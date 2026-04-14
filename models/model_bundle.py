from __future__ import annotations

import base64
import io
import logging
from typing import Optional

import torch
from huggingface_hub import logging as hf_logging
from PIL import Image
from transformers import (AutoModel, AutoModelForCausalLM,
                          AutoModelForSequenceClassification, AutoProcessor,
                          AutoTokenizer, LlavaNextForConditionalGeneration,
                          LlavaNextProcessor)

hf_logging.set_verbosity_error()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_image(source: str) -> Image.Image:
    if source.startswith("data:image") or (
        len(source) > 260 and "/" not in source[:10]
    ):
        # base64
        header, _, data = source.partition(",")
        raw = base64.b64decode(data if data else source)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    return Image.open(source).convert("RGB")


# ---------------------------------------------------------------------------
# Visual Captioner (Qwen2-VL)
# ---------------------------------------------------------------------------


class VisualCaptioner:

    DEFAULT_MODEL = "Qwen/Qwen2-VL-2B-Instruct"

    def __init__(
        self, model_name: str = DEFAULT_MODEL, device: Optional[torch.device] = None
    ):
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.device = device or _device()
        logger.debug("Loading visual captioner: %s", model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            device_map="auto" if self.device.type == "cuda" else None,
        )
        if self.device.type != "cuda":
            self.model = self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def caption(
        self,
        keyframes: list[str],
        prompt: str = "Can you describe what is happening in the video in detail?",
    ) -> str:
        captions: list[str] = []
        for src in keyframes:
            image = _load_image(src)
            conversation = [
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": prompt}],
                }
            ]
            text = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
            inputs = self.processor(
                images=[image], text=[text], padding=True, return_tensors="pt"
            ).to(self.device)
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )
            # Strip the input tokens from the output
            generated = output_ids[0, inputs["input_ids"].shape[1] :]
            captions.append(
                self.processor.decode(generated, skip_special_tokens=True).strip()
            )
        return " | ".join(captions) if captions else ""


# ---------------------------------------------------------------------------
# NLI Scorer (DeBERTa-v3-small)
# ---------------------------------------------------------------------------


class NLIScorer:

    DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-small"
    ENTAILMENT_IDX = 1  # index of "entailment" in the label list

    def __init__(
        self, model_name: str = DEFAULT_MODEL, device: Optional[torch.device] = None
    ):
        self.device = device or _device()
        logger.debug("Loading NLI scorer: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
            self.device
        )
        self.model.eval()
        # Resolve entailment label index from config if available
        id2label = self.model.config.id2label
        for idx, label in id2label.items():
            if "entail" in label.lower():
                self.ENTAILMENT_IDX = int(idx)
                break

    @torch.inference_mode()
    def entailment_score(self, premise: str, hypothesis: str) -> float:
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)
        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        return float(probs[0, self.ENTAILMENT_IDX].item())


# ---------------------------------------------------------------------------
# Text Encoder (bge-small-en-v1.5)
# ---------------------------------------------------------------------------


class TextEncoder:

    DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

    def __init__(
        self, model_name: str = DEFAULT_MODEL, device: Optional[torch.device] = None
    ):
        self.device = device or _device()
        logger.debug("Loading text encoder: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, texts: list[str], batch_size: int = 64) -> torch.Tensor:
        all_embeddings: list[torch.Tensor] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            outputs = self.model(**inputs)
            # CLS-token pooling (standard for bge)
            embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
            all_embeddings.append(embeddings)
        return torch.cat(all_embeddings, dim=0)

    def similarity(self, query: str, passages: list[str]) -> list[float]:
        q_emb = self.encode([query])  # (1, D)
        p_emb = self.encode(passages)  # (N, D)
        scores = (q_emb @ p_emb.T).squeeze(0)  # (N,)
        return scores.tolist()


# ---------------------------------------------------------------------------
# Generative LLM wrapper (shared for decomposer, hop reader, aggregator)
# ---------------------------------------------------------------------------


class GenerativeLLM:

    def __init__(
        self,
        model_name: str,
        device: Optional[torch.device] = None,
        load_in_4bit: bool = False,
        max_new_tokens_cap: int = 1024,
        context_window: int = 4096,
    ):
        self.device = device or _device()
        self.max_new_tokens_cap = max_new_tokens_cap
        self.context_window = context_window
        logger.debug(
            "Loading generative LLM: %s  (max_new_tokens_cap=%d, context_window=%d)",
            model_name,
            max_new_tokens_cap,
            context_window,
        )

        quantization_config = None
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            device_map="auto" if self.device.type == "cuda" else None,
        )
        if self.device.type != "cuda":
            self.model = self.model.to(self.device)
        self.model.eval()
        self._supports_chat_template = hasattr(self.tokenizer, "apply_chat_template")

    @torch.inference_mode()
    def generate(
        self,
        prompt: str | list[dict],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        do_sample: bool = False,
    ) -> str:
        if isinstance(prompt, list) and self._supports_chat_template:
            text = self.tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
        else:
            text = prompt if isinstance(prompt, str) else str(prompt)

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.context_window,
        ).to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        effective_max = min(max_new_tokens, self.max_new_tokens_cap)
        kwargs = {
            "max_new_tokens": effective_max,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if do_sample:
            kwargs["temperature"] = temperature
        else:
            kwargs["temperature"] = 1
            kwargs["top_p"] = 0.7
            kwargs["top_k"] = 0.6

        output_ids = self.model.generate(**inputs, **kwargs)
        generated = output_ids[0, input_len:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


def load_default_bundle(
    captioner_model: str = "Qwen/Qwen2-VL-2B-Instruct",
    nli_model: str = "cross-encoder/nli-deberta-v3-small",
    encoder_model: str = "BAAI/bge-small-en-v1.5",
    decomposer_model: str = "microsoft/Phi-3-mini-4k-instruct",
    hop_model: str = "Qwen/Qwen2.5-3B-Instruct",
    aggregator_model: str = "mistralai/Mistral-7B-Instruct-v0.3",
    load_in_4bit: bool = False,
) -> dict:
    device = _device()
    captioner = VisualCaptioner(captioner_model, device=device)
    nli = NLIScorer(nli_model, device=device)
    encoder = TextEncoder(encoder_model, device=device)
    decomposer_llm = GenerativeLLM(
        decomposer_model, device=device, load_in_4bit=load_in_4bit
    )
    hop_llm = GenerativeLLM(hop_model, device=device, load_in_4bit=load_in_4bit)
    aggregator_llm = GenerativeLLM(
        aggregator_model, device=device, load_in_4bit=load_in_4bit
    )
    consistency_llm = hop_llm
    return {
        "captioner": captioner,
        "caption_fn": captioner.caption,
        "nli": nli,
        "encoder": encoder,
        "decomposer_llm": decomposer_llm,
        "hop_llm": hop_llm,
        "aggregator_llm": aggregator_llm,
        "consistency_llm": consistency_llm,
    }


def load_single_llm_bundle(
    llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    captioner_model: str = "Qwen/Qwen2-VL-2B-Instruct",
    nli_model: str = "cross-encoder/nli-deberta-v3-small",
    encoder_model: str = "BAAI/bge-small-en-v1.5",
    load_in_4bit: bool = False,
    context_window: int = 2048,
) -> dict:
    device = _device()
    logger.debug(
        "Loading single-LLM bundle: llm=%s  captioner=%s  4bit=%s  ctx=%d",
        llm_model,
        captioner_model,
        load_in_4bit,
        context_window,
    )

    shared_llm = GenerativeLLM(
        llm_model,
        device=device,
        load_in_4bit=load_in_4bit,
        max_new_tokens_cap=512,
        context_window=context_window,
    )
    captioner = VisualCaptioner(captioner_model, device=device)

    return {
        "captioner": captioner,
        "caption_fn": captioner.caption,
        "nli": NLIScorer(nli_model, device=device),
        "encoder": TextEncoder(encoder_model, device=device),
        "decomposer_llm": shared_llm,
        "hop_llm": shared_llm,
        "aggregator_llm": shared_llm,
        "consistency_llm": shared_llm,
    }
