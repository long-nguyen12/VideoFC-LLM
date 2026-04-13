"""
models/model_bundle.py
----------------------
Loads and wraps all HuggingFace / PyTorch models required by the pipeline.
Each wrapper exposes a minimal interface so modules stay model-agnostic.

Model roles
-----------
  caption_model  : LLaVA-1.6 (7B) or moondream2 (1.8B)  — keyframe → caption
  nli_model      : DeBERTa-v3-small (184M)               — entailment scorer
  text_encoder   : bge-small-en-v1.5 (33M)               — dense retrieval encoder
  decomposer_llm : Phi-3-mini or Mistral-7B               — claim decomposition
  hop_llm        : Qwen2.5-3B or Phi-3-mini               — per-hop reader + summariser
  aggregator_llm : Mistral-7B or LLaMA-3.1-8B             — verdict aggregation
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
from PIL import Image
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModel,
    AutoProcessor,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
    AutoModelForCausalLM,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_image(source: str) -> Image.Image:
    """Accept either a file path or a base64-encoded JPEG/PNG string."""
    if source.startswith("data:image") or (len(source) > 260 and "/" not in source[:10]):
        # base64
        header, _, data = source.partition(",")
        raw = base64.b64decode(data if data else source)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    return Image.open(source).convert("RGB")


# ---------------------------------------------------------------------------
# Visual Captioner (Qwen2-VL)
# ---------------------------------------------------------------------------

class VisualCaptioner:
    """
    Wraps Qwen2-VL for keyframe captioning.
    Defaults to Qwen/Qwen2-VL-2B-Instruct as it is highly efficient and capable.
    """

    DEFAULT_MODEL = "Qwen/Qwen2-VL-2B-Instruct"

    def __init__(self, model_name: str = DEFAULT_MODEL, device: Optional[torch.device] = None):
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        self.device = device or _device()
        logger.info("Loading visual captioner: %s", model_name)
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
    def caption(self, keyframes: list[str], prompt: str = "Describe this image concisely.") -> str:
        """
        Caption a list of keyframes and return one aggregated description.
        For multi-frame segments the captions are concatenated with ' | '.
        """
        captions: list[str] = []
        for src in keyframes:
            image = _load_image(src)
            conversation = [
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": prompt}],
                }
            ]
            text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            inputs = self.processor(images=[image], text=[text], padding=True, return_tensors="pt").to(self.device)
            output_ids = self.model.generate(
                **inputs, 
                max_new_tokens=128, 
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )
            # Strip the input tokens from the output
            generated = output_ids[0, inputs["input_ids"].shape[1]:]
            captions.append(self.processor.decode(generated, skip_special_tokens=True).strip())
        return " | ".join(captions) if captions else ""


# ---------------------------------------------------------------------------
# NLI Scorer (DeBERTa-v3-small)
# ---------------------------------------------------------------------------

class NLIScorer:
    """
    Wraps cross-encoder/nli-deberta-v3-small.
    Returns the entailment probability for a (premise, hypothesis) pair.
    Label order from the model: contradiction=0, entailment=1, neutral=2
    (checked against the model config; adjust ENTAILMENT_IDX if needed).
    """

    DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-small"
    ENTAILMENT_IDX = 1  # index of "entailment" in the label list

    def __init__(self, model_name: str = DEFAULT_MODEL, device: Optional[torch.device] = None):
        self.device = device or _device()
        logger.info("Loading NLI scorer: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        # Resolve entailment label index from config if available
        id2label = self.model.config.id2label
        for idx, label in id2label.items():
            if "entail" in label.lower():
                self.ENTAILMENT_IDX = int(idx)
                break

    @torch.inference_mode()
    def entailment_score(self, premise: str, hypothesis: str) -> float:
        """Return the entailment probability in [0, 1]."""
        inputs = self.tokenizer(
            premise, hypothesis,
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
    """
    Wraps BAAI/bge-small-en-v1.5 for dense semantic similarity.
    Used by the retriever to embed queries and passages.
    """

    DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

    def __init__(self, model_name: str = DEFAULT_MODEL, device: Optional[torch.device] = None):
        self.device = device or _device()
        logger.info("Loading text encoder: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, texts: list[str], batch_size: int = 64) -> torch.Tensor:
        """
        Encode a list of strings into L2-normalised embeddings.
        Returns shape (N, hidden_dim).
        """
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
        """Return cosine similarities between query and each passage."""
        q_emb = self.encode([query])           # (1, D)
        p_emb = self.encode(passages)          # (N, D)
        scores = (q_emb @ p_emb.T).squeeze(0) # (N,)
        return scores.tolist()


# ---------------------------------------------------------------------------
# Generative LLM wrapper (shared for decomposer, hop reader, aggregator)
# ---------------------------------------------------------------------------

class GenerativeLLM:
    """
    Generic HuggingFace causal-LM wrapper.
    Supports chat-template models (Phi-3, Mistral, Qwen, LLaMA-3).

    model_name examples:
        "Qwen/Qwen2.5-1.5B-Instruct"   # single-model / hardware-constrained
        "Qwen/Qwen2.5-3B-Instruct"
        "microsoft/Phi-3-mini-4k-instruct"
        "mistralai/Mistral-7B-Instruct-v0.3"
        "meta-llama/Meta-Llama-3.1-8B-Instruct"

    Parameters
    ----------
    max_new_tokens_cap : Hard ceiling on generated tokens regardless of what
                         the caller requests. Set this to 512 for 2B models to
                         avoid running past the useful generation length.
    context_window     : Maximum input token length fed to the model.
                         Qwen2.5-1.5B / 2B support 32 K, but keeping this at
                         2048 on CPU avoids OOM on constrained hardware.
    """

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
        logger.info("Loading generative LLM: %s  (max_new_tokens_cap=%d, context_window=%d)",
                    model_name, max_new_tokens_cap, context_window)

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
        """
        Accept either:
          - a plain string (used as the full prompt verbatim), or
          - a list of chat dicts: [{"role": "system", "content": ...}, ...]

        Returns the raw generated text (stripped).
        """
        if isinstance(prompt, list) and self._supports_chat_template:
            text = self.tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
        else:
            text = prompt if isinstance(prompt, str) else str(prompt)

        inputs = self.tokenizer(
            text, return_tensors="pt",
            truncation=True, max_length=self.context_window,
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
            kwargs["temperature"] = None
            kwargs["top_p"] = None
            kwargs["top_k"] = None

        output_ids = self.model.generate(**inputs, **kwargs)
        generated = output_ids[0, input_len:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Model Bundle
# ---------------------------------------------------------------------------

@dataclass
class ModelBundle:
    """
    Container for all models used by the pipeline.
    Pass this into run_pipeline() as a single argument.
    """
    captioner: VisualCaptioner
    nli: NLIScorer
    encoder: TextEncoder
    decomposer_llm: GenerativeLLM
    hop_llm: GenerativeLLM
    aggregator_llm: GenerativeLLM

    # Convenience wrapper so pipeline code can call models.caption_fn(keyframes)
    def caption_fn(self, keyframes: list[str]) -> str:
        return self.captioner.caption(keyframes)


def load_default_bundle(
    captioner_model: str = "Qwen/Qwen2-VL-2B-Instruct",
    nli_model: str = "cross-encoder/nli-deberta-v3-small",
    encoder_model: str = "BAAI/bge-small-en-v1.5",
    decomposer_model: str = "microsoft/Phi-3-mini-4k-instruct",
    hop_model: str = "Qwen/Qwen2.5-3B-Instruct",
    aggregator_model: str = "mistralai/Mistral-7B-Instruct-v0.3",
    load_in_4bit: bool = False,
) -> ModelBundle:
    """
    Full multi-model bundle. Loads a separate checkpoint for each pipeline
    role. Requires ~20 GB VRAM (or ~12 GB with load_in_4bit=True).
    Override any model name to swap in a different checkpoint.
    """
    device = _device()
    return ModelBundle(
        captioner=VisualCaptioner(captioner_model, device=device),
        nli=NLIScorer(nli_model, device=device),
        encoder=TextEncoder(encoder_model, device=device),
        decomposer_llm=GenerativeLLM(decomposer_model, device=device, load_in_4bit=load_in_4bit),
        hop_llm=GenerativeLLM(hop_model, device=device, load_in_4bit=load_in_4bit),
        aggregator_llm=GenerativeLLM(aggregator_model, device=device, load_in_4bit=load_in_4bit),
    )


def load_single_llm_bundle(
    llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    captioner_model: str = "Qwen/Qwen2-VL-2B-Instruct",
    nli_model: str = "cross-encoder/nli-deberta-v3-small",
    encoder_model: str = "BAAI/bge-small-en-v1.5",
    load_in_4bit: bool = False,
    context_window: int = 2048,
) -> ModelBundle:
    """
    Hardware-constrained bundle: one small LLM shared across all three
    generative roles (decomposer, hop reader, aggregator).

    The single LLM is loaded once and the same Python object is assigned to
    all three ModelBundle slots — no duplication of weights in memory.

    Recommended models by VRAM budget
    ----------------------------------
    <4 GB  : "Qwen/Qwen2.5-0.5B-Instruct"   or  "Qwen/Qwen2.5-1.5B-Instruct"
    4–6 GB : "Qwen/Qwen2.5-3B-Instruct"      or  "microsoft/Phi-3-mini-4k-instruct"
    6–8 GB : any of the above with load_in_4bit=False

    The captioner defaults to moondream2 (1.8 B, ~3.5 GB), which is the
    lightest VLM that produces usable captions. Set captioner_model=None to
    skip the captioner entirely and always use the synthetic caption path
    (appropriate when the dataset provides no keyframes).

    Parameters
    ----------
    llm_model       : HuggingFace model ID for the single shared LLM.
    captioner_model : VLM for keyframe captioning. Pass None to disable.
    nli_model       : NLI classifier (184 M, always separate — no generation).
    encoder_model   : Dense retrieval encoder (33 M, always separate).
    load_in_4bit    : Enable bitsandbytes 4-bit quantisation on the LLM.
                      Cuts VRAM roughly in half; requires bitsandbytes>=0.43.
    context_window  : Max input token length. Keep at 2048 on CPU to avoid
                      OOM; raise to 4096 on GPU if VRAM permits.

    Returns
    -------
    ModelBundle  where decomposer_llm, hop_llm, aggregator_llm all point to
                 the same GenerativeLLM instance.
    """
    device = _device()
    logger.info(
        "Loading single-LLM bundle: llm=%s  captioner=%s  4bit=%s  ctx=%d",
        llm_model, captioner_model, load_in_4bit, context_window,
    )

    # The aggregator prompt is the longest; cap new tokens at 512 for small
    # models so generation does not run past the useful JSON response.
    shared_llm = GenerativeLLM(
        llm_model,
        device=device,
        load_in_4bit=load_in_4bit,
        max_new_tokens_cap=512,
        context_window=context_window,
    )

    if captioner_model is not None:
        captioner = VisualCaptioner(captioner_model, device=device)
    else:
        # Dummy captioner — caption_fn always returns an empty string so the
        # pipeline falls back to the synthetic caption from the dataset adapter.
        class _NullCaptioner:
            def caption(self, keyframes: list[str], **_) -> str:
                return ""
        captioner = _NullCaptioner()  # type: ignore[assignment]

    return ModelBundle(
        captioner=captioner,
        nli=NLIScorer(nli_model, device=device),
        encoder=TextEncoder(encoder_model, device=device),
        decomposer_llm=shared_llm,   # ─┐
        hop_llm=shared_llm,          #  ├─ same object, weights loaded once
        aggregator_llm=shared_llm,   # ─┘
    )
