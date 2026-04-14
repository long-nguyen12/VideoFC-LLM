from __future__ import annotations

import base64
import copy
import io
import json
import logging
import re
from typing import List, Optional, Union

import torch
from huggingface_hub import logging as hf_logging
from PIL import Image
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
    Qwen2VLForConditionalGeneration,
)
from modules.utils import safe_json_parse as _safe_json_parse

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
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.debug("Loading visual captioner: %s on %s", model_name, self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        dtype = (
            torch.bfloat16
            if (self.device.type == "cuda" and torch.cuda.is_bf16_supported())
            else torch.float16
        )
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map=self.device.type if self.device.type == "cuda" else None,
        )
        if self.device.type != "cuda":
            self.model = self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def caption(
        self,
        keyframes: List[str],
        prompt: str = "Can you describe what is happening in this image in detail?",
    ) -> str:
        if not keyframes:
            return ""

        captions = []
        for src in keyframes:
            image = _load_image(src)
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
            inputs = self.processor(
                text=[text], images=[image], padding=True, return_tensors="pt"
            ).to(self.device)

            output_ids = self.model.generate(
                **inputs, max_new_tokens=512, do_sample=False
            )
            prompt_len = inputs["attention_mask"][0].sum().item()
            generated = output_ids[0, prompt_len:]
            caption = self.processor.decode(generated, skip_special_tokens=True).strip()
            captions.append(caption)

        return " | ".join(captions)


class GenerativeLLM:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device: Optional[torch.device] = None,
        load_in_4bit: bool = False,
        max_new_tokens_cap: int = 256,
        context_window: int = 8192,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.max_new_tokens_cap = max_new_tokens_cap
        self.context_window = context_window

        logger.debug(
            "Loading generative LLM: %s (cap=%d, ctx=%d, 4bit=%s)",
            model_name,
            max_new_tokens_cap,
            context_window,
            load_in_4bit,
        )

        quantization_config = None
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=(
                    torch.bfloat16
                    if (self.device.type == "cuda" and torch.cuda.is_bf16_supported())
                    else torch.float16
                ),
            )

        dtype = (
            None
            if load_in_4bit
            else (
                torch.bfloat16
                if (self.device.type == "cuda" and torch.cuda.is_bf16_supported())
                else torch.float16
            )
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            device_map=self.device.type if self.device.type == "cuda" else None,
        )

        if self.device.type != "cuda" or not hasattr(self.model, "hf_device_map"):
            self.model = self.model.to(self.device)
        self.model.eval()
        self._supports_chat_template = hasattr(self.tokenizer, "apply_chat_template")

    @torch.inference_mode()
    def generate(
        self,
        prompt: Union[str, List[dict]],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        do_sample: bool = False,
    ) -> str:
        do_sample = do_sample and temperature > 0.0

        if isinstance(prompt, list) and self._supports_chat_template:
            text = self.tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
        else:
            text = prompt if isinstance(prompt, str) else str(prompt)

        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self.context_window
        ).to(self.device)

        input_len = inputs["input_ids"].shape[1]
        effective_max = min(max_new_tokens, self.max_new_tokens_cap)

        kwargs = {
            "max_new_tokens": effective_max,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if do_sample:
            kwargs["temperature"] = max(temperature, 1e-5)
            kwargs["top_p"] = 0.9
            kwargs["top_k"] = 50

        output_ids = self.model.generate(**inputs, **kwargs)
        generated = output_ids[0, input_len:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    @torch.inference_mode()
    def generate_json(
        self,
        prompt: Union[str, List[dict]],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        max_retries: int = 2,
    ) -> Optional[dict]:
        """Generates text and safely extracts a JSON object with retry logic."""
        current_prompt = prompt

        for attempt in range(max_retries + 1):
            current_temp = temperature if attempt == 0 else 0.0

            if attempt > 0 and isinstance(current_prompt, list):
                current_prompt = copy.deepcopy(current_prompt)
                current_prompt[-1][
                    "content"
                ] += (
                    "\n\nCRITICAL: Output ONLY valid JSON. Start with { and end with }."
                )
            try:
                raw = self.generate(
                    current_prompt, max_new_tokens=max_new_tokens, temperature=current_temp
                )
                parsed = self.extract_json(raw)

                if parsed is not None:
                    return parsed
                logger.warning(
                    "JSON extraction failed (attempt %d/%d). Raw: %s",
                    attempt + 1,
                    max_retries + 1,
                    raw,
                )
            except Exception as exc:
                logger.warning(
                    "LLM generation or JSON extraction error (attempt %d/%d): %s",
                    attempt + 1,
                    max_retries + 1,
                    exc,
                )

        logger.error("Failed to extract valid JSON after %d attempts", max_retries + 1)
        return None

    @staticmethod
    def extract_json(text: str) -> Optional[dict]:
        if not text or not isinstance(text, str):
            return None

        # Try 1: Raw parse (fast path for well-formed output)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try 2: Strip whitespace only
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Try 3: Remove markdown fences if present
        no_fences = re.sub(
            r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE
        )
        try:
            return json.loads(no_fences)
        except json.JSONDecodeError:
            pass

        # Try 4: Brace-matching extraction (handles trailing noise)
        start = no_fences.find("{")
        if start == -1:
            return None

        depth, end, in_str, escaped = 0, -1, False, False
        for i, c in enumerate(no_fences[start:], start):
            if escaped:
                escaped = False
                continue
            if c == "\\" and in_str:
                escaped = True
                continue
            if c == '"' and not escaped:
                in_str = not in_str
                continue
            if in_str:
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        if end == -1:
            return None

        try:
            return json.loads(no_fences[start:end])
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse JSON from extracted text: %s", exc)
            return None


# ---------------------------------------------------------------------------
# Bundle Loaders
# ---------------------------------------------------------------------------


def load_default_bundle(
    captioner_model: str = "Qwen/Qwen2-VL-2B-Instruct",
    llm_model: str = "Qwen/Qwen2.5-3B-Instruct",
    load_in_4bit: bool = False,
) -> dict:
    device = _device()
    shared_llm = GenerativeLLM(llm_model, device=device, load_in_4bit=load_in_4bit)
    captioner = VisualCaptioner(captioner_model, device=device)

    return {
        "captioner": captioner,
        "caption_fn": captioner.caption,
        "decomposer_llm": shared_llm,
        "hop_llm": shared_llm,
        "aggregator_llm": shared_llm,
        "consistency_llm": shared_llm,
    }


def load_single_llm_bundle(
    llm_model: str = "Qwen/Qwen2.5-7B-Instruct",
    captioner_model: str = "Qwen/Qwen2-VL-2B-Instruct",
    load_in_4bit: bool = False,
    context_window: int = 8192,
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
        max_new_tokens_cap=1024,
        context_window=context_window,
    )
    captioner = VisualCaptioner(captioner_model, device=device)

    return {
        "captioner": captioner,
        "caption_fn": captioner.caption,
        "decomposer_llm": shared_llm,
        "hop_llm": shared_llm,
        "aggregator_llm": shared_llm,
        "consistency_llm": shared_llm,
    }
