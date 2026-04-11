"""
modules/module1_claim_decomposer.py
------------------------------------
Module 1 — Claim Decomposer

Breaks a composite claim into an ordered list of atomic sub-questions using
a small instruction-tuned LLM (Phi-3-mini or Mistral-7B).

Each sub-question is:
  - independently verifiable
  - answerable from external evidence or the video content
  - ordered so that earlier answers condition later questions

Input  : claim_text, VideoSegment, visual_caption, conflict_flag, GenerativeLLM
Output : ClaimDecomposition
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from schemas import ClaimDecomposition, SubQuestion, VideoSegment
from models import GenerativeLLM

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a claim decomposer for a video fact-checking system.
Given a composite claim and a video context summary, decompose the claim
into an ordered list of atomic sub-questions. Each sub-question must be:
- independently verifiable
- answerable from external evidence or the video content
- ordered so that earlier answers can condition later questions

Respond ONLY with valid JSON — no markdown fences, no preamble:
{
  "claim_id": "<string>",
  "sub_questions": [
    {
      "hop": <integer starting at 1>,
      "question": "<string>",
      "depends_on_hops": [<integer>, ...],
      "evidence_type": "video" | "web" | "kb" | "any"
    }
  ]
}"""


def _build_user_message(
    claim_text: str,
    visual_caption: str,
    transcript_excerpt: str,
    start_ts: float,
    end_ts: float,
    conflict_flag: bool,
    claim_id: str,
    rationale_hint: str = "",
) -> str:
    hint_block = f'\nKnown rationale context (use to guide decomposition):\n"{rationale_hint}"\n' if rationale_hint else ""
    return (
        f'Claim ID: "{claim_id}"\n'
        f'Claim: "{claim_text}"\n'
        f'Visual caption: "{visual_caption}"\n'
        f'Transcript excerpt: "{transcript_excerpt}"\n'
        f"Timestamp: {start_ts}s – {end_ts}s\n"
        f"Modal conflict flag: {conflict_flag}\n"
        f"{hint_block}\n"
        "Decompose this claim into atomic sub-questions."
    )


def _build_prompt(
    claim_text: str,
    visual_caption: str,
    transcript_excerpt: str,
    start_ts: float,
    end_ts: float,
    conflict_flag: bool,
    claim_id: str,
    rationale_hint: str = "",
) -> list[dict[str, str]]:
    """Return a chat-template message list."""
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _build_user_message(
                claim_text, visual_caption, transcript_excerpt,
                start_ts, end_ts, conflict_flag, claim_id,
                rationale_hint=rationale_hint,
            ),
        },
    ]


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def _safe_json_parse(raw: str) -> dict[str, Any]:
    """
    Robustly extract a JSON object from the model output.
    Strips markdown code fences if present.
    """
    # Remove ```json ... ``` or ``` ... ``` wrappers
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())
    # Find the first '{' ... last '}' in case there is surrounding text
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in model output:\n{raw}")
    return json.loads(cleaned[start:end])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def decompose_claim(
    claim_text: str,
    claim_id: str,
    segment: VideoSegment,
    visual_caption: str,
    conflict_flag: bool,
    llm: GenerativeLLM,
    max_retries: int = 2,
    rationale_hint: str = "",
) -> ClaimDecomposition:
    """
    Decompose a composite claim into atomic sub-questions.

    Parameters
    ----------
    claim_text     : The full claim string to verify.
    claim_id       : Stable identifier for this claim.
    segment        : The associated VideoSegment (provides transcript + timestamps).
    visual_caption : Caption produced by the visual captioner for this segment.
    conflict_flag  : Whether Module 2 detected a cross-modal conflict.
    llm            : A loaded GenerativeLLM instance.
    max_retries    : Number of generation retries on parse failure.
    rationale_hint : Optional gold-rationale summary injected into the prompt
                     (from RationaleContext.prompt_summary()). Guides the LLM
                     toward evidence-aligned sub-questions when available.

    Returns
    -------
    ClaimDecomposition
    """
    prompt = _build_prompt(
        claim_text=claim_text,
        visual_caption=visual_caption,
        transcript_excerpt=segment.transcript[:300],
        start_ts=segment.start_ts,
        end_ts=segment.end_ts,
        conflict_flag=conflict_flag,
        claim_id=claim_id,
        rationale_hint=rationale_hint,
    )

    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            raw = llm.generate(prompt, max_new_tokens=512)
            data = _safe_json_parse(raw)

            sub_questions = [
                SubQuestion(
                    hop=sq["hop"],
                    question=sq["question"],
                    depends_on_hops=sq.get("depends_on_hops", []),
                    evidence_type=sq.get("evidence_type", "any"),
                )
                for sq in data.get("sub_questions", [])
            ]

            if not sub_questions:
                raise ValueError("LLM returned zero sub-questions.")

            return ClaimDecomposition(
                claim_id=data.get("claim_id", claim_id),
                claim_text=claim_text,
                segment_id=segment.segment_id,
                sub_questions=sub_questions,
            )

        except Exception as exc:
            last_exc = exc
            logger.warning("Claim decomposition attempt %d/%d failed: %s", attempt + 1, max_retries + 1, exc)

    # Fallback: single sub-question covering the whole claim
    logger.error("All decomposition attempts failed (%s). Using fallback single-hop.", last_exc)
    return ClaimDecomposition(
        claim_id=claim_id,
        claim_text=claim_text,
        segment_id=segment.segment_id,
        sub_questions=[
            SubQuestion(
                hop=1,
                question=claim_text,
                depends_on_hops=[],
                evidence_type="any",
            )
        ],
    )
