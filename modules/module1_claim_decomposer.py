from __future__ import annotations

import json
import logging
import re

from models import GenerativeLLM
from modules.utils import safe_json_parse as _safe_json_parse
from modules.prompt_template import (
    _DECOMPOSITION_PROMPT_TEMPLATE as _SYSTEM_PROMPT_TEMPLATE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def _build_user_message(
    claim_text: str,
    visual_caption: str,
    transcript_excerpt: str,
    conflict_flag: bool,
    claim_id: str,
    rationale_hint: str = "",
) -> str:
    hint_block = (
        f'\nKnown rationale context (use to guide decomposition):\n"{rationale_hint}"\n'
        if rationale_hint
        else ""
    )
    return (
        f'Claim ID: "{claim_id}"\n'
        f'Claim: "{claim_text}"\n'
        f'Visual: "{visual_caption}"\n'
        f'Transcript: "{transcript_excerpt}"\n'
        f"Conflict flag: {conflict_flag}\n"
        f"{hint_block}\n"
    )


def _build_prompt(
    claim_text: str,
    visual_caption: str,
    transcript_excerpt: str,
    conflict_flag: bool,
    claim_id: str,
    rationale_hint: str = "",
    max_sub_questions: int = 5,
) -> list[dict[str, str]]:
    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(max_sub_questions=max_sub_questions)
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": _build_user_message(
                claim_text,
                visual_caption,
                transcript_excerpt,
                conflict_flag,
                claim_id,
                rationale_hint=rationale_hint,
            ),
        },
    ]


# JSON parsing — shared implementation lives in modules/utils.py

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def decompose_claim(
    claim_text: str,
    claim_id: str,
    segment: dict,
    visual_caption: str,
    conflict_flag: bool,
    llm: GenerativeLLM,
    max_retries: int = 2,
    rationale_hint: str = "",
    max_sub_questions: int = 5,
) -> dict:
    def _esc(s: str) -> str:
        return s.replace('"', '\\"').replace("\\", "\\\\") if s else ""

    prompt = _build_prompt(
        claim_text=_esc(claim_text),
        visual_caption=_esc(visual_caption),
        transcript_excerpt=_esc(segment["transcript"]),
        conflict_flag=conflict_flag,
        claim_id=_esc(claim_id),
        rationale_hint=_esc(rationale_hint),
        max_sub_questions=max_sub_questions,
    )

    data = llm.generate_json(
        prompt,
        max_new_tokens=1024,
        temperature=0.0,
        max_retries=max_retries,
    )

    if data is None:
        logger.error(
            "Claim decomposition failed after %d attempts. claim_id=%s segment_id=%s",
            max_retries + 1,
            claim_id,
            segment.get("segment_id"),
        )
        return {
            "claim_id": claim_id,
            "claim_text": claim_text,
            "segment_id": segment["segment_id"],
            "sub_questions": [
                {
                    "hop": 1,
                    "question": claim_text,
                    "depends_on_hops": [],
                    "evidence_type": "any",
                }
            ],
        }

    data = _normalize_decomposition_output(data, claim_id)
    is_valid, err_msg = _validate_decomposition_output(data, claim_id)
    if not is_valid:
        logger.warning(
            "Decomposition output failed validation (claim_id=%s): %s. Raw keys: %s",
            claim_id,
            err_msg,
            list(data.keys()) if isinstance(data, dict) else type(data),
        )
        return {
            "claim_id": claim_id,
            "claim_text": claim_text,
            "segment_id": segment["segment_id"],
            "sub_questions": [
                {
                    "hop": 1,
                    "question": claim_text,
                    "depends_on_hops": [],
                    "evidence_type": "any",
                }
            ],
        }

    sub_questions = [
        {
            "hop": sq["hop"],
            "question": sq["question"],
            "depends_on_hops": sq.get("depends_on_hops", []),
            "evidence_type": sq.get("evidence_type", "any"),
        }
        for sq in data.get("sub_questions", [])
        if isinstance(sq.get("hop"), int) and isinstance(sq.get("question"), str)
    ][:max_sub_questions]

    if not sub_questions:
        logger.warning(
            "No valid sub-questions extracted; using fallback. claim_id=%s", claim_id
        )
        sub_questions = [
            {
                "hop": 1,
                "question": claim_text,
                "depends_on_hops": [],
                "evidence_type": "any",
            }
        ]

    return {
        "claim_id": data.get("claim_id", claim_id),
        "claim_text": claim_text,
        "segment_id": segment["segment_id"],
        "sub_questions": sub_questions,
    }


def _normalize_decomposition_output(data: dict, claim_id: str) -> dict:
    """
    Normalize common near-miss outputs into the required schema.
    This prevents dropping to fallback when the model emits a single
    sub-question object at the top level.
    """
    if not isinstance(data, dict):
        return data

    normalized = dict(data)

    # Common case: model returns one sub-question object directly.
    if (
        "sub_questions" not in normalized
        and "hop" in normalized
        and "question" in normalized
    ):
        single_sq = {
            "hop": normalized.get("hop", 1),
            "question": normalized.get("question", ""),
            "depends_on_hops": normalized.get("depends_on_hops", []),
            "evidence_type": normalized.get("evidence_type", "any"),
        }
        normalized = {
            "claim_id": claim_id,
            "sub_questions": [single_sq],
        }
        return normalized

    # Sometimes sub_questions is emitted as a single object instead of a list.
    if isinstance(normalized.get("sub_questions"), dict):
        normalized["sub_questions"] = [normalized["sub_questions"]]

    return normalized


def _validate_decomposition_output(data: dict, claim_id: str) -> tuple[bool, str]:
    if not isinstance(data, dict):
        return False, "Output is not a JSON object"
    if "sub_questions" not in data:
        return False, "Missing 'sub_questions' field"
    if not isinstance(data["sub_questions"], list):
        return False, "'sub_questions' must be a list"

    for i, sq in enumerate(data["sub_questions"]):
        if not isinstance(sq.get("hop"), int) or sq["hop"] < 1:
            return False, f"sub_questions[{i}]: 'hop' must be integer >= 1"
        if not isinstance(sq.get("question"), str) or not sq["question"].strip():
            return False, f"sub_questions[{i}]: 'question' must be non-empty string"
        if sq.get("evidence_type") not in ("video", "web", "kb", "any"):
            return False, f"sub_questions[{i}]: invalid 'evidence_type'"
        if not isinstance(sq.get("depends_on_hops", []), list):
            return False, f"sub_questions[{i}]: 'depends_on_hops' must be a list"

    return True, ""
