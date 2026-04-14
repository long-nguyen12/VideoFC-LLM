from __future__ import annotations

import json
import logging
import re

from models import GenerativeLLM
from modules.utils import safe_json_parse as _safe_json_parse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
You are a claim decomposer for a video fact-checking system.
Given a composite claim and a video context summary, decompose the claim
into an ordered list of atomic sub-questions. Each sub-question must be:
- independently verifiable
- answerable from external evidence or the video content
- ordered so that earlier answers can condition later questions
- Produce at most {max_sub_questions} sub-questions.

OUTPUT FORMAT RULES (MANDATORY):
1. Respond ONLY with a valid JSON object. Do not include markdown, code blocks, explanations, greetings, or any text outside the JSON.
2. Use double quotes for ALL keys and string values. Single quotes are invalid JSON and will cause parsing errors.
3. Ensure proper JSON escaping for special characters (e.g., \\", \\\\, \\n).
4. Do not include trailing commas, comments, or schema annotations in the output.
5. If decomposition is impossible or input is ambiguous, return: {{"claim_id": null, "sub_questions": [], "error": "brief reason"}}

REQUIRED JSON SCHEMA:
{{
  "claim_id": "<string or null>",
  "sub_questions": [
    {{
      "hop": <integer starting at 1>,
      "question": "<string>",
      "depends_on_hops": [<integer>, ...],
      "evidence_type": "video" | "web" | "kb" | "any"
    }}
  ]
}}
"""


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
        f'Visual: "{visual_caption}"\n'
        f'Transcript: "{transcript_excerpt}"\n'
        f"Conflict flag: {conflict_flag}\n"
        f"{hint_block}\n"
        "Decompose this claim. Output ONLY valid JSON. Start your response with { and end with }."
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
    max_sub_questions: int = 5,
) -> list[dict[str, str]]:
    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(max_sub_questions=max_sub_questions)
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": _build_user_message(
                claim_text, visual_caption, transcript_excerpt,
                start_ts, end_ts, conflict_flag, claim_id,
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
    prompt = _build_prompt(
        claim_text=claim_text,
        visual_caption=visual_caption,
        transcript_excerpt=segment["transcript"],
        start_ts=segment["start_ts"],
        end_ts=segment["end_ts"],
        conflict_flag=conflict_flag,
        claim_id=claim_id,
        rationale_hint=rationale_hint,
        max_sub_questions=max_sub_questions,
    )

    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            raw = llm.generate(prompt, max_new_tokens=512)
            data = _safe_json_parse(raw)
            print(f"LLM raw output (attempt {attempt + 1}): {raw}, parsed as {data}")
            sub_questions = [
                {
                    "hop": sq["hop"],
                    "question": sq["question"],
                    "depends_on_hops": sq.get("depends_on_hops", []),
                    "evidence_type": sq.get("evidence_type", "any"),
                }
                for sq in data.get("sub_questions", [])
            ]

            if not sub_questions:
                raise ValueError("LLM returned zero sub-questions.")

            # Hard-cap: silently drop any excess sub-questions
            sub_questions = sub_questions[:max_sub_questions]

            return {
                "claim_id": data.get("claim_id", claim_id),
                "claim_text": claim_text,
                "segment_id": segment["segment_id"],
                "sub_questions": sub_questions,
            }

        except Exception as exc:
            last_exc = exc
            logger.warning("Claim decomposition attempt %d/%d failed: %s", attempt + 1, max_retries + 1, exc)

    # Fallback: single sub-question covering the whole claim
    logger.error("All decomposition attempts failed (%s). Using fallback single-hop.", last_exc)
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
