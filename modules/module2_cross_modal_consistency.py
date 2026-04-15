from __future__ import annotations

import logging

from models import GenerativeLLM

logger = logging.getLogger(__name__)

# Entailment/consistency score floor below which a pair is considered conflicting.
NLI_CONFLICT_FLOOR: float = 0.40


def _clamp_01(value: object) -> float:
    try:
        score = float(value)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, score))


from modules.prompt_template import (
    _CONSISTENCY_MODAL_SYSTEM_PROMPT as _LLM_MODAL_SYSTEM_PROMPT,
)


def _build_modal_llm_prompt(
    claim_text: str,
    visual_caption: str,
    transcript: str,
    content: str = "",
) -> list[dict[str, str]]:
    # Escape double quotes in inputs to prevent JSON injection issues
    def _escape_for_prompt(s: str) -> str:
        return s.replace('"', '\\"').replace("\\\\", "\\")

    claim_esc = _escape_for_prompt(claim_text)
    visual_esc = _escape_for_prompt(visual_caption)
    transcript_esc = _escape_for_prompt(transcript)
    content_esc = _escape_for_prompt(content) if content.strip() else ""

    content_block = ""
    if content.strip():
        content_block = f'Article/content: "{content_esc}"\n\n'

    user_content = (
        f'Claim: "{claim_esc}"\n'
        f'Visual caption: "{visual_esc}"\n'
        f'Transcript: "{transcript_esc}"\n\n'
        f"{content_block}"
    )
    return [
        {"role": "system", "content": _LLM_MODAL_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def compute_modal_consistency_llm(
    claim_text: str,
    visual_caption: str,
    transcript: str,
    segment_id: str,
    llm: GenerativeLLM,
    content: str = "",
    conflict_floor: float = NLI_CONFLICT_FLOOR,
    max_retries: int = 2,
) -> dict:
    prompt = _build_modal_llm_prompt(
        claim_text=claim_text,
        visual_caption=visual_caption,
        transcript=transcript,
        content=content,
    )

    data = llm.generate_json(prompt, max_new_tokens=512, max_retries=max_retries)

    if data is None:
        logger.error(
            "All LLM modal consistency attempts failed for segment %s. "
            "Returning conservative conflict report.",
            segment_id,
        )
        return {
            "segment_id": segment_id,
            "vc_score": 0.0,
            "tc_score": 0.0,
            "vt_score": 0.0,
            "ac_score": 0.0 if content.strip() else None,
            "conflict_flag": True,
            "dominant_conflict": None,
        }

    vc = _clamp_01(data.get("vc_score"))
    tc = _clamp_01(data.get("tc_score"))
    vt = _clamp_01(data.get("vt_score"))

    ac_raw = data.get("ac_score")
    ac_score = _clamp_01(ac_raw) if ac_raw is not None else None

    scores: dict[str, float] = {
        "V↔C": vc,
        "T↔C": tc,
        "V↔T": vt,
    }
    if content.strip() and ac_score is not None:
        scores["A↔C"] = ac_score

    conflict_flag = any(s < conflict_floor for s in scores.values())
    dominant_conflict = min(scores, key=scores.get) if conflict_flag else None

    llm_dominant = data.get("dominant_conflict")
    if conflict_flag and llm_dominant in scores:
        dominant_conflict = llm_dominant

    return {
        "segment_id": segment_id,
        "vc_score": round(vc, 4),
        "tc_score": round(tc, 4),
        "vt_score": round(vt, 4),
        "ac_score": round(ac_score, 4) if ac_score is not None else None,
        "conflict_flag": conflict_flag,
        "dominant_conflict": dominant_conflict,
    }
