from __future__ import annotations

import logging

from models import GenerativeLLM, NLIScorer
from modules.utils import safe_json_parse as _safe_json_parse

logger = logging.getLogger(__name__)

# Entailment/consistency score floor below which a pair is considered conflicting.
NLI_CONFLICT_FLOOR: float = 0.40


_LLM_MODAL_SYSTEM_PROMPT = """\
You are a strict cross-modal consistency scorer for fact-checking.

Compare three pairs and score consistency in [0, 1]:
- V↔C: visual_caption vs claim
- T↔C: transcript vs claim
- V↔T: visual_caption vs transcript

If article/content text is provided, also score:
- A↔C: article_content vs claim

OUTPUT FORMAT RULES (MANDATORY):
1. Return ONLY valid JSON.
2. No markdown/code fences/explanations.
3. Use exactly these keys:
   - "vc_score": number in [0, 1]
   - "tc_score": number in [0, 1]
   - "vt_score": number in [0, 1]
   - "ac_score": number in [0, 1] or null
   - "dominant_conflict": one of "V↔C", "T↔C", "V↔T", "A↔C", or null
"""


def _clamp_01(value: object) -> float:
    try:
        score = float(value)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, score))


def _build_modal_llm_prompt(
    claim_text: str,
    visual_caption: str,
    transcript: str,
    content: str = "",
) -> list[dict[str, str]]:
    content_block = ""
    if content.strip():
        content_block = f'Article/content: "{content}"\n\n'

    user_content = (
        f'Claim: "{claim_text}"\n'
        f'Visual caption: "{visual_caption}"\n'
        f'Transcript: "{transcript}"\n\n'
        f"{content_block}"
        "Output JSON only."
    )
    return [
        {"role": "system", "content": _LLM_MODAL_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def compute_modal_consistency(
    claim_text: str,
    visual_caption: str,
    transcript: str,
    segment_id: str,
    nli: NLIScorer,
    conflict_floor: float = NLI_CONFLICT_FLOOR,
) -> dict:
    logger.debug("Computing cross-modal consistency for segment %s", segment_id)

    vc = nli.entailment_score(visual_caption, claim_text)
    tc = nli.entailment_score(transcript, claim_text)
    vt = nli.entailment_score(visual_caption, transcript)

    scores: dict[str, float] = {
        "V↔C": vc,
        "T↔C": tc,
        "V↔T": vt,
    }
    conflict_flag = any(s < conflict_floor for s in scores.values())
    dominant_conflict: str | None = (
        min(scores, key=scores.get) if conflict_flag else None
    )

    return {
        "segment_id": segment_id,
        "vc_score": round(vc, 4),
        "tc_score": round(tc, 4),
        "vt_score": round(vt, 4),
        "conflict_flag": conflict_flag,
        "dominant_conflict": dominant_conflict,
    }


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
    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            raw = llm.generate(prompt, max_new_tokens=512)
            data = _safe_json_parse(raw, context="module2_cross_modal_consistency")

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
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "LLM modal consistency attempt %d/%d failed for segment %s: %s",
                attempt + 1,
                max_retries + 1,
                segment_id,
                exc,
            )

    logger.error(
        "All LLM modal consistency attempts failed for segment %s (%s). "
        "Returning conservative conflict report.",
        segment_id,
        last_exc,
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
