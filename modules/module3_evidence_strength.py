from __future__ import annotations

import logging

from models import GenerativeLLM
from modules.utils import safe_json_parse as _safe_json_parse

logger = logging.getLogger(__name__)


THRESHOLDS: dict[str, float] = {
    "coverage": 0.75,
    "confidence": 0.65,
    "consistency": 0.60,
    "min_hop_confidence": 0.50,
    "nli_conflict_floor": 0.40,
}

_EVIDENCE_SCORE_SYSTEM_PROMPT = """\
You are an evidence relevance scorer for fact-checking.
Given one question and one evidence passage, score how strongly the passage supports answering the question.

Return ONLY valid JSON:
{
  "score": <float in [0.0, 1.0]>
}
"""


def _clamp_01(value: object) -> float:
    try:
        score = float(value)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, score))


def _llm_passage_score(
    llm: GenerativeLLM,
    question: str,
    passage_text: str,
    max_retries: int = 1,
) -> float:
    prompt = [
        {"role": "system", "content": _EVIDENCE_SCORE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f'Question: "{question[:500]}"\n'
                f'Passage: "{passage_text[:2000]}"\n'
                "Output JSON only."
            ),
        },
    ]

    for _ in range(max_retries + 1):
        try:
            raw = llm.generate(prompt, max_new_tokens=96)
            data = _safe_json_parse(raw)
            if not isinstance(data, dict):
                continue
            return _clamp_01(data.get("score"))
        except Exception:
            continue
    return 0.0


def score_evidence(
    claim: dict,
    evidence: list[dict],
    modal_report: dict,
    llm: GenerativeLLM,
    thresholds: dict[str, float] = THRESHOLDS,
) -> dict:
    hop_scores: dict[int, float] = {}

    for sq in claim["sub_questions"]:
        relevant = [e for e in evidence if sq["hop"] in e.get("hop_ids", [])]
        if not relevant:
            hop_scores[sq["hop"]] = 0.0
            logger.debug("Hop %d: no relevant evidence.", sq["hop"])
            continue

        per_passage_scores = [
            _llm_passage_score(llm, sq["question"], e.get("passage_text", ""))
            for e in relevant
        ]
        hop_scores[sq["hop"]] = max(per_passage_scores) if per_passage_scores else 0.0
        logger.debug("Hop %d: best evidence score=%.3f", sq["hop"], hop_scores[sq["hop"]])

    n = len(hop_scores)
    if n == 0:
        logger.warning(
            "No hops found in claim %s - returning zero-score report.", claim["claim_id"]
        )
        return {
            "claim_id": claim["claim_id"],
            "coverage_score": 0.0,
            "confidence_score": 0.0,
            "consistency_score": 0.0,
            "gate_pass": False,
            "weak_aspects": [sq["question"] for sq in claim["sub_questions"]],
        }

    coverage = sum(
        1 for s in hop_scores.values() if s > thresholds["min_hop_confidence"]
    ) / n
    confidence = sum(hop_scores.values()) / n
    consistency = min(
        modal_report.get("vc_score", 0.0),
        modal_report.get("tc_score", 0.0),
        modal_report.get("vt_score", 0.0),
    )

    gate_pass = (
        coverage >= thresholds["coverage"]
        and confidence >= thresholds["confidence"]
        and consistency >= thresholds["consistency"]
    )

    hop_index = {sq["hop"]: sq for sq in claim["sub_questions"]}
    weak_aspects = [
        hop_index[hop]["question"]
        for hop, s in hop_scores.items()
        if s < thresholds["min_hop_confidence"]
    ]

    logger.info(
        "Claim %s - coverage=%.2f confidence=%.2f consistency=%.2f gate=%s weak=%d",
        claim["claim_id"],
        coverage,
        confidence,
        consistency,
        gate_pass,
        len(weak_aspects),
    )

    return {
        "claim_id": claim["claim_id"],
        "coverage_score": round(coverage, 4),
        "confidence_score": round(confidence, 4),
        "consistency_score": round(consistency, 4),
        "gate_pass": gate_pass,
        "weak_aspects": weak_aspects,
    }
