from __future__ import annotations

import logging

from models import GenerativeLLM
from modules.utils import safe_json_parse as _safe_json_parse

logger = logging.getLogger(__name__)

NLI_CONFLICT_FLOOR: float = 0.40


def compute_evidence_saliency(
    hop_results: list[dict],
    evidence: list[dict],
    verdict_label: str,
    llm: GenerativeLLM,
) -> list[dict]:
    evidence_map: dict[str, dict] = {e["evidence_id"]: e for e in evidence}
    saliency_list: list[dict] = []

    score_system_prompt = """\
You are scoring how much one evidence passage supports a final verdict label.
Return ONLY valid JSON:
{"score": <float in [0.0, 1.0]>}
"""

    for hop in hop_results:
        if not hop.get("supported_by"):
            continue

        raw_scores: dict[str, float] = {}
        for eid in hop.get("supported_by", []):
            if eid not in evidence_map:
                logger.debug("Evidence %s not found in evidence pool - skipping.", eid)
                continue

            passage = evidence_map[eid].get("passage_text", "")
            prompt = [
                {"role": "system", "content": score_system_prompt},
                {
                    "role": "user",
                    "content": (
                        f'Verdict label: "{verdict_label}"\n'
                        f'Passage: "{passage[:2000]}"\n'
                        "Output JSON only."
                    ),
                },
            ]
            try:
                data = _safe_json_parse(llm.generate(prompt, max_new_tokens=96))
                if not isinstance(data, dict):
                    score = 0.0
                else:
                    score = float(data.get("score", 0.0))
            except Exception:
                score = 0.0
            raw_scores[eid] = max(0.0, min(1.0, score))

        total = sum(raw_scores.values()) or 1.0
        for eid, score in raw_scores.items():
            passage_text = evidence_map[eid].get("passage_text", "")
            sentences = [s.strip() for s in passage_text.split(".") if s.strip()]
            verdict_tokens = set(verdict_label.lower().split())
            key_span = max(
                sentences,
                key=lambda s: len(set(s.lower().split()) & verdict_tokens),
                default=sentences[0] if sentences else passage_text[:120],
            )
            saliency_list.append(
                {
                    "evidence_id": eid,
                    "hop": hop["hop"],
                    "saliency_score": round(score / total, 4),
                    "key_span": key_span,
                }
            )

    return saliency_list


_PAIR_LABELS: dict[str, str] = {
    "V↔C": "Visual content contradicts claim",
    "T↔C": "Transcript contradicts claim",
    "V↔T": "Visual content contradicts transcript",
}


def build_modal_annotations(
    modal_report: dict,
    segment: dict,
    conflict_floor: float = NLI_CONFLICT_FLOOR,
) -> list[dict]:
    pair_scores: dict[str, float] = {
        "V↔C": modal_report.get("vc_score", 0.0),
        "T↔C": modal_report.get("tc_score", 0.0),
        "V↔T": modal_report.get("vt_score", 0.0),
    }
    annotations: list[dict] = []

    for pair, score in pair_scores.items():
        if score < conflict_floor:
            label = _PAIR_LABELS.get(pair, pair)
            annotations.append(
                {
                    "pair": pair,
                    "score": round(score, 4),
                    "timestamp": segment["start_ts"],
                    "human_note": (
                        f"{label} at {segment['start_ts']}s "
                        f"({pair} consistency score: {score:.2f})."
                    ),
                }
            )

    return annotations


_SUMMARY_SYSTEM_PROMPT = """\
You are an explainability assistant for a video fact-checking system.
Given one intermediate reasoning answer and its supporting evidence IDs,
write exactly one plain-language sentence that a non-expert could read.

Respond ONLY with valid JSON:
{"summary": "<exactly one sentence>"}
"""


def _build_summary_prompt(hop: dict) -> list[dict[str, str]]:
    user_content = (
        f'Hop {hop["hop"]}: Q: "{hop["question"]}"\n'
        f'Answer: "{hop["answer"]}" (confidence: {hop["confidence"]:.2f})\n'
        f"Supporting evidence: {hop.get('supported_by')}\n\n"
        "Summarise in one sentence. Output JSON only."
    )
    return [
        {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def generate_hop_summaries(
    hop_results: list[dict],
    llm: GenerativeLLM,
) -> list[str]:
    summaries: list[str] = []

    for hop in hop_results:
        if hop["answer_unknown"]:
            summaries.append(
                f"Hop {hop['hop']} could not be resolved: "
                f'no sufficient evidence was found for "{hop["question"]}".'
            )
            continue

        try:
            prompt = _build_summary_prompt(hop)
            raw = llm.generate(prompt, max_new_tokens=128)
            data = _safe_json_parse(raw)
            if not isinstance(data, dict):
                summaries.append(hop["answer"])
            else:
                summaries.append(data.get("summary", hop["answer"]))
        except Exception as exc:
            logger.warning(
                "Hop %d summary generation failed: %s - using answer as fallback.",
                hop["hop"],
                exc,
            )
            summaries.append(hop["answer"])

    return summaries


def build_explainability_report(
    verdict: dict,
    hop_results: list[dict],
    evidence: list[dict],
    modal_report: dict,
    segment: dict,
    llm: GenerativeLLM,
) -> dict:
    logger.info("Building explainability report for claim %s.", verdict["claim_id"])

    saliency = compute_evidence_saliency(hop_results, evidence, verdict["verdict"], llm)
    annotations = build_modal_annotations(modal_report, segment)
    summaries = generate_hop_summaries(hop_results, llm)

    return {
        "claim_id": verdict["claim_id"],
        "segment_id": verdict["segment_id"],
        "verdict": verdict["verdict"],
        "confidence": verdict["confidence"],
        "evidence_saliency": saliency,
        "modal_annotations": annotations,
        "hop_summaries": summaries,
        "counterfactual": verdict["counterfactual"],
        "gate_passed": verdict["gate_passed"],
        "retrieval_rounds": verdict["retrieval_rounds"],
    }
