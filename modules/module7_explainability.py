"""
modules/module7_explainability.py
----------------------------------
Module 7 — Explainability Module

The single accountability point for all transparency outputs. Transforms
internal pipeline objects into a structured ExplainabilityReport consumable
by UIs, auditors, and evaluation scripts without access to intermediate state.

Three computations happen here:

1. Evidence saliency
   For each hop's supporting passages, compute a normalised NLI entailment
   score against the verdict label. The highest-overlap sentence is extracted
   as the key span.

2. Modal annotations
   Each conflicting modality pair (V↔C, T↔C, V↔T) is rendered as a
   timestamp-anchored, plain-language annotation.

3. Hop summaries
   The per-hop reader LLM (same model as Module 5, no extra cost) writes
   one plain-language sentence per hop. Unknown hops get a canned fallback.

The counterfactual, gate_passed, and retrieval_rounds are passed through
from FinalVerdict unchanged.

Input  : FinalVerdict, list[HopResult], list[EvidenceRef],
         ModalConflictReport, VideoSegment, NLIScorer, GenerativeLLM
Output : ExplainabilityReport
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from schemas import (
    EvidenceRef,
    EvidenceSaliency,
    ExplainabilityReport,
    FinalVerdict,
    HopResult,
    ModalAnnotation,
    ModalConflictReport,
    VideoSegment,
)
from models import GenerativeLLM, NLIScorer

logger = logging.getLogger(__name__)

NLI_CONFLICT_FLOOR: float = 0.40


# ---------------------------------------------------------------------------
# 1. Evidence saliency
# ---------------------------------------------------------------------------

def compute_evidence_saliency(
    hop_results: list[HopResult],
    evidence: list[EvidenceRef],
    verdict_label: str,
    nli: NLIScorer,
) -> list[EvidenceSaliency]:
    """
    For each evidence passage cited in a hop, score its entailment against the
    verdict label and normalise within the hop. Extract the most salient sentence
    as the key span.
    """
    evidence_map: dict[str, EvidenceRef] = {e.evidence_id: e for e in evidence}
    saliency_list: list[EvidenceSaliency] = []

    for hop in hop_results:
        if not hop.supported_by:
            continue

        raw_scores: dict[str, float] = {}
        for eid in hop.supported_by:
            if eid not in evidence_map:
                logger.debug("Evidence %s not found in evidence pool — skipping.", eid)
                continue
            passage = evidence_map[eid].passage_text
            raw_scores[eid] = nli.entailment_score(passage, verdict_label)

        total = sum(raw_scores.values()) or 1.0

        for eid, score in raw_scores.items():
            passage_text = evidence_map[eid].passage_text
            # Key span: sentence with highest token overlap with the verdict label
            sentences = [s.strip() for s in passage_text.split(".") if s.strip()]
            verdict_tokens = set(verdict_label.lower().split())
            key_span = max(
                sentences,
                key=lambda s: len(set(s.lower().split()) & verdict_tokens),
                default=sentences[0] if sentences else passage_text[:120],
            )

            saliency_list.append(
                EvidenceSaliency(
                    evidence_id=eid,
                    hop=hop.hop,
                    saliency_score=round(score / total, 4),
                    key_span=key_span,
                )
            )

    return saliency_list


# ---------------------------------------------------------------------------
# 2. Modal annotations
# ---------------------------------------------------------------------------

_PAIR_LABELS: dict[str, str] = {
    "V↔C": "Visual content contradicts claim",
    "T↔C": "Transcript contradicts claim",
    "V↔T": "Visual content contradicts transcript",
}


def build_modal_annotations(
    modal_report: ModalConflictReport,
    segment: VideoSegment,
    conflict_floor: float = NLI_CONFLICT_FLOOR,
) -> list[ModalAnnotation]:
    """
    Convert the ModalConflictReport into a list of timestamped plain-language
    annotations — one per conflicting pair.
    """
    pair_scores: dict[str, float] = {
        "V↔C": modal_report.vc_score,
        "T↔C": modal_report.tc_score,
        "V↔T": modal_report.vt_score,
    }
    annotations: list[ModalAnnotation] = []

    for pair, score in pair_scores.items():
        if score < conflict_floor:
            label = _PAIR_LABELS.get(pair, pair)
            annotations.append(
                ModalAnnotation(
                    pair=pair,
                    score=round(score, 4),
                    timestamp=segment.start_ts,
                    human_note=(
                        f"{label} at {segment.start_ts}s "
                        f"({pair} NLI score: {score:.2f})."
                    ),
                )
            )

    return annotations


# ---------------------------------------------------------------------------
# 3. Hop summaries
# ---------------------------------------------------------------------------

_SUMMARY_SYSTEM_PROMPT = """\
You are an explainability assistant for a video fact-checking system.
Given one intermediate reasoning answer and its supporting evidence IDs,
write exactly one plain-language sentence that a non-expert could read.
Do not use jargon. Do not start with "I".

You MUST respond strictly with the following JSON format and nothing else. Do not provide any conversational text before or after the JSON:
{ "summary": "<one sentence>" }"""


def _build_summary_prompt(hop: HopResult) -> list[dict[str, str]]:
    user_content = (
        f'Hop {hop.hop}: Q: "{hop.question}"\n'
        f'Answer: "{hop.answer}" (confidence: {hop.confidence:.2f})\n'
        f"Supporting evidence: {hop.supported_by}\n\n"
        "Summarise in one sentence. Output ONLY valid JSON. Start your response with { and end with }."
    )
    return [
        {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]


def _safe_json_parse(raw: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())
    start = cleaned.find("{")
    end   = cleaned.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON found in: {raw[:200]}")
    return json.loads(cleaned[start:end])


def generate_hop_summaries(
    hop_results: list[HopResult],
    llm: GenerativeLLM,
) -> list[str]:
    """
    Generate one plain-language sentence per hop.
    Unknown hops receive a deterministic fallback (no LLM call).
    """
    summaries: list[str] = []

    for hop in hop_results:
        if hop.answer_unknown:
            summaries.append(
                f"Hop {hop.hop} could not be resolved: "
                f'no sufficient evidence was found for "{hop.question}".'
            )
            continue

        try:
            prompt = _build_summary_prompt(hop)
            raw = llm.generate(prompt, max_new_tokens=128)
            data = _safe_json_parse(raw)
            summaries.append(data.get("summary", hop.answer))
        except Exception as exc:
            logger.warning("Hop %d summary generation failed: %s — using answer as fallback.", hop.hop, exc)
            summaries.append(hop.answer)

    return summaries


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_explainability_report(
    verdict: FinalVerdict,
    hop_results: list[HopResult],
    evidence: list[EvidenceRef],
    modal_report: ModalConflictReport,
    segment: VideoSegment,
    nli: NLIScorer,
    llm: GenerativeLLM,
) -> ExplainabilityReport:
    """
    Build the final ExplainabilityReport — the sole output of the pipeline.

    Parameters
    ----------
    verdict      : FinalVerdict from Module 6.
    hop_results  : All HopResults from Module 5.
    evidence     : Full evidence pool (initial + retrieved).
    modal_report : ModalConflictReport from Module 2.
    segment      : Source VideoSegment.
    nli          : NLI scorer (DeBERTa-v3-small) — reused, no extra load.
    llm          : Per-hop reader LLM — reused for hop summarisation.

    Returns
    -------
    ExplainabilityReport
    """
    logger.info("Building explainability report for claim %s.", verdict.claim_id)

    saliency    = compute_evidence_saliency(hop_results, evidence, verdict.verdict, nli)
    annotations = build_modal_annotations(modal_report, segment)
    summaries   = generate_hop_summaries(hop_results, llm)

    return ExplainabilityReport(
        claim_id=verdict.claim_id,
        segment_id=verdict.segment_id,
        verdict=verdict.verdict,
        confidence=verdict.confidence,
        evidence_saliency=saliency,
        modal_annotations=annotations,
        hop_summaries=summaries,
        counterfactual=verdict.counterfactual,
        gate_passed=verdict.gate_passed,
        retrieval_rounds=verdict.retrieval_rounds,
    )
