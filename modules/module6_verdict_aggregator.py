"""
modules/module6_verdict_aggregator.py
--------------------------------------
Module 6 — Verdict Aggregator

Synthesises all hop answers, the modal conflict report, and the evidence
strength report into a final labelled verdict with a step-by-step reasoning
trace and a counterfactual.

Uses Mistral-7B or LLaMA-3.1-8B — the only module that needs a larger
context window because it must hold all hop answers simultaneously.

Possible verdicts:
  "supported"              — evidence and modalities consistently support the claim
  "refuted"                — evidence clearly contradicts the claim
  "insufficient_evidence"  — gate never passed; verdict unreliable
  "misleading_context"     — claim is technically accurate but omits key context

Input  : ClaimDecomposition, VideoSegment, visual_caption, list[HopResult],
         ModalConflictReport, EvidenceStrengthReport, retrieval_rounds,
         GenerativeLLM
Output : FinalVerdict
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from schemas import (
    ClaimDecomposition,
    EvidenceStrengthReport,
    FinalVerdict,
    HopResult,
    ModalConflictReport,
    ReasoningStep,
    VideoSegment,
)
from models import GenerativeLLM

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

_AGGREGATOR_SYSTEM_PROMPT = """\
You are a fact-checking verdict aggregator.
You receive the original composite claim, intermediate answers from N reasoning
hops, a cross-modal conflict report, and an evidence gate status.
Produce a single final verdict with a step-by-step reasoning trace and a
counterfactual statement.

Respond ONLY with valid JSON — no markdown fences, no preamble:
{
  "claim_id": "<string>",
  "verdict": "supported" | "refuted" | "insufficient_evidence" | "misleading_context",
  "confidence": <float 0.0–1.0>,
  "reasoning_trace": [
    {
      "step": <integer starting at 1>,
      "finding": "<string>",
      "source_hop": <integer | null>,
      "evidence_ids": ["<string>", ...]
    }
  ],
  "modal_conflict_used": <true | false>,
  "counterfactual": "<string — what would need to change for the verdict to flip>"
}"""


def _format_hop_answers(hop_results: list[HopResult]) -> str:
    lines = []
    for h in hop_results:
        status = "UNKNOWN" if h.answer_unknown else f"confidence={h.confidence:.2f}"
        lines.append(
            f'Hop {h.hop}: Q: "{h.question}" → A: "{h.answer}" ({status})'
            f' supported_by={h.supported_by}'
        )
    return "\n".join(lines) if lines else "No hop answers available."


def _build_aggregator_prompt(
    claim: ClaimDecomposition,
    segment: VideoSegment,
    visual_caption: str,
    hop_results: list[HopResult],
    modal_report: ModalConflictReport,
    strength_report: EvidenceStrengthReport,
) -> list[dict[str, str]]:
    user_content = (
        f'Original claim: "{claim.claim_text}"\n'
        f'Claim ID: "{claim.claim_id}"\n'
        f'Visual caption: "{visual_caption}"\n'
        f"Timestamp: {segment.start_ts}s – {segment.end_ts}s\n\n"
        f"Hop answers:\n{_format_hop_answers(hop_results)}\n\n"
        f"Cross-modal conflict report:\n"
        f"  Visual↔Claim score:      {modal_report.vc_score:.4f}\n"
        f"  Transcript↔Claim score:  {modal_report.tc_score:.4f}\n"
        f"  Visual↔Transcript score: {modal_report.vt_score:.4f}\n"
        f"  Conflict flag:            {modal_report.conflict_flag}\n"
        f"  Dominant conflict:        {modal_report.dominant_conflict}\n\n"
        f"Evidence gate passed: {strength_report.gate_pass}\n"
        f"Coverage score:       {strength_report.coverage_score:.4f}\n"
        f"Confidence score:     {strength_report.confidence_score:.4f}\n"
        f"Consistency score:    {strength_report.consistency_score:.4f}\n\n"
        "Produce the verdict."
    )
    return [
        {"role": "system", "content": _AGGREGATOR_SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def _safe_json_parse(raw: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())
    start = cleaned.find("{")
    end   = cleaned.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON found in aggregator output: {raw[:300]}")
    return json.loads(cleaned[start:end])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def aggregate_verdict(
    claim: ClaimDecomposition,
    segment: VideoSegment,
    visual_caption: str,
    hop_results: list[HopResult],
    modal_report: ModalConflictReport,
    strength_report: EvidenceStrengthReport,
    retrieval_rounds: int,
    llm: GenerativeLLM,
    max_retries: int = 2,
) -> FinalVerdict:
    """
    Synthesise all pipeline signals into a final verdict.

    Parameters
    ----------
    claim             : Decomposed claim from Module 1.
    segment           : Source video segment.
    visual_caption    : Caption from the visual captioner.
    hop_results       : All completed hop answers from Module 5.
    modal_report      : Cross-modal consistency report from Module 2.
    strength_report   : Evidence strength report from Module 3/4.
    retrieval_rounds  : Number of retrieval rounds consumed (for transparency).
    llm               : Aggregator LLM (Mistral-7B or LLaMA-3.1-8B).
    max_retries       : Parse retry budget.

    Returns
    -------
    FinalVerdict
    """
    # If gate never passed, emit insufficient_evidence without calling the LLM
    if not strength_report.gate_pass:
        logger.warning(
            "Claim %s: gate_pass=False after %d retrieval rounds. "
            "Emitting insufficient_evidence verdict.",
            claim.claim_id, retrieval_rounds,
        )
        return FinalVerdict(
            claim_id=claim.claim_id,
            segment_id=segment.segment_id,
            verdict="insufficient_evidence",
            confidence=0.0,
            reasoning_trace=[
                ReasoningStep(
                    step=1,
                    finding=(
                        f"Evidence gate did not pass after {retrieval_rounds} "
                        f"retrieval round(s). Weak aspects: "
                        f"{'; '.join(strength_report.weak_aspects)}"
                    ),
                    source_hop=None,
                    evidence_ids=[],
                )
            ],
            modal_conflict_used=modal_report.conflict_flag,
            counterfactual=(
                "The verdict could change if sufficient evidence were found for: "
                + "; ".join(strength_report.weak_aspects)
            ),
            retrieval_rounds=retrieval_rounds,
            gate_passed=False,
        )

    prompt = _build_aggregator_prompt(
        claim, segment, visual_caption,
        hop_results, modal_report, strength_report,
    )

    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            raw = llm.generate(prompt, max_new_tokens=1024)
            data = _safe_json_parse(raw)

            reasoning_trace = [
                ReasoningStep(
                    step=rs["step"],
                    finding=rs["finding"],
                    source_hop=rs.get("source_hop"),
                    evidence_ids=rs.get("evidence_ids", []),
                )
                for rs in data.get("reasoning_trace", [])
            ]

            return FinalVerdict(
                claim_id=data.get("claim_id", claim.claim_id),
                segment_id=segment.segment_id,
                verdict=data.get("verdict", "insufficient_evidence"),
                confidence=float(data.get("confidence", 0.0)),
                reasoning_trace=reasoning_trace,
                modal_conflict_used=bool(data.get("modal_conflict_used", False)),
                counterfactual=data.get("counterfactual", ""),
                retrieval_rounds=retrieval_rounds,
                gate_passed=strength_report.gate_pass,
            )

        except Exception as exc:
            last_exc = exc
            logger.warning(
                "Aggregator attempt %d/%d failed: %s", attempt + 1, max_retries + 1, exc
            )

    # Fallback on complete failure
    logger.error("All aggregator attempts failed (%s). Using fallback verdict.", last_exc)
    return FinalVerdict(
        claim_id=claim.claim_id,
        segment_id=segment.segment_id,
        verdict="insufficient_evidence",
        confidence=0.0,
        reasoning_trace=[
            ReasoningStep(
                step=1,
                finding=f"Aggregator LLM failed to produce a valid verdict: {last_exc}",
                source_hop=None,
                evidence_ids=[],
            )
        ],
        modal_conflict_used=modal_report.conflict_flag,
        counterfactual="",
        retrieval_rounds=retrieval_rounds,
        gate_passed=strength_report.gate_pass,
    )
