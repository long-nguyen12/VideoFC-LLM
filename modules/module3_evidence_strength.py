"""
modules/module3_evidence_strength.py
-------------------------------------
Module 3 — Evidence Strength Scorer

Determines whether the current evidence pool is sufficient to support verdict
generation, without requiring any human sufficiency labels.

The gate is derived from three NLI-based metrics:
  coverage_score   — fraction of hops that have at least one evidence passage
                     with entailment score > min_hop_confidence
  confidence_score — mean max-entailment score across all hops
  consistency_score— min of vc_score, tc_score, vt_score from Module 2

All three must meet their respective thresholds for gate_pass = True.
Sub-questions that fall short are returned as weak_aspects, which are used
by Module 4 (targeted retrieval) to scope follow-up queries.

Input  : ClaimDecomposition, list[EvidenceRef], ModalConflictReport, NLIScorer
Output : EvidenceStrengthReport
"""

from __future__ import annotations

import logging

from schemas import (
    ClaimDecomposition,
    EvidenceRef,
    EvidenceStrengthReport,
    ModalConflictReport,
)
from models import NLIScorer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

THRESHOLDS: dict[str, float] = {
    "coverage":           0.75,   # fraction of hops needing strong evidence
    "confidence":         0.65,   # mean NLI entailment across passages
    "consistency":        0.60,   # cross-modal agreement floor
    "min_hop_confidence": 0.50,   # per-hop floor before flagging as weak
    "nli_conflict_floor": 0.40,   # below this → conflict_flag = True (Module 2)
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_evidence(
    claim: ClaimDecomposition,
    evidence: list[EvidenceRef],
    modal_report: ModalConflictReport,
    nli: NLIScorer,
    thresholds: dict[str, float] = THRESHOLDS,
) -> EvidenceStrengthReport:
    """
    Score the current evidence pool against all sub-questions.

    Parameters
    ----------
    claim        : Decomposed claim from Module 1.
    evidence     : Current evidence list (initial + any retrieved so far).
    modal_report : Cross-modal consistency report from Module 2.
    nli          : A loaded NLIScorer.
    thresholds   : Override defaults for tuning.

    Returns
    -------
    EvidenceStrengthReport
    """
    hop_scores: dict[int, float] = {}

    for sq in claim.sub_questions:
        relevant = [e for e in evidence if sq.hop in e.hop_ids]

        if not relevant:
            hop_scores[sq.hop] = 0.0
            logger.debug("Hop %d: no relevant evidence.", sq.hop)
            continue

        # Score each passage for this hop's question; keep the best
        per_passage_scores = [
            nli.entailment_score(e.passage_text, sq.question)
            for e in relevant
        ]
        hop_scores[sq.hop] = max(per_passage_scores)
        logger.debug("Hop %d: best evidence score=%.3f", sq.hop, hop_scores[sq.hop])

    n = len(hop_scores)
    if n == 0:
        logger.warning("No hops found in claim %s — returning zero-score report.", claim.claim_id)
        return EvidenceStrengthReport(
            claim_id=claim.claim_id,
            coverage_score=0.0,
            confidence_score=0.0,
            consistency_score=0.0,
            gate_pass=False,
            weak_aspects=[sq.question for sq in claim.sub_questions],
        )

    coverage = sum(
        1 for s in hop_scores.values() if s > thresholds["min_hop_confidence"]
    ) / n

    confidence = sum(hop_scores.values()) / n

    consistency = min(
        modal_report.vc_score,
        modal_report.tc_score,
        modal_report.vt_score,
    )

    gate_pass = (
        coverage    >= thresholds["coverage"]
        and confidence  >= thresholds["confidence"]
        and consistency >= thresholds["consistency"]
    )

    # Identify sub-questions whose best evidence is below the per-hop floor
    hop_index = {sq.hop: sq for sq in claim.sub_questions}
    weak_aspects = [
        hop_index[hop].question
        for hop, s in hop_scores.items()
        if s < thresholds["min_hop_confidence"]
    ]

    logger.info(
        "Claim %s — coverage=%.2f confidence=%.2f consistency=%.2f gate=%s weak=%d",
        claim.claim_id, coverage, confidence, consistency, gate_pass, len(weak_aspects),
    )

    return EvidenceStrengthReport(
        claim_id=claim.claim_id,
        coverage_score=round(coverage, 4),
        confidence_score=round(confidence, 4),
        consistency_score=round(consistency, 4),
        gate_pass=gate_pass,
        weak_aspects=weak_aspects,
    )
