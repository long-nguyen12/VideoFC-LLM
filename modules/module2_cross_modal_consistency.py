"""
modules/module2_cross_modal_consistency.py
------------------------------------------
Module 2 — Cross-Modal Consistency

Checks entailment consistency across three modality pairs using DeBERTa-v3-small:
  V↔C  visual caption  ↔ claim
  T↔C  transcript      ↔ claim
  V↔T  visual caption  ↔ transcript

A conflict is flagged when any pair's entailment score falls below the
nli_conflict_floor threshold. The lowest-scoring pair is reported as the
dominant conflict and fed into Module 1 (claim decomposer), Module 3
(evidence scorer), Module 6 (aggregator), and Module 7 (explainability).

Input  : claim_text, visual_caption, transcript, segment_id, NLIScorer
Output : ModalConflictReport
"""

from __future__ import annotations

import logging

from models import NLIScorer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Threshold (mirrors the global THRESHOLDS dict in the framework spec)
# ---------------------------------------------------------------------------

NLI_CONFLICT_FLOOR: float = 0.40


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_modal_consistency(
    claim_text: str,
    visual_caption: str,
    transcript: str,
    segment_id: str,
    nli: NLIScorer,
    conflict_floor: float = NLI_CONFLICT_FLOOR,
) -> dict:
    """
    Score cross-modal consistency and return a ModalConflictReport.

    Parameters
    ----------
    claim_text     : The claim to be fact-checked.
    visual_caption : Caption generated from keyframes (Module 0 / caption_fn).
    transcript     : Pre-annotated transcript from the dataset.
    segment_id     : ID of the video segment, for traceability.
    nli            : A loaded NLIScorer (DeBERTa-v3-small).
    conflict_floor : Entailment score below which a pair is considered conflicting.

    Returns
    -------
    dict
    """
    logger.debug("Computing cross-modal consistency for segment %s", segment_id)

    # Score all three pairs
    vc = nli.entailment_score(visual_caption, claim_text)
    tc = nli.entailment_score(transcript, claim_text)
    vt = nli.entailment_score(visual_caption, transcript)

    scores: dict[str, float] = {
        "V↔C": vc,
        "T↔C": tc,
        "V↔T": vt,
    }

    conflict_flag = any(s < conflict_floor for s in scores.values())
    dominant_conflict: str | None = None

    if conflict_flag:
        dominant_conflict = min(scores, key=scores.get)  # type: ignore[arg-type]
        logger.info(
            "Segment %s: modal conflict detected. Dominant=%s (score=%.3f)",
            segment_id, dominant_conflict, scores[dominant_conflict],
        )
    else:
        logger.debug("Segment %s: no modal conflict. Scores=%s", segment_id, scores)

    return {
        "segment_id": segment_id,
        "vc_score": round(vc, 4),
        "tc_score": round(tc, 4),
        "vt_score": round(vt, 4),
        "conflict_flag": conflict_flag,
        "dominant_conflict": dominant_conflict,
    }
