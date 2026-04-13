"""
pipeline.py
-----------
Full pipeline orchestrator.

run_pipeline() is the single entry point. It wires all seven modules in
the correct order and returns an ExplainabilityReport — the sole output
visible to callers.

Usage
-----
    from pipeline import run_pipeline
    from models import load_default_bundle
    from modules import DenseRetriever

    models   = load_default_bundle()
    retriever = DenseRetriever(models.encoder)
    retriever.index(my_passage_corpus)

    report = run_pipeline(
        claim_text="The bridge collapsed on 14 January 2024.",
        claim_id="claim-001",
        segment=my_segment,
        initial_evidence=my_initial_evidence,
        models=models,
        retriever=retriever,
    )
    print(report.model_dump_json(indent=2))
"""

from __future__ import annotations

import logging

from schemas import EvidenceRef, ExplainabilityReport, VideoSegment
from models import ModelBundle
from modules import (
    DenseRetriever,
    MAX_RETRIEVAL_ROUNDS,
    aggregate_verdict,
    build_explainability_report,
    compute_modal_consistency,
    decompose_claim,
    gated_retrieval_loop,
    run_multihop,
)

logger = logging.getLogger(__name__)


def run_pipeline(
    claim_text: str,
    claim_id: str,
    segment: VideoSegment,
    initial_evidence: list[EvidenceRef],
    models: ModelBundle,
    retriever: DenseRetriever,
) -> ExplainabilityReport:
    """
    Execute the full fact-checking pipeline.

    Parameters
    ----------
    claim_text        : The claim string to verify.
    claim_id          : Stable identifier for this claim.
    segment           : VideoSegment containing transcript and keyframe paths.
    initial_evidence  : Pre-given evidence passages from the dataset.
    models            : ModelBundle with all loaded HuggingFace models.
    retriever         : DenseRetriever pre-indexed on a passage corpus.

    Returns
    -------
    ExplainabilityReport
        The sole public output. Contains the verdict, confidence, evidence
        saliency, modal annotations, hop summaries, and counterfactual.
    """
    logger.info("=== Pipeline start | claim_id=%s segment=%s ===", claim_id, segment.segment_id)

    # ------------------------------------------------------------------
    # Step 1 — Visual captioning
    # Compute a single visual caption from keyframes.
    # ------------------------------------------------------------------
    logger.info("[1/7] Visual captioning")
    visual_caption: str = models.caption_fn(segment.keyframes)
    print("DEBUG: Visual caption:\n", visual_caption)  # Debug print
    logger.debug("Visual caption: %s", visual_caption[:120])

    # ------------------------------------------------------------------
    # Step 2 — Cross-modal consistency
    # Score all three modality pairs (V↔C, T↔C, V↔T) with NLI.
    # ------------------------------------------------------------------
    logger.info("[2/7] Cross-modal consistency")
    modal_report = compute_modal_consistency(
        claim_text=claim_text,
        visual_caption=visual_caption,
        transcript=segment.transcript,
        segment_id=segment.segment_id,
        nli=models.nli,
    )

    # ------------------------------------------------------------------
    # Step 3 — Claim decomposition
    # Break the composite claim into ordered atomic sub-questions.
    # ------------------------------------------------------------------
    logger.info("[3/7] Claim decomposition")
    claim = decompose_claim(
        claim_text=claim_text,
        claim_id=claim_id,
        segment=segment,
        visual_caption=visual_caption,
        conflict_flag=modal_report.conflict_flag,
        llm=models.decomposer_llm,
    )
    logger.info("Decomposed into %d sub-questions.", len(claim.sub_questions))

    # ------------------------------------------------------------------
    # Step 4 — Gated evidence retrieval
    # Score initial evidence; retrieve targeted passages for weak aspects.
    # ------------------------------------------------------------------
    logger.info("[4/7] Gated evidence retrieval")
    evidence, strength_report = gated_retrieval_loop(
        claim=claim,
        segment=segment,
        evidence=list(initial_evidence),   # work on a copy
        modal_report=modal_report,
        retriever=retriever,
        nli=models.nli,
    )
    retrieval_rounds = 0 if strength_report.gate_pass else MAX_RETRIEVAL_ROUNDS
    logger.info(
        "Evidence gate: pass=%s | coverage=%.2f confidence=%.2f consistency=%.2f",
        strength_report.gate_pass,
        strength_report.coverage_score,
        strength_report.confidence_score,
        strength_report.consistency_score,
    )

    # ------------------------------------------------------------------
    # Step 5 — Multi-hop reasoning
    # Execute sub-questions sequentially; each hop conditions the next.
    # ------------------------------------------------------------------
    logger.info("[5/7] Multi-hop reasoning")
    hop_results = run_multihop(
        claim=claim,
        evidence=evidence,
        segment=segment,
        llm=models.hop_llm,
        retriever=retriever,
    )
    known = sum(1 for h in hop_results if not h.answer_unknown)
    logger.info("Completed %d/%d hops.", known, len(hop_results))

    # ------------------------------------------------------------------
    # Step 6 — Verdict aggregation
    # Synthesise all signals into a labelled verdict with reasoning trace.
    # ------------------------------------------------------------------
    logger.info("[6/7] Verdict aggregation")
    verdict = aggregate_verdict(
        claim=claim,
        segment=segment,
        visual_caption=visual_caption,
        hop_results=hop_results,
        modal_report=modal_report,
        strength_report=strength_report,
        retrieval_rounds=retrieval_rounds,
        llm=models.aggregator_llm,
    )
    logger.info("Verdict: %s (confidence=%.2f)", verdict.verdict, verdict.confidence)

    # ------------------------------------------------------------------
    # Step 7 — Explainability
    # Produce the structured explanation report returned to the caller.
    # ------------------------------------------------------------------
    logger.info("[7/7] Explainability report")
    report = build_explainability_report(
        verdict=verdict,
        hop_results=hop_results,
        evidence=evidence,
        modal_report=modal_report,
        segment=segment,
        nli=models.nli,
        llm=models.hop_llm,
    )

    logger.info("=== Pipeline complete | verdict=%s ===", report.verdict)
    return report
