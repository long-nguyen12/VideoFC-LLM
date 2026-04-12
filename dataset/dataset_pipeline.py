"""
dataset/dataset_pipeline.py
----------------------------
Dataset-aware pipeline entry point.

Wraps the core run_pipeline() with two dataset-specific adaptations:

1. Synthetic visual caption bypass
   The dataset does not supply keyframe images. When VideoSegment.keyframes
   is empty, the VLM captioner is skipped and the pre-built synthetic caption
   (headline + description from record_to_visual_caption) is used directly.
   This avoids a 7B model call for text-only records.

2. Rationale injection (optional, controlled by use_rationale_hints flag)
   The gold rationale from original_rationales is summarised and injected into
   Module 1 (claim decomposer) via the rationale_hint parameter. This guides
   the sub-question decomposition toward the known evidence structure, which
   improves hop alignment during training and supervised evaluation.
   Set use_rationale_hints=False for blind / test-set evaluation.

3. Evaluation helpers
   run_dataset_record() returns both the ExplainabilityReport and a
   DatasetEvalResult that records the gold label, predicted label, and
   whether the predicted verdict matches the gold verdict.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from schemas.data_models import EvidenceRef, ExplainabilityReport, VideoSegment
from models.model_bundle import ModelBundle
from modules.module1_claim_decomposer import decompose_claim
from modules.module2_cross_modal_consistency import compute_modal_consistency
from modules.module3_evidence_strength import score_evidence
from modules.module4_targeted_retrieval import DenseRetriever, gated_retrieval_loop
from modules.module5_multihop_reasoning import run_multihop
from modules.module6_verdict_aggregator import aggregate_verdict
from modules.module7_explainability import build_explainability_report
from modules.module4_targeted_retrieval import MAX_RETRIEVAL_ROUNDS

from dataset.dataset_schemas import DatasetRecord
from dataset.dataset_adapter import (
    PipelineInputs,
    RationaleContext,
    record_to_pipeline_inputs,
)
from dataset.label_mapper import verdict_to_label

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evaluation result
# ---------------------------------------------------------------------------

@dataclass
class DatasetEvalResult:
    """
    Pairs a pipeline output with its ground-truth label for evaluation.

    Attributes
    ----------
    claim_id        : Pipeline claim ID.
    gold_verdict    : Gold verdict from the dataset (via label_mapper).
    gold_label      : Integer class index of gold_verdict.
    pred_verdict    : Predicted verdict from the pipeline.
    pred_label      : Integer class index of pred_verdict.
    pred_confidence : Pipeline confidence score.
    correct         : True if pred_verdict == gold_verdict.
    report          : Full ExplainabilityReport.
    """
    claim_id: str
    gold_verdict: str
    gold_label: int
    pred_verdict: str
    pred_label: int
    pred_confidence: float
    correct: bool
    report: ExplainabilityReport


# ---------------------------------------------------------------------------
# Dataset-aware pipeline
# ---------------------------------------------------------------------------

def run_dataset_pipeline(
    inputs: PipelineInputs,
    models: ModelBundle,
    retriever: DenseRetriever,
    use_rationale_hints: bool = True,
    max_sub_questions: int = 5,
) -> ExplainabilityReport:
    """
    Execute the pipeline from a PipelineInputs object.

    Key differences from run_pipeline():
      - If inputs.segment.keyframes is empty, inputs.visual_caption is used
        directly (no VLM call).
      - If use_rationale_hints=True, the gold rationale summary is injected
        into Module 1 via rationale_hint.

    Parameters
    ----------
    inputs               : PipelineInputs from record_to_pipeline_inputs().
    models               : Loaded ModelBundle.
    retriever            : DenseRetriever pre-indexed on a passage corpus.
    use_rationale_hints  : Whether to inject gold rationale into Module 1.
                           Set False for blind test-set evaluation.
    max_sub_questions    : Cap on sub-questions from the decomposer.
                           Use 3 when running with a ≤2B single-model bundle.

    Returns
    -------
    ExplainabilityReport
    """
    segment = inputs.segment
    claim_text = inputs.claim_text
    claim_id = inputs.claim_id

    logger.info(
        "=== Dataset pipeline start | claim_id=%s segment=%s ===",
        claim_id, segment.segment_id,
    )

    # ------------------------------------------------------------------
    # Step 1 — Visual captioning (or synthetic bypass)
    # ------------------------------------------------------------------
    logger.info("[1/7] Visual captioning")
    if segment.keyframes:
        visual_caption = models.caption_fn(segment.keyframes)
        logger.debug("VLM caption: %s", visual_caption[:120])
    else:
        visual_caption = inputs.visual_caption
        logger.debug("Synthetic caption (no keyframes): %s", visual_caption[:120])

    # ------------------------------------------------------------------
    # Step 2 — Cross-modal consistency
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
    # Step 3 — Claim decomposition (with optional rationale hint)
    # ------------------------------------------------------------------
    logger.info("[3/7] Claim decomposition")
    rationale_hint = ""
    if use_rationale_hints:
        rationale_hint = inputs.rationale_context.prompt_summary()

    claim = decompose_claim(
        claim_text=claim_text,
        claim_id=claim_id,
        segment=segment,
        visual_caption=visual_caption,
        conflict_flag=modal_report.conflict_flag,
        llm=models.decomposer_llm,
        rationale_hint=rationale_hint,
        max_sub_questions=max_sub_questions,
    )
    logger.info("Decomposed into %d sub-questions.", len(claim.sub_questions))

    # ------------------------------------------------------------------
    # Step 4 — Gated evidence retrieval
    # ------------------------------------------------------------------
    logger.info("[4/7] Gated evidence retrieval")
    evidence, strength_report = gated_retrieval_loop(
        claim=claim,
        segment=segment,
        evidence=list(inputs.initial_evidence),
        modal_report=modal_report,
        retriever=retriever,
        nli=models.nli,
    )
    retrieval_rounds = 0 if strength_report.gate_pass else MAX_RETRIEVAL_ROUNDS
    logger.info(
        "Gate: pass=%s coverage=%.2f confidence=%.2f consistency=%.2f",
        strength_report.gate_pass,
        strength_report.coverage_score,
        strength_report.confidence_score,
        strength_report.consistency_score,
    )

    # ------------------------------------------------------------------
    # Step 5 — Multi-hop reasoning
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

    logger.info("=== Dataset pipeline complete | verdict=%s ===", report.verdict)
    return report


def run_dataset_record(
    record: DatasetRecord,
    models: ModelBundle,
    retriever: DenseRetriever,
    use_rationale_hints: bool = True,
    keyframe_paths: list[str] | None = None,
) -> DatasetEvalResult:
    """
    Run the full pipeline on a single DatasetRecord and return an eval result.

    Parameters
    ----------
    record               : Parsed DatasetRecord.
    models               : Loaded ModelBundle.
    retriever            : DenseRetriever pre-indexed on a passage corpus.
    use_rationale_hints  : Whether to inject gold rationale hints (Module 1).
    keyframe_paths       : Optional extracted keyframe paths. If None, the
                           synthetic caption is used.

    Returns
    -------
    DatasetEvalResult
    """
    inputs = record_to_pipeline_inputs(record, keyframe_paths=keyframe_paths)

    # Index the record-specific evidence corpus so dynamic retrieval (Modules 4 & 5) works
    retriever.index(list(inputs.initial_evidence))

    report = run_dataset_pipeline(
        inputs=inputs,
        models=models,
        retriever=retriever,
        use_rationale_hints=use_rationale_hints,
    )
    pred_label = verdict_to_label(report.verdict)
    return DatasetEvalResult(
        claim_id=inputs.claim_id,
        gold_verdict=inputs.gold_verdict,
        gold_label=inputs.gold_label,
        pred_verdict=report.verdict,
        pred_label=pred_label,
        pred_confidence=report.confidence,
        correct=(report.verdict == inputs.gold_verdict),
        report=report,
    )
