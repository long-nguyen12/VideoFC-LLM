from __future__ import annotations

import logging

from modules.module1_claim_decomposer import decompose_claim
from modules.module2_cross_modal_consistency import (
    compute_modal_consistency,
    compute_modal_consistency_llm,
)
from modules.module3_evidence_strength import score_evidence
from modules.module4_targeted_retrieval import DenseRetriever, gated_retrieval_loop
from modules.module5_multihop_reasoning import run_multihop
from modules.module6_verdict_aggregator import aggregate_verdict
from modules.module7_explainability import build_explainability_report
from modules.module4_targeted_retrieval import MAX_RETRIEVAL_ROUNDS

from dataset import (
    record_to_pipeline_inputs,
    verdict_to_label,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset-aware pipeline
# ---------------------------------------------------------------------------


def run_fc_pipeline(
    inputs: dict,
    models: dict,
    retriever: DenseRetriever,
    use_rationale_hints: bool = True,
    max_sub_questions: int = 5,
) -> dict:
    segment = inputs["segment"]
    claim_text = inputs["claim_text"]
    claim_id = inputs["claim_id"]
    content = inputs.get("content") or inputs.get("rationale_context", {}).get(
        "article_content", ""
    )

    logger.info(
        "=== Dataset pipeline start | claim_id=%s segment=%s ===",
        claim_id,
        segment["segment_id"],
    )

    # ------------------------------------------------------------------
    # Step 1 — Visual captioning (or synthetic bypass)
    # ------------------------------------------------------------------
    logger.info("[1/7] Visual captioning")
    if segment.get("keyframes", []):
        visual_caption = models["caption_fn"](segment["keyframes"])
        logger.debug("VLM caption: %s", visual_caption)
    else:
        visual_caption = inputs["visual_caption"]
        logger.debug("Synthetic caption (no keyframes): %s", visual_caption)

    # ------------------------------------------------------------------
    # Step 2 — Cross-modal consistency
    # ------------------------------------------------------------------
    logger.info("[2/7] Cross-modal consistency")
    modal_report = compute_modal_consistency_llm(
        claim_text=claim_text,
        visual_caption=visual_caption,
        transcript=segment["transcript"],
        segment_id=segment["segment_id"],
        content=content,
        llm=models["consistency_llm"],
    )
    print(f"DEBUG: Modal report for claim_id={claim_id} = {modal_report}")

    # ------------------------------------------------------------------
    # Step 3 — Claim decomposition (with optional rationale hint)
    # ------------------------------------------------------------------
    logger.info("[3/7] Claim decomposition")
    rationale_hint = ""
    if use_rationale_hints:
        ctx = inputs["rationale_context"]
        lines = [f"Known verdict: {ctx.get('snopes_rating', '')}"]
        if ctx.get("main_rationale"):
            lines.append(f"Main rationale: {ctx['main_rationale']}")
        for i, r in enumerate(ctx.get("additional_rationales", []), 1):
            lines.append(f"Supporting rationale {i}: {r}")
        rationale_hint = "\n".join(lines)[:600]

    claim = decompose_claim(
        claim_text=claim_text,
        claim_id=claim_id,
        segment=segment,
        visual_caption=visual_caption,
        conflict_flag=modal_report["conflict_flag"],
        llm=models["decomposer_llm"],
        rationale_hint=rationale_hint,
        max_sub_questions=max_sub_questions,
    )
    logger.info("Decomposed into %d sub-questions.", len(claim["sub_questions"]))

    # ------------------------------------------------------------------
    # Step 4 — Gated evidence retrieval
    # ------------------------------------------------------------------
    logger.info("[4/7] Gated evidence retrieval")
    evidence, strength_report = gated_retrieval_loop(
        claim=claim,
        segment=segment,
        evidence=list(inputs["initial_evidence"]),
        modal_report=modal_report,
        retriever=retriever,
        nli=models["nli"],
    )
    retrieval_rounds = 0 if strength_report["gate_pass"] else MAX_RETRIEVAL_ROUNDS
    logger.info(
        "Gate: pass=%s coverage=%.2f confidence=%.2f consistency=%.2f",
        strength_report["gate_pass"],
        strength_report["coverage_score"],
        strength_report["confidence_score"],
        strength_report["consistency_score"],
    )

    # ------------------------------------------------------------------
    # Step 5 — Multi-hop reasoning
    # ------------------------------------------------------------------
    logger.info("[5/7] Multi-hop reasoning")
    hop_results = run_multihop(
        claim=claim,
        evidence=evidence,
        segment=segment,
        llm=models["hop_llm"],
        retriever=retriever,
    )
    known = sum(1 for h in hop_results if not h["answer_unknown"])
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
        llm=models["aggregator_llm"],
    )
    logger.info(
        "Verdict: %s (confidence=%.2f)", verdict["verdict"], verdict["confidence"]
    )

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
        nli=models["nli"],
        llm=models["hop_llm"],
    )

    logger.info(
        "=== Dataset pipeline complete | verdict=%s ===",
        report.get("verdict", "unknown"),
    )
    return report


def run_dataset_record(
    record: dict,
    models: dict,
    retriever: DenseRetriever,
    use_rationale_hints: bool = True,
    keyframe_paths: list[str] | None = None,
) -> dict:
    inputs = record_to_pipeline_inputs(record, keyframe_paths=keyframe_paths)

    retriever.index(list(inputs["initial_evidence"]))

    report = run_fc_pipeline(
        inputs=inputs,
        models=models,
        retriever=retriever,
        use_rationale_hints=use_rationale_hints,
    )
    pred_label = verdict_to_label(report.get("verdict", "unknown"))
    return {
        "claim_id": inputs["claim_id"],
        "gold_verdict": inputs["gold_verdict"],
        "gold_label": inputs["gold_label"],
        "pred_verdict": report.get("verdict", "unknown"),
        "pred_label": pred_label,
        "pred_confidence": report.get("confidence", 0.0),
        "correct": report.get("verdict", "unknown") == inputs["gold_verdict"],
        "report": report,
    }
