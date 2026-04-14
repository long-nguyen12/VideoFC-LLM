"""
dataset/
--------
Dataset integration layer for the video fact-checking pipeline.

Public surface
--------------
  load_for_pipeline        — load records + keyframe paths for LLM pipeline
  run_pipeline_evaluation  — run the full LLM pipeline on a dataset split
  record_to_pipeline_inputs— converts a record into all pipeline inputs
  run_dataset_record       — run the full pipeline on one record + eval result
  compute_metrics          — compute P/R/F1 from a list of eval results
  EvaluationSummary        — structured metric report
  rating_to_verdict        — Snopes rating → pipeline verdict
  VERDICT_TO_LABEL         — verdict → integer label index
  NUM_LABELS               — number of verdict classes (4)
"""

from .true_dataset_loader import (
    load_for_pipeline,
    run_pipeline_evaluation,
    record_to_pipeline_inputs,
    record_to_segment,
    record_to_evidence,
    record_to_rationale_context,
    record_to_visual_caption,
    rating_to_verdict,
    verdict_to_label,
    rating_to_label,
    label_to_verdict,
    VERDICT_TO_LABEL,
    LABEL_TO_VERDICT,
    NUM_LABELS,
    VERDICT_DISPLAY,
)
from run_pipeline import (
    run_dataset_pipeline,
    run_dataset_record,
)
from .evaluation import compute_metrics, EvaluationSummary, log_summary

__all__ = [
    "load_for_pipeline",
    "run_pipeline_evaluation",
    "record_to_pipeline_inputs",
    "record_to_segment",
    "record_to_evidence",
    "record_to_rationale_context",
    "record_to_visual_caption",
    "run_dataset_pipeline",
    "run_dataset_record",
    "compute_metrics",
    "EvaluationSummary",
    "log_summary",
    "rating_to_verdict",
    "verdict_to_label",
    "rating_to_label",
    "label_to_verdict",
    "VERDICT_TO_LABEL",
    "LABEL_TO_VERDICT",
    "NUM_LABELS",
    "VERDICT_DISPLAY",
]
