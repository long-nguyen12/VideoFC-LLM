"""
dataset/
--------
Dataset integration layer for the video fact-checking pipeline.

Public surface
--------------
  DatasetRecord            — raw record schema (mirrors dataset JSON exactly)
  DatasetLoader            — lazy file iterator (JSONL / JSON)
  DirectoryLoader          — lazy directory iterator (per-record JSON files)
  load_split               — load train_val / test using dataset split files
  load_for_pipeline        — load records + keyframe paths for LLM pipeline
  run_pipeline_evaluation  — run the full LLM pipeline on a dataset split
  record_to_pipeline_inputs— converts a record into all pipeline inputs
  PipelineInputs           — typed container for pipeline inputs
  RationaleContext         — gold rationale data for prompt injection
  run_dataset_record       — run the full pipeline on one record + eval result
  DatasetEvalResult        — (gold, predicted, correct) evaluation container
  compute_metrics          — compute P/R/F1 from a list of eval results
  EvaluationSummary        — structured metric report
  rating_to_verdict        — Snopes rating → pipeline verdict
  VERDICT_TO_LABEL         — verdict → integer label index
  NUM_LABELS               — number of verdict classes (4)
"""

from .dataset_schemas import DatasetRecord
from .dataset_loader import DatasetLoader, DirectoryLoader, split_records, load_split
from .true_dataset_loader import load_for_pipeline, run_pipeline_evaluation
from .dataset_adapter import (
    record_to_pipeline_inputs,
    record_to_segment,
    record_to_evidence,
    record_to_rationale_context,
    record_to_visual_caption,
    PipelineInputs,
    RationaleContext,
)
from .dataset_pipeline import (
    run_dataset_pipeline,
    run_dataset_record,
    DatasetEvalResult,
)
from .evaluation import compute_metrics, EvaluationSummary, log_summary
from .label_mapper import (
    rating_to_verdict,
    verdict_to_label,
    rating_to_label,
    label_to_verdict,
    VERDICT_TO_LABEL,
    LABEL_TO_VERDICT,
    NUM_LABELS,
    VERDICT_DISPLAY,
)

__all__ = [
    "DatasetRecord",
    "DatasetLoader",
    "DirectoryLoader",
    "split_records",
    "load_split",
    "load_for_pipeline",
    "run_pipeline_evaluation",
    "record_to_pipeline_inputs",
    "record_to_segment",
    "record_to_evidence",
    "record_to_rationale_context",
    "record_to_visual_caption",
    "PipelineInputs",
    "RationaleContext",
    "run_dataset_pipeline",
    "run_dataset_record",
    "DatasetEvalResult",
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

