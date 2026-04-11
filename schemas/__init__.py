"""
schemas/
--------
Pydantic data models for inter-module communication in the fact-checking pipeline.
"""

from .data_models import (
    VideoSegment,
    EvidenceRef,
    ModalConflictReport,
    ModalAnnotation,
    SubQuestion,
    ClaimDecomposition,
    EvidenceStrengthReport,
    HopResult,
    ReasoningStep,
    FinalVerdict,
    EvidenceSaliency,
    ExplainabilityReport,
)

__all__ = [
    "VideoSegment",
    "EvidenceRef",
    "ModalConflictReport",
    "ModalAnnotation",
    "SubQuestion",
    "ClaimDecomposition",
    "EvidenceStrengthReport",
    "HopResult",
    "ReasoningStep",
    "FinalVerdict",
    "EvidenceSaliency",
    "ExplainabilityReport",
]
