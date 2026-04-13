"""
schemas/data_models.py
----------------------
All shared Pydantic schemas used for inter-module communication.
Every module imports from here; no schema is defined elsewhere.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Input types
# ---------------------------------------------------------------------------

class VideoSegment(BaseModel):
    """One video segment from the dataset."""
    segment_id: str
    start_ts: float
    end_ts: float
    transcript: str                      # pre-annotated, given in dataset
    keyframes: list[str]                 # file paths or base64-encoded JPEG/PNG strings


class EvidenceRef(BaseModel):
    """One retrieved or pre-given evidence passage."""
    evidence_id: str
    source_url: str
    source_date: str
    passage_text: str
    retrieval_score: float
    hop_ids: list[int] = Field(default_factory=list)  # which hops this evidence was fetched for


# ---------------------------------------------------------------------------
# Module 1 — Claim Decomposition
# ---------------------------------------------------------------------------

class SubQuestion(BaseModel):
    hop: int
    question: str
    depends_on_hops: list[int] = Field(default_factory=list)
    evidence_type: str          # "video" | "web" | "kb" | "any"


class ClaimDecomposition(BaseModel):
    claim_id: str
    claim_text: str
    segment_id: str
    sub_questions: list[SubQuestion]


# ---------------------------------------------------------------------------
# Module 2 — Cross-Modal Consistency
# ---------------------------------------------------------------------------

class ModalConflictReport(BaseModel):
    segment_id: str
    vc_score: float             # visual caption ↔ claim  NLI entailment score
    tc_score: float             # transcript    ↔ claim  NLI entailment score
    vt_score: float             # visual caption ↔ transcript NLI entailment score
    conflict_flag: bool
    dominant_conflict: Optional[str] = None   # "V↔C" | "T↔C" | "V↔T" | null


# ---------------------------------------------------------------------------
# Module 3 — Evidence Strength
# ---------------------------------------------------------------------------

class EvidenceStrengthReport(BaseModel):
    claim_id: str
    coverage_score: float       # fraction of sub-questions with strong evidence
    confidence_score: float     # mean NLI entailment across evidence passages
    consistency_score: float    # agreement between evidence and video modalities
    gate_pass: bool
    weak_aspects: list[str]     # sub-question texts that lack strong evidence


# ---------------------------------------------------------------------------
# Module 5 — Multi-Hop Reasoning
# ---------------------------------------------------------------------------

class HopResult(BaseModel):
    claim_id: str = ""
    hop: int
    question: str = ""
    answer: str                 # ≤ 2 sentences
    confidence: float
    answer_unknown: bool
    supported_by: list[str] = Field(default_factory=list)  # evidence_ids


# ---------------------------------------------------------------------------
# Module 6 — Final Verdict
# ---------------------------------------------------------------------------

class ReasoningStep(BaseModel):
    step: int
    finding: str
    source_hop: Optional[int] = None
    evidence_ids: list[str] = Field(default_factory=list)


class FinalVerdict(BaseModel):
    claim_id: str
    segment_id: str
    verdict: str
    confidence: float
    reasoning_trace: list[ReasoningStep] = Field(default_factory=list)
    modal_conflict_used: bool = False
    counterfactual: str = ""
    retrieval_rounds: int = 0
    gate_passed: bool = False


# ---------------------------------------------------------------------------
# Module 7 — Explainability
# ---------------------------------------------------------------------------

class EvidenceSaliency(BaseModel):
    evidence_id: str
    hop: int
    saliency_score: float       # NLI contribution weight, normalised to [0, 1] within hop
    key_span: str               # most salient sentence from the passage


class ModalAnnotation(BaseModel):
    pair: str                   # "V↔C" | "T↔C" | "V↔T"
    score: float
    timestamp: float
    human_note: str


class ExplainabilityReport(BaseModel):
    claim_id: str
    segment_id: str
    verdict: str
    confidence: float
    evidence_saliency: list[EvidenceSaliency] = Field(default_factory=list)
    modal_annotations: list[ModalAnnotation] = Field(default_factory=list)
    hop_summaries: list[str] = Field(default_factory=list)
    counterfactual: str = ""
    gate_passed: bool = False
    retrieval_rounds: int = 0
