"""
tests/test_schemas.py
---------------------
Unit tests for all Pydantic data schemas.
Verifies field types, defaults, and round-trip JSON serialisation.
"""

import json
import pytest
from schemas.data_models import (
    VideoSegment,
    EvidenceRef,
    SubQuestion,
    ClaimDecomposition,
    ModalConflictReport,
    EvidenceStrengthReport,
    HopResult,
    ReasoningStep,
    FinalVerdict,
    EvidenceSaliency,
    ModalAnnotation,
    ExplainabilityReport,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def video_segment():
    return VideoSegment(
        segment_id="seg-001",
        start_ts=0.0,
        end_ts=30.5,
        transcript="The bridge was declared structurally sound in 2023.",
        keyframes=["frame_00.jpg", "frame_15.jpg"],
    )


@pytest.fixture
def evidence_ref():
    return EvidenceRef(
        evidence_id="ev-001",
        source_url="https://example.com/article",
        source_date="2024-01-15",
        passage_text="Inspectors found no structural deficiencies in the bridge.",
        retrieval_score=0.92,
        hop_ids=[1],
    )


@pytest.fixture
def claim_decomposition():
    return ClaimDecomposition(
        claim_id="claim-001",
        claim_text="The bridge collapsed due to neglected maintenance.",
        segment_id="seg-001",
        sub_questions=[
            SubQuestion(hop=1, question="Was the bridge maintained?",
                        depends_on_hops=[], evidence_type="web"),
            SubQuestion(hop=2, question="Did the bridge collapse?",
                        depends_on_hops=[1], evidence_type="any"),
        ],
    )


@pytest.fixture
def modal_conflict_report():
    return ModalConflictReport(
        segment_id="seg-001",
        vc_score=0.72,
        tc_score=0.55,
        vt_score=0.68,
        conflict_flag=False,
        dominant_conflict=None,
    )


@pytest.fixture
def hop_result():
    return HopResult(
        claim_id="claim-001",
        hop=1,
        question="Was the bridge maintained?",
        answer="Records show regular maintenance was performed until 2022.",
        confidence=0.81,
        answer_unknown=False,
        supported_by=["ev-001"],
    )


@pytest.fixture
def final_verdict():
    return FinalVerdict(
        claim_id="claim-001",
        segment_id="seg-001",
        verdict="refuted",
        confidence=0.87,
        reasoning_trace=[
            ReasoningStep(step=1, finding="Maintenance logs contradict the claim.",
                          source_hop=1, evidence_ids=["ev-001"]),
        ],
        modal_conflict_used=False,
        counterfactual="The claim would be supported if maintenance records showed neglect.",
        retrieval_rounds=1,
        gate_passed=True,
    )


# ---------------------------------------------------------------------------
# VideoSegment
# ---------------------------------------------------------------------------

class TestVideoSegment:
    def test_fields(self, video_segment):
        assert video_segment.segment_id == "seg-001"
        assert video_segment.start_ts == 0.0
        assert video_segment.end_ts == 30.5
        assert "bridge" in video_segment.transcript
        assert len(video_segment.keyframes) == 2

    def test_json_roundtrip(self, video_segment):
        data = json.loads(video_segment.model_dump_json())
        restored = VideoSegment(**data)
        assert restored == video_segment

    def test_keyframes_empty_list(self):
        seg = VideoSegment(segment_id="x", start_ts=0.0, end_ts=1.0,
                           transcript="", keyframes=[])
        assert seg.keyframes == []


# ---------------------------------------------------------------------------
# EvidenceRef
# ---------------------------------------------------------------------------

class TestEvidenceRef:
    def test_fields(self, evidence_ref):
        assert evidence_ref.evidence_id == "ev-001"
        assert evidence_ref.retrieval_score == 0.92
        assert evidence_ref.hop_ids == [1]

    def test_default_hop_ids(self):
        ev = EvidenceRef(evidence_id="x", source_url="", source_date="",
                         passage_text="text", retrieval_score=0.5)
        assert ev.hop_ids == []

    def test_json_roundtrip(self, evidence_ref):
        restored = EvidenceRef(**json.loads(evidence_ref.model_dump_json()))
        assert restored == evidence_ref


# ---------------------------------------------------------------------------
# ClaimDecomposition
# ---------------------------------------------------------------------------

class TestClaimDecomposition:
    def test_sub_questions_order(self, claim_decomposition):
        hops = [sq.hop for sq in claim_decomposition.sub_questions]
        assert hops == sorted(hops)

    def test_depends_on_hops_default(self):
        sq = SubQuestion(hop=1, question="Q?", evidence_type="any")
        assert sq.depends_on_hops == []

    def test_evidence_type_values(self, claim_decomposition):
        for sq in claim_decomposition.sub_questions:
            assert sq.evidence_type in ("video", "web", "kb", "any")

    def test_json_roundtrip(self, claim_decomposition):
        data = json.loads(claim_decomposition.model_dump_json())
        restored = ClaimDecomposition(**data)
        assert restored.claim_id == claim_decomposition.claim_id
        assert len(restored.sub_questions) == len(claim_decomposition.sub_questions)


# ---------------------------------------------------------------------------
# ModalConflictReport
# ---------------------------------------------------------------------------

class TestModalConflictReport:
    def test_no_conflict(self, modal_conflict_report):
        assert not modal_conflict_report.conflict_flag
        assert modal_conflict_report.dominant_conflict is None

    def test_with_conflict(self):
        report = ModalConflictReport(
            segment_id="s", vc_score=0.25, tc_score=0.80, vt_score=0.75,
            conflict_flag=True, dominant_conflict="V↔C",
        )
        assert report.conflict_flag
        assert report.dominant_conflict == "V↔C"

    def test_scores_are_floats(self, modal_conflict_report):
        assert isinstance(modal_conflict_report.vc_score, float)
        assert isinstance(modal_conflict_report.tc_score, float)
        assert isinstance(modal_conflict_report.vt_score, float)


# ---------------------------------------------------------------------------
# EvidenceStrengthReport
# ---------------------------------------------------------------------------

class TestEvidenceStrengthReport:
    def test_gate_pass_true(self):
        r = EvidenceStrengthReport(
            claim_id="c", coverage_score=0.8, confidence_score=0.7,
            consistency_score=0.65, gate_pass=True, weak_aspects=[],
        )
        assert r.gate_pass
        assert r.weak_aspects == []

    def test_gate_pass_false_with_weak_aspects(self):
        r = EvidenceStrengthReport(
            claim_id="c", coverage_score=0.5, confidence_score=0.4,
            consistency_score=0.3, gate_pass=False,
            weak_aspects=["Was the bridge maintained?"],
        )
        assert not r.gate_pass
        assert len(r.weak_aspects) == 1


# ---------------------------------------------------------------------------
# HopResult
# ---------------------------------------------------------------------------

class TestHopResult:
    def test_known_hop(self, hop_result):
        assert not hop_result.answer_unknown
        assert hop_result.confidence > 0.0
        assert hop_result.supported_by == ["ev-001"]

    def test_unknown_hop(self):
        hop = HopResult(hop=2, answer="", confidence=0.0,
                        answer_unknown=True, supported_by=[])
        assert hop.answer_unknown
        assert hop.supported_by == []

    def test_default_claim_id(self):
        hop = HopResult(hop=1, answer="x", confidence=0.5, answer_unknown=False)
        assert hop.claim_id == ""


# ---------------------------------------------------------------------------
# FinalVerdict
# ---------------------------------------------------------------------------

class TestFinalVerdict:
    def test_verdict_values(self, final_verdict):
        assert final_verdict.verdict in (
            "supported", "refuted", "insufficient_evidence", "misleading_context"
        )

    def test_reasoning_trace(self, final_verdict):
        assert len(final_verdict.reasoning_trace) == 1
        step = final_verdict.reasoning_trace[0]
        assert step.step == 1
        assert step.source_hop == 1

    def test_json_roundtrip(self, final_verdict):
        data = json.loads(final_verdict.model_dump_json())
        restored = FinalVerdict(**data)
        assert restored.verdict == final_verdict.verdict
        assert restored.confidence == final_verdict.confidence


# ---------------------------------------------------------------------------
# ExplainabilityReport
# ---------------------------------------------------------------------------

class TestExplainabilityReport:
    def test_full_report(self):
        report = ExplainabilityReport(
            claim_id="claim-001",
            segment_id="seg-001",
            verdict="refuted",
            confidence=0.87,
            evidence_saliency=[
                EvidenceSaliency(evidence_id="ev-001", hop=1,
                                 saliency_score=1.0, key_span="maintenance was performed")
            ],
            modal_annotations=[
                ModalAnnotation(pair="V↔C", score=0.25, timestamp=0.0,
                                human_note="Visual content contradicts claim at 0.0s.")
            ],
            hop_summaries=["Records show the bridge was maintained regularly."],
            counterfactual="The claim would be supported if maintenance records showed neglect.",
            gate_passed=True,
            retrieval_rounds=1,
        )
        assert report.verdict == "refuted"
        assert len(report.evidence_saliency) == 1
        assert len(report.modal_annotations) == 1
        assert len(report.hop_summaries) == 1

    def test_default_empty_fields(self):
        report = ExplainabilityReport(
            claim_id="c", segment_id="s",
            verdict="insufficient_evidence", confidence=0.0,
        )
        assert report.evidence_saliency == []
        assert report.modal_annotations == []
        assert report.hop_summaries == []
        assert report.counterfactual == ""

    def test_json_roundtrip(self):
        report = ExplainabilityReport(
            claim_id="c", segment_id="s", verdict="supported", confidence=0.9,
            hop_summaries=["The event occurred as described."],
            gate_passed=True, retrieval_rounds=0,
        )
        data = json.loads(report.model_dump_json())
        restored = ExplainabilityReport(**data)
        assert restored == report
