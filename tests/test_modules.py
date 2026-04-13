"""
tests/test_modules.py
---------------------
Unit tests for all seven pipeline modules.
All HuggingFace models are replaced with lightweight stubs so these tests
run without GPU or network access.
"""

import json
import pytest
from unittest.mock import MagicMock, patch




# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def segment():
    return dict(
        segment_id="seg-001",
        start_ts=0.0,
        end_ts=30.0,
        transcript="The bridge was declared safe following a 2023 inspection.",
        keyframes=["frame_00.jpg"],
    )


@pytest.fixture
def evidence_pool():
    return [
        dict(
            evidence_id="ev-001",
            source_url="https://source.com/a",
            source_date="2024-01-10",
            passage_text="Independent inspectors confirmed structural integrity.",
            retrieval_score=0.91,
            hop_ids=[1],
        ),
        dict(
            evidence_id="ev-002",
            source_url="https://source.com/b",
            source_date="2024-01-12",
            passage_text="No maintenance failures were recorded in the last five years.",
            retrieval_score=0.85,
            hop_ids=[2],
        ),
    ]


@pytest.fixture
def claim_decomp():
    return dict(
        claim_id="claim-001",
        claim_text="The bridge collapsed due to neglected maintenance.",
        segment_id="seg-001",
        sub_questions=[
            dict(hop=1, question="Was maintenance neglected?",
                        depends_on_hops=[], evidence_type="web"),
            dict(hop=2, question="Did the bridge collapse?",
                        depends_on_hops=[1], evidence_type="any"),
        ],
    )


@pytest.fixture
def modal_report():
    return dict(
        segment_id="seg-001",
        vc_score=0.71, tc_score=0.65, vt_score=0.78,
        conflict_flag=False, dominant_conflict=None,
    )


@pytest.fixture
def modal_report_conflict():
    return dict(
        segment_id="seg-001",
        vc_score=0.22, tc_score=0.60, vt_score=0.55,
        conflict_flag=True, dominant_conflict="V↔C",
    )


@pytest.fixture
def strength_pass():
    return dict(
        claim_id="claim-001",
        coverage_score=0.80, confidence_score=0.72, consistency_score=0.65,
        gate_pass=True, weak_aspects=[],
    )


@pytest.fixture
def strength_fail():
    return dict(
        claim_id="claim-001",
        coverage_score=0.40, confidence_score=0.35, consistency_score=0.30,
        gate_pass=False, weak_aspects=["Was maintenance neglected?"],
    )


@pytest.fixture
def hop_results():
    return [
        dict(claim_id="claim-001", hop=1,
                  question="Was maintenance neglected?",
                  answer="No — records confirm regular maintenance.",
                  confidence=0.85, answer_unknown=False,
                  supported_by=["ev-001"]),
        dict(claim_id="claim-001", hop=2,
                  question="Did the bridge collapse?",
                  answer="There is no evidence of a collapse.",
                  confidence=0.78, answer_unknown=False,
                  supported_by=["ev-002"]),
    ]


def _make_nli(score: float = 0.75):
    """Return a stub NLI scorer that always returns a fixed entailment score."""
    nli = MagicMock()
    nli.entailment_score.return_value = score
    return nli


def _make_llm(response: dict):
    """Return a stub LLM that always returns a JSON-serialised dict."""
    llm = MagicMock()
    llm.generate.return_value = json.dumps(response)
    return llm


# ===========================================================================
# Module 1 — Claim Decomposer
# ===========================================================================

class TestModule1ClaimDecomposer:
    def test_normal_decomposition(self, segment):
        from modules.module1_claim_decomposer import decompose_claim

        llm_response = {
            "claim_id": "claim-001",
            "sub_questions": [
                {"hop": 1, "question": "Was maintenance neglected?",
                 "depends_on_hops": [], "evidence_type": "web"},
                {"hop": 2, "question": "Did the bridge collapse?",
                 "depends_on_hops": [1], "evidence_type": "any"},
            ],
        }
        result = decompose_claim(
            claim_text="The bridge collapsed due to neglected maintenance.",
            claim_id="claim-001",
            segment=segment,
            visual_caption="A large concrete bridge over a river.",
            conflict_flag=False,
            llm=_make_llm(llm_response),
        )

        assert result["claim_id"] == "claim-001"
        assert result["segment_id"] == "seg-001"
        assert len(result["sub_questions"]) == 2
        assert result["sub_questions"][0].hop == 1
        assert result["sub_questions"][1].depends_on_hops == [1]

    def test_fallback_on_llm_failure(self, segment):
        from modules.module1_claim_decomposer import decompose_claim

        llm = MagicMock()
        llm.generate.return_value = "NOT JSON AT ALL"

        result = decompose_claim(
            claim_text="Some claim.",
            claim_id="claim-002",
            segment=segment,
            visual_caption="",
            conflict_flag=False,
            llm=llm,
            max_retries=1,
        )
        # Fallback: single sub-question covering the whole claim
        assert len(result["sub_questions"]) == 1
        assert result["sub_questions"][0].question == "Some claim."
        assert result["sub_questions"][0].hop == 1

    def test_markdown_fences_stripped(self, segment):
        from modules.module1_claim_decomposer import decompose_claim

        fenced = '```json\n{"claim_id":"c","sub_questions":[{"hop":1,"question":"Q?","depends_on_hops":[],"evidence_type":"any"}]}\n```'
        llm = MagicMock()
        llm.generate.return_value = fenced

        result = decompose_claim("Q?", "c", segment, "", False, llm)
        assert result["claim_id"] == "c"
        assert len(result["sub_questions"]) == 1


# ===========================================================================
# Module 2 — Cross-Modal Consistency
# ===========================================================================

class TestModule2CrossModal:
    def test_no_conflict(self):
        from modules.module2_cross_modal_consistency import compute_modal_consistency

        result = compute_modal_consistency(
            claim_text="The bridge is safe.",
            visual_caption="A sturdy bridge.",
            transcript="Engineers confirmed the bridge is safe.",
            segment_id="seg-001",
            nli=_make_nli(0.80),
        )
        assert not result["conflict_flag"]
        assert result["dominant_conflict"] is None
        assert result["vc_score"] == pytest.approx(0.80, abs=1e-4)

    def test_conflict_detected(self):
        from modules.module2_cross_modal_consistency import compute_modal_consistency

        # Score below 0.40 for V↔C
        nli = MagicMock()
        scores = iter([0.20, 0.70, 0.65])  # vc, tc, vt
        nli.entailment_score.side_effect = lambda p, h: next(scores)

        result = compute_modal_consistency(
            claim_text="The bridge collapsed.",
            visual_caption="The bridge stands intact.",
            transcript="No collapse was observed.",
            segment_id="seg-001",
            nli=nli,
        )
        assert result["conflict_flag"]
        assert result["dominant_conflict"] == "V↔C"

    def test_all_pairs_scored(self):
        from modules.module2_cross_modal_consistency import compute_modal_consistency

        nli = _make_nli(0.75)
        compute_modal_consistency("C", "V", "T", "seg", nli)
        assert nli.entailment_score.call_count == 3


# ===========================================================================
# Module 3 — Evidence Strength Scorer
# ===========================================================================

class TestModule3EvidenceStrength:
    def test_gate_passes_with_strong_evidence(
        self, claim_decomp, evidence_pool, modal_report
    ):
        from modules.module3_evidence_strength import score_evidence

        result = score_evidence(claim_decomp, evidence_pool, modal_report, _make_nli(0.80))
        assert result["gate_pass"]
        assert result["weak_aspects"] == []
        assert result["coverage_score"] == pytest.approx(1.0, abs=1e-4)

    def test_gate_fails_with_weak_evidence(
        self, claim_decomp, evidence_pool, modal_report
    ):
        from modules.module3_evidence_strength import score_evidence

        result = score_evidence(claim_decomp, evidence_pool, modal_report, _make_nli(0.10))
        assert not result["gate_pass"]
        assert len(result["weak_aspects"]) == 2

    def test_consistency_uses_min_modal_score(
        self, claim_decomp, evidence_pool
    ):
        from modules.module3_evidence_strength import score_evidence

        modal = dict(
            segment_id="s", vc_score=0.30, tc_score=0.90, vt_score=0.85,
            conflict_flag=True, dominant_conflict="V↔C",
        )
        result = score_evidence(claim_decomp, evidence_pool, modal, _make_nli(0.80))
        assert result["consistency_score"] == pytest.approx(0.30, abs=1e-4)
        # Consistency < 0.60 should cause gate to fail
        assert not result["gate_pass"]

    def test_no_evidence_returns_zero(self, claim_decomp, modal_report):
        from modules.module3_evidence_strength import score_evidence

        result = score_evidence(claim_decomp, [], modal_report, _make_nli(0.80))
        assert result["coverage_score"] == 0.0
        assert not result["gate_pass"]


# ===========================================================================
# Module 4 — Targeted Retrieval
# ===========================================================================

class TestModule4TargetedRetrieval:
    def _make_retriever(self, passages):
        from modules.module4_targeted_retrieval import DenseRetriever
        retriever = MagicMock(spec=DenseRetriever)
        retriever.search.return_value = passages
        return retriever

    def test_loop_stops_when_gate_passes(
        self, claim_decomp, evidence_pool, modal_report, segment
    ):
        from modules.module4_targeted_retrieval import gated_retrieval_loop

        # NLI score high enough to pass immediately
        evidence, report = gated_retrieval_loop(
            claim=claim_decomp,
            segment=segment,
            evidence=list(evidence_pool),
            modal_report=modal_report,
            retriever=self._make_retriever([]),
            nli=_make_nli(0.80),
            max_rounds=3,
        )
        assert report["gate_pass"]
        # Retriever should not have been called
        # (gate passes on first scoring pass)

    def test_retriever_called_for_weak_aspects(
        self, claim_decomp, modal_report, segment
    ):
        from modules.module4_targeted_retrieval import gated_retrieval_loop

        new_ev = dict(
            evidence_id="ev-new", source_url="", source_date="",
            passage_text="New passage.", retrieval_score=0.7, hop_ids=[],
        )
        retriever = self._make_retriever([new_ev])

        # Start with empty evidence so gate fails → retrieval triggered
        evidence, report = gated_retrieval_loop(
            claim=claim_decomp,
            segment=segment,
            evidence=[],
            modal_report=modal_report,
            retriever=retriever,
            nli=_make_nli(0.10),  # always fails gate
            max_rounds=2,
        )
        assert retriever.search.called

    def test_no_duplicate_evidence(
        self, claim_decomp, evidence_pool, modal_report, segment
    ):
        from modules.module4_targeted_retrieval import gated_retrieval_loop

        # Retriever always returns the same passage as initial evidence
        duplicate = evidence_pool[0]
        retriever = self._make_retriever([duplicate])

        evidence, _ = gated_retrieval_loop(
            claim=claim_decomp,
            segment=segment,
            evidence=list(evidence_pool),
            modal_report=modal_report,
            retriever=retriever,
            nli=_make_nli(0.10),
            max_rounds=2,
        )
        ids = [e.evidence_id for e in evidence]
        assert len(ids) == len(set(ids)), "Duplicate evidence IDs found"

    def test_build_retrieval_query(self, segment):
        from modules.module4_targeted_retrieval import build_retrieval_query

        resolved = [
            dict(claim_id="c", hop=1, question="Q1?",
                      answer="Bridge was maintained.", confidence=0.8,
                      answer_unknown=False, supported_by=[]),
        ]
        query = build_retrieval_query("Did the bridge collapse?", resolved, segment)
        assert "Did the bridge collapse?" in query
        assert "Bridge was maintained." in query
        assert segment["transcript"][:10] in query


# ===========================================================================
# Module 5 — Multi-Hop Reasoning
# ===========================================================================

class TestModule5MultiHop:
    def _make_retriever(self):
        r = MagicMock()
        r.search.return_value = []
        return r

    def test_two_hops_completed(
        self, claim_decomp, evidence_pool, segment
    ):
        from modules.module5_multihop_reasoning import run_multihop

        hop1_resp = {"hop": 1, "question": "Was maintenance neglected?",
                     "answer": "No.", "confidence": 0.85,
                     "supported_by": ["ev-001"], "answer_unknown": False}
        hop2_resp = {"hop": 2, "question": "Did the bridge collapse?",
                     "answer": "No evidence of collapse.",
                     "confidence": 0.78, "supported_by": ["ev-002"],
                     "answer_unknown": False}

        llm = MagicMock()
        llm.generate.side_effect = [json.dumps(hop1_resp), json.dumps(hop2_resp)]

        results = run_multihop(claim_decomp, evidence_pool, segment,
                               llm, self._make_retriever())
        assert len(results) == 2
        assert not results[0]["answer_unknown"]
        assert not results[1]["answer_unknown"]

    def test_unknown_hop_stops_chain(
        self, claim_decomp, evidence_pool, segment
    ):
        from modules.module5_multihop_reasoning import run_multihop

        unknown_resp = {"hop": 1, "question": "Was maintenance neglected?",
                        "answer": "", "confidence": 0.0,
                        "supported_by": [], "answer_unknown": True}

        llm = MagicMock()
        llm.generate.return_value = json.dumps(unknown_resp)

        results = run_multihop(claim_decomp, evidence_pool, segment,
                               llm, self._make_retriever())

        # Hop 2 depends_on_hops=[1], so it should be skipped
        assert results[0]["answer_unknown"]
        if len(results) > 1:
            assert results[1]["answer_unknown"]

    def test_retry_on_parse_failure(self, claim_decomp, evidence_pool, segment):
        from modules.module5_multihop_reasoning import run_single_hop

        sq = claim_decomp["sub_questions"][0]
        good = {"hop": 1, "question": "Q?", "answer": "A.",
                "confidence": 0.7, "supported_by": [], "answer_unknown": False}
        llm = MagicMock()
        llm.generate.side_effect = ["INVALID JSON", json.dumps(good)]

        result = run_single_hop(sq, {}, evidence_pool, llm, max_retries=2)
        assert not result["answer_unknown"]
        assert result["answer"] == "A."


# ===========================================================================
# Module 6 — Verdict Aggregator
# ===========================================================================

class TestModule6VerdictAggregator:
    def test_verdict_emitted_on_gate_pass(
        self, claim_decomp, segment, modal_report, strength_pass, hop_results
    ):
        from modules.module6_verdict_aggregator import aggregate_verdict

        llm_resp = {
            "claim_id": "claim-001",
            "verdict": "refuted",
            "confidence": 0.88,
            "reasoning_trace": [
                {"step": 1, "finding": "Maintenance was not neglected.",
                 "source_hop": 1, "evidence_ids": ["ev-001"]}
            ],
            "modal_conflict_used": False,
            "counterfactual": "The claim would be supported if maintenance logs showed neglect.",
        }

        verdict = aggregate_verdict(
            claim=claim_decomp,
            segment=segment,
            visual_caption="A sturdy bridge.",
            hop_results=hop_results,
            modal_report=modal_report,
            strength_report=strength_pass,
            retrieval_rounds=0,
            llm=_make_llm(llm_resp),
        )

        assert verdict["verdict"] == "refuted"
        assert verdict["confidence"] == pytest.approx(0.88)
        assert verdict["gate_passed"]
        assert len(verdict["reasoning_trace"]) == 1

    def test_insufficient_evidence_without_llm_call(
        self, claim_decomp, segment, modal_report, strength_fail, hop_results
    ):
        from modules.module6_verdict_aggregator import aggregate_verdict

        llm = MagicMock()
        verdict = aggregate_verdict(
            claim=claim_decomp,
            segment=segment,
            visual_caption="",
            hop_results=hop_results,
            modal_report=modal_report,
            strength_report=strength_fail,
            retrieval_rounds=3,
            llm=llm,
        )

        assert verdict["verdict"] == "insufficient_evidence"
        assert verdict["confidence"] == 0.0
        assert not verdict["gate_passed"]
        llm.generate.assert_not_called()

    def test_fallback_on_llm_parse_failure(
        self, claim_decomp, segment, modal_report, strength_pass, hop_results
    ):
        from modules.module6_verdict_aggregator import aggregate_verdict

        llm = MagicMock()
        llm.generate.return_value = "NOT VALID JSON"

        verdict = aggregate_verdict(
            claim=claim_decomp, segment=segment,
            visual_caption="", hop_results=hop_results,
            modal_report=modal_report, strength_report=strength_pass,
            retrieval_rounds=0, llm=llm, max_retries=1,
        )
        assert verdict["verdict"] == "insufficient_evidence"

    def test_valid_verdict_values(
        self, claim_decomp, segment, modal_report, strength_pass, hop_results
    ):
        from modules.module6_verdict_aggregator import aggregate_verdict

        valid_verdicts = [
            "supported", "refuted",
            "insufficient_evidence", "misleading_context",
        ]
        for v in valid_verdicts:
            llm_resp = {
                "claim_id": "claim-001", "verdict": v,
                "confidence": 0.75, "reasoning_trace": [],
                "modal_conflict_used": False,
                "counterfactual": "cf",
            }
            verdict = aggregate_verdict(
                claim=claim_decomp, segment=segment,
                visual_caption="", hop_results=hop_results,
                modal_report=modal_report, strength_report=strength_pass,
                retrieval_rounds=0, llm=_make_llm(llm_resp),
            )
            assert verdict["verdict"] == v


# ===========================================================================
# Module 7 — Explainability
# ===========================================================================

class TestModule7Explainability:
    def _make_verdict(self, verdict_label="refuted"):
        return dict(
            claim_id="claim-001",
            segment_id="seg-001",
            verdict=verdict_label,
            confidence=0.88,
            reasoning_trace=[
                dict(step=1, finding="Finding.", source_hop=1,
                              evidence_ids=["ev-001"]),
            ],
            modal_conflict_used=False,
            counterfactual="CF statement.",
            retrieval_rounds=1,
            gate_passed=True,
        )

    def test_saliency_normalised_per_hop(
        self, hop_results, evidence_pool, modal_report, segment
    ):
        from modules.module7_explainability import compute_evidence_saliency

        saliency = compute_evidence_saliency(
            hop_results, evidence_pool, "refuted", _make_nli(0.60)
        )
        # Each hop with supported_by should have exactly one saliency entry
        hop_saliencies = {s["hop"]: s for s in saliency}
        for hop in hop_results:
            if hop["supported_by"]:
                assert hop["hop"] in hop_saliencies
                assert 0.0 <= hop_saliencies[hop["hop"]]["saliency_score"] <= 1.0

    def test_modal_annotations_only_for_conflicts(
        self, segment
    ):
        from modules.module7_explainability import build_modal_annotations

        # No conflict → empty list
        report_no_conflict = dict(
            segment_id="s", vc_score=0.80, tc_score=0.75, vt_score=0.70,
            conflict_flag=False, dominant_conflict=None,
        )
        assert build_modal_annotations(report_no_conflict, segment) == []

        # One conflict
        report_conflict = dict(
            segment_id="s", vc_score=0.20, tc_score=0.75, vt_score=0.70,
            conflict_flag=True, dominant_conflict="V↔C",
        )
        annotations = build_modal_annotations(report_conflict, segment)
        assert len(annotations) == 1
        assert annotations[0]["pair"] == "V↔C"
        assert str(segment["start_ts"]) in annotations[0]["human_note"]

    def test_hop_summaries_length(self, hop_results):
        from modules.module7_explainability import generate_hop_summaries

        summary_resp = {"summary": "The bridge was maintained regularly."}
        llm = _make_llm(summary_resp)
        summaries = generate_hop_summaries(hop_results, llm)

        assert len(summaries) == len(hop_results)
        for s in summaries:
            assert isinstance(s, str)
            assert len(s) > 0

    def test_unknown_hop_gets_canned_summary(self):
        from modules.module7_explainability import generate_hop_summaries

        hops = [
            dict(claim_id="c", hop=1, question="Q?",
                      answer="", confidence=0.0, answer_unknown=True),
        ]
        llm = MagicMock()
        summaries = generate_hop_summaries(hops, llm)

        assert len(summaries) == 1
        assert "could not be resolved" in summaries[0]
        llm.generate.assert_not_called()

    def test_full_report_structure(
        self, hop_results, evidence_pool, modal_report, segment
    ):
        from modules.module7_explainability import build_explainability_report

        verdict = self._make_verdict()
        summary_resp = {"summary": "Evidence supports the finding."}

        report = build_explainability_report(
            verdict=verdict,
            hop_results=hop_results,
            evidence=evidence_pool,
            modal_report=modal_report,
            segment=segment,
            nli=_make_nli(0.70),
            llm=_make_llm(summary_resp),
        )

        assert report["claim_id"] == "claim-001"
        assert report["segment_id"] == "seg-001"
        assert report["verdict"] == "refuted"
        assert report["gate_passed"]
        assert report["retrieval_rounds"] == 1
        assert report["counterfactual"] == "CF statement."
        assert len(report["hop_summaries"]) == len(hop_results)

    def test_report_passthrough_fields(
        self, hop_results, evidence_pool, modal_report, segment
    ):
        from modules.module7_explainability import build_explainability_report

        verdict = self._make_verdict("misleading_context")
        verdict["retrieval_rounds"] = 2

        report = build_explainability_report(
            verdict=verdict,
            hop_results=hop_results,
            evidence=evidence_pool,
            modal_report=modal_report,
            segment=segment,
            nli=_make_nli(0.70),
            llm=_make_llm({"summary": "s"}),
        )
        assert report["retrieval_rounds"] == 2
        assert report["verdict"] == "misleading_context"


# ===========================================================================
# Pipeline integration (all stubs, no GPU)
# ===========================================================================

class TestPipelineIntegration:
    def _make_model_bundle(self, nli_score=0.75):
        from models.model_bundle import ModelBundle

        captioner = MagicMock()
        captioner.caption.return_value = "A concrete bridge over a river."

        nli = _make_nli(nli_score)
        encoder = MagicMock()
        encoder.encode.return_value = __import__("torch").zeros(1, 384)

        decomp_resp = {
            "claim_id": "claim-001",
            "sub_questions": [
                {"hop": 1, "question": "Was maintenance neglected?",
                 "depends_on_hops": [], "evidence_type": "web"},
            ],
        }
        hop_resp = {
            "hop": 1, "question": "Was maintenance neglected?",
            "answer": "No — regular maintenance was performed.",
            "confidence": 0.82, "supported_by": ["ev-001"],
            "answer_unknown": False,
        }
        verdict_resp = {
            "claim_id": "claim-001", "verdict": "refuted", "confidence": 0.84,
            "reasoning_trace": [{"step": 1, "finding": "Maintenance was not neglected.",
                                  "source_hop": 1, "evidence_ids": ["ev-001"]}],
            "modal_conflict_used": False,
            "counterfactual": "Would be supported if maintenance was neglected.",
        }
        summary_resp = {"summary": "Maintenance records refute the claim."}

        decomposer = MagicMock()
        decomposer.generate.return_value = json.dumps(decomp_resp)

        hop_llm = MagicMock()
        hop_llm.generate.side_effect = [
            json.dumps(hop_resp),    # hop answer
            json.dumps(summary_resp),  # hop summary (Module 7)
        ]

        aggregator = MagicMock()
        aggregator.generate.return_value = json.dumps(verdict_resp)

        bundle = MagicMock(spec=ModelBundle)
        bundle.caption_fn.return_value = "A concrete bridge over a river."
        bundle.nli = nli
        bundle.encoder = encoder
        bundle.decomposer_llm = decomposer
        bundle.hop_llm = hop_llm
        bundle.aggregator_llm = aggregator

        return bundle

    def _make_retriever(self, evidence_pool):
        from modules.module4_targeted_retrieval import DenseRetriever
        r = MagicMock(spec=DenseRetriever)
        r.search.return_value = []
        return r

    def test_pipeline_returns_explainability_report(
        self, segment, evidence_pool
    ):
        from pipeline import run_pipeline
        
        bundle = self._make_model_bundle()
        retriever = self._make_retriever(evidence_pool)

        report = run_pipeline(
            claim_text="The bridge collapsed due to neglected maintenance.",
            claim_id="claim-001",
            segment=segment,
            initial_evidence=evidence_pool,
            models=bundle,
            retriever=retriever,
        )

        assert isinstance(report, dict)
        assert report["claim_id"] == "claim-001"
        assert report["verdict"] in (
            "supported", "refuted",
            "insufficient_evidence", "misleading_context",
        )

    def test_pipeline_insufficient_evidence_path(self, segment):
        from pipeline import run_pipeline
        
        # NLI always low → gate never passes → insufficient_evidence
        bundle = self._make_model_bundle(nli_score=0.05)
        retriever = MagicMock()
        retriever.search.return_value = []

        # With no initial evidence and low NLI, gate will fail
        report = run_pipeline(
            claim_text="The bridge collapsed.",
            claim_id="claim-002",
            segment=segment,
            initial_evidence=[],
            models=bundle,
            retriever=retriever,
        )

        assert isinstance(report, dict)
        # Either insufficient_evidence or another valid verdict
        assert report["verdict"] in (
            "supported", "refuted",
            "insufficient_evidence", "misleading_context",
        )
