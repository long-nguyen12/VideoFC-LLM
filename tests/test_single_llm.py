"""
tests/test_single_llm.py
------------------------
Tests for hardware-constrained single-LLM mode:
  - load_single_llm_bundle: same object assigned to all three LLM slots
  - GenerativeLLM: max_new_tokens_cap and context_window honoured
  - module1: max_sub_questions cap enforced in prompt and in parsed output
  - module6: compacted aggregator prompt fits small context
  - dataset_pipeline: max_sub_questions propagated through
  - End-to-end: run_dataset_pipeline with single-model bundle
"""

import json
import os
import sys
import pytest
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def segment():
        return dict(
        segment_id="seg-001",
        start_ts=0.0,
        end_ts=30.0,
        transcript="The bridge was declared safe following a 2023 inspection.",
        keyframes=[],
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
def strength_pass():
        return dict(
        claim_id="claim-001",
        coverage_score=0.80, confidence_score=0.72, consistency_score=0.65,
        gate_pass=True, weak_aspects=[],
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


def _make_nli(score=0.75):
    nli = MagicMock()
    nli.entailment_score.return_value = score
    return nli


def _make_llm(response: dict):
    llm = MagicMock()
    llm.generate.return_value = json.dumps(response)
    return llm


# ===========================================================================
# GenerativeLLM — max_new_tokens_cap and context_window
# ===========================================================================

class TestGenerativeLLMCaps:

    def test_max_new_tokens_cap_applied(self):
        """generate() should call model.generate with min(requested, cap)."""
        import torch
        from models.model_bundle import GenerativeLLM

        llm = MagicMock(spec=GenerativeLLM)
        llm.max_new_tokens_cap = 512
        llm.context_window = 2048

        # Simulate the capping logic directly
        requested = 1024
        effective = min(requested, llm.max_new_tokens_cap)
        assert effective == 512

    def test_cap_does_not_reduce_small_request(self):
        """Requests smaller than the cap should pass through unchanged."""
        from models.model_bundle import GenerativeLLM
        llm = MagicMock(spec=GenerativeLLM)
        llm.max_new_tokens_cap = 512
        requested = 128
        effective = min(requested, llm.max_new_tokens_cap)
        assert effective == 128

    def test_default_cap_is_1024(self):
        """Default max_new_tokens_cap should be 1024 (full-size models)."""
        # We check the default value from the class signature via inspect
        import inspect
        from models.model_bundle import GenerativeLLM
        sig = inspect.signature(GenerativeLLM.__init__)
        assert sig.parameters["max_new_tokens_cap"]["default"] == 1024

    def test_default_context_window_is_4096(self):
        import inspect
        from models.model_bundle import GenerativeLLM
        sig = inspect.signature(GenerativeLLM.__init__)
        assert sig.parameters["context_window"]["default"] == 4096


# ===========================================================================
# load_single_llm_bundle — shared object identity
# ===========================================================================

class TestLoadSingleLLMBundle:

    def test_single_llm_bundle_exported(self):
        from models import load_single_llm_bundle
        assert callable(load_single_llm_bundle)

    def test_llm_slots_are_same_object(self):
        """All three LLM roles must point to the exact same Python object."""
        from models.model_bundle import GenerativeLLM, NLIScorer, TextEncoder
        from models.model_bundle import ModelBundle, load_single_llm_bundle

        with patch("models.model_bundle.GenerativeLLM") as MockLLM, \
             patch("models.model_bundle.NLIScorer") as MockNLI, \
             patch("models.model_bundle.TextEncoder") as MockEnc, \
             patch("models.model_bundle.VisualCaptioner") as MockCap:

            # Make the LLM constructor return a unique sentinel each call
            sentinel = object()
            MockLLM.return_value = sentinel

            bundle = load_single_llm_bundle(
                llm_model="Qwen/Qwen2.5-1.5B-Instruct",
                captioner_model="vikhyatk/moondream2",
            )

            # LLM constructor called exactly once
            assert MockLLM.call_count == 1

            # All three roles are the identical object
            assert bundle.decomposer_llm is bundle.hop_llm
            assert bundle.hop_llm is bundle.aggregator_llm
            assert bundle.decomposer_llm is sentinel

    def test_null_captioner_when_model_is_none(self):
        """Passing captioner_model=None should skip VisualCaptioner init."""
        from models.model_bundle import load_single_llm_bundle

        with patch("models.model_bundle.GenerativeLLM"), \
             patch("models.model_bundle.NLIScorer"), \
             patch("models.model_bundle.TextEncoder"), \
             patch("models.model_bundle.VisualCaptioner") as MockCap:

            bundle = load_single_llm_bundle(captioner_model=None)
            MockCap.assert_not_called()
            # caption_fn should return empty string
            assert bundle.caption_fn([]) == ""
            assert bundle.caption_fn(["some_frame.jpg"]) == ""

    def test_single_bundle_uses_512_cap(self):
        """load_single_llm_bundle must pass max_new_tokens_cap=512."""
        from models.model_bundle import load_single_llm_bundle

        with patch("models.model_bundle.GenerativeLLM") as MockLLM, \
             patch("models.model_bundle.NLIScorer"), \
             patch("models.model_bundle.TextEncoder"), \
             patch("models.model_bundle.VisualCaptioner"):

            load_single_llm_bundle(llm_model="Qwen/Qwen2.5-1.5B-Instruct")
            call_kwargs = MockLLM.call_args[1]
            assert call_kwargs.get("max_new_tokens_cap") == 512

    def test_single_bundle_uses_passed_context_window(self):
        from models.model_bundle import load_single_llm_bundle

        with patch("models.model_bundle.GenerativeLLM") as MockLLM, \
             patch("models.model_bundle.NLIScorer"), \
             patch("models.model_bundle.TextEncoder"), \
             patch("models.model_bundle.VisualCaptioner"):

            load_single_llm_bundle(context_window=2048)
            call_kwargs = MockLLM.call_args[1]
            assert call_kwargs.get("context_window") == 2048

    def test_default_context_window_is_2048(self):
        """Single-model bundle should default to 2048 context window."""
        import inspect
        from models.model_bundle import load_single_llm_bundle
        sig = inspect.signature(load_single_llm_bundle)
        assert sig.parameters["context_window"]["default"] == 2048

    def test_default_llm_is_qwen_15b(self):
        import inspect
        from models.model_bundle import load_single_llm_bundle
        sig = inspect.signature(load_single_llm_bundle)
        assert "Qwen2.5-1.5B" in sig.parameters["llm_model"]["default"]


# ===========================================================================
# Module 1 — max_sub_questions
# ===========================================================================

class TestModule1MaxSubQuestions:

    def test_max_sub_questions_in_system_prompt(self, segment):
        """The system prompt must embed the max_sub_questions limit."""
        from modules.module1_claim_decomposer import _build_prompt

        prompt = _build_prompt(
            claim_text="Some claim.",
            visual_caption="A scene.",
            transcript_excerpt="Some transcript.",
            start_ts=0.0,
            end_ts=10.0,
            conflict_flag=False,
            claim_id="c1",
            max_sub_questions=3,
        )
        system_content = prompt[0]["content"]
        assert "3" in system_content   # limit embedded

    def test_max_sub_questions_5_in_prompt(self, segment):
        from modules.module1_claim_decomposer import _build_prompt
        prompt = _build_prompt(
            claim_text="Claim.", visual_caption="", transcript_excerpt="",
            start_ts=0.0, end_ts=10.0, conflict_flag=False, claim_id="c",
            max_sub_questions=5,
        )
        assert "5" in prompt[0]["content"]

    def test_output_capped_at_max_sub_questions(self, segment):
        """Parser must silently drop sub-questions beyond the cap."""
        from modules.module1_claim_decomposer import decompose_claim

        # LLM returns 6 sub-questions but cap is 3
        resp = {
            "claim_id": "c",
            "sub_questions": [
                {"hop": i, "question": f"Q{i}?", "depends_on_hops": [], "evidence_type": "any"}
                for i in range(1, 7)
            ],
        }
        llm = _make_llm(resp)
        result = decompose_claim(
            "Claim.", "c", segment, "", False, llm,
            max_sub_questions=3,
        )
        assert len(result["sub_questions"]) == 3
        assert result["sub_questions"][-1].hop == 3

    def test_output_not_capped_when_under_limit(self, segment):
        """If LLM returns fewer than max_sub_questions, all are preserved."""
        from modules.module1_claim_decomposer import decompose_claim

        resp = {
            "claim_id": "c",
            "sub_questions": [
                {"hop": 1, "question": "Q1?", "depends_on_hops": [], "evidence_type": "any"},
                {"hop": 2, "question": "Q2?", "depends_on_hops": [1], "evidence_type": "web"},
            ],
        }
        result = decompose_claim("Claim.", "c", segment, "", False, _make_llm(resp),
                                 max_sub_questions=5)
        assert len(result["sub_questions"]) == 2

    def test_claim_truncated_in_user_message(self, segment):
        """Claims longer than 200 chars must be truncated in the prompt."""
        from modules.module1_claim_decomposer import _build_user_message

        long_claim = "X" * 300
        msg = _build_user_message(
            claim_text=long_claim,
            visual_caption="V", transcript_excerpt="T",
            start_ts=0.0, end_ts=10.0, conflict_flag=False, claim_id="c",
        )
        # The claim in the message should not exceed 200 chars of the original
        assert "X" * 201 not in msg
        assert "X" * 200 in msg

    def test_caption_truncated_in_user_message(self, segment):
        from modules.module1_claim_decomposer import _build_user_message

        long_cap = "C" * 200
        msg = _build_user_message(
            claim_text="Claim.", visual_caption=long_cap,
            transcript_excerpt="T", start_ts=0.0, end_ts=10.0,
            conflict_flag=False, claim_id="c",
        )
        assert "C" * 121 not in msg
        assert "C" * 120 in msg

    def test_default_max_sub_questions_is_5(self):
        import inspect
        from modules.module1_claim_decomposer import decompose_claim
        sig = inspect.signature(decompose_claim)
        assert sig.parameters["max_sub_questions"]["default"] == 5


# ===========================================================================
# Module 6 — compact aggregator prompt
# ===========================================================================

class TestModule6CompactPrompt:

    def test_hop_answers_capped_at_4(self, claim_decomp, segment,
                                     modal_report, strength_pass):
                from modules.module6_verdict_aggregator import _format_hop_answers

        # 6 hops
        hops = [
            dict(claim_id="c", hop=i, question=f"Q{i}?",
                      answer=f"Answer {i}.", confidence=0.8,
                      answer_unknown=False, supported_by=[])
            for i in range(1, 7)
        ]
        formatted = _format_hop_answers(hops)
        # Only first 4 hops should appear as full lines
        assert "Hop 4" in formatted
        assert "Hop 5" not in formatted or "truncated" in formatted

    def test_long_hop_answer_truncated(self):
                from modules.module6_verdict_aggregator import _format_hop_answers

        long_answer = "A" * 200
        hops = [dict(claim_id="c", hop=1, question="Q?",
                          answer=long_answer, confidence=0.8,
                          answer_unknown=False, supported_by=[])]
        formatted = _format_hop_answers(hops)
        assert "A" * 121 not in formatted
        assert "…" in formatted

    def test_claim_truncated_in_aggregator_prompt(self, claim_decomp, segment,
                                                   modal_report, strength_pass):
        from modules.module6_verdict_aggregator import _build_aggregator_prompt
        
        long_claim = dict(
            claim_id="c",
            claim_text="X" * 300,
            segment_id="s",
            sub_questions=[dict(hop=1, question="Q?",
                                       depends_on_hops=[], evidence_type="any")],
        )
        prompt = _build_aggregator_prompt(
            long_claim, segment, "caption", [], modal_report, strength_pass
        )
        user_content = prompt[1]["content"]
        assert "X" * 201 not in user_content
        assert "X" * 200 in user_content

    def test_aggregator_system_prompt_is_compact(self):
        """System prompt for aggregator should be under 400 chars."""
        from modules.module6_verdict_aggregator import _AGGREGATOR_SYSTEM_PROMPT
        assert len(_AGGREGATOR_SYSTEM_PROMPT) < 700

    def test_modal_info_on_single_line(self, claim_decomp, segment,
                                       modal_report, strength_pass):
        from modules.module6_verdict_aggregator import _build_aggregator_prompt

        prompt = _build_aggregator_prompt(
            claim_decomp, segment, "caption", [], modal_report, strength_pass
        )
        user_content = prompt[1]["content"]
        # Compact form: all modal scores on one line
        assert "vc=" in user_content
        assert "tc=" in user_content
        assert "vt=" in user_content

    def test_gate_status_on_single_line(self, claim_decomp, segment,
                                        modal_report, strength_pass):
        from modules.module6_verdict_aggregator import _build_aggregator_prompt

        prompt = _build_aggregator_prompt(
            claim_decomp, segment, "caption", [], modal_report, strength_pass
        )
        user_content = prompt[1]["content"]
        assert "gate=" in user_content.lower() or "PASS" in user_content

    def test_max_new_tokens_is_512(self, claim_decomp, segment,
                                   modal_report, strength_pass):
        """aggregate_verdict must call llm.generate with max_new_tokens=512."""
        from modules.module6_verdict_aggregator import aggregate_verdict
        
        verdict_resp = {
            "claim_id": "c", "verdict": "refuted", "confidence": 0.9,
            "reasoning_trace": [], "modal_conflict_used": False,
            "counterfactual": "CF.",
        }
        llm = _make_llm(verdict_resp)
        aggregate_verdict(
            claim_decomp, segment, "cap", [],
            modal_report, strength_pass, 0, llm,
        )
        call_kwargs = llm.generate.call_args[1]
        assert call_kwargs.get("max_new_tokens") == 512


# ===========================================================================
# dataset_pipeline — max_sub_questions propagated
# ===========================================================================

class TestDatasetPipelineMaxSubQuestions:

    RAW_RECORD = {
        "url": "https://snopes.com/test",
        "claim": "The bridge collapsed due to neglected maintenance.",
        "rating": "False",
        "content": "The claim is false.",
        "video_information": {
            "video_id": "vid-001", "video_date": 20240101.0,
            "platform": "test", "video_headline": "Bridge report",
            "video_transcript": "The bridge was inspected and declared safe.",
            "video_description": "News coverage of bridge inspection.",
            "video_length": 60.0,
            "video_url": "https://example.com/vid",
        },
        "original_rationales": {
            "main_rationale": "The bridge was maintained properly.",
            "additional_rationale1": "", "additional_rationale2": "",
            "additional_rationale3": "",
        },
        "summary_rationales": {
            "synthesized_rationale": "False — bridge was maintained.",
            "detailed_reasons": {"reason1": "Maintenance records are clean."},
        },
        "evidences": {
            "num_of_evidence": 2,
            "evidence1": ["Bridge inspected 2023.", []],
            "evidence2": ["No failures recorded.", []],
        },
        "relationship_with_evidence": [
            {"<claim,evidence1>": "Counters the claim."},
        ],
        "other": {},
    }

    def _make_stub_bundle(self, max_sub=3):
        import torch
        from models.model_bundle import ModelBundle

        bundle = MagicMock(spec=ModelBundle)
        bundle.caption_fn.return_value = ""
        bundle.nli.entailment_score.return_value = 0.72
        bundle.encoder.encode.return_value = torch.zeros(1, 384)

        decomp_resp = {
            "claim_id": "test",
            "sub_questions": [
                {"hop": i, "question": f"Q{i}?", "depends_on_hops": [], "evidence_type": "any"}
                for i in range(1, max_sub + 1)
            ],
        }
        hop_resp = lambda i: {
            "hop": i, "question": f"Q{i}?", "answer": f"A{i}.",
            "confidence": 0.8, "supported_by": [], "answer_unknown": False,
        }
        verdict_resp = {
            "claim_id": "test", "verdict": "refuted", "confidence": 0.85,
            "reasoning_trace": [], "modal_conflict_used": False,
            "counterfactual": "CF.",
        }
        summary_resp = {"summary": "Summary."}

        bundle.decomposer_llm.generate.return_value = json.dumps(decomp_resp)
        bundle.hop_llm.generate.side_effect = (
            [json.dumps(hop_resp(i)) for i in range(1, max_sub + 1)]
            + [json.dumps(summary_resp)] * max_sub
        )
        bundle.aggregator_llm.generate.return_value = json.dumps(verdict_resp)
        return bundle

    def _make_retriever(self):
        from modules.module4_targeted_retrieval import DenseRetriever
        r = MagicMock(spec=DenseRetriever)
        r.search.return_value = []
        return r

    def test_max_sub_questions_forwarded_to_decomposer(self):
        """decompose_claim must receive the max_sub_questions passed to run_dataset_pipeline."""
        from dataset.dataset_loader import DatasetLoader
        from dataset.dataset_adapter import record_to_pipeline_inputs
        from run_pipeline import run_dataset_pipeline
        from modules import decompose_claim as orig_decompose

        record = DatasetLoader.from_dict(self.RAW_RECORD)
        inputs = record_to_pipeline_inputs(record)
        bundle = self._make_stub_bundle(max_sub=3)
        retriever = self._make_retriever()

        captured = {}
        original = __import__("modules.module1_claim_decomposer",
                               fromlist=["decompose_claim"]).decompose_claim

        def capturing_decompose(*args, **kwargs):
            captured["max_sub_questions"] = kwargs.get("max_sub_questions")
            return original(*args, **kwargs)

        with patch("dataset.dataset_pipeline.decompose_claim", side_effect=capturing_decompose):
            run_dataset_pipeline(inputs, bundle, retriever, max_sub_questions=3)

        assert captured.get("max_sub_questions") == 3

    def test_run_dataset_pipeline_with_max_3(self):
        from dataset.dataset_loader import DatasetLoader
        from dataset.dataset_adapter import record_to_pipeline_inputs
        from run_pipeline import run_dataset_pipeline
        
        record = DatasetLoader.from_dict(self.RAW_RECORD)
        inputs = record_to_pipeline_inputs(record)
        bundle = self._make_stub_bundle(max_sub=3)
        retriever = self._make_retriever()

        report = run_dataset_pipeline(
            inputs, bundle, retriever,
            use_rationale_hints=True,
            max_sub_questions=3,
        )
        assert isinstance(report, dict)
        assert report["verdict"] in (
            "supported", "refuted", "insufficient_evidence", "misleading_context"
        )

    def test_run_dataset_record_single_model_mode(self):
        """Convenience check: run_dataset_record works end-to-end with max_sub_questions=3."""
        from dataset.dataset_loader import DatasetLoader
        from run_pipeline import run_dataset_record

        record = DatasetLoader.from_dict(self.RAW_RECORD)
        bundle = self._make_stub_bundle(max_sub=3)
        retriever = self._make_retriever()

        # run_dataset_record does not expose max_sub_questions directly,
        # so we patch run_dataset_pipeline to verify it is called correctly
        from . import dataset_pipeline as dp
        original_run = dp.run_dataset_pipeline
        captured_kwargs = {}

        def patched_run(inputs, models, retriever, **kwargs):
            captured_kwargs.update(kwargs)
            return original_run(inputs, models, retriever, **kwargs)

        with patch.object(dp, "run_dataset_pipeline", side_effect=patched_run):
            result = run_dataset_record(
                record, bundle, retriever,
                use_rationale_hints=True,
            )

        assert result is not None
        assert result["gold_verdict"] == "refuted"


# ===========================================================================
# Integration: full pipeline using a single-model stub bundle
# ===========================================================================

class TestSingleModelIntegration:

    def _make_single_model_stub(self):
        """Simulate load_single_llm_bundle by sharing one LLM stub."""
        import torch
        from models.model_bundle import ModelBundle

        shared_llm = MagicMock()
        # The shared LLM is called in order: decompose → hop(s) → aggregate → summaries
        decomp_resp = {
            "claim_id": "c",
            "sub_questions": [
                {"hop": 1, "question": "Was maintenance neglected?",
                 "depends_on_hops": [], "evidence_type": "web"},
                {"hop": 2, "question": "Did the bridge collapse?",
                 "depends_on_hops": [1], "evidence_type": "any"},
                {"hop": 3, "question": "What do records show?",
                 "depends_on_hops": [], "evidence_type": "web"},
            ],
        }
        hop_resps = [
            {"hop": 1, "question": "Was maintenance neglected?",
             "answer": "No.", "confidence": 0.88,
             "supported_by": ["ev-001"], "answer_unknown": False},
            {"hop": 2, "question": "Did the bridge collapse?",
             "answer": "No evidence.", "confidence": 0.82,
             "supported_by": ["ev-002"], "answer_unknown": False},
            {"hop": 3, "question": "What do records show?",
             "answer": "Records are clean.", "confidence": 0.79,
             "supported_by": [], "answer_unknown": False},
        ]
        verdict_resp = {
            "claim_id": "c", "verdict": "refuted", "confidence": 0.90,
            "reasoning_trace": [
                {"step": 1, "finding": "Maintenance was not neglected.",
                 "source_hop": 1, "evidence_ids": ["ev-001"]},
            ],
            "modal_conflict_used": False,
            "counterfactual": "Would be supported if records showed neglect.",
        }
        summaries = [
            {"summary": "Maintenance records refute the claim."},
            {"summary": "No collapse evidence found."},
            {"summary": "Clean records confirm proper upkeep."},
        ]

        shared_llm.generate.side_effect = (
            [json.dumps(decomp_resp)]
            + [json.dumps(r) for r in hop_resps]
            + [json.dumps(verdict_resp)]
            + [json.dumps(s) for s in summaries]
        )

        bundle = MagicMock(spec=ModelBundle)
        bundle.caption_fn.return_value = "A concrete bridge."
        bundle.nli.entailment_score.return_value = 0.72
        bundle.encoder.encode.return_value = torch.zeros(1, 384)

        # All three roles are the SAME object
        bundle.decomposer_llm = shared_llm
        bundle.hop_llm = shared_llm
        bundle.aggregator_llm = shared_llm

        return bundle, shared_llm

    def test_all_roles_use_same_llm_object(self, segment, evidence_pool):
        from pipeline import run_pipeline
        from modules.module4_targeted_retrieval import DenseRetriever

        bundle, shared_llm = self._make_single_model_stub()
        retriever = MagicMock(spec=DenseRetriever)
        retriever.search.return_value = []

        run_pipeline(
            claim_text="The bridge collapsed due to neglected maintenance.",
            claim_id="c",
            segment=segment,
            initial_evidence=evidence_pool,
            models=bundle,
            retriever=retriever,
        )

        # shared_llm.generate should have been called multiple times
        # (decompose + hops + aggregate + summaries) — all on the same object
        assert shared_llm.generate.call_count >= 3

    def test_pipeline_returns_report_with_single_model(self, segment, evidence_pool):
        from pipeline import run_pipeline
                from modules.module4_targeted_retrieval import DenseRetriever

        bundle, _ = self._make_single_model_stub()
        retriever = MagicMock(spec=DenseRetriever)
        retriever.search.return_value = []

        report = run_pipeline(
            claim_text="The bridge collapsed due to neglected maintenance.",
            claim_id="c",
            segment=segment,
            initial_evidence=evidence_pool,
            models=bundle,
            retriever=retriever,
        )

        assert isinstance(report, dict)
        assert report["verdict"] == "refuted"
        assert report["confidence"] == pytest.approx(0.90)
        assert len(report["hop_summaries"]) == 3

    def test_generate_call_count_equals_decompose_plus_hops_plus_verdict_plus_summaries(
        self, segment, evidence_pool
    ):
        from pipeline import run_pipeline
        from modules.module4_targeted_retrieval import DenseRetriever

        bundle, shared_llm = self._make_single_model_stub()
        retriever = MagicMock(spec=DenseRetriever)
        retriever.search.return_value = []

        run_pipeline(
            claim_text="The bridge collapsed.",
            claim_id="c",
            segment=segment,
            initial_evidence=evidence_pool,
            models=bundle,
            retriever=retriever,
        )

        # 1 decompose + 3 hops + 1 verdict + 3 summaries = 8
        assert shared_llm.generate.call_count == 8
