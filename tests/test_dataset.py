"""
tests/test_dataset.py
---------------------
Unit tests for the full dataset integration layer:
  dataset_schemas   — DatasetRecord parsing
  label_mapper      — Snopes rating → verdict mapping
  dataset_adapter   — record → pipeline types conversion
  dataset_loader    — file loading (via in-memory stubs)
  dataset_pipeline  — dataset-aware pipeline with synthetic caption bypass
  evaluation        — metric computation
"""

import json
import os
import sys
import tempfile
import pytest
from unittest.mock import MagicMock

# Ensure project root is on the path (mirrors conftest.py)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")


# ---------------------------------------------------------------------------
# Shared raw record fixture (the Snopes/Obama record from the dataset)
# ---------------------------------------------------------------------------

RAW_RECORD = {
    "url": "https://www.snopes.com/fact-check/obama-snubs-scalias-funeral/",
    "claim": "President Obama snubbed the funeral of Supreme Court Justice Antonin Scalia in order to play golf.",
    "rating": "Mostly False",
    "content": "President Obama opted to attend a visitation for Justice Scalia but did not attend his funeral.",
    "video_information": {
        "video_id": "1942500",
        "video_date": 20160217.0,
        "platform": "nbcnews",
        "video_headline": "Obama speaks out on battle to fill Scalia's seat",
        "video_transcript": "And now to the heated debate over replacing supreme court justice Antonin Scalia.",
        "video_description": "President Obama is responding firmly to Republican opposition.",
        "video_length": 157.918844,
        "video_url": "https://www.nbcnews.com/news/us-news/white-house-obama-will-not-attend-justice-scalia-s-funeral-n520236",
    },
    "original_rationales": {
        "main_rationale": "President Obama didn't fail to pay respect to Justice Scalia or play golf instead of attending his funeral.",
        "additional_rationale1": "President Obama opted to attend a visitation for Justice Scalia but did not attend his funeral.",
        "additional_rationale2": "The optics of paying his respects to Scalia are tricky for Obama.",
        "additional_rationale3": "In actuality, the President spent that weekend reading through lengthy dossiers.",
    },
    "summary_rationales": {
        "synthesized_rationale": "The claim is rated Mostly False because it lacks evidence.",
        "detailed_reasons": {
            "reason1": "Presidents do not have a precedent of attending all Supreme Court justices funerals.",
            "reason2": "Obama attended the visitation, undermining the claim.",
            "reason3": "Reports suggesting Obama was playing golf were speculative.",
        },
    },
    "evidences": {
        "num_of_evidence": 9,
        "evidence1": [
            "Instead, the president will pay his respects on Friday.",
            [],
        ],
        "evidence2": [
            "Earnest revealed the president's plans during the daily briefing.",
            [],
        ],
        "evidence3": [
            "I wouldn't have expected President Obama to attend the funeral Mass.",
            [],
        ],
        "evidence4": [
            "Facing questions again about the decision, Josh Earnest implied disruption.",
            [],
        ],
        "evidence5": [
            "Former President George W. Bush attended the funeral for Chief Justice William Rehnquist.",
            [],
        ],
        "evidence6": [
            "Of the approximately 100 justices who have served on the court.",
            [],
        ],
        "evidence7": [
            "Oh man...is Obama planning to golf through Scalia's funeral?",
            ["https://t.co/LrSJmVKpBp", "https://twitter.com/guypbenson/status/700062310220562432"],
        ],
        "evidence8": [
            "WATCH: White House Says THIS About Obama Golfing During Scalia Funeral.",
            ["https://t.co/y5um4IalgV"],
        ],
        "evidence9": [
            "@POTUS must be too busy golfing to attend the funeral.",
            ["https://twitter.com/eric_poitras/status/701129931493941249"],
        ],
    },
    "relationship_with_evidence": [
        {"<claim,evidence1>": "Evidence directly counters the claim."},
        {"<claim,evidence2>": "Evidence clarifies Obama's approach."},
        {"<main_rationale,evidence3>": "Evidence supports the main rationale."},
        {"<claim,evidence7>": "Tweet aligns with the claim's assertion."},
        {"<claim,evidence9>": "Tweet reinforces the golf speculation."},
    ],
    "other": {"iframe_video_links": []},
}


@pytest.fixture
def raw_record():
    return dict(RAW_RECORD)


@pytest.fixture
def dataset_record(raw_record):
    from dataset.true_dataset_loader import rating_to_verdict
    return DatasetRecord.from_dict(raw_record)


# ===========================================================================
# dataset_schemas
# ===========================================================================

class TestDatasetSchemas:

    def test_from_dict_parses_top_level(self, raw_record):
        from dataset.true_dataset_loader import rating_to_verdict
        r = DatasetRecord.from_dict(raw_record)
        assert r.claim == raw_record["claim"]
        assert r.rating == "Mostly False"
        assert r.url.startswith("https://")

    def test_video_information_parsed(self, dataset_record):
        vi = dataset_record.video_information
        assert vi.video_id == "1942500"
        assert vi.video_date == 20160217.0
        assert vi.platform == "nbcnews"
        assert vi.video_length == pytest.approx(157.918844)
        assert "Scalia" in vi.video_transcript

    def test_original_rationales_parsed(self, dataset_record):
        r = dataset_record.original_rationales
        assert "Obama" in r.main_rationale
        assert r.additional_rationale1 != ""
        assert r.additional_rationale2 != ""
        assert r.additional_rationale3 != ""

    def test_original_rationales_all_rationales(self, dataset_record):
        all_r = dataset_record.original_rationales.all_rationales()
        assert len(all_r) == 4
        assert all(isinstance(s, str) and s["strip"]() for s in all_r)

    def test_summary_rationales_parsed(self, dataset_record):
        s = dataset_record.summary_rationales
        assert "Mostly False" in s["synthesized_rationale"]
        assert len(s["all_reasons"]()) == 3

    def test_detailed_reasons_ordered(self, dataset_record):
        reasons = dataset_record.summary_rationales.all_reasons()
        assert reasons[0]["startswith"]("Presidents do not")
        assert reasons[1]["startswith"]("Obama attended")
        assert reasons[2]["startswith"]("Reports suggesting")

    def test_evidences_parsed(self, dataset_record):
        ev = dataset_record.evidences
        assert ev.num_of_evidence == 9
        assert len(ev.entries) == 9

    def test_evidence_urls(self, dataset_record):
        ev7 = dataset_record.evidences.entries[6]   # evidence7 is index 6 (0-based)
        assert ev7.evidence_index == 7
        assert len(ev7.urls) == 2
        assert ev7.urls[0]["startswith"]("https://")

    def test_evidence_no_urls(self, dataset_record):
        ev1 = dataset_record.evidences.entries[0]
        assert ev1.evidence_index == 1
        assert ev1.urls == []

    def test_relationship_entries_parsed(self, dataset_record):
        rels = dataset_record.relationship_with_evidence
        assert len(rels) == 5

    def test_relationship_left_right(self, dataset_record):
        rels = dataset_record.relationship_with_evidence
        claim_rels = [r for r in rels if r.left == "claim"]
        rationale_rels = [r for r in rels if "rationale" in r.left]
        assert len(claim_rels) == 4
        assert len(rationale_rels) == 1

    def test_relationship_evidence_index(self, dataset_record):
        rels = dataset_record.relationship_with_evidence
        indices = [r.evidence_index for r in rels]
        assert 1 in indices
        assert 7 in indices
        assert 9 in indices

    def test_missing_additional_rationale_defaults(self):
        from dataset.true_dataset_loader import rating_to_verdict
        raw = dict(RAW_RECORD)
        raw["original_rationales"] = {"main_rationale": "Only main."}
        r = DatasetRecord.from_dict(raw)
        assert r.original_rationales.additional_rationale1 == ""
        assert len(r.original_rationales.all_rationales()) == 1

    def test_evidence_block_from_dict_handles_missing_keys(self):
        from dataset.true_dataset_loader import rating_to_verdict
        raw = {"num_of_evidence": 2, "evidence1": ["text one", []]}
        # evidence2 missing — should not raise
        block = EvidencesBlock.from_dict(raw)
        assert block.num_of_evidence == 2
        assert len(block.entries) == 1


# ===========================================================================
# label_mapper
# ===========================================================================

class TestLabelMapper:

    @pytest.mark.parametrize("rating,expected", [
        ("True",                "supported"),
        ("Mostly True",         "supported"),
        ("Correct Attribution", "supported"),
        ("False",               "refuted"),
        ("Mostly False",        "refuted"),
        ("Labeled Satire",      "refuted"),
        ("Mixture",             "misleading_context"),
        ("Outdated",            "misleading_context"),
        ("Miscaptioned",        "misleading_context"),
        ("Unproven",            "insufficient_evidence"),
        ("Unrated",             "insufficient_evidence"),
        ("Research In Progress","insufficient_evidence"),
    ])
    def test_known_ratings(self, rating, expected):
        from dataset.true_dataset_loader import rating_to_verdict, VERDICT_TO_LABEL, label_to_verdict, rating_to_label, NUM_LABELS
        assert rating_to_verdict(rating) == expected

    def test_unknown_rating_fallback(self):
        from dataset.true_dataset_loader import rating_to_verdict, VERDICT_TO_LABEL, label_to_verdict, rating_to_label, NUM_LABELS
        assert rating_to_verdict("Banana") == "insufficient_evidence"

    def test_case_insensitive(self):
        from dataset.true_dataset_loader import rating_to_verdict, VERDICT_TO_LABEL, label_to_verdict, rating_to_label, NUM_LABELS
        assert rating_to_verdict("mostly false") == "refuted"
        assert rating_to_verdict("MOSTLY FALSE") == "refuted"
        assert rating_to_verdict("  True  ")    == "supported"

    def test_verdict_to_label_range(self):
        from dataset.true_dataset_loader import rating_to_verdict, VERDICT_TO_LABEL, label_to_verdict, rating_to_label, NUM_LABELS
        assert len(VERDICT_TO_LABEL) == NUM_LABELS
        assert set(VERDICT_TO_LABEL.values()) == set(range(NUM_LABELS))

    def test_label_to_verdict_roundtrip(self):
        from dataset.true_dataset_loader import rating_to_verdict, VERDICT_TO_LABEL, label_to_verdict, rating_to_label, NUM_LABELS
        for verdict, label in VERDICT_TO_LABEL.items():
            assert label_to_verdict(label) == verdict

    def test_rating_to_label_mostly_false(self):
        from dataset.true_dataset_loader import rating_to_verdict, VERDICT_TO_LABEL, label_to_verdict, rating_to_label, NUM_LABELS
        lbl = rating_to_label("Mostly False")
        assert lbl == VERDICT_TO_LABEL["refuted"]

    def test_verdict_confidence_floor(self):
        from dataset.true_dataset_loader import rating_to_verdict, VERDICT_TO_LABEL, label_to_verdict, rating_to_label, NUM_LABELS
        assert verdict_confidence_floor("supported") > 0.0
        assert verdict_confidence_floor("refuted") > 0.0
        assert verdict_confidence_floor("misleading_context") > 0.0
        assert verdict_confidence_floor("insufficient_evidence") == 0.0


# ===========================================================================
# dataset_adapter
# ===========================================================================

class TestDatasetAdapter:

    def test_record_to_segment(self, dataset_record):
        from dataset.true_dataset_loader import record_to_pipeline_inputs, _yyyymmdd_to_iso, _assign_hop_ids, record_to_segment, record_to_visual_caption, record_to_evidence, record_to_rationale_context
        seg = record_to_segment(dataset_record)
        assert seg.segment_id == "1942500"
        assert seg.start_ts == 0.0
        assert seg.end_ts == pytest.approx(157.918844)
        assert "Scalia" in seg.transcript
        assert seg.keyframes == []

    def test_record_to_segment_with_keyframes(self, dataset_record):
        from dataset.true_dataset_loader import record_to_pipeline_inputs, _yyyymmdd_to_iso, _assign_hop_ids, record_to_segment, record_to_visual_caption, record_to_evidence, record_to_rationale_context
        seg = record_to_segment(dataset_record, keyframe_paths=["f1.jpg", "f2.jpg"])
        assert seg.keyframes == ["f1.jpg", "f2.jpg"]

    def test_record_to_visual_caption(self, dataset_record):
        from dataset.true_dataset_loader import record_to_pipeline_inputs, _yyyymmdd_to_iso, _assign_hop_ids, record_to_segment, record_to_visual_caption, record_to_evidence, record_to_rationale_context
        cap = record_to_visual_caption(dataset_record)
        assert "Obama" in cap or "Scalia" in cap
        assert len(cap) > 10

    def test_record_to_evidence_count(self, dataset_record):
        from dataset.true_dataset_loader import record_to_pipeline_inputs, _yyyymmdd_to_iso, _assign_hop_ids, record_to_segment, record_to_visual_caption, record_to_evidence, record_to_rationale_context
        evs = record_to_evidence(dataset_record)
        assert len(evs) == 9

    def test_evidence_ids(self, dataset_record):
        from dataset.true_dataset_loader import record_to_pipeline_inputs, _yyyymmdd_to_iso, _assign_hop_ids, record_to_segment, record_to_visual_caption, record_to_evidence, record_to_rationale_context
        evs = record_to_evidence(dataset_record)
        ids = [e.evidence_id for e in evs]
        assert "1942500-ev1" in ids
        assert "1942500-ev9" in ids

    def test_evidence_retrieval_score(self, dataset_record):
        from dataset.true_dataset_loader import record_to_pipeline_inputs, _yyyymmdd_to_iso, _assign_hop_ids, record_to_segment, record_to_visual_caption, record_to_evidence, record_to_rationale_context
        evs = record_to_evidence(dataset_record)
        assert all(e.retrieval_score == 1.0 for e in evs)

    def test_evidence_hop_ids_assigned(self, dataset_record):
        from dataset.true_dataset_loader import record_to_pipeline_inputs, _yyyymmdd_to_iso, _assign_hop_ids, record_to_segment, record_to_visual_caption, record_to_evidence, record_to_rationale_context
        evs = record_to_evidence(dataset_record)
        ev_map = {e.evidence_id: e for e in evs}
        # evidence1 appears in a claim relationship → should have hop_id assigned
        ev1 = ev_map["1942500-ev1"]
        assert len(ev1.hop_ids) >= 1

    def test_evidence_source_url_fallback(self, dataset_record):
        from dataset.true_dataset_loader import record_to_pipeline_inputs, _yyyymmdd_to_iso, _assign_hop_ids, record_to_segment, record_to_visual_caption, record_to_evidence, record_to_rationale_context
        evs = record_to_evidence(dataset_record)
        # evidence1 has no URLs → should fall back to video_url
        ev1 = next(e for e in evs if e.evidence_id == "1942500-ev1")
        assert "nbcnews" in ev1.source_url

    def test_evidence_source_url_from_list(self, dataset_record):
        from dataset.true_dataset_loader import record_to_pipeline_inputs, _yyyymmdd_to_iso, _assign_hop_ids, record_to_segment, record_to_visual_caption, record_to_evidence, record_to_rationale_context
        evs = record_to_evidence(dataset_record)
        ev7 = next(e for e in evs if e.evidence_id == "1942500-ev7")
        # evidence7 has URLs → first one should be used
        assert ev7.source_url.startswith("https://")
        assert "nbcnews" not in ev7.source_url   # not the fallback

    def test_record_to_rationale_context(self, dataset_record):
        from dataset.true_dataset_loader import record_to_pipeline_inputs, _yyyymmdd_to_iso, _assign_hop_ids, record_to_segment, record_to_visual_caption, record_to_evidence, record_to_rationale_context
        ctx = record_to_rationale_context(dataset_record)
        assert ctx.gold_verdict == "refuted"
        assert ctx.snopes_rating == "Mostly False"
        assert "Obama" in ctx.main_rationale
        assert len(ctx.additional_rationales) == 3
        assert len(ctx.detailed_reasons) == 3

    def test_rationale_context_prompt_summary(self, dataset_record):
        from dataset.true_dataset_loader import record_to_pipeline_inputs, _yyyymmdd_to_iso, _assign_hop_ids, record_to_segment, record_to_visual_caption, record_to_evidence, record_to_rationale_context
        ctx = record_to_rationale_context(dataset_record)
        summary = ctx.prompt_summary()
        assert "Mostly False" in summary
        assert len(summary) <= 600

    def test_rationale_context_prompt_summary_truncation(self, dataset_record):
        from dataset.true_dataset_loader import record_to_pipeline_inputs, _yyyymmdd_to_iso, _assign_hop_ids, record_to_segment, record_to_visual_caption, record_to_evidence, record_to_rationale_context
        ctx = record_to_rationale_context(dataset_record)
        summary = ctx.prompt_summary(max_chars=50)
        assert len(summary) <= 50

    def test_record_to_pipeline_inputs(self, dataset_record):
        from dataset.true_dataset_loader import record_to_pipeline_inputs, _yyyymmdd_to_iso, _assign_hop_ids, record_to_segment, record_to_visual_caption, record_to_evidence, record_to_rationale_context
        inputs = record_to_pipeline_inputs(dataset_record)
        assert inputs["claim_text"] == dataset_record.claim
        assert inputs["gold_verdict"] == "refuted"
        assert inputs["gold_label"] >= 0
        assert inputs["segment"]["segment_id"] == "1942500"
        assert len(inputs["initial_evidence"]) == 9
        assert inputs["visual_caption"] != ""
        assert inputs["claim_id"] != ""

    def test_claim_id_is_safe_string(self, dataset_record):
        from dataset.true_dataset_loader import record_to_pipeline_inputs, _yyyymmdd_to_iso, _assign_hop_ids, record_to_segment, record_to_visual_caption, record_to_evidence, record_to_rationale_context
        import re
        inputs = record_to_pipeline_inputs(dataset_record)
        # Should contain only alphanumeric, underscore, dash
        assert re.match(r'^[a-zA-Z0-9_-]+$', inputs["claim_id"])
        assert len(inputs["claim_id"]) <= 64

    def test_date_conversion(self):
        from dataset.true_dataset_loader import record_to_pipeline_inputs, _yyyymmdd_to_iso, _assign_hop_ids, record_to_segment, record_to_visual_caption, record_to_evidence, record_to_rationale_context
        assert _yyyymmdd_to_iso(20160217.0) == "2016-02-17"
        assert _yyyymmdd_to_iso(20000101.0) == "2000-01-01"
        assert _yyyymmdd_to_iso(0.0) == ""  # invalid

    def test_hop_id_assignment_claim_rels_first(self, dataset_record):
        from dataset.true_dataset_loader import record_to_pipeline_inputs, _yyyymmdd_to_iso, _assign_hop_ids, record_to_segment, record_to_visual_caption, record_to_evidence, record_to_rationale_context
        hop_map = _assign_hop_ids(
            dataset_record.evidences.entries,
            dataset_record.relationship_with_evidence,
        )
        # Evidence 1 and 2 are in claim relationships → should have hop 1 and 2
        assigned = {idx: hops for idx, hops in hop_map.items() if hops}
        assert len(assigned) >= 2

    def test_no_duplicate_hop_ids_per_evidence(self, dataset_record):
        from dataset.true_dataset_loader import record_to_pipeline_inputs, _yyyymmdd_to_iso, _assign_hop_ids, record_to_segment, record_to_visual_caption, record_to_evidence, record_to_rationale_context
        hop_map = _assign_hop_ids(
            dataset_record.evidences.entries,
            dataset_record.relationship_with_evidence,
        )
        for idx, hops in hop_map.items():
            assert len(hops) == len(set(hops)), f"Duplicate hop_ids for evidence {idx}"


# ===========================================================================
# dataset_loader
# ===========================================================================

class TestDatasetLoader:

    def _write_jsonl(self, records: list[dict], path: str) -> None:
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def _write_json_array(self, records: list[dict], path: str) -> None:
        with open(path, "w") as f:
            json.dump(records, f)

    def test_load_jsonl(self, raw_record):
        from dataset.true_dataset_loader import split_records
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            f.write(json.dumps(raw_record) + "\n")
            f.write(json.dumps(raw_record) + "\n")
            tmp_path = f.name
        try:
            loader = DatasetLoader(tmp_path)
            records = loader.load_all()
            assert len(records) == 2
            assert records[0]["claim"] == raw_record["claim"]
        finally:
            os.unlink(tmp_path)

    def test_load_json_array(self, raw_record):
        from dataset.true_dataset_loader import split_records
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump([raw_record, raw_record], f)
            tmp_path = f.name
        try:
            records = DatasetLoader(tmp_path).load_all()
            assert len(records) == 2
        finally:
            os.unlink(tmp_path)

    def test_load_single_json_object(self, raw_record):
        from dataset.true_dataset_loader import split_records
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(raw_record, f)
            tmp_path = f.name
        try:
            records = DatasetLoader(tmp_path).load_all()
            assert len(records) == 1
        finally:
            os.unlink(tmp_path)

    def test_max_records(self, raw_record):
        from dataset.true_dataset_loader import split_records
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            for _ in range(5):
                f.write(json.dumps(raw_record) + "\n")
            tmp_path = f.name
        try:
            records = DatasetLoader(tmp_path, max_records=2).load_all()
            assert len(records) == 2
        finally:
            os.unlink(tmp_path)

    def test_skip_malformed_lines(self, raw_record):
        from dataset.true_dataset_loader import split_records
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            f.write(json.dumps(raw_record) + "\n")
            f.write("NOT JSON\n")
            f.write(json.dumps(raw_record) + "\n")
            tmp_path = f.name
        try:
            # skip_errors=True by default — malformed line should be skipped
            records = DatasetLoader(tmp_path, skip_errors=True).load_all()
            assert len(records) == 2
        finally:
            os.unlink(tmp_path)

    def test_from_dict(self, raw_record):
        from dataset.true_dataset_loader import split_records
        record = DatasetLoader.from_dict(raw_record)
        assert record.rating == "Mostly False"

    def test_from_json_string(self, raw_record):
        from dataset.true_dataset_loader import split_records
        record = DatasetLoader.from_json_string(json.dumps(raw_record))
        assert record.claim == raw_record["claim"]

    def test_file_not_found(self):
        from dataset.true_dataset_loader import split_records
        with pytest.raises(FileNotFoundError):
            DatasetLoader("/nonexistent/path/file.jsonl")

    def test_split_records(self, raw_record):
        from dataset.true_dataset_loader import split_records
        records = [DatasetLoader.from_dict(raw_record) for _ in range(20)]
        train, val, test = split_records(records, train_frac=0.7, val_frac=0.15, seed=0)
        assert len(train) + len(val) + len(test) == 20
        assert len(train) > len(val)
        assert len(train) > len(test)

    def test_split_deterministic(self, raw_record):
        from dataset.true_dataset_loader import split_records
        records = [DatasetLoader.from_dict(raw_record) for _ in range(10)]
        t1, v1, s1 = split_records(records, seed=42)
        t2, v2, s2 = split_records(records, seed=42)
        assert len(t1) == len(t2)
        assert len(v1) == len(v2)


# ===========================================================================
# dataset_pipeline
# ===========================================================================

class TestDatasetPipeline:

    def _make_stub_bundle(self):
        from models.model_bundle import ModelBundle
        import torch

        bundle = MagicMock(spec=ModelBundle)
        bundle.caption_fn.return_value = "Obama speaking at a podium."
        bundle.nli.entailment_score.return_value = 0.72
        bundle.encoder.encode.return_value = torch.zeros(1, 384)

        decomp_resp = {
            "claim_id": "test-claim",
            "sub_questions": [
                {"hop": 1, "question": "Did Obama attend Scalia's funeral?",
                 "depends_on_hops": [], "evidence_type": "web"},
                {"hop": 2, "question": "Was Obama playing golf?",
                 "depends_on_hops": [1], "evidence_type": "web"},
            ],
        }
        hop1 = {"hop": 1, "question": "Did Obama attend Scalia's funeral?",
                "answer": "No — Obama attended the visitation, not the funeral.",
                "confidence": 0.89, "supported_by": ["1942500-ev1"],
                "answer_unknown": False}
        hop2 = {"hop": 2, "question": "Was Obama playing golf?",
                "answer": "No evidence supports the golf claim.",
                "confidence": 0.84, "supported_by": ["1942500-ev7"],
                "answer_unknown": False}
        verdict_resp = {
            "claim_id": "test-claim", "verdict": "refuted", "confidence": 0.90,
            "reasoning_trace": [
                {"step": 1, "finding": "Obama attended visitation, not funeral.",
                 "source_hop": 1, "evidence_ids": ["1942500-ev1"]},
            ],
            "modal_conflict_used": False,
            "counterfactual": "Would be supported if Obama had played golf instead.",
        }
        summary = {"summary": "Obama paid his respects at the visitation."}

        bundle.decomposer_llm.generate.return_value = json.dumps(decomp_resp)
        bundle.hop_llm.generate.side_effect = (
            [json.dumps(hop1), json.dumps(hop2)]   # hop answers
            + [json.dumps(summary), json.dumps(summary)]  # summaries
        )
        bundle.aggregator_llm.generate.return_value = json.dumps(verdict_resp)
        return bundle

    def _make_stub_retriever(self):
        from modules.module4_targeted_retrieval import DenseRetriever
        r = MagicMock(spec=DenseRetriever)
        r.search.return_value = []
        return r

    def test_synthetic_caption_used_when_no_keyframes(self, dataset_record):
        from dataset.true_dataset_loader import record_to_pipeline_inputs, _yyyymmdd_to_iso, _assign_hop_ids, record_to_segment, record_to_visual_caption, record_to_evidence, record_to_rationale_context
        from run_pipeline import run_dataset_pipeline

        inputs = record_to_pipeline_inputs(dataset_record)
        assert inputs["segment"]["keyframes"] == []   # no frames in dataset

        bundle = self._make_stub_bundle()
        retriever = self._make_stub_retriever()

        report = run_dataset_pipeline(inputs, bundle, retriever, use_rationale_hints=True)

        # VLM captioner should NOT have been called (no keyframes)
        bundle.caption_fn.assert_not_called()
        assert report is not None

    def test_vlm_called_when_keyframes_present(self, dataset_record):
        from dataset.true_dataset_loader import record_to_pipeline_inputs, _yyyymmdd_to_iso, _assign_hop_ids, record_to_segment, record_to_visual_caption, record_to_evidence, record_to_rationale_context
        from run_pipeline import run_dataset_pipeline

        inputs = record_to_pipeline_inputs(dataset_record, keyframe_paths=["f.jpg"])
        bundle = self._make_stub_bundle()
        retriever = self._make_stub_retriever()

        run_dataset_pipeline(inputs, bundle, retriever)
        bundle.caption_fn.assert_called_once_with(["f.jpg"])

    def test_rationale_hint_injected_into_decomposer(self, dataset_record):
        from dataset.true_dataset_loader import record_to_pipeline_inputs, _yyyymmdd_to_iso, _assign_hop_ids, record_to_segment, record_to_visual_caption, record_to_evidence, record_to_rationale_context
        from run_pipeline import run_dataset_pipeline

        inputs = record_to_pipeline_inputs(dataset_record)
        bundle = self._make_stub_bundle()
        retriever = self._make_stub_retriever()

        run_dataset_pipeline(inputs, bundle, retriever, use_rationale_hints=True)

        # The prompt passed to the decomposer LLM should include rationale context
        call_args = bundle.decomposer_llm.generate.call_args
        prompt = call_args[0][0]   # first positional arg
        # prompt is a list of chat dicts
        user_content = next(m["content"] for m in prompt if m["role"] == "user")
        assert "Mostly False" in user_content or "Obama" in user_content

    def test_rationale_hint_omitted_when_disabled(self, dataset_record):
        from dataset.true_dataset_loader import record_to_pipeline_inputs, _yyyymmdd_to_iso, _assign_hop_ids, record_to_segment, record_to_visual_caption, record_to_evidence, record_to_rationale_context
        from run_pipeline import run_dataset_pipeline

        inputs = record_to_pipeline_inputs(dataset_record)
        bundle = self._make_stub_bundle()
        retriever = self._make_stub_retriever()

        run_dataset_pipeline(inputs, bundle, retriever, use_rationale_hints=False)

        call_args = bundle.decomposer_llm.generate.call_args
        prompt = call_args[0][0]
        user_content = next(m["content"] for m in prompt if m["role"] == "user")
        # Rationale context block should not appear
        assert "Known rationale context" not in user_content

    def test_run_dataset_record_returns_eval_result(self, dataset_record):
        from run_pipeline import run_dataset_record

        bundle = self._make_stub_bundle()
        retriever = self._make_stub_retriever()

        result = run_dataset_record(dataset_record, bundle, retriever)

        assert result["gold_verdict"] == "refuted"
        assert result["pred_verdict"] in (
            "supported", "refuted", "insufficient_evidence", "misleading_context"
        )
        assert isinstance(result["correct"], bool)
        assert result["claim_id"] != ""
        assert result["report"] is not None

    def test_correct_flag_set_properly(self, dataset_record):
        from run_pipeline import run_dataset_record

        bundle = self._make_stub_bundle()
        retriever = self._make_stub_retriever()

        result = run_dataset_record(dataset_record, bundle, retriever)
        assert result["correct"] == (result["pred_verdict"] == result["gold_verdict"])


# ===========================================================================
# evaluation
# ===========================================================================

class TestEvaluation:

    def _make_result(self, gold_verdict, pred_verdict, confidence=0.8):
        """Build a minimal evaluation-result dict stub."""
        from dataset.true_dataset_loader import verdict_to_label

        report = {
            "claim_id": "c",
            "segment_id": "s",
            "verdict": pred_verdict,
            "confidence": confidence,
        }
        return {
            "claim_id": "c",
            "gold_verdict": gold_verdict,
            "gold_label": verdict_to_label(gold_verdict),
            "pred_verdict": pred_verdict,
            "pred_label": verdict_to_label(pred_verdict),
            "pred_confidence": confidence,
            "correct": gold_verdict == pred_verdict,
            "report": report,
        }

    def test_perfect_accuracy(self):
        from dataset.evaluation import compute_metrics
        results = [
            self._make_result("refuted",    "refuted"),
            self._make_result("supported",  "supported"),
            self._make_result("refuted",    "refuted"),
        ]
        summary = compute_metrics(results)
        assert summary.accuracy == pytest.approx(1.0)
        assert summary.num_correct == 3
        assert summary.num_records == 3

    def test_zero_accuracy(self):
        from dataset.evaluation import compute_metrics
        results = [
            self._make_result("refuted",   "supported"),
            self._make_result("supported", "refuted"),
        ]
        summary = compute_metrics(results)
        assert summary.accuracy == pytest.approx(0.0)
        assert summary.num_correct == 0

    def test_partial_accuracy(self):
        from dataset.evaluation import compute_metrics
        results = [
            self._make_result("refuted", "refuted"),     # correct
            self._make_result("refuted", "supported"),   # wrong
        ]
        summary = compute_metrics(results)
        assert summary.accuracy == pytest.approx(0.5)

    def test_per_class_tp_fp_fn(self):
        from dataset.evaluation import compute_metrics
        results = [
            self._make_result("refuted",   "refuted"),    # TP for refuted
            self._make_result("refuted",   "supported"),  # FN for refuted, FP for supported
            self._make_result("supported", "supported"),  # TP for supported
        ]
        summary = compute_metrics(results)
        refuted_cls = next(c for c in summary.per_class if c.verdict == "refuted")
        supported_cls = next(c for c in summary.per_class if c.verdict == "supported")
        assert refuted_cls.tp == 1
        assert refuted_cls.fn == 1
        assert supported_cls.tp == 1
        assert supported_cls.fp == 1

    def test_precision_recall_f1(self):
        from dataset.evaluation import compute_metrics
        results = [
            self._make_result("refuted", "refuted"),
            self._make_result("refuted", "refuted"),
            self._make_result("refuted", "supported"),
        ]
        summary = compute_metrics(results)
        refuted_cls = next(c for c in summary.per_class if c.verdict == "refuted")
        assert refuted_cls.precision == pytest.approx(1.0)   # 2 TP, 0 FP
        assert refuted_cls.recall    == pytest.approx(2/3, abs=1e-4)
        assert refuted_cls.f1        == pytest.approx(2 * 1.0 * (2/3) / (1.0 + 2/3), abs=1e-4)

    def test_macro_f1_unweighted(self):
        from dataset.evaluation import compute_metrics
        results = [
            self._make_result("refuted",  "refuted"),
            self._make_result("supported","supported"),
        ]
        summary = compute_metrics(results)
        # Both active classes have F1=1.0 → macro should be 1.0
        assert summary.macro_f1 == pytest.approx(1.0)

    def test_zero_division_safety(self):
        from dataset.evaluation import compute_metrics
        # All predictions are "refuted" — supported class has 0 TP + 0 FP
        results = [
            self._make_result("refuted",  "refuted"),
            self._make_result("refuted",  "refuted"),
        ]
        summary = compute_metrics(results)
        # Should not raise; unsupported classes have support=0 and are excluded
        assert summary.macro_f1 >= 0.0

    def test_distributions_tracked(self):
        from dataset.evaluation import compute_metrics
        results = [
            self._make_result("refuted",   "refuted"),
            self._make_result("refuted",   "supported"),
            self._make_result("supported", "refuted"),
        ]
        summary = compute_metrics(results)
        assert summary.gold_distribution["refuted"] == 2
        assert summary.gold_distribution["supported"] == 1
        assert summary.pred_distribution["refuted"] == 2
        assert summary.pred_distribution["supported"] == 1

    def test_to_dict_structure(self):
        from dataset.evaluation import compute_metrics
        results = [self._make_result("refuted", "refuted")]
        d = compute_metrics(results).to_dict()
        assert "accuracy" in d
        assert "macro_f1" in d
        assert "per_class" in d
        assert isinstance(d["per_class"], list)

    def test_to_json_parses(self):
        from dataset.evaluation import compute_metrics
        results = [self._make_result("refuted", "refuted")]
        j = compute_metrics(results).to_json()
        parsed = json.loads(j)
        assert parsed["num_records"] == 1

    def test_str_output(self):
        from dataset.evaluation import compute_metrics
        results = [self._make_result("refuted", "refuted")]
        s = str(compute_metrics(results))
        assert "Accuracy" in s
        assert "Macro" in s

    def test_empty_results_raises(self):
        from dataset.evaluation import compute_metrics
        with pytest.raises(ValueError):
            compute_metrics([])
