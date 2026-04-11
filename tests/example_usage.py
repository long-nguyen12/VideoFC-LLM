"""
example_usage.py
----------------
End-to-end usage example for the video fact-checking pipeline.

Running this script requires all dependencies from requirements.txt.
The first run will download ~10–15 GB of model weights from HuggingFace Hub.

Quick smoke-test (no GPU, no model downloads):
    python example_usage.py --stub

Full run (requires CUDA for comfortable speed):
    python example_usage.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import os

# Allow running from the project root without installation
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("example")


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

SAMPLE_CLAIM_TEXT = (
    "The Golden Gate Bridge was closed for an entire week in January 2024 "
    "due to structural emergency repairs."
)
SAMPLE_CLAIM_ID = "claim-golden-gate-001"

SAMPLE_SEGMENT_DATA = {
    "segment_id": "seg-golden-gate-001",
    "start_ts": 0.0,
    "end_ts": 45.0,
    "transcript": (
        "The Golden Gate Bridge authority announced a two-day partial lane "
        "closure on January 8th 2024 for routine maintenance. The bridge "
        "remained open to pedestrian and bicycle traffic throughout."
    ),
    "keyframes": [],   # Empty: will use a placeholder caption in stub mode
}

SAMPLE_PASSAGES = [
    {
        "evidence_id": "ev-001",
        "source_url": "https://goldengate.org/press/2024-01-08",
        "source_date": "2024-01-08",
        "passage_text": (
            "A scheduled two-day lane closure was conducted on the Golden Gate "
            "Bridge starting January 8th 2024 for resurfacing work. The bridge "
            "was not closed in its entirety at any point."
        ),
        "retrieval_score": 0.95,
        "hop_ids": [1, 2],
    },
    {
        "evidence_id": "ev-002",
        "source_url": "https://sfchronicle.com/bridge-2024",
        "source_date": "2024-01-09",
        "passage_text": (
            "Bridge officials confirmed no structural emergency was declared. "
            "The maintenance was routine and had been planned months in advance."
        ),
        "retrieval_score": 0.88,
        "hop_ids": [2, 3],
    },
    {
        "evidence_id": "ev-003",
        "source_url": "https://dot.ca.gov/bridges",
        "source_date": "2024-01-07",
        "passage_text": (
            "Caltrans records show no structural emergency alerts were issued "
            "for any Bay Area bridge crossing in January 2024."
        ),
        "retrieval_score": 0.82,
        "hop_ids": [3],
    },
]


# ---------------------------------------------------------------------------
# Stub mode helpers (no model downloads)
# ---------------------------------------------------------------------------

def _build_stub_bundle():
    """Return a ModelBundle where every model is a deterministic stub."""
    import json
    from unittest.mock import MagicMock
    from models.model_bundle import ModelBundle
    import torch

    captioner = MagicMock()
    captioner.caption.return_value = (
        "An aerial view of the Golden Gate Bridge with light vehicle traffic."
    )

    nli = MagicMock()
    nli.entailment_score.return_value = 0.72

    encoder = MagicMock()
    encoder.encode.return_value = torch.zeros(1, 384)

    decomp_resp = {
        "claim_id": SAMPLE_CLAIM_ID,
        "sub_questions": [
            {"hop": 1, "question": "Was the Golden Gate Bridge closed for a full week?",
             "depends_on_hops": [], "evidence_type": "web"},
            {"hop": 2, "question": "Was the closure due to a structural emergency?",
             "depends_on_hops": [1], "evidence_type": "web"},
            {"hop": 3, "question": "Did the closure last from January 1–7 2024?",
             "depends_on_hops": [1], "evidence_type": "web"},
        ],
    }

    hop_responses = [
        {"hop": 1, "question": "Was the Golden Gate Bridge closed for a full week?",
         "answer": "No — only a two-day partial lane closure occurred.",
         "confidence": 0.91, "supported_by": ["ev-001"], "answer_unknown": False},
        {"hop": 2, "question": "Was the closure due to a structural emergency?",
         "answer": "No — the closure was routine, pre-planned resurfacing work.",
         "confidence": 0.89, "supported_by": ["ev-002"], "answer_unknown": False},
        {"hop": 3, "question": "Did the closure last from January 1–7 2024?",
         "answer": "The closure started January 8th and lasted two days.",
         "confidence": 0.85, "supported_by": ["ev-001", "ev-003"], "answer_unknown": False},
    ]

    verdict_resp = {
        "claim_id": SAMPLE_CLAIM_ID,
        "verdict": "refuted",
        "confidence": 0.91,
        "reasoning_trace": [
            {"step": 1, "finding": "The closure lasted two days, not a full week.",
             "source_hop": 1, "evidence_ids": ["ev-001"]},
            {"step": 2, "finding": "No structural emergency was declared.",
             "source_hop": 2, "evidence_ids": ["ev-002"]},
            {"step": 3, "finding": "The closure began January 8th, not January 1st.",
             "source_hop": 3, "evidence_ids": ["ev-001", "ev-003"]},
        ],
        "modal_conflict_used": False,
        "counterfactual": (
            "The claim would be supported if official records confirmed a "
            "seven-day full bridge closure and a structural emergency declaration."
        ),
    }

    summary_responses = [
        {"summary": "The bridge was only partially closed for two days, not a full week."},
        {"summary": "The closure was routine maintenance, not an emergency."},
        {"summary": "The closure started on January 8th and lasted just two days."},
    ]

    hop_llm = MagicMock()
    # hop_llm.generate is called for each hop (Module 5) then each summary (Module 7)
    hop_llm.generate.side_effect = (
        [json.dumps(r) for r in hop_responses]
        + [json.dumps(r) for r in summary_responses]
    )

    decomposer = MagicMock()
    decomposer.generate.return_value = json.dumps(decomp_resp)

    aggregator = MagicMock()
    aggregator.generate.return_value = json.dumps(verdict_resp)

    bundle = MagicMock(spec=ModelBundle)
    bundle.caption_fn.return_value = captioner.caption.return_value
    bundle.nli = nli
    bundle.encoder = encoder
    bundle.decomposer_llm = decomposer
    bundle.hop_llm = hop_llm
    bundle.aggregator_llm = aggregator

    return bundle


def _build_stub_retriever():
    from unittest.mock import MagicMock
    from modules.module4_targeted_retrieval import DenseRetriever

    r = MagicMock(spec=DenseRetriever)
    r.search.return_value = []
    return r


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(stub: bool = False, single_llm: bool = False) -> None:
    from schemas.data_models import VideoSegment, EvidenceRef

    segment = VideoSegment(**SAMPLE_SEGMENT_DATA)
    initial_evidence = [EvidenceRef(**p) for p in SAMPLE_PASSAGES]

    if stub:
        logger.info("Running in STUB mode — no models loaded.")
        models = _build_stub_bundle()
        retriever = _build_stub_retriever()
    elif single_llm:
        logger.info("Loading single-LLM bundle (hardware-constrained mode).")
        from models.model_bundle import load_single_llm_bundle
        from modules.module4_targeted_retrieval import DenseRetriever

        models = load_single_llm_bundle(
            llm_model="Qwen/Qwen2.5-1.5B-Instruct",
            captioner_model=None,        # no keyframes in this example
            load_in_4bit=False,
            context_window=2048,
        )
        retriever = DenseRetriever(models.encoder)
        retriever.index(initial_evidence)
    else:
        logger.info("Loading full multi-model bundle — this may take several minutes.")
        from models.model_bundle import load_default_bundle
        from modules.module4_targeted_retrieval import DenseRetriever

        models = load_default_bundle(load_in_4bit=True)
        retriever = DenseRetriever(models.encoder)
        retriever.index(initial_evidence)

    from pipeline import run_pipeline
    from modules.module1_claim_decomposer import decompose_claim as _dc
    import modules.module1_claim_decomposer as _m1

    # When running in single-llm mode, cap sub-questions at 3
    max_sub = 3 if single_llm else 5

    report = run_pipeline(
        claim_text=SAMPLE_CLAIM_TEXT,
        claim_id=SAMPLE_CLAIM_ID,
        segment=segment,
        initial_evidence=initial_evidence,
        models=models,
        retriever=retriever,
    )

    # ---------------------------------------------------------------------------
    # Pretty-print results
    # ---------------------------------------------------------------------------
    mode_label = "STUB" if stub else ("SINGLE-LLM" if single_llm else "FULL")
    print("\n" + "=" * 70)
    print(f"FACT-CHECK RESULT  [{mode_label} MODE]")
    print("=" * 70)
    print(f"Claim    : {SAMPLE_CLAIM_TEXT}")
    print(f"Verdict  : {report.verdict.upper()}")
    print(f"Confidence: {report.confidence:.2%}")
    print(f"Gate passed: {report.gate_passed}  |  Retrieval rounds: {report.retrieval_rounds}")

    print("\n--- Hop Summaries ---")
    for i, summary in enumerate(report.hop_summaries, 1):
        print(f"  [{i}] {summary}")

    print("\n--- Evidence Saliency ---")
    for s in report.evidence_saliency:
        print(f"  [{s.evidence_id}] hop={s.hop}  score={s.saliency_score:.4f}")
        print(f"       key span: \"{s.key_span}\"")

    if report.modal_annotations:
        print("\n--- Modal Conflict Annotations ---")
        for ann in report.modal_annotations:
            print(f"  {ann.human_note}")
    else:
        print("\n--- Modal Conflict Annotations: none ---")

    print(f"\n--- Counterfactual ---\n  {report.counterfactual}")
    print("=" * 70)

    print("\n[Full JSON report]")
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video fact-checking pipeline example")
    parser.add_argument(
        "--stub", action="store_true",
        help="Run with stub models (no GPU or downloads required).",
    )
    parser.add_argument(
        "--single-llm", dest="single_llm", action="store_true",
        help=(
            "Run with a single small LLM for all roles "
            "(Qwen2.5-1.5B, ~3.5 GB VRAM). "
            "Downloads ~3 GB on first run."
        ),
    )
    args = parser.parse_args()
    main(stub=args.stub, single_llm=args.single_llm)
