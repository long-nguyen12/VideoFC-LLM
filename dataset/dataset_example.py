"""
dataset_example.py
------------------
Demonstrates the full dataset integration:
  1. Parse a raw dataset record
  2. Inspect all mapped fields
  3. Run the pipeline (stub mode — no GPU/downloads needed)
  4. Evaluate against the gold label
  5. Simulate a batch evaluation over multiple records

Run:
    python dataset_example.py          # full demo
    python dataset_example.py --quiet  # suppress verbose field printing
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.WARNING,  # suppress pipeline debug noise by default
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("dataset_example")


# ---------------------------------------------------------------------------
# The Snopes/Obama sample record (one full record from the dataset)
# ---------------------------------------------------------------------------

RAW_RECORD = {
    "url": "https://www.snopes.com/fact-check/obama-snubs-scalias-funeral/",
    "claim": "President Obama snubbed the funeral of Supreme Court Justice Antonin Scalia in order to play golf.",
    "rating": "Mostly False",
    "content": "President Obama opted to attend a visitation for Justice Scalia but did not attend his funeral.\nPresident Obama didn't fail to pay respect to Justice Scalia or play golf instead of attending his funeral.",
    "video_information": {
        "video_id": "1942500",
        "video_date": 20160217.0,
        "platform": "nbcnews",
        "video_headline": "Obama speaks out on battle to fill Scalia's seat",
        "video_transcript": (
            "And now to the heated debate over replacing supreme court justice Antonin Scalia, "
            "president Obama now waging into the Republican opposition against him naming Scalia "
            "successor. The constitution is pretty clear about what is supposed to happen now."
        ),
        "video_description": "President Obama is responding firmly to Republican opposition to him naming Scalia's successor.",
        "video_length": 157.918844,
        "video_url": "https://www.nbcnews.com/news/us-news/white-house-obama-will-not-attend-justice-scalia-s-funeral-n520236",
    },
    "original_rationales": {
        "main_rationale": "President Obama didn't fail to pay respect to Justice Scalia or play golf instead of attending his funeral, and Presidents do not historically attend all funerals of Supreme Court justices.",
        "additional_rationale1": "President Obama opted to attend a visitation for Justice Scalia but did not attend his funeral.",
        "additional_rationale2": "The optics of paying his respects to Scalia are tricky for Obama, who would have been the subject of constant cutaways to his reactions.",
        "additional_rationale3": "In actuality, the President spent that weekend reading through lengthy dossiers and job histories of potential candidates for Scalia's replacement.",
    },
    "summary_rationales": {
        "synthesized_rationale": (
            "The claim that President Obama snubbed Justice Scalia's funeral to play golf is rated "
            "'Mostly False' because it lacks evidence and is based on a misinterpretation of "
            "presidential duties and attendance at funerals."
        ),
        "detailed_reasons": {
            "reason1": "Presidents do not have a precedent of attending the funerals of all Supreme Court justices. Historically, fewer than half of recent funerals included the president or vice president.",
            "reason2": "The article supports that Obama did attend the visitation and intended to pay his respects appropriately, undermining the claim.",
            "reason3": "Reports suggesting Obama was playing golf were based on speculation rather than fact.",
        },
    },
    "evidences": {
        "num_of_evidence": 9,
        "evidence1": ["Instead, the president will pay his respects on Friday, when Scalia's body lies in repose in the Great Hall of the Supreme Court building.", []],
        "evidence2": ["Earnest revealed the president's plans during the daily briefing, saying Obama and first lady Michelle Obama will go to the Supreme Court on Friday to pay their respects.", []],
        "evidence3": ["'I wouldn't have expected President Obama to attend the funeral Mass, and I see no reason to fault him for not attending,' said Ed Whelan.", []],
        "evidence4": ["Josh Earnest implied that one reason was the potential for the extensive presidential security detail to be disruptive.", []],
        "evidence5": ["Former President George W. Bush attended the funeral for Chief Justice William Rehnquist.", []],
        "evidence6": ["Of the approximately 100 justices who have served on the court and left, a little fewer than half have died while still holding the position.", []],
        "evidence7": ["Oh man...is Obama planning to golf through Scalia's funeral?", ["https://t.co/LrSJmVKpBp", "https://twitter.com/guypbenson/status/700062310220562432"]],
        "evidence8": ["WATCH: White House Says THIS About Obama Golfing During Scalia Funeral.", ["https://t.co/y5um4IalgV"]],
        "evidence9": ["@POTUS must be too busy golfing to attend the funeral of the late Justice Scalia.", ["https://twitter.com/eric_poitras/status/701129931493941249"]],
    },
    "relationship_with_evidence": [
        {"<claim,evidence1>": "Evidence directly counters the claim that President Obama snubbed the funeral."},
        {"<claim,evidence2>": "Evidence clarifies that Obama planned to attend the visitation."},
        {"<main_rationale,evidence3>": "Evidence supports the main rationale by confirming Obama's absence was not a snub."},
        {"<main_rationale,evidence4>": "Evidence supports that the White House chose a respectful arrangement."},
        {"<main_rationale,evidence5>": "Evidence supports historical precedent for not attending justices' funerals."},
        {"<claim,evidence7>": "Tweet aligns with the claim's assertion about golf."},
        {"<claim,evidence9>": "Tweet reinforces the speculation around the golf claim."},
    ],
    "other": {"iframe_video_links": []},
}


# ---------------------------------------------------------------------------
# Stub model bundle (no GPU / no downloads)
# ---------------------------------------------------------------------------

def _build_stub_bundle():
    import json
    from unittest.mock import MagicMock
    from models.model_bundle import ModelBundle
    import torch

    bundle = MagicMock(spec=ModelBundle)
    bundle.caption_fn.return_value = "A news anchor delivers a report on the Supreme Court vacancy."
    bundle.nli.entailment_score.return_value = 0.68
    bundle.encoder.encode.return_value = torch.zeros(1, 384)

    decomp_resp = {
        "claim_id": "stub-claim",
        "sub_questions": [
            {"hop": 1, "question": "Did President Obama attend Justice Scalia's funeral?",
             "depends_on_hops": [], "evidence_type": "web"},
            {"hop": 2, "question": "Was President Obama playing golf instead of attending?",
             "depends_on_hops": [1], "evidence_type": "web"},
            {"hop": 3, "question": "What is the historical precedent for presidents attending Supreme Court funerals?",
             "depends_on_hops": [], "evidence_type": "web"},
        ],
    }

    hop_responses = [
        {"hop": 1, "question": "Did President Obama attend Justice Scalia's funeral?",
         "answer": "No — Obama attended a visitation the day before but not the funeral itself.",
         "confidence": 0.91, "supported_by": ["1942500-ev1", "1942500-ev2"], "answer_unknown": False},
        {"hop": 2, "question": "Was President Obama playing golf instead of attending?",
         "answer": "No evidence supports the golf claim; he was reviewing candidate dossiers.",
         "confidence": 0.88, "supported_by": ["1942500-ev3"], "answer_unknown": False},
        {"hop": 3, "question": "What is the historical precedent for presidents attending Supreme Court funerals?",
         "answer": "Fewer than half of recent justices' funerals were attended by the sitting president.",
         "confidence": 0.85, "supported_by": ["1942500-ev5"], "answer_unknown": False},
    ]

    verdict_resp = {
        "claim_id": "stub-claim",
        "verdict": "refuted",
        "confidence": 0.91,
        "reasoning_trace": [
            {"step": 1, "finding": "Obama attended the visitation, not the funeral — a deliberate and precedented choice.",
             "source_hop": 1, "evidence_ids": ["1942500-ev1", "1942500-ev2"]},
            {"step": 2, "finding": "The golf claim is based on speculation with no factual support.",
             "source_hop": 2, "evidence_ids": ["1942500-ev3"]},
            {"step": 3, "finding": "Presidential absence from Supreme Court funerals is historically common.",
             "source_hop": 3, "evidence_ids": ["1942500-ev5"]},
        ],
        "modal_conflict_used": False,
        "counterfactual": "The claim would be supported if contemporaneous evidence showed Obama played golf on the day of the funeral instead of preparing for Scalia's replacement.",
    }

    summary_responses = [
        {"summary": "Obama attended a respectful visitation but not the formal funeral, consistent with historical precedent."},
        {"summary": "No credible evidence supports the assertion that Obama chose golf over the funeral."},
        {"summary": "Presidents historically do not attend all Supreme Court justices' funerals."},
    ]

    bundle.decomposer_llm.generate.return_value = json.dumps(decomp_resp)
    bundle.hop_llm.generate.side_effect = (
        [json.dumps(r) for r in hop_responses]
        + [json.dumps(r) for r in summary_responses]
    )
    bundle.aggregator_llm.generate.return_value = json.dumps(verdict_resp)

    return bundle


def _build_stub_retriever():
    from unittest.mock import MagicMock
    from modules.module4_targeted_retrieval import DenseRetriever
    r = MagicMock(spec=DenseRetriever)
    r.search.return_value = []
    return r


# ---------------------------------------------------------------------------
# Demo helpers
# ---------------------------------------------------------------------------

def _print_section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print('=' * 70)


def _show_record_fields(record, verbose: bool) -> None:
    from dataset.dataset_adapter import (
        record_to_evidence,
        record_to_visual_caption,
        record_to_rationale_context,
    )

    _print_section("1. PARSED DATASET RECORD")
    print(f"  Claim   : {record.claim}")
    print(f"  Rating  : {record.rating}")
    print(f"  URL     : {record.url}")
    print(f"  Video ID: {record.video_information.video_id}")
    print(f"  Platform: {record.video_information.platform}")
    print(f"  Duration: {record.video_information.video_length:.1f}s")

    if verbose:
        print(f"\n  Transcript excerpt:")
        print(f"    {record.video_information.video_transcript[:200]}...")

        _print_section("2. SYNTHETIC VISUAL CAPTION")
        caption = record_to_visual_caption(record)
        print(f"  {caption}")

        _print_section("3. EVIDENCE PASSAGES")
        evs = record_to_evidence(record)
        for ev in evs[:4]:   # show first 4
            hop_str = str(ev.hop_ids) if ev.hop_ids else "unassigned"
            print(f"  [{ev.evidence_id}] hops={hop_str}")
            print(f"    {ev.passage_text[:100]}...")
            print(f"    source: {ev.source_url}")
        print(f"  ... ({len(evs)} total)")

        _print_section("4. RATIONALE CONTEXT")
        ctx = record_to_rationale_context(record)
        print(f"  Gold verdict : {ctx.gold_verdict}")
        print(f"  Snopes rating: {ctx.snopes_rating}")
        print(f"  Main rationale: {ctx.main_rationale[:120]}...")
        print(f"  Detailed reasons:")
        for i, r in enumerate(ctx.detailed_reasons, 1):
            print(f"    [{i}] {r[:100]}...")
        print(f"\n  Prompt summary (≤600 chars):")
        print(f"    {ctx.prompt_summary()}")


def _show_pipeline_result(result) -> None:
    _print_section("5. PIPELINE RESULT")

    report = result.report
    print(f"  Verdict     : {report.verdict.upper()}")
    print(f"  Confidence  : {report.confidence:.2%}")
    print(f"  Gate passed : {report.gate_passed}")
    print(f"  Retrieval rounds: {report.retrieval_rounds}")

    print(f"\n  Gold verdict : {result.gold_verdict.upper()}")
    print(f"  Correct      : {'✓' if result.correct else '✗'}  ({result.pred_verdict} vs {result.gold_verdict})")

    print(f"\n  Hop Summaries:")
    for i, s in enumerate(report.hop_summaries, 1):
        print(f"    [{i}] {s}")

    print(f"\n  Evidence Saliency (top 3):")
    sorted_sal = sorted(report.evidence_saliency, key=lambda x: x.saliency_score, reverse=True)[:3]
    for s in sorted_sal:
        print(f"    [{s.evidence_id}] hop={s.hop}  score={s.saliency_score:.4f}")
        print(f"         key span: \"{s.key_span[:80]}\"")

    if report.modal_annotations:
        print(f"\n  Modal Conflict Annotations:")
        for ann in report.modal_annotations:
            print(f"    {ann.human_note}")

    print(f"\n  Counterfactual:")
    print(f"    {report.counterfactual}")


def _show_batch_eval(records) -> None:
    from dataset.dataset_pipeline import run_dataset_record
    from dataset.evaluation import compute_metrics

    _print_section("6. BATCH EVALUATION (simulated)")

    bundle    = _build_stub_bundle()
    retriever = _build_stub_retriever()
    results   = []

    for i, rec in enumerate(records):
        # Rotate stub verdict to simulate diverse predictions
        verdicts = ["refuted", "supported", "refuted", "misleading_context", "refuted"]
        import json
        verdict_resp = {
            "claim_id": f"claim-{i}", "verdict": verdicts[i % len(verdicts)],
            "confidence": 0.80, "reasoning_trace": [],
            "modal_conflict_used": False, "counterfactual": "CF.",
        }
        bundle.aggregator_llm.generate.return_value = json.dumps(verdict_resp)

        try:
            result = run_dataset_record(rec, bundle, retriever, use_rationale_hints=True)
            results.append(result)
            print(f"  [{i+1}/{len(records)}] {result.pred_verdict:<22}  gold={result.gold_verdict:<22}  {'✓' if result.correct else '✗'}")
        except Exception as exc:
            print(f"  [{i+1}/{len(records)}] ERROR: {exc}")

    if results:
        summary = compute_metrics(results)
        print(f"\n{summary}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(verbose: bool = True) -> None:
    from dataset.dataset_loader import DatasetLoader
    from dataset.dataset_adapter import record_to_pipeline_inputs
    from dataset.dataset_pipeline import run_dataset_record

    print("\n" + "=" * 70)
    print("  VIDEO FACT-CHECKING DATASET INTEGRATION DEMO")
    print("=" * 70)

    # 1. Parse record
    record = DatasetLoader.from_dict(RAW_RECORD)

    # 2. Show parsed fields
    _show_record_fields(record, verbose=verbose)

    # 3. Run pipeline (stub mode)
    _print_section("RUNNING PIPELINE (stub mode)")
    bundle    = _build_stub_bundle()
    retriever = _build_stub_retriever()
    result    = run_dataset_record(record, bundle, retriever, use_rationale_hints=True)

    # 4. Show pipeline output
    _show_pipeline_result(result)

    # 5. Batch eval simulation (5 copies of the same record with different fake ratings)
    import copy
    from dataset.dataset_schemas import DatasetRecord

    ratings = ["Mostly False", "True", "Mostly False", "Mixture", "False"]
    batch_records = []
    for rating in ratings:
        raw = dict(RAW_RECORD)
        raw["rating"] = rating
        batch_records.append(DatasetLoader.from_dict(raw))

    _show_batch_eval(batch_records)

    # 6. Full JSON report
    if verbose:
        _print_section("FULL JSON REPORT")
        print(result.report.model_dump_json(indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset integration demo")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose field output")
    args = parser.parse_args()
    main(verbose=not args.quiet)
