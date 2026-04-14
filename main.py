"""
run_evaluation.py
-----------------
Main entry point to evaluate the Video Fact-Checking Pipeline on the TRUE dataset.

Usage:
    python run_evaluation.py --split test --max-records 100
    python run_evaluation.py --split train_val --use-stubs
    python run_evaluation.py --split test --4bit --output final_test_results.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from dataset import load_for_pipeline, run_dataset_record, compute_metrics, log_summary
from models.model_bundle import load_default_bundle, load_single_llm_bundle
from modules.module4_targeted_retrieval import DenseRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("run_evaluation")


def build_stub_models():
    """Loads lightweight mock models for rapid end-to-end testing."""
    from dataset.dataset_example import _build_stub_bundle, _build_stub_retriever
    return _build_stub_bundle(), _build_stub_retriever()


def main():
    parser = argparse.ArgumentParser(description="Evaluate video fact-checking pipeline.")
    parser.add_argument("--dataset-root", type=str, default="data/TRUE_Dataset", help="Path to TRUE dataset root.")
    parser.add_argument("--split", type=str, default="test", choices=["train_val", "test"], help="Dataset split to evaluate.")
    parser.add_argument("--max-records", type=int, default=None, help="Limit number of records (for testing).")
    parser.add_argument("--use-stubs", action="store_true", help="Use stubbed models to bypass GPU inference.")
    parser.add_argument("--use-rationale-hints", action="store_true", help="Inject gold rationale to guide reasoning.")
    parser.add_argument("--output", type=str, default=None, help="Path to save results JSON. If omitted, auto-generates a timestamped filename.")
    parser.add_argument("--4bit", action="store_true", dest="load_in_4bit", help="Load generative LLMs in 4-bit precision.")
    parser.add_argument("--single-model", action="store_true", help="Use a single shared LLM across all roles to save VRAM.")
    
    args = parser.parse_args()
    
    if args.output:
        out_path = Path(args.output)
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path(f"evaluation_{args.split}_{timestamp}.json")
    
    logger.info(f"Loading '{args.split}' split from {args.dataset_root}...")
    try:
        items = load_for_pipeline(
            path=args.dataset_root,
            split=args.split,
            limit_samples=args.max_records,
        )
    except FileNotFoundError as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)
        
    if not items:
        logger.warning("No records loaded. Exiting.")
        sys.exit(0)
        
    if args.use_stubs:
        logger.info("Initializing STUB models (fast mode, no GPU)...")
        bundle, retriever = build_stub_models()
    else:
        logger.info("Initializing REAL models (this may take a while)...")
        # Ensure imports and model initializations run appropriately
        import torch
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        
        if args.single_model:
            bundle = load_single_llm_bundle(load_in_4bit=args.load_in_4bit)
        else:
            bundle = load_default_bundle(load_in_4bit=args.load_in_4bit)
            
        retriever = DenseRetriever(bundle.encoder)
        
    results = []
    
    logger.info(f"Starting evaluation loop over {len(items)} records...")
    for i, (record, kf_paths) in enumerate(items, 1):
        vid_info = record["video_information"]
        vid = vid_info["video_id"]
        logger.info(f"Processing [{i}/{len(items)}] video_id={vid}...")
        print(f"DEBUG: kf_paths for {vid} from load_for_pipeline = {kf_paths}")
        
        try:
            # Note: run_dataset_record() internally indexes the record's specific evidence corpus 
            # using the retriever object, before running the pipeline.
            result = run_dataset_record(
                record=record,
                models=bundle,
                retriever=retriever,
                use_rationale_hints=args.use_rationale_hints,
                keyframe_paths=kf_paths if kf_paths else None,
            )
            results.append(result)
            
            is_correct = "✓" if result["correct"] else "✗"
            logger.info(
                f"  -> Pred: {result['pred_verdict']:<22} Gold: {result['gold_verdict']:<22} {is_correct}"
            )
            
            partial_summary = compute_metrics(results)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({
                    "summary": partial_summary.to_dict(),
                    "args": vars(args),
                    "status": f"In progress ({i}/{len(items)})",
                    "results": [
                        {
                            "claim_id": r["claim_id"],
                            "pred_verdict": r["pred_verdict"],
                            "gold_verdict": r["gold_verdict"],
                            "correct": r["correct"],
                            "report": r["report"],
                        }
                        for r in results
                    ]
                }, f, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to process record {vid}: {e}", exc_info=True)
            
    if not results:
        logger.warning("No successful evaluations.")
        sys.exit(0)
        
    logger.info("Computing metrics...")
    summary = compute_metrics(results)
    
    # Log to screen
    print("\n" + "=" * 60)
    print(" EVALUATION SUMMARY ")
    print("=" * 60)
    print(str(summary))
    print("=" * 60 + "\n")
    print(results)
    # Final dump updates the status to Completed
    full_output = {
        "summary": summary.to_dict(),
        "args": vars(args),
        "status": "Completed",
        "results": [
            {
                "claim_id": r["claim_id"],
                "pred_verdict": r["pred_verdict"],
                "gold_verdict": r["gold_verdict"],
                "correct": r["correct"],
                "report": r["report"]
            }
            for r in results
        ]
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)
        
    logger.info(f"Finished evaluating {len(results)} records. Results saved to {out_path.absolute()}")

if __name__ == "__main__":
    main()
