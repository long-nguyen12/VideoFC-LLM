"""
dataset/true_dataset_loader.py
-------------------------------
Unified PyTorch dataset pipeline for the TRUE dataset.

Supports one workflow:

  1. LLM pipeline evaluation — via load_for_pipeline() / run_pipeline_evaluation()


Usage — LLM pipeline evaluation
---------------------------------
    from dataset.true_dataset_loader import load_for_pipeline, run_pipeline_evaluation

    # Option A: load records + keyframe paths, then run pipeline yourself
    items = load_for_pipeline(path="data/TRUE_Dataset", split="test")
    for record, kf_paths in items:
        result = run_dataset_record(record, models, retriever, keyframe_paths=kf_paths)

    # Option B: run the full evaluation loop in one call
    results = run_pipeline_evaluation(
        path="data/TRUE_Dataset",
        split="test",
        models=bundle,
        retriever=retriever,
    )
"""

from __future__ import annotations

import logging
import random
import re
from glob import glob
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


logger = logging.getLogger(__name__)

DATA_PATH = "data/TRUE_Dataset"


# --- Label Dictionary ---
_RATING_TO_VERDICT: dict[str, str] = {
    # Clearly true
    "true": "true",
    "mostly true": "true",
    "correct attribution": "true",
    # Clearly false
    "false": "false",
    "mostly false": "false",
    # Nuanced / misleading-context cases
    "mixture": "false",
    "outdated": "false",
    "fake": "false",
}

VERDICT_TO_LABEL: dict[str, int] = {
    "true": 0,
    "false": 1,
}

def rating_to_verdict(rating: str) -> str:
    return _RATING_TO_VERDICT.get(rating.lower().strip(), "insufficient_evidence")

def verdict_to_label(verdict: str) -> int:
    return VERDICT_TO_LABEL.get(verdict, 1)

def rating_to_label(rating: str) -> int:
    return verdict_to_label(rating_to_verdict(rating))

LABEL_TO_VERDICT = {v: k for k, v in VERDICT_TO_LABEL.items()}
def label_to_verdict(label: int) -> str:
    return LABEL_TO_VERDICT.get(label, "false")

NUM_LABELS = len(VERDICT_TO_LABEL)
VERDICT_DISPLAY = {
    "true": "True",
    "false": "False",
}


def _yyyymmdd_to_iso(value: float) -> str:
    from datetime import datetime
    try:
        return datetime.strptime(str(int(value)), "%Y%m%d").date().isoformat()
    except Exception:
        return ""

def _assign_hop_ids(evidence_entries: list[dict], relationships: list[dict]) -> dict[int, list[int]]:
    hop_map = {e.get('evidence_index', 0): [] for e in evidence_entries}
    evidence_to_hop = {}
    next_hop = 1
    claim_rels = [r for r in relationships if r.get('left') == 'claim']
    other_rels = [r for r in relationships if r.get('left') != 'claim']
    for rel in claim_rels + other_rels:
        idx = rel.get('evidence_index', 0)
        if idx == 0 or idx not in hop_map: continue
        if idx not in evidence_to_hop:
            evidence_to_hop[idx] = next_hop
            next_hop += 1
        hop_id = evidence_to_hop[idx]
        if hop_id not in hop_map[idx]: hop_map[idx].append(hop_id)
    return hop_map

def record_to_segment(record: dict, keyframe_paths: Optional[list[str]] = None) -> dict:
    vi = record.get("video_information", {})
    return {
        "segment_id": vi.get("video_id", ""),
        "start_ts": 0.0,
        "end_ts": vi.get("video_length", 0.0),
        "transcript": vi.get("video_transcript", ""),
        "keyframes": keyframe_paths or [],
    }

def record_to_visual_caption(record: dict) -> str:
    vi = record.get("video_information", {})
    parts = [p.strip() for p in [vi.get("video_headline", ""), vi.get("video_description", "")] if p.strip()]
    return ". ".join(parts)

def record_to_evidence(record: dict) -> list[dict]:
    vi = record.get("video_information", {})
    iso_date = _yyyymmdd_to_iso(vi.get("video_date", 0))
    evidences = record.get("evidences", {}).get("entries", [])
    relationships = record.get("relationship_with_evidence", [])
    hop_map = _assign_hop_ids(evidences, relationships)
    res = []
    for entry in evidences:
        urls = entry.get("urls", [])
        source_url = urls[0] if urls else vi.get("video_url", "")
        idx = entry.get("evidence_index", 0)
        res.append({
            "evidence_id": f"{vi.get('video_id', '')}-ev{idx}",
            "source_url": source_url,
            "source_date": iso_date,
            "passage_text": entry.get("passage_text", ""),
            "retrieval_score": 1.0,
            "hop_ids": hop_map.get(idx, []),
        })
    return res

def record_to_rationale_context(record: dict) -> dict:
    orig = record.get("original_rationales", {})
    summ = record.get("summary_rationales", {})
    add = []
    for k in ["additional_rationale1", "additional_rationale2", "additional_rationale3"]:
        v = orig.get(k, "")
        if v.strip(): add.append(v)
    
    # all_reasons() mapped to a list of dicts/strings
    # we just extract reasons
    detailed = summ.get("reasons", [])
    if not detailed and "all_reasons" in summ:
        if callable(summ["all_reasons"]): detailed = summ["all_reasons"]()

    return {
        "main_rationale": orig.get("main_rationale", ""),
        "additional_rationales": add,
        "synthesized_rationale": summ.get("synthesized_rationale", ""),
        "detailed_reasons": detailed,
        "gold_verdict": rating_to_verdict(record.get("rating", "")),
        "snopes_rating": record.get("rating", ""),
        "snopes_url": record.get("url", ""),
        "article_content": record.get("content", ""),
    }

def record_to_pipeline_inputs(record: dict, keyframe_paths: Optional[list[str]] = None) -> dict:
    vi = record.get("video_information", {})
    raw_id = f"{vi.get('video_id', '')}_{record.get('claim', '')[:40]}"
    claim_id = re.sub(r"[^a-zA-Z0-9_-]", "_", raw_id)[:64]
    return {
        "claim_text": record.get("claim", ""),
        "claim_id": claim_id,
        "segment": record_to_segment(record, keyframe_paths),
        "visual_caption": record_to_visual_caption(record),
        "initial_evidence": record_to_evidence(record),
        "rationale_context": record_to_rationale_context(record),
        "gold_verdict": rating_to_verdict(record.get("rating", "")),
        "gold_label": rating_to_label(record.get("rating", "")),
    }


def _load_dir(directory: str | Path, max_records: Optional[int] = None) -> list[dict]:
    import json
    records = []
    dir_path = Path(directory)
    if not dir_path.is_dir():
        return records
    count = 0
    for fp in sorted(dir_path.glob("*.json")):
        if max_records is not None and count >= max_records:
            break
        try:
            with open(fp, encoding="utf-8") as fh:
                records.append(json.load(fh))
            count += 1
        except Exception as exc:
            pass
    return records


def split_records(
    records: list[dict], train_frac: float = 0.8, seed: int = 42
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    n_train = int(len(shuffled) * train_frac)
    return shuffled[:n_train], shuffled[n_train:]

# ---------------------------------------------------------------------------
# Label taxonomy (from true_dataset.py)
# ---------------------------------------------------------------------------

# Maps each rating string → (binary_label, fine_sub_label)
RATING_TO_FINE: dict[str, tuple[int, int]] = {
    "true": (0, 0),
    "mostly true": (0, 1),
    "correct attribution": (0, 2),
    "false": (1, 0),
    "mostly false": (1, 1),
    "mixture": (1, 2),
    "fake": (1, 3),
    "miscaptioned": (1, 4),
}

# Number of fine-grained sub-labels per coarse class
NUM_FINE_PER_COARSE: tuple[int, int] = (3, 5)  # 3 for TRUE, 5 for FALSE
TOTAL_FINE_CLASSES: int = sum(NUM_FINE_PER_COARSE)  # 8

# Flat fine-grained label: unique index across all classes
# TRUE sub-labels:  0, 1, 2
# FALSE sub-labels: 3, 4, 5, 6, 7
RATING_TO_FLAT_FINE: dict[str, int] = {
    rating: (fine if coarse == 0 else NUM_FINE_PER_COARSE[0] + fine)
    for rating, (coarse, fine) in RATING_TO_FINE.items()
}


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------


def rating_to_binary(rating_str: str) -> int:
    """Return 0 (TRUE) or 1 (FALSE) for a rating string."""
    r = rating_str.lower()
    if r in {"mostly true", "true", "correct attribution"}:
        return 0
    if r in {"false", "mostly false", "mixture", "fake", "miscaptioned"}:
        return 1
    logger.warning("Unknown rating for binary label: %r", r)
    return 0


# ---------------------------------------------------------------------------
# Text / field helpers
# ---------------------------------------------------------------------------


def clean_data(text: Optional[str]) -> str:
    """Strip HTML tags and leading/trailing whitespace."""
    if text is None or str(text) == "nan":
        return ""
    text = re.sub(r"(<p>|</p>|@)+", "", str(text))
    return text.strip()


# ---------------------------------------------------------------------------
# File-system path helpers
# ---------------------------------------------------------------------------


def resolve_video_path(claim_id: str, data_path: str = DATA_PATH) -> str:
    """
    Resolve the local path to the video file for *claim_id*.

    Checks train_val_video/ first, then test_video/.
    Falls back to the test_video path (may not exist) if not found.
    """
    if not claim_id:
        return ""
    root = Path(data_path)
    candidates = [
        root / "train_val_video" / f"{claim_id}.mp4",
        root / "test_video" / f"{claim_id}.mp4",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return str(root / "test_video" / f"{claim_id}.mp4")


def resolve_keyframe_path(claim_id: str, data_path: str = DATA_PATH) -> list[str]:
    """
    Return a list of extracted keyframe paths (*.jpeg) for *claim_id*.

    Checks train_val_output/ then test_output/.
    Returns an empty list if no frames are found.
    """
    if not claim_id:
        return []
    root = Path(data_path)
    candidates = [
        root / "train_val_output" / claim_id / claim_id,
        root / "test_output" / claim_id / claim_id,
    ]
    print(candidates)
    for path in candidates:
        print(f"Checking path: {path} (exists={path.exists()})")
        if path.exists():
            search_pattern = str(path / "*.jpeg")
            frames = glob(search_pattern)
            print(
                f"Glob pattern {search_pattern} returned {len(frames)} frames: {frames}"
            )
            if frames:
                return sorted(frames)
    return []


# ---------------------------------------------------------------------------
# Split helper
# ---------------------------------------------------------------------------


def split_records(
    records: list[dict],
    train_frac: float = 0.8,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Deterministically split a list of records into train and val subsets.

    Parameters
    ----------
    records    : Full list of dict objects.
    train_frac : Fraction of data for training (remainder → validation).
    seed       : Random seed for reproducibility.

    Returns
    -------
    (train_records, val_records)
    """
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    n_train = int(len(shuffled) * train_frac)
    train = shuffled[:n_train]
    val = shuffled[n_train:]
    logger.info("Split: train=%d val=%d", len(train), len(val))
    return train, val


# ---------------------------------------------------------------------------
# Public API: load data from disk
# ---------------------------------------------------------------------------




def get_dataset(
    path: str = DATA_PATH,
    seed: int = 42,
    train_frac: float = 0.8,
    limit_samples: Optional[int] = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Load and split train/val/test records from the TRUE dataset directory.

    Parameters
    ----------
    path          : Root path to the TRUE dataset (contains train_val/ and test/).
    seed          : Random seed for train/val split reproducibility.
    train_frac    : Fraction of train_val data used for training.
    limit_samples : If set, cap the number of records loaded per split.

    Returns
    -------
    (train_records, val_records, test_records)
    """
    root = Path(path)
    logger.info("Loading TRUE dataset from %s", root)

    train_val_records = _load_dir(root / "train_val", max_records=limit_samples)
    test_records = _load_dir(root / "test", max_records=limit_samples)

    train_records, val_records = split_records(
        train_val_records, train_frac=train_frac, seed=seed
    )

    logger.info(
        "Dataset loaded — train: %d | val: %d | test: %d",
        len(train_records),
        len(val_records),
        len(test_records),
    )
    return train_records, val_records, test_records


# ---------------------------------------------------------------------------
# Public API: pipeline bridge — load records for dataset_pipeline.py
# ---------------------------------------------------------------------------


def load_for_pipeline(
    path: str = DATA_PATH,
    split: str = "test",
    seed: int = 42,
    train_frac: float = 0.8,
    limit_samples: Optional[int] = None,
) -> list[tuple[dict, list[str]]]:
    """
    Load dicts with resolved keyframe paths, ready for the
    LLM evaluation pipeline (``dataset_pipeline.run_dataset_record``).

    This bridges true_dataset_loader's file loading and keyframe resolution
    to dataset_pipeline's expected inputs.

    Parameters
    ----------
    path          : Root path to the TRUE dataset.
    split         : Which split to return: ``"train"``, ``"val"``, or ``"test"``.
    seed          : Random seed for the train/val split.
    train_frac    : Fraction of train_val used for training.
    limit_samples : Cap on number of records loaded per split (for debugging).

    Returns
    -------
    list of (dict, keyframe_paths) tuples.
        *keyframe_paths* is a ``list[str]`` of resolved JPEG file paths
        (may be empty if no extracted keyframes exist for the record).

    Example
    -------
    >>> items = load_for_pipeline(split="test")
    >>> for record, kf_paths in items:
    ...     result = run_dataset_record(record, models, retriever,
    ...                                 keyframe_paths=kf_paths)
    """
    train_records, val_records, test_records = get_dataset(
        path=path,
        seed=seed,
        train_frac=train_frac,
        limit_samples=limit_samples,
    )

    split_map = {
        "train": train_records,
        "val": val_records,
        "test": test_records,
        "train_val": train_records + val_records,
    }
    if split not in split_map:
        raise ValueError(
            f"Unknown split {split!r}. Choose from {list(split_map.keys())}"
        )
    records = split_map[split]

    items: list[tuple[dict, list[str]]] = []
    for record in records:
        claim_id = record.get("video_information", {}).get("video_id")
        kf_paths = resolve_keyframe_path(claim_id, data_path=path)
        items.append((record, kf_paths))

    logger.info(
        "load_for_pipeline: split=%s → %d records (%d with keyframes)",
        split,
        len(items),
        sum(1 for _, kf in items if kf),
    )
    return items


def run_pipeline_evaluation(
    path: str = DATA_PATH,
    split: str = "test",
    models: "ModelBundle" = None,  # type: ignore[assignment]
    retriever: "DenseRetriever" = None,  # type: ignore[assignment]
    use_rationale_hints: bool = False,
    seed: int = 42,
    train_frac: float = 0.8,
    limit_samples: Optional[int] = None,
) -> list["DatasetEvalResult"]:
    """
    End-to-end convenience: load a split and run every record through the
    7-module LLM fact-checking pipeline, returning evaluation results.

    Combines ``load_for_pipeline()`` + ``run_dataset_record()`` in a single
    call.  Results are collected incrementally so partial progress is
    available even if a record fails.

    Parameters
    ----------
    path                : Root path to the TRUE dataset.
    split               : ``"train"``, ``"val"``, ``"test"``, or ``"train_val"``.
    models              : Loaded ``ModelBundle`` (from ``models.model_bundle``).
    retriever           : ``DenseRetriever`` pre-indexed on a passage corpus.
    use_rationale_hints : Inject gold rationale into Module 1.
                          Set ``False`` for blind evaluation.
    seed                : Random seed for train/val split.
    train_frac          : Fraction of train_val used for training.
    limit_samples       : Cap on records loaded per split (for debugging).

    Returns
    -------
    list[DatasetEvalResult]
        One result per successfully processed record.

    Example
    -------
    >>> results = run_pipeline_evaluation(
    ...     split="test", models=bundle, retriever=retriever,
    ... )
    >>> accuracy = sum(r.correct for r in results) / len(results)
    """
    # Lazy import to avoid circular dependencies at module level
    from run_pipeline import (
        run_dataset_record,
        DatasetEvalResult,
    )  # noqa: F811

    if models is None or retriever is None:
        raise ValueError("Both `models` and `retriever` must be provided.")

    items = load_for_pipeline(
        path=path,
        split=split,
        seed=seed,
        train_frac=train_frac,
        limit_samples=limit_samples,
    )

    results: list[DatasetEvalResult] = []
    total = len(items)

    for i, (record, kf_paths) in enumerate(items, 1):
        vid = record.get("video_information", {}).get("video_id")
        logger.info("Pipeline eval [%d/%d] video_id=%s", i, total, vid)
        try:
            result = run_dataset_record(
                record=record,
                models=models,
                retriever=retriever,
                use_rationale_hints=use_rationale_hints,
                keyframe_paths=kf_paths if kf_paths else None,
            )
            results.append(result)
            tag = "✓" if result.correct else "✗"
            logger.info(
                "  → Pred: %-22s Gold: %-22s %s",
                result.pred_verdict,
                result.gold_verdict,
                tag,
            )
        except Exception as exc:
            logger.error("Failed on record %s: %s", vid, exc, exc_info=True)

    logger.info(
        "Pipeline evaluation complete: %d/%d records processed, accuracy=%.2f%%",
        len(results),
        total,
        100.0 * sum(r.correct for r in results) / max(len(results), 1),
    )
    return results
