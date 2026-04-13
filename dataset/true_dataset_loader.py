"""
dataset/true_dataset_loader.py
-------------------------------
Unified PyTorch dataset pipeline for the TRUE dataset.

Combines:
  - The lazy, schema-validated file parsing from dataset_loader.py
    (DirectoryLoader → DatasetRecord via dataset_schemas.py)
  - The correct label encoding, text cleaning, video/keyframe path
    resolution, and DataLoader construction from true_dataset.py

Supports two workflows:

  1. Classifier training (CrossTransVFC) — via create_dataloaders()
  2. LLM pipeline evaluation — via load_for_pipeline() / run_pipeline_evaluation()

Usage — Classifier training
----------------------------
    from dataset.true_dataset_loader import create_dataloaders

    train_loader, val_loader, test_loader = create_dataloaders(
        path="data/TRUE_Dataset",
        batch_size=32,
    )

    for batch in train_loader:
        claim_ids   = batch["claim_id"]       # list[str]
        claims      = batch["claim"]          # list[str]
        labels      = batch["label"]          # FloatTensor (B, 2) one-hot binary
        fine_labels = batch["fine_label"]     # LongTensor  (B,)  fine sub-label
        flat_labels = batch["flat_fine_label"]# LongTensor  (B,)  flat 0..7 index
        keyframes   = batch["keyframes"]      # list[list[str]]
        ...

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

from dataset.dataset_loader import DirectoryLoader
from dataset.dataset_schemas import DatasetRecord

logger = logging.getLogger(__name__)

DATA_PATH = "data/TRUE_Dataset"

# ---------------------------------------------------------------------------
# Label taxonomy (from true_dataset.py)
# ---------------------------------------------------------------------------

# Maps each rating string → (binary_label, fine_sub_label)
RATING_TO_FINE: dict[str, tuple[int, int]] = {
    "true":                (0, 0),
    "mostly true":         (0, 1),
    "correct attribution": (0, 2),
    "false":               (1, 0),
    "mostly false":        (1, 1),
    "mixture":             (1, 2),
    "fake":                (1, 3),
    "miscaptioned":        (1, 4),
}

# Number of fine-grained sub-labels per coarse class
NUM_FINE_PER_COARSE: tuple[int, int] = (3, 5)   # 3 for TRUE, 5 for FALSE
TOTAL_FINE_CLASSES: int = sum(NUM_FINE_PER_COARSE)  # 8

# Flat fine-grained label: unique index across all classes
# TRUE sub-labels:  0, 1, 2
# FALSE sub-labels: 3, 4, 5, 6, 7
RATING_TO_FLAT_FINE: dict[str, int] = {
    rating: (
        fine
        if coarse == 0
        else NUM_FINE_PER_COARSE[0] + fine
    )
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


def rating_to_multilabel(rating_str: str) -> tuple[int, int, int]:
    """
    Return (binary_label, fine_sub_label, flat_fine_label) for a rating string.

    Returns (0, 0, 0) and logs a warning for unknown ratings.
    """
    r = rating_str.lower()
    if r not in RATING_TO_FINE:
        logger.warning("Unknown rating for multilabel: %r", r)
        return 0, 0, 0
    binary, fine_sub = RATING_TO_FINE[r]
    flat_fine = RATING_TO_FLAT_FINE[r]
    return binary, fine_sub, flat_fine


def one_hot(index: int, num_classes: int) -> np.ndarray:
    """Convert a class index to a one-hot numpy vector."""
    v = np.zeros(num_classes, dtype=np.int64)
    v[index] = 1
    return v


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
        root / "test_video"      / f"{claim_id}.mp4",
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
        root / "train_val_output" / claim_id ,
        root / "test_output"      / claim_id ,
    ]
    print(candidates)
    for path in candidates:
        print(f"Checking path: {path} (exists={path.exists()})")
        if path.exists():
            search_pattern = str(path / "*.jpeg")
            frames = glob(search_pattern)
            print(f"Glob pattern {search_pattern} returned {len(frames)} frames: {frames}")
            if frames:
                return sorted(frames)
    return []


# ---------------------------------------------------------------------------
# Encoding: DatasetRecord → sample dict
# ---------------------------------------------------------------------------

def encode_record(record: DatasetRecord, data_path: str = DATA_PATH) -> dict:
    """
    Convert a validated DatasetRecord into a model-ready sample dict.

    Uses the schema-parsed fields from DatasetRecord (via dataset_schemas.py)
    and applies all encoding logic from true_dataset.py.
    """
    # — Claim & label —
    claim = clean_data(record.claim)
    rating_str = record.rating

    binary_idx = rating_to_binary(rating_str)
    binary_label, fine_sub_idx, flat_fine_idx = rating_to_multilabel(rating_str)

    # — Video information (from schema) —
    vi = record.video_information
    claim_id           = clean_data(vi.video_id)
    video_transcript   = clean_data(vi.video_transcript)
    video_description  = clean_data(vi.video_description)
    video_headline     = clean_data(vi.video_headline)
    video_url          = resolve_video_path(claim_id, data_path)

    # — Evidence text (from EvidencesBlock) —
    text_evidence: list[str] = [
        clean_data(e.passage_text)
        for e in record.evidences.entries
        if e.passage_text.strip()
    ]

    # — Keyframes —
    keyframes = resolve_keyframe_path(claim_id, data_path)

    return {
        "claim_id":        claim_id,
        "claim":           claim,
        "label":           torch.tensor(one_hot(binary_idx, 2), dtype=torch.float),
        "fine_label":      fine_sub_idx,
        "flat_fine_label": flat_fine_idx,
        "rating":          rating_str,
        "text_evidence":   text_evidence,
        "video_transcript":  video_transcript,
        "video_url":         video_url,
        "video_description": video_description,
        "video_headline":    video_headline,
        "image_evidence":    [],          # reserved for future use
        "url":               record.url,
        "content":           clean_data(record.content),
        "keyframes":         keyframes,
    }


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class TRUEDataset(Dataset):
    """
    PyTorch Dataset for the TRUE video fact-checking dataset.

    Parameters
    ----------
    records   : List of DatasetRecord objects (from DirectoryLoader).
    data_path : Root path for resolving video/keyframe files.
    """

    def __init__(
        self,
        records: list[DatasetRecord],
        data_path: str = DATA_PATH,
    ) -> None:
        self._encoded: list[dict] = []
        for record in records:
            try:
                self._encoded.append(encode_record(record, data_path))
            except Exception as exc:
                logger.warning(
                    "Skipping record (id=%s) due to encode error: %s",
                    getattr(getattr(record, "video_information", None), "video_id", "?"),
                    exc,
                )

    def __len__(self) -> int:
        return len(self._encoded)

    def __getitem__(self, idx: int) -> dict:
        return self._encoded[idx]

    def to_list(self) -> list[dict]:
        """Return all encoded samples as a plain list."""
        return self._encoded


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_fn(batch: list[dict]) -> dict:
    """
    Collate a list of sample dicts into a batch dict for DataLoader.

    Tensors are stacked; all other fields are kept as lists.
    """
    return {
        "claim_id":          [item["claim_id"]           for item in batch],
        "claim":             [item["claim"]              for item in batch],
        "label":             torch.stack([item["label"]  for item in batch]),
        "fine_label":        torch.tensor([item["fine_label"]       for item in batch], dtype=torch.long),
        "flat_fine_label":   torch.tensor([item["flat_fine_label"]  for item in batch], dtype=torch.long),
        "rating":            [item["rating"]             for item in batch],
        "text_evidence":     [item["text_evidence"]      for item in batch],
        "video_transcript":  [item["video_transcript"]   for item in batch],
        "video_url":         [item["video_url"]          for item in batch],
        "video_description": [item["video_description"]  for item in batch],
        "video_headline":    [item["video_headline"]     for item in batch],
        "image_evidence":    [item["image_evidence"]     for item in batch],
        "url":               [item["url"]                for item in batch],
        "content":           [item["content"]            for item in batch],
        "keyframes":         [item["keyframes"]          for item in batch],
    }


# ---------------------------------------------------------------------------
# Split helper
# ---------------------------------------------------------------------------

def split_records(
    records: list[DatasetRecord],
    train_frac: float = 0.8,
    seed: int = 42,
) -> tuple[list[DatasetRecord], list[DatasetRecord]]:
    """
    Deterministically split a list of records into train and val subsets.

    Parameters
    ----------
    records    : Full list of DatasetRecord objects.
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
    val   = shuffled[n_train:]
    logger.info("Split: train=%d val=%d", len(train), len(val))
    return train, val


# ---------------------------------------------------------------------------
# Public API: load data from disk
# ---------------------------------------------------------------------------

def _load_dir(directory: str | Path, max_records: Optional[int] = None) -> list[DatasetRecord]:
    """Load all records from a directory using DirectoryLoader."""
    loader = DirectoryLoader(dir_path=directory, max_records=max_records, skip_errors=True)
    return loader.load_all()


def get_dataset(
    path: str = DATA_PATH,
    seed: int = 42,
    train_frac: float = 0.8,
    limit_samples: Optional[int] = None,
) -> tuple[list[DatasetRecord], list[DatasetRecord], list[DatasetRecord]]:
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
    test_records      = _load_dir(root / "test",      max_records=limit_samples)

    train_records, val_records = split_records(train_val_records, train_frac=train_frac, seed=seed)

    logger.info(
        "Dataset loaded — train: %d | val: %d | test: %d",
        len(train_records), len(val_records), len(test_records),
    )
    return train_records, val_records, test_records


# ---------------------------------------------------------------------------
# Public API: create DataLoaders
# ---------------------------------------------------------------------------

def create_dataloaders(
    path: str = DATA_PATH,
    batch_size: int = 32,
    shuffle_train: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 42,
    train_frac: float = 0.8,
    limit_samples: Optional[int] = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders for the TRUE dataset.

    Parameters
    ----------
    path          : Root path to the TRUE dataset.
    batch_size    : Batch size for all loaders.
    shuffle_train : Whether to shuffle the training loader.
    num_workers   : Number of DataLoader worker processes.
    pin_memory    : Whether to pin memory (useful for GPU training).
    seed          : Random seed for train/val split.
    train_frac    : Fraction of train_val used for training.
    limit_samples : Cap on number of records loaded per split (for debugging).

    Returns
    -------
    (train_loader, val_loader, test_loader)
    """
    train_records, val_records, test_records = get_dataset(
        path=path,
        seed=seed,
        train_frac=train_frac,
        limit_samples=limit_samples,
    )

    train_dataset = TRUEDataset(train_records, data_path=path)
    val_dataset   = TRUEDataset(val_records,   data_path=path)
    test_dataset  = TRUEDataset(test_records,  data_path=path)

    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, **loader_kwargs)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,         **loader_kwargs)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,         **loader_kwargs)

    logger.info(
        "DataLoaders created — train: %d batches | val: %d batches | test: %d batches",
        len(train_loader), len(val_loader), len(test_loader),
    )
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Public API: pipeline bridge — load records for dataset_pipeline.py
# ---------------------------------------------------------------------------

def load_for_pipeline(
    path: str = DATA_PATH,
    split: str = "test",
    seed: int = 42,
    train_frac: float = 0.8,
    limit_samples: Optional[int] = None,
) -> list[tuple[DatasetRecord, list[str]]]:
    """
    Load DatasetRecords with resolved keyframe paths, ready for the
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
    list of (DatasetRecord, keyframe_paths) tuples.
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
        path=path, seed=seed, train_frac=train_frac,
        limit_samples=limit_samples,
    )

    split_map = {
        "train":     train_records,
        "val":       val_records,
        "test":      test_records,
        "train_val": train_records + val_records,
    }
    if split not in split_map:
        raise ValueError(
            f"Unknown split {split!r}. Choose from {list(split_map.keys())}"
        )
    records = split_map[split]

    items: list[tuple[DatasetRecord, list[str]]] = []
    for record in records:
        claim_id = record.video_information.video_id
        kf_paths = resolve_keyframe_path(claim_id, data_path=path)
        items.append((record, kf_paths))

    logger.info(
        "load_for_pipeline: split=%s → %d records (%d with keyframes)",
        split, len(items), sum(1 for _, kf in items if kf),
    )
    return items


def run_pipeline_evaluation(
    path: str = DATA_PATH,
    split: str = "test",
    models: "ModelBundle" = None,           # type: ignore[assignment]
    retriever: "DenseRetriever" = None,     # type: ignore[assignment]
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
    from dataset.dataset_pipeline import run_dataset_record, DatasetEvalResult  # noqa: F811

    if models is None or retriever is None:
        raise ValueError("Both `models` and `retriever` must be provided.")

    items = load_for_pipeline(
        path=path, split=split, seed=seed,
        train_frac=train_frac, limit_samples=limit_samples,
    )

    results: list[DatasetEvalResult] = []
    total = len(items)

    for i, (record, kf_paths) in enumerate(items, 1):
        vid = record.video_information.video_id
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
                result.pred_verdict, result.gold_verdict, tag,
            )
        except Exception as exc:
            logger.error("Failed on record %s: %s", vid, exc, exc_info=True)

    logger.info(
        "Pipeline evaluation complete: %d/%d records processed, accuracy=%.2f%%",
        len(results), total,
        100.0 * sum(r.correct for r in results) / max(len(results), 1),
    )
    return results

