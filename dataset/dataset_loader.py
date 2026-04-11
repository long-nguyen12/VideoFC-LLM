"""
dataset/dataset_loader.py
--------------------------
Loads dataset files (JSONL or JSON array) and yields DatasetRecord objects.
Also provides DirectoryLoader for per-record JSON directories (TRUE dataset).

Supported file formats
----------------------
  .jsonl / .ndjson   — one JSON object per line
  .json              — either a JSON array [...] or a single object {...}
  directory of .json  — one record per file (TRUE dataset format)

Usage
-----
    # Single file:
    from dataset.dataset_loader import DatasetLoader

    loader = DatasetLoader("data/train.jsonl")
    for record in loader:
        inputs = record_to_pipeline_inputs(record)
        report = run_pipeline(...)

    # Or load everything into memory:
    records = loader.load_all()

    # Or load a single record from a raw dict (useful for one-off tests):
    record = DatasetLoader.from_dict(raw_dict)

    # Directory of per-record JSON files (TRUE dataset):
    from dataset.dataset_loader import DirectoryLoader

    loader = DirectoryLoader("data/TRUE_Dataset/train_val")
    for record in loader:
        ...

    # Using the dataset's own split files:
    from dataset.dataset_loader import load_split

    train_records = load_split("data/TRUE_Dataset", split="train_val")
    test_records  = load_split("data/TRUE_Dataset", split="test")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Generator, Iterator

from dataset.dataset_schemas import DatasetRecord

logger = logging.getLogger(__name__)


class DatasetLoader:
    """
    Lazy iterator over a dataset file.

    Parameters
    ----------
    path        : Path to a .jsonl, .ndjson, or .json file.
    max_records : Optional cap on number of records to yield (useful for
                  debugging / smoke-tests).
    skip_errors : If True, malformed records are logged and skipped rather
                  than raising an exception.
    """

    def __init__(
        self,
        path: str | Path,
        max_records: int | None = None,
        skip_errors: bool = True,
    ) -> None:
        self.path = Path(path)
        self.max_records = max_records
        self.skip_errors = skip_errors

        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.path}")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[DatasetRecord]:
        return self._iter_records()

    def __len__(self) -> int:
        """Count records without fully loading them (single pass)."""
        return sum(1 for _ in self._iter_raw())

    def load_all(self) -> list[DatasetRecord]:
        """Load all records into memory and return as a list."""
        return list(self._iter_records())

    @staticmethod
    def from_dict(raw: dict) -> DatasetRecord:
        """Parse a single raw Python dict into a DatasetRecord."""
        return DatasetRecord.from_dict(raw)

    @staticmethod
    def from_json_string(s: str) -> DatasetRecord:
        """Parse a raw JSON string into a DatasetRecord."""
        return DatasetRecord.from_dict(json.loads(s))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _iter_raw(self) -> Generator[dict, None, None]:
        """
        Yield raw Python dicts from the file.
        Handles JSONL (one object per line) and JSON array / single object.
        """
        suffix = self.path.suffix.lower()

        with open(self.path, encoding="utf-8") as fh:
            if suffix in (".jsonl", ".ndjson"):
                for lineno, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as exc:
                        logger.warning("Skipping malformed JSON at line %d: %s", lineno, exc)
            else:
                # .json — could be an array or a single object
                try:
                    data = json.load(fh)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Cannot parse JSON file {self.path}: {exc}") from exc

                if isinstance(data, list):
                    yield from data
                elif isinstance(data, dict):
                    yield data
                else:
                    raise ValueError(
                        f"Unsupported JSON root type {type(data)} in {self.path}"
                    )

    def _iter_records(self) -> Iterator[DatasetRecord]:
        count = 0
        for raw in self._iter_raw():
            if self.max_records is not None and count >= self.max_records:
                break
            try:
                record = DatasetRecord.from_dict(raw)
                yield record
                count += 1
            except Exception as exc:
                if self.skip_errors:
                    logger.warning(
                        "Skipping record (url=%s) due to parse error: %s",
                        raw.get("url", "?"), exc,
                    )
                else:
                    raise


# ---------------------------------------------------------------------------
# Convenience: split a list of records into train/val/test
# ---------------------------------------------------------------------------

def split_records(
    records: list[DatasetRecord],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> tuple[list[DatasetRecord], list[DatasetRecord], list[DatasetRecord]]:
    """
    Deterministically split records into train / val / test sets.

    Parameters
    ----------
    records    : Full list of DatasetRecord objects.
    train_frac : Fraction of data for training.
    val_frac   : Fraction of data for validation.
    seed       : Random seed for reproducibility.

    Returns
    -------
    (train, val, test)  — three lists summing to len(records).
    """
    import random
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    train = shuffled[:n_train]
    val   = shuffled[n_train : n_train + n_val]
    test  = shuffled[n_train + n_val:]

    logger.info("Split: train=%d val=%d test=%d", len(train), len(val), len(test))
    return train, val, test


# ---------------------------------------------------------------------------
# DirectoryLoader: iterate over a directory of per-record JSON files
# ---------------------------------------------------------------------------

class DirectoryLoader:
    """
    Lazy iterator over a directory containing one .json file per record.

    This is the storage layout used by the TRUE dataset:
        data/TRUE_Dataset/train_val/10372904.json
        data/TRUE_Dataset/train_val/10085809.json
        ...

    Files are yielded in sorted order by filename for reproducibility.
    An optional ``video_ids`` filter can restrict loading to a specific
    subset (e.g. from a split file like train_val_set.txt).

    Parameters
    ----------
    dir_path    : Path to the directory containing .json files.
    video_ids   : Optional list/set of video IDs to load. When provided,
                  only files whose stem matches a video ID are loaded.
                  This is used with the split .txt files.
    max_records : Optional cap on number of records to yield.
    skip_errors : If True, malformed files are logged and skipped.
    """

    def __init__(
        self,
        dir_path: str | Path,
        video_ids: list[str] | set[str] | None = None,
        max_records: int | None = None,
        skip_errors: bool = True,
    ) -> None:
        self.dir_path = Path(dir_path)
        self.video_ids = set(video_ids) if video_ids is not None else None
        self.max_records = max_records
        self.skip_errors = skip_errors

        if not self.dir_path.is_dir():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.dir_path}"
            )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[DatasetRecord]:
        return self._iter_records()

    def __len__(self) -> int:
        """Count eligible files without fully parsing them."""
        return sum(1 for _ in self._iter_paths())

    def load_all(self) -> list[DatasetRecord]:
        """Load all records into memory and return as a list."""
        return list(self._iter_records())

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _iter_paths(self) -> Generator[Path, None, None]:
        """
        Yield .json file paths in sorted order, optionally filtered
        by video_ids.
        """
        for fp in sorted(self.dir_path.glob("*.json")):
            if self.video_ids is not None and fp.stem not in self.video_ids:
                continue
            yield fp

    def _iter_records(self) -> Iterator[DatasetRecord]:
        count = 0
        for fp in self._iter_paths():
            if self.max_records is not None and count >= self.max_records:
                break
            try:
                with open(fp, encoding="utf-8") as fh:
                    raw = json.load(fh)
                record = DatasetRecord.from_dict(raw)
                yield record
                count += 1
            except Exception as exc:
                if self.skip_errors:
                    logger.warning(
                        "Skipping file %s due to error: %s", fp.name, exc,
                    )
                else:
                    raise

        logger.info(
            "DirectoryLoader: yielded %d records from %s",
            count, self.dir_path,
        )


# ---------------------------------------------------------------------------
# load_split: load train_val or test using the dataset's own split files
# ---------------------------------------------------------------------------

def _read_split_ids(split_file: Path) -> list[str]:
    """Read video IDs from a split .txt file (one ID per line)."""
    ids: list[str] = []
    with open(split_file, encoding="utf-8") as fh:
        for line in fh:
            vid = line.strip()
            if vid:
                ids.append(vid)
    return ids


def load_split(
    dataset_root: str | Path,
    split: str = "train_val",
    max_records: int | None = None,
    skip_errors: bool = True,
) -> list[DatasetRecord]:
    """
    Load a dataset split using the TRUE dataset's file layout.

    Expected directory structure::

        dataset_root/
        ├── train_val/           # directory of per-record .json files
        ├── test/                # directory of per-record .json files
        ├── train_val_set.txt    # one video_id per line
        └── test_set.txt         # one video_id per line

    Parameters
    ----------
    dataset_root : Path to the TRUE dataset root (e.g. "data/TRUE_Dataset").
    split        : "train_val" or "test".
    max_records  : Optional cap on number of records.
    skip_errors  : If True, malformed records are logged and skipped.

    Returns
    -------
    list[DatasetRecord]
    """
    root = Path(dataset_root)
    split_file = root / f"{split}_set.txt"
    split_dir  = root / split

    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    video_ids = _read_split_ids(split_file)
    logger.info(
        "Loading split '%s': %d video IDs from %s",
        split, len(video_ids), split_file,
    )

    loader = DirectoryLoader(
        dir_path=split_dir,
        video_ids=video_ids,
        max_records=max_records,
        skip_errors=skip_errors,
    )
    return loader.load_all()
