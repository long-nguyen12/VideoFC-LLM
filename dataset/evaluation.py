"""
dataset/evaluation.py
----------------------
Evaluation utilities for batch dataset runs.

Computes per-class and macro metrics from a list of evaluation dicts:
  accuracy, precision, recall, F1 (macro + per-class)

All metrics are computed without external dependencies (no scikit-learn
required) so the module runs in the same lightweight environment as the rest
of the codebase. Import scikit-learn versions as optional drop-ins if needed.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from dataset.true_dataset_loader import LABEL_TO_VERDICT, NUM_LABELS, VERDICT_DISPLAY

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-class and aggregate metric containers
# ---------------------------------------------------------------------------

@dataclass
class ClassMetrics:
    label: int
    verdict: str
    display_name: str
    support: int           # number of gold instances of this class
    tp: int = 0            # true positives
    fp: int = 0            # false positives
    fn: int = 0            # false negatives

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


@dataclass
class EvaluationSummary:
    num_records: int
    num_correct: int
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class: list[ClassMetrics] = field(default_factory=list)

    # Distribution of gold vs predicted verdicts
    gold_distribution: dict[str, int] = field(default_factory=dict)
    pred_distribution: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_records": self.num_records,
            "num_correct": self.num_correct,
            "accuracy": round(self.accuracy, 4),
            "macro_precision": round(self.macro_precision, 4),
            "macro_recall": round(self.macro_recall, 4),
            "macro_f1": round(self.macro_f1, 4),
            "gold_distribution": self.gold_distribution,
            "pred_distribution": self.pred_distribution,
            "per_class": [
                {
                    "label": c.label,
                    "verdict": c.verdict,
                    "display_name": c.display_name,
                    "support": c.support,
                    "precision": round(c.precision, 4),
                    "recall": round(c.recall, 4),
                    "f1": round(c.f1, 4),
                }
                for c in sorted(self.per_class, key=lambda x: x.label)
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def __str__(self) -> str:
        lines = [
            f"Records : {self.num_records}",
            f"Correct : {self.num_correct}",
            f"Accuracy: {self.accuracy:.4f}",
            f"Macro P : {self.macro_precision:.4f}",
            f"Macro R : {self.macro_recall:.4f}",
            f"Macro F1: {self.macro_f1:.4f}",
            "",
            f"{'Class':<28} {'Supp':>6} {'P':>8} {'R':>8} {'F1':>8}",
            "-" * 62,
        ]
        for c in sorted(self.per_class, key=lambda x: x.label):
            lines.append(
                f"{c.display_name:<28} {c.support:>6} "
                f"{c.precision:>8.4f} {c.recall:>8.4f} {c.f1:>8.4f}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_metrics(results: list[dict]) -> EvaluationSummary:
    """
    Compute classification metrics from a list of evaluation dicts.

    Parameters
    ----------
    results : list[dict]

    Returns
    -------
    EvaluationSummary
    """
    if not results:
        raise ValueError("Cannot compute metrics on empty results list.")

    # Initialise per-class counters
    classes: dict[int, ClassMetrics] = {
        label: ClassMetrics(
            label=label,
            verdict=LABEL_TO_VERDICT[label],
            display_name=VERDICT_DISPLAY.get(LABEL_TO_VERDICT[label], LABEL_TO_VERDICT[label]),
            support=0,
        )
        for label in range(NUM_LABELS)
    }

    gold_dist: dict[str, int] = defaultdict(int)
    pred_dist: dict[str, int] = defaultdict(int)
    correct = 0

    for r in results:
        g, p = r["gold_label"], r["pred_label"]
        gold_dist[r["gold_verdict"]] += 1
        pred_dist[r["pred_verdict"]] += 1

        classes[g].support += 1

        if g == p:
            classes[g].tp += 1
            correct += 1
        else:
            classes[g].fn += 1
            classes[p].fp += 1

    n = len(results)
    accuracy = correct / n

    # Macro averages (unweighted across classes with support > 0)
    active = [c for c in classes.values() if c.support > 0]
    macro_p  = sum(c.precision for c in active) / len(active) if active else 0.0
    macro_r  = sum(c.recall    for c in active) / len(active) if active else 0.0
    macro_f1 = sum(c.f1        for c in active) / len(active) if active else 0.0

    return EvaluationSummary(
        num_records=n,
        num_correct=correct,
        accuracy=accuracy,
        macro_precision=macro_p,
        macro_recall=macro_r,
        macro_f1=macro_f1,
        per_class=list(classes.values()),
        gold_distribution=dict(gold_dist),
        pred_distribution=dict(pred_dist),
    )


def log_summary(summary: EvaluationSummary, prefix: str = "") -> None:
    tag = f"[{prefix}] " if prefix else ""
    logger.info("%sAccuracy=%.4f  Macro-F1=%.4f  (%d records)",
                tag, summary.accuracy, summary.macro_f1, summary.num_records)
    for c in sorted(summary.per_class, key=lambda x: x.label):
        if c.support > 0:
            logger.info(
                "%s  %-28s support=%d  P=%.3f  R=%.3f  F1=%.3f",
                tag, c.display_name, c.support,
                c.precision, c.recall, c.f1,
            )
