"""
dataset/label_mapper.py
------------------------
Maps Snopes-style rating strings to the pipeline's four-class verdict space
and to integer label indices for training / evaluation.

Snopes rating taxonomy (observed in the wild)
---------------------------------------------
Positive / True family
  "True"
  "Mostly True"

Contextual / nuanced family
  "Mixture"
  "Outdated"
  "Unproven"
  "Correct Attribution"
  "Miscaptioned"

Negative / False family
  "False"
  "Mostly False"
  "Labeled Satire"

Ambiguous / unknown
  "Unrated"
  "Research In Progress"

Pipeline verdict space
----------------------
  "supported"              ← claim is corroborated by evidence
  "refuted"                ← claim is clearly contradicted by evidence
  "misleading_context"     ← claim is partially true but decontextualised
  "insufficient_evidence"  ← cannot determine; evidence gate failed or label unknown

Label index mapping (for classification head training)
  0 → supported
  1 → refuted
  2 → misleading_context
  3 → insufficient_evidence
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Mapping tables
# ---------------------------------------------------------------------------

# Maps normalised (lower-cased, stripped) Snopes rating → pipeline verdict
_RATING_TO_VERDICT: dict[str, str] = {
    # Clearly true
    "true":                    "supported",
    "mostly true":             "supported",
    "correct attribution":     "supported",

    # Clearly false
    "false":                   "refuted",
    "mostly false":            "refuted",
    "labeled satire":          "refuted",

    # Nuanced / misleading-context cases
    "mixture":                 "misleading_context",
    "outdated":                "misleading_context",
    "miscaptioned":            "misleading_context",

    # Ambiguous / cannot determine
    "unproven":                "insufficient_evidence",
    "unrated":                 "insufficient_evidence",
    "research in progress":    "insufficient_evidence",
}

# Integer label index used for classification
VERDICT_TO_LABEL: dict[str, int] = {
    "supported":             0,
    "refuted":               1,
    "misleading_context":    2,
    "insufficient_evidence": 3,
}

LABEL_TO_VERDICT: dict[int, str] = {v: k for k, v in VERDICT_TO_LABEL.items()}

NUM_LABELS: int = len(VERDICT_TO_LABEL)

# Human-readable display labels (used in reports / UI)
VERDICT_DISPLAY: dict[str, str] = {
    "supported":             "Supported",
    "refuted":               "Refuted",
    "misleading_context":    "Misleading Context",
    "insufficient_evidence": "Insufficient Evidence",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rating_to_verdict(rating: str) -> str:
    """
    Convert a raw Snopes rating string to a pipeline verdict label.

    Normalisation: lower-case + strip whitespace before lookup.
    Unknown ratings fall back to "insufficient_evidence".

    Parameters
    ----------
    rating : Raw rating string from the dataset, e.g. "Mostly False".

    Returns
    -------
    str  One of the four pipeline verdict labels.
    """
    return _RATING_TO_VERDICT.get(rating.lower().strip(), "insufficient_evidence")


def verdict_to_label(verdict: str) -> int:
    """Return the integer class index for a pipeline verdict string."""
    return VERDICT_TO_LABEL.get(verdict, VERDICT_TO_LABEL["insufficient_evidence"])


def rating_to_label(rating: str) -> int:
    """Convenience: raw Snopes rating → integer class index."""
    return verdict_to_label(rating_to_verdict(rating))


def label_to_verdict(label: int) -> str:
    """Return the pipeline verdict string for an integer label index."""
    return LABEL_TO_VERDICT.get(label, "insufficient_evidence")


def all_ratings() -> list[str]:
    """Return all known Snopes rating strings."""
    return list(_RATING_TO_VERDICT.keys())


def verdict_confidence_floor(verdict: str) -> float:
    """
    Return a minimum confidence floor appropriate for each verdict class.
    Used when the pipeline must emit a verdict without LLM aggregation
    (e.g. when gold labels are available for training signal).
    """
    floors = {
        "supported":             0.70,
        "refuted":               0.70,
        "misleading_context":    0.55,   # harder to be confident about
        "insufficient_evidence": 0.0,    # no floor — always uncertain
    }
    return floors.get(verdict, 0.0)
