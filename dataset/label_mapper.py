from __future__ import annotations

# ---------------------------------------------------------------------------
# Mapping tables
# ---------------------------------------------------------------------------

# Maps normalised (lower-cased, stripped) Snopes rating → pipeline verdict
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

# Integer label index used for classification
VERDICT_TO_LABEL: dict[str, int] = {
    "true": 0,
    "false": 1,
}

LABEL_TO_VERDICT: dict[int, str] = {v: k for k, v in VERDICT_TO_LABEL.items()}

NUM_LABELS: int = len(VERDICT_TO_LABEL)

# Human-readable display labels (used in reports / UI)
VERDICT_DISPLAY: dict[str, str] = {
    "true": "True",
    "false": "False",
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
    return _RATING_TO_VERDICT.get(rating.lower().strip())


def verdict_to_label(verdict: str) -> int:
    """Return the integer class index for a pipeline verdict string."""
    return VERDICT_TO_LABEL.get(verdict)


def rating_to_label(rating: str) -> int:
    """Convenience: raw Snopes rating → integer class index."""
    return verdict_to_label(rating_to_verdict(rating))


def label_to_verdict(label: int) -> str:
    """Return the pipeline verdict string for an integer label index."""
    return LABEL_TO_VERDICT.get(label)


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
        "true": 0.70,
        "false": 0.70,
    }
    return floors.get(verdict, 0.0)
