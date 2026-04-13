"""
modules/utils.py
----------------
Shared utilities used across all pipeline modules.

Do NOT import pipeline-internal modules here — this file must remain
dependency-free (only stdlib) so any module can safely import from it.
"""

from __future__ import annotations

import json
import re
from typing import Any


def safe_json_parse(raw: str, context: str = "") -> dict[str, Any]:
    """
    Robustly extract a JSON object from a model's raw text output.

    Handles:
      - Markdown code fences  (```json ... ```)
      - Leading/trailing prose before or after the JSON object
      - Whitespace padding

    Parameters
    ----------
    raw     : Raw string returned by the LLM.
    context : Optional label for the error message (e.g. "aggregator").

    Returns
    -------
    dict

    Raises
    ------
    ValueError  if no JSON object boundary can be found.
    json.JSONDecodeError  if the extracted substring is not valid JSON.
    """
    # Strip markdown code fences
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())

    # Find outermost { ... }
    start = cleaned.find("{")
    end   = cleaned.rfind("}") + 1
    if start == -1 or end == 0:
        loc = f" in {context}" if context else ""
        raise ValueError(f"No JSON object found{loc}: {raw[:300]!r}")

    return json.loads(cleaned[start:end])
