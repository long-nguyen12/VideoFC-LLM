from __future__ import annotations

import json
import re
from typing import Any


def safe_json_parse(text: str) -> dict[str, Any]:
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end != 0:
            candidate = text[start:end]
            return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Strategy 3: Remove markdown code fences if present
    try:
        cleaned = re.sub(
            r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE
        ).strip()
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    return None
