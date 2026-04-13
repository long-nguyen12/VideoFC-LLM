from __future__ import annotations

import json
import re
from typing import Any


def safe_json_parse(raw: str, context: str = "") -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())

    # Find outermost { ... }
    start = cleaned.find("{")
    end   = cleaned.rfind("}") + 1
    if start == -1 or end == 0:
        loc = f" in {context}" if context else ""
        raise ValueError(f"No JSON object found{loc}: {raw[:300]!r}")

    return json.loads(cleaned[start:end])
