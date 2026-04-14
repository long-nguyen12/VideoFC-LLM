from __future__ import annotations

import json
import logging
import re

from models import GenerativeLLM
from modules.module4_targeted_retrieval import DenseRetriever, build_retrieval_query
from modules.utils import safe_json_parse as _safe_json_parse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

_HOP_SYSTEM_PROMPT = """\
You are a single-hop evidence reader for a video fact-checking system.
You receive one atomic sub-question, optionally an answer from a previous hop,
and a set of retrieved evidence passages.
Produce a concise intermediate answer (≤ 2 sentences) with citations.

OUTPUT FORMAT RULES (MANDATORY):
1. Respond ONLY with a valid JSON object. Do not include markdown, code blocks, explanations, greetings, or any text outside the JSON.
2. Use double quotes for ALL keys and string values. Single quotes are invalid JSON and will cause parsing errors.
3. Ensure proper JSON escaping for special characters (e.g., \\", \\\\, \\n).
4. Do not include trailing commas, comments, or schema annotations in the output.
5. The "answer" field must contain at most 2 sentences. Keep it concise and factual.
6. The "confidence" field must be a float between 0.0 and 1.0, inclusive.
7. The "supported_by" field must be a list of evidence_id strings from the provided passages.
8. Set "answer_unknown" to true ONLY if the passages genuinely do not contain sufficient information to answer the question. When true, "answer" should briefly state what is missing.

REQUIRED JSON SCHEMA:
{{
  "hop": <integer>,
  "question": "<string>",
  "answer": "<string, ≤ 2 sentences>",
  "confidence": <float 0.0–1.0>,
  "supported_by": ["<evidence_id>", ...],
  "answer_unknown": <true | false>
}}
"""


def _format_passages(passages: list[dict]) -> str:
    if not passages:
        return "No passages available."
    lines = []
    for p in passages:
        lines.append(f"[{p['evidence_id']}] ({p['source_url']}, {p['source_date']}): {p['passage_text']}")
    return "\n".join(lines)


def _build_hop_prompt(
    sq: dict,
    prior_answers: dict[int, str],
    passages: list[dict],
) -> list[dict[str, str]]:
    prior_text = ""
    if sq.get("depends_on_hops") and prior_answers:
        lines = [
            f"Previous answer (hop {hop}): \"{ans}\""
            for hop, ans in prior_answers.items()
            if hop in sq["depends_on_hops"]
        ]
        prior_text = "\n".join(lines) + "\n"

    user_content = (
        f'Sub-question (hop {sq["hop"]}): "{sq["question"]}"\n'
        f"{prior_text}"
        f"Retrieved passages:\n{_format_passages(passages)}\n\n"
        "Answer concisely. Output ONLY valid JSON. Start your response with { and end with }."
    )
    return [
        {"role": "system", "content": _HOP_SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]


# JSON parsing — shared implementation lives in modules/utils.py


# ---------------------------------------------------------------------------
# Single-hop execution
# ---------------------------------------------------------------------------

def run_single_hop(
    sq: dict,
    prior_answers: dict[int, str],
    passages: list[dict],
    llm: GenerativeLLM,
    claim_id: str = "",
    max_retries: int = 2,
) -> dict:
    prompt = _build_hop_prompt(sq, prior_answers, passages)

    for attempt in range(max_retries + 1):
        try:
            raw = llm.generate(prompt, max_new_tokens=256)
            data = _safe_json_parse(raw)

            result = {
                "claim_id": claim_id,
                "hop": data.get("hop", sq["hop"]),
                "question": data.get("question", sq["question"]),
                "answer": data.get("answer", ""),
                "confidence": float(data.get("confidence", 0.0)),
                "answer_unknown": bool(data.get("answer_unknown", False)),
                "supported_by": data.get("supported_by", []),
            }

            # Accept on first non-unknown result, or on final attempt
            if not result["answer_unknown"] or attempt == max_retries:
                return result

            logger.debug("Hop %d attempt %d: answer_unknown=True, retrying.", sq["hop"], attempt + 1)

        except Exception as exc:
            logger.warning("Hop %d attempt %d parse error: %s", sq["hop"], attempt + 1, exc)

    # Complete failure
    logger.error("Hop %d failed after %d attempts — marking answer_unknown.", sq["hop"], max_retries + 1)
    return {
        "claim_id": claim_id,
        "hop": sq["hop"],
        "question": sq["question"],
        "answer": "",
        "confidence": 0.0,
        "answer_unknown": True,
        "supported_by": [],
    }


# ---------------------------------------------------------------------------
# Multi-hop orchestration
# ---------------------------------------------------------------------------

def run_multihop(
    claim: dict,
    evidence: list[dict],
    segment: dict,
    llm: GenerativeLLM,
    retriever: DenseRetriever,
) -> list[dict]:
    hop_results: list[dict] = []
    unknown_hops: set[int] = set()

    for sq in claim["sub_questions"]:
        # Skip if a dependency hop is unknown
        if any(dep in unknown_hops for dep in sq.get("depends_on_hops", [])):
            logger.warning(
                "Skipping hop %d — depends on unknown hop(s) %s.",
                sq["hop"], sq.get("depends_on_hops", []),
            )
            hop_results.append(
                {
                    "claim_id": claim["claim_id"],
                    "hop": sq["hop"],
                    "question": sq["question"],
                    "answer": "",
                    "confidence": 0.0,
                    "answer_unknown": True,
                    "supported_by": [],
                }
            )
            unknown_hops.add(sq["hop"])
            continue

        # Gather evidence for this hop
        relevant = [e for e in evidence if sq["hop"] in e["hop_ids"]]

        # If no evidence is pre-assigned and hop is not video-only, retrieve on the fly
        if not relevant and sq.get("evidence_type", "any") != "video":
            query = build_retrieval_query(sq["question"], hop_results, segment)
            retrieved = retriever.search(query, llm=llm, top_k=3)
            for p in retrieved:
                if sq["hop"] not in p["hop_ids"]:
                    p["hop_ids"].append(sq["hop"])
            existing_by_id = {e["evidence_id"]: e for e in evidence}
            for p in retrieved:
                existing = existing_by_id.get(p["evidence_id"])
                if existing is None:
                    evidence.append(p)
                    existing_by_id[p["evidence_id"]] = p
                else:
                    existing_hops = existing.setdefault("hop_ids", [])
                    for hop_id in p.get("hop_ids", []):
                        if hop_id not in existing_hops:
                            existing_hops.append(hop_id)
            relevant = retrieved

        prior = {h["hop"]: h["answer"] for h in hop_results if h["hop"] in sq.get("depends_on_hops", [])}

        result = run_single_hop(sq, prior, relevant, llm, claim_id=claim["claim_id"])
        hop_results.append(result)

        if result["answer_unknown"]:
            unknown_hops.add(sq["hop"])

    return hop_results
