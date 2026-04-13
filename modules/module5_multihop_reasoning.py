"""
modules/module5_multihop_reasoning.py
--------------------------------------
Module 5 — Multi-Hop Reasoning

Executes sub-questions sequentially. Each hop's answer conditions the next
sub-question's retrieval query so reasoning evolves across hops rather than
remaining static.

Uses a small instruction-tuned LLM (Qwen2.5-3B or Phi-3-mini) constrained
to short, JSON-structured outputs per hop.

answer_unknown=True is the escape hatch — after max_retries failed parses
the hop is marked unknown and downstream hops that depend on it are skipped,
preventing hallucinated chain-of-reasoning.

Input  : ClaimDecomposition, list[EvidenceRef], VideoSegment,
         GenerativeLLM, DenseRetriever
Output : list[HopResult]
"""

from __future__ import annotations

import json
import logging
import re

from schemas import (
    ClaimDecomposition,
    EvidenceRef,
    HopResult,
    SubQuestion,
    VideoSegment,
)
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


def _format_passages(passages: list[EvidenceRef]) -> str:
    if not passages:
        return "No passages available."
    lines = []
    for p in passages:
        lines.append(f"[{p.evidence_id}] ({p.source_url}, {p.source_date}): {p.passage_text}")
    return "\n".join(lines)


def _build_hop_prompt(
    sq: SubQuestion,
    prior_answers: dict[int, str],
    passages: list[EvidenceRef],
) -> list[dict[str, str]]:
    prior_text = ""
    if sq.depends_on_hops and prior_answers:
        lines = [
            f"Previous answer (hop {hop}): \"{ans}\""
            for hop, ans in prior_answers.items()
            if hop in sq.depends_on_hops
        ]
        prior_text = "\n".join(lines) + "\n"

    user_content = (
        f'Sub-question (hop {sq.hop}): "{sq.question}"\n'
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
    sq: SubQuestion,
    prior_answers: dict[int, str],
    passages: list[EvidenceRef],
    llm: GenerativeLLM,
    claim_id: str = "",
    max_retries: int = 2,
) -> HopResult:
    """
    Execute one hop with up to max_retries parse retries.
    Returns a HopResult with answer_unknown=True on total failure.
    """
    prompt = _build_hop_prompt(sq, prior_answers, passages)

    for attempt in range(max_retries + 1):
        try:
            raw = llm.generate(prompt, max_new_tokens=256)
            data = _safe_json_parse(raw)

            result = HopResult(
                claim_id=claim_id,
                hop=data.get("hop", sq.hop),
                question=data.get("question", sq.question),
                answer=data.get("answer", ""),
                confidence=float(data.get("confidence", 0.0)),
                answer_unknown=bool(data.get("answer_unknown", False)),
                supported_by=data.get("supported_by", []),
            )

            # Accept on first non-unknown result, or on final attempt
            if not result.answer_unknown or attempt == max_retries:
                return result

            logger.debug("Hop %d attempt %d: answer_unknown=True, retrying.", sq.hop, attempt + 1)

        except Exception as exc:
            logger.warning("Hop %d attempt %d parse error: %s", sq.hop, attempt + 1, exc)

    # Complete failure
    logger.error("Hop %d failed after %d attempts — marking answer_unknown.", sq.hop, max_retries + 1)
    return HopResult(
        claim_id=claim_id,
        hop=sq.hop,
        question=sq.question,
        answer="",
        confidence=0.0,
        answer_unknown=True,
        supported_by=[],
    )


# ---------------------------------------------------------------------------
# Multi-hop orchestration
# ---------------------------------------------------------------------------

def run_multihop(
    claim: ClaimDecomposition,
    evidence: list[EvidenceRef],
    segment: VideoSegment,
    llm: GenerativeLLM,
    retriever: DenseRetriever,
) -> list[HopResult]:
    """
    Execute all sub-questions in dependency order.

    If a hop has no evidence assigned yet (hop_ids not matching) and its
    evidence_type is not "video", a targeted retrieval query is issued before
    running the hop.

    Execution stops early if a hop returns answer_unknown=True and subsequent
    hops depend on it.

    Parameters
    ----------
    claim     : Decomposed claim from Module 1.
    evidence  : Evidence pool (post Module 4).
    segment   : Source video segment.
    llm       : Per-hop generative LLM (Qwen2.5-3B or Phi-3-mini).
    retriever : DenseRetriever for on-the-fly hop-level retrieval.

    Returns
    -------
    list[HopResult]  — one per executed hop (may be shorter than sub_questions
                       if an unknown hop terminates the chain early).
    """
    hop_results: list[HopResult] = []
    unknown_hops: set[int] = set()

    for sq in claim.sub_questions:
        # Skip if a dependency hop is unknown
        if any(dep in unknown_hops for dep in sq.depends_on_hops):
            logger.warning(
                "Skipping hop %d — depends on unknown hop(s) %s.",
                sq.hop, sq.depends_on_hops,
            )
            hop_results.append(
                HopResult(
                    claim_id=claim.claim_id,
                    hop=sq.hop,
                    question=sq.question,
                    answer="",
                    confidence=0.0,
                    answer_unknown=True,
                    supported_by=[],
                )
            )
            unknown_hops.add(sq.hop)
            continue

        # Gather evidence for this hop
        relevant = [e for e in evidence if sq.hop in e.hop_ids]

        # If no evidence is pre-assigned and hop is not video-only, retrieve on the fly
        if not relevant and sq.evidence_type != "video":
            query = build_retrieval_query(sq.question, hop_results, segment)
            retrieved = retriever.search(query, top_k=3)
            for p in retrieved:
                if sq.hop not in p.hop_ids:
                    p.hop_ids.append(sq.hop)
            existing_ids = {e.evidence_id for e in evidence}
            for p in retrieved:
                if p.evidence_id not in existing_ids:
                    evidence.append(p)
            relevant = retrieved

        prior = {h.hop: h.answer for h in hop_results if h.hop in sq.depends_on_hops}

        result = run_single_hop(sq, prior, relevant, llm, claim_id=claim.claim_id)
        hop_results.append(result)

        if result.answer_unknown:
            unknown_hops.add(sq.hop)

    return hop_results
