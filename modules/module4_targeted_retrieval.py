from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from models import GenerativeLLM
from modules.module3_evidence_strength import score_evidence
from modules.utils import safe_json_parse as _safe_json_parse

logger = logging.getLogger(__name__)

MAX_RETRIEVAL_ROUNDS: int = 3

_RETRIEVE_SCORE_SYSTEM_PROMPT = """\
You are a passage ranking model for fact-checking retrieval.
Given one query and one passage, estimate relevance for answering the query.

Return ONLY valid JSON:
{
  "score": <float in [0.0, 1.0]>
}
"""


def _clamp_01(value: object) -> float:
    try:
        score = float(value)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, score))


def _token_overlap_score(query: str, text: str) -> float:
    q = set(re.findall(r"[a-z0-9]+", query.lower()))
    t = set(re.findall(r"[a-z0-9]+", text.lower()))
    if not q or not t:
        return 0.0
    return len(q & t) / max(len(q), 1)


def _llm_retrieval_score(
    llm: GenerativeLLM,
    query: str,
    passage_text: str,
    max_retries: int = 1,
) -> float:
    prompt = [
        {"role": "system", "content": _RETRIEVE_SCORE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f'Query: "{query[:500]}"\n'
                f'Passage: "{passage_text[:2000]}"\n'
                "Output JSON only."
            ),
        },
    ]
    for _ in range(max_retries + 1):
        try:
            raw = llm.generate(prompt, max_new_tokens=96)
            data = _safe_json_parse(raw)
            if not isinstance(data, dict):
                continue
            return _clamp_01(data.get("score"))
        except Exception:
            continue
    return 0.0


@dataclass
class DenseRetriever:
    _passages: list[dict] = field(default_factory=list, repr=False)

    def index(self, passages: list[dict]) -> None:
        self._passages = passages or []
        logger.info("DenseRetriever: indexed %d passages.", len(self._passages))

    def search(
        self,
        query: str,
        llm: GenerativeLLM,
        top_k: int = 3,
        candidate_pool: int = 30,
    ) -> list[dict]:
        if not self._passages:
            logger.debug("DenseRetriever.search called on empty index.")
            return []

        # Lightweight lexical prefilter before LLM scoring.
        ranked_idx = sorted(
            range(len(self._passages)),
            key=lambda i: _token_overlap_score(
                query, self._passages[i].get("passage_text", "")
            ),
            reverse=True,
        )
        candidate_idx = ranked_idx[: max(top_k, min(candidate_pool, len(ranked_idx)))]

        scored: list[tuple[int, float]] = []
        for idx in candidate_idx:
            passage_text = self._passages[idx].get("passage_text", "")
            score = _llm_retrieval_score(llm, query, passage_text)
            scored.append((idx, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        chosen = scored[: min(top_k, len(scored))]

        results: list[dict] = []
        for idx, score in chosen:
            src = self._passages[idx]
            results.append(
                {
                    "evidence_id": src["evidence_id"],
                    "source_url": src["source_url"],
                    "source_date": src["source_date"],
                    "passage_text": src.get("passage_text", ""),
                    "retrieval_score": float(score),
                    "hop_ids": list(src.get("hop_ids", [])),
                }
            )
        return results


def build_retrieval_query(
    weak_aspect: str,
    resolved_hops: list[dict],
    segment: dict,
) -> str:
    known_facts = " ".join(
        h["answer"] for h in resolved_hops if not h.get("answer_unknown", False)
    )
    query = weak_aspect
    if known_facts:
        query += f" Context: {known_facts}"
    query += f" Video context: {segment.get('transcript', '')[:200]}"
    return query


def gated_retrieval_loop(
    claim: dict,
    segment: dict,
    evidence: list[dict],
    modal_report: dict,
    retriever: DenseRetriever,
    llm: GenerativeLLM,
    max_rounds: int = MAX_RETRIEVAL_ROUNDS,
    resolved_hops: list[dict] | None = None,
) -> tuple[list[dict], dict]:
    resolved_hops = resolved_hops or []
    report = None

    for round_idx in range(max_rounds):
        report = score_evidence(claim, evidence, modal_report, llm)

        if report["gate_pass"] or not report["weak_aspects"]:
            logger.info(
                "Retrieval loop: gate passed after %d round(s) for claim %s.",
                round_idx,
                claim["claim_id"],
            )
            break

        logger.info(
            "Retrieval round %d/%d - %d weak aspects for claim %s.",
            round_idx + 1,
            max_rounds,
            len(report["weak_aspects"]),
            claim["claim_id"],
        )

        for aspect in report["weak_aspects"]:
            query = build_retrieval_query(aspect, resolved_hops, segment)
            new_passages = retriever.search(query, llm=llm, top_k=3)

            matched_hop = next(
                (
                    sq["hop"]
                    for sq in claim["sub_questions"]
                    if sq["question"] == aspect
                ),
                None,
            )
            for p in new_passages:
                if matched_hop is not None and matched_hop not in p["hop_ids"]:
                    p["hop_ids"].append(matched_hop)

            existing_by_id = {e["evidence_id"]: e for e in evidence}
            for p in new_passages:
                existing = existing_by_id.get(p["evidence_id"])
                if existing is None:
                    evidence.append(p)
                    existing_by_id[p["evidence_id"]] = p
                    continue

                existing_hops = existing.setdefault("hop_ids", [])
                for hop_id in p.get("hop_ids", []):
                    if hop_id not in existing_hops:
                        existing_hops.append(hop_id)
                existing["retrieval_score"] = max(
                    float(existing.get("retrieval_score", 0.0)),
                    float(p.get("retrieval_score", 0.0)),
                )

    if report is None or not report["gate_pass"]:
        report = score_evidence(claim, evidence, modal_report, llm)
        if not report["gate_pass"]:
            logger.warning(
                "Retrieval loop exhausted for claim %s - gate still not passed.",
                claim["claim_id"],
            )

    return evidence, report
