from __future__ import annotations

import hashlib
import logging
import re
import urllib.parse
import urllib.request

from models import GenerativeLLM
from modules.module3_evidence_strength import score_evidence
from modules.prompt_template import _RETRIEVE_SCORE_SYSTEM_PROMPT, _EVIDENCE_SCORE_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

MAX_RETRIEVAL_ROUNDS: int = 3
_WEB_TOP_K: int = 3
_INTERNAL_TOP_K: int = 3
_RELEVANCE_THRESHOLD: float = 0.40


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _passage_id(text: str, prefix: str = "web") -> str:
    digest = hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()[:10]
    return f"{prefix}-{digest}"


# ---------------------------------------------------------------------------
# LLM scoring helpers
# ---------------------------------------------------------------------------


def _llm_relevance_score(
    llm: GenerativeLLM,
    query: str,
    passage_text: str,
    max_retries: int = 1,
) -> float:
    prompt = [
        {"role": "system", "content": _RETRIEVE_SCORE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f'Query: "{query}"\nPassage: "{passage_text}"\n',
        },
    ]
    data = llm.generate_json(prompt, max_new_tokens=96, max_retries=max_retries)
    if not isinstance(data, dict):
        return 0.0
    return _clamp_01(data.get("score"))


def _llm_evidence_score(
    llm: GenerativeLLM,
    question: str,
    passage_text: str,
    max_retries: int = 1,
) -> float:
    prompt = [
        {"role": "system", "content": _EVIDENCE_SCORE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f'Question: "{question}"\nPassage: "{passage_text}"\n',
        },
    ]
    data = llm.generate_json(prompt, max_new_tokens=96, max_retries=max_retries)
    if not isinstance(data, dict):
        return 0.0
    return _clamp_01(data.get("score"))


# ---------------------------------------------------------------------------
# Internal evidence filtering
# ---------------------------------------------------------------------------


def filter_internal_evidence(
    weak_aspects: list[str],
    all_evidence: list[dict],
    claim_sub_questions: list[dict],
    llm: GenerativeLLM,
    top_k: int = _INTERNAL_TOP_K,
) -> list[dict]:
    """
    Re-rank and filter the existing internal evidence pool using LLM scoring.
    Returns passages that score above threshold for at least one weak aspect,
    with hop_ids updated to reflect which aspects they cover.
    """
    aspect_to_hop: dict[str, int] = {
        sq["question"]: sq["hop"] for sq in claim_sub_questions
    }

    scored: list[tuple[dict, int, float]] = []
    for aspect in weak_aspects:
        hop_id = aspect_to_hop.get(aspect)

        candidates = sorted(
            all_evidence,
            key=lambda e: _token_overlap_score(aspect, e.get("passage_text", "")),
            reverse=True,
        )[:max(top_k, 10)]

        for passage in candidates:
            score = _llm_evidence_score(llm, aspect, passage.get("passage_text", ""))
            if score >= _RELEVANCE_THRESHOLD:
                scored.append((passage, hop_id, score))

    seen: set[str] = set()
    results: list[dict] = []
    for passage, hop_id, score in sorted(scored, key=lambda x: x[2], reverse=True):
        eid = passage["evidence_id"]
        if eid not in seen:
            seen.add(eid)
            entry = dict(passage)
            entry["retrieval_score"] = float(score)
            if hop_id is not None and hop_id not in entry.get("hop_ids", []):
                entry = dict(entry)
                entry["hop_ids"] = list(entry.get("hop_ids", [])) + [hop_id]
            results.append(entry)

    logger.info(
        "Internal filter: %d passages scored above threshold from pool of %d.",
        len(results),
        len(all_evidence),
    )
    return results[:top_k * len(weak_aspects)]


# ---------------------------------------------------------------------------
# External web retrieval
# ---------------------------------------------------------------------------


def _ddg_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Lightweight DuckDuckGo HTML scrape — no API key required.
    Returns list of dicts with keys: title, snippet, url.
    """
    encoded = urllib.parse.quote_plus(query)
    url = f"https://html.duckduckgo.com/html/?q={encoded}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        logger.warning("DDG search failed for query %r: %s", query, exc)
        return []

    results: list[dict] = []
    snippet_pattern = re.compile(
        r'class="result__snippet"[^>]*>(.*?)</a>', re.DOTALL
    )
    title_pattern = re.compile(
        r'class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>', re.DOTALL
    )
    url_pattern = re.compile(
        r'class="result__url"[^>]*>(.*?)</span>', re.DOTALL
    )

    titles = title_pattern.findall(html)
    snippets = snippet_pattern.findall(html)
    urls_raw = url_pattern.findall(html)

    for i in range(min(max_results, len(titles), len(snippets))):
        href, title_html = titles[i]
        snippet_html = snippets[i]
        raw_url = urls_raw[i] if i < len(urls_raw) else href

        title = re.sub(r"<[^>]+>", "", title_html).strip()
        snippet = re.sub(r"<[^>]+>", "", snippet_html).strip()
        source_url = raw_url.strip().lstrip("/").strip()
        if not source_url.startswith("http"):
            source_url = "https://" + source_url

        if snippet:
            results.append(
                {"title": title, "snippet": snippet, "url": source_url}
            )

    return results


def fetch_external_evidence(
    weak_aspects: list[str],
    claim_sub_questions: list[dict],
    segment: dict,
    resolved_hops: list[dict],
    llm: GenerativeLLM,
    top_k: int = _WEB_TOP_K,
) -> list[dict]:
    """
    For each weak aspect, build a search query, scrape DuckDuckGo,
    score snippets with LLM, and return top passages as evidence dicts.
    """
    aspect_to_hop: dict[str, int] = {
        sq["question"]: sq["hop"] for sq in claim_sub_questions
    }

    known_facts = " ".join(
        h["answer"] for h in resolved_hops if not h.get("answer_unknown", False)
    )

    new_passages: list[dict] = []

    for aspect in weak_aspects:
        hop_id = aspect_to_hop.get(aspect)

        query = aspect
        if known_facts:
            query += f" {known_facts}"
        transcript_snippet = segment.get("transcript", "")[:120]
        if transcript_snippet:
            query += f" {transcript_snippet}"

        logger.info("Web search query: %r (hop=%s)", query[:120], hop_id)
        search_hits = _ddg_search(query, max_results=top_k + 2)

        scored: list[tuple[dict, float]] = []
        for hit in search_hits:
            passage_text = f"{hit['title']}. {hit['snippet']}"
            score = _llm_relevance_score(llm, aspect, passage_text)
            if score >= _RELEVANCE_THRESHOLD:
                scored.append((hit, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        for hit, score in scored[:top_k]:
            passage_text = f"{hit['title']}. {hit['snippet']}"
            eid = _passage_id(passage_text, prefix="web")
            new_passages.append(
                {
                    "evidence_id": eid,
                    "source_url": hit["url"],
                    "source_date": "",
                    "passage_text": passage_text,
                    "retrieval_score": float(score),
                    "hop_ids": [hop_id] if hop_id is not None else [],
                }
            )

    logger.info(
        "External fetch: %d passages retrieved for %d weak aspects.",
        len(new_passages),
        len(weak_aspects),
    )
    return new_passages


# ---------------------------------------------------------------------------
# Evidence merging
# ---------------------------------------------------------------------------


def merge_evidence(existing: list[dict], new_passages: list[dict]) -> list[dict]:
    """
    Merge new_passages into existing, deduplicating by evidence_id.
    Updates hop_ids and keeps the highest retrieval_score for duplicates.
    """
    index: dict[str, dict] = {e["evidence_id"]: e for e in existing}

    for p in new_passages:
        eid = p["evidence_id"]
        if eid not in index:
            index[eid] = p
            existing.append(p)
        else:
            entry = index[eid]
            for hop_id in p.get("hop_ids", []):
                if hop_id not in entry.get("hop_ids", []):
                    entry.setdefault("hop_ids", []).append(hop_id)
            entry["retrieval_score"] = max(
                float(entry.get("retrieval_score", 0.0)),
                float(p.get("retrieval_score", 0.0)),
            )

    return existing


# ---------------------------------------------------------------------------
# Public API — drop-in replacement for gated_retrieval_loop
# ---------------------------------------------------------------------------


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
    query += f" Video context: {segment.get('transcript', '')}"
    return query


def gated_retrieval_loop(
    claim: dict,
    segment: dict,
    evidence: list[dict],
    modal_report: dict,
    llm: GenerativeLLM,
    max_rounds: int = MAX_RETRIEVAL_ROUNDS,
    resolved_hops: list[dict] | None = None,
) -> tuple[list[dict], dict]:
    """
    Gated retrieval loop combining internal re-ranking and external web RAG.

    Round logic:
      1. Score current evidence with LLM via module3.score_evidence.
      2. If gate passes → done.
      3. Otherwise:
         a. Re-rank internal evidence pool for weak aspects (LLM scored).
         b. Fetch new external passages via DuckDuckGo web search.
         c. Merge both into the evidence list.
      4. Repeat up to max_rounds.
    """
    resolved_hops = resolved_hops or []
    report = None

    for round_idx in range(max_rounds):
        report = score_evidence(claim, evidence, modal_report, llm)

        if report["gate_pass"] or not report["weak_aspects"]:
            logger.info(
                "Retrieval gate passed after %d round(s) for claim %s.",
                round_idx,
                claim["claim_id"],
            )
            break

        logger.info(
            "Retrieval round %d/%d — %d weak aspects for claim %s.",
            round_idx + 1,
            max_rounds,
            len(report["weak_aspects"]),
            claim["claim_id"],
        )

        internal_passages = filter_internal_evidence(
            weak_aspects=report["weak_aspects"],
            all_evidence=evidence,
            claim_sub_questions=claim["sub_questions"],
            llm=llm,
        )

        external_passages = fetch_external_evidence(
            weak_aspects=report["weak_aspects"],
            claim_sub_questions=claim["sub_questions"],
            segment=segment,
            resolved_hops=resolved_hops,
            llm=llm,
        )

        evidence = merge_evidence(evidence, internal_passages + external_passages)

    if report is None or not report["gate_pass"]:
        report = score_evidence(claim, evidence, modal_report, llm)
        if not report["gate_pass"]:
            logger.warning(
                "Retrieval loop exhausted for claim %s — gate still not passed.",
                claim["claim_id"],
            )

    return evidence, report


# ---------------------------------------------------------------------------
# Backwards-compatible retriever shim for module5
# ---------------------------------------------------------------------------


class DenseRetriever:
    """
    Thin shim kept for module5_multihop_reasoning compatibility.
    Wraps the functional internal filter so module5 can call retriever.search().
    """

    def __init__(self) -> None:
        self._passages: list[dict] = []

    def index(self, passages: list[dict]) -> None:
        self._passages = passages or []
        logger.info("DenseRetriever.index: %d passages.", len(self._passages))

    def search(
        self,
        query: str,
        llm: GenerativeLLM,
        top_k: int = _INTERNAL_TOP_K,
        candidate_pool: int = 30,
    ) -> list[dict]:
        if not self._passages:
            return []

        candidates = sorted(
            self._passages,
            key=lambda p: _token_overlap_score(query, p.get("passage_text", "")),
            reverse=True,
        )[:max(top_k, min(candidate_pool, len(self._passages)))]

        scored = [
            (p, _llm_relevance_score(llm, query, p.get("passage_text", "")))
            for p in candidates
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for p, score in scored[:top_k]:
            entry = dict(p)
            entry["retrieval_score"] = float(score)
            results.append(entry)
        return results
