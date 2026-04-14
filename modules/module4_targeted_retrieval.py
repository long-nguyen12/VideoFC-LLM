from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch

from models import NLIScorer, TextEncoder
from modules.module3_evidence_strength import score_evidence

logger = logging.getLogger(__name__)

MAX_RETRIEVAL_ROUNDS: int = 3


# ---------------------------------------------------------------------------
# Dense Retriever
# ---------------------------------------------------------------------------

@dataclass
class DenseRetriever:
    encoder: TextEncoder
    _passages: list[dict] = field(default_factory=list, repr=False)
    _embeddings: torch.Tensor | None = field(default=None, repr=False)

    def index(self, passages: list[dict]) -> None:
        """Encode and store a corpus of passages."""
        self._passages = passages
        if not passages:
            self._embeddings = None
            logger.debug("DenseRetriever: index cleared (0 passages).")
            return
        texts = [p["passage_text"] for p in passages]
        self._embeddings = self.encoder.encode(texts)   # (N, D)
        logger.info("DenseRetriever: indexed %d passages.", len(passages))

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        if self._embeddings is None or len(self._passages) == 0:
            logger.debug("DenseRetriever.search called on empty index.")
            return []

        q_emb = self.encoder.encode([query])                       # (1, D)
        scores = (q_emb @ self._embeddings.T).squeeze(0)           # (N,)
        k = min(top_k, len(self._passages))
        top_indices = torch.topk(scores, k).indices.tolist()

        results: list[dict] = []
        for idx in top_indices:
            src = self._passages[idx]
            results.append(
                {
                    "evidence_id": src["evidence_id"],
                    "source_url": src["source_url"],
                    "source_date": src["source_date"],
                    "passage_text": src["passage_text"],
                    "retrieval_score": float(scores[idx].item()),
                    # Preserve existing hop_ids and allow caller to append a matched hop.
                    "hop_ids": list(src.get("hop_ids", [])),
                }
            )
        return results


# ---------------------------------------------------------------------------
# Query builder
# ---------------------------------------------------------------------------

def build_retrieval_query(
    weak_aspect: str,
    resolved_hops: list[dict],
    segment: dict,
) -> str:
    """
    Construct a retrieval query by combining the weak aspect with context
    from already-resolved hops and the segment transcript.
    """
    known_facts = " ".join(
        h["answer"] for h in resolved_hops if not h["answer_unknown"]
    )
    query = weak_aspect
    if known_facts:
        query += f" Context: {known_facts}"
    query += f" Video context: {segment['transcript'][:200]}"
    return query


# ---------------------------------------------------------------------------
# Gated retrieval loop
# ---------------------------------------------------------------------------

def gated_retrieval_loop(
    claim: dict,
    segment: dict,
    evidence: list[dict],
    modal_report: dict,
    retriever: DenseRetriever,
    nli: NLIScorer,
    max_rounds: int = MAX_RETRIEVAL_ROUNDS,
    resolved_hops: list[dict] | None = None,
) -> tuple[list[dict], dict]:
    resolved_hops = resolved_hops or []
    report = None

    for round_idx in range(max_rounds):
        report = score_evidence(claim, evidence, modal_report, nli)

        if report["gate_pass"] or not report["weak_aspects"]:
            logger.info(
                "Retrieval loop: gate passed after %d round(s) for claim %s.",
                round_idx, claim["claim_id"],
            )
            break

        logger.info(
            "Retrieval round %d/%d — %d weak aspects for claim %s.",
            round_idx + 1, max_rounds, len(report["weak_aspects"]), claim["claim_id"],
        )

        for aspect in report["weak_aspects"]:
            query = build_retrieval_query(aspect, resolved_hops, segment)
            new_passages = retriever.search(query, top_k=3)

            # Assign hop_ids based on which sub-question matched the aspect
            matched_hop = next(
                (sq["hop"] for sq in claim["sub_questions"] if sq["question"] == aspect), None
            )
            for p in new_passages:
                if matched_hop is not None and matched_hop not in p["hop_ids"]:
                    p["hop_ids"].append(matched_hop)

            # Deduplicate by evidence_id, but merge hop_ids back into existing rows.
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

    # Final score in case we exhausted rounds without passing
    if report is None or not report["gate_pass"]:
        report = score_evidence(claim, evidence, modal_report, nli)
        if not report["gate_pass"]:
            logger.warning(
                "Retrieval loop exhausted for claim %s — gate still not passed.",
                claim["claim_id"],
            )

    return evidence, report
