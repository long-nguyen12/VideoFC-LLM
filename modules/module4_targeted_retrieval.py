"""
modules/module4_targeted_retrieval.py
--------------------------------------
Module 4 — Targeted Retrieval

Triggered only when gate_pass is False. Re-retrieves evidence for weak_aspects
only, conditioned on what prior hops have already resolved. Loops up to
MAX_RETRIEVAL_ROUNDS times, re-scoring after each round.

The retriever uses the bge-small-en-v1.5 text encoder for dense cosine
similarity search over an in-memory passage corpus. For production use,
swap DenseRetriever for a BM25 or FAISS-backed store.

Input  : ClaimDecomposition, VideoSegment, list[EvidenceRef],
         ModalConflictReport, DenseRetriever, NLIScorer
Output : (list[EvidenceRef], EvidenceStrengthReport)
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field

import torch

from schemas import (
    ClaimDecomposition,
    EvidenceRef,
    EvidenceStrengthReport,
    HopResult,
    ModalConflictReport,
    VideoSegment,
)
from models import NLIScorer, TextEncoder
from modules.module3_evidence_strength import score_evidence

logger = logging.getLogger(__name__)

MAX_RETRIEVAL_ROUNDS: int = 3


# ---------------------------------------------------------------------------
# Dense Retriever
# ---------------------------------------------------------------------------

@dataclass
class DenseRetriever:
    """
    In-memory dense retriever backed by bge-small-en-v1.5.

    Usage
    -----
    retriever = DenseRetriever(encoder)
    retriever.index(passage_list)        # add passages once
    results = retriever.search(query, top_k=3)
    """

    encoder: TextEncoder
    _passages: list[EvidenceRef] = field(default_factory=list, repr=False)
    _embeddings: torch.Tensor | None = field(default=None, repr=False)

    def index(self, passages: list[EvidenceRef]) -> None:
        """Encode and store a corpus of passages."""
        self._passages = passages
        texts = [p.passage_text for p in passages]
        self._embeddings = self.encoder.encode(texts)   # (N, D)
        logger.info("DenseRetriever: indexed %d passages.", len(passages))

    def search(self, query: str, top_k: int = 3) -> list[EvidenceRef]:
        """
        Return the top-k passages most similar to the query.
        New passages get fresh UUIDs and hop_ids=[].
        """
        if self._embeddings is None or len(self._passages) == 0:
            logger.warning("DenseRetriever.search called on empty index.")
            return []

        q_emb = self.encoder.encode([query])                       # (1, D)
        scores = (q_emb @ self._embeddings.T).squeeze(0)           # (N,)
        k = min(top_k, len(self._passages))
        top_indices = torch.topk(scores, k).indices.tolist()

        results: list[EvidenceRef] = []
        for idx in top_indices:
            src = self._passages[idx]
            results.append(
                EvidenceRef(
                    evidence_id=src.evidence_id,
                    source_url=src.source_url,
                    source_date=src.source_date,
                    passage_text=src.passage_text,
                    retrieval_score=float(scores[idx].item()),
                    hop_ids=[],  # assigned by the caller after retrieval
                )
            )
        return results


# ---------------------------------------------------------------------------
# Query builder
# ---------------------------------------------------------------------------

def build_retrieval_query(
    weak_aspect: str,
    resolved_hops: list[HopResult],
    segment: VideoSegment,
) -> str:
    """
    Construct a retrieval query by combining the weak aspect with context
    from already-resolved hops and the segment transcript.
    """
    known_facts = " ".join(
        h.answer for h in resolved_hops if not h.answer_unknown
    )
    query = weak_aspect
    if known_facts:
        query += f" Context: {known_facts}"
    query += f" Video context: {segment.transcript[:200]}"
    return query


# ---------------------------------------------------------------------------
# Gated retrieval loop
# ---------------------------------------------------------------------------

def gated_retrieval_loop(
    claim: ClaimDecomposition,
    segment: VideoSegment,
    evidence: list[EvidenceRef],
    modal_report: ModalConflictReport,
    retriever: DenseRetriever,
    nli: NLIScorer,
    max_rounds: int = MAX_RETRIEVAL_ROUNDS,
    resolved_hops: list[HopResult] | None = None,
) -> tuple[list[EvidenceRef], EvidenceStrengthReport]:
    """
    Iteratively retrieve targeted evidence for weak sub-questions until
    the gate passes or max_rounds is exhausted.

    Parameters
    ----------
    claim          : Decomposed claim from Module 1.
    segment        : Source video segment (for transcript context).
    evidence       : Starting evidence list (initial + pre-given).
    modal_report   : Cross-modal report from Module 2.
    retriever      : A DenseRetriever with an indexed corpus.
    nli            : NLI scorer for re-scoring after each round.
    max_rounds     : Maximum retrieval iterations.
    resolved_hops  : Optional already-completed hop answers (used to condition queries).

    Returns
    -------
    (final_evidence, final_strength_report)
    """
    resolved_hops = resolved_hops or []
    report: EvidenceStrengthReport | None = None

    for round_idx in range(max_rounds):
        report = score_evidence(claim, evidence, modal_report, nli)

        if report.gate_pass or not report.weak_aspects:
            logger.info(
                "Retrieval loop: gate passed after %d round(s) for claim %s.",
                round_idx, claim.claim_id,
            )
            break

        logger.info(
            "Retrieval round %d/%d — %d weak aspects for claim %s.",
            round_idx + 1, max_rounds, len(report.weak_aspects), claim.claim_id,
        )

        for aspect in report.weak_aspects:
            query = build_retrieval_query(aspect, resolved_hops, segment)
            new_passages = retriever.search(query, top_k=3)

            # Assign hop_ids based on which sub-question matched the aspect
            matched_hop = next(
                (sq.hop for sq in claim.sub_questions if sq.question == aspect), None
            )
            for p in new_passages:
                if matched_hop is not None and matched_hop not in p.hop_ids:
                    p.hop_ids.append(matched_hop)

            # Deduplicate by evidence_id
            existing_ids = {e.evidence_id for e in evidence}
            for p in new_passages:
                if p.evidence_id not in existing_ids:
                    evidence.append(p)
                    existing_ids.add(p.evidence_id)

    # Final score in case we exhausted rounds without passing
    if report is None or not report.gate_pass:
        report = score_evidence(claim, evidence, modal_report, nli)
        if not report.gate_pass:
            logger.warning(
                "Retrieval loop exhausted for claim %s — gate still not passed.",
                claim.claim_id,
            )

    return evidence, report
