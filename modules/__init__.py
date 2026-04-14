"""
modules/
--------
The 7 discrete pipeline modules.
"""

from .module1_claim_decomposer import decompose_claim
from .module3_evidence_strength import score_evidence
from .module4_targeted_retrieval import DenseRetriever, gated_retrieval_loop
from .module5_multihop_reasoning import run_multihop
from .module6_verdict_aggregator import aggregate_verdict
from .module7_explainability import build_explainability_report

__all__ = [
    "decompose_claim",
    "score_evidence",
    "DenseRetriever",
    "gated_retrieval_loop",
    "run_multihop",
    "aggregate_verdict",
    "build_explainability_report",
]
