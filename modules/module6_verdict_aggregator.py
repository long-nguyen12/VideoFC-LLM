from __future__ import annotations

import logging

from models import GenerativeLLM

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

from modules.prompt_template import _AGGREGATOR_SYSTEM_PROMPT


def _format_hop_answers(hop_results: list[dict], max_hops: int = 4) -> str:
    lines = []
    for h in hop_results[:max_hops]:
        status = "UNKNOWN" if h["answer_unknown"] else f"conf={h['confidence']:.2f}"
        # Truncate long answers to keep the prompt short for small models
        answer = h["answer"] + "…" if len(h["answer"]) > 120 else h["answer"]
        lines.append(f'Hop {h["hop"]}: Q: "{h["question"]}" → A: "{answer}" ({status})')
    if len(hop_results) > max_hops:
        lines.append(f"… ({len(hop_results) - max_hops} more hops truncated)")
    return "\n".join(lines) if lines else "No hop answers available."


def _build_aggregator_prompt(
    claim: dict,
    segment: dict,
    visual_caption: str,
    hop_results: list[dict],
    modal_report: dict,
    strength_report: dict,
) -> list[dict[str, str]]:
    # Truncate free-text fields so the total prompt fits in a small context window
    caption_short = visual_caption
    conflict_line = (
        f"conflict={modal_report['conflict_flag']}  dominant={modal_report['dominant_conflict']}  "
        f"vc={modal_report['vc_score']:.2f}  tc={modal_report['tc_score']:.2f}  vt={modal_report['vt_score']:.2f}"
    )
    gate_line = (
        f"gate={'PASS' if strength_report['gate_pass'] else 'FAIL'}  "
        f"cov={strength_report['coverage_score']:.2f}  "
        f"conf={strength_report['confidence_score']:.2f}  "
        f"cons={strength_report['consistency_score']:.2f}"
    )
    user_content = (
        f'Claim: "{claim["claim_text"]}"\n'
        f'Claim ID: "{claim["claim_id"]}"\n'
        f'Visual: "{caption_short}"\n\n'
        f"Hops:\n{_format_hop_answers(hop_results)}\n\n"
        f"Modal: {conflict_line}\n"
        f"Gate:  {gate_line}\n\n"
    )
    return [
        {"role": "system", "content": _AGGREGATOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]



# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def aggregate_verdict(
    claim: dict,
    segment: dict,
    visual_caption: str,
    hop_results: list[dict],
    modal_report: dict,
    strength_report: dict,
    retrieval_rounds: int,
    llm: GenerativeLLM,
    max_retries: int = 2,
) -> dict:
    # If gate never passed, emit insufficient_evidence without calling the LLM
    if not strength_report["gate_pass"]:
        logger.warning(
            "Claim %s: gate_pass=False after %d retrieval rounds. "
            "Emitting insufficient_evidence verdict.",
            claim["claim_id"],
            retrieval_rounds,
        )
        return {
            "claim_id": claim["claim_id"],
            "segment_id": segment["segment_id"],
            "verdict": "insufficient_evidence",
            "confidence": 0.0,
            "reasoning_trace": [
                {
                    "step": 1,
                    "finding": (
                        f"Evidence gate did not pass after {retrieval_rounds} "
                        f"retrieval round(s). Weak aspects: "
                        f"{'; '.join(strength_report['weak_aspects'])}"
                    ),
                    "source_hop": None,
                    "evidence_ids": [],
                }
            ],
            "modal_conflict_used": modal_report["conflict_flag"],
            "counterfactual": (
                "The verdict could change if sufficient evidence were found for: "
                + "; ".join(strength_report["weak_aspects"])
            ),
            "retrieval_rounds": retrieval_rounds,
            "gate_passed": False,
        }

    prompt = _build_aggregator_prompt(
        claim,
        segment,
        visual_caption,
        hop_results,
        modal_report,
        strength_report,
    )

    data = llm.generate_json(prompt, max_new_tokens=512, max_retries=max_retries)

    if data is None:
        logger.error("All aggregator attempts failed. Using fallback verdict.")
        return {
            "claim_id": claim["claim_id"],
            "segment_id": segment["segment_id"],
            "verdict": "insufficient_evidence",
            "confidence": 0.0,
            "reasoning_trace": [
                {
                    "step": 1,
                    "finding": "Aggregator LLM failed to produce a valid verdict.",
                    "source_hop": None,
                    "evidence_ids": [],
                }
            ],
            "modal_conflict_used": modal_report["conflict_flag"],
            "counterfactual": "",
            "retrieval_rounds": retrieval_rounds,
            "gate_passed": strength_report["gate_pass"],
        }

    reasoning_trace = [
        {
            "step": rs["step"],
            "finding": rs["finding"],
            "source_hop": rs.get("source_hop"),
            "evidence_ids": rs.get("evidence_ids", []),
        }
        for rs in data.get("reasoning_trace", [])
    ]

    return {
        "claim_id": data.get("claim_id", claim["claim_id"]),
        "segment_id": segment["segment_id"],
        "verdict": data.get("verdict", "insufficient_evidence"),
        "confidence": float(data.get("confidence", 0.0)),
        "reasoning_trace": reasoning_trace,
        "modal_conflict_used": bool(data.get("modal_conflict_used", False)),
        "counterfactual": data.get("counterfactual", ""),
        "retrieval_rounds": retrieval_rounds,
        "gate_passed": strength_report["gate_pass"],
    }
