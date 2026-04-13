"""
modules/module6_verdict_aggregator.py
--------------------------------------
Module 6 — Verdict Aggregator

Synthesises all hop answers, the modal conflict report, and the evidence
strength report into a final labelled verdict with a step-by-step reasoning
trace and a counterfactual.

Uses Mistral-7B or LLaMA-3.1-8B — the only module that needs a larger
context window because it must hold all hop answers simultaneously.

Possible verdicts:
  "supported"              — evidence and modalities consistently support the claim
  "refuted"                — evidence clearly contradicts the claim
  "insufficient_evidence"  — gate never passed; verdict unreliable
  "misleading_context"     — claim is technically accurate but omits key context

Input  : ClaimDecomposition, VideoSegment, visual_caption, list[HopResult],
         ModalConflictReport, EvidenceStrengthReport, retrieval_rounds,
         GenerativeLLM
Output : FinalVerdict
"""

from __future__ import annotations

import json
import logging
import re

from models import GenerativeLLM
from modules.utils import safe_json_parse as _safe_json_parse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

_AGGREGATOR_SYSTEM_PROMPT = """\
You are a fact-checking verdict aggregator.
Given a claim, hop answers, a cross-modal conflict report, and evidence gate status, produce a verdict.

OUTPUT FORMAT RULES (MANDATORY):
1. Respond ONLY with a valid JSON object. Do not include markdown, code blocks, explanations, greetings, or any text outside the JSON.
2. Use double quotes for ALL keys and string values. Single quotes are invalid JSON and will cause parsing errors.
3. Ensure proper JSON escaping for special characters (e.g., \\", \\\\, \\n).
4. Do not include trailing commas, comments, or schema annotations in the output.
5. The "reasoning_trace" array must contain at most 3 steps. Each "finding" must be under 20 words.
6. The "counterfactual" field must contain exactly one sentence.
7. The "verdict" field must be exactly one of: "yes" or "no".
8. The "confidence" field must be a float between 0.0 and 1.0, inclusive.

REQUIRED JSON SCHEMA:
{{
  "claim_id": "<string>",
  "verdict": "<one of: yes | no >",
  "confidence": <float 0.0-1.0>,
  "reasoning_trace": [
    {{
      "step": <integer starting at 1>,
      "finding": "<string, under 20 words>",
      "source_hop": <integer or null>,
      "evidence_ids": ["<string>"]
    }}
  ],
  "modal_conflict_used": <true | false>,
  "counterfactual": "<string, exactly one sentence>"
}}
"""


def _format_hop_answers(hop_results: list[dict], max_hops: int = 4) -> str:
    lines = []
    for h in hop_results[:max_hops]:
        status = "UNKNOWN" if h["answer_unknown"] else f"conf={h['confidence']:.2f}"
        # Truncate long answers to keep the prompt short for small models
        answer = h["answer"][:120] + "…" if len(h["answer"]) > 120 else h["answer"]
        lines.append(f'Hop {h["hop"]}: Q: "{h["question"][:80]}" → A: "{answer}" ({status})')
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
    caption_short  = visual_caption[:120]
    conflict_line  = (
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
        f'Claim: "{claim["claim_text"][:200]}"\n'
        f'Claim ID: "{claim["claim_id"]}"\n'
        f'Visual: "{caption_short}"\n\n'
        f"Hops:\n{_format_hop_answers(hop_results)}\n\n"
        f"Modal: {conflict_line}\n"
        f"Gate:  {gate_line}\n\n"
        "Produce the verdict. Output ONLY valid JSON. Start your response with { and end with }."
    )
    return [
        {"role": "system", "content": _AGGREGATOR_SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]


# JSON parsing — shared implementation lives in modules/utils.py


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
    """
    Synthesise all pipeline signals into a final verdict.

    Parameters
    ----------
    claim             : Decomposed claim from Module 1.
    segment           : Source video segment.
    visual_caption    : Caption from the visual captioner.
    hop_results       : All completed hop answers from Module 5.
    modal_report      : Cross-modal consistency report from Module 2.
    strength_report   : Evidence strength report from Module 3/4.
    retrieval_rounds  : Number of retrieval rounds consumed (for transparency).
    llm               : Aggregator LLM (Mistral-7B or LLaMA-3.1-8B).
    max_retries       : Parse retry budget.

    Returns
    -------
    dict
    """
    # If gate never passed, emit insufficient_evidence without calling the LLM
    if not strength_report["gate_pass"]:
        logger.warning(
            "Claim %s: gate_pass=False after %d retrieval rounds. "
            "Emitting insufficient_evidence verdict.",
            claim["claim_id"], retrieval_rounds,
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
        claim, segment, visual_caption,
        hop_results, modal_report, strength_report,
    )

    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            raw = llm.generate(prompt, max_new_tokens=512)
            data = _safe_json_parse(raw)

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

        except Exception as exc:
            last_exc = exc
            logger.warning(
                "Aggregator attempt %d/%d failed: %s", attempt + 1, max_retries + 1, exc
            )

    # Fallback on complete failure
    logger.error("All aggregator attempts failed (%s). Using fallback verdict.", last_exc)
    return {
        "claim_id": claim["claim_id"],
        "segment_id": segment["segment_id"],
        "verdict": "insufficient_evidence",
        "confidence": 0.0,
        "reasoning_trace": [
            {
                "step": 1,
                "finding": f"Aggregator LLM failed to produce a valid verdict: {last_exc}",
                "source_hop": None,
                "evidence_ids": [],
            }
        ],
        "modal_conflict_used": modal_report["conflict_flag"],
        "counterfactual": "",
        "retrieval_rounds": retrieval_rounds,
        "gate_passed": strength_report["gate_pass"],
    }
