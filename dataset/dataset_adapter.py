"""
dataset/dataset_adapter.py
---------------------------
Converts a DatasetRecord into the pipeline's native input types:
  VideoSegment       ← video_information (transcript, headline, description)
  list[EvidenceRef]  ← evidences block, with hop_ids assigned from relationships
  RationaleContext   ← rationale fields packaged for prompt injection

Design decisions
----------------

VideoSegment construction
  The dataset provides video_transcript as the transcript field.
  video_headline and video_description are prepended to give the decomposer
  richer context (they provide the "visual" summary in the absence of real
  keyframes). video_date is converted from YYYYMMDD float → ISO string and
  stored in segment metadata.

  Keyframes are empty by default — the dataset does not supply frame files.
  If frame extraction is performed externally, pass keyframe_paths to
  record_to_segment(). When keyframes are absent, the captioner is skipped
  and video_headline + video_description are used as the visual caption.

EvidenceRef construction
  Each evidence entry (evidence1…N) becomes one EvidenceRef.
  evidence_id = "<video_id>-ev<index>"
  source_url  = first URL in the evidence's URL list (or video_url fallback)
  source_date = video_date as ISO string
  retrieval_score = 1.0 (dataset-provided evidence is treated as gold)
  hop_ids     = assigned by _assign_hop_ids() using relationship_with_evidence

Hop ID assignment
  relationship_with_evidence keys encode which evidence is relevant to the
  claim vs. which is relevant to specific rationale points. We use the
  evidence index to number hops: the first unique evidence in a claim
  relationship goes to hop 1, the second to hop 2, etc.  Rationale
  relationships that mention an evidence not yet seen in a claim relationship
  are assigned to the next available hop.

RationaleContext
  Packages original and summary rationales into a single object that
  individual modules can optionally inject into their prompts (e.g. to
  bias the decomposer or the aggregator toward the gold rationale structure).
  This is kept separate from VideoSegment so it can be omitted during
  blind evaluation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional

from schemas.data_models import EvidenceRef, VideoSegment
from dataset.dataset_schemas import DatasetRecord, RawEvidence, RelationshipEntry
from dataset.label_mapper import rating_to_verdict


# ---------------------------------------------------------------------------
# Rationale context (optional prompt enrichment)
# ---------------------------------------------------------------------------

@dataclass
class RationaleContext:
    """
    Gold rationale data extracted from the dataset record.
    Passed optionally into pipeline modules to guide generation.

    Attributes
    ----------
    main_rationale        : The primary Snopes rationale sentence.
    additional_rationales : Up to three supplementary rationale strings.
    synthesized_rationale : The longer synthesized explanation.
    detailed_reasons      : Ordered list of structured reasoning sentences.
    gold_verdict          : Pipeline verdict derived from the Snopes rating.
    snopes_rating         : Raw Snopes rating string (e.g. "Mostly False").
    snopes_url            : URL of the Snopes article.
    article_content       : Full Snopes article body (for retrieval seeding).
    """
    main_rationale: str
    additional_rationales: list[str] = field(default_factory=list)
    synthesized_rationale: str = ""
    detailed_reasons: list[str] = field(default_factory=list)
    gold_verdict: str = "insufficient_evidence"
    snopes_rating: str = ""
    snopes_url: str = ""
    article_content: str = ""

    def prompt_summary(self, max_chars: int = 600) -> str:
        """
        Return a compact prompt-injectable summary of the gold rationale.
        Truncated to max_chars to fit within small-model context windows.
        """
        lines = [f"Known verdict: {self.snopes_rating}"]
        if self.main_rationale:
            lines.append(f"Main rationale: {self.main_rationale}")
        for i, r in enumerate(self.additional_rationales, 1):
            lines.append(f"Supporting rationale {i}: {r}")
        text = "\n".join(lines)
        return text[:max_chars] if len(text) > max_chars else text


# ---------------------------------------------------------------------------
# Date helper
# ---------------------------------------------------------------------------

def _yyyymmdd_to_iso(value: float) -> str:
    """Convert 20160217.0 → '2016-02-17'. Returns empty string on failure."""
    try:
        s = str(int(value))
        dt = datetime.strptime(s, "%Y%m%d")
        return dt.date().isoformat()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Hop ID assignment
# ---------------------------------------------------------------------------

def _assign_hop_ids(
    evidence_entries: list[RawEvidence],
    relationships: list[RelationshipEntry],
) -> dict[int, list[int]]:
    """
    Derive hop_ids for each evidence entry from relationship annotations.

    Strategy
    --------
    1. Walk relationship entries in order. Entries whose left side is "claim"
       are processed first (they are the primary claim-evidence links).
    2. Each distinct evidence_index encountered in a "claim" relationship
       is assigned to the next available hop (1, 2, 3, …).
    3. Entries whose left side contains "rationale" are then processed;
       evidence already assigned to a hop retains that hop; new evidence
       gets the next available hop.
    4. Evidence entries with no relationship annotation are left with
       hop_ids=[] so the retrieval loop can re-score them freely.

    Returns
    -------
    dict mapping evidence_index (1-based) → list[int] of hop IDs.
    """
    hop_map: dict[int, list[int]] = {e.evidence_index: [] for e in evidence_entries}
    evidence_to_hop: dict[int, int] = {}
    next_hop = 1

    # Sort: claim-relationships first, then rationale-relationships
    claim_rels  = [r for r in relationships if r.left == "claim"]
    other_rels  = [r for r in relationships if r.left != "claim"]

    for rel in claim_rels + other_rels:
        idx = rel.evidence_index
        if idx == 0 or idx not in hop_map:
            continue
        if idx not in evidence_to_hop:
            evidence_to_hop[idx] = next_hop
            next_hop += 1
        hop_id = evidence_to_hop[idx]
        if hop_id not in hop_map[idx]:
            hop_map[idx].append(hop_id)

    return hop_map


# ---------------------------------------------------------------------------
# VideoSegment construction
# ---------------------------------------------------------------------------

def record_to_segment(
    record: DatasetRecord,
    keyframe_paths: Optional[list[str]] = None,
) -> VideoSegment:
    """
    Build a VideoSegment from a DatasetRecord.

    The transcript is the video_transcript field.  The headline and
    description are NOT appended to the transcript — they are preserved
    separately in the visual caption step (see record_to_visual_caption).

    Parameters
    ----------
    record         : Parsed DatasetRecord.
    keyframe_paths : Optional list of local frame file paths. When omitted,
                     the captioner will be skipped (see pipeline adapter).

    Returns
    -------
    VideoSegment
    """
    vi = record.video_information
    return VideoSegment(
        segment_id=vi.video_id,
        start_ts=0.0,
        end_ts=vi.video_length,
        transcript=vi.video_transcript,
        keyframes=keyframe_paths or [],
    )


def record_to_visual_caption(record: DatasetRecord) -> str:
    """
    Build a synthetic visual caption from headline + description when no
    real keyframes are available.  This is used instead of calling the VLM.

    Format: "<headline>. <description>"
    """
    vi = record.video_information
    parts = [p.strip() for p in [vi.video_headline, vi.video_description] if p.strip()]
    return ". ".join(parts)


# ---------------------------------------------------------------------------
# EvidenceRef construction
# ---------------------------------------------------------------------------

def record_to_evidence(record: DatasetRecord) -> list[EvidenceRef]:
    """
    Convert the dataset's evidences block into a list of EvidenceRef objects
    with hop_ids assigned from the relationship annotations.

    Parameters
    ----------
    record : Parsed DatasetRecord.

    Returns
    -------
    list[EvidenceRef]  — one per evidence entry, ordered by evidence index.
    """
    vi = record.video_information
    iso_date = _yyyymmdd_to_iso(vi.video_date)

    hop_map = _assign_hop_ids(
        record.evidences.entries,
        record.relationship_with_evidence,
    )

    result: list[EvidenceRef] = []
    for entry in record.evidences.entries:
        # Use the first listed URL as the canonical source, fall back to video_url
        source_url = entry.urls[0] if entry.urls else vi.video_url

        result.append(EvidenceRef(
            evidence_id=f"{vi.video_id}-ev{entry.evidence_index}",
            source_url=source_url,
            source_date=iso_date,
            passage_text=entry.passage_text,
            retrieval_score=1.0,   # gold evidence; full weight
            hop_ids=hop_map.get(entry.evidence_index, []),
        ))

    return result


# ---------------------------------------------------------------------------
# RationaleContext construction
# ---------------------------------------------------------------------------

def record_to_rationale_context(record: DatasetRecord) -> RationaleContext:
    """
    Build a RationaleContext from the rationale fields of a DatasetRecord.

    Parameters
    ----------
    record : Parsed DatasetRecord.

    Returns
    -------
    RationaleContext
    """
    orig = record.original_rationales
    summ = record.summary_rationales

    return RationaleContext(
        main_rationale=orig.main_rationale,
        additional_rationales=[
            r for r in [
                orig.additional_rationale1,
                orig.additional_rationale2,
                orig.additional_rationale3,
            ] if r.strip()
        ],
        synthesized_rationale=summ.synthesized_rationale,
        detailed_reasons=summ.all_reasons(),
        gold_verdict=rating_to_verdict(record.rating),
        snopes_rating=record.rating,
        snopes_url=record.url,
        article_content=record.content,
    )


# ---------------------------------------------------------------------------
# Combined convenience function
# ---------------------------------------------------------------------------

@dataclass
class PipelineInputs:
    """All pipeline inputs derived from a single DatasetRecord."""
    claim_text: str
    claim_id: str
    segment: VideoSegment
    visual_caption: str          # synthetic caption (headline + description)
    initial_evidence: list[EvidenceRef]
    rationale_context: RationaleContext
    gold_verdict: str            # pipeline verdict from Snopes rating
    gold_label: int              # integer class index


def record_to_pipeline_inputs(
    record: DatasetRecord,
    keyframe_paths: Optional[list[str]] = None,
) -> PipelineInputs:
    """
    Convert a DatasetRecord into all pipeline inputs in one call.

    Parameters
    ----------
    record         : Parsed DatasetRecord.
    keyframe_paths : Optional extracted keyframe file paths.

    Returns
    -------
    PipelineInputs
    """
    from dataset.label_mapper import rating_to_label

    segment  = record_to_segment(record, keyframe_paths)
    evidence = record_to_evidence(record)
    caption  = record_to_visual_caption(record)
    rationale_ctx = record_to_rationale_context(record)
    gold_verdict  = rating_to_verdict(record.rating)
    gold_label    = rating_to_label(record.rating)

    # claim_id: stable hash of video_id + normalised claim text
    raw_id = f"{record.video_information.video_id}_{record.claim[:40]}"
    claim_id = re.sub(r"[^a-zA-Z0-9_-]", "_", raw_id)[:64]

    return PipelineInputs(
        claim_text=record.claim,
        claim_id=claim_id,
        segment=segment,
        visual_caption=caption,
        initial_evidence=evidence,
        rationale_context=rationale_ctx,
        gold_verdict=gold_verdict,
        gold_label=gold_label,
    )
