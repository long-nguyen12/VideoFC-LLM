"""
dataset/dataset_schemas.py
---------------------------
Pydantic schemas that mirror the raw dataset JSON record structure exactly.

These are *input* schemas only — no pipeline logic lives here.
The DatasetRecord is the canonical representation of one row in the dataset.
The DatasetAdapter (dataset_adapter.py) converts it into pipeline types.

Dataset field inventory
-----------------------
Top level
  url                   : str          Snopes article URL for this claim
  claim                 : str          The composite claim text
  rating                : str          Snopes verdict label (e.g. "Mostly False")
  content               : str          Full Snopes article body

video_information
  video_id              : str
  video_date            : float        YYYYMMDD as float (e.g. 20160217.0)
  platform              : str          e.g. "nbcnews"
  video_headline        : str
  video_transcript      : str          Pre-annotated speech transcript
  video_description     : str
  video_length          : float        Seconds
  video_url             : str

original_rationales
  main_rationale        : str
  additional_rationale1 : str
  additional_rationale2 : str
  additional_rationale3 : str

summary_rationales
  synthesized_rationale : str
  detailed_reasons      : dict[str, str]  keys: "reason1", "reason2", ...

evidences
  num_of_evidence       : int
  evidence1..N          : [passage_text: str, urls: list[str]]

relationship_with_evidence
  list of dicts with keys like "<claim,evidence1>", "<main_rationale,evidence2>"
  and string values describing the alignment
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Video information block
# ---------------------------------------------------------------------------

class VideoInformation(BaseModel):
    video_id: str
    video_date: float                  # YYYYMMDD as float
    platform: str
    video_headline: str
    video_transcript: str
    video_description: str
    video_length: float                # seconds
    video_url: str


# ---------------------------------------------------------------------------
# Rationale blocks
# ---------------------------------------------------------------------------

class OriginalRationales(BaseModel):
    main_rationale: str
    additional_rationale1: str = ""
    additional_rationale2: str = ""
    additional_rationale3: str = ""

    def all_rationales(self) -> list[str]:
        """Return all non-empty rationale strings in order."""
        return [
            r for r in [
                self.main_rationale,
                self.additional_rationale1,
                self.additional_rationale2,
                self.additional_rationale3,
            ]
            if r.strip()
        ]


class DetailedReasons(BaseModel):
    """Flexible container — the dataset uses reason1, reason2, reason3 keys."""
    model_config = {"extra": "allow"}

    def reasons(self) -> list[str]:
        """Return reason values in key-sorted order."""
        return [
            v for k, v in sorted(self.model_extra.items())  # type: ignore[union-attr]
            if k.startswith("reason") and isinstance(v, str) and v.strip()
        ]


class SummaryRationales(BaseModel):
    synthesized_rationale: str
    detailed_reasons: DetailedReasons

    def all_reasons(self) -> list[str]:
        return self.detailed_reasons.reasons()


# ---------------------------------------------------------------------------
# Evidence block
# ---------------------------------------------------------------------------

class RawEvidence(BaseModel):
    """
    One evidence entry.  In the raw JSON each entry is:
      "evidence1": ["passage text", ["url1", "url2"]]
    Parsed into a structured object by DatasetRecord.__init__.
    """
    evidence_key: str               # "evidence1", "evidence2", …
    evidence_index: int             # 1-based integer index
    passage_text: str
    urls: list[str] = Field(default_factory=list)


class EvidencesBlock(BaseModel):
    """
    Container for all evidence entries plus the raw dict for flexible access.
    The num_of_evidence field is preserved for validation.
    """
    num_of_evidence: int
    entries: list[RawEvidence] = Field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "EvidencesBlock":
        n = int(raw.get("num_of_evidence", 0))
        entries: list[RawEvidence] = []
        for i in range(1, n + 1):
            key = f"evidence{i}"
            if key not in raw:
                continue
            value = raw[key]
            if isinstance(value, list) and len(value) >= 2:
                passage_text = str(value[0])
                urls = [str(u) for u in value[1]] if isinstance(value[1], list) else []
            elif isinstance(value, list) and len(value) == 1:
                passage_text = str(value[0])
                urls = []
            else:
                passage_text = str(value)
                urls = []
            entries.append(RawEvidence(
                evidence_key=key,
                evidence_index=i,
                passage_text=passage_text,
                urls=urls,
            ))
        return cls(num_of_evidence=n, entries=entries)


# ---------------------------------------------------------------------------
# Evidence–claim relationship annotations
# ---------------------------------------------------------------------------

class RelationshipEntry(BaseModel):
    """
    One entry in relationship_with_evidence.
    Each dict has a single key like "<claim,evidence1>" or
    "<main_rationale,evidence3>" and a string description as value.
    """
    key: str                        # e.g. "<claim,evidence1>"
    left: str                       # "claim" | "main_rationale" | "additional_rationaleN"
    right: str                      # "evidence1", "evidence2", …
    evidence_index: int             # parsed integer index from right
    description: str                # full alignment text

    @classmethod
    def from_raw_dict(cls, d: dict[str, str]) -> list["RelationshipEntry"]:
        """Parse one raw relationship dict (which has a single key)."""
        entries = []
        for key, description in d.items():
            # key format: "<left,right>"
            inner = key.strip("<>")
            if "," not in inner:
                continue
            left, right = inner.split(",", 1)
            left = left.strip()
            right = right.strip()
            # Extract numeric index from "evidence3" → 3
            idx_str = "".join(c for c in right if c.isdigit())
            evidence_index = int(idx_str) if idx_str else 0
            entries.append(cls(
                key=key,
                left=left,
                right=right,
                evidence_index=evidence_index,
                description=description,
            ))
        return entries


# ---------------------------------------------------------------------------
# Top-level DatasetRecord
# ---------------------------------------------------------------------------

class DatasetRecord(BaseModel):
    """
    One complete record from the video fact-checking dataset.
    All fields map directly to the raw JSON keys.
    """
    # Claim
    url: str
    claim: str
    rating: str                         # raw Snopes label
    content: str                        # full Snopes article body

    # Video
    video_information: VideoInformation

    # Rationales
    original_rationales: OriginalRationales
    summary_rationales: SummaryRationales

    # Evidence
    evidences: EvidencesBlock

    # Evidence–claim relationships
    relationship_with_evidence: list[RelationshipEntry] = Field(default_factory=list)

    # Misc
    other: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DatasetRecord":
        """
        Parse a raw JSON dict into a DatasetRecord.
        Handles the non-standard structure of evidences and relationships.
        """
        # Parse evidences block separately
        evidences_block = EvidencesBlock.from_dict(raw.get("evidences", {}))

        # Parse relationship list
        raw_relationships: list[dict] = raw.get("relationship_with_evidence", [])
        relationships: list[RelationshipEntry] = []
        for entry_dict in raw_relationships:
            relationships.extend(RelationshipEntry.from_raw_dict(entry_dict))

        return cls(
            url=raw["url"],
            claim=raw["claim"],
            rating=raw["rating"],
            content=raw.get("content", ""),
            video_information=VideoInformation(**raw["video_information"]),
            original_rationales=OriginalRationales(**raw["original_rationales"]),
            summary_rationales=SummaryRationales(
                synthesized_rationale=raw["summary_rationales"]["synthesized_rationale"],
                detailed_reasons=DetailedReasons(
                    **raw["summary_rationales"].get("detailed_reasons", {})
                ),
            ),
            evidences=evidences_block,
            relationship_with_evidence=relationships,
            other=raw.get("other", {}),
        )
