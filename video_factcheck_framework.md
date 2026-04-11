# Explainable Video Fact-Checking Framework with Small LLMs

## Overview

This framework performs explainable, claim-level fact-checking over video content. It assumes a dataset where each entry contains a **video segment** with a pre-annotated transcript, a **claim** linked to that segment, and an **initial set of evidence passages**. The framework checks whether existing evidence is sufficient, retrieves more if needed, runs multi-hop reasoning, detects cross-modal inconsistencies between the video content and the claim, and produces a final verdict with a full reasoning trace surfaced by a dedicated explainability module.

**What is pre-given in the dataset:**
- Transcript
- Initial evidence passages per claim

**What the framework computes:**
- Visual captions from keyframes
- Cross-modal consistency scores
- Evidence sufficiency gate
- Multi-hop reasoning chain
- Final verdict
- Explainability report (saliency attributions, conflict annotations, counterfactual)

---

## Architecture Overview

```
VideoSegment  ──────────────────────────────────────────┐
  transcript (given)                                     │
  keyframes  → caption_fn        visual caption (computed)
                                                         │
Claim text                                               │
Initial evidence (given)                                 │
         │                                               │
         ├───────────────────────────────────────────────┤
         │                                               │
         ▼                                               ▼
┌─────────────────────┐              ┌────────────────────────────────┐
│  1. Claim decomposer │              │  2. Cross-modal consistency     │
└──────────┬──────────┘              └───────────────┬────────────────┘
           │ ClaimDecomposition                       │ ModalConflictReport
           │                                          │
           └──────────────────┬───────────────────────┘
                              │
               ┌──────────────▼──────────────┐
               │  3. Evidence strength scorer  │
               └──────────────┬──────────────┘
                         ┌────┴─────┐
                    sufficient?  weak aspects?
                         │             │
                         │   ┌─────────▼──────────────┐
                         │   │  4. Targeted retrieval   │◄──┐
                         │   └─────────┬──────────────┘   │
                         │             │ new passages      │ loop (max 3)
                         │   ┌─────────▼──────────────┐   │
                         │   │    re-score evidence     │───┘
                         │             │
                    ┌────▼─────────────▼────┐
                    │  5. Multi-hop reasoning │
                    └──────────┬────────────┘
                               │ HopResult[]
                    ┌──────────▼────────────┐
                    │  6. Verdict aggregator  │
                    └──────────┬────────────┘
                               │ FinalVerdict
                    ┌──────────▼────────────┐
                    │  7. Explainability      │
                    └──────────┬────────────┘
                               │
                         ExplainabilityReport
```

---

## Data Schemas

All modules communicate through typed JSON objects.

### VideoSegment

```python
class VideoSegment(BaseModel):
    segment_id: str
    start_ts: float
    end_ts: float
    transcript: str           # pre-annotated, given in dataset
    keyframes: list[str]      # file paths or base64 strings
```

### EvidenceRef

```python
class EvidenceRef(BaseModel):
    evidence_id: str
    source_url: str
    source_date: str
    passage_text: str
    retrieval_score: float
    hop_ids: list[int]        # which hops this evidence was retrieved for
```

### ClaimDecomposition

```python
class SubQuestion(BaseModel):
    hop: int
    question: str
    depends_on_hops: list[int]
    evidence_type: str        # "video" | "web" | "kb" | "any"

class ClaimDecomposition(BaseModel):
    claim_id: str
    claim_text: str
    segment_id: str
    sub_questions: list[SubQuestion]
```

### ModalConflictReport

```python
class ModalConflictReport(BaseModel):
    segment_id: str
    vc_score: float           # visual ↔ claim NLI entailment score
    tc_score: float           # transcript ↔ claim NLI entailment score
    vt_score: float           # visual ↔ transcript NLI entailment score
    conflict_flag: bool
    dominant_conflict: str | None  # "V↔C" | "T↔C" | "V↔T" | null
```

### EvidenceStrengthReport

```python
class EvidenceStrengthReport(BaseModel):
    claim_id: str
    coverage_score: float     # fraction of sub-questions with strong evidence
    confidence_score: float   # mean NLI entailment across evidence passages
    consistency_score: float  # agreement between evidence and video content
    gate_pass: bool
    weak_aspects: list[str]   # sub-question texts that lack strong evidence
```

### HopResult

```python
class HopResult(BaseModel):
    claim_id: str
    hop: int
    question: str
    answer: str               # ≤ 2 sentences
    confidence: float
    answer_unknown: bool
    supported_by: list[str]   # evidence_ids
```

### FinalVerdict

```python
class ReasoningStep(BaseModel):
    step: int
    finding: str
    source_hop: int | None
    evidence_ids: list[str]

class FinalVerdict(BaseModel):
    claim_id: str
    segment_id: str
    verdict: str              # "supported" | "refuted" | "insufficient_evidence" | "misleading_context"
    confidence: float
    reasoning_trace: list[ReasoningStep]
    modal_conflict_used: bool
    counterfactual: str
    retrieval_rounds: int
    gate_passed: bool
```

### ExplainabilityReport

```python
class EvidenceSaliency(BaseModel):
    evidence_id: str
    hop: int
    saliency_score: float     # NLI contribution weight (0–1)
    key_span: str             # most salient phrase in the passage

class ModalAnnotation(BaseModel):
    pair: str                 # "V↔C" | "T↔C" | "V↔T"
    score: float
    timestamp: float
    human_note: str           # e.g. "Visual content contradicts claim at 12.4s"

class ExplainabilityReport(BaseModel):
    claim_id: str
    segment_id: str
    verdict: str
    confidence: float
    evidence_saliency: list[EvidenceSaliency]
    modal_annotations: list[ModalAnnotation]
    hop_summaries: list[str]  # one plain-language sentence per hop
    counterfactual: str
    gate_passed: bool
    retrieval_rounds: int
```

---

## Module 1 — Claim Decomposer

Breaks a composite claim into an ordered list of atomic sub-questions. Uses a small instruction-tuned LLM (Phi-3-mini or Mistral 7B).

### Prompt template

```
SYSTEM:
You are a claim decomposer for a video fact-checking system.
Given a composite claim and a video context summary, decompose the claim
into an ordered list of atomic sub-questions. Each sub-question must be:
- independently verifiable
- answerable from external evidence or the video content
- ordered so that earlier answers can condition later questions

Respond ONLY with valid JSON:
{
  "claim_id": string,
  "sub_questions": [
    {
      "hop": integer,
      "question": string,
      "depends_on_hops": [integer],
      "evidence_type": "video" | "web" | "kb" | "any"
    }
  ]
}

USER:
Claim: "{{claim_text}}"
Visual caption: "{{visual_caption}}"
Transcript excerpt: "{{transcript_excerpt}}"
Timestamp: {{start_ts}}s – {{end_ts}}s
Modal conflict flag: {{conflict_flag}}

Decompose this claim.
```

### Implementation

```python
def decompose_claim(
    claim_text: str,
    segment: VideoSegment,
    visual_caption: str,
    conflict_flag: bool,
    llm,
) -> ClaimDecomposition:

    prompt = build_decomposer_prompt(
        claim_text=claim_text,
        visual_caption=visual_caption,
        transcript_excerpt=segment.transcript[:300],
        start_ts=segment.start_ts,
        end_ts=segment.end_ts,
        conflict_flag=conflict_flag,
    )
    raw = llm.generate(prompt, max_tokens=512)
    data = safe_json_parse(raw)
    return ClaimDecomposition(**data, segment_id=segment.segment_id)
```

---

## Module 2 — Cross-Modal Consistency

Checks consistency across three pairs: visual content vs. claim, transcript vs. claim, and visual content vs. transcript. Uses a lightweight NLI classifier (DeBERTa-v3-small, 184M params — not an LLM).

The visual caption is computed once from keyframes via a vision LLM (LLaVA-1.6 7B or moondream2). The transcript is used directly from the dataset. Both are encoded with a shared text encoder (`bge-small-en-v1.5` or `all-MiniLM-L6-v2`) so embedding spaces are directly comparable.

```python
def compute_modal_consistency(
    claim_text: str,
    visual_caption: str,
    transcript: str,
    segment_id: str,
    nli_model,
) -> ModalConflictReport:

    vc = nli_model.entailment_score(visual_caption, claim_text)
    tc = nli_model.entailment_score(transcript, claim_text)
    vt = nli_model.entailment_score(visual_caption, transcript)

    scores = {"V↔C": vc, "T↔C": tc, "V↔T": vt}
    conflict_flag = any(s < THRESHOLDS["nli_conflict_floor"] for s in scores.values())
    dominant = min(scores, key=scores.get) if conflict_flag else None

    return ModalConflictReport(
        segment_id=segment_id,
        vc_score=vc,
        tc_score=tc,
        vt_score=vt,
        conflict_flag=conflict_flag,
        dominant_conflict=dominant,
    )
```

`dominant_conflict` feeds directly into the explainability module's `ModalAnnotation` list.

---

## Module 3 — Evidence Strength Scorer

Determines whether the initial (or accumulated) evidence is sufficient to support verdict generation, and identifies which sub-questions lack strong coverage. No sufficiency labels are required — the gate is derived entirely from NLI entailment scores and cross-modal consistency.

```python
THRESHOLDS = {
    "coverage":           0.75,   # fraction of hops needing strong evidence
    "confidence":         0.65,   # mean NLI entailment across passages
    "consistency":        0.60,   # cross-modal agreement floor
    "min_hop_confidence": 0.50,   # per-hop floor before flagging as weak
    "nli_conflict_floor": 0.40,   # below this → conflict_flag = True
}

def score_evidence(
    claim: ClaimDecomposition,
    evidence: list[EvidenceRef],
    modal_report: ModalConflictReport,
    nli_model,
) -> EvidenceStrengthReport:

    hop_scores = {}
    for sq in claim.sub_questions:
        relevant = [e for e in evidence if sq.hop in e.hop_ids]
        if not relevant:
            hop_scores[sq.hop] = 0.0
            continue
        scores = [nli_model.entailment_score(sq.question, e.passage_text)
                  for e in relevant]
        hop_scores[sq.hop] = max(scores)

    n = len(hop_scores)
    coverage    = sum(s > THRESHOLDS["min_hop_confidence"]
                      for s in hop_scores.values()) / n
    confidence  = sum(hop_scores.values()) / n
    consistency = min(
        modal_report.vc_score,
        modal_report.tc_score,
        modal_report.vt_score,
    )

    weak_aspects = [
        claim.sub_questions[hop - 1].question
        for hop, s in hop_scores.items()
        if s < THRESHOLDS["min_hop_confidence"]
    ]

    return EvidenceStrengthReport(
        claim_id=claim.claim_id,
        coverage_score=coverage,
        confidence_score=confidence,
        consistency_score=consistency,
        gate_pass=(
            coverage    >= THRESHOLDS["coverage"] and
            confidence  >= THRESHOLDS["confidence"] and
            consistency >= THRESHOLDS["consistency"]
        ),
        weak_aspects=weak_aspects,
    )
```

---

## Module 4 — Targeted Retrieval

Only triggered when `gate_pass` is False. Queries are scoped specifically to `weak_aspects` and conditioned on what prior hops already resolved — avoiding broad re-retrieval and keeping context windows tight for small models.

```python
def build_retrieval_query(
    weak_aspect: str,
    resolved_hops: list[HopResult],
    segment: VideoSegment,
) -> str:
    known_facts = " ".join(
        h.answer for h in resolved_hops if not h.answer_unknown
    )
    return (
        f"{weak_aspect} "
        f"Context: {known_facts} "
        f"Video context: {segment.transcript[:200]}"
    )

MAX_RETRIEVAL_ROUNDS = 3

def gated_retrieval_loop(
    claim: ClaimDecomposition,
    segment: VideoSegment,
    evidence: list[EvidenceRef],
    modal_report: ModalConflictReport,
    retriever,
    nli_model,
) -> tuple[list[EvidenceRef], EvidenceStrengthReport]:

    for _ in range(MAX_RETRIEVAL_ROUNDS):
        report = score_evidence(claim, evidence, modal_report, nli_model)

        if report.gate_pass or not report.weak_aspects:
            break

        for aspect in report.weak_aspects:
            query = build_retrieval_query(aspect, [], segment)
            new_passages = retriever.search(query, top_k=3)
            evidence.extend(new_passages)

    return evidence, report
```

After `MAX_RETRIEVAL_ROUNDS`, the loop exits regardless and the final strength report is passed downstream.

---

## Module 5 — Multi-Hop Reasoning

Executes sub-questions sequentially. Each hop's answer conditions the next sub-question query, so retrieval evolves across hops rather than remaining static. Uses a small RAG-setup LLM (Qwen2.5-3B or Phi-3-mini).

### Per-hop prompt template

```
SYSTEM:
You are a single-hop evidence reader. You receive one sub-question,
optionally an answer from a previous hop, and a set of retrieved passages.
Produce a concise intermediate answer with citations.

Respond ONLY with valid JSON:
{
  "hop": integer,
  "question": string,
  "answer": string,
  "confidence": float,
  "supported_by": [string],
  "answer_unknown": boolean
}

USER:
Sub-question (hop {{hop}}): "{{question}}"
Previous answer (hop {{depends_on_hop}}): "{{previous_answer}}"
Retrieved passages:
{{#each passages}}
[{{id}}] ({{source}}, {{date}}): {{text}}
{{/each}}

Answer concisely.
```

`answer_unknown: true` is the escape hatch. After two failed retries the hop is marked unknown and remaining hops are skipped, preventing hallucinated downstream answers.

### Implementation

```python
def run_multihop(
    claim: ClaimDecomposition,
    evidence: list[EvidenceRef],
    segment: VideoSegment,
    llm,
    retriever,
) -> list[HopResult]:

    hop_results: list[HopResult] = []

    for sq in claim.sub_questions:
        prior = {h.hop: h.answer for h in hop_results
                 if h.hop in sq.depends_on_hops}

        relevant = [e for e in evidence if sq.hop in e.hop_ids]
        if not relevant and sq.evidence_type != "video":
            query = build_retrieval_query(sq.question, hop_results, segment)
            relevant = retriever.search(query, top_k=3)

        result = run_single_hop(sq, prior, relevant, llm)
        hop_results.append(result)

        if result.answer_unknown:
            break

    return hop_results


def run_single_hop(
    sq: SubQuestion,
    prior_answers: dict[int, str],
    passages: list[EvidenceRef],
    llm,
    max_retries: int = 2,
) -> HopResult:

    for attempt in range(max_retries + 1):
        prompt = build_hop_prompt(sq, prior_answers, passages)
        raw = llm.generate(prompt, max_tokens=256)
        try:
            data = safe_json_parse(raw)
            result = HopResult(**data)
            if not result.answer_unknown or attempt == max_retries:
                return result
        except Exception:
            continue

    return HopResult(
        hop=sq.hop, answer="", confidence=0.0,
        answer_unknown=True, supported_by=[]
    )
```

---

## Module 6 — Verdict Aggregator

Synthesises all hop answers and the modal conflict report into a final verdict. Uses Mistral 7B or LLaMA 3.1 8B — the only module that needs a larger context window.

### Prompt template

```
SYSTEM:
You are a fact-checking verdict aggregator. You receive the original
composite claim, intermediate answers from N reasoning hops, and a
cross-modal conflict report. Produce a final verdict.

Respond ONLY with valid JSON:
{
  "claim_id": string,
  "verdict": "supported" | "refuted" | "insufficient_evidence" | "misleading_context",
  "confidence": float,
  "reasoning_trace": [
    {
      "step": integer,
      "finding": string,
      "source_hop": integer | null,
      "evidence_ids": [string]
    }
  ],
  "modal_conflict_used": boolean,
  "counterfactual": string
}

USER:
Original claim: "{{claim_text}}"
Visual caption: "{{visual_caption}}"
Timestamp: {{start_ts}}s – {{end_ts}}s

Hop answers:
{{#each hops}}
Hop {{hop}}: Q: "{{question}}" → A: "{{answer}}" (confidence: {{confidence}})
{{/each}}

Cross-modal conflict report:
- Visual↔Claim score:      {{vc_score}}
- Transcript↔Claim score:  {{tc_score}}
- Visual↔Transcript score: {{vt_score}}
- Conflict flag:            {{conflict_flag}}
- Dominant conflict:        {{dominant_conflict}}

Evidence gate passed: {{gate_pass}}

Produce the verdict.
```

### Implementation

```python
def aggregate_verdict(
    claim: ClaimDecomposition,
    segment: VideoSegment,
    visual_caption: str,
    hop_results: list[HopResult],
    modal_report: ModalConflictReport,
    strength_report: EvidenceStrengthReport,
    retrieval_rounds: int,
    llm,
) -> FinalVerdict:

    prompt = build_aggregator_prompt(
        claim, segment, visual_caption,
        hop_results, modal_report, strength_report
    )
    raw = llm.generate(prompt, max_tokens=1024)
    data = safe_json_parse(raw)

    return FinalVerdict(
        **data,
        segment_id=segment.segment_id,
        retrieval_rounds=retrieval_rounds,
        gate_passed=strength_report.gate_pass,
    )
```

---

## Module 7 — Explainability Module

Transforms the internal pipeline outputs into a structured, human-readable explanation. This is the single point of accountability for all transparency outputs: it computes evidence saliency scores, renders modal conflict annotations with timestamps, produces plain-language hop summaries, and emits a counterfactual. Downstream consumers (UIs, auditors, evaluation scripts) read only the `ExplainabilityReport` — they do not need access to intermediate pipeline objects.

### Design principles

**Saliency via NLI contribution.** For each evidence passage, the saliency score is the NLI entailment score of that passage against the final verdict label, normalised across passages within the same hop. This gives a grounded, model-derived importance weight without requiring gradient access.

**Modal annotations are timestamp-anchored.** Each conflict pair is resolved to a human-readable note tied to the segment timestamp, so reviewers can seek directly to the relevant moment.

**Hop summaries use a small generative LLM.** A single sentence per hop is generated by the same per-hop reader model (Qwen2.5-3B / Phi-3-mini), conditioned on the hop answer and its supporting evidence IDs.

**Counterfactual is passed through from the verdict aggregator.** No additional LLM call is needed.

### Prompt template — hop summariser

```
SYSTEM:
You are an explainability assistant. Given one intermediate reasoning answer
and its supporting evidence IDs, write exactly one plain-language sentence
that a non-expert could read. Do not use jargon. Do not start with "I".

Respond ONLY with a JSON object:
{ "summary": string }

USER:
Hop {{hop}}: Q: "{{question}}"
Answer: "{{answer}}" (confidence: {{confidence}})
Supporting evidence: {{supported_by}}

Summarise in one sentence.
```

### Implementation

```python
def build_modal_annotations(
    modal_report: ModalConflictReport,
    segment: VideoSegment,
) -> list[ModalAnnotation]:
    annotations = []
    pair_scores = {
        "V↔C": modal_report.vc_score,
        "T↔C": modal_report.tc_score,
        "V↔T": modal_report.vt_score,
    }
    pair_labels = {
        "V↔C": "Visual content contradicts claim",
        "T↔C": "Transcript contradicts claim",
        "V↔T": "Visual content contradicts transcript",
    }
    for pair, score in pair_scores.items():
        if score < THRESHOLDS["nli_conflict_floor"]:
            annotations.append(ModalAnnotation(
                pair=pair,
                score=score,
                timestamp=segment.start_ts,
                human_note=(
                    f"{pair_labels[pair]} at {segment.start_ts}s "
                    f"({pair} NLI score: {score:.2f})."
                ),
            ))
    return annotations


def compute_evidence_saliency(
    hop_results: list[HopResult],
    evidence: list[EvidenceRef],
    verdict_label: str,
    nli_model,
) -> list[EvidenceSaliency]:
    saliency_list = []
    evidence_map = {e.evidence_id: e for e in evidence}

    for hop in hop_results:
        raw_scores = {}
        for eid in hop.supported_by:
            passage = evidence_map[eid].passage_text
            raw_scores[eid] = nli_model.entailment_score(passage, verdict_label)

        total = sum(raw_scores.values()) or 1.0
        for eid, score in raw_scores.items():
            passage_text = evidence_map[eid].passage_text
            # Key span: sentence with highest token overlap with verdict label
            sentences = passage_text.split(". ")
            key_span = max(sentences, key=lambda s: len(set(s.lower().split())
                           & set(verdict_label.lower().split())), default=sentences[0])
            saliency_list.append(EvidenceSaliency(
                evidence_id=eid,
                hop=hop.hop,
                saliency_score=round(score / total, 4),
                key_span=key_span.strip(),
            ))

    return saliency_list


def generate_hop_summaries(
    hop_results: list[HopResult],
    llm,
) -> list[str]:
    summaries = []
    for hop in hop_results:
        if hop.answer_unknown:
            summaries.append(
                f"Hop {hop.hop} could not be resolved: "
                f"no sufficient evidence found for \"{hop.question}\"."
            )
            continue
        prompt = build_hop_summary_prompt(hop)
        raw = llm.generate(prompt, max_tokens=128)
        data = safe_json_parse(raw)
        summaries.append(data.get("summary", hop.answer))
    return summaries


def build_explainability_report(
    verdict: FinalVerdict,
    hop_results: list[HopResult],
    evidence: list[EvidenceRef],
    modal_report: ModalConflictReport,
    segment: VideoSegment,
    nli_model,
    llm,
) -> ExplainabilityReport:

    saliency      = compute_evidence_saliency(
                        hop_results, evidence, verdict.verdict, nli_model)
    annotations   = build_modal_annotations(modal_report, segment)
    hop_summaries = generate_hop_summaries(hop_results, llm)

    return ExplainabilityReport(
        claim_id=verdict.claim_id,
        segment_id=verdict.segment_id,
        verdict=verdict.verdict,
        confidence=verdict.confidence,
        evidence_saliency=saliency,
        modal_annotations=annotations,
        hop_summaries=hop_summaries,
        counterfactual=verdict.counterfactual,
        gate_passed=verdict.gate_passed,
        retrieval_rounds=verdict.retrieval_rounds,
    )
```

---

## Full Pipeline

```python
def run_pipeline(
    claim_text: str,
    segment: VideoSegment,
    initial_evidence: list[EvidenceRef],
    models: ModelBundle,
) -> ExplainabilityReport:

    # 1. Visual captioning — only computed step on the video side
    visual_caption = models.caption_fn(segment.keyframes)

    # 2. Cross-modal consistency
    modal_report = compute_modal_consistency(
        claim_text, visual_caption,
        segment.transcript, segment.segment_id,
        models.nli
    )

    # 3. Claim decomposition
    claim = decompose_claim(
        claim_text, segment, visual_caption,
        modal_report.conflict_flag, models.decomposer_llm
    )

    # 4. Gated evidence retrieval
    evidence, strength_report = gated_retrieval_loop(
        claim, segment, list(initial_evidence),
        modal_report, models.retriever, models.nli
    )

    # 5. Multi-hop reasoning
    hop_results = run_multihop(
        claim, evidence, segment,
        models.hop_llm, models.retriever
    )

    # 6. Verdict
    retrieval_rounds = (
        0 if strength_report.gate_pass else MAX_RETRIEVAL_ROUNDS
    )
    verdict = aggregate_verdict(
        claim, segment, visual_caption,
        hop_results, modal_report, strength_report,
        retrieval_rounds, models.aggregator_llm
    )

    # 7. Explainability — sole output returned to the caller
    return build_explainability_report(
        verdict, hop_results, evidence,
        modal_report, segment,
        models.nli, models.hop_llm
    )
```

The pipeline returns an `ExplainabilityReport` rather than a raw `FinalVerdict`. All verdict fields are preserved inside the report; callers never need to unwrap intermediate objects.

---

## Small LLM Role Assignment

| Module | Model | Params | Justification |
|---|---|---|---|
| Visual captioner | LLaVA-1.6 / moondream2 | 7B / 1.8B | Only VLM in the pipeline; runs on-device |
| Text encoder | bge-small-en-v1.5 | 33M | Shared across all text streams; fast semantic similarity |
| NLI scorer | DeBERTa-v3-small | 184M | More reliable entailment than prompting a generative model; also drives saliency scores in Module 7 |
| Claim decomposer | Phi-3-mini / Mistral 7B | 3.8B / 7B | Structured few-shot template task |
| Per-hop reader | Qwen2.5-3B / Phi-3-mini | 3B / 3.8B | Short context enforces concise intermediate answers; reused for hop summarisation in Module 7 |
| Verdict aggregator | Mistral 7B / LLaMA 3.1 8B | 7B / 8B | Needs full context across all hops and conflict signals |

---

## Explainability Outputs

All explainability outputs are consolidated in `ExplainabilityReport` (Module 7). No transparency signal is scattered across other modules.

**Evidence saliency** — `evidence_saliency[]` lists every supporting passage with a normalised importance weight and the most salient span, so reviewers know exactly which text drove each hop's answer.

**Modal annotations** — `modal_annotations[]` maps each conflicting modality pair to a timestamped, plain-language note, e.g.: *"Visual content contradicts claim at 12.4s (V↔C NLI score: 0.21)."*

**Hop summaries** — `hop_summaries[]` provides one plain-language sentence per reasoning hop, making the multi-hop chain readable without requiring access to raw evidence passages.

**Counterfactual** — `counterfactual` states what would need to change for the verdict to flip, e.g.: *"The claim would be supported if the event date matched the claimed date."*

**Gate transparency** — `gate_passed` and `retrieval_rounds` surface evidence quality directly in the report, so consumers can distinguish confident verdicts from those produced under weak evidence.

---

## Threshold Reference

```python
THRESHOLDS = {
    "coverage":           0.75,
    "confidence":         0.65,
    "consistency":        0.60,
    "min_hop_confidence": 0.50,
    "nli_conflict_floor": 0.40,
}

MAX_RETRIEVAL_ROUNDS = 3
```

Tune by plotting precision/recall for `gate_pass` against final verdict correctness on a held-out split.

---

## Key Design Decisions

**OCR removed entirely.** The pipeline relies on the transcript (pre-annotated) and visual captions (computed from keyframes). On-screen text detection added a brittle, noisy signal without a commensurate gain in verdict quality given the NLI-based consistency checks already in place.

**Explainability is a first-class module, not a post-hoc annotation.** Module 7 consolidates all transparency outputs into a single `ExplainabilityReport`. It computes saliency during the same NLI pass already used for verdict scoring, so there is no additional model overhead.

**The pipeline returns `ExplainabilityReport`, not `FinalVerdict`.** All verdict fields are preserved inside the report. This enforces the contract that explainability is not optional — callers cannot bypass Module 7.

**No sufficiency labels required.** The evidence gate is derived from NLI entailment scores and cross-modal consistency alone.

**Targeted retrieval over full re-retrieval.** Only `weak_aspects` are re-queried, conditioned on resolved hops, keeping retrieval scoped and context windows tight for small models.

**Role specialisation over one large model.** Each module is sized for its task. The NLI scorer (184M) handles both entailment and saliency. The per-hop reader (3B) is reused for hop summarisation. Only the aggregator needs a larger context.

**Honest verdicts.** If `gate_pass` remains False after `MAX_RETRIEVAL_ROUNDS`, the verdict is labelled `insufficient_evidence`. The `gate_passed` and `retrieval_rounds` fields surface this transparently in every output.

**Cross-modal conflict as a first-class signal.** `conflict_flag` feeds the claim decomposer, the evidence scorer, the aggregator prompt, and the explainability module's `modal_annotations` — not a post-hoc annotation.
