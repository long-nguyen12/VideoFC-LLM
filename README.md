# Video Fact-Checking Framework

Explainable, claim-level fact-checking over video content using small HuggingFace models and PyTorch.

## Architecture

```
VideoSegment (transcript + keyframes)
        │
        ▼
[1] Visual Captioner       LLaVA-1.6 / moondream2
        │
        ▼
[2] Cross-Modal Consistency   DeBERTa-v3-small NLI  ──► ModalConflictReport
        │
        ▼
[3] Claim Decomposer        Phi-3-mini / Mistral-7B ──► ClaimDecomposition
        │
        ▼
[4] Gated Evidence Retrieval  bge-small-en-v1.5      ──► EvidenceStrengthReport
        │
        ▼
[5] Multi-Hop Reasoning     Qwen2.5-3B / Phi-3-mini  ──► HopResult[]
        │
        ▼
[6] Verdict Aggregator      Mistral-7B / LLaMA-3.1-8B ──► FinalVerdict
        │
        ▼
[7] Explainability Module   (NLI + hop LLM reused)   ──► ExplainabilityReport  ← sole output
```

## Project Structure

```
video_factcheck/
├── pipeline.py                   Entry point: run_pipeline()
├── example_usage.py              Runnable end-to-end example
├── requirements.txt
├── pyproject.toml
├── schemas/
│   └── data_models.py            All Pydantic schemas
├── models/
│   └── model_bundle.py           HuggingFace model wrappers + ModelBundle
├── modules/
│   ├── module1_claim_decomposer.py
│   ├── module2_cross_modal_consistency.py
│   ├── module3_evidence_strength.py
│   ├── module4_targeted_retrieval.py   (includes DenseRetriever)
│   ├── module5_multihop_reasoning.py
│   ├── module6_verdict_aggregator.py
│   └── module7_explainability.py
└── tests/
    ├── test_schemas.py
    └── test_modules.py
```

## Installation

```bash
pip install -r requirements.txt
```

For GPU with 4-bit quantisation (recommended for 7B models on <24 GB VRAM):
```bash
pip install bitsandbytes
```

## Quick Start

### Stub mode (no downloads, instant)
```bash
python example_usage.py --stub
```

### Full mode (downloads ~10–15 GB on first run)
```bash
python example_usage.py
```

### In code

```python
from pipeline import run_pipeline
from models import load_default_bundle
from modules import DenseRetriever
from schemas import VideoSegment, EvidenceRef

# Load all models
models = load_default_bundle(load_in_4bit=True)

# Index a passage corpus for retrieval
retriever = DenseRetriever(models.encoder)
retriever.index(passage_corpus)   # list[EvidenceRef]

# Build a segment
segment = VideoSegment(
    segment_id="seg-001",
    start_ts=0.0,
    end_ts=30.0,
    transcript="The bridge was declared safe following a 2023 inspection.",
    keyframes=["frame_00.jpg"],
)

# Run
report = run_pipeline(
    claim_text="The bridge collapsed due to neglected maintenance.",
    claim_id="claim-001",
    segment=segment,
    initial_evidence=initial_evidence,
    models=models,
    retriever=retriever,
)

print(report.verdict)            # "refuted"
print(report.confidence)         # 0.91
print(report.hop_summaries)      # ["Maintenance was not neglected.", ...]
print(report.model_dump_json(indent=2))
```

## Running Tests

```bash
# All tests (no GPU required — all models are stubbed)
pytest

# With coverage
pytest --cov
```

## Model Defaults

| Role | Default model | Size |
|---|---|---|
| Visual captioner | `llava-hf/llava-v1.6-mistral-7b-hf` | 7B |
| NLI scorer | `cross-encoder/nli-deberta-v3-small` | 184M |
| Text encoder | `BAAI/bge-small-en-v1.5` | 33M |
| Claim decomposer | `microsoft/Phi-3-mini-4k-instruct` | 3.8B |
| Per-hop reader | `Qwen/Qwen2.5-3B-Instruct` | 3B |
| Verdict aggregator | `mistralai/Mistral-7B-Instruct-v0.3` | 7B |

Override any model via `load_default_bundle(aggregator_model="meta-llama/Meta-Llama-3.1-8B-Instruct")`.

## Key Design Decisions

- **OCR removed.** Transcript + visual captions cover all text modalities.
- **Explainability is a first-class module.** `run_pipeline()` returns `ExplainabilityReport`, not `FinalVerdict`.
- **No GPU required to test.** Every test uses deterministic stubs.
- **Gate-first verdict.** If evidence never passes the quality gate after 3 retrieval rounds, `insufficient_evidence` is emitted without calling the aggregator LLM.
- **Role specialisation.** The NLI scorer drives both entailment scoring and saliency attribution. The hop LLM is reused for hop summarisation in Module 7.
