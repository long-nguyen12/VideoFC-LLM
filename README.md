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

### Single-LLM mode (hardware-constrained, ~3.5 GB VRAM)
```bash
python example_usage.py --single-llm
```

### Full multi-model mode (downloads ~10–15 GB on first run)
```bash
python example_usage.py
```

### In code — single-LLM bundle

```python
from models import load_single_llm_bundle
from modules import DenseRetriever
from dataset import record_to_pipeline_inputs, run_dataset_pipeline
from dataset.dataset_loader import DatasetLoader

models = load_single_llm_bundle(
    llm_model="Qwen/Qwen2.5-1.5B-Instruct",  # ~3.5 GB
    captioner_model=None,   # skip VLM when no keyframes
    load_in_4bit=False,
    context_window=2048,
)
retriever = DenseRetriever(models.encoder)
retriever.index(initial_evidence)

inputs = record_to_pipeline_inputs(record)
report = run_dataset_pipeline(
    inputs=inputs,
    models=models,
    retriever=retriever,
    use_rationale_hints=True,
    max_sub_questions=3,    # keep hops short for 1.5B model
)
print(report.verdict, report.confidence)
```

### In code — full multi-model bundle

```python
from pipeline import run_pipeline
from models import load_default_bundle
from modules import DenseRetriever
from schemas import VideoSegment, EvidenceRef

models = load_default_bundle(load_in_4bit=True)
retriever = DenseRetriever(models.encoder)
retriever.index(passage_corpus)

segment = VideoSegment(
    segment_id="seg-001", start_ts=0.0, end_ts=30.0,
    transcript="The bridge was declared safe in 2023.", keyframes=["frame_00.jpg"],
)
report = run_pipeline(
    claim_text="The bridge collapsed due to neglected maintenance.",
    claim_id="claim-001", segment=segment,
    initial_evidence=initial_evidence, models=models, retriever=retriever,
)
print(report.verdict)
```

## Running Tests

```bash
# All tests (no GPU required — all models are stubbed)
pytest

# With coverage
pytest --cov
```

## VRAM Budget Reference

| Configuration | LLM | VLM | NLI + Encoder | Total |
|---|---|---|---|---|
| **Single-LLM, no VLM** | Qwen2.5-1.5B (~3.5 GB) | — | ~0.5 GB | **~4 GB** |
| **Single-LLM, 4-bit, no VLM** | Qwen2.5-1.5B 4-bit (~1.8 GB) | — | ~0.5 GB | **~2.5 GB** |
| **Single-LLM + moondream2** | Qwen2.5-1.5B (~3.5 GB) | moondream2 (~3.5 GB) | ~0.5 GB | **~7.5 GB** |
| **Single-LLM 3B** | Qwen2.5-3B (~6 GB) | — | ~0.5 GB | **~6.5 GB** |
| **Full multi-model** | Phi-3-mini + Qwen2.5-3B + Mistral-7B | LLaVA-1.6-7B | ~0.5 GB | **~22 GB** |
| **Full multi-model, 4-bit** | same, quantised | same, quantised | ~0.5 GB | **~12 GB** |

`load_in_4bit=True` requires `bitsandbytes>=0.43` and CUDA.  
On CPU, use `context_window=2048` to avoid OOM during tokenisation.

## Model Defaults

### Full bundle (`load_default_bundle`)

| Role | Default model | Size |
|---|---|---|
| Visual captioner | `llava-hf/llava-v1.6-mistral-7b-hf` | 7B |
| NLI scorer | `cross-encoder/nli-deberta-v3-small` | 184M |
| Text encoder | `BAAI/bge-small-en-v1.5` | 33M |
| Claim decomposer | `microsoft/Phi-3-mini-4k-instruct` | 3.8B |
| Per-hop reader | `Qwen/Qwen2.5-3B-Instruct` | 3B |
| Verdict aggregator | `mistralai/Mistral-7B-Instruct-v0.3` | 7B |

### Single-model bundle (`load_single_llm_bundle`)

| Role | Default model | Size |
|---|---|---|
| Visual captioner | `vikhyatk/moondream2` (or `None`) | 1.8B |
| NLI scorer | `cross-encoder/nli-deberta-v3-small` | 184M |
| Text encoder | `BAAI/bge-small-en-v1.5` | 33M |
| All three LLM roles | `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B |

Override with: `load_single_llm_bundle(llm_model="Qwen/Qwen2.5-3B-Instruct")`

## Key Design Decisions

- **OCR removed.** Transcript + visual captions cover all text modalities.
- **Explainability is a first-class module.** `run_pipeline()` returns `ExplainabilityReport`, not `FinalVerdict`.
- **No GPU required to test.** Every test uses deterministic stubs.
- **Gate-first verdict.** If evidence never passes the quality gate after 3 retrieval rounds, `insufficient_evidence` is emitted without calling the aggregator LLM.
- **Role specialisation.** The NLI scorer drives both entailment scoring and saliency attribution. The hop LLM is reused for hop summarisation in Module 7.
