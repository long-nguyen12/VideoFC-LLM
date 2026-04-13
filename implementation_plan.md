# Remove Pydantic Schemas & Simplify Dataset

Two goals:
1. **Replace all Pydantic schemas** (`schemas/data_models.py`) with plain `dict`s across the entire pipeline
2. **Consolidate the `dataset/` directory** using `true_dataset_loader.py` as the single source of truth, with its correct 8-class label taxonomy

---

## Current dataset/ structure (6 files, lots of redundancy)

| File | Lines | Purpose | Problem |
|---|---|---|---|
| `dataset_schemas.py` | 284 | Pydantic models for raw JSON parsing | Unnecessary — raw JSON dicts work fine |
| `dataset_loader.py` | 366 | File/directory iterator → `DatasetRecord` | Can be simplified to yield raw dicts |
| `dataset_adapter.py` | 349 | `DatasetRecord` → pipeline Pydantic types | Entire file eliminated (modules take dicts) |
| `label_mapper.py` | 93 | Binary label mapping | Inconsistent — missing `miscaptioned`, `fake` |
| `true_dataset_loader.py` | 412 | Correct 8-class taxonomy + loading | **Keep as canonical source** |
| `evaluation.py` | 199 | Metrics computation | Keep (already uses plain dataclasses) |

## Target dataset/ structure (3 files)

| File | Purpose |
|---|---|
| `true_dataset_loader.py` | Loading, label taxonomy, keyframe resolution, pipeline bridge |
| `evaluation.py` | Metrics computation (unchanged) |
| `__init__.py` | Re-exports from the two above |

> [!IMPORTANT]
> The correct label taxonomy lives in `true_dataset_loader.py` lines 54–75. This is the **only** label mapping that survives. `label_mapper.py` is deleted.

## Label taxonomy (from `true_dataset_loader.py` — the source of truth)

```python
RATING_TO_FINE = {
    "true":                 (0, 0),   # flat=0
    "mostly true":          (0, 1),   # flat=1
    "correct attribution":  (0, 2),   # flat=2
    "false":                (1, 0),   # flat=3
    "mostly false":         (1, 1),   # flat=4
    "mixture":              (1, 2),   # flat=5
    "fake":                 (1, 3),   # flat=6
    "miscaptioned":         (1, 4),   # flat=7
}
```

---

## Proposed Changes

### schemas/ package

#### [DELETE] [data_models.py](file:///c:/Users/hnguyen/Documents/PhD/Code/VideoFC-LLM/schemas/data_models.py)
All Pydantic schemas removed. Modules pass/receive plain `dict`s.

#### [MODIFY] [__init__.py](file:///c:/Users/hnguyen/Documents/PhD/Code/VideoFC-LLM/schemas/__init__.py)
Empty the file — nothing to export.

---

### dataset/ package

#### [DELETE] [dataset_schemas.py](file:///c:/Users/hnguyen/Documents/PhD/Code/VideoFC-LLM/dataset/dataset_schemas.py)
`DatasetRecord` and all sub-models removed. Raw JSON dicts are used directly.

#### [DELETE] [dataset_adapter.py](file:///c:/Users/hnguyen/Documents/PhD/Code/VideoFC-LLM/dataset/dataset_adapter.py)
The entire adapter layer (`record_to_segment`, `record_to_evidence`, `PipelineInputs`, `RationaleContext`) is eliminated. Its conversion logic moves into `true_dataset_loader.py` as simple dict-building helper functions.

#### [DELETE] [label_mapper.py](file:///c:/Users/hnguyen/Documents/PhD/Code/VideoFC-LLM/dataset/label_mapper.py)
Replaced by `true_dataset_loader.py`'s `RATING_TO_FINE` / `RATING_TO_FLAT_FINE` / `rating_to_binary()`. 

#### [DELETE] [dataset_loader.py](file:///c:/Users/hnguyen/Documents/PhD/Code/VideoFC-LLM/dataset/dataset_loader.py)
`DirectoryLoader` and `load_split` are already used by `true_dataset_loader.py`. The loading logic will be inlined into `true_dataset_loader.py` (simplified to yield raw dicts instead of `DatasetRecord` objects).

#### [MODIFY] [true_dataset_loader.py](file:///c:/Users/hnguyen/Documents/PhD/Code/VideoFC-LLM/dataset/true_dataset_loader.py)
Becomes the **single dataset file**. Changes:

1. **Remove** `from dataset.dataset_loader import DirectoryLoader` and `from dataset.dataset_schemas import DatasetRecord` — inline the directory-loading logic to yield raw `dict`s from JSON files
2. **Add** label-mapping functions absorbed from `label_mapper.py`:
   - `rating_to_verdict(rating) → str` (binary: "true"/"false")
   - `verdict_to_label(verdict) → int` (0 or 1)
   - `rating_to_label(rating) → int`
   - `label_to_verdict(label) → str`
   - Keep existing `RATING_TO_FINE`, `RATING_TO_FLAT_FINE`, `rating_to_binary()`
3. **Add** dict-building helpers absorbed from `dataset_adapter.py`:
   - `record_to_segment(raw_dict, keyframe_paths) → dict` — builds a segment dict from raw JSON
   - `record_to_evidence(raw_dict) → list[dict]` — builds evidence dicts from raw JSON
   - `record_to_visual_caption(raw_dict) → str`
   - `record_to_rationale_context(raw_dict) → dict`
   - `record_to_pipeline_inputs(raw_dict, keyframe_paths) → dict` — combined convenience
4. **Update** `load_for_pipeline()` to return `list[tuple[dict, list[str]]]` instead of `list[tuple[DatasetRecord, list[str]]]`
5. **Update** `run_pipeline_evaluation()` to work with raw dicts

#### [MODIFY] [evaluation.py](file:///c:/Users/hnguyen/Documents/PhD/Code/VideoFC-LLM/dataset/evaluation.py)
- Update imports: get label constants from `true_dataset_loader` instead of deleted `label_mapper`

#### [MODIFY] [__init__.py](file:///c:/Users/hnguyen/Documents/PhD/Code/VideoFC-LLM/dataset/__init__.py)
- Remove imports from deleted files
- Re-export from `true_dataset_loader` and `evaluation` only

---

### Modules (all 7 — same transformation as before)

#### [MODIFY] [module1_claim_decomposer.py](file:///c:/Users/hnguyen/Documents/PhD/Code/VideoFC-LLM/modules/module1_claim_decomposer.py)
- Remove `from schemas import ClaimDecomposition, SubQuestion, VideoSegment`
- All Pydantic types → `dict`, attribute access → key access

#### [MODIFY] [module2_cross_modal_consistency.py](file:///c:/Users/hnguyen/Documents/PhD/Code/VideoFC-LLM/modules/module2_cross_modal_consistency.py)
- Remove `from schemas import ModalConflictReport`
- Return `dict` instead of `ModalConflictReport`

#### [MODIFY] [module3_evidence_strength.py](file:///c:/Users/hnguyen/Documents/PhD/Code/VideoFC-LLM/modules/module3_evidence_strength.py)
- Remove schema imports, all typed params → `dict`

#### [MODIFY] [module4_targeted_retrieval.py](file:///c:/Users/hnguyen/Documents/PhD/Code/VideoFC-LLM/modules/module4_targeted_retrieval.py)
- Remove schema imports
- `DenseRetriever._passages: list[dict]`, all methods use dict access

#### [MODIFY] [module5_multihop_reasoning.py](file:///c:/Users/hnguyen/Documents/PhD/Code/VideoFC-LLM/modules/module5_multihop_reasoning.py)
- Remove schema imports, all typed params → `dict`, attribute access → key access

#### [MODIFY] [module6_verdict_aggregator.py](file:///c:/Users/hnguyen/Documents/PhD/Code/VideoFC-LLM/modules/module6_verdict_aggregator.py)
- Remove schema imports, all constructors → dict literals

#### [MODIFY] [module7_explainability.py](file:///c:/Users/hnguyen/Documents/PhD/Code/VideoFC-LLM/modules/module7_explainability.py)
- Remove schema imports, all constructors → dict literals

---

### Pipeline orchestrators

#### [MODIFY] [pipeline.py](file:///c:/Users/hnguyen/Documents/PhD/Code/VideoFC-LLM/pipeline.py)
- Remove `from schemas import ...`
- All typed params → `dict`

#### [MODIFY] [run_pipeline.py](file:///c:/Users/hnguyen/Documents/PhD/Code/VideoFC-LLM/run_pipeline.py)
- Remove schema and adapter imports
- Import dict-building helpers from `true_dataset_loader` instead
- `DatasetEvalResult.report` type → `dict`

---

## Key transformation rules

| Before | After |
|---|---|
| `segment.transcript` | `segment["transcript"]` |
| `VideoSegment(segment_id=..., ...)` | `{"segment_id": ..., ...}` |
| `from schemas import VideoSegment` | *(removed)* |
| `DatasetRecord.from_dict(raw)` | `raw` (use the JSON dict directly) |
| `from dataset.label_mapper import ...` | `from dataset.true_dataset_loader import ...` |
| `record.video_information.video_id` | `raw["video_information"]["video_id"]` |
| `report.model_dump_json()` | `json.dumps(report, indent=2)` |

> [!WARNING]
> This removes all Pydantic runtime validation. Missing keys or wrong types will only surface at access time. Acceptable for research code.

---

## Open Questions

> [!IMPORTANT]
> The `evaluation.py` currently imports `LABEL_TO_VERDICT`, `NUM_LABELS`, `VERDICT_DISPLAY` from `label_mapper.py`. After deletion, these constants need to move into `true_dataset_loader.py`. Should we keep the binary-only evaluation (2 classes: true/false) or switch to the full 8-class evaluation using `RATING_TO_FLAT_FINE`?

---

## Verification Plan

### Automated Tests
- `python -c "from modules import *; from pipeline import run_pipeline"` — verify imports resolve
- `python -c "from dataset.true_dataset_loader import *"` — verify consolidated dataset module loads
- `python -m pytest tests/ -x` — run existing test suite (tests will need updates for dict access)

### Manual Verification
- Spot-check that dict keys match old Pydantic field names in each changed file
