import os
import json
import re

def create_true_dataset_loader():
    with open('dataset/dataset_adapter.py', 'r', encoding='utf-8') as f:
        adapter_code = f.read()

    with open('dataset/true_dataset_loader.py', 'r', encoding='utf-8') as f:
        loader_code = f.read()

    with open('dataset/label_mapper.py', 'r', encoding='utf-8') as f:
        label_code = f.read()

    # Extract label_mapper parts
    # _RATING_TO_VERDICT
    rating_match = re.search(r'(_RATING_TO_VERDICT: dict\[str, str\] = \{.*?\})', label_code, re.DOTALL)
    rating_dict = rating_match.group(1) if rating_match else ""

    verdict_label_match = re.search(r'(VERDICT_TO_LABEL: dict\[str, int\] = \{.*?\})', label_code, re.DOTALL)
    verdict_label_dict = verdict_label_match.group(1) if verdict_label_match else ""

    label_api = """
def rating_to_verdict(rating: str) -> str:
    return _RATING_TO_VERDICT.get(rating.lower().strip(), "insufficient_evidence")

def verdict_to_label(verdict: str) -> int:
    return VERDICT_TO_LABEL.get(verdict, 1)

def rating_to_label(rating: str) -> int:
    return verdict_to_label(rating_to_verdict(rating))

LABEL_TO_VERDICT = {v: k for k, v in VERDICT_TO_LABEL.items()}
def label_to_verdict(label: int) -> str:
    return LABEL_TO_VERDICT.get(label, "false")

NUM_LABELS = len(VERDICT_TO_LABEL)
VERDICT_DISPLAY = {
    "true": "True",
    "false": "False",
}
"""

    # We need to inline _load_dir since DirectoryLoader is deleted.
    load_dir_code = """
def _load_dir(directory: str | Path, max_records: Optional[int] = None) -> list[dict]:
    import json
    records = []
    dir_path = Path(directory)
    if not dir_path.is_dir():
        return records
    count = 0
    for fp in sorted(dir_path.glob("*.json")):
        if max_records is not None and count >= max_records:
            break
        try:
            with open(fp, encoding="utf-8") as fh:
                records.append(json.load(fh))
            count += 1
        except Exception as exc:
            pass
    return records
"""

    # Convert split_records to dict
    split_code = """
def split_records(
    records: list[dict], train_frac: float = 0.8, seed: int = 42
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    n_train = int(len(shuffled) * train_frac)
    return shuffled[:n_train], shuffled[n_train:]
"""

    # Extract adapter dict builders (already changed to return dict by my prev run, except the ones taking DatasetRecord)
    # We must replace DatasetRecord -> dict, and record.field -> record['field'] etc.
    # It's actually easier to just manually write out the adapter functions here correctly.
    adapter_functions = """
def _yyyymmdd_to_iso(value: float) -> str:
    from datetime import datetime
    try:
        return datetime.strptime(str(int(value)), "%Y%m%d").date().isoformat()
    except Exception:
        return ""

def _assign_hop_ids(evidence_entries: list[dict], relationships: list[dict]) -> dict[int, list[int]]:
    hop_map = {e.get('evidence_index', 0): [] for e in evidence_entries}
    evidence_to_hop = {}
    next_hop = 1
    claim_rels = [r for r in relationships if r.get('left') == 'claim']
    other_rels = [r for r in relationships if r.get('left') != 'claim']
    for rel in claim_rels + other_rels:
        idx = rel.get('evidence_index', 0)
        if idx == 0 or idx not in hop_map: continue
        if idx not in evidence_to_hop:
            evidence_to_hop[idx] = next_hop
            next_hop += 1
        hop_id = evidence_to_hop[idx]
        if hop_id not in hop_map[idx]: hop_map[idx].append(hop_id)
    return hop_map

def record_to_segment(record: dict, keyframe_paths: Optional[list[str]] = None) -> dict:
    vi = record.get("video_information", {})
    return {
        "segment_id": vi.get("video_id", ""),
        "start_ts": 0.0,
        "end_ts": vi.get("video_length", 0.0),
        "transcript": vi.get("video_transcript", ""),
        "keyframes": keyframe_paths or [],
    }

def record_to_visual_caption(record: dict) -> str:
    vi = record.get("video_information", {})
    parts = [p.strip() for p in [vi.get("video_headline", ""), vi.get("video_description", "")] if p.strip()]
    return ". ".join(parts)

def record_to_evidence(record: dict) -> list[dict]:
    vi = record.get("video_information", {})
    iso_date = _yyyymmdd_to_iso(vi.get("video_date", 0))
    evidences = record.get("evidences", {}).get("entries", [])
    relationships = record.get("relationship_with_evidence", [])
    hop_map = _assign_hop_ids(evidences, relationships)
    res = []
    for entry in evidences:
        urls = entry.get("urls", [])
        source_url = urls[0] if urls else vi.get("video_url", "")
        idx = entry.get("evidence_index", 0)
        res.append({
            "evidence_id": f"{vi.get('video_id', '')}-ev{idx}",
            "source_url": source_url,
            "source_date": iso_date,
            "passage_text": entry.get("passage_text", ""),
            "retrieval_score": 1.0,
            "hop_ids": hop_map.get(idx, []),
        })
    return res

def record_to_rationale_context(record: dict) -> dict:
    orig = record.get("original_rationales", {})
    summ = record.get("summary_rationales", {})
    add = []
    for k in ["additional_rationale1", "additional_rationale2", "additional_rationale3"]:
        v = orig.get(k, "")
        if v.strip(): add.append(v)
    
    # all_reasons() mapped to a list of dicts/strings
    # we just extract reasons
    detailed = summ.get("reasons", [])
    if not detailed and "all_reasons" in summ:
        if callable(summ["all_reasons"]): detailed = summ["all_reasons"]()

    return {
        "main_rationale": orig.get("main_rationale", ""),
        "additional_rationales": add,
        "synthesized_rationale": summ.get("synthesized_rationale", ""),
        "detailed_reasons": detailed,
        "gold_verdict": rating_to_verdict(record.get("rating", "")),
        "snopes_rating": record.get("rating", ""),
        "snopes_url": record.get("url", ""),
        "article_content": record.get("content", ""),
    }

def record_to_pipeline_inputs(record: dict, keyframe_paths: Optional[list[str]] = None) -> dict:
    vi = record.get("video_information", {})
    raw_id = f"{vi.get('video_id', '')}_{record.get('claim', '')[:40]}"
    claim_id = re.sub(r"[^a-zA-Z0-9_-]", "_", raw_id)[:64]
    return {
        "claim_text": record.get("claim", ""),
        "claim_id": claim_id,
        "segment": record_to_segment(record, keyframe_paths),
        "visual_caption": record_to_visual_caption(record),
        "initial_evidence": record_to_evidence(record),
        "rationale_context": record_to_rationale_context(record),
        "gold_verdict": rating_to_verdict(record.get("rating", "")),
        "gold_label": rating_to_label(record.get("rating", "")),
    }
"""

    true_ds = loader_code
    
    # Fix imports
    true_ds = re.sub(r'from dataset\.dataset_loader import DirectoryLoader\n', '', true_ds)
    true_ds = re.sub(r'from dataset\.dataset_schemas import DatasetRecord\n', '', true_ds)
    
    # Replace DatasetRecord occurrences
    true_ds = true_ds.replace("DatasetRecord", "dict")
    true_ds = true_ds.replace('record.video_information.video_id', 'record.get("video_information", {}).get("video_id")')

    # Remove _load_dir from true_dataset_loader to replace it
    true_ds = re.sub(r'def _load_dir.*?return loader\.load_all\(\)\n', '', true_ds, flags=re.DOTALL)
    
    # Remove split_records from true_dataset_loader
    true_ds = re.sub(r'def split_records.*?logger\.info\([^)]+\)\n    return train, val\n', '', true_ds, flags=re.DOTALL)

    # Insert our new code BEFORE the helpers
    insert_idx = true_ds.find('# ---------------------------------------------------------------------------')

    new_content = (
        true_ds[:insert_idx] +
        "\n# --- Label Dictionary ---\n" + rating_dict + "\n\n" + verdict_label_dict + "\n" + label_api + "\n" +
        adapter_functions + "\n" + load_dir_code + "\n" + split_code + "\n" +
        true_ds[insert_idx:]
    )

    with open('dataset/true_dataset_loader.py', 'w', encoding='utf-8') as f:
        f.write(new_content)

if __name__ == '__main__':
    create_true_dataset_loader()
