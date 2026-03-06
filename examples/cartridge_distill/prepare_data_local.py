#!/usr/bin/env python3
"""
Prepare on-policy training data LOCALLY from already-downloaded HF parquet files.

Groups prompts by patient_id so consecutive batches (batch_size=32) are same-patient.
This enables the teacher to look up documents once per batch instead of per sample.

Usage:
    python prepare_data_local.py

Input:
    data/hf_shards/shard_{0,1,2}/train-*.parquet  (already downloaded)

Output:
    data/on_policy/train.parquet   ~130K prompts sorted by patient_id
    data/on_policy/val.parquet     ~10K prompts
"""

import os, sys, json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import requests

sys.path.insert(0, str(Path(__file__).parents[3] / "cartridges"))
os.environ.setdefault("CARTRIDGES_DIR", str(Path(__file__).parents[3] / "cartridges"))
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/tmp")


# ---------------------------------------------------------------------------
# 1. Build patient name → patient_id mapping from LongHealth benchmark
# ---------------------------------------------------------------------------

def build_patient_name_map() -> dict[str, str]:
    """Returns {patient_name: patient_id} e.g. {'Paul Wells': 'patient_01'}"""
    print("Fetching LongHealth patient names...")
    url = "https://raw.githubusercontent.com/kbressem/LongHealth/refs/heads/main/data/benchmark_v5.json"
    data = requests.get(url, timeout=30).json()
    mapping = {patient["name"]: pid for pid, patient in data.items()}
    print(f"  {len(mapping)} patients: {list(mapping.keys())}")
    return mapping


def extract_patient_id(text: str, name_map: dict[str, str]) -> str:
    for name, pid in name_map.items():
        if name in text:
            return pid
    return "unknown"


# ---------------------------------------------------------------------------
# 2. Read all locally-cached shard parquets
# ---------------------------------------------------------------------------

SHARD_DIRS = [
    Path("data/hf_shards/shard_0"),
    Path("data/hf_shards/shard_1"),
    Path("data/hf_shards/shard_2"),
]

def read_shard(shard_dir: Path, name_map: dict) -> list[dict]:
    """Read all parquet files in a shard dir, extract user prompts."""
    rows = []
    parquet_files = sorted(shard_dir.glob("*.parquet"))
    print(f"  {shard_dir.name}: {len(parquet_files)} parquet files")

    for pf in parquet_files:
        table = pq.read_table(str(pf))
        df = table.to_pandas()

        for _, row in df.iterrows():
            messages = row["messages"]
            # messages may be a list of dicts, numpy array, or pandas object
            if messages is None:
                continue
            try:
                msg_list = list(messages)
            except Exception:
                continue
            if not msg_list:
                continue

            # Each conversation is [user, assistant]. Extract the user turn.
            user_content = None
            for msg in msg_list:
                if isinstance(msg, dict):
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                else:
                    role = getattr(msg, "role", "")
                    content = getattr(msg, "content", "")
                if role == "user" and content:
                    user_content = str(content)
                    break

            if not user_content:
                continue

            patient_id = extract_patient_id(user_content, name_map)

            rows.append({
                "prompt": [{"role": "user", "content": user_content}],
                "patient_id": patient_id,
                "data_source": "longhealth_synthesized",
                "reward_model": {"ground_truth": "", "style": "rule"},
            })

    print(f"    → {len(rows):,} prompts")
    return rows


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------

def main():
    # Resolve shard dirs relative to this script's location
    script_dir = Path(__file__).parent
    workspace = script_dir.parents[2]  # verl/examples/cartridge_distill → cartridges-workspace/
    shard_dirs = [workspace / d for d in SHARD_DIRS]

    # Check shards exist
    for d in shard_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Shard not found: {d}. Run data/download_all.py first.")

    name_map = build_patient_name_map()

    print("\nReading shards...")
    all_rows = []
    for shard_dir in shard_dirs:
        all_rows.extend(read_shard(shard_dir, name_map))

    print(f"\nTotal prompts: {len(all_rows):,}")

    df = pd.DataFrame(all_rows)

    # Report patient distribution
    counts = df["patient_id"].value_counts()
    print("\nPrompts per patient:")
    for pid, count in counts.items():
        print(f"  {pid}: {count:,}")

    # Sort by patient_id so consecutive batches of 32 are same-patient
    df = df.sort_values("patient_id").reset_index(drop=True)

    # Train/val split: last 10% (still sorted by patient, so proportional split)
    n_val = min(5000, len(df) // 10)
    train_df = df.iloc[:-n_val].copy()
    val_df = df.iloc[-n_val:].copy()

    print(f"\nTrain: {len(train_df):,} rows  |  Val: {len(val_df):,} rows")

    out_dir = workspace / "data" / "on_policy"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.parquet"
    val_path = out_dir / "val.parquet"
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    print(f"\n✓ {train_path} ({train_path.stat().st_size / 1e6:.1f} MB)")
    print(f"✓ {val_path} ({val_path.stat().st_size / 1e6:.1f} MB)")
    print("\nNext: modal volume put on-policy-data data/on_policy/ /")


if __name__ == "__main__":
    main()
