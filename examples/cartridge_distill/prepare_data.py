#!/usr/bin/env python3
"""
Prepare on-policy training data from the SAME prompts the off-policy baseline uses.

Extracts the user-turn prompts from the paper's 196K synthesized HuggingFace data
and formats them as veRL parquet files. This ensures both methods train on the
exact same question distribution — the only difference is who generates the answers.

Usage:
    # Requires CARTRIDGES_DIR and CARTRIDGES_OUTPUT_DIR set, or:
    CARTRIDGES_DIR=/opt/cartridges CARTRIDGES_OUTPUT_DIR=/tmp \
        python examples/cartridge_distill/prepare_data.py

Output:
    ~/data/cartridge_distill/train.parquet   (from shards 0+1, ~130K prompts)
    ~/data/cartridge_distill/val.parquet     (from shard 2, ~65K prompts, or small subset)
"""

import json
import os
import re
from pathlib import Path

import pandas as pd


# ---------- patient name → patient_id mapping (for teacher document lookup) ----------
PATIENT_NAMES = {}  # filled lazily


def _load_patient_mapping():
    """Build patient_name → patient_id mapping from LongHealth benchmark."""
    global PATIENT_NAMES
    if PATIENT_NAMES:
        return
    import requests

    url = "https://raw.githubusercontent.com/kbressem/LongHealth/refs/heads/main/data/benchmark_v5.json"
    print(f"Downloading LongHealth metadata from {url}...")
    data = requests.get(url, timeout=30).json()
    for pid, patient in data.items():
        PATIENT_NAMES[patient["name"]] = pid
    print(f"Loaded {len(PATIENT_NAMES)} patient names")


def _extract_patient_id_from_prompt(prompt_text: str) -> str:
    """Try to match a patient name in the prompt to get the patient_id."""
    _load_patient_mapping()
    for name, pid in PATIENT_NAMES.items():
        if name in prompt_text:
            return pid
    return "unknown"


# ---------- HF data extraction ----------

def extract_prompts_from_hf_shard(repo_id: str, limit: int = None) -> list[dict]:
    """Download one HF shard and extract user-turn prompts.

    Each conversation in the paper's data is [user, assistant].
    We take the user message as the prompt for on-policy rollout.
    """
    # Use datasets + huggingface_hub to download
    from huggingface_hub import HfApi
    import pyarrow.parquet as pq
    import io
    import requests

    api = HfApi()
    repo_files = api.list_repo_files(repo_id, repo_type="dataset")
    parquet_files = sorted(f for f in repo_files if f.startswith("data/") and f.endswith(".parquet"))

    if not parquet_files:
        raise ValueError(f"No parquet files in {repo_id}")

    prompts = []
    for pf in parquet_files:
        url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{pf}"
        print(f"  Downloading {pf}...")
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        table = pq.read_table(io.BytesIO(resp.content))
        df = table.to_pandas()

        for _, row in df.iterrows():
            messages = row.get("messages", None)
            if messages is None or len(messages) == 0:
                continue

            # Find the first user message
            user_content = None
            for msg in messages:
                role = msg["role"] if isinstance(msg, dict) else ""
                if role == "user":
                    user_content = msg["content"] if isinstance(msg, dict) else ""
                    break

            if not user_content:
                continue

            patient_id = _extract_patient_id_from_prompt(user_content)

            prompts.append({
                "prompt": [{"role": "user", "content": user_content}],
                "data_source": "longhealth_synthesized",
                "patient_id": patient_id,
                # Dummy reward — distillation uses KL loss, not reward
                "reward_model": {"ground_truth": "", "style": "rule"},
            })

            if limit and len(prompts) >= limit:
                return prompts

    return prompts


def main():
    # The paper's three HF shards (196K total conversations)
    hf_shards = [
        "hazyresearch/m07d11_longhealth_synthesize_llama-3.2-3b_p10_n65536-0",
        "hazyresearch/m07d11_longhealth_synthesize_llama-3.2-3b_p10_n65536-1",
        "hazyresearch/m07d11_longhealth_synthesize_llama-3.2-3b_p10_n65536-2",
    ]

    # Extract prompts from shards 0+1 for training, shard 2 for val
    train_prompts = []
    for shard in hf_shards[:2]:
        print(f"\nExtracting prompts from {shard}...")
        train_prompts.extend(extract_prompts_from_hf_shard(shard))
    print(f"\nTotal train prompts: {len(train_prompts)}")

    print(f"\nExtracting prompts from {hf_shards[2]}...")
    val_prompts = extract_prompts_from_hf_shard(hf_shards[2], limit=500)
    print(f"Val prompts: {len(val_prompts)}")

    out_dir = Path(os.environ.get("DATA_DIR", str(Path.home() / "data"))) / "cartridge_distill"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.DataFrame(train_prompts)
    val_df = pd.DataFrame(val_prompts)

    train_path = out_dir / "train.parquet"
    val_path = out_dir / "val.parquet"

    train_df.to_parquet(train_path)
    val_df.to_parquet(val_path)

    print(f"\n✓ Saved {train_path} ({len(train_df)} rows)")
    print(f"✓ Saved {val_path} ({len(val_df)} rows)")
    print(f"\nSample prompt:\n{train_df.iloc[0]['prompt']}")


if __name__ == "__main__":
    main()
