#!/usr/bin/env python3
"""
Test the teacher logprob pipeline locally.
Debugs: patient name matching, document lookup, Tokasaurus teacher call.

Usage:
    python test_teacher_locally.py
"""

import json
import base64
import requests
import numpy as np
from transformers import AutoTokenizer

TOKASAURUS_URL = "https://kiran1234c--tokasaurus-cartridge-server-serve.modal.run"
MODEL = "meta-llama/Llama-3.2-3B-Instruct"

# Step 1: Load patient data
print("1. Loading LongHealth data...")
data = requests.get(
    "https://raw.githubusercontent.com/kbressem/LongHealth/refs/heads/main/data/benchmark_v5.json"
).json()

patient_docs = {}
patient_names = {}
for pid, patient in data.items():
    patient_docs[pid] = "\n\n".join(
        f"--- {doc_id} ---\n{text}" for doc_id, text in patient["texts"].items()
    )
    patient_names[pid] = patient["name"]

print(f"   Loaded {len(patient_docs)} patients:")
for pid, name in patient_names.items():
    doc_len = len(patient_docs[pid])
    print(f"   {pid}: {name} ({doc_len} chars)")

# Step 2: Test patient name matching against actual prompts from our data
print("\n2. Testing patient name matching against real prompts...")

# Load actual prompts from our training data
import pandas as pd
from pathlib import Path
train_path = Path.home() / "data" / "cartridge_distill" / "train.parquet"
if train_path.exists():
    df = pd.read_parquet(train_path)
    print(f"   Loaded {len(df)} training prompts")
    
    matched_count = 0
    unmatched_prompts = []
    for idx, row in df.head(20).iterrows():
        prompt_text = str(row["prompt"])  # The raw prompt before chat template
        
        found = None
        for pid, pname in patient_names.items():
            if pname and pname in prompt_text:
                found = (pid, pname)
                break
        
        if found:
            matched_count += 1
        else:
            unmatched_prompts.append(prompt_text[:80])
    
    print(f"   Matched: {matched_count}/20")
    if unmatched_prompts:
        print(f"   Unmatched samples:")
        for p in unmatched_prompts[:3]:
            print(f"     {p}...")
else:
    print(f"   ⚠ No training data at {train_path}. Run prepare_data.py first.")
    # Test with synthetic prompts
    for pid in list(patient_names.keys())[:5]:
        prompt = f"You are answering questions about a patient. Patient: {patient_names[pid]}, DOB: 1970-01-01."
        found = None
        for test_pid, pname in patient_names.items():
            if pname and pname in prompt:
                found = (test_pid, pname)
                break
        status = f"✓ {found[1]}" if found else "✗ NO MATCH"
        print(f"   {pid}: {status}")

# Step 3: Test what veRL's decoded prompt looks like
# In the training loop, we decode valid_ids[:300] and search for patient names.
# But valid_ids come from tokenizer.apply_chat_template() which wraps the prompt.
# The key question: does the patient name survive in the first 300 tokens after decoding?
print("\n3. Simulating veRL's decode + match (no tokenizer needed)...")
# In reality, decoded text looks like:
# "system\n\nCutting Knowledge Date: December 2023\n\n...user\n\nYou are answering questions about a patient. Patient: Anna Sample, DOB:..."
# The patient name appears after the system prompt header (~50 tokens)
# Our code decodes 300 tokens which is ~200 chars — should include the patient name

# The REAL issue: in the training loop, valid_ids = input_ids[attention_mask.bool()]
# This gives us [prompt_ids + response_ids]. When decoded, it should contain "Patient: Anna Sample"
# Let's verify with a simulated decode:
for pid in list(patient_names.keys())[:5]:
    # This is what the decoded text should look like (chat template adds headers)
    simulated_decode = f"system\n\nCutting Knowledge Date: December 2023\n\nuser\n\nYou are answering questions about a patient. Patient: {patient_names[pid]}, DOB: 1970-01-01"
    found = None
    for test_pid, pname in patient_names.items():
        if pname and pname in simulated_decode:
            found = (test_pid, pname)
            break
    status = f"✓ {found[1]}" if found else "✗ NO MATCH"
    print(f"   {pid}: {status}")

# Step 4: Test teacher logprob call (skip tokenizer, use raw token IDs)
print("\n4. Testing teacher logprob call to Tokasaurus...")
test_pid = list(patient_names.keys())[0]
test_doc = patient_docs[test_pid]
# Use simple text prompt as token IDs (not ideal but tests the HTTP path)
test_prompt = f"What treatment did {patient_names[test_pid]} receive?"
test_response = "The patient received chemotherapy."

# Test TEXT teacher call (no tokenizer needed — Tokasaurus tokenizes internally)
full_text = test_doc[:5000] + "\n\nQuestion: What treatment?\n\nAnswer: The patient received chemotherapy."
payload = {
    "model": "default",
    "prompt": full_text,
    "max_tokens": 1,
    "temperature": 0.0,
    "logprobs_in_fingerprint": True,
}
print(f"   Sending {len(full_text)} char text prompt...")
try:
    resp = requests.post(f"{TOKASAURUS_URL}/custom/cartridge/completions", json=payload, timeout=300)
    print(f"   HTTP {resp.status_code}")
    if resp.status_code == 200:
        result = resp.json()
        fp = json.loads(result.get("system_fingerprint", "{}"))
        packed = fp.get("packed_chosen_logprobs")
        if packed and packed[0]:
            all_lp = np.frombuffer(base64.b64decode(packed[0]), dtype=np.float32)
            print(f"   ✓ Got {len(all_lp)} logprobs, last 5: {all_lp[-5:]}")
        else:
            print(f"   ✗ No logprobs! FP keys: {list(fp.keys())}")
    else:
        print(f"   ✗ Error: {resp.text[:300]}")
except Exception as e:
    print(f"   ✗ {e}")

print("\nDone!")
