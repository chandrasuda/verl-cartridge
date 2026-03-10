#!/usr/bin/env python3
"""Standalone cartridge eval script — runs in a subprocess with GPU access.

Called by ray_trainer.py::_eval_cartridge_on_longhealth() because the Ray
TaskRunner actor has no GPU (Ray allocates it to WorkerDict).  Running as a
fresh subprocess with CUDA_VISIBLE_DEVICES=0 gives us a clean CUDA context.

Usage:
    CUDA_VISIBLE_DEVICES=0 python cartridge_eval_subprocess.py \
        --ckpt /path/to/cache-step0.pt \
        --step 0 \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --eval-json /results/onpolicy/eval_scores.json
"""

import argparse
import json
import os
import re
import time

import requests as http_req
import torch
from transformers import AutoTokenizer

from cartridges.cache import AttnConfig, TrainableCache
from cartridges.generation import flex_generate
from cartridges.models.llama.modeling_llama import FlexLlamaForCausalLM


def extract_answer(text):
    m = re.search(r"\b([A-E])\b", text.strip()[:20])
    return m.group(1) if m else "?"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to cache-stepN.pt")
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--model", required=True, help="HF model name")
    parser.add_argument("--eval-json", required=True, help="Path to write eval results")
    args = parser.parse_args()

    # ---- Load LongHealth questions ----
    data = http_req.get(
        "https://raw.githubusercontent.com/kbressem/LongHealth/"
        "refs/heads/main/data/benchmark_v5.json"
    ).json()
    questions = []
    for pid, patient in data.items():
        for q in patient["questions"]:
            options = "\n".join(
                L + ") " + q["answer_" + L.lower()] for L in "ABCDE"
            )
            prompt = (
                f"You are answering a multiple choice question about patient {patient['name']}.\n\n"
                f"Question: {q['question']}\n\nOptions:\n{options}\n\n"
                f"Answer with ONLY the letter (A, B, C, D, or E):"
            )
            answer_map = {q["answer_" + L.lower()]: L for L in "ABCDE"}
            questions.append({
                "prompt": prompt,
                "correct": answer_map.get(q["correct"], "?"),
            })
    print(f"[subprocess-eval] {len(questions)} questions loaded")

    # ---- Load model ----
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = FlexLlamaForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).cuda().eval()
    print(f"[subprocess-eval] Model loaded: {args.model}")

    # ---- Load cache checkpoint ----
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if "fixed_keys" in ckpt and "frozen_keys" not in ckpt:
        ckpt["frozen_keys"] = ckpt.pop("fixed_keys")
        ckpt["frozen_values"] = ckpt.pop("fixed_values")

    tk = ckpt["trainable_keys"]
    fk = ckpt["frozen_keys"]
    nl, nh, hd = len(tk), tk[0].size(1), tk[0].size(3)
    nf = fk[0].size(2) if fk else 0

    ik = [
        torch.cat([fk[i], tk[i]], dim=2).contiguous() if nf > 0 else tk[i]
        for i in range(nl)
    ]
    iv = [
        torch.cat([ckpt["frozen_values"][i], ckpt["trainable_values"][i]], dim=2).contiguous()
        if nf > 0 else ckpt["trainable_values"][i]
        for i in range(nl)
    ]
    cache = TrainableCache(
        config=AttnConfig(n_layers=nl, n_heads=nh, head_dim=hd),
        init_keys=ik, init_values=iv, num_frozen_tokens=nf,
    ).cuda()
    del ckpt
    print(f"[subprocess-eval] Cache loaded: {nl} layers, {nh} heads, {nf} frozen tokens")

    # ---- Run eval ----
    t0 = time.time()
    correct = 0
    for qi, q in enumerate(questions):
        ids = tokenizer.encode(q["prompt"])
        input_ids = torch.tensor(ids, dtype=torch.long, device="cuda")
        seq_ids = torch.zeros_like(input_ids)
        position_ids = torch.arange(len(ids), dtype=torch.long, device="cuda")
        cache.clear()
        gen_output = flex_generate(
            model=model, tokenizer=tokenizer, cache=cache,
            input_ids=input_ids, seq_ids=seq_ids, position_ids=position_ids,
            max_new_tokens=10, temperature=0.0,
        )
        gen_tokens = gen_output.get(0, [])
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        pred = extract_answer(gen_text)
        if qi < 5:
            print(f"  Q{qi}: gen='{gen_text[:80]}' pred={pred} correct={q['correct']}")
        if pred == q["correct"]:
            correct += 1

    total = len(questions)
    acc = correct / total * 100
    elapsed = time.time() - t0
    total_tokens = args.step * 32 * 350

    print(f"\n{'=' * 60}")
    print(f"  EVAL @ step {args.step}: {correct}/{total} ({acc:.1f}%) [{elapsed:.0f}s]")
    print(f"{'=' * 60}\n")

    # ---- Save results (read-modify-write) ----
    if os.path.exists(args.eval_json):
        with open(args.eval_json) as f:
            results = json.load(f)
    else:
        results = {"method": "on_policy", "evals": []}

    results["evals"].append({
        "optimizer_step": args.step,
        "total_tokens": total_tokens,
        "scores": {"score": round(acc / 100, 4)},
        "num_eval_questions": total,
        "correct": correct,
    })

    os.makedirs(os.path.dirname(args.eval_json), exist_ok=True)
    with open(args.eval_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[subprocess-eval] Saved to {args.eval_json}")


if __name__ == "__main__":
    main()