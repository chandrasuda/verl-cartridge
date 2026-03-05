"""
Evaluate on-policy cartridge checkpoints on LongHealth while training runs.

Loads each cache-step*.pt from the results volume, runs FlexLlama + cartridge
locally on a GPU, and builds an accuracy-vs-step curve.

Usage:
    # One-shot: eval all checkpoints found so far
    modal run examples/cartridge_distill/eval_checkpoints.py

    # Poll mode: keep checking for new checkpoints every 5 min
    modal run examples/cartridge_distill/eval_checkpoints.py --poll
"""

import modal

NUM_EVAL_QUESTIONS = 40  # 40 questions from patients 1-8

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install("git")
    .env({"CUDA_HOME": "/usr/local/cuda", "CARTRIDGES_DIR": "/opt/cartridges", "CARTRIDGES_OUTPUT_DIR": "/tmp"})
    .pip_install("torch==2.6.0", "packaging", "numpy", "requests")
    .run_commands(
        "pip install https://github.com/Dao-AILab/flash-attention/releases/download/"
        "v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl",
    )
    .run_commands(
        "git clone https://github.com/HazyResearch/cartridges.git /opt/cartridges "
        "&& pip install -e /opt/cartridges"
    )
    .pip_install("transformers==4.53.0")
)

results_volume = modal.Volume.from_name("onpolicy-results", create_if_missing=True)
app = modal.App("cartridge-eval", image=image)


@app.function(
    gpu="A10G",
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=14400,  # 4 hours
    volumes={"/results": results_volume},
    min_containers=0,
    max_containers=1,
)
def evaluate(poll: bool = False):
    """Evaluate all cartridge checkpoints on LongHealth."""
    import json
    import os
    import re
    import time
    import glob
    import torch
    import requests as http_req

    # --- Load LongHealth ---
    print("Loading LongHealth benchmark...")
    data = http_req.get(
        "https://raw.githubusercontent.com/kbressem/LongHealth/refs/heads/main/data/benchmark_v5.json"
    ).json()

    questions = []
    for pid, patient in data.items():
        if int(pid.split("_")[1]) > 8:
            continue
        # Collect documents for this patient
        doc_text = "\n\n".join(
            f"--- {did} ---\n{txt}" for did, txt in patient["texts"].items()
        )
        for q in patient["questions"]:
            options = "\n".join(
                f"{letter}) {q[f'answer_{letter.lower()}']}" for letter in "ABCDE"
            )
            prompt = (
                f"You are answering a multiple choice question about patient {patient['name']}.\n\n"
                f"Question: {q['question']}\n\nOptions:\n{options}\n\n"
                f"Answer with ONLY the letter (A, B, C, D, or E):"
            )
            answer_map = {
                q["answer_a"]: "A", q["answer_b"]: "B", q["answer_c"]: "C",
                q["answer_d"]: "D", q["answer_e"]: "E",
            }
            questions.append({
                "prompt": prompt,
                "correct": answer_map.get(q["correct"], "?"),
                "patient_id": pid,
                "doc_text": doc_text,
            })

    eval_qs = questions[:NUM_EVAL_QUESTIONS]
    print(f"Evaluating {len(eval_qs)} questions per checkpoint\n")

    # --- Load model + tokenizer once ---
    print("Loading model...")
    from transformers import AutoTokenizer
    from cartridges.models.llama.modeling_llama import FlexLlamaForCausalLM
    from cartridges.cache import TrainableCache
    from cartridges.generation import flex_generate

    MODEL = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = FlexLlamaForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda().eval()
    print(f"✓ Model loaded: {MODEL}\n")

    def extract_answer(text):
        match = re.search(r'\b([A-E])\b', text.strip()[:20])
        return match.group(1) if match else "?"

    def eval_with_cartridge(cache, label=""):
        """Evaluate using FlexLlama + cartridge (same as off-policy eval)."""
        correct = 0
        total = 0
        for q in eval_qs:
            try:
                input_ids = tokenizer.encode(q["prompt"], return_tensors="pt").cuda()
                cache.clear()
                output_ids = flex_generate(
                    model=model,
                    tokenizer=tokenizer,
                    cache=cache,
                    input_ids=input_ids,
                    max_new_tokens=10,
                    temperature=0.0,
                )
                answer_text = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
                predicted = extract_answer(answer_text)
                if predicted == q["correct"]:
                    correct += 1
                total += 1
            except Exception as e:
                if total < 3:
                    print(f"  ⚠ {e}")
                total += 1

        acc = correct / total * 100 if total > 0 else 0
        print(f"  {label}: {correct}/{total} ({acc:.1f}%)")
        return {"correct": correct, "total": total, "accuracy": acc}

    def eval_baseline():
        """Evaluate without cartridge (no document context)."""
        correct = 0
        total = 0
        for q in eval_qs:
            try:
                input_ids = tokenizer.encode(q["prompt"], return_tensors="pt").cuda()
                with torch.no_grad():
                    out = model.generate(input_ids, max_new_tokens=10, do_sample=False)
                answer_text = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
                predicted = extract_answer(answer_text)
                if predicted == q["correct"]:
                    correct += 1
                total += 1
            except Exception as e:
                if total < 3:
                    print(f"  ⚠ {e}")
                total += 1

        acc = correct / total * 100 if total > 0 else 0
        print(f"  Baseline: {correct}/{total} ({acc:.1f}%)")
        return {"correct": correct, "total": total, "accuracy": acc}

    # --- Load existing results ---
    eval_path = "/results/onpolicy_eval.json"
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            results = json.load(f)
    else:
        results = {"evals": [], "evaluated_steps": []}

    evaluated = set(results.get("evaluated_steps", []))

    def save_results():
        results["evaluated_steps"] = sorted(list(evaluated))
        with open(eval_path, "w") as f:
            json.dump(results, f, indent=2)
        results_volume.commit()

    # --- Eval baseline once ---
    if "baseline" not in results:
        print("=== Baseline (no cartridge) ===")
        results["baseline"] = eval_baseline()
        save_results()

    # --- Scan and eval checkpoints ---
    def scan_and_eval():
        ckpt_dir = "/results/onpolicy/cartridge_checkpoints"
        if not os.path.exists(ckpt_dir):
            print(f"No checkpoint dir yet: {ckpt_dir}")
            return False

        ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "cache-step*.pt")))
        new_found = False

        for ckpt_path in ckpts:
            match = re.search(r"cache-step(\d+)\.pt", ckpt_path)
            if not match:
                continue
            step = int(match.group(1))
            if step in evaluated:
                continue

            new_found = True
            print(f"\n=== Step {step}: {os.path.basename(ckpt_path)} ===")

            try:
                # Fix key naming if needed
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                if "fixed_keys" in ckpt and "frozen_keys" not in ckpt:
                    ckpt["frozen_keys"] = ckpt.pop("fixed_keys")
                    ckpt["frozen_values"] = ckpt.pop("fixed_values")
                    torch.save(ckpt, ckpt_path)
                del ckpt

                cache = TrainableCache.from_pretrained(ckpt_path).cuda()
                print(f"  Loaded cache: {cache._num_trainable_tokens} trainable tokens")

                result = eval_with_cartridge(cache, label=f"Step {step}")
                result["step"] = step
                results["evals"].append(result)
                evaluated.add(step)
                save_results()

                del cache
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"  ✗ Failed to eval step {step}: {e}")

        return new_found

    # First scan
    results_volume.reload()
    scan_and_eval()
    save_results()

    if poll:
        print("\n--- Poll mode: checking for new checkpoints every 5 min ---")
        while True:
            time.sleep(300)
            results_volume.reload()
            if scan_and_eval():
                save_results()
            else:
                print(f"  No new checkpoints at {time.strftime('%H:%M:%S')}")

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("EVAL SUMMARY")
    print("=" * 60)
    if "baseline" in results:
        print(f"  Baseline:    {results['baseline']['accuracy']:.1f}%")
    for e in sorted(results.get("evals", []), key=lambda x: x["step"]):
        print(f"  Step {e['step']:>4}:   {e['accuracy']:.1f}%")
    print("=" * 60)


@app.local_entrypoint()
def main(poll: bool = False):
    evaluate.remote(poll=poll)
