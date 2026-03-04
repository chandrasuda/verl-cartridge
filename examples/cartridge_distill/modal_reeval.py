"""
Re-evaluate saved cartridge checkpoints on LongHealth.
Downloads checkpoints from the Modal volume, loads the model once,
then runs eval for each checkpoint step.
"""
import modal
import json
import time

# ---------------------------------------------------------------------------
# Modal setup — same image as training
# ---------------------------------------------------------------------------
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install("git")
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "CARTRIDGES_DIR": "/opt/cartridges",
        "CARTRIDGES_OUTPUT_DIR": "/results",
    })
    .pip_install("torch==2.6.0", "packaging", "numpy")
    .run_commands(
        "pip install https://github.com/Dao-AILab/flash-attention/releases/download/"
        "v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl",
    )
    .run_commands(
        "git clone https://github.com/HazyResearch/cartridges.git /opt/cartridges "
        "&& pip install -e /opt/cartridges"
    )
)

volume = modal.Volume.from_name("offpolicy-results", create_if_missing=True)
app = modal.App("offpolicy-reeval", image=image, volumes={"/results": volume})

# The completed run's directory on the volume
RUN_DIR = "2026-03-04-11-19-04-_container_entrypoint/a4ad6ac9-0262-4a3b-ad82-5e178a5b8327"

# Checkpoint steps to re-evaluate (the ones missing from our JSON)
CHECKPOINT_STEPS = [1536, 2048, 2560, 2694]

GLOBAL_BATCH_SIZE = 32
PACKED_SEQ_LEN = 2048


@app.function(
    gpu="A100-80GB",
    timeout=3600,  # 1 hour max
    secrets=[modal.Secret.from_name("huggingface-secret")],
    memory=65536,
)
def reeval():
    import sys
    sys.path.insert(0, "/opt/cartridges")

    import torch
    from transformers import AutoTokenizer
    from cartridges.models.llama.modeling_llama import FlexLlamaForCausalLM
    from cartridges.models.config import HFModelConfig
    from cartridges.cache import TrainableCache
    from cartridges.train import CacheAndModel, GenerationEvalConfig, evaluate_generations
    from cartridges.data.longhealth.evals import LongHealthMultipleChoiceGenerateDataset

    device = "cuda"

    # ── Load model + tokenizer once ──────────────────────────────────
    print("Loading model...")
    model_config = HFModelConfig(
        pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct",
        model_cls=FlexLlamaForCausalLM,
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    model = model_config.instantiate().to(device).to(torch.bfloat16)
    for param in model.parameters():
        param.requires_grad = False
    print("Model loaded.")

    # ── Load eval dataset ────────────────────────────────────────────
    patient_ids = [f"patient_{i:02d}" for i in range(1, 11)]
    eval_dataset_config = LongHealthMultipleChoiceGenerateDataset.Config(
        patient_ids=patient_ids,
    )
    eval_dataset = eval_dataset_config.instantiate(tokenizer=tokenizer, seed=42)
    eval_gen_config = GenerationEvalConfig(
        dataset=eval_dataset_config,
        name_for_wandb="longhealth_p10",
        generate_max_new_tokens=512,
        batch_size=32,
        temperature=0.3,
    )
    print(f"Eval dataset: {len(eval_dataset)} questions")

    # ── Evaluate each checkpoint ─────────────────────────────────────
    results = []
    for step in CHECKPOINT_STEPS:
        ckpt_name = f"cache-step{step}.pt" if step != 2694 else "cache_last.pt"
        ckpt_path = f"/results/{RUN_DIR}/{ckpt_name}"

        # Also try the exact name
        import os
        if not os.path.exists(ckpt_path):
            ckpt_path = f"/results/{RUN_DIR}/cache-step{step}.pt"
        if not os.path.exists(ckpt_path):
            print(f"  ⚠ Checkpoint not found for step {step}, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"  Evaluating step {step}  ({ckpt_path})")
        print(f"{'='*60}")

        # Load cache from checkpoint
        cache = TrainableCache.from_pretrained(ckpt_path, device=device)
        cache = cache.to(device)

        # Wrap model + cache
        wrapped = CacheAndModel(cache, model)

        # Run eval
        eval_results = evaluate_generations(
            config=eval_gen_config,
            model=wrapped,
            tokenizer=tokenizer,
            dataset=eval_dataset,
            optimizer_step=step,
            local_rank=0,
            log_to_wandb=False,
        )

        if eval_results:
            import pandas as pd
            df = pd.DataFrame(eval_results)
            score_cols = [c for c in df.columns if c.endswith("score")]
            scores = {c: round(float(df[c].mean()), 4) for c in score_cols}
        else:
            scores = {"score": 0.0}

        entry = {
            "optimizer_step": step,
            "total_tokens": step * GLOBAL_BATCH_SIZE * PACKED_SEQ_LEN,
            "total_packed_seqs": step * GLOBAL_BATCH_SIZE,
            "scores": scores,
            "num_eval_questions": len(eval_results) if eval_results else 0,
            "timestamp": time.time(),
        }
        results.append(entry)

        print(f"\n  Step {step}: {scores}")

        # Free cache memory
        del cache, wrapped
        torch.cuda.empty_cache()

    # ── Merge with existing eval_scores.json ─────────────────────────
    existing_path = "/results/eval_scores.json"
    if os.path.exists(existing_path):
        with open(existing_path) as f:
            existing = json.load(f)
    else:
        existing = {"method": "off_policy", "evals": []}

    # Remove any existing entries for these steps, then add new ones
    existing_steps = {e["optimizer_step"] for e in results}
    existing["evals"] = [
        e for e in existing["evals"] if e["optimizer_step"] not in existing_steps
    ]
    existing["evals"].extend(results)
    existing["evals"].sort(key=lambda e: e["optimizer_step"])

    with open(existing_path, "w") as f:
        json.dump(existing, f, indent=2)

    volume.commit()

    print(f"\n{'='*60}")
    print("  RE-EVAL COMPLETE")
    print(f"{'='*60}")
    for e in existing["evals"]:
        print(f"  Step {e['optimizer_step']:>5}  |  score={e['scores'].get('score', 'N/A')}")
    print(f"\n  Total: {len(existing['evals'])} data points")

    return existing


@app.local_entrypoint()
def main():
    result = reeval.remote()
    # Save locally too
    with open("results/off_policy.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nSaved to results/off_policy.json")
