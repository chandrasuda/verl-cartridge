"""
Modal training instance for on-policy cartridge distillation.

Runs veRL training on an A10G GPU with the cartridge patches applied.
Tokasaurus inference runs on a separate Modal deployment (already running).

Usage:
    modal run examples/cartridge_distill/modal_train.py
"""

import modal

TOKASAURUS_URL = "https://kiran1234c--tokasaurus-cartridge-server-serve.modal.run"
GPU = "A100-80GB"

# Build image with veRL + cartridges + all patches applied
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git", "patch")
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "CARTRIDGES_DIR": "/opt/cartridges",
        "CARTRIDGES_OUTPUT_DIR": "/tmp/cartridge_output",
    })
    .pip_install("torch==2.6.0", "packaging", "numpy")
    .run_commands(
        # Pre-built wheel from GitHub — avoids compiling from source entirely
        "pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl",
        "pip install flashinfer-python==0.2.0.post2 --extra-index-url https://flashinfer.ai/whl/cu124/torch2.6/",
    )
    # Install cartridges — clone and install in editable mode to ensure all subpackages are included
    .run_commands(
        "git clone https://github.com/HazyResearch/cartridges.git /opt/cartridges && pip install -e /opt/cartridges"
    )
    # Install veRL from our fork (has cartridge support baked in)
    # Cache bust: change the echo to force re-clone on code changes
    .run_commands("echo 'verl-fork-v8-teacher-mb4'")
    .run_commands(
        "git clone https://github.com/chandrasuda/verl-cartridge.git /opt/verl-cartridge "
        "&& pip install -e /opt/verl-cartridge"
    )
    # Install tokasaurus
    .run_commands(
        "pip install git+https://github.com/chandrasuda/tokasaurus.git@geoff/cartridges"
    )
    .pip_install("requests")
    .pip_install(
        "transformers==4.53.0",
        "ray[default]",
        "omegaconf",
        "hydra-core",
        "pandas",
        "pyarrow",
        "aiohttp",
        "codetiming",
        "torchdata",
        "peft",
        "cachetools",
    )
)

results_volume = modal.Volume.from_name("onpolicy-results", create_if_missing=True)
data_volume = modal.Volume.from_name("training-data")
app = modal.App("onpolicy-training", image=image)


@app.function(
    gpu=GPU,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=86400,  # 24 hours
    min_containers=0,
    max_containers=1,
    scaledown_window=600,
    volumes={"/results": results_volume, "/data": data_volume},
)
def train():
    """Run cartridge distillation training."""
    import subprocess
    import os
    import sys

    # Verify GPU
    try:
        import torch
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    except Exception as e:
        print(f"GPU check failed: {e}")

    # Verify our fork is installed correctly
    from verl.workers.config.actor import CartridgeConfig
    print(f"✓ veRL fork installed with CartridgeConfig")

    # Verify pre-processed data from volume
    train_parquet = "/data/cartridge_distill/train.parquet"
    val_parquet = "/data/cartridge_distill/val.parquet"
    assert os.path.exists(train_parquet), f"Missing {train_parquet} — upload to 'training-data' volume"
    assert os.path.exists(val_parquet), f"Missing {val_parquet} — upload to 'training-data' volume"
    print(f"✓ Training data: {train_parquet}")
    print(f"✓ Validation data: {val_parquet}")

    # Pre-download LongHealth data for the teacher
    import requests as req
    data = req.get('https://raw.githubusercontent.com/kbressem/LongHealth/refs/heads/main/data/benchmark_v5.json').json()
    import json
    with open('/tmp/longhealth_data.json', 'w') as f:
        json.dump(data, f)
    print(f'Saved LongHealth data ({len(data)} patients) to /tmp/longhealth_data.json')

    # Create dummy reward function (cartridge uses KL loss, not rewards)
    os.makedirs("/tmp/reward", exist_ok=True)
    with open("/tmp/reward/dummy_reward.py", "w") as f:
        f.write("def compute_score(data_source, solution_str, ground_truth, extra_info=None):\n    return 0.0\n")

    # Run training
    env = os.environ.copy()
    env["TOKASAURUS_URL"] = TOKASAURUS_URL

    cmd = [
        sys.executable, "-m", "verl.trainer.main_ppo",
        "algorithm.adv_estimator=grpo",
        #
        "data.train_files=/data/cartridge_distill/train.parquet",
        "data.val_files=/data/cartridge_distill/val.parquet",
        "data.train_batch_size=32",
        "data.max_prompt_length=512",
        "data.max_response_length=512",
        "data.filter_overlong_prompts=True",
        "data.truncation=right",
        #
        "actor_rollout_ref.model.path=meta-llama/Llama-3.2-3B-Instruct",
        "actor_rollout_ref.model.external_lib=cartridges",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        #
        "actor_rollout_ref.actor.strategy=fsdp",
        "actor_rollout_ref.actor.ppo_mini_batch_size=32",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.kl_loss_coef=1.0",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.actor.ppo_epochs=1",
        "actor_rollout_ref.actor.fsdp_config.param_offload=False",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
        "actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16",
        "actor_rollout_ref.actor.fsdp_config.use_orig_params=True",
        "actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16",
        "+actor_rollout_ref.actor.cartridge.enabled=True",
        # No checkpoint_path → KVFromText init (same as off-policy baseline)
        "+actor_rollout_ref.actor.cartridge.num_tokens=2048",
        "+actor_rollout_ref.actor.cartridge.num_frozen_tokens=1",
        "+actor_rollout_ref.actor.cartridge.lr=0.02",
        #
        "actor_rollout_ref.rollout.name=tokasaurus",
        "actor_rollout_ref.rollout.temperature=0.7",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4",
        "actor_rollout_ref.rollout.agent.num_workers=1",
        f"+actor_rollout_ref.rollout.custom.tokasaurus_url={TOKASAURUS_URL}",
        # Cartridge synced from actor after each step (no pre-loaded cartridge)
        "+actor_rollout_ref.rollout.custom.cartridges=[]",
        #
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        #
        "algorithm.use_kl_in_reward=False",
        # Dummy reward (cartridge distillation uses KL loss, not reward-based RL)
        "reward.custom_reward_function.path=/tmp/reward/dummy_reward.py",
        "reward.custom_reward_function.name=compute_score",
        #
        "trainer.critic_warmup=0",
        'trainer.logger=["console"]',
        "trainer.project_name=cartridge_distill",
        "trainer.experiment_name=longhealth_llama3b_onpolicy",
        "trainer.n_gpus_per_node=1",
        "trainer.nnodes=1",
        "trainer.save_freq=-1",  # Disable full-model checkpointing (CacheAndModel missing .config)
        "trainer.test_freq=-1",  # Disable reward-based test (dummy reward = useless)
        "+trainer.cartridge_save_freq=10",  # Save cache .pt every 10 steps so we can eval early
        "trainer.default_local_dir=/results/onpolicy",
        "trainer.total_epochs=100",  # Large — actual limit is total_training_steps below
        "trainer.total_training_steps=300",  # ~15 hours at ~180s/step
        "trainer.val_before_train=False",
    ]

    print(f"\n{'='*60}")
    print("Starting on-policy cartridge distillation training")
    print(f"Tokasaurus URL: {TOKASAURUS_URL}")
    print(f"GPU: {GPU}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, env=env)
    
    if result.returncode != 0:
        print(f"Training failed with exit code {result.returncode}")
        return result.returncode

    # Save the trained cartridge to persistent volume
    print("\n" + "=" * 60)
    print("Training complete. Saving results to volume...")
    print("=" * 60)

    # Copy all cartridge checkpoints to volume
    import shutil, glob
    ckpt_dir = "/results/onpolicy/cartridge_checkpoints"
    if os.path.exists(ckpt_dir):
        ckpts = glob.glob(os.path.join(ckpt_dir, "cache-step*.pt"))
        print(f"✓ Found {len(ckpts)} cartridge checkpoints")
        for c in sorted(ckpts):
            print(f"  {os.path.basename(c)}")
    else:
        print(f"⚠ No checkpoint dir at {ckpt_dir}")

    # Also copy the latest synced cartridge
    src = "/tmp/verl_cartridge_latest.pt"
    dst = "/results/onpolicy/on_policy_cartridge_final.pt"
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"✓ Saved final cartridge: {dst}")

    results_volume.commit()

    return result.returncode


@app.function(
    gpu="A10G",
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=1800,
    volumes={"/results": results_volume},
)
def eval_cartridge():
    """Evaluate on-policy cartridge on LongHealth."""
    import torch
    import json
    import re
    import requests as http_req

    print("Loading model + cartridge...")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from cartridges.cache import TrainableCache

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-3B-Instruct", torch_dtype=torch.bfloat16
    ).cuda().eval()

    # Load on-policy cartridge
    ckpt_path = "/results/onpolicy/on_policy_cartridge_final.pt"
    import os
    if os.path.exists(ckpt_path):
        # Rename fixed_keys if needed
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "fixed_keys" in ckpt and "frozen_keys" not in ckpt:
            ckpt["frozen_keys"] = ckpt.pop("fixed_keys")
            ckpt["frozen_values"] = ckpt.pop("fixed_values")
            torch.save(ckpt, ckpt_path + ".fixed")
            ckpt_path = ckpt_path + ".fixed"
        del ckpt
        cartridge = TrainableCache.from_pretrained(ckpt_path).cuda()
        print(f"✓ Loaded on-policy cartridge: {cartridge._num_trainable_tokens} trainable tokens")
    else:
        print(f"⚠ No cartridge found at {ckpt_path}")
        return

    # Load LongHealth questions
    data = http_req.get("https://raw.githubusercontent.com/kbressem/LongHealth/refs/heads/main/data/benchmark_v5.json").json()

    questions = []
    for pid, patient in data.items():
        if int(pid.split("_")[1]) > 8: continue
        for q in patient["questions"]:
            options = f"A) {q['answer_a']}\nB) {q['answer_b']}\nC) {q['answer_c']}\nD) {q['answer_d']}\nE) {q['answer_e']}"
            answer_map = {q["answer_a"]: "A", q["answer_b"]: "B", q["answer_c"]: "C", q["answer_d"]: "D", q["answer_e"]: "E"}
            correct = answer_map.get(q["correct"], "?")
            prompt = f"Question about patient {patient['name']}:\n{q['question']}\nOptions:\n{options}\nAnswer (A/B/C/D/E):"
            questions.append({"prompt": prompt, "correct": correct})

    print(f"Evaluating {len(questions[:40])} questions...")

    # We can't easily eval with cartridge locally (needs FlexLlama)
    # But we can eval baseline without cartridge
    correct_baseline = 0
    for q in questions[:40]:
        ids = tokenizer.encode(q["prompt"], return_tensors="pt").cuda()
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=5, temperature=0.0, do_sample=False)
        answer = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        m = re.search(r'\b([A-E])\b', answer[:20])
        pred = m.group(1) if m else "?"
        if pred == q["correct"]:
            correct_baseline += 1

    print(f"\nBaseline (no cartridge): {correct_baseline}/40 ({correct_baseline/40*100:.1f}%)")
    print(f"On-policy cartridge saved at: {ckpt_path}")
    print("Note: eval WITH cartridge requires FlexLlamaForCausalLM which needs FlexAttention setup.")
    print("Use Tokasaurus for cartridge eval (upload the .pt file to HuggingFace or pass as local).")


@app.local_entrypoint()
def main():
    exit_code = train.remote()
    print(f"\nTraining finished with exit code: {exit_code}")
