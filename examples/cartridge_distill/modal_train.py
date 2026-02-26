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
        "CARTRIDGES_DIR": "/opt/cartridges/cartridges",
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
    # Install veRL
    .run_commands(
        "pip install git+https://github.com/volcengine/verl.git"
    )
    # Install tokasaurus
    .run_commands(
        "pip install git+https://github.com/chandrasuda/tokasaurus.git@geoff/cartridges"
    )
    .pip_install("requests")
    .run_commands("echo 'patch-version-8'")  # Force image rebuild to pick up latest patches
    .run_commands(
        "python3 -c \""
        "import urllib.request, zipfile, shutil, os; "
        "urllib.request.urlretrieve("
        "'https://github.com/chandrasuda/on-policy-cartridge-training/archive/refs/heads/main.zip', "
        "'/tmp/p.zip'); "
        "zipfile.ZipFile('/tmp/p.zip').extractall('/opt/'); "
        "shutil.move('/opt/on-policy-cartridge-training-main', '/opt/patches'); "
        "os.remove('/tmp/p.zip'); "
        "print(os.listdir('/opt/patches/verl_patches/'))"
        "\""
    )
    # Remaining deps not covered by the above
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

cartridge_volume = modal.Volume.from_name("cartridge-checkpoints", create_if_missing=True)
app = modal.App("cartridge-training", image=image)


@app.function(
    gpu=GPU,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=7200,  # 2 hours
    min_containers=0,
    max_containers=1,
    scaledown_window=600,
    volumes={"/cartridge_output": cartridge_volume},
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

    # First, apply the veRL patches
    # Find where veRL is installed
    import verl
    verl_path = os.path.dirname(os.path.dirname(verl.__file__))
    print(f"veRL installed at: {verl_path}")

    # Patches baked into image at /opt/patches
    import shutil
    print(f"✓ Patches at /opt/patches: {os.listdir('/opt/patches/verl_patches/')}")

    # Copy the tokasaurus rollout bridge
    toka_dst = os.path.join(verl_path, "verl", "workers", "rollout", "tokasaurus_rollout")
    toka_src = "/opt/patches/verl_patches/rollout/tokasaurus_rollout"
    if os.path.exists(toka_dst):
        shutil.rmtree(toka_dst)
    shutil.copytree(toka_src, toka_dst)
    print(f"Copied tokasaurus_rollout to {toka_dst}")

    # Apply patches by directly overwriting files in the installed veRL package.
    # git apply doesn't work on pip-installed site-packages (no .git).
    # Instead, we use a Python script that reads the patch and applies edits.
    import importlib
    import verl.workers.config.actor as actor_mod
    import verl.workers.actor.dp_actor as dp_actor_mod
    import verl.workers.rollout.replica as replica_mod
    import verl.workers.fsdp_workers as fsdp_mod
    import verl.trainer.ppo.ray_trainer as trainer_mod

    # For each patched file, copy our pre-patched version from the workspace
    # We keep full patched copies in the repo for this purpose
    patch_script = "/tmp/patches/apply_patches.py"
    subprocess.run([sys.executable, "-c", f"""
import subprocess, sys
# Apply patches using the `patch` command directly on the installed files
import verl, os
sp = os.path.dirname(os.path.dirname(verl.__file__))

patches = [
    ('config/actor_config.patch', 'verl/workers/config/actor.py'),
    ('actor/dp_actor.patch', 'verl/workers/actor/dp_actor.py'),
    ('rollout/replica.patch', 'verl/workers/rollout/replica.py'),
    ('fsdp_workers.patch', 'verl/workers/fsdp_workers.py'),
    ('ray_trainer.patch', 'verl/trainer/ppo/ray_trainer.py'),
    ('rollout/base.patch', 'verl/workers/rollout/base.py'),
    ('agent_loop.patch', 'verl/experimental/agent_loop/agent_loop.py'),
]
for patch_name, target in patches:
    patch_path = f'/opt/patches/verl_patches/{{patch_name}}'
    target_path = os.path.join(sp, target)
    if not os.path.exists(target_path):
        print(f'⚠ Target not found: {{target_path}}')
        continue
    r = subprocess.run(['patch', '-p1', '--forward', '-i', patch_path, target_path],
                       capture_output=True, text=True)
    if r.returncode == 0:
        print(f'✓ Patched {{target}}')
    elif 'already applied' in r.stdout or 'Reversed' in r.stdout:
        print(f'✓ Already patched {{target}}')
    else:
        print(f'⚠ Failed to patch {{target}}: {{r.stderr[:200]}}')
"""], check=False)

    # Prepare data
    subprocess.run([
        sys.executable, "/opt/patches/training/prepare_data.py"
    ], check=True)

    # Pre-download LongHealth data for the teacher (ref worker may not have HTTP access)
    subprocess.run([sys.executable, "-c", """
import requests, json
data = requests.get('https://raw.githubusercontent.com/kbressem/LongHealth/refs/heads/main/data/benchmark_v5.json').json()
with open('/tmp/longhealth_data.json', 'w') as f:
    json.dump(data, f)
print(f'Saved LongHealth data ({len(data)} patients) to /tmp/longhealth_data.json')
"""], check=True)

    # Run training
    env = os.environ.copy()
    env["TOKASAURUS_URL"] = TOKASAURUS_URL

    cmd = [
        sys.executable, "-m", "verl.trainer.main_ppo",
        "algorithm.adv_estimator=grpo",
        #
        "data.train_files=/root/data/cartridge_distill/train.parquet",
        "data.val_files=/root/data/cartridge_distill/val.parquet",
        "data.train_batch_size=8",
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
        "actor_rollout_ref.actor.ppo_mini_batch_size=8",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
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
        "+actor_rollout_ref.actor.cartridge.checkpoint_path=hazyresearch/cartridge-wauoq23f",
        "+actor_rollout_ref.actor.cartridge.num_tokens=2048",
        "+actor_rollout_ref.actor.cartridge.lr=0.02",
        #
        "actor_rollout_ref.rollout.name=tokasaurus",
        "actor_rollout_ref.rollout.temperature=0.7",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4",
        "actor_rollout_ref.rollout.agent.num_workers=1",
        f"+actor_rollout_ref.rollout.custom.tokasaurus_url={TOKASAURUS_URL}",
        "+actor_rollout_ref.rollout.custom.cartridges=[{id: hazyresearch/cartridge-wauoq23f, source: huggingface}]",
        #
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        #
        "algorithm.use_kl_in_reward=False",
        # Dummy reward (cartridge distillation uses KL loss, not reward-based RL)
        "reward.custom_reward_function.path=/opt/patches/training/dummy_reward.py",
        "reward.custom_reward_function.name=compute_score",
        #
        "trainer.critic_warmup=0",
        'trainer.logger=["console"]',
        "trainer.project_name=cartridge_distill",
        "trainer.experiment_name=longhealth_llama3b_onpolicy",
        "trainer.n_gpus_per_node=1",
        "trainer.nnodes=1",
        "trainer.save_freq=-1",  # Disable checkpointing (CacheAndModel missing .config attr)
        "trainer.test_freq=5",
        "trainer.total_epochs=2",
        "trainer.val_before_train=False",
        "trainer.test_freq=-1",
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
    print("Training complete. Saving cartridge to /cartridge_output/...")
    print("=" * 60)

    # The cartridge was saved to /tmp/verl_cartridge_latest.pt during the last sync
    import shutil
    src = "/tmp/verl_cartridge_latest.pt"
    dst = "/cartridge_output/on_policy_cartridge.pt"
    if os.path.exists(src):
        shutil.copy2(src, dst)
        cartridge_volume.commit()
        print(f"✓ Saved on-policy cartridge to volume: {dst}")
    else:
        print(f"⚠ Cartridge not found at {src}. It may not have been synced.")

    return result.returncode


@app.function(
    gpu="A10G",
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=1800,
    volumes={"/cartridge_output": cartridge_volume},
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
    ckpt_path = "/cartridge_output/on_policy_cartridge.pt"
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
