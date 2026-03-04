"""
Quick 5-step speed test for on-policy cartridge training.
Measures per-step time to estimate total training duration.
"""
import modal

TOKASAURUS_URL = "https://kiran1234c--tokasaurus-cartridge-server-serve.modal.run"
GPU = "A100-80GB"

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
        "pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl",
        "pip install flashinfer-python==0.2.0.post2 --extra-index-url https://flashinfer.ai/whl/cu124/torch2.6/",
    )
    .run_commands(
        "git clone https://github.com/HazyResearch/cartridges.git /opt/cartridges && pip install -e /opt/cartridges"
    )
    .run_commands(
        "pip install git+https://github.com/volcengine/verl.git"
    )
    .run_commands(
        "pip install git+https://github.com/chandrasuda/tokasaurus.git@geoff/cartridges"
    )
    .pip_install("requests")
    .run_commands("echo 'speed-test-v1'")
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

app = modal.App("speed-test", image=image)


@app.function(
    gpu=GPU,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,  # 1 hour max
)
def test_speed():
    import subprocess
    import os
    import sys
    import time

    # Verify GPU
    import torch
    props = torch.cuda.get_device_properties(0)
    vram = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
    print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {vram / 1e9:.1f} GB")

    # Apply patches (same as modal_train.py)
    import verl
    verl_path = os.path.dirname(os.path.dirname(verl.__file__))
    print(f"veRL path: {verl_path}")

    patches_dir = "/opt/patches/verl_patches"
    for patch_name in sorted(os.listdir(patches_dir)):
        if patch_name.endswith(".patch"):
            patch_path = os.path.join(patches_dir, patch_name)
            print(f"Applying {patch_name}...")
            result = subprocess.run(
                ["patch", "-p1", "-d", verl_path, "-i", patch_path, "--forward"],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                print(f"  ✓ Applied")
            else:
                if "already applied" in result.stdout or "Reversed" in result.stdout:
                    print(f"  ⊘ Already applied")
                else:
                    print(f"  ✗ Failed: {result.stderr[:200]}")

    # Create minimal training data (just 200 prompts for the speed test)
    os.makedirs("/root/data/cartridge_distill", exist_ok=True)
    import pandas as pd

    prompts = []
    for i in range(200):
        prompts.append({
            "prompt": [
                {"role": "user", "content": f"What is the diagnosis for patient {i % 10 + 1}? Explain in detail."}
            ],
            "patient_id": f"patient_{(i % 10) + 1:02d}",
        })

    df = pd.DataFrame(prompts)
    df.to_parquet("/root/data/cartridge_distill/train.parquet")
    df[:20].to_parquet("/root/data/cartridge_distill/val.parquet")
    print(f"Created {len(df)} training prompts")

    # Prepare LongHealth data for teacher document lookup
    import requests as req
    url = "https://raw.githubusercontent.com/kbressem/LongHealth/refs/heads/main/data/benchmark_v5.json"
    data = req.get(url).json()
    import json
    with open('/tmp/longhealth_data.json', 'w') as f:
        json.dump(data, f)
    print(f'LongHealth: {len(data)} patients')

    env = os.environ.copy()
    env["LONGHEALTH_DATA_PATH"] = "/tmp/longhealth_data.json"

    cmd = [
        sys.executable, "-m", "verl.trainer.main_ppo",
        "algorithm.adv_estimator=grpo",
        "data.train_files=/root/data/cartridge_distill/train.parquet",
        "data.val_files=/root/data/cartridge_distill/val.parquet",
        "data.train_batch_size=32",
        "data.max_prompt_length=512",
        "data.max_response_length=512",
        "data.filter_overlong_prompts=True",
        #
        f"actor_rollout_ref.model.path=meta-llama/Llama-3.2-3B-Instruct",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        #
        "actor_rollout_ref.actor.strategy=fsdp",
        "actor_rollout_ref.actor.ppo_mini_batch_size=32",
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
        "+actor_rollout_ref.actor.cartridge.num_tokens=2048",
        "+actor_rollout_ref.actor.cartridge.num_frozen_tokens=1",
        "+actor_rollout_ref.actor.cartridge.lr=0.02",
        #
        "actor_rollout_ref.rollout.name=tokasaurus",
        "actor_rollout_ref.rollout.temperature=0.7",
        "actor_rollout_ref.rollout.n=1",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4",
        "actor_rollout_ref.rollout.agent.num_workers=1",
        f"+actor_rollout_ref.rollout.custom.tokasaurus_url={TOKASAURUS_URL}",
        "+actor_rollout_ref.rollout.custom.cartridges=[]",
        #
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        #
        "algorithm.use_kl_in_reward=False",
        "reward.custom_reward_function.path=/opt/patches/training/dummy_reward.py",
        "reward.custom_reward_function.name=compute_score",
        #
        "trainer.critic_warmup=0",
        'trainer.logger=["console"]',
        "trainer.project_name=speed_test",
        "trainer.experiment_name=speed_test_5steps",
        "trainer.n_gpus_per_node=1",
        "trainer.nnodes=1",
        "trainer.save_freq=-1",
        "trainer.test_freq=-1",
        "trainer.default_local_dir=/tmp/speed_test",
        "trainer.total_epochs=100",
        "trainer.total_training_steps=5",  # JUST 5 STEPS
        "trainer.val_before_train=False",
    ]

    print(f"\n{'='*60}")
    print(f"SPEED TEST: 5 steps, batch=32")
    print(f"Tokasaurus: {TOKASAURUS_URL}")
    print(f"{'='*60}\n")

    start = time.time()
    result = subprocess.run(cmd, env=env)
    elapsed = time.time() - start

    print(f"\n{'='*60}")
    print(f"SPEED TEST RESULTS")
    print(f"  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Per-step (avg): {elapsed/5:.1f}s")
    print(f"  2700 steps ETA: {elapsed/5 * 2700 / 3600:.1f} hours")
    print(f"  Cost @ $3.73/hr A100 + $1.10/hr A10G: ${elapsed/5 * 2700 / 3600 * 4.83:.0f}")
    print(f"{'='*60}")

    return {
        "total_time_sec": elapsed,
        "per_step_sec": elapsed / 5,
        "eta_2700_steps_hours": elapsed / 5 * 2700 / 3600,
    }


@app.local_entrypoint()
def main():
    result = test_speed.remote()
    print(f"\nResult: {result}")
