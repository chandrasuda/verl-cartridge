"""
Quick 5-step speed test for on-policy cartridge training.
Runs the actual training loop for 5 steps and reports timing.
"""
import modal
import time

TOKASAURUS_URL = "https://kiran1234c--tokasaurus-cartridge-server-serve.modal.run"
GPU = "A100-80GB"

# Same image as modal_train.py
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git")
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "CARTRIDGES_DIR": "/opt/cartridges",
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
    # Install veRL from our fork (has CartridgeConfig, top-k CE, KVFromText fix, etc.)
    .run_commands(
        "git clone https://github.com/chandrasuda/verl-cartridge.git /opt/verl-cartridge "
        "&& pip install -e /opt/verl-cartridge"
    )
    .run_commands("pip install git+https://github.com/chandrasuda/tokasaurus.git@geoff/cartridges")
    .pip_install(
        "requests", "transformers==4.53.0", "ray[default]", "omegaconf", "hydra-core",
        "pandas", "pyarrow", "aiohttp", "codetiming", "torchdata", "peft",
        "cachetools", "datasets", "tiktoken",
    )
    .run_commands("echo 'fork-v2'")  # force rebuild
)

app = modal.App("speed-test", image=image)


@app.function(
    gpu=GPU,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=7200,  # 2 hours
)
def test_speed():
    import subprocess, os, sys, json, time

    import verl
    verl_path = os.path.dirname(os.path.dirname(verl.__file__))
    print(f"veRL installed at: {verl_path}")

    # Verify CartridgeConfig exists (installed from fork)
    from verl.workers.config.actor import CartridgeConfig
    print(f"✓ CartridgeConfig found: {CartridgeConfig}")

    # 1. Create minimal training data (just 200 prompts for the speed test)
    os.makedirs("/root/data/cartridge_distill", exist_ok=True)
    import pandas as pd
    prompts = []
    for i in range(200):
        prompts.append({
            "prompt": [{"role": "user", "content": f"Question {i}: What is the diagnosis for a patient with chest pain?"}],
            "patient_id": f"patient_{(i % 10) + 1:02d}",
            "data_source": "longhealth",
            "reward_model": {"ground_truth": "A"},
        })
    df = pd.DataFrame(prompts)
    df.to_parquet("/root/data/cartridge_distill/train.parquet")
    df[:20].to_parquet("/root/data/cartridge_distill/val.parquet")
    print(f"Created {len(df)} training prompts")

    # 3. Write dummy reward
    os.makedirs("/tmp/reward", exist_ok=True)
    with open("/tmp/reward/dummy_reward.py", "w") as f:
        f.write("def compute_score(data_source, solution_str, ground_truth, extra_info=None):\n    return 0.0\n")

    # 4. Prepare LongHealth data (for teacher document lookup)
    import requests as req
    url = "https://raw.githubusercontent.com/kbressem/LongHealth/refs/heads/main/data/benchmark_v5.json"
    r = req.get(url)
    with open("/tmp/longhealth_data.json", "w") as f:
        json.dump(r.json(), f)
    print(f"Downloaded LongHealth data ({len(r.json())} patients)")

    # 5. Run 5 steps
    env = os.environ.copy()
    env["HYDRA_FULL_ERROR"] = "1"
    env["NCCL_DEBUG"] = "WARN"
    env["TOKENIZERS_PARALLELISM"] = "false"

    cmd = [
        sys.executable, "-m", "verl.trainer.main_ppo",
        "algorithm.adv_estimator=grpo",
        "data.train_files=/root/data/cartridge_distill/train.parquet",
        "data.val_files=/root/data/cartridge_distill/val.parquet",
        "data.train_batch_size=8",  # smaller for speed test (faster turnaround)
        "data.max_prompt_length=512",
        "data.max_response_length=256",  # shorter for speed
        "data.filter_overlong_prompts=True",
        "actor_rollout_ref.model.path=meta-llama/Llama-3.2-3B-Instruct",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
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
        "+actor_rollout_ref.actor.cartridge.num_tokens=2048",
        "+actor_rollout_ref.actor.cartridge.num_frozen_tokens=1",
        "+actor_rollout_ref.actor.cartridge.lr=0.02",
        "actor_rollout_ref.rollout.name=tokasaurus",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.temperature=0.7",
        "actor_rollout_ref.rollout.n=1",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4",
        "actor_rollout_ref.rollout.agent.num_workers=1",
        f"+actor_rollout_ref.rollout.custom.tokasaurus_url={TOKASAURUS_URL}",
        "+actor_rollout_ref.rollout.custom.cartridges=[]",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        "algorithm.use_kl_in_reward=False",
        "reward.custom_reward_function.path=/tmp/reward/dummy_reward.py",
        "reward.custom_reward_function.name=compute_score",
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
        "trainer.total_training_steps=5",
        "trainer.val_before_train=False",
    ]

    print(f"\n{'='*60}")
    print("LAUNCHING 5-STEP SPEED TEST")
    print(f"Tokasaurus: {TOKASAURUS_URL}")
    print(f"{'='*60}\n")

    t0 = time.time()
    result = subprocess.run(cmd, env=env)
    elapsed = time.time() - t0

    per_step = elapsed / 5
    eta_1000 = 1000 * per_step / 3600

    print(f"\n{'='*60}")
    print(f"SPEED TEST RESULTS")
    print(f"  Exit code: {result.returncode}")
    print(f"  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Per-step (incl. init): {per_step:.0f}s")
    print(f"  1000 steps ETA: {eta_1000:.1f} hours")
    print(f"  Cost @ $3.73/hr actor + $2.78/hr toka: ${eta_1000 * (3.73 + 2.78):.0f}")
    print(f"{'='*60}")

    return {
        "exit_code": result.returncode,
        "total_sec": elapsed,
        "per_step_sec": per_step,
        "eta_1000_hours": eta_1000,
    }


@app.local_entrypoint()
def main():
    result = test_speed.remote()
    print(f"\nResult: {result}")
