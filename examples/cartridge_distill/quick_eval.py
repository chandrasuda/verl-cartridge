"""Quick eval of specific checkpoints. Runs on separate GPU, doesn't touch training."""
import modal

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git")
    .env({"CUDA_HOME": "/usr/local/cuda", "CARTRIDGES_DIR": "/opt/cartridges", "CARTRIDGES_OUTPUT_DIR": "/tmp"})
    .pip_install("torch==2.6.0", "packaging", "numpy", "requests")
    .run_commands(
        "pip install https://github.com/Dao-AILab/flash-attention/releases/download/"
        "v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl",
    )
    .run_commands("git clone https://github.com/HazyResearch/cartridges.git /opt/cartridges && pip install -e /opt/cartridges")
    .pip_install("transformers==4.53.0")
)

results_volume = modal.Volume.from_name("onpolicy-results", create_if_missing=True)
app = modal.App("quick-eval", image=image)


@app.function(
    gpu="A10G",
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,
    volumes={"/results": results_volume},
)
def run_eval(steps: list[int] = [5, 140]):
    import json, os, re, torch, requests as http_req
    from transformers import AutoTokenizer
    from cartridges.models.llama.modeling_llama import FlexLlamaForCausalLM
    from cartridges.cache import TrainableCache
    from cartridges.generation import flex_generate

    MODEL = "meta-llama/Llama-3.2-3B-Instruct"
    NUM_EVAL = 40

    # Load LongHealth
    data = http_req.get(
        "https://raw.githubusercontent.com/kbressem/LongHealth/refs/heads/main/data/benchmark_v5.json"
    ).json()
    questions = []
    for pid, patient in data.items():
        if int(pid.split("_")[1]) > 8:
            continue
        for q in patient["questions"]:
            options = "\n".join(f"{L}) {q[f'answer_{L.lower()}']}" for L in "ABCDE")
            prompt = (
                f"You are answering a multiple choice question about patient {patient['name']}.\n\n"
                f"Question: {q['question']}\n\nOptions:\n{options}\n\n"
                f"Answer with ONLY the letter (A, B, C, D, or E):"
            )
            answer_map = {q[f"answer_{L.lower()}"]: L for L in "ABCDE"}
            questions.append({"prompt": prompt, "correct": answer_map.get(q["correct"], "?")})
    eval_qs = questions[:NUM_EVAL]
    print(f"Eval: {len(eval_qs)} questions\n")

    def extract_answer(text):
        m = re.search(r'\b([A-E])\b', text.strip()[:20])
        return m.group(1) if m else "?"

    # Load model
    print(f"Loading {MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # Baseline uses standard Llama (FlexLlama doesn't support .generate())
    from transformers import AutoModelForCausalLM
    base_model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda().eval()

    print("=== Baseline (no cartridge) ===")
    correct_bl = 0
    for i, q in enumerate(eval_qs):
        ids = tokenizer.encode(q["prompt"], return_tensors="pt").cuda()
        with torch.no_grad():
            out = base_model.generate(ids, max_new_tokens=10, do_sample=False)
        gen_text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        pred = extract_answer(gen_text)
        if i < 3:
            print(f"    Q{i}: gen='{gen_text[:60]}' pred={pred} correct={q['correct']}")
        if pred == q["correct"]:
            correct_bl += 1
    print(f"  Baseline: {correct_bl}/{len(eval_qs)} ({correct_bl/len(eval_qs)*100:.1f}%)\n")

    # Free base model, load FlexLlama for cartridge eval
    del base_model
    torch.cuda.empty_cache()
    model = FlexLlamaForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda().eval()
    print("✓ FlexLlama loaded for cartridge eval\n")

    # Test with off-policy pretrained cartridge first (sanity check)
    print("=== Off-policy cartridge (hazyresearch/cartridge-wauoq23f) ===")
    try:
        from huggingface_hub import hf_hub_download
        offpol_path = hf_hub_download("hazyresearch/cartridge-wauoq23f", "cartridge.pt")
        offpol_cache = TrainableCache.from_pretrained(offpol_path).cuda()
        print(f"  Loaded: {offpol_cache._num_trainable_tokens} trainable tokens")
        correct_op = 0
        for qi, q in enumerate(eval_qs):
            ids = tokenizer.encode(q["prompt"], return_tensors="pt").cuda()
            flat_ids = ids.squeeze(0)
            seq_ids_t = torch.zeros_like(flat_ids)
            pos_ids = torch.arange(len(flat_ids), device=ids.device)
            offpol_cache.clear()
            gen_output = flex_generate(
                model=model, tokenizer=tokenizer, cache=offpol_cache,
                input_ids=flat_ids, seq_ids=seq_ids_t, position_ids=pos_ids,
                max_new_tokens=10, temperature=0.0,
            )
            gen_tokens = gen_output.get(0, [])
            gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            pred = extract_answer(gen_text)
            if qi < 3:
                print(f"    Q{qi}: gen='{gen_text[:80]}' pred={pred} correct={q['correct']}")
            if pred == q["correct"]:
                correct_op += 1
        print(f"  Off-policy: {correct_op}/{len(eval_qs)} ({correct_op/len(eval_qs)*100:.1f}%)\n")
        del offpol_cache
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  ✗ Off-policy eval failed: {e}\n")

    # Eval each on-policy step
    results = {"baseline": correct_bl / len(eval_qs) * 100, "evals": {}}
    for step in steps:
        ckpt_path = f"/results/onpolicy/cartridge_checkpoints/cache-step{step}.pt"
        if not os.path.exists(ckpt_path):
            print(f"=== Step {step}: NOT FOUND ===")
            continue

        print(f"=== Step {step} ===")
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if "fixed_keys" in ckpt and "frozen_keys" not in ckpt:
                ckpt["frozen_keys"] = ckpt.pop("fixed_keys")
                ckpt["frozen_values"] = ckpt.pop("fixed_values")
                torch.save(ckpt, ckpt_path)
            del ckpt

            # Work around cartridges bug: from_pretrained reads .size(1) instead of .size(2)
            # for num_frozen_tokens, giving n_heads instead of actual frozen count.
            # Manually reconstruct with correct dimensions.
            ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            trainable_keys = ckpt_data["trainable_keys"]
            frozen_keys = ckpt_data["frozen_keys"]
            n_layers = len(trainable_keys)
            n_heads = trainable_keys[0].size(1)
            head_dim = trainable_keys[0].size(3)
            num_frozen = frozen_keys[0].size(2) if frozen_keys else 0
            from cartridges.cache import AttnConfig
            init_keys = [
                torch.cat([frozen_keys[i], trainable_keys[i]], dim=2).contiguous()
                if num_frozen > 0 else trainable_keys[i]
                for i in range(n_layers)
            ]
            init_values = [
                torch.cat([ckpt_data["frozen_values"][i], ckpt_data["trainable_values"][i]], dim=2).contiguous()
                if num_frozen > 0 else ckpt_data["trainable_values"][i]
                for i in range(n_layers)
            ]
            cache = TrainableCache(
                config=AttnConfig(n_layers=n_layers, n_heads=n_heads, head_dim=head_dim),
                init_keys=init_keys, init_values=init_values,
                num_frozen_tokens=num_frozen,
            ).cuda()
            del ckpt_data
            correct = 0
            for qi, q in enumerate(eval_qs):
                ids = tokenizer.encode(q["prompt"], return_tensors="pt").cuda()
                flat_ids = ids.squeeze(0)  # (seq_len,)
                seq_ids = torch.zeros_like(flat_ids)  # all belong to sequence 0
                pos_ids = torch.arange(len(flat_ids), device=ids.device)
                cache.clear()
                gen_output = flex_generate(
                    model=model, tokenizer=tokenizer, cache=cache,
                    input_ids=flat_ids, seq_ids=seq_ids, position_ids=pos_ids,
                    max_new_tokens=10, temperature=0.0,
                )
                # flex_generate returns {seq_id: [token_ids]} dict
                gen_tokens = gen_output.get(0, [])
                gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                pred = extract_answer(gen_text)
                if qi < 3:
                    print(f"    Q{qi}: gen='{gen_text[:80]}' pred={pred} correct={q['correct']}")
                if pred == q["correct"]:
                    correct += 1
            acc = correct / len(eval_qs) * 100
            print(f"  Step {step}: {correct}/{len(eval_qs)} ({acc:.1f}%)\n")
            results["evals"][step] = acc
            del cache
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  ✗ Failed: {e}\n")

    print("=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"  Baseline:  {results['baseline']:.1f}%")
    for step, acc in sorted(results["evals"].items()):
        print(f"  Step {step:>4}:  {acc:.1f}%")
    return results


@app.local_entrypoint()
def main():
    results = run_eval.remote(steps=[5, 100, 140])
    print(f"\nResults: {results}")
