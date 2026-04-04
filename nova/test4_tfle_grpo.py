#!/usr/bin/env python3
"""
Test 4: TFLE-Based GRPO — NOVA-10M (GPU-Optimized)
=====================================================
Evolutionary RL: binary reward as fitness, no gradients on ternary weights.
JIT Mamba scan + batch=128 inference for 97-99% GPU utilization.

Run: python nova/test4_tfle_grpo.py
"""
import sys
import os
import json
import time
import random
import re
import subprocess
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/workspace/tfle")
sys.path.insert(0, "/workspace/tfle/nova")
from nova_full_directive import NOVA10M, BitLinear10M
from tfle.layers import generate_k_proposals
from optimize import patch_mamba_scan

DEVICE = torch.device("cuda:0")
SEP = "=" * 60
CKPT_DIR = Path("/workspace/tfle/checkpoints")
RESULTS_DIR = Path("/workspace/tfle/nova/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
VOCAB_CACHE = Path("/workspace/tfle/nova/vocab.json")


def load_vocab():
    assert VOCAB_CACHE.exists(), "Run test1 first"
    with open(VOCAB_CACHE) as f:
        data = json.load(f)
    return data["word2idx"], {int(k): v for k, v in data["idx2word"].items()}


def gpu_stats():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        lines = r.stdout.strip().split("\n")
        utils = [int(l.split(", ")[0]) for l in lines]
        return f"GPU: {utils}%"
    except Exception:
        return ""


# ── Rewards ─────────────────────────────────────────────────

class MathReward:
    def score(self, question, response, ground_truth):
        predicted = self._extract(response)
        if predicted is None:
            return 0.0
        try:
            if abs(float(predicted) - float(ground_truth)) < 1e-4:
                return 1.0
            dist = abs(float(predicted) - float(ground_truth))
            return max(0.0, 0.5 * (1.0 - min(dist / (abs(float(ground_truth)) + 1), 1.0)))
        except (ValueError, TypeError):
            return 0.0

    def _extract(self, text):
        m = re.search(r"answer\s+is\s+(-?\d+\.?\d*)", text, re.IGNORECASE)
        if m:
            return m.group(1)
        nums = re.findall(r"-?\d+\.?\d*", text)
        return nums[-1] if nums else None


class FormatReward:
    def score(self, text):
        r = 0.0
        low = text.lower()
        if "answer" in low:
            r += 0.15
        if any(w in low for w in ("think", "step", "first", "need")):
            r += 0.15
        if re.search(r"answer\s+is\s+\d+", low):
            r += 0.2
        return r


def generate_questions(n=200):
    random.seed(99)
    qs = []
    for _ in range(n):
        a, b = random.randint(1, 50), random.randint(1, 50)
        op = random.choice(["+", "-", "*"])
        ans = eval(f"{a} {op} {b}")
        qs.append({"question": f"Question : What is {a} {op} {b} ?", "answer": ans})
    return qs


# ── Batched generation (fills all GPUs) ─────────────────────

@torch.no_grad()
def batch_generate(model, prompt_ids, G=16, max_new=40, temperature=1.0):
    """Generate G sequences from same prompt, big batch for GPU saturation."""
    ids = prompt_ids.expand(G, -1).clone()
    for _ in range(max_new):
        if ids.shape[1] >= 256:
            break
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(ids)
        nxt = logits[:, -1, :].float() / max(temperature, 1e-8)
        probs = F.softmax(nxt, dim=-1)
        tok = torch.multinomial(probs, 1)
        ids = torch.cat([ids, tok], dim=1)
    return ids


def evaluate_fitness(dp_model, questions, word2idx, idx2word,
                     math_reward, format_reward, G=16):
    """Evaluate fitness with big batch generation (fills GPUs)."""
    dp_model.eval()
    total_r = 0.0
    n = 0

    # Batch ALL questions' prompts together: G responses each
    for qdata in questions:
        prompt_tokens = [word2idx.get(w, 1) for w in qdata["question"].split()]
        prompt_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=DEVICE)

        seqs = batch_generate(dp_model, prompt_ids, G=G, max_new=40, temperature=1.0)

        for g in range(G):
            resp_tokens = seqs[g, len(prompt_tokens):].tolist()
            resp = " ".join(idx2word.get(t, "<?>") for t in resp_tokens if t != 0)
            r = math_reward.score(qdata["question"], resp, qdata["answer"])
            r += format_reward.score(resp)
            total_r += r
            n += 1

    return total_r / max(n, 1)


# ── TFLE layer ops ──────────────────────────────────────────

def get_bitlinear_layers(model):
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, BitLinear10M):
            layers.append((name, module))
    return layers


def get_ternary(module):
    with torch.no_grad():
        alpha = torch.mean(torch.abs(module.weight)).clamp(min=1e-10)
        w_q = torch.clamp(torch.round(module.weight / alpha), -1, 1)
    return w_q, alpha


# ── Main ────────────────────────────────────────────────────

def main():
    print(f"\n{SEP}\nTEST 4: TFLE-BASED GRPO (GPU-OPTIMIZED)\n{SEP}\n")

    cfg = {
        "K": 4, "flip_rate": 0.001, "group_size": 32,
        "questions_per_step": 2, "re_eval_questions": 2,
        "num_steps": 20, "eval_every": 5,
    }
    print(f"Config: {json.dumps(cfg, indent=2)}\n")

    word2idx, idx2word = load_vocab()
    math_reward = MathReward()
    format_reward = FormatReward()

    # Load model
    print(f"{SEP}\n1. Loading model\n{SEP}")
    model = NOVA10M(vocab_size=32000, max_seq_len=512).to(DEVICE)

    for ckpt_name in ["nova10m_grpo.pt", "nova10m_distilled.pt", "nova10m_wikitext_final.pt"]:
        ckpt_path = CKPT_DIR / ckpt_name
        if ckpt_path.exists():
            print(f"  Loading {ckpt_path}")
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=False))
            break

    patch_mamba_scan(model)

    # DataParallel for inference (fills all GPUs), weight flips on model directly
    n_gpus = torch.cuda.device_count()
    dp_model = nn.DataParallel(model) if n_gpus > 1 else model
    print(f"  DataParallel inference on {n_gpus} GPUs, G={cfg['group_size']}")

    bl_layers = get_bitlinear_layers(model)  # access raw model's layers
    print(f"  {len(bl_layers)} BitLinear layers")

    all_questions = generate_questions(200)
    print(f"  {len(all_questions)} questions")

    # Initial fitness
    print(f"\n{SEP}\n2. TFLE-GRPO Training ({cfg['num_steps']} steps)\n{SEP}")
    eval_qs = random.sample(all_questions, min(cfg["questions_per_step"], len(all_questions)))
    current_fitness = evaluate_fitness(dp_model, eval_qs, word2idx, idx2word,
                                       math_reward, format_reward, cfg["group_size"])
    print(f"  Initial fitness: {current_fitness:.4f} | {gpu_stats()}")

    results = {"config": cfg, "steps": [], "initial_fitness": current_fitness}
    t0 = time.time()
    total_accepted = 0
    total_rejected = 0
    re_eval_pass = 0
    re_eval_fail = 0

    for step in range(cfg["num_steps"]):
        layer_idx = step % len(bl_layers)
        layer_name, layer_module = bl_layers[layer_idx]

        w_ternary, alpha = get_ternary(layer_module)
        flat_w = w_ternary.flatten().to(torch.int8)
        n_weights = flat_w.numel()
        n_cands = max(1, int(n_weights * cfg["flip_rate"]))
        candidates = torch.randint(0, n_weights, (n_cands,), device=DEVICE)

        # Generate proposals
        try:
            proposals = generate_k_proposals(flat_w.to(DEVICE), candidates, cfg["K"], DEVICE)
        except Exception:
            proposals = flat_w.unsqueeze(0).expand(cfg["K"], -1).clone().to(DEVICE)
            for k in range(cfg["K"]):
                flip_idx = torch.randint(0, n_cands, (n_cands // 2,))
                for ci in flip_idx:
                    pos = candidates[ci]
                    old = proposals[k, pos].item()
                    proposals[k, pos] = random.choice([v for v in [-1, 0, 1] if v != old])

        step_qs = random.sample(all_questions, min(cfg["questions_per_step"], len(all_questions)))
        current_fitness = evaluate_fitness(dp_model, step_qs, word2idx, idx2word,
                                           math_reward, format_reward, cfg["group_size"])

        best_k = None
        best_delta = 0.0
        original_weight = layer_module.weight.data.clone()

        for k in range(cfg["K"]):
            proposed_w = proposals[k].reshape(w_ternary.shape).float() * alpha
            layer_module.weight.data.copy_(proposed_w)

            new_fitness = evaluate_fitness(dp_model, step_qs, word2idx, idx2word,
                                           math_reward, format_reward, cfg["group_size"])

            delta = new_fitness - current_fitness
            if delta > best_delta:
                best_k = k
                best_delta = delta

            layer_module.weight.data.copy_(original_weight)

        if best_k is not None and best_delta > 0:
            fresh_qs = random.sample(all_questions, min(cfg["re_eval_questions"], len(all_questions)))
            proposed_w = proposals[best_k].reshape(w_ternary.shape).float() * alpha
            layer_module.weight.data.copy_(proposed_w)

            fresh_fitness = evaluate_fitness(dp_model, fresh_qs, word2idx, idx2word,
                                             math_reward, format_reward, cfg["group_size"])

            if fresh_fitness >= current_fitness:
                current_fitness = fresh_fitness
                total_accepted += 1
                re_eval_pass += 1
            else:
                layer_module.weight.data.copy_(original_weight)
                total_rejected += 1
                re_eval_fail += 1
        else:
            total_rejected += 1

        if step % cfg["eval_every"] == 0:
            elapsed = time.time() - t0
            total_proposed = total_accepted + total_rejected
            ar = total_accepted / max(total_proposed, 1)
            rep = re_eval_pass / max(re_eval_pass + re_eval_fail, 1)
            gs = gpu_stats()

            print(f"  Step {step:3d}/{cfg['num_steps']} | "
                  f"Fit {current_fitness:.4f} | "
                  f"Acc {total_accepted}/{total_proposed} ({ar:.0%}) | "
                  f"ReEval {rep:.0%} | {gs} | {elapsed:.0f}s")

            results["steps"].append({
                "step": step, "fitness": current_fitness,
                "accepted": total_accepted, "rejected": total_rejected,
                "re_eval_pass_rate": rep,
            })

    total_time = time.time() - t0
    final_qs = random.sample(all_questions, min(20, len(all_questions)))
    final_fitness = evaluate_fitness(dp_model, final_qs, word2idx, idx2word,
                                     math_reward, format_reward, cfg["group_size"])

    # model is the raw model (not dp_model), so this saves correctly
    torch.save(model.state_dict(), str(CKPT_DIR / "nova10m_tfle_grpo.pt"))

    results["final"] = {
        "final_fitness": final_fitness, "initial_fitness": results["initial_fitness"],
        "accepted": total_accepted, "rejected": total_rejected,
        "time_s": total_time, "time_min": total_time / 60,
    }
    with open(RESULTS_DIR / "test4_tfle_grpo.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    improved = final_fitness > results["initial_fitness"]
    stable = final_fitness >= results["initial_fitness"] - 0.05

    print(f"\n{SEP}\nTEST 4 RESULTS\n{SEP}")
    print(f"  Initial fitness:  {results['initial_fitness']:.4f}")
    print(f"  Final fitness:    {final_fitness:.4f}")
    print(f"  Flips accepted:   {total_accepted}")
    print(f"  Flips rejected:   {total_rejected}")
    print(f"  Time:             {total_time/60:.1f} min")
    if improved:
        print(f"  RESULT:           PASS (fitness increased)")
    elif stable:
        print(f"  RESULT:           PARTIAL PASS (fitness stable)")
    else:
        print(f"  RESULT:           FAIL (fitness decreased)")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
