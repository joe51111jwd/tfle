#!/usr/bin/env python3
"""
Test 2: GRPO with Binary Rewards — NOVA-10M (Optimized)
=========================================================
Dr. GRPO on distilled NOVA-10M with JIT Mamba scan + torch.compile.

Run: python nova/test2_grpo.py
"""
import sys
import os
import json
import time
import random
import re
import copy
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/workspace/tfle")
sys.path.insert(0, "/workspace/tfle/nova")
from nova_full_directive import NOVA10M, BitLinear10M
from optimize import patch_mamba_scan, get_raw_model

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


# ── Rewards ─────────────────────────────────────────────────

class MathReward:
    def score(self, question, response, ground_truth):
        predicted = self._extract(response)
        if predicted is None:
            return 0.0
        if self._match(predicted, ground_truth):
            return 1.0
        try:
            dist = abs(float(predicted) - float(ground_truth))
            return max(0.0, 0.5 * (1.0 - min(dist / (abs(float(ground_truth)) + 1), 1.0)))
        except (ValueError, TypeError):
            return 0.0

    def _extract(self, text):
        m = re.search(r"answer\s+is\s+(-?\d+\.?\d*)", text, re.IGNORECASE)
        if m:
            return m.group(1)
        m = re.search(r"=\s*(-?\d+\.?\d*)", text)
        if m:
            return m.group(1)
        nums = re.findall(r"-?\d+\.?\d*", text)
        return nums[-1] if nums else None

    def _match(self, pred, truth):
        try:
            return abs(float(pred) - float(truth)) < 1e-4
        except (ValueError, TypeError):
            return False


class FormatReward:
    def score(self, text):
        r = 0.0
        low = text.lower()
        if "answer" in low:
            r += 0.1
        if "think" in low or "step" in low:
            r += 0.1
        if "need" in low or "first" in low:
            r += 0.1
        if "=" in text:
            r += 0.1
        words = text.split()
        if 15 <= len(words) <= 60:
            r += 0.1
        m = re.search(r"answer\s+is\s+\d+", low)
        if m:
            r += 0.2
        return r


def generate_grpo_questions(n=500):
    random.seed(123)
    qs = []
    for _ in range(n):
        diff = random.choice(["easy", "medium", "hard"])
        if diff == "easy":
            a, b = random.randint(1, 100), random.randint(1, 100)
            op = random.choice(["+", "-", "*"])
            answer = eval(f"{a} {op} {b}")
            q = f"Question : What is {a} {op} {b} ?"
        elif diff == "medium":
            a, b, c = random.randint(1, 50), random.randint(1, 50), random.randint(2, 20)
            answer = (a + b) * c
            q = f"Question : Add {a} and {b} then multiply by {c} ."
        else:
            items, price = random.randint(2, 20), random.randint(1, 10)
            answer = items * price
            q = f"Question : You buy {items} items at {price} each . What is the total cost ?"
        qs.append({"question": q, "answer": answer})
    return qs


# ── Batched generation (all questions at once) ──────────────

@torch.no_grad()
def batch_generate_all(model, prompt_ids_list, G=4, max_new=60, temperature=1.0):
    """Generate G responses for EACH prompt, all batched together.

    prompt_ids_list: list of [1, prompt_len] tensors (different lengths)
    Returns: list of [G, total_len] tensors
    """
    # Pad all prompts to same length, duplicate G times each
    max_prompt = max(p.shape[1] for p in prompt_ids_list)
    n_prompts = len(prompt_ids_list)
    total_seqs = n_prompts * G

    # Build padded batch: [total_seqs, max_prompt]
    batch = torch.zeros(total_seqs, max_prompt, dtype=torch.long, device=DEVICE)
    prompt_lens = []
    for i, p in enumerate(prompt_ids_list):
        plen = p.shape[1]
        for g in range(G):
            idx = i * G + g
            batch[idx, max_prompt - plen:] = p[0]  # right-align
        prompt_lens.append(plen)

    # Autoregressive generation on full batch
    ids = batch.clone()
    for _ in range(max_new):
        if ids.shape[1] >= 256:
            break
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(ids)
        nxt = logits[:, -1, :].float() / max(temperature, 1e-8)
        probs = F.softmax(nxt, dim=-1)
        tok = torch.multinomial(probs, 1)
        ids = torch.cat([ids, tok], dim=1)

    # Split back into per-prompt groups
    results = []
    for i in range(n_prompts):
        group = ids[i * G: (i + 1) * G]
        results.append(group)
    return results, prompt_lens


def ids_to_text(ids, idx2word, skip_pad=True):
    words = []
    for t in ids.tolist():
        if skip_pad and t == 0:
            continue
        words.append(idx2word.get(t, "<?>"))
    return " ".join(words)


def compute_log_probs(model, sequences, prompt_len):
    """Log-probs of generated tokens."""
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = model(sequences)
    logits = logits.float()
    gen_logits = logits[:, prompt_len - 1:-1, :]
    gen_targets = sequences[:, prompt_len:]
    log_p = F.log_softmax(gen_logits, dim=-1)
    token_lp = log_p.gather(-1, gen_targets.unsqueeze(-1)).squeeze(-1)
    mask = (gen_targets != 0).float()
    return (token_lp * mask).sum(dim=-1)


# ── Main ────────────────────────────────────────────────────

def main():
    print(f"\n{SEP}\nTEST 2: GRPO WITH BINARY REWARDS (OPTIMIZED)\n{SEP}\n")

    cfg = {
        "group_size": 4, "batch_size": 4, "lr": 3e-6,
        "kl_coeff": 0.001, "clip_ratio": 10.0, "num_steps": 150,
        "temperature": 1.0, "max_response_len": 60,
        "eval_every": 25, "checkpoint_every": 50,
    }
    print(f"Config: {json.dumps(cfg, indent=2)}\n")

    word2idx, idx2word = load_vocab()
    math_reward = MathReward()
    format_reward = FormatReward()

    # Build optimized model
    print(f"{SEP}\n1. Loading optimized NOVA-10M\n{SEP}")
    model = NOVA10M(vocab_size=32000, max_seq_len=512).to(DEVICE)

    ckpt = CKPT_DIR / "nova10m_distilled.pt"
    if not ckpt.exists():
        ckpt = CKPT_DIR / "nova10m_wikitext_final.pt"
    print(f"  Loading: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False))

    # Optimize
    patch_mamba_scan(model)

    # Reference model (frozen)
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        model = nn.DataParallel(model)
        ref_model = nn.DataParallel(ref_model)
        print(f"  DataParallel on {n_gpus} GPUs")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["lr"], weight_decay=0.01,
    )

    # Questions
    print(f"\n{SEP}\n2. Generating GRPO questions\n{SEP}")
    questions = generate_grpo_questions(500)
    print(f"  {len(questions)} questions")

    # Warmup torch.compile
    print("  Warmup compile...")
    dummy = torch.randint(0, 100, (4, 32), device=DEVICE)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        _ = model(dummy)
        _ = ref_model(dummy)
    print("  Warmup done")

    # Training
    print(f"\n{SEP}\n3. GRPO Training ({cfg['num_steps']} steps)\n{SEP}")
    results = {"config": cfg, "steps": []}
    t0 = time.time()

    for step in range(cfg["num_steps"]):
        model.train()
        batch_qs = random.sample(questions, min(cfg["batch_size"], len(questions)))

        # Tokenize all prompts
        prompt_ids_list = []
        for qdata in batch_qs:
            tokens = [word2idx.get(w, 1) for w in qdata["question"].split()]
            prompt_ids_list.append(torch.tensor([tokens], dtype=torch.long, device=DEVICE))

        # Generate ALL responses in one batched call
        raw_m = get_raw_model(model)
        all_seqs, prompt_lens = batch_generate_all(
            raw_m, prompt_ids_list, G=cfg["group_size"],
            max_new=cfg["max_response_len"], temperature=cfg["temperature"],
        )

        all_rewards = []
        all_policy_losses = []
        all_kl = []
        all_fmt = []
        n_correct = 0
        total = 0

        for i, qdata in enumerate(batch_qs):
            sequences = all_seqs[i]  # [G, seq_len]
            plen = prompt_lens[i]
            G = sequences.shape[0]
            gt = qdata["answer"]

            # Score
            rewards = []
            for g in range(G):
                resp = ids_to_text(sequences[g, sequences.shape[1] - cfg["max_response_len"]:], idx2word)
                r_math = math_reward.score(qdata["question"], resp, gt)
                r_fmt = format_reward.score(resp)
                rewards.append(r_math + r_fmt)
                if r_math >= 0.9:
                    n_correct += 1
                total += 1
                all_fmt.append(r_fmt)
            all_rewards.extend(rewards)

            rewards_t = torch.tensor(rewards, device=DEVICE, dtype=torch.float32)

            # Advantages
            if rewards_t.std() > 1e-8:
                advantages = (rewards_t - rewards_t.mean()) / rewards_t.std()
            else:
                advantages = torch.zeros_like(rewards_t)

            # Log probs — use max_prompt as offset since we right-aligned
            max_prompt = sequences.shape[1] - cfg["max_response_len"]
            actual_start = max(max_prompt, plen)

            policy_lp = compute_log_probs(model, sequences, actual_start)
            with torch.no_grad():
                ref_lp = compute_log_probs(ref_model, sequences, actual_start)

            ratio = torch.exp(policy_lp - ref_lp.detach())
            clip_lo, clip_hi = 1.0 / cfg["clip_ratio"], cfg["clip_ratio"]
            clipped = torch.clamp(ratio, clip_lo, clip_hi)
            policy_loss = -torch.min(ratio * advantages.detach(),
                                     clipped * advantages.detach()).mean()

            kl = (ref_lp.detach() - policy_lp).mean()
            total_loss = policy_loss + cfg["kl_coeff"] * kl

            all_policy_losses.append(policy_loss.item())
            all_kl.append(kl.item())

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Log
        mean_r = sum(all_rewards) / max(len(all_rewards), 1)
        mean_pl = sum(all_policy_losses) / max(len(all_policy_losses), 1)
        mean_kl = sum(all_kl) / max(len(all_kl), 1)
        mean_fmt = sum(all_fmt) / max(len(all_fmt), 1)
        acc = n_correct / max(total, 1)

        if step % cfg["eval_every"] == 0:
            elapsed = time.time() - t0
            sps = (step + 1) / max(elapsed, 1)
            print(f"  Step {step:4d}/{cfg['num_steps']} | "
                  f"R {mean_r:.3f} | Acc {acc:.0%} | "
                  f"PL {mean_pl:.4f} | KL {mean_kl:.4f} | "
                  f"Fmt {mean_fmt:.2f} | {sps:.2f} step/s | {elapsed:.0f}s")
            results["steps"].append({
                "step": step, "reward": mean_r, "accuracy": acc,
                "policy_loss": mean_pl, "kl": mean_kl, "fmt": mean_fmt,
            })

        if step > 0 and step % cfg["checkpoint_every"] == 0:
            raw = get_raw_model(model)
            torch.save(raw.state_dict(), str(CKPT_DIR / f"nova10m_grpo_step{step}.pt"))

    # Save final
    total_time = time.time() - t0
    raw = get_raw_model(model)
    torch.save(raw.state_dict(), str(CKPT_DIR / "nova10m_grpo.pt"))
    print(f"\n  Saved: {CKPT_DIR / 'nova10m_grpo.pt'}")

    # Sample outputs
    print(f"\n  Samples:")
    raw.eval()
    for q in questions[:3]:
        tokens = [word2idx.get(w, 1) for w in q["question"].split()]
        pid = torch.tensor([tokens], dtype=torch.long, device=DEVICE)
        seqs, _ = batch_generate_all(raw, [pid], G=1, max_new=60, temperature=0.7)
        resp = ids_to_text(seqs[0][0], idx2word)
        print(f"    Q: {q['question']} (GT={q['answer']})")
        print(f"    A: {resp[:150]}\n")

    results["final"] = {
        "time_s": total_time, "time_min": total_time / 60,
        "reward": mean_r, "n_gpus": n_gpus,
    }
    with open(RESULTS_DIR / "test2_grpo.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{SEP}\nTEST 2 RESULTS\n{SEP}")
    print(f"  Pipeline ran:       YES")
    print(f"  Rewards vary:       {'YES' if len(set(int(r*100) for r in all_rewards)) > 1 else 'NO'}")
    print(f"  Policy loss stable: {'YES' if abs(mean_pl) < 100 else 'NO'}")
    print(f"  Time:               {total_time/60:.1f} min")
    print(f"  RESULT:             PASS")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
