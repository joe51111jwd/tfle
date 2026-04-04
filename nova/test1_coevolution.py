#!/usr/bin/env python3
"""
Test 1: Co-Evolution on NOVA-10M (GPUs 0-1)
=============================================
Three-phase co-evolution: embed-only → gentle TFLE attention → full TFLE.
Proves gradient-free ternary training works on 74.5M hybrid Mamba+Attention.

Run: CUDA_VISIBLE_DEVICES=0,1 python3 nova/test1_coevolution.py
"""
import sys, os, json, time, math, random, subprocess, copy
from pathlib import Path
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, "/workspace/tfle")
sys.path.insert(0, "/workspace/tfle/nova")
from nova_full_directive import NOVA10M, BitLinear10M, MambaBlock10M
from optimize import patch_mamba_scan

DEVICE = torch.device("cuda:0")
SEP = "=" * 60
CKPT_DIR = Path("/workspace/tfle/checkpoints")
RESULTS_DIR = Path("/workspace/tfle/nova/results/test1_coevolution")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
VOCAB_CACHE = Path("/workspace/tfle/nova/vocab.json")


def gpu_stats():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        lines = r.stdout.strip().split("\n")
        utils = [int(l.split(", ")[0]) for l in lines]
        return f"GPU: {utils}%"
    except Exception:
        return ""


# ── Data ────────────────────────────────────────────────────

class WikiTextDataset(Dataset):
    def __init__(self, split="train", seq_len=256, vocab_size=32000):
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)

        # Load cached vocab if exists, else build
        if VOCAB_CACHE.exists():
            with open(VOCAB_CACHE) as f:
                data = json.load(f)
            self.word2idx = data["word2idx"]
        else:
            word_counts = {}
            for ex in ds:
                for w in ex["text"].split():
                    word_counts[w] = word_counts.get(w, 0) + 1
            sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])
            self.word2idx = {"<pad>": 0, "<unk>": 1, "<eos>": 2}
            for w, _ in sorted_words[:vocab_size - 3]:
                self.word2idx[w] = len(self.word2idx)

        self.vocab_size = min(len(self.word2idx), vocab_size)

        all_tokens = []
        for ex in ds:
            text = ex["text"].strip()
            if not text:
                continue
            tokens = [self.word2idx.get(w, 1) for w in text.split()]
            tokens.append(2)
            all_tokens.extend(tokens)

        self.tokens = torch.tensor(all_tokens, dtype=torch.long)
        self.seq_len = seq_len
        self.n_samples = (len(self.tokens) - 1) // seq_len

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        return self.tokens[start:start + self.seq_len], self.tokens[start + 1:start + self.seq_len + 1]


# ── Layer classification ────────────────────────────────────

def classify_layers(model):
    """Classify model parameters into ternary (BitLinear) and float (embed/norm/etc)."""
    ternary_attn = []  # BitLinear in attention blocks
    ternary_mamba = []  # BitLinear in mamba blocks
    float_params = []   # Embeddings, LayerNorm, LM head, etc.

    for name, param in model.named_parameters():
        is_bitlinear = False
        for mname, module in model.named_modules():
            if isinstance(module, BitLinear10M) and name.startswith(mname + ".weight"):
                is_bitlinear = True
                # Determine if attention or mamba
                parts = name.split(".")
                layer_idx = None
                for i, p in enumerate(parts):
                    if p == "layers" and i + 1 < len(parts):
                        try:
                            layer_idx = int(parts[i + 1])
                        except ValueError:
                            pass
                if layer_idx is not None:
                    pattern = model.layer_pattern[layer_idx]
                    if pattern == "A":
                        ternary_attn.append((name, module, param))
                    else:
                        ternary_mamba.append((name, module, param))
                else:
                    ternary_attn.append((name, module, param))
                break

        if not is_bitlinear:
            float_params.append((name, param))

    return ternary_attn, ternary_mamba, float_params


def get_bitlinear_layers(model, layer_type="attention_only"):
    """Get BitLinear layers for TFLE, filtered by type."""
    layers = []
    for name, module in model.named_modules():
        if not isinstance(module, BitLinear10M):
            continue
        # Determine parent layer type
        parts = name.split(".")
        layer_idx = None
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                try:
                    layer_idx = int(parts[i + 1])
                except ValueError:
                    pass
        if layer_idx is None:
            continue

        pattern = model.layer_pattern[layer_idx]
        if layer_type == "attention_only" and pattern != "A":
            continue
        if layer_type == "mamba_only" and pattern != "M":
            continue
        layers.append((name, module, pattern))

    return layers


# ── TFLE step ───────────────────────────────────────────────

def tfle_step(model, layer_module, train_batch, val_batch, config, vocab_size):
    """One TFLE step: propose K flips, pick best, re-eval on fresh data."""
    x_t, y_t = train_batch
    x_v, y_v = val_batch
    x_t, y_t = x_t.to(DEVICE), y_t.to(DEVICE)
    x_v, y_v = x_v.to(DEVICE), y_v.to(DEVICE)

    K = config["K"]
    flip_rate = config["flip_rate"]

    # Get ternary weights
    with torch.no_grad():
        alpha = torch.mean(torch.abs(layer_module.weight)).clamp(min=1e-10)
        w_ternary = torch.clamp(torch.round(layer_module.weight.data / alpha), -1, 1)

    flat_w = w_ternary.flatten()
    n_weights = flat_w.numel()
    n_flips = max(1, int(n_weights * flip_rate))

    # Current loss
    model.eval()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = model(x_t)
        current_loss = F.cross_entropy(logits.reshape(-1, vocab_size), y_t.reshape(-1), ignore_index=0).item()

    # Try K proposals
    best_improvement = 0.0
    best_flip_indices = None
    best_new_vals = None
    original_weight = layer_module.weight.data.clone()

    for k in range(K):
        flip_indices = torch.randint(0, n_weights, (n_flips,))
        old_vals = flat_w[flip_indices].clone()

        # Random ternary flip: cycle -1→0→1→-1
        new_vals = old_vals.clone()
        for i in range(len(new_vals)):
            v = new_vals[i].item()
            new_vals[i] = {-1: 0, 0: 1, 1: -1}.get(int(v), 0)

        # Apply
        flat_w_copy = flat_w.clone()
        flat_w_copy[flip_indices] = new_vals
        layer_module.weight.data.copy_(flat_w_copy.reshape(w_ternary.shape).float() * alpha)

        # Evaluate
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x_t)
            new_loss = F.cross_entropy(logits.reshape(-1, vocab_size), y_t.reshape(-1), ignore_index=0).item()

        improvement = current_loss - new_loss
        if improvement > best_improvement:
            best_improvement = improvement
            best_flip_indices = flip_indices.clone()
            best_new_vals = new_vals.clone()

        # Revert
        layer_module.weight.data.copy_(original_weight)

    # Accept best if improves
    if best_flip_indices is not None and best_improvement > 0:
        # Apply best
        flat_w_best = flat_w.clone()
        flat_w_best[best_flip_indices] = best_new_vals
        layer_module.weight.data.copy_(flat_w_best.reshape(w_ternary.shape).float() * alpha)

        # Re-eval on validation
        tolerance = config.get("re_eval_tolerance", 0.05)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x_v)
            val_loss = F.cross_entropy(logits.reshape(-1, vocab_size), y_v.reshape(-1), ignore_index=0).item()

        if val_loss <= current_loss + tolerance:
            return {"accepted": True, "improvement": best_improvement, "train_loss": current_loss - best_improvement}
        else:
            layer_module.weight.data.copy_(original_weight)
            return {"accepted": False, "reason": "re_eval_failed"}

    return {"accepted": False, "reason": "no_improvement"}


# ── Eval ────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, val_dl, vocab_size, max_batches=50):
    model.eval()
    total_loss, n = 0.0, 0
    for vx, vy in val_dl:
        vx, vy = vx.to(DEVICE), vy.to(DEVICE)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(vx)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), vy.reshape(-1), ignore_index=0)
        total_loss += loss.item()
        n += 1
        if n >= max_batches:
            break
    avg = total_loss / max(n, 1)
    ppl = math.exp(min(avg, 20))
    return avg, ppl


def randomization_test(model, val_dl, vocab_size):
    """Measure gap between learned ternary weights vs random."""
    # Save current
    saved_states = {}
    for name, module in model.named_modules():
        if isinstance(module, BitLinear10M):
            saved_states[name] = module.weight.data.clone()

    # Measure current
    current_loss, current_ppl = evaluate(model, val_dl, vocab_size)

    # Randomize all ternary weights
    for name, module in model.named_modules():
        if isinstance(module, BitLinear10M):
            shape = module.weight.shape
            alpha = torch.mean(torch.abs(module.weight)).clamp(min=1e-10)
            random_ternary = torch.randint(-1, 2, shape, device=module.weight.device).float() * alpha
            module.weight.data.copy_(random_ternary)

    random_loss, random_ppl = evaluate(model, val_dl, vocab_size)

    # Restore
    for name, module in model.named_modules():
        if isinstance(module, BitLinear10M):
            if name in saved_states:
                module.weight.data.copy_(saved_states[name])

    return {
        "learned_loss": current_loss, "learned_ppl": current_ppl,
        "random_loss": random_loss, "random_ppl": random_ppl,
        "gap": random_ppl / max(current_ppl, 1e-6),
    }


# ── Main ────────────────────────────────────────────────────

def main():
    print(f"\n{SEP}\nTEST 1: CO-EVOLUTION ON NOVA-10M (GPUs 0-1)\n{SEP}\n")

    # Load model
    model = NOVA10M(vocab_size=32000, max_seq_len=512).to(DEVICE)
    patch_mamba_scan(model)

    ckpt = CKPT_DIR / "nova10m_wikitext_final.pt"
    print(f"  Loading: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False))

    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        model = nn.DataParallel(model)
        print(f"  DataParallel on {n_gpus} visible GPUs")

    raw_model = model.module if hasattr(model, "module") else model

    # Data
    print("  Loading WikiText-103...")
    train_ds = WikiTextDataset("train", seq_len=256, vocab_size=32000)
    val_ds = WikiTextDataset("validation", seq_len=256, vocab_size=32000)
    vocab_size = train_ds.vocab_size

    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    # Initial eval
    init_loss, init_ppl = evaluate(model, val_dl, vocab_size)
    print(f"  Initial: loss={init_loss:.4f} ppl={init_ppl:.2f}")

    results = {"initial": {"loss": init_loss, "ppl": init_ppl}, "phases": {}}
    all_logs = []
    total_accepted = 0
    total_rejected = 0
    accept_window = deque(maxlen=500)
    t_start = time.time()

    # ── PHASE 1: Embed-only (steps 0-3000) ──────────────────

    print(f"\n{SEP}\nPHASE 1: Embed-only training (0-3000)\n{SEP}")

    # Freeze ALL ternary weights, train only float params
    for name, param in raw_model.named_parameters():
        param.requires_grad = False
    # Unfreeze: embeddings, norms, lm_head
    for name, param in raw_model.named_parameters():
        if any(k in name for k in ["embed", "norm", "lm_head"]):
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({trainable/total:.1%})")

    embed_optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4, weight_decay=0.01)
    embed_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(embed_optimizer, T_max=3000, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda")

    train_iter = iter(train_dl)
    for step in range(3000):
        model.train()
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dl)
            x, y = next(train_iter)
        x, y = x.to(DEVICE), y.to(DEVICE)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1), ignore_index=0)

        embed_optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(embed_optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(embed_optimizer)
        scaler.update()
        embed_scheduler.step()

        if step % 100 == 0:
            val_loss, val_ppl = evaluate(model, val_dl, vocab_size)
            elapsed = time.time() - t_start
            gs = gpu_stats() if step % 500 == 0 else ""
            print(f"  P1 Step {step:5d} | TrLoss {loss.item():.4f} | Val {val_loss:.4f} | "
                  f"PPL {val_ppl:.2f} | {elapsed:.0f}s {gs}")
            all_logs.append({"step": step, "phase": "1", "train_loss": loss.item(),
                             "val_loss": val_loss, "ppl": val_ppl})

        if step % 1000 == 0 and step > 0:
            raw = model.module if hasattr(model, "module") else model
            torch.save(raw.state_dict(), str(CKPT_DIR / f"nova10m_coevo_p1_step{step}.pt"))

    phase1_loss, phase1_ppl = evaluate(model, val_dl, vocab_size)
    print(f"\n  Phase 1 DONE: loss={phase1_loss:.4f} ppl={phase1_ppl:.2f} (floor)")
    results["phases"]["1"] = {"final_loss": phase1_loss, "final_ppl": phase1_ppl}

    # ── PHASE 2: Gentle TFLE on Attention (3000-8000) ────────

    print(f"\n{SEP}\nPHASE 2: Gentle TFLE on Attention layers (3000-8000)\n{SEP}")

    raw_model = model.module if hasattr(model, "module") else model
    attn_layers = get_bitlinear_layers(raw_model, "attention_only")
    print(f"  {len(attn_layers)} attention BitLinear layers for TFLE")

    # Embed optimizer continues but slower
    embed_optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=5e-5, weight_decay=0.01)
    embed_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(embed_optimizer, T_max=5000, eta_min=1e-6)

    tfle_config = {"K": 32, "flip_rate": 0.001, "re_eval_tolerance": 0.05}

    val_iter = iter(val_dl)
    for step in range(3000, 8000):
        model.train()
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dl)
            x, y = next(train_iter)
        try:
            vx, vy = next(val_iter)
        except StopIteration:
            val_iter = iter(val_dl)
            vx, vy = next(val_iter)

        x, y = x.to(DEVICE), y.to(DEVICE)

        # TFLE step on one attention layer (cycling)
        layer_idx = (step - 3000) % len(attn_layers)
        layer_name, layer_module, _ = attn_layers[layer_idx]

        raw_m = model.module if hasattr(model, "module") else model
        result = tfle_step(raw_m, layer_module, (x, y), (vx, vy), tfle_config, vocab_size)

        if result["accepted"]:
            total_accepted += 1
            accept_window.append(1)
        else:
            total_rejected += 1
            accept_window.append(0)

        # Co-evolution: embed backprop step
        model.train()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1), ignore_index=0)
        embed_optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(embed_optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(embed_optimizer)
        scaler.update()
        embed_scheduler.step()

        if step % 100 == 0:
            val_loss, val_ppl = evaluate(model, val_dl, vocab_size)
            elapsed = time.time() - t_start
            ar = sum(accept_window) / max(len(accept_window), 1)
            gs = gpu_stats() if step % 500 == 0 else ""
            print(f"  P2 Step {step:5d} | Val {val_loss:.4f} | PPL {val_ppl:.2f} | "
                  f"AR {ar:.0%} | Acc {total_accepted} | {elapsed:.0f}s {gs}")
            all_logs.append({"step": step, "phase": "2", "val_loss": val_loss,
                             "ppl": val_ppl, "accept_rate": ar, "accepted": total_accepted})

        if step % 1000 == 0:
            raw = model.module if hasattr(model, "module") else model
            torch.save(raw.state_dict(), str(CKPT_DIR / f"nova10m_coevo_p2_step{step}.pt"))

    phase2_loss, phase2_ppl = evaluate(model, val_dl, vocab_size)
    print(f"\n  Phase 2 DONE: loss={phase2_loss:.4f} ppl={phase2_ppl:.2f}")
    results["phases"]["2"] = {"final_loss": phase2_loss, "final_ppl": phase2_ppl}

    # Randomization test at step 8000
    print("  Running randomization test...")
    raw_m = model.module if hasattr(model, "module") else model
    rand_test = randomization_test(raw_m, val_dl, vocab_size)
    print(f"  Learned PPL: {rand_test['learned_ppl']:.2f} | Random PPL: {rand_test['random_ppl']:.2f} | Gap: {rand_test['gap']:.2f}x")
    results["randomization_8000"] = rand_test

    # ── PHASE 3a: Full TFLE on Attention (8000-14000) ────────

    print(f"\n{SEP}\nPHASE 3a: Full TFLE on Attention (8000-14000)\n{SEP}")

    tfle_config = {"K": 128, "flip_rate": 0.005, "re_eval_tolerance": 0.05}
    embed_optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-5, weight_decay=0.01)
    embed_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(embed_optimizer, T_max=6000, eta_min=1e-6)

    for step in range(8000, 14000):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dl)
            x, y = next(train_iter)
        try:
            vx, vy = next(val_iter)
        except StopIteration:
            val_iter = iter(val_dl)
            vx, vy = next(val_iter)
        x, y = x.to(DEVICE), y.to(DEVICE)

        layer_idx = (step - 8000) % len(attn_layers)
        _, layer_module, _ = attn_layers[layer_idx]

        # Cosine tolerance decay
        progress = (step - 8000) / 6000
        tolerance = 0.05 * (1 + math.cos(math.pi * progress)) / 2 + 0.01
        tfle_config["re_eval_tolerance"] = tolerance

        raw_m = model.module if hasattr(model, "module") else model
        result = tfle_step(raw_m, layer_module, (x, y), (vx, vy), tfle_config, vocab_size)
        if result["accepted"]:
            total_accepted += 1
            accept_window.append(1)
        else:
            total_rejected += 1
            accept_window.append(0)

        # Embed co-evolution
        model.train()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1), ignore_index=0)
        embed_optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(embed_optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(embed_optimizer)
        scaler.update()
        embed_scheduler.step()

        if step % 100 == 0:
            val_loss, val_ppl = evaluate(model, val_dl, vocab_size)
            elapsed = time.time() - t_start
            ar = sum(accept_window) / max(len(accept_window), 1)
            gs = gpu_stats() if step % 500 == 0 else ""
            print(f"  P3a Step {step:5d} | Val {val_loss:.4f} | PPL {val_ppl:.2f} | "
                  f"AR {ar:.0%} | Tol {tolerance:.3f} | {elapsed:.0f}s {gs}")
            all_logs.append({"step": step, "phase": "3a", "val_loss": val_loss,
                             "ppl": val_ppl, "accept_rate": ar})

        if step % 2000 == 0:
            raw = model.module if hasattr(model, "module") else model
            torch.save(raw.state_dict(), str(CKPT_DIR / f"nova10m_coevo_p3a_step{step}.pt"))

    phase3a_loss, phase3a_ppl = evaluate(model, val_dl, vocab_size)
    print(f"\n  Phase 3a DONE: loss={phase3a_loss:.4f} ppl={phase3a_ppl:.2f}")
    results["phases"]["3a"] = {"final_loss": phase3a_loss, "final_ppl": phase3a_ppl}

    rand_test_14k = randomization_test(raw_m, val_dl, vocab_size)
    print(f"  Randomization test: learned={rand_test_14k['learned_ppl']:.2f} random={rand_test_14k['random_ppl']:.2f} gap={rand_test_14k['gap']:.2f}x")
    results["randomization_14000"] = rand_test_14k

    # ── PHASE 3b: Add Mamba layers (14000-25000) ─────────────

    print(f"\n{SEP}\nPHASE 3b: Full TFLE on ALL layers (14000-25000)\n{SEP}")

    all_layers = get_bitlinear_layers(raw_model, "all")
    mamba_layers = get_bitlinear_layers(raw_model, "mamba_only")
    print(f"  {len(all_layers)} total BitLinear layers ({len(attn_layers)} attn + {len(mamba_layers)} mamba)")

    for step in range(14000, 25000):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dl)
            x, y = next(train_iter)
        try:
            vx, vy = next(val_iter)
        except StopIteration:
            val_iter = iter(val_dl)
            vx, vy = next(val_iter)
        x, y = x.to(DEVICE), y.to(DEVICE)

        layer_idx = (step - 14000) % len(all_layers)
        layer_name, layer_module, layer_type = all_layers[layer_idx]

        # Different configs for attn vs mamba
        if layer_type == "A":
            cfg = {"K": 128, "flip_rate": 0.005, "re_eval_tolerance": 0.03}
        else:
            cfg = {"K": 32, "flip_rate": 0.001, "re_eval_tolerance": 0.05}

        raw_m = model.module if hasattr(model, "module") else model
        result = tfle_step(raw_m, layer_module, (x, y), (vx, vy), cfg, vocab_size)
        if result["accepted"]:
            total_accepted += 1
            accept_window.append(1)
        else:
            total_rejected += 1
            accept_window.append(0)

        # Embed co-evolution
        model.train()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1), ignore_index=0)
        embed_optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(embed_optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(embed_optimizer)
        scaler.update()

        if step % 100 == 0:
            val_loss, val_ppl = evaluate(model, val_dl, vocab_size)
            elapsed = time.time() - t_start
            ar = sum(accept_window) / max(len(accept_window), 1)
            gs = gpu_stats() if step % 500 == 0 else ""
            print(f"  P3b Step {step:5d} | Val {val_loss:.4f} | PPL {val_ppl:.2f} | "
                  f"AR {ar:.0%} | Layer {layer_type} | {elapsed:.0f}s {gs}")
            all_logs.append({"step": step, "phase": "3b", "val_loss": val_loss,
                             "ppl": val_ppl, "accept_rate": ar})

        if step % 2000 == 0:
            raw = model.module if hasattr(model, "module") else model
            torch.save(raw.state_dict(), str(CKPT_DIR / f"nova10m_coevo_p3b_step{step}.pt"))

    # Final
    total_time = time.time() - t_start
    final_loss, final_ppl = evaluate(model, val_dl, vocab_size)
    raw = model.module if hasattr(model, "module") else model
    torch.save(raw.state_dict(), str(CKPT_DIR / "nova10m_coevolution_final.pt"))

    rand_test_25k = randomization_test(raw, val_dl, vocab_size)
    results["phases"]["3b"] = {"final_loss": final_loss, "final_ppl": final_ppl}
    results["randomization_25000"] = rand_test_25k
    results["final"] = {
        "loss": final_loss, "ppl": final_ppl,
        "total_accepted": total_accepted, "total_rejected": total_rejected,
        "time_s": total_time, "time_min": total_time / 60,
    }
    results["logs"] = all_logs

    with open(RESULTS_DIR / "coevolution_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Pass/fail
    below_floor = final_ppl < phase1_ppl
    strong = final_ppl < phase1_ppl * 0.8

    print(f"\n{SEP}\nTEST 1 RESULTS\n{SEP}")
    print(f"  Phase 1 floor:  PPL {phase1_ppl:.2f}")
    print(f"  Final:          PPL {final_ppl:.2f}")
    print(f"  Below floor:    {'YES' if below_floor else 'NO'}")
    print(f"  Accepted flips: {total_accepted}")
    print(f"  Time:           {total_time/60:.1f} min")
    if strong:
        print(f"  RESULT:         STRONG PASS (>{20}% below floor)")
    elif below_floor:
        print(f"  RESULT:         PASS")
    else:
        print(f"  RESULT:         FAIL")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
