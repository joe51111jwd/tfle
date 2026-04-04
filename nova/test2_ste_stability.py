#!/usr/bin/env python3
"""
Test 2: STE Long-Horizon Stability (GPUs 2-3)
===============================================
Continue STE training for 50K steps. Find when val loss diverges.

Run: CUDA_VISIBLE_DEVICES=2,3 python3 nova/test2_ste_stability.py
"""
import sys, os, json, time, math, subprocess
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, "/workspace/tfle")
sys.path.insert(0, "/workspace/tfle/nova")
from nova_full_directive import NOVA10M
from optimize import patch_mamba_scan

DEVICE = torch.device("cuda:0")
SEP = "=" * 60
CKPT_DIR = Path("/workspace/tfle/checkpoints")
RESULTS_DIR = Path("/workspace/tfle/nova/results/test2_ste_stability")
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


class WikiTextDataset(Dataset):
    def __init__(self, split="train", seq_len=256, vocab_size=32000):
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)

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


def main():
    print(f"\n{SEP}\nTEST 2: STE LONG-HORIZON STABILITY (GPUs 2-3)\n{SEP}\n")

    cfg = {
        "lr": 1e-4, "scheduler": "cosine", "weight_decay": 0.01,
        "grad_clip": 1.0, "batch_size": 16, "seq_len": 256,
        "eval_every": 500, "checkpoint_every": 5000, "total_steps": 50000,
    }
    print(f"Config: {json.dumps(cfg, indent=2)}\n")

    # Model
    model = NOVA10M(vocab_size=32000, max_seq_len=512).to(DEVICE)
    patch_mamba_scan(model)

    ckpt = CKPT_DIR / "nova10m_wikitext_final.pt"
    print(f"  Loading: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False))

    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        model = nn.DataParallel(model)
        print(f"  DataParallel on {n_gpus} visible GPUs")

    # Data
    print("  Loading WikiText-103...")
    train_ds = WikiTextDataset("train", seq_len=256, vocab_size=32000)
    val_ds = WikiTextDataset("validation", seq_len=256, vocab_size=32000)
    vocab = train_ds.vocab_size

    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=4, pin_memory=True)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["total_steps"], eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda")

    # Training
    print(f"\n{SEP}\nSTE Training {cfg['total_steps']} steps\n{SEP}")

    model.train()
    total_loss = 0.0
    best_val_loss = float("inf")
    best_val_step = 0
    diverge_step = None
    t0 = time.time()
    train_iter = iter(train_dl)
    results = {"config": cfg, "steps": []}

    for step in range(cfg["total_steps"]):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dl)
            x, y = next(train_iter)

        x, y = x.to(DEVICE), y.to(DEVICE)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1), ignore_index=0)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()

        if step % cfg["eval_every"] == 0 and step > 0:
            # Full validation
            model.eval()
            vl_sum, vl_n = 0.0, 0
            with torch.no_grad():
                for vx, vy in val_dl:
                    vx, vy = vx.to(DEVICE), vy.to(DEVICE)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        vlogits = model(vx)
                        vl = F.cross_entropy(vlogits.reshape(-1, vocab), vy.reshape(-1), ignore_index=0)
                    vl_sum += vl.item()
                    vl_n += 1
                    if vl_n >= 50:
                        break
            val_loss = vl_sum / vl_n
            val_ppl = math.exp(min(val_loss, 20))
            model.train()

            avg_train = total_loss / (step + 1)
            train_ppl = math.exp(min(avg_train, 20))
            gap = val_loss - avg_train
            elapsed = time.time() - t0
            lr = scheduler.get_last_lr()[0]
            sps = (step + 1) / max(elapsed, 1)
            gs = gpu_stats() if step % 2500 == 0 else ""

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_step = step

            # Detect divergence: val loss > best + 1.0
            if diverge_step is None and val_loss > best_val_loss + 1.0:
                diverge_step = step
                print(f"  *** DIVERGENCE DETECTED at step {step} ***")

            print(f"  Step {step:5d}/{cfg['total_steps']} | Train {avg_train:.4f} | "
                  f"Val {val_loss:.4f} | Gap {gap:.4f} | PPL {val_ppl:.2f} | "
                  f"LR {lr:.6f} | {sps:.2f} stp/s {gs}")

            results["steps"].append({
                "step": step, "train_loss": avg_train, "val_loss": val_loss,
                "train_val_gap": gap, "ppl": val_ppl, "lr": lr,
                "best_val": best_val_loss, "best_step": best_val_step,
            })

        if step % cfg["checkpoint_every"] == 0 and step > 0:
            raw = model.module if hasattr(model, "module") else model
            torch.save(raw.state_dict(), str(CKPT_DIR / f"nova10m_ste_step{step}.pt"))

    # Final
    total_time = time.time() - t0
    raw = model.module if hasattr(model, "module") else model
    torch.save(raw.state_dict(), str(CKPT_DIR / "nova10m_ste_50k.pt"))

    results["final"] = {
        "total_steps": cfg["total_steps"],
        "avg_train_loss": total_loss / cfg["total_steps"],
        "best_val_loss": best_val_loss,
        "best_val_step": best_val_step,
        "diverge_step": diverge_step,
        "time_s": total_time, "time_min": total_time / 60,
    }

    with open(RESULTS_DIR / "ste_stability_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{SEP}\nTEST 2 RESULTS\n{SEP}")
    print(f"  Best val loss:    {best_val_loss:.4f} at step {best_val_step}")
    print(f"  Final val loss:   {results['steps'][-1]['val_loss']:.4f}" if results['steps'] else "  No eval steps")
    print(f"  Diverge step:     {diverge_step if diverge_step else 'None (stable)'}")
    print(f"  Time:             {total_time/60:.1f} min")
    if diverge_step:
        print(f"  RESULT:           STE DEGRADES at step {diverge_step}")
    else:
        print(f"  RESULT:           STE STABLE through {cfg['total_steps']} steps")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
