#!/usr/bin/env python3
"""
Fast STE Pretrain — NOVA-10M on WikiText-103
=============================================
Uses JIT Mamba scan for max GPU utilization.
All 4 GPUs via DataParallel. Batch=16 (fills 32GB VRAM).

Run: python nova/pretrain_fast.py
"""
import sys, os, json, time, subprocess
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
CKPT_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path("/workspace/tfle/nova/results")
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
        mems = [int(l.split(", ")[1]) for l in lines]
        return f"GPU: {utils}% | VRAM: {[f'{m}MB' for m in mems]}"
    except Exception:
        return ""


class WikiTextDataset(Dataset):
    def __init__(self, split="train", seq_len=256, vocab_size=32000):
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)

        print(f"  Building vocab from {split}...")
        word_counts = {}
        for ex in ds:
            for w in ex["text"].split():
                word_counts[w] = word_counts.get(w, 0) + 1

        sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])
        self.word2idx = {"<pad>": 0, "<unk>": 1, "<eos>": 2}
        for w, _ in sorted_words[:vocab_size - 3]:
            self.word2idx[w] = len(self.word2idx)
        self.vocab_size = min(len(self.word2idx), vocab_size)
        self.idx2word = {v: k for k, v in self.word2idx.items()}

        print(f"  Tokenizing {len(ds)} examples (vocab={self.vocab_size})...")
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
        print(f"  Total tokens: {len(self.tokens):,} | Samples: {self.n_samples:,}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = self.tokens[start:start + self.seq_len]
        y = self.tokens[start + 1:start + self.seq_len + 1]
        return x, y


def main():
    print(f"\n{SEP}\nNOVA-10M FAST STE PRETRAIN — WikiText-103, 4x RTX 5090\n{SEP}\n")

    # Build model with JIT scan
    model = NOVA10M(vocab_size=32000, max_seq_len=512).to(DEVICE)
    patch_mamba_scan(model)
    p = model.count_parameters()
    print(f"  Params: {p['total']:,}")

    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        model = nn.DataParallel(model)
        print(f"  DataParallel on {n_gpus} GPUs")

    # Load data
    train_ds = WikiTextDataset("train", seq_len=256, vocab_size=32000)
    val_ds = WikiTextDataset("validation", seq_len=256, vocab_size=32000)

    # Cache vocab for other scripts
    if not VOCAB_CACHE.exists():
        vocab_data = {
            "word2idx": train_ds.word2idx,
            "idx2word": {str(k): v for k, v in train_ds.idx2word.items()},
        }
        with open(VOCAB_CACHE, "w") as f:
            json.dump(vocab_data, f)
        print(f"  Vocab cached to {VOCAB_CACHE}")

    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    vocab = train_ds.vocab_size
    print(f"  Vocab: {vocab} | Train: {len(train_ds):,} | Val: {len(val_ds):,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda")

    # Training
    model.train()
    total_loss = 0.0
    best_val_loss = float("inf")
    t0 = time.time()
    steps = 10000
    train_iter = iter(train_dl)

    print(f"\n{SEP}\nTraining {steps} steps\n{SEP}")

    for step in range(steps):
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
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()

        if step % 100 == 0:
            avg = total_loss / (step + 1)
            elapsed = time.time() - t0
            sps = (step + 1) / max(elapsed, 1)
            lr = scheduler.get_last_lr()[0]
            gs = gpu_stats() if step % 500 == 0 else ""

            val_str = ""
            if step % 1000 == 0 and step > 0:
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
                val_avg = vl_sum / vl_n
                best_val_loss = min(best_val_loss, val_avg)
                val_str = f" | Val: {val_avg:.4f} | Best: {best_val_loss:.4f}"
                model.train()

            print(f"  Step {step:5d}/{steps} | Loss: {loss.item():.4f} | Avg: {avg:.4f} | "
                  f"LR: {lr:.6f} | {sps:.2f} stp/s{val_str} {gs}")

        if step % 2000 == 0 and step > 0:
            raw = model.module if hasattr(model, "module") else model
            torch.save(raw.state_dict(), str(CKPT_DIR / f"nova10m_wikitext_step{step}.pt"))
            print(f"  Checkpoint saved: step {step}")

    # Final save
    elapsed = time.time() - t0
    raw = model.module if hasattr(model, "module") else model
    torch.save(raw.state_dict(), str(CKPT_DIR / "nova10m_wikitext_final.pt"))

    result = {
        "steps": steps, "avg_loss": total_loss / steps, "best_val_loss": best_val_loss,
        "time_s": elapsed, "time_min": elapsed / 60, "gpus": n_gpus, "vocab": vocab,
    }
    with open(RESULTS_DIR / "pretrain_fast.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{SEP}")
    print(f"DONE: avg={total_loss/steps:.4f} best_val={best_val_loss:.4f} ({elapsed/60:.1f}min)")
    print(f"Checkpoint: {CKPT_DIR / 'nova10m_wikitext_final.pt'}")
    print(f"{SEP}")


if __name__ == "__main__":
    main()
