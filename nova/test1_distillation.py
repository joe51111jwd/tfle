#!/usr/bin/env python3
"""
Test 1: Reasoning Distillation Pipeline — NOVA-10M
===================================================
Fine-tune pretrained NOVA-10M on synthetic reasoning traces.
Validates: loss decreases, format compliance, pipeline stability.

Run: python nova/test1_distillation.py
"""
import sys
import os
import json
import time
import random
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, "/workspace/tfle")
from nova_full_directive import NOVA10M, BitLinear10M

DEVICE = torch.device("cuda:0")
SEP = "=" * 60
CKPT_DIR = Path("/workspace/tfle/checkpoints")
RESULTS_DIR = Path("/workspace/tfle/nova/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
VOCAB_CACHE = Path("/workspace/tfle/nova/vocab.json")


# ── Vocabulary ──────────────────────────────────────────────

def build_vocab(vocab_size=32000):
    """Rebuild exact WikiText-103 vocab from pretrain, with caching."""
    if VOCAB_CACHE.exists():
        with open(VOCAB_CACHE) as f:
            data = json.load(f)
        word2idx = data["word2idx"]
        idx2word = {int(k): v for k, v in data["idx2word"].items()}
        print(f"  Loaded cached vocab: {len(word2idx)} words")
        return word2idx, idx2word, len(word2idx)

    from datasets import load_dataset
    print("  Building WikiText-103 vocabulary (one-time)...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

    word_counts = {}
    for ex in ds:
        for w in ex["text"].split():
            word_counts[w] = word_counts.get(w, 0) + 1

    sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])
    word2idx = {"<pad>": 0, "<unk>": 1, "<eos>": 2}
    for w, _ in sorted_words[:vocab_size - 3]:
        word2idx[w] = len(word2idx)

    idx2word = {v: k for k, v in word2idx.items()}
    actual = len(word2idx)
    print(f"  Vocab built: {actual} words")

    with open(VOCAB_CACHE, "w") as f:
        json.dump({
            "word2idx": word2idx,
            "idx2word": {str(k): v for k, v in idx2word.items()},
        }, f)

    return word2idx, idx2word, actual


# ── Synthetic Data ──────────────────────────────────────────

def generate_traces(n=10000):
    """Generate synthetic math reasoning traces using vocab-safe words."""
    random.seed(42)
    traces = []

    for _ in range(n):
        r = random.random()

        if r < 0.4:
            a, b = random.randint(1, 100), random.randint(1, 100)
            op, verb = random.choice([("+", "add"), ("-", "subtract"), ("*", "multiply")])
            result = eval(f"{a} {op} {b}")
            text = (
                f"Question : What is {a} {op} {b} ? "
                f"Let me think about this . I need to {verb} {a} and {b} . "
                f"The result of {a} {op} {b} is {result} . "
                f"The answer is {result} ."
            )
        elif r < 0.7:
            a, b = random.randint(1, 50), random.randint(1, 50)
            c = random.randint(2, 20)
            s1 = a + b
            result = s1 * c
            text = (
                f"Question : Add {a} and {b} then multiply by {c} . "
                f"Let me think step by step . First I add {a} and {b} to get {s1} . "
                f"Then I multiply {s1} by {c} . "
                f"That gives {s1} times {c} which is {result} . "
                f"The answer is {result} ."
            )
        else:
            items = random.randint(2, 20)
            price = random.randint(1, 10)
            total = items * price
            text = (
                f"Question : You buy {items} items at {price} each . What is the total cost ? "
                f"Let me work through this . Each item costs {price} and there are {items} of them . "
                f"I multiply {items} by {price} to get the total . "
                f"{items} times {price} is {total} . "
                f"The answer is {total} ."
            )

        traces.append(text)

    print(f"  Generated {len(traces)} reasoning traces")
    return traces


class TraceDataset(Dataset):
    def __init__(self, traces, word2idx, seq_len=256):
        self.samples = []
        self.oov = 0
        self.total = 0

        for text in traces:
            tokens = []
            for w in text.split():
                self.total += 1
                idx = word2idx.get(w, 1)
                if idx == 1 and w not in ("<unk>",):
                    self.oov += 1
                tokens.append(idx)
            tokens.append(2)  # <eos>

            if len(tokens) > seq_len + 1:
                tokens = tokens[:seq_len + 1]
            while len(tokens) < seq_len + 1:
                tokens.append(0)

            self.samples.append(torch.tensor(tokens, dtype=torch.long))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        t = self.samples[idx]
        return t[:-1], t[1:]


# ── Generation ──────────────────────────────────────────────

@torch.no_grad()
def generate_text(model, word2idx, idx2word, prompt_words, max_tokens=80, temperature=0.8):
    """Autoregressive generation."""
    model.eval()
    tokens = [word2idx.get(w, 1) for w in prompt_words]
    ids = torch.tensor([tokens], dtype=torch.long, device=DEVICE)

    for _ in range(max_tokens):
        if ids.shape[1] >= 256:
            break
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(ids)
        nxt = logits[0, -1, :].float()
        if temperature > 0:
            probs = F.softmax(nxt / temperature, dim=-1)
            tok = torch.multinomial(probs, 1)
        else:
            tok = nxt.argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, tok.unsqueeze(0)], dim=1)
        if tok.item() == 2:
            break

    return " ".join(idx2word.get(t, "<?>") for t in ids[0].tolist())


def evaluate_format(model, word2idx, idx2word, n=20):
    """Check how many outputs show structured reasoning."""
    prompts = [
        "Question : What is 5 + 3 ?",
        "Question : What is 12 * 4 ?",
        "Question : What is 100 - 37 ?",
        "Question : Add 10 and 20 then multiply by 3 .",
        "Question : You buy 5 items at 3 each . What is the total cost ?",
        "Question : What is 7 + 8 ?",
        "Question : What is 25 * 2 ?",
        "Question : What is 50 - 13 ?",
        "Question : Add 6 and 14 then multiply by 5 .",
        "Question : You buy 10 items at 2 each . What is the total cost ?",
        "Question : What is 33 + 67 ?",
        "Question : What is 9 * 9 ?",
        "Question : What is 200 - 88 ?",
        "Question : Add 15 and 25 then multiply by 4 .",
        "Question : You buy 8 items at 7 each . What is the total cost ?",
        "Question : What is 44 + 56 ?",
        "Question : What is 11 * 6 ?",
        "Question : What is 75 - 29 ?",
        "Question : Add 30 and 40 then multiply by 2 .",
        "Question : You buy 3 items at 9 each . What is the total cost ?",
    ]

    compliant = 0
    samples = []
    for i in range(min(n, len(prompts))):
        words = prompts[i].split()
        out = generate_text(model, word2idx, idx2word, words, max_tokens=80)
        has_answer = "answer" in out.lower()
        has_reasoning = any(w in out.lower() for w in ("think", "step", "first", "multiply", "add", "get", "need"))
        ok = has_answer and has_reasoning
        if ok:
            compliant += 1
        samples.append({"prompt": prompts[i], "output": out[:300], "compliant": ok})

    return compliant / max(len(samples), 1), samples


# ── Main ────────────────────────────────────────────────────

def main():
    print(f"\n{SEP}\nTEST 1: REASONING DISTILLATION PIPELINE\n{SEP}\n")

    cfg = {
        "lr": 2e-5, "epochs": 3, "batch_size": 32, "seq_len": 256,
        "warmup_steps": 100, "weight_decay": 0.01, "grad_clip": 1.0,
        "n_traces": 10000,
    }
    print(f"Config: {json.dumps(cfg, indent=2)}\n")

    # 1) Vocab
    print(f"{SEP}\n1. Building vocabulary\n{SEP}")
    word2idx, idx2word, vocab_size = build_vocab()

    # 2) Model
    print(f"\n{SEP}\n2. Loading NOVA-10M\n{SEP}")
    model = NOVA10M(vocab_size=32000, max_seq_len=512).to(DEVICE)
    p = model.count_parameters()
    print(f"  Params: {p['total']:,}")

    ckpt = CKPT_DIR / "nova10m_wikitext_final.pt"
    print(f"  Loading checkpoint: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False))
    print("  Loaded.")

    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        model = nn.DataParallel(model)
        print(f"  DataParallel on {n_gpus} GPUs")

    # 3) Data
    print(f"\n{SEP}\n3. Generating reasoning traces\n{SEP}")
    traces = generate_traces(cfg["n_traces"])
    split = int(0.9 * len(traces))
    train_ds = TraceDataset(traces[:split], word2idx, cfg["seq_len"])
    val_ds = TraceDataset(traces[split:], word2idx, cfg["seq_len"])
    oov_rate = train_ds.oov / max(train_ds.total, 1)
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | OOV: {oov_rate:.1%}")

    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                          num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                        num_workers=4, pin_memory=True)

    # 4) Optimizer — full fine-tune (model is small enough for 4x32GB)
    print(f"\n{SEP}\n4. Training\n{SEP}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                  weight_decay=cfg["weight_decay"])
    total_steps = cfg["epochs"] * len(train_dl)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")

    results = {"config": cfg, "epochs": [], "samples": []}
    best_val = float("inf")
    step = 0
    t_all = time.time()

    for epoch in range(cfg["epochs"]):
        model.train()
        epoch_loss = 0.0
        n_bat = 0
        t0 = time.time()

        for bi, (x, y) in enumerate(train_dl):
            x, y = x.to(DEVICE), y.to(DEVICE)

            # Warmup
            if step < cfg["warmup_steps"]:
                lr_now = cfg["lr"] * (step + 1) / cfg["warmup_steps"]
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_now

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, 32000), y.reshape(-1), ignore_index=0)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            if step >= cfg["warmup_steps"]:
                scheduler.step()

            epoch_loss += loss.item()
            n_bat += 1
            step += 1

            if bi % 50 == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(f"  E{epoch+1} {bi:4d}/{len(train_dl)} | Loss {loss.item():.4f} | LR {lr:.2e}")

        # Validation
        model.eval()
        vl_sum, vl_n = 0.0, 0
        with torch.no_grad():
            for vx, vy in val_dl:
                vx, vy = vx.to(DEVICE), vy.to(DEVICE)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    vlogits = model(vx)
                    vl = F.cross_entropy(vlogits.reshape(-1, 32000), vy.reshape(-1), ignore_index=0)
                vl_sum += vl.item()
                vl_n += 1

        avg_t = epoch_loss / n_bat
        avg_v = vl_sum / vl_n
        best_val = min(best_val, avg_v)
        elapsed = time.time() - t0

        print(f"\n  Epoch {epoch+1} DONE: train={avg_t:.4f} val={avg_v:.4f} best={best_val:.4f} ({elapsed:.0f}s)")
        results["epochs"].append({
            "epoch": epoch + 1, "train_loss": avg_t, "val_loss": avg_v,
            "best_val": best_val, "time_s": elapsed,
        })

        # Sample generations
        raw = model.module if hasattr(model, "module") else model
        comp, samps = evaluate_format(raw, word2idx, idx2word, n=5)
        print(f"  Format compliance: {comp:.0%}")
        for s in samps[:2]:
            print(f"    Q: {s['prompt']}")
            print(f"    A: {s['output'][:150]}")
            print()
        results["samples"].append({"epoch": epoch + 1, "compliance": comp, "examples": samps})

    # 5) Save
    total_time = time.time() - t_all
    raw = model.module if hasattr(model, "module") else model
    save_path = CKPT_DIR / "nova10m_distilled.pt"
    torch.save(raw.state_dict(), str(save_path))
    print(f"\n  Checkpoint saved: {save_path}")

    # 6) Final eval
    final_comp, final_samps = evaluate_format(raw, word2idx, idx2word, n=20)

    results["final"] = {
        "total_time_s": total_time, "total_time_min": total_time / 60,
        "best_val": best_val, "format_compliance": final_comp,
        "oov_rate": oov_rate, "n_gpus": n_gpus,
        "final_samples": final_samps[:10],
    }
    with open(RESULTS_DIR / "test1_distillation.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    loss_down = results["epochs"][-1]["train_loss"] < results["epochs"][0]["train_loss"]

    print(f"\n{SEP}\nTEST 1 RESULTS\n{SEP}")
    print(f"  Loss decreased:     {'YES' if loss_down else 'NO'}")
    print(f"  Final train loss:   {results['epochs'][-1]['train_loss']:.4f}")
    print(f"  Final val loss:     {results['epochs'][-1]['val_loss']:.4f}")
    print(f"  Format compliance:  {final_comp:.0%}")
    print(f"  Time:               {total_time/60:.1f} min")
    print(f"  RESULT:             {'PASS' if loss_down else 'FAIL'}")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
