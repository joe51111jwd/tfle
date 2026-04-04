#!/usr/bin/env python3
"""
Test 3: Data Pipeline for Scale — NOVA-10M (GPU-Optimized)
============================================================
Streaming FineWeb-Edu + StarCoder. Batch=16 train (maxes 32GB VRAM).
JIT Mamba scan for 84-99% GPU utilization.

Run: python nova/test3_data_pipeline.py
"""
import sys
import os
import json
import time
import random
import subprocess
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/workspace/tfle")
sys.path.insert(0, "/workspace/tfle/nova")
from nova_full_directive import NOVA10M
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
    return data["word2idx"]


def gpu_stats():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        lines = r.stdout.strip().split("\n")
        parts = [l.split(", ") for l in lines]
        utils = [int(p[0]) for p in parts]
        mems = [f"{int(p[1])}/{int(p[2])}" for p in parts]
        return f"GPU util: {utils} | VRAM(MB): {mems}"
    except Exception:
        return ""


# ── Streaming Data Loader ──────────────────────────────────

class PretrainDataLoader:
    def __init__(self, word2idx, seq_length=256, batch_size=16,
                 text_ratio=0.8, buffer_size=5000):
        self.word2idx = word2idx
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.text_ratio = text_ratio
        self.buffer_size = buffer_size
        self.text_tokens = 0
        self.code_tokens = 0

    def _tokenize(self, text):
        return [self.word2idx.get(w, 1) for w in text.split()]

    def _stream_text(self):
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                          split="train", streaming=True)
        for ex in ds:
            text = ex.get("text", "")
            if text and len(text) > 50:
                tokens = self._tokenize(text)
                if tokens:
                    yield tokens, "text"

    def _stream_code(self):
        from datasets import load_dataset
        try:
            ds = load_dataset("bigcode/starcoderdata", data_dir="python",
                              split="train", streaming=True)
        except Exception:
            try:
                ds = load_dataset("bigcode/the-stack-dedup", data_dir="data/python",
                                  split="train", streaming=True)
            except Exception:
                print("  WARNING: No code dataset, text-only mode")
                return
        for ex in ds:
            content = ex.get("content", ex.get("text", ""))
            if content and len(content) > 50:
                tokens = self._tokenize(content)
                if tokens:
                    yield tokens, "code"

    def stream(self):
        text_gen = self._stream_text()
        code_gen = self._stream_code()

        buffer = []
        token_buf = []
        source_buf = "text"

        def fill_buffer():
            nonlocal token_buf, source_buf
            while len(buffer) < self.buffer_size:
                try:
                    if random.random() < self.text_ratio:
                        tokens, src = next(text_gen)
                    else:
                        tokens, src = next(code_gen)
                except StopIteration:
                    try:
                        tokens, src = next(text_gen)
                    except StopIteration:
                        return False
                token_buf.extend(tokens)
                token_buf.append(2)  # <eos>
                while len(token_buf) >= self.seq_length + 1:
                    seq = token_buf[:self.seq_length + 1]
                    token_buf = token_buf[self.seq_length + 1:]
                    if src == "text":
                        self.text_tokens += self.seq_length
                    else:
                        self.code_tokens += self.seq_length
                    buffer.append(seq)
            return True

        while True:
            if len(buffer) < self.batch_size:
                if not fill_buffer():
                    break
                if len(buffer) < self.batch_size:
                    break
            random.shuffle(buffer)
            batch_seqs = [buffer.pop() for _ in range(min(self.batch_size, len(buffer)))]
            if len(batch_seqs) < self.batch_size:
                continue
            stacked = torch.tensor(batch_seqs, dtype=torch.long)
            yield stacked[:, :-1], stacked[:, 1:]

    def get_ratio(self):
        total = self.text_tokens + self.code_tokens
        if total == 0:
            return 0, 0
        return self.text_tokens / total, self.code_tokens / total


# ── Tests ───────────────────────────────────────────────────

def test_pipeline(word2idx):
    print(f"\n{SEP}\nPart A: Data Pipeline Validation\n{SEP}")
    loader = PretrainDataLoader(word2idx, seq_length=256, batch_size=16, buffer_size=3000)

    tokens_seen = 0
    t0 = time.time()
    target = 500

    idx2word = {int(k): v for k, v in json.load(open(VOCAB_CACHE))["idx2word"].items()}

    for i, (x, y) in enumerate(loader.stream()):
        tokens_seen += x.numel()
        if i == 0:
            sample = " ".join(idx2word.get(t, "<?>") for t in x[0, :40].tolist())
            print(f"  First batch: {x.shape} | Sample: {sample[:120]}")
        if (i + 1) % 100 == 0:
            tps = tokens_seen / max(time.time() - t0, 1e-6)
            print(f"  Batch {i+1}: {tps:,.0f} tok/s | {tokens_seen/1e6:.1f}M tokens")
        if i + 1 >= target:
            break

    elapsed = time.time() - t0
    tps = tokens_seen / max(elapsed, 1e-6)
    text_r, code_r = loader.get_ratio()

    result = {
        "throughput_tokens_per_sec": tps, "total_tokens": tokens_seen,
        "batches": i + 1, "time_s": elapsed,
        "text_ratio": text_r, "code_ratio": code_r,
    }
    print(f"\n  Throughput:   {tps:,.0f} tok/s")
    print(f"  Text/Code:    {text_r:.0%} / {code_r:.0%}")
    return result


def short_pretrain(word2idx):
    print(f"\n{SEP}\nPart B: 2000-step pretrain on mixed data (batch=16, 4 GPUs max)\n{SEP}")

    model = NOVA10M(vocab_size=32000, max_seq_len=512).to(DEVICE)
    ckpt = CKPT_DIR / "nova10m_wikitext_final.pt"
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False))
    patch_mamba_scan(model)

    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        model = nn.DataParallel(model)
    print(f"  Model loaded | {n_gpus} GPUs | batch=16 (4/GPU, fills 32GB VRAM)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda")

    loader = PretrainDataLoader(word2idx, seq_length=256, batch_size=16, buffer_size=3000)

    model.train()
    total_loss = 0.0
    t0 = time.time()
    step = 0
    losses = []

    for x, y in loader.stream():
        x, y = x.to(DEVICE), y.to(DEVICE)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, 32000), y.reshape(-1), ignore_index=0)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        losses.append(loss.item())
        step += 1

        if step % 100 == 0:
            avg = total_loss / step
            elapsed = time.time() - t0
            sps = step / elapsed
            lr = scheduler.get_last_lr()[0]
            gs = gpu_stats()
            print(f"  Step {step:4d}/2000 | Loss {loss.item():.4f} | Avg {avg:.4f} | "
                  f"LR {lr:.6f} | {sps:.2f} step/s | {gs}")

        if step >= 2000:
            break

    elapsed = time.time() - t0
    first_200 = sum(losses[:200]) / 200 if len(losses) >= 200 else losses[0]
    last_200 = sum(losses[-200:]) / 200 if len(losses) >= 200 else losses[-1]
    trend = "decreasing" if last_200 < first_200 - 0.05 else ("flat" if abs(last_200 - first_200) < 0.1 else "increasing")

    result = {
        "steps": step, "avg_loss": total_loss / max(step, 1),
        "first_200_avg": first_200, "last_200_avg": last_200,
        "loss_trend": trend, "time_s": elapsed, "time_min": elapsed / 60,
    }
    print(f"\n  Steps: {step} | Avg: {result['avg_loss']:.4f} | Trend: {trend} | {elapsed/60:.1f} min")
    return result


def main():
    print(f"\n{SEP}\nTEST 3: DATA PIPELINE FOR SCALE (GPU-OPTIMIZED)\n{SEP}\n")

    word2idx = load_vocab()
    print(f"  Vocab: {len(word2idx)} words")

    pipeline_results = test_pipeline(word2idx)
    pretrain_results = short_pretrain(word2idx)

    all_results = {"pipeline": pipeline_results, "pretrain_2k": pretrain_results}
    with open(RESULTS_DIR / "test3_data_pipeline.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    tps = pipeline_results["throughput_tokens_per_sec"]
    trend = pretrain_results["loss_trend"]

    print(f"\n{SEP}\nTEST 3 RESULTS\n{SEP}")
    print(f"  Throughput:     {tps:,.0f} tok/s ({'PASS' if tps > 50000 else 'PARTIAL'})")
    print(f"  Loss trend:     {trend}")
    passed = tps > 10000 and trend in ("decreasing", "flat")
    print(f"  RESULT:         {'PASS' if passed else 'FAIL'}")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
