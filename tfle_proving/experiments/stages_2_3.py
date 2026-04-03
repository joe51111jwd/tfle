"""Stage 2 (BPE) + Stage 3 (Code) — run sequentially on one GPU.

Stage 2: Bigram-extended vocab (768 tokens) on Shakespeare
Stage 3: Character-level on Python code corpus
"""

from __future__ import annotations

import argparse
import ast
import inspect
import os
import sys
import textwrap
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, ROOT)

from tfle.config import TFLEConfig, FitnessType, CoolingSchedule, SelectionMethod
from tfle_proving.data.loader import download_shakespeare
from tfle_proving.data.tokenizer import BigramTokenizer
from tfle_proving.models.char_lm import CharLM, STECharLM
from tfle_proving.training.tfle_text_trainer_v2 import TFLETextTrainerV2
from tfle_proving.training.ste_text_trainer import STETextTrainer
from tfle_proving.training.utils import setup_device, compute_perplexity, save_results, plot_loss_curves


# ========== DATA ==========

class TokenDataset(Dataset):
    """Generic token-level dataset for next-token prediction."""
    def __init__(self, tokens: list[int], context_len: int, stride: int = 1):
        self.data = torch.tensor(tokens, dtype=torch.long)
        self.context_len = context_len
        self.stride = stride
        self.n = (len(self.data) - context_len - 1) // stride + 1

    def __len__(self): return self.n

    def __getitem__(self, idx):
        start = idx * self.stride
        return self.data[start:start + self.context_len], self.data[start + self.context_len]


def get_python_corpus() -> str:
    """Collect Python code from stdlib + synthetic functions."""
    code_parts = []

    # Grab source from common stdlib modules
    safe_modules = [
        'json', 'os.path', 'collections', 'functools', 'itertools',
        'textwrap', 'string', 'math', 'statistics', 'random',
        'datetime', 'pathlib', 'dataclasses', 'typing', 'enum',
        'abc', 'copy', 'operator', 'bisect', 'heapq',
    ]
    for mod_name in safe_modules:
        try:
            mod = __import__(mod_name)
            src = inspect.getsource(mod)
            code_parts.append(src[:20000])  # Cap per module
        except Exception:
            pass

    # Synthetic Python functions
    synthetics = [
        'def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n',
        'def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n',
        'def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr\n',
        'def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1\n',
        'class Stack:\n    def __init__(self):\n        self.items = []\n\n    def push(self, item):\n        self.items.append(item)\n\n    def pop(self):\n        return self.items.pop()\n\n    def is_empty(self):\n        return len(self.items) == 0\n',
        'def flatten(lst):\n    result = []\n    for item in lst:\n        if isinstance(item, list):\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result\n',
        'def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)\n\ndef merge(left, right):\n    result = []\n    i = j = 0\n    while i < len(left) and j < len(right):\n        if left[i] <= right[j]:\n            result.append(left[i])\n            i += 1\n        else:\n            result.append(right[j])\n            j += 1\n    result.extend(left[i:])\n    result.extend(right[j:])\n    return result\n',
        'def count_words(text):\n    words = text.lower().split()\n    counts = {}\n    for word in words:\n        counts[word] = counts.get(word, 0) + 1\n    return counts\n',
    ]
    code_parts.extend(synthetics * 20)  # Repeat for more data

    corpus = "\n\n".join(code_parts)
    print(f"Python corpus: {len(corpus):,} characters")
    return corpus


def make_config(total_steps, K=128, flip_rate=0.008):
    return TFLEConfig(
        fitness_type=FitnessType.TASK_LOSS,
        flip_rate=flip_rate,
        selection_method=SelectionMethod.TRACE_WEIGHTED,
        protection_threshold=0.3,
        num_parallel_proposals=K,
        initial_temperature=1.5, min_temperature=0.1,
        cooling_schedule=CoolingSchedule.COSINE,
        reheat_on_plateau=True, plateau_window=3000, reheat_factor=2.5,
        trace_decay=0.95, separate_pos_neg_traces=True,
        total_training_steps=total_steps,
        min_candidates_per_step=10, max_candidates_fraction=0.05,
        cdll_alpha_start=0.05,
        exploration_rate=0.005, exploration_min=0.001,
        depth_scaled_flip_rate=True, flip_rate_depth_scale=0.85,
        depth_scaled_temperature=True, temperature_depth_scale=0.8,
    )


# ========== STAGE 2 ==========

def run_stage2(device, steps=20000):
    print(f"\n{'='*60}")
    print("STAGE 2: BPE Token Prediction (768 vocab)")
    print(f"{'='*60}")

    base = os.path.join(ROOT, "tfle_proving")
    results_dir = os.path.join(base, "results", "stage2")
    checkpoint_dir = os.path.join(base, "checkpoints", "stage2")

    text = download_shakespeare()
    tokenizer = BigramTokenizer(text, n_bigrams=512)
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}")

    tokens = tokenizer.encode(text)
    split = int(len(tokens) * 0.9)
    ctx_len = 128

    train_ds = TokenDataset(tokens[:split], ctx_len, stride=3)
    val_ds = TokenDataset(tokens[split:], ctx_len, stride=1)
    print(f"Train: {len(train_ds):,}, Val: {len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)

    # TFLE model — larger for BPE
    hidden_sizes = [1024, 512, 512]
    embed_dim = 48
    config = make_config(steps, K=128)

    model = CharLM(vocab_size, embed_dim, ctx_len, hidden_sizes, config, device)
    print(f"Ternary params: {model.get_ternary_param_count():,}")

    trainer = TFLETextTrainerV2(
        model, config, train_loader, val_loader, device,
        embed_lr=1e-3, reeval=False,  # No reeval for from-scratch
    )
    tfle_log = trainer.train(steps, eval_every=500, checkpoint_every=5000,
                              checkpoint_dir=checkpoint_dir, results_dir=results_dir)

    # STE baseline
    ste = STECharLM(vocab_size, embed_dim, ctx_len, hidden_sizes).to(device)
    ste_trainer = STETextTrainer(ste, train_loader, val_loader, device, lr=1e-3)
    ste_log = ste_trainer.train(steps, eval_every=500, results_dir=results_dir)

    # Compare
    tfle_final = tfle_log[-1] if tfle_log else {}
    ste_final = ste_log[-1] if ste_log else {}
    gap = tfle_final.get("val_perplexity", 999) / max(ste_final.get("val_perplexity", 1), 0.01)

    summary = {
        "vocab_size": vocab_size,
        "tfle_final_loss": tfle_final.get("val_loss"),
        "tfle_final_ppl": tfle_final.get("val_perplexity"),
        "ste_final_loss": ste_final.get("val_loss"),
        "ste_final_ppl": ste_final.get("val_perplexity"),
        "gap": f"{gap:.1f}x",
        "status": "GOOD" if gap < 2 else ("PASS" if gap < 5 else "FAIL"),
    }
    save_results(summary, os.path.join(results_dir, "stage2_summary.json"))
    plot_loss_curves(tfle_log, ste_log, os.path.join(results_dir, "stage2_curves.png"),
                     title="Stage 2: BPE Token Prediction")

    print(f"\nStage 2: {summary['status']} — TFLE ppl={summary['tfle_final_ppl']:.1f}, "
          f"STE ppl={summary['ste_final_ppl']:.1f}, gap={summary['gap']}")
    return summary


# ========== STAGE 3 ==========

def run_stage3(device, steps=20000):
    print(f"\n{'='*60}")
    print("STAGE 3: Code Prediction (Python)")
    print(f"{'='*60}")

    base = os.path.join(ROOT, "tfle_proving")
    results_dir = os.path.join(base, "results", "stage3")
    checkpoint_dir = os.path.join(base, "checkpoints", "stage3")

    code_text = get_python_corpus()
    vocab_size = 256  # byte-level
    ctx_len = 128

    split = int(len(code_text) * 0.9)
    train_data = torch.tensor([ord(c) % 256 for c in code_text[:split]], dtype=torch.long)
    val_data = torch.tensor([ord(c) % 256 for c in code_text[split:]], dtype=torch.long)

    class CodeDataset(Dataset):
        def __init__(self, data, ctx_len, stride=3):
            self.data = data; self.ctx_len = ctx_len; self.stride = stride
            self.n = (len(data) - ctx_len - 1) // stride + 1
        def __len__(self): return self.n
        def __getitem__(self, i):
            s = i * self.stride
            return self.data[s:s+self.ctx_len], self.data[s+self.ctx_len]

    train_ds = CodeDataset(train_data, ctx_len); val_ds = CodeDataset(val_data, ctx_len, stride=1)
    print(f"Train: {len(train_ds):,}, Val: {len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)

    hidden_sizes = [512, 512, 256]
    config = make_config(steps, K=128)

    model = CharLM(vocab_size, 32, ctx_len, hidden_sizes, config, device)
    print(f"Ternary params: {model.get_ternary_param_count():,}")

    trainer = TFLETextTrainerV2(
        model, config, train_loader, val_loader, device,
        embed_lr=1e-3, reeval=False,
    )
    tfle_log = trainer.train(steps, eval_every=500, checkpoint_every=5000,
                              checkpoint_dir=checkpoint_dir, results_dir=results_dir)

    # STE baseline
    ste = STECharLM(vocab_size, 32, ctx_len, hidden_sizes).to(device)
    ste_trainer = STETextTrainer(ste, train_loader, val_loader, device, lr=1e-3)
    ste_log = ste_trainer.train(steps, eval_every=500, results_dir=results_dir)

    # Code-specific metrics: try to parse generated samples
    model_device = device
    syntax_valid = 0
    for _ in range(100):
        sample = model.generate_text("def ", length=150, temperature=0.7)
        try:
            ast.parse(sample)
            syntax_valid += 1
        except SyntaxError:
            pass

    tfle_final = tfle_log[-1] if tfle_log else {}
    ste_final = ste_log[-1] if ste_log else {}
    gap = tfle_final.get("val_perplexity", 999) / max(ste_final.get("val_perplexity", 1), 0.01)

    summary = {
        "tfle_final_loss": tfle_final.get("val_loss"),
        "tfle_final_ppl": tfle_final.get("val_perplexity"),
        "ste_final_loss": ste_final.get("val_loss"),
        "ste_final_ppl": ste_final.get("val_perplexity"),
        "gap": f"{gap:.1f}x",
        "syntax_valid_rate": f"{syntax_valid}/100",
        "status": "GOOD" if gap < 2 else ("PASS" if gap < 5 else "FAIL"),
    }
    save_results(summary, os.path.join(results_dir, "stage3_summary.json"))
    plot_loss_curves(tfle_log, ste_log, os.path.join(results_dir, "stage3_curves.png"),
                     title="Stage 3: Code Prediction")

    print(f"\nStage 3: {summary['status']} — TFLE ppl={summary['tfle_final_ppl']:.1f}, "
          f"STE ppl={summary['ste_final_ppl']:.1f}, syntax={summary['syntax_valid_rate']}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20000)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0")
    print(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(0)}")

    s2 = run_stage2(device, args.steps)
    s3 = run_stage3(device, args.steps)

    print(f"\n{'='*60}")
    print("STAGES 2+3 COMPLETE")
    print(f"Stage 2: {s2['status']} (gap={s2['gap']})")
    print(f"Stage 3: {s3['status']} (gap={s3['gap']}, syntax={s3['syntax_valid_rate']})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
