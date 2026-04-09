#!/usr/bin/env python3
"""
NOVA Benchmark Evaluation Runner
==================================
Runs GSM8K, MATH-500, MMLU, HumanEval on a NOVA checkpoint.
Supports base (raw) and bracket inference modes.

Usage:
  python nova/scripts/eval.py --checkpoint /data/checkpoints/ckpt_step50000.pt
  python nova/scripts/eval.py --checkpoint ... --bracket --bracket_size 64
  python nova/scripts/eval.py --checkpoint ... --suite full
  python nova/scripts/eval.py --checkpoint ... --benchmark gsm8k --num_samples 100
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nova.model import Nova2_4B
from nova.model.config import NOVA_1B, NOVA_1B_QWEN, NOVA_2_4B, NOVA_10M


# ── Benchmark configs ───────────────────────────────────

BENCHMARKS = {
    "gsm8k": {
        "dataset": "gsm8k", "subset": "main", "split": "test",
        "scorer": "numeric", "description": "Grade school math (1319 questions)",
    },
    "math500": {
        "dataset": "hendrycks/competition_math", "subset": None, "split": "test",
        "scorer": "numeric", "description": "Competition math (500 questions)",
        "max_samples": 500,
    },
    "mmlu": {
        "dataset": "cais/mmlu", "subset": "all", "split": "test",
        "scorer": "exact_match", "description": "Broad knowledge (14042 questions)",
    },
    "humaneval": {
        "dataset": "openai/openai_humaneval", "subset": None, "split": "test",
        "scorer": "code_pass", "description": "Python code generation (164 problems)",
    },
}

TARGETS = {
    "gsm8k": {"base": "20-35%", "bracket": "50-65%"},
    "math500": {"base": "10-20%", "bracket": "30-45%"},
    "mmlu": {"base": "29-34%", "bracket": "35-42%"},
    "humaneval": {"base": "2-8%", "bracket": "8-15%"},
}

SUITES = {
    "quick": ["gsm8k"],
    "core": ["gsm8k", "math500", "mmlu"],
    "full": ["gsm8k", "math500", "mmlu", "humaneval"],
}


# ── Model loading ───────────────────────────────────────

def load_model(checkpoint_path: str, config_name: str = "NOVA_1B_QWEN") -> Nova2_4B:
    configs = {"NOVA_1B": NOVA_1B, "NOVA_1B_QWEN": NOVA_1B_QWEN,
               "NOVA_2_4B": NOVA_2_4B, "NOVA_10M": NOVA_10M}
    config = configs.get(config_name, NOVA_1B_QWEN)

    model = Nova2_4B.from_config(config)

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    # Handle torch.compile _orig_mod prefix
    cleaned = {}
    for k, v in state.items():
        k = k.replace("_orig_mod.", "").replace("module.", "")
        cleaned[k] = v
    model.load_state_dict(cleaned, strict=False)

    model.eval()
    model.cuda()
    print(f"Loaded {config_name} from {checkpoint_path}")
    return model


# ── Generation ──────────────────────────────────────────

@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 512,
             temperature: float = 0.0) -> str:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
    if input_ids.shape[1] > 2048:
        input_ids = input_ids[:, -2048:]

    for _ in range(max_new_tokens):
        if input_ids.shape[1] >= 4096:
            break
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids)
        next_logits = logits[0, -1, :].float()

        if temperature <= 0:
            next_token = next_logits.argmax(dim=-1, keepdim=True)
        else:
            probs = torch.softmax(next_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1)

        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    output_ids = input_ids[0, len(tokenizer.encode(prompt)):].tolist()
    return tokenizer.decode(output_ids, skip_special_tokens=True)


# ── Scoring ─────────────────────────────────────────────

def extract_number(text: str) -> float | None:
    import re
    # GSM8K format: #### NUMBER
    m = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if m:
        return float(m.group(1).replace(",", ""))
    # General: last number in text
    nums = re.findall(r"-?[\d,]+\.?\d*", text)
    if nums:
        return float(nums[-1].replace(",", ""))
    return None


def score_numeric(prediction: str, reference: str) -> bool:
    pred_num = extract_number(prediction)
    ref_num = extract_number(reference)
    if pred_num is None or ref_num is None:
        return False
    return abs(pred_num - ref_num) < 1e-4


def score_exact(prediction: str, reference: str) -> bool:
    return prediction.strip().upper() == reference.strip().upper()


def score_code(prediction: str, test_code: str) -> bool:
    import subprocess
    import tempfile
    code = prediction + "\n" + test_code
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        f.flush()
        try:
            result = subprocess.run(
                [sys.executable, f.name],
                capture_output=True, timeout=10,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, Exception):
            return False
        finally:
            os.unlink(f.name)


SCORERS = {
    "numeric": score_numeric,
    "exact_match": score_exact,
    "code_pass": score_code,
}


# ── Benchmark loader ───────────────────────────────────

def load_benchmark(name: str, max_samples: int | None = None):
    from datasets import load_dataset
    cfg = BENCHMARKS[name]
    ds = load_dataset(cfg["dataset"], cfg.get("subset"), split=cfg["split"])
    samples = []

    for i, ex in enumerate(ds):
        if max_samples and i >= max_samples:
            break
        if name == "gsm8k":
            samples.append({
                "prompt": f"Solve this math problem step by step:\n{ex['question']}\n\nAnswer:",
                "reference": ex["answer"],
            })
        elif name == "math500":
            samples.append({
                "prompt": f"Solve this math problem:\n{ex['problem']}\n\nAnswer:",
                "reference": ex.get("solution", ex.get("answer", "")),
            })
        elif name == "mmlu":
            choices = ex.get("choices", [])
            letters = "ABCD"
            choice_text = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices))
            samples.append({
                "prompt": f"{ex['question']}\n{choice_text}\n\nAnswer:",
                "reference": letters[ex["answer"]] if isinstance(ex["answer"], int) else str(ex["answer"]),
            })
        elif name == "humaneval":
            samples.append({
                "prompt": ex["prompt"],
                "reference": ex.get("test", ex.get("canonical_solution", "")),
            })

    return samples


# ── Run benchmark ───────────────────────────────────────

def run_benchmark(model, tokenizer, name: str, max_samples: int | None = None,
                  bracket_size: int = 0, verbose: bool = False) -> dict:
    print(f"\n{'='*50}")
    print(f"Running: {name} ({BENCHMARKS[name]['description']})")
    if bracket_size > 1:
        print(f"  Mode: bracket inference (size={bracket_size})")
    else:
        print(f"  Mode: base (greedy)")
    print(f"{'='*50}")

    samples = load_benchmark(name, max_samples)
    scorer = SCORERS[BENCHMARKS[name]["scorer"]]

    correct = 0
    total = len(samples)
    results = []
    t0 = time.time()

    for i, sample in enumerate(samples):
        if bracket_size > 1:
            # Generate multiple candidates, pick best by self-judging
            candidates = []
            temps = torch.linspace(0.3, 1.0, bracket_size).tolist()
            for t in temps:
                resp = generate(model, tokenizer, sample["prompt"],
                                max_new_tokens=512, temperature=t)
                candidates.append(resp)
            # Simple majority vote on extracted answer
            from collections import Counter
            answers = [extract_number(c) for c in candidates]
            answers = [a for a in answers if a is not None]
            if answers:
                prediction = str(Counter(answers).most_common(1)[0][0])
            else:
                prediction = candidates[0] if candidates else ""
        else:
            prediction = generate(model, tokenizer, sample["prompt"],
                                  max_new_tokens=512, temperature=0.0)

        is_correct = scorer(prediction, sample["reference"])
        if is_correct:
            correct += 1
        results.append({"correct": is_correct, "prediction": prediction[:200]})

        if verbose and (i < 3 or (i + 1) % 50 == 0):
            print(f"  [{i+1}/{total}] {'OK' if is_correct else 'WRONG'} "
                  f"| pred={prediction[:60]}... | ref={sample['reference'][:60]}...")

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            acc = correct / (i + 1)
            print(f"  Progress: {i+1}/{total} | Acc: {acc:.1%} | {elapsed:.0f}s")

    elapsed = time.time() - t0
    accuracy = correct / max(total, 1)

    print(f"\n  {name}: {accuracy:.1%} ({correct}/{total}) in {elapsed:.0f}s")
    if name in TARGETS:
        mode = "bracket" if bracket_size > 1 else "base"
        print(f"  Target ({mode}): {TARGETS[name][mode]}")

    return {
        "benchmark": name, "accuracy": accuracy,
        "correct": correct, "total": total,
        "bracket_size": bracket_size, "time_s": elapsed,
        "results": results[:20],  # Save first 20 for inspection
    }


# ── Main ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NOVA Benchmark Evaluation")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--config", default="NOVA_1B_QWEN", help="Model config name")
    parser.add_argument("--benchmark", default=None, help="Single benchmark to run")
    parser.add_argument("--suite", default="core", choices=SUITES.keys(),
                        help="Benchmark suite (quick/core/full)")
    parser.add_argument("--num_samples", type=int, default=None, help="Limit samples per benchmark")
    parser.add_argument("--bracket", action="store_true", help="Use bracket inference")
    parser.add_argument("--bracket_size", type=int, default=16, help="Bracket size (candidates)")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--verbose", action="store_true", help="Print per-question results")
    args = parser.parse_args()

    # Load model
    model = load_model(args.checkpoint, args.config)

    # Load tokenizer
    try:
        from nova.training.tokenizer_setup import get_qwen_tokenizer
        tokenizer = get_qwen_tokenizer()
    except Exception:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Determine benchmarks
    if args.benchmark:
        benchmarks = [args.benchmark]
    else:
        benchmarks = SUITES[args.suite]

    bracket_size = args.bracket_size if args.bracket else 0

    # Run
    all_results = {}
    for name in benchmarks:
        if name not in BENCHMARKS:
            print(f"Unknown benchmark: {name}, skipping")
            continue
        result = run_benchmark(model, tokenizer, name,
                               max_samples=args.num_samples,
                               bracket_size=bracket_size,
                               verbose=args.verbose)
        all_results[name] = result

    # Summary
    print(f"\n{'='*60}")
    print("NOVA Benchmark Results")
    print(f"{'='*60}")
    print(f"{'Benchmark':<15} {'Accuracy':>10} {'Correct':>10} {'Target':>15}")
    print("-" * 55)
    for name, r in all_results.items():
        mode = "bracket" if bracket_size > 1 else "base"
        target = TARGETS.get(name, {}).get(mode, "—")
        print(f"{name:<15} {r['accuracy']:>9.1%} {r['correct']:>5}/{r['total']:<4} {target:>15}")
    print(f"{'='*60}")

    # Save
    output_path = args.output or f"eval_results_{int(time.time())}.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
