"""Local trace generation from a quantized teacher model.

Loads a teacher (e.g. DeepSeek-R1-Distill-Qwen-14B) in int4 via bitsandbytes,
generates reasoning traces in <think>...</think><answer>...</answer> format,
verifies answers where possible, and saves as JSONL.

Usage:
    python -m nova.training.generate_traces \
        --teacher deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
        --num-traces 10000 \
        --output ./traces \
        --gpus 0,1,2,3
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class TraceConfig:
    teacher_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    num_traces: int = 10000
    output_dir: str = "./traces"
    gpus: list[int] | None = None
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    batch_size: int = 4


def _extract_answer(text: str) -> str | None:
    """Extract the final answer from a <answer>...</answer> block."""
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


def _extract_thinking(text: str) -> str | None:
    """Extract the thinking from a <think>...</think> block."""
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


def _verify_math_answer(answer: str, expected: str) -> bool:
    """Check if a math answer matches the expected value numerically."""
    def _parse_number(s: str) -> float | None:
        s = s.strip().replace(",", "").replace("$", "").replace("%", "")
        # extract last number in the string
        nums = re.findall(r"-?\d+\.?\d*", s)
        if nums:
            try:
                return float(nums[-1])
            except ValueError:
                return None
        return None

    got = _parse_number(answer)
    expected_num = _parse_number(expected)
    if got is not None and expected_num is not None:
        return abs(got - expected_num) < 1e-4
    return answer.strip().lower() == expected.strip().lower()


def _verify_code_answer(code: str) -> bool:
    """Try to execute a code snippet and check it doesn't error."""
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()
            result = subprocess.run(
                [sys.executable, f.name],
                capture_output=True,
                timeout=10,
                text=True,
            )
            os.unlink(f.name)
            return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


class TraceGenerator:
    """Generates reasoning traces from a quantized teacher model.

    Loads the teacher in int4 using bitsandbytes, then generates
    chain-of-thought traces for each prompt and verifies correctness.
    """

    def __init__(self, config: TraceConfig, gpu_id: int = 0):
        self.config = config
        self.gpu_id = gpu_id
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load the teacher model in int4 quantization."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        logger.info(f"Loading {self.config.teacher_model} in int4 on GPU {self.gpu_id}")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.teacher_model,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.teacher_model,
            quantization_config=bnb_config,
            device_map={"": self.gpu_id},
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        self.model.eval()
        logger.info(f"Teacher loaded on GPU {self.gpu_id}")

    def _build_prompt(self, problem: dict) -> str:
        """Build a prompt that elicits structured reasoning."""
        question = problem["question"]
        return (
            f"Solve the following problem step by step. "
            f"Show your reasoning inside <think>...</think> tags, "
            f"then give your final answer inside <answer>...</answer> tags.\n\n"
            f"Problem: {question}\n"
        )

    @torch.no_grad()
    def generate_trace(self, problem: dict) -> dict | None:
        """Generate a single reasoning trace for a problem.

        Returns a dict with problem, thinking, answer fields or None on failure.
        """
        prompt = self._build_prompt(problem)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_new_tokens,
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        # ensure the response has the expected structure
        thinking = _extract_thinking(response)
        answer = _extract_answer(response)

        if not thinking or not answer:
            # try to salvage: wrap the whole response as thinking, last line as answer
            lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
            if len(lines) >= 2:
                thinking = " ".join(lines[:-1])
                answer = lines[-1]
            elif lines:
                thinking = lines[0]
                answer = lines[0]
            else:
                return None

        # verify correctness where possible
        verified = None
        expected = problem.get("answer")
        if expected is not None:
            verified = _verify_math_answer(answer, str(expected))

        problem_type = problem.get("type", "math")
        if problem_type == "code" and answer:
            verified = _verify_code_answer(answer)

        return {
            "problem": problem["question"],
            "thinking": thinking,
            "answer": answer,
            "verified": verified,
            "source": problem.get("source", "unknown"),
        }

    def generate_batch(self, problems: list[dict]) -> list[dict]:
        """Generate traces for a batch of problems."""
        traces = []
        for problem in problems:
            trace = self.generate_trace(problem)
            if trace is not None:
                traces.append(trace)
        return traces

    def generate_all(self, problems: list[dict]) -> list[dict]:
        """Generate traces for all problems, with progress logging."""
        if self.model is None:
            self.load_model()

        traces = []
        total = len(problems)
        t0 = time.time()

        for i, problem in enumerate(problems):
            trace = self.generate_trace(problem)
            if trace is not None:
                traces.append(trace)

            if (i + 1) % 10 == 0 or i == total - 1:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                logger.info(
                    f"GPU {self.gpu_id}: {i + 1}/{total} traces "
                    f"({len(traces)} valid, {rate:.1f} traces/s)"
                )

        return traces


def _worker_generate(
    teacher_model: str,
    gpu_id: int,
    problems: list[dict],
    output_path: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    """Worker function for parallel generation on a single GPU."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    config = TraceConfig(
        teacher_model=teacher_model,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    generator = TraceGenerator(config, gpu_id=gpu_id)
    traces = generator.generate_all(problems)

    with open(output_path, "w") as f:
        for trace in traces:
            f.write(json.dumps(trace) + "\n")

    return output_path


def generate_parallel(
    config: TraceConfig,
    problems: list[dict],
) -> list[dict]:
    """Split problems across GPUs and generate in parallel.

    Each GPU loads a separate copy of the teacher model and processes
    its shard independently. Results are merged at the end.
    """
    gpus = config.gpus or [0]
    n_gpus = len(gpus)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # split problems across GPUs
    shards = [[] for _ in range(n_gpus)]
    for i, problem in enumerate(problems):
        shards[i % n_gpus].append(problem)

    shard_paths = [str(output_dir / f"shard_{gpu_id}.jsonl") for gpu_id in gpus]

    logger.info(f"Generating {len(problems)} traces across {n_gpus} GPUs: {gpus}")
    for i, gpu_id in enumerate(gpus):
        logger.info(f"  GPU {gpu_id}: {len(shards[i])} problems")

    # launch one process per GPU
    all_traces = []
    with ProcessPoolExecutor(max_workers=n_gpus) as executor:
        futures = {}
        for i, gpu_id in enumerate(gpus):
            if not shards[i]:
                continue
            future = executor.submit(
                _worker_generate,
                config.teacher_model,
                gpu_id,
                shards[i],
                shard_paths[i],
                config.max_new_tokens,
                config.temperature,
                config.top_p,
            )
            futures[future] = gpu_id

        for future in as_completed(futures):
            gpu_id = futures[future]
            try:
                shard_path = future.result()
                with open(shard_path) as f:
                    for line in f:
                        all_traces.append(json.loads(line))
                logger.info(f"GPU {gpu_id} finished: {shard_path}")
            except Exception as e:
                logger.error(f"GPU {gpu_id} failed: {e}")

    # write merged output
    merged_path = output_dir / "traces.jsonl"
    with open(merged_path, "w") as f:
        for trace in all_traces:
            f.write(json.dumps(trace) + "\n")

    n_verified = sum(1 for t in all_traces if t.get("verified") is True)
    n_failed = sum(1 for t in all_traces if t.get("verified") is False)
    n_unknown = sum(1 for t in all_traces if t.get("verified") is None)
    logger.info(
        f"Done: {len(all_traces)} traces "
        f"(verified={n_verified}, failed={n_failed}, unknown={n_unknown})"
    )

    return all_traces


def load_problems(num_traces: int) -> list[dict]:
    """Load math problems from GSM8K and MATH for trace generation."""
    problems = []
    try:
        from datasets import load_dataset

        ds = load_dataset("gsm8k", "main", split="train")
        for ex in ds:
            if len(problems) >= num_traces // 2:
                break
            m = re.search(r"####\s*(-?\d+\.?\d*)", ex.get("answer", ""))
            if m:
                problems.append({
                    "question": ex["question"],
                    "answer": m.group(1),
                    "source": "gsm8k",
                    "type": "math",
                })
    except Exception as e:
        logger.warning(f"Could not load GSM8K: {e}")

    try:
        from datasets import load_dataset

        ds = load_dataset("hendrycks/competition_math", split="train")
        for ex in ds:
            if len(problems) >= num_traces:
                break
            answer = ex.get("answer", "")
            if answer:
                problems.append({
                    "question": ex["problem"],
                    "answer": answer,
                    "source": "math",
                    "type": "math",
                })
    except Exception as e:
        logger.warning(f"Could not load MATH dataset: {e}")

    if not problems:
        logger.warning("No datasets available, generating synthetic problems")
        import random
        random.seed(42)
        for _ in range(num_traces):
            a, b = random.randint(1, 100), random.randint(1, 100)
            op = random.choice(["+", "-", "*"])
            answer = eval(f"{a} {op} {b}")
            problems.append({
                "question": f"What is {a} {op} {b}?",
                "answer": str(answer),
                "source": "synthetic",
                "type": "math",
            })

    return problems[:num_traces]


def main():
    parser = argparse.ArgumentParser(description="Generate reasoning traces from a teacher model")
    parser.add_argument("--teacher", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                        help="Teacher model name or path")
    parser.add_argument("--num-traces", type=int, default=10000,
                        help="Number of traces to generate")
    parser.add_argument("--output", default="./traces",
                        help="Output directory")
    parser.add_argument("--gpus", default="0",
                        help="Comma-separated GPU IDs (e.g. 0,1,2,3)")
    parser.add_argument("--max-new-tokens", type=int, default=2048,
                        help="Max tokens per generation")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    )

    gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]

    config = TraceConfig(
        teacher_model=args.teacher,
        num_traces=args.num_traces,
        output_dir=args.output,
        gpus=gpu_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        batch_size=args.batch_size,
    )

    problems = load_problems(config.num_traces)
    logger.info(f"Loaded {len(problems)} problems")

    if len(gpu_ids) > 1:
        traces = generate_parallel(config, problems)
    else:
        generator = TraceGenerator(config, gpu_id=gpu_ids[0])
        traces = generator.generate_all(problems)

        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "traces.jsonl"
        with open(output_path, "w") as f:
            for trace in traces:
                f.write(json.dumps(trace) + "\n")

    n_verified = sum(1 for t in traces if t.get("verified") is True)
    logger.info(f"Generated {len(traces)} traces ({n_verified} verified correct)")


if __name__ == "__main__":
    main()
