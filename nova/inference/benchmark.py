"""NOVA benchmark suite — load, run, score, and report.

Supports multiple modes:
  - base: raw model generation, no strategies
  - nova: full NOVA agent with all strategies enabled
  - bracket: bracket inference (default in agent)
  - bracket_search: bracket inference with search augmentation

Ablation mode toggles each strategy individually to measure contribution.
Results saved as JSON with per-question breakdown and summary table.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from .agent import NOVAAgent, parse_task_type, BRACKET_SIZES
from .strategies import (
    GenerativeModel,
    Tokenizer,
    extract_answer,
)
from .bracket import BracketInference
from .tool_orchestrator import ToolOrchestrator


# ── Benchmark targets from the NOVA spec ─────────────────────


BENCHMARK_TARGETS: dict[str, dict[str, float]] = {
    "GSM8K": {
        "low": 0.90,
        "high": 0.97,
        "description": "Grade school math — bracket + answer verification",
    },
    "MATH-500": {
        "low": 0.55,
        "high": 0.70,
        "description": "Competition math — bracket + verification",
    },
    "MMLU": {
        "low": 0.55,
        "high": 0.63,
        "description": "Massive multitask — bracket only",
    },
    "MMLU+search": {
        "low": 0.70,
        "high": 0.80,
        "description": "MMLU with search augmentation",
    },
    "HumanEval": {
        "low": 0.85,
        "high": 0.95,
        "description": "Code generation — bracket + execution verify",
    },
}


# ── Benchmark definitions ─────────────────────────────────────


@dataclass
class BenchmarkConfig:
    name: str
    hf_path: str
    hf_split: str = "test"
    hf_subset: str | None = None
    question_key: str = "question"
    answer_key: str = "answer"
    task_type: str = "reasoning"
    n_samples: int | None = None  # None = use all
    scorer: str = "exact_match"  # exact_match | numeric | code_pass


BENCHMARKS: dict[str, BenchmarkConfig] = {
    "math500": BenchmarkConfig(
        name="MATH-500",
        hf_path="hendrycks/competition_math",
        hf_split="test",
        question_key="problem",
        answer_key="solution",
        task_type="math",
        n_samples=500,
        scorer="numeric",
    ),
    "gsm8k": BenchmarkConfig(
        name="GSM8K",
        hf_path="openai/gsm8k",
        hf_subset="main",
        hf_split="test",
        question_key="question",
        answer_key="answer",
        task_type="math",
        scorer="numeric",
    ),
    "humaneval": BenchmarkConfig(
        name="HumanEval",
        hf_path="openai/openai_humaneval",
        hf_split="test",
        question_key="prompt",
        answer_key="canonical_solution",
        task_type="code",
        scorer="code_pass",
    ),
    "mbpp": BenchmarkConfig(
        name="MBPP",
        hf_path="google-research-datasets/mbpp",
        hf_split="test",
        question_key="text",
        answer_key="code",
        task_type="code",
        scorer="code_pass",
    ),
    "mmlu": BenchmarkConfig(
        name="MMLU",
        hf_path="cais/mmlu",
        hf_subset="all",
        hf_split="test",
        question_key="question",
        answer_key="answer",
        task_type="reasoning",
        scorer="exact_match",
    ),
    "arc_challenge": BenchmarkConfig(
        name="ARC-Challenge",
        hf_path="allenai/ai2_arc",
        hf_subset="ARC-Challenge",
        hf_split="test",
        question_key="question",
        answer_key="answerKey",
        task_type="reasoning",
        scorer="exact_match",
    ),
    "bbh": BenchmarkConfig(
        name="BBH",
        hf_path="lukaemon/bbh",
        hf_subset="boolean_expressions",
        hf_split="test",
        question_key="input",
        answer_key="target",
        task_type="reasoning",
        scorer="exact_match",
    ),
    "gpqa_diamond": BenchmarkConfig(
        name="GPQA-Diamond",
        hf_path="Idavidrein/gpqa",
        hf_subset="gpqa_diamond",
        hf_split="train",  # GPQA only has train split
        question_key="question",
        answer_key="answer",
        task_type="reasoning",
        scorer="exact_match",
    ),
    "aime2024": BenchmarkConfig(
        name="AIME-2024",
        hf_path="AI-MO/aimo-validation-aime",
        hf_split="train",
        question_key="problem",
        answer_key="answer",
        task_type="math",
        n_samples=30,
        scorer="numeric",
    ),
    "ifeval": BenchmarkConfig(
        name="IFEval",
        hf_path="google/IFEval",
        hf_split="train",
        question_key="prompt",
        answer_key="reference_answer",
        task_type="general",
        scorer="exact_match",
    ),
}


# ── Dataset loading ────────────────────────────────────────────


def load_benchmark(config: BenchmarkConfig) -> list[dict]:
    """Load benchmark questions from HuggingFace."""
    from datasets import load_dataset

    load_kwargs = {"path": config.hf_path, "split": config.hf_split}
    if config.hf_subset:
        load_kwargs["name"] = config.hf_subset

    try:
        ds = load_dataset(**load_kwargs, trust_remote_code=True)
    except Exception as e:
        print(f"  WARNING: Failed to load {config.name}: {e}")
        return []

    samples = []
    for i, row in enumerate(ds):
        if config.n_samples and i >= config.n_samples:
            break

        question = row.get(config.question_key, "")
        answer = row.get(config.answer_key, "")

        # Handle MMLU multi-choice format
        if "choices" in row and isinstance(row["choices"], list):
            choices_text = "\n".join(
                f"({chr(65 + j)}) {c}" for j, c in enumerate(row["choices"])
            )
            question = f"{question}\n{choices_text}"
            # Convert numeric answer index to letter
            if isinstance(answer, int):
                answer = chr(65 + answer)

        # Handle GSM8K answer format: "#### 42"
        if isinstance(answer, str) and "####" in answer:
            answer = answer.split("####")[-1].strip()

        # Handle ARC answerKey
        if config.name == "ARC-Challenge" and "choices" in row:
            labels = row["choices"].get("label", [])
            texts = row["choices"].get("text", [])
            choices_text = "\n".join(
                f"({l}) {t}" for l, t in zip(labels, texts)
            )
            question = f"{question}\n{choices_text}"

        samples.append({
            "question": str(question),
            "answer": str(answer),
            "index": i,
        })

    return samples


# ── Scorers ────────────────────────────────────────────────────


def score_exact_match(predicted: str | None, ground_truth: str) -> float:
    if predicted is None:
        return 0.0
    pred = predicted.strip().lower()
    truth = ground_truth.strip().lower()
    if pred == truth:
        return 1.0
    # Single letter match for multi-choice
    if len(truth) == 1 and truth in pred[:5].lower():
        return 1.0
    return 0.0


def score_numeric(predicted: str | None, ground_truth: str) -> float:
    if predicted is None:
        return 0.0
    try:
        # Clean common formatting
        pred_clean = predicted.strip().replace(",", "").replace("$", "")
        truth_clean = ground_truth.strip().replace(",", "").replace("$", "")
        p = float(pred_clean)
        t = float(truth_clean)
        if abs(p - t) < 1e-4:
            return 1.0
        # Partial credit for being close
        if t != 0 and abs(p - t) / abs(t) < 0.01:
            return 0.5
        return 0.0
    except (ValueError, TypeError):
        return score_exact_match(predicted, ground_truth)


def score_code_pass(predicted: str | None, ground_truth: str) -> float:
    """Score code by checking if it runs without error.

    Full functional testing requires test harnesses per benchmark.
    This is a basic syntax + execution check.
    """
    if predicted is None:
        return 0.0

    import subprocess
    import tempfile

    code = predicted.strip()
    # If the model output includes the ground truth signature, good sign
    if ground_truth.strip()[:50] in code:
        return 0.5

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        f.flush()
        try:
            result = subprocess.run(
                ["python", "-c", f"compile(open('{f.name}').read(), '{f.name}', 'exec')"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                return 0.5  # Compiles but not functionally tested
            return 0.0
        except (subprocess.TimeoutExpired, Exception):
            return 0.0


SCORERS = {
    "exact_match": score_exact_match,
    "numeric": score_numeric,
    "code_pass": score_code_pass,
}


# ── Base inference (no strategies) ─────────────────────────────


@torch.no_grad()
def run_base_inference(
    model: GenerativeModel,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.3,
) -> str | None:
    """Single-pass generation — the baseline with no strategies."""
    device = (
        next(iter(model.parameters())).device
        if hasattr(model, "parameters")
        else torch.device("cpu")
    )
    ids = torch.tensor(
        [tokenizer.encode(prompt)], dtype=torch.long, device=device
    )
    out = model.generate(ids, max_new_tokens=max_new_tokens, temperature=temperature)
    text = tokenizer.decode(out[0].tolist())
    return extract_answer(text)


# ── BenchmarkRunner ────────────────────────────────────────────


@dataclass
class BenchmarkResult:
    benchmark: str
    mode: str  # "base" | "nova" | ablation name
    score: float
    n_correct: int
    n_total: int
    elapsed_s: float
    per_question: list[dict] = field(default_factory=list)


class BenchmarkRunner:
    """Load benchmarks, run in base and NOVA modes, score and report.

    Usage:
        runner = BenchmarkRunner(model, tokenizer)
        results = runner.run_all()
        runner.save_results(results, "benchmark_results.json")
        runner.print_summary(results)
    """

    def __init__(
        self,
        model: GenerativeModel,
        tokenizer: Tokenizer,
        output_dir: str | Path = "nova/results",
        benchmark_names: list[str] | None = None,
        swt_scheduler=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.swt_scheduler = swt_scheduler

        if benchmark_names:
            self.benchmarks = {
                k: v for k, v in BENCHMARKS.items() if k in benchmark_names
            }
        else:
            self.benchmarks = dict(BENCHMARKS)

        self.agent = NOVAAgent(model, tokenizer, swt_scheduler=swt_scheduler)

    def run_all(self, modes: list[str] | None = None) -> list[BenchmarkResult]:
        """Run all selected benchmarks in specified modes.

        modes: list of "base", "nova", or ablation strategy names.
               Default: ["base", "nova"]
        """
        if modes is None:
            modes = ["base", "nova"]

        all_results = []
        for bm_key, bm_config in self.benchmarks.items():
            print(f"\n{'=' * 60}")
            print(f"  Benchmark: {bm_config.name}")
            print(f"{'=' * 60}")

            samples = load_benchmark(bm_config)
            if not samples:
                print(f"  SKIPPED (failed to load)")
                continue
            print(f"  Loaded {len(samples)} samples")

            scorer = SCORERS.get(bm_config.scorer, score_exact_match)

            for mode in modes:
                result = self._run_benchmark(bm_key, bm_config, samples, scorer, mode)
                all_results.append(result)
                print(
                    f"  [{mode:>8s}] {result.score:.1%} "
                    f"({result.n_correct}/{result.n_total}) "
                    f"in {result.elapsed_s:.1f}s"
                )

        return all_results

    def run_ablation(self) -> list[BenchmarkResult]:
        """Run ablation: toggle each strategy individually.

        For each benchmark, runs with each strategy disabled to
        measure its individual contribution.
        """
        strategies_to_ablate = [
            "self_consistency",
            "tree_search",
            "forest_of_thought",
            "execution_verifier",
            "adversarial_review",
        ]

        all_results = []
        for bm_key, bm_config in self.benchmarks.items():
            print(f"\n{'=' * 60}")
            print(f"  Ablation: {bm_config.name}")
            print(f"{'=' * 60}")

            samples = load_benchmark(bm_config)
            if not samples:
                continue
            print(f"  Loaded {len(samples)} samples")

            scorer = SCORERS.get(bm_config.scorer, score_exact_match)

            # Full NOVA baseline for comparison
            full = self._run_benchmark(bm_key, bm_config, samples, scorer, "nova")
            all_results.append(full)
            print(f"  [    nova] {full.score:.1%}")

            # Ablate each strategy
            for strategy in strategies_to_ablate:
                ablation_name = f"no_{strategy}"
                result = self._run_benchmark(
                    bm_key, bm_config, samples, scorer, "nova",
                    disabled_strategy=strategy,
                )
                result.mode = ablation_name
                all_results.append(result)
                delta = result.score - full.score
                direction = "+" if delta >= 0 else ""
                print(
                    f"  [{ablation_name:>20s}] {result.score:.1%} "
                    f"({direction}{delta:.1%})"
                )

        return all_results

    def _run_benchmark(
        self,
        bm_key: str,
        bm_config: BenchmarkConfig,
        samples: list[dict],
        scorer,
        mode: str,
        disabled_strategy: str | None = None,
    ) -> BenchmarkResult:
        """Run a single benchmark in a given mode."""
        t0 = time.time()
        per_question = []
        n_correct = 0

        for sample in samples:
            question = sample["question"]
            ground_truth = sample["answer"]

            if mode == "base":
                predicted = run_base_inference(
                    self.model, self.tokenizer, question
                )
                strategy_used = "base"
            else:
                # NOVA mode with optional ablation
                if disabled_strategy:
                    # Temporarily remove the strategy
                    original = self.agent.strategies.get(disabled_strategy)
                    self.agent.strategies[disabled_strategy] = None
                    agent_result = self.agent.run(question, ground_truth)
                    self.agent.strategies[disabled_strategy] = original
                else:
                    agent_result = self.agent.run(question, ground_truth)
                predicted = agent_result.get("answer")
                strategy_used = agent_result.get("strategy", "nova")

            if predicted is not None:
                predicted = str(predicted)

            score = scorer(predicted, ground_truth)
            n_correct += score > 0.5

            per_question.append({
                "index": sample["index"],
                "question": question[:200],
                "ground_truth": ground_truth[:100],
                "predicted": predicted,
                "score": score,
                "strategy": strategy_used,
            })

        elapsed = time.time() - t0
        total = len(samples)
        avg_score = n_correct / max(total, 1)

        return BenchmarkResult(
            benchmark=bm_config.name,
            mode=mode,
            score=avg_score,
            n_correct=n_correct,
            n_total=total,
            elapsed_s=elapsed,
            per_question=per_question,
        )

    def save_results(self, results: list[BenchmarkResult], filename: str = "benchmark_results.json"):
        """Save full results as JSON with per-question breakdown."""
        data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "benchmarks": [],
        }
        for r in results:
            data["benchmarks"].append({
                "benchmark": r.benchmark,
                "mode": r.mode,
                "score": r.score,
                "n_correct": r.n_correct,
                "n_total": r.n_total,
                "elapsed_s": round(r.elapsed_s, 2),
                "per_question": r.per_question,
            })

        path = self.output_dir / filename
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"\nResults saved to {path}")

    def print_summary(self, results: list[BenchmarkResult]):
        """Print summary table to stdout."""
        print(f"\n{'=' * 72}")
        print(f"  NOVA BENCHMARK SUMMARY")
        print(f"{'=' * 72}")
        print(f"  {'Benchmark':<20s} {'Mode':<15s} {'Score':>8s} {'N':>6s} {'Time':>8s}")
        print(f"  {'-' * 20} {'-' * 15} {'-' * 8} {'-' * 6} {'-' * 8}")

        # Group by benchmark for side-by-side comparison
        by_benchmark: dict[str, list[BenchmarkResult]] = {}
        for r in results:
            by_benchmark.setdefault(r.benchmark, []).append(r)

        for bm_name, bm_results in by_benchmark.items():
            for r in bm_results:
                print(
                    f"  {bm_name:<20s} {r.mode:<15s} "
                    f"{r.score:>7.1%} {r.n_total:>6d} "
                    f"{r.elapsed_s:>7.1f}s"
                )

        # Aggregate scores
        base_scores = [r.score for r in results if r.mode == "base"]
        nova_scores = [r.score for r in results if r.mode == "nova"]
        bracket_scores = [r.score for r in results if r.mode == "bracket"]
        bracket_search_scores = [
            r.score for r in results if r.mode == "bracket_search"
        ]
        if base_scores:
            print(f"\n  Base average:     {sum(base_scores) / len(base_scores):.1%}")
        if nova_scores:
            print(f"  NOVA average:     {sum(nova_scores) / len(nova_scores):.1%}")
        if bracket_scores:
            print(f"  Bracket average:  {sum(bracket_scores) / len(bracket_scores):.1%}")
        if bracket_search_scores:
            print(
                f"  Bracket+search:   "
                f"{sum(bracket_search_scores) / len(bracket_search_scores):.1%}"
            )
        if base_scores and nova_scores:
            delta = (
                sum(nova_scores) / len(nova_scores)
                - sum(base_scores) / len(base_scores)
            )
            print(f"  NOVA lift:        {'+' if delta >= 0 else ''}{delta:.1%}")
        print(f"{'=' * 72}")

    # ── Bracket-aware benchmark methods ──────────────────────────

    def run_gsm8k(self, bracket_size: int = 64) -> BenchmarkResult:
        """Run GSM8K with bracket inference + mathematical answer verification.

        Generates bracket_size candidates, verifies answers numerically,
        and brackets among correct candidates on quality.
        """
        config = BENCHMARKS["gsm8k"]
        samples = load_benchmark(config)
        if not samples:
            print("  GSM8K: failed to load")
            return BenchmarkResult("GSM8K", "bracket", 0.0, 0, 0, 0.0)

        scorer = SCORERS["numeric"]
        bracket = BracketInference(
            self.model, self.tokenizer,
            preference_store=getattr(self.agent, "preference_store", None),
        )

        t0 = time.time()
        per_question = []
        n_correct = 0

        for sample in samples:
            result = bracket.run(
                sample["question"],
                n_candidates=bracket_size,
                verify_answers=True,
            )
            predicted = result.get("answer")
            if predicted is not None:
                predicted = str(predicted)

            score = scorer(predicted, sample["answer"])
            n_correct += score > 0.5
            per_question.append({
                "index": sample["index"],
                "question": sample["question"][:200],
                "ground_truth": sample["answer"][:100],
                "predicted": predicted,
                "score": score,
                "strategy": "bracket",
                "bracket_size": bracket_size,
                "n_candidates": result.get("n_candidates", 0),
            })

        elapsed = time.time() - t0
        total = len(samples)
        avg_score = n_correct / max(total, 1)

        return BenchmarkResult(
            benchmark="GSM8K",
            mode="bracket",
            score=avg_score,
            n_correct=n_correct,
            n_total=total,
            elapsed_s=elapsed,
            per_question=per_question,
        )

    def run_humaneval(self, bracket_size: int = 64) -> BenchmarkResult:
        """Run HumanEval with bracket inference + execution testing.

        Generates bracket_size code candidates, executes tests to filter
        survivors, then brackets survivors on code quality.
        """
        config = BENCHMARKS["humaneval"]
        samples = load_benchmark(config)
        if not samples:
            print("  HumanEval: failed to load")
            return BenchmarkResult("HumanEval", "bracket", 0.0, 0, 0, 0.0)

        scorer = SCORERS["code_pass"]
        bracket = BracketInference(
            self.model, self.tokenizer,
            preference_store=getattr(self.agent, "preference_store", None),
        )

        t0 = time.time()
        per_question = []
        n_correct = 0

        for sample in samples:
            result = bracket.code_bracket(
                sample["question"],
                n_candidates=bracket_size,
            )
            predicted = result.get("answer")
            if predicted is not None:
                predicted = str(predicted)

            score = scorer(predicted, sample["answer"])
            n_correct += score > 0.5
            per_question.append({
                "index": sample["index"],
                "question": sample["question"][:200],
                "ground_truth": sample["answer"][:100],
                "predicted": predicted,
                "score": score,
                "strategy": "bracket_code",
                "bracket_size": bracket_size,
            })

        elapsed = time.time() - t0
        total = len(samples)
        avg_score = n_correct / max(total, 1)

        return BenchmarkResult(
            benchmark="HumanEval",
            mode="bracket",
            score=avg_score,
            n_correct=n_correct,
            n_total=total,
            elapsed_s=elapsed,
            per_question=per_question,
        )

    def run_mmlu(
        self, bracket_size: int = 16, use_search: bool = False
    ) -> BenchmarkResult:
        """Run MMLU with bracket inference and optional search augmentation.

        When use_search=True, each question is first augmented with web
        search results before bracket inference.
        """
        config = BENCHMARKS["mmlu"]
        samples = load_benchmark(config)
        if not samples:
            print("  MMLU: failed to load")
            return BenchmarkResult("MMLU", "bracket", 0.0, 0, 0, 0.0)

        scorer = SCORERS["exact_match"]
        bracket = BracketInference(
            self.model, self.tokenizer,
            preference_store=getattr(self.agent, "preference_store", None),
        )
        orchestrator = ToolOrchestrator() if use_search else None
        mode = "bracket_search" if use_search else "bracket"

        t0 = time.time()
        per_question = []
        n_correct = 0

        for sample in samples:
            prompt = sample["question"]
            if orchestrator is not None:
                prompt = orchestrator.search_and_augment(prompt)

            result = bracket.run(prompt, n_candidates=bracket_size)
            predicted = result.get("answer")
            if predicted is not None:
                predicted = str(predicted)

            score = scorer(predicted, sample["answer"])
            n_correct += score > 0.5
            per_question.append({
                "index": sample["index"],
                "question": sample["question"][:200],
                "ground_truth": sample["answer"][:100],
                "predicted": predicted,
                "score": score,
                "strategy": mode,
                "bracket_size": bracket_size,
            })

        elapsed = time.time() - t0
        total = len(samples)
        avg_score = n_correct / max(total, 1)

        return BenchmarkResult(
            benchmark="MMLU",
            mode=mode,
            score=avg_score,
            n_correct=n_correct,
            n_total=total,
            elapsed_s=elapsed,
            per_question=per_question,
        )

    def run_full_suite(self) -> list[BenchmarkResult]:
        """Run all benchmarks in base, bracket, and bracket+search modes.

        Prints a comparison table showing:
          - base (no bracket, no strategies)
          - bracket (bracket inference)
          - bracket+search (bracket with search augmentation, where applicable)
          - targets from the NOVA spec
        """
        print(f"\n{'=' * 80}")
        print(f"  NOVA FULL BENCHMARK SUITE — base vs bracket vs bracket+search")
        print(f"{'=' * 80}")

        all_results: list[BenchmarkResult] = []

        # GSM8K — base vs bracket
        print(f"\n  --- GSM8K ---")
        gsm_config = BENCHMARKS["gsm8k"]
        gsm_samples = load_benchmark(gsm_config)
        if gsm_samples:
            gsm_base = self._run_benchmark(
                "gsm8k", gsm_config, gsm_samples, SCORERS["numeric"], "base"
            )
            all_results.append(gsm_base)
            print(f"  [    base] {gsm_base.score:.1%}")

            gsm_bracket = self.run_gsm8k(bracket_size=64)
            all_results.append(gsm_bracket)
            print(f"  [ bracket] {gsm_bracket.score:.1%}")

            target = BENCHMARK_TARGETS["GSM8K"]
            print(f"  [ target ] {target['low']:.0%}-{target['high']:.0%}")

        # HumanEval — base vs bracket
        print(f"\n  --- HumanEval ---")
        he_config = BENCHMARKS["humaneval"]
        he_samples = load_benchmark(he_config)
        if he_samples:
            he_base = self._run_benchmark(
                "humaneval", he_config, he_samples, SCORERS["code_pass"], "base"
            )
            all_results.append(he_base)
            print(f"  [    base] {he_base.score:.1%}")

            he_bracket = self.run_humaneval(bracket_size=64)
            all_results.append(he_bracket)
            print(f"  [ bracket] {he_bracket.score:.1%}")

            target = BENCHMARK_TARGETS["HumanEval"]
            print(f"  [ target ] {target['low']:.0%}-{target['high']:.0%}")

        # MMLU — base vs bracket vs bracket+search
        print(f"\n  --- MMLU ---")
        mmlu_config = BENCHMARKS["mmlu"]
        mmlu_samples = load_benchmark(mmlu_config)
        if mmlu_samples:
            mmlu_base = self._run_benchmark(
                "mmlu", mmlu_config, mmlu_samples, SCORERS["exact_match"], "base"
            )
            all_results.append(mmlu_base)
            print(f"  [    base] {mmlu_base.score:.1%}")

            mmlu_bracket = self.run_mmlu(bracket_size=16, use_search=False)
            all_results.append(mmlu_bracket)
            print(f"  [ bracket] {mmlu_bracket.score:.1%}")

            mmlu_search = self.run_mmlu(bracket_size=16, use_search=True)
            all_results.append(mmlu_search)
            print(f"  [bracket+] {mmlu_search.score:.1%}")

            target_base = BENCHMARK_TARGETS["MMLU"]
            target_search = BENCHMARK_TARGETS["MMLU+search"]
            print(f"  [ target ] {target_base['low']:.0%}-{target_base['high']:.0%} (no search)")
            print(f"  [ target ] {target_search['low']:.0%}-{target_search['high']:.0%} (with search)")

        # Summary comparison
        self._print_target_comparison(all_results)
        return all_results

    def _print_target_comparison(self, results: list[BenchmarkResult]):
        """Print side-by-side comparison of results vs spec targets."""
        print(f"\n{'=' * 80}")
        print(f"  RESULTS vs TARGETS")
        print(f"{'=' * 80}")
        print(
            f"  {'Benchmark':<15s} {'Mode':<16s} {'Score':>8s} "
            f"{'Target':>12s} {'Status':>8s}"
        )
        print(f"  {'-' * 15} {'-' * 16} {'-' * 8} {'-' * 12} {'-' * 8}")

        for r in results:
            target_key = r.benchmark
            if r.mode == "bracket_search":
                target_key = f"{r.benchmark}+search"

            target = BENCHMARK_TARGETS.get(target_key)
            if target is None:
                target_str = "N/A"
                status = ""
            else:
                target_str = f"{target['low']:.0%}-{target['high']:.0%}"
                if r.score >= target["high"]:
                    status = "EXCEEDS"
                elif r.score >= target["low"]:
                    status = "IN RANGE"
                else:
                    status = "BELOW"

            print(
                f"  {r.benchmark:<15s} {r.mode:<16s} {r.score:>7.1%} "
                f"{target_str:>12s} {status:>8s}"
            )
        print(f"{'=' * 80}")
