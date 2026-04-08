"""NOVA agent loop — task parsing, planning, execution, and learning.

Integrates all inference strategies via DifficultyRouter and manages
the full reason-act-verify cycle. Bracket inference is the DEFAULT
method for all tasks. Handles SWT wake updates when the model
encounters novel task patterns.
"""

from __future__ import annotations

import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch

from .strategies import (
    AdversarialReview,
    DifficultyRouter,
    ExecutionVerifier,
    ForestOfThought,
    GenerativeModel,
    SelfConsistencyVoter,
    StrategyPipeline,
    Tokenizer,
    TreeSearch,
    extract_answer,
)
from .bracket import BracketInference
from .tool_orchestrator import ToolOrchestrator


# ── Bracket sizing by difficulty ──────────────────────────────

BRACKET_SIZES: dict[str, int] = {
    "trivial": 1,
    "easy": 4,
    "simple": 4,      # alias — router uses "simple"
    "medium": 16,
    "hard": 64,
}


# ── Task types ─────────────────────────────────────────────────


class TaskType(Enum):
    MATH = "math"
    CODE = "code"
    REASONING = "reasoning"
    KNOWLEDGE = "knowledge"
    GENERAL = "general"


# ── Memory systems ─────────────────────────────────────────────


@dataclass
class ShortTermMemory:
    """Conversation-scoped context window."""

    entries: deque = field(default_factory=lambda: deque(maxlen=32))

    def add(self, role: str, content: str):
        self.entries.append({"role": role, "content": content, "ts": time.time()})

    def get_context(self, max_tokens: int = 2048) -> str:
        parts = []
        total = 0
        for entry in reversed(self.entries):
            text = f"{entry['role']}: {entry['content']}"
            est_tokens = len(text.split())
            if total + est_tokens > max_tokens:
                break
            parts.append(text)
            total += est_tokens
        return "\n".join(reversed(parts))

    def clear(self):
        self.entries.clear()


@dataclass
class TaskPattern:
    """Observed pattern for a task type + difficulty combo."""

    task_type: str
    difficulty: str
    best_strategy: str
    success_rate: float = 0.0
    attempts: int = 0
    successes: int = 0
    avg_time_s: float = 0.0

    def update(self, success: bool, elapsed: float):
        self.attempts += 1
        if success:
            self.successes += 1
        self.success_rate = self.successes / self.attempts
        self.avg_time_s = (
            self.avg_time_s * (self.attempts - 1) + elapsed
        ) / self.attempts


class LongTermMemory:
    """Cross-session task pattern memory.

    Tracks which strategies work best for which task types and
    difficulty levels. Used by the planner to override router defaults.
    """

    def __init__(self):
        self.patterns: dict[str, TaskPattern] = {}
        self.novel_tasks: list[dict] = []

    def get_pattern(self, task_type: str, difficulty: str) -> TaskPattern | None:
        key = f"{task_type}:{difficulty}"
        return self.patterns.get(key)

    def update_pattern(
        self, task_type: str, difficulty: str, strategy: str,
        success: bool, elapsed: float,
    ):
        key = f"{task_type}:{difficulty}"
        if key not in self.patterns:
            self.patterns[key] = TaskPattern(
                task_type=task_type,
                difficulty=difficulty,
                best_strategy=strategy,
            )
        pattern = self.patterns[key]
        pattern.update(success, elapsed)
        if success and pattern.success_rate > 0.5:
            pattern.best_strategy = strategy

    def record_novel_task(self, prompt: str, task_type: str, difficulty: str):
        self.novel_tasks.append({
            "prompt": prompt[:200],
            "task_type": task_type,
            "difficulty": difficulty,
            "ts": time.time(),
        })

    def get_novel_task_count(self) -> int:
        return len(self.novel_tasks)

    def drain_novel_tasks(self) -> list[dict]:
        tasks = self.novel_tasks.copy()
        self.novel_tasks.clear()
        return tasks


# ── Task parser ────────────────────────────────────────────────


MATH_PATTERNS = re.compile(
    r"(?:\b(?:solve|equation|integral|derivative|prove|compute|calculate|sum|product|"
    r"factor|simplify|evaluate|find\s+the\s+value|how\s+many|probability|"
    r"triangle|circle|area|volume|matrix)\b"
    r"|what\s+is\s+\d"
    r"|\\frac|\\sqrt|\\int|\\sum"
    r"|\d+\s*[+\-*/]\s*\d+)",
    re.IGNORECASE,
)

CODE_PATTERNS = re.compile(
    r"\b(function|def\s|class\s|implement|algorithm|code|program|"
    r"write\s+a?\s*(function|script|program|method)|"
    r"return\s+type|input.*output|O\(n|time\s+complexity|"
    r"list|array|string|dict|hash|tree|graph|sort|search)\b",
    re.IGNORECASE,
)

REASONING_PATTERNS = re.compile(
    r"\b(explain|why|how\s+does|what\s+would|if\s+.+then|"
    r"reason|logic|deduce|infer|conclude|analyze|"
    r"compare|contrast|argument|premise|fallacy|"
    r"true\s+or\s+false|which\s+of\s+the\s+following)\b",
    re.IGNORECASE,
)

KNOWLEDGE_PATTERNS = re.compile(
    r"\b(who\s+is|who\s+was|what\s+year|when\s+did|when\s+was|"
    r"capital\s+of|president\s+of|population\s+of|"
    r"current|latest|founded\s+in|born\s+in|died\s+in|"
    r"where\s+is|where\s+was|stock\s+price|exchange\s+rate)\b",
    re.IGNORECASE,
)


def parse_task_type(prompt: str) -> TaskType:
    """Classify the task type from the prompt text."""
    math_hits = len(MATH_PATTERNS.findall(prompt))
    code_hits = len(CODE_PATTERNS.findall(prompt))
    reasoning_hits = len(REASONING_PATTERNS.findall(prompt))
    knowledge_hits = len(KNOWLEDGE_PATTERNS.findall(prompt))

    scores = {
        TaskType.MATH: math_hits * 2,
        TaskType.CODE: code_hits * 2,
        TaskType.REASONING: reasoning_hits,
        TaskType.KNOWLEDGE: knowledge_hits * 2,
        TaskType.GENERAL: 1,
    }

    return max(scores, key=scores.get)


# ── Planner ────────────────────────────────────────────────────


def select_strategy(
    task_type: TaskType,
    difficulty: str,
    memory: LongTermMemory,
) -> str:
    """Choose strategy based on task type, difficulty, and past patterns."""
    # Check if memory has a proven strategy for this combo
    pattern = memory.get_pattern(task_type.value, difficulty)
    if pattern is not None and pattern.attempts >= 3 and pattern.success_rate > 0.6:
        return pattern.best_strategy

    # Default strategy map
    strategy_map = {
        TaskType.MATH: {
            "simple": "self_consistency",
            "medium": "self_consistency",
            "hard": "forest_of_thought",
        },
        TaskType.CODE: {
            "simple": "direct",
            "medium": "execution_verifier",
            "hard": "execution_verifier",
        },
        TaskType.REASONING: {
            "simple": "direct",
            "medium": "self_consistency",
            "hard": "adversarial_review",
        },
        TaskType.KNOWLEDGE: {
            "simple": "direct",
            "medium": "self_consistency",
            "hard": "self_consistency",
        },
        TaskType.GENERAL: {
            "simple": "direct",
            "medium": "self_consistency",
            "hard": "tree_search",
        },
    }

    return strategy_map.get(task_type, strategy_map[TaskType.GENERAL]).get(
        difficulty, "self_consistency"
    )


# ── Environment ────────────────────────────────────────────────


class Environment:
    """Manages action execution and feedback collection."""

    def __init__(
        self,
        model: GenerativeModel,
        tokenizer: Tokenizer,
        strategies: dict[str, Any],
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.strategies = strategies

    def execute(self, prompt: str, strategy_name: str) -> dict:
        """Execute the chosen strategy and return results."""
        strategy = self.strategies.get(strategy_name)
        if strategy is None:
            # Fallback to direct generation
            return self._direct_generate(prompt)

        if strategy_name == "execution_verifier":
            return strategy(prompt)
        if strategy_name == "direct":
            return self._direct_generate(prompt)

        # All other strategies follow the same call signature
        return strategy(prompt)

    def _direct_generate(self, prompt: str) -> dict:
        """Single-pass generation with no strategy."""
        device = (
            next(iter(self.model.parameters())).device
            if hasattr(self.model, "parameters")
            else torch.device("cpu")
        )
        ids = torch.tensor(
            [self.tokenizer.encode(prompt)], dtype=torch.long, device=device
        )
        out = self.model.generate(ids, max_new_tokens=512, temperature=0.3)
        text = self.tokenizer.decode(out[0].tolist())
        return {
            "answer": extract_answer(text),
            "full_text": text,
            "strategy": "direct",
        }


# ── NOVAAgent ──────────────────────────────────────────────────


class NOVAAgent:
    """Full agent loop: parse -> plan -> execute -> verify -> learn.

    Integrates all inference strategies, manages memory, and triggers
    SWT wake updates when novel tasks accumulate.
    """

    # How many novel tasks before triggering an SWT wake update
    SWT_NOVEL_THRESHOLD = 10

    def __init__(
        self,
        model: GenerativeModel,
        tokenizer: Tokenizer,
        swt_scheduler=None,
        swt_novel_threshold: int | None = None,
        preference_store: Any | None = None,
        search_enabled: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.swt_scheduler = swt_scheduler
        self.swt_novel_threshold = swt_novel_threshold or self.SWT_NOVEL_THRESHOLD
        self.preference_store = preference_store
        self.search_enabled = search_enabled

        self.short_memory = ShortTermMemory()
        self.long_memory = LongTermMemory()

        # Build strategy instances
        self.voter = SelfConsistencyVoter(model, tokenizer, n_samples=16)
        self.tree = TreeSearch(model, tokenizer, beam_width=4, max_depth=8)
        self.forest = ForestOfThought(model, tokenizer, n_trees=4)
        self.exec_verifier = ExecutionVerifier(model, tokenizer)
        self.adversarial = AdversarialReview(model, tokenizer)
        self.router = DifficultyRouter(
            model, tokenizer,
            medium_strategy=self.voter,
            hard_strategy=self.forest,
        )
        self.pipeline = StrategyPipeline(
            model, tokenizer,
            execution_verifier=self.exec_verifier,
            adversarial_review=self.adversarial,
            difficulty_router=self.router,
        )

        # Bracket inference — default method for all tasks
        self.bracket = BracketInference(
            model, tokenizer, preference_store=preference_store
        )

        # Tool orchestrator — web search integration
        self.tool_orchestrator = ToolOrchestrator()

        self.strategies = {
            "direct": None,  # handled by Environment._direct_generate
            "self_consistency": self.voter,
            "tree_search": self.tree,
            "forest_of_thought": self.forest,
            "execution_verifier": self.exec_verifier,
            "adversarial_review": self.adversarial,
            "router": self.router,
            "pipeline": self.pipeline,
        }

        self.env = Environment(model, tokenizer, self.strategies)

    def run(
        self,
        prompt: str,
        ground_truth: str | None = None,
        use_bracket: bool = True,
    ) -> dict:
        """Full agent cycle for a single prompt.

        By default routes through bracket inference. Set use_bracket=False
        to fall back to the legacy strategy-only path.

        Returns dict with answer, task_type, difficulty, strategy, timing, success.
        """
        t0 = time.time()

        # 1. Parse
        task_type = parse_task_type(prompt)
        self.short_memory.add("user", prompt)

        # 2. Estimate difficulty via router
        difficulty_info = self.router._estimate_difficulty(prompt)
        difficulty = difficulty_info[0]
        entropy = difficulty_info[1]

        # 3. Plan — select strategy (used as fallback or inside bracket)
        strategy_name = select_strategy(task_type, difficulty, self.long_memory)

        # 4. Execute — bracket is the default path
        if use_bracket:
            result = self._bracket_dispatch(prompt, task_type, difficulty)
            strategy_name = result.get("strategy", f"bracket_{task_type.value}")
        else:
            result = self.env.execute(prompt, strategy_name)

        answer = result.get("answer")
        elapsed = time.time() - t0

        # 5. Verify
        success = False
        if ground_truth is not None and answer is not None:
            success = self._verify(answer, ground_truth)

        # 6. Learn — update long-term memory
        self.long_memory.update_pattern(
            task_type.value, difficulty, strategy_name, success, elapsed
        )

        # Track novelty for SWT
        pattern = self.long_memory.get_pattern(task_type.value, difficulty)
        is_novel = pattern is None or pattern.attempts <= 1
        if is_novel:
            self.long_memory.record_novel_task(prompt, task_type.value, difficulty)

        # 7. SWT wake update if enough novel tasks accumulated
        self._maybe_swt_wake()

        self.short_memory.add("assistant", str(answer))

        return {
            "answer": answer,
            "task_type": task_type.value,
            "difficulty": difficulty,
            "entropy": entropy,
            "strategy": strategy_name,
            "elapsed_s": elapsed,
            "success": success,
            "is_novel": is_novel,
            "result_detail": result,
        }

    def _bracket_dispatch(
        self, prompt: str, task_type: TaskType, difficulty: str
    ) -> dict:
        """Route prompt through bracket inference based on task type.

        Routing logic:
          - CODE: bracket.code_bracket with execution verification
          - MATH: bracket with answer verification
          - KNOWLEDGE (search enabled): search first, then bracket augmented prompt
          - Everything else: pure bracket inference
        """
        n_candidates = BRACKET_SIZES.get(difficulty, 16)

        if task_type == TaskType.CODE:
            return self.bracket.code_bracket(prompt, n_candidates=n_candidates)

        if task_type == TaskType.MATH:
            return self.bracket.run(
                prompt, n_candidates=n_candidates, verify_answers=True
            )

        if task_type == TaskType.KNOWLEDGE and self.search_enabled:
            augmented = self.tool_orchestrator.search_and_augment(prompt)
            return self.bracket.run(augmented, n_candidates=n_candidates)

        # REASONING, GENERAL, or KNOWLEDGE without search
        return self.bracket.run(prompt, n_candidates=n_candidates)

    def run_batch(
        self,
        prompts: list[str],
        ground_truths: list[str | None] | None = None,
        use_bracket: bool = True,
    ) -> list[dict]:
        """Run the agent on a batch of prompts sequentially."""
        if ground_truths is None:
            ground_truths = [None] * len(prompts)
        return [
            self.run(p, gt, use_bracket=use_bracket)
            for p, gt in zip(prompts, ground_truths)
        ]

    def _verify(self, predicted: str, ground_truth: str) -> bool:
        """Check if predicted answer matches ground truth."""
        pred = predicted.strip().lower()
        truth = ground_truth.strip().lower()

        if pred == truth:
            return True

        # Numeric comparison
        try:
            p_val = float(pred.replace(",", ""))
            t_val = float(truth.replace(",", ""))
            return abs(p_val - t_val) < 1e-4
        except (ValueError, TypeError):
            pass

        # Substring containment for short answers
        if len(truth) < 20 and truth in pred:
            return True

        return False

    def _maybe_swt_wake(self):
        """Trigger SWT wake update if novel task threshold is reached."""
        if self.swt_scheduler is None:
            return
        if self.long_memory.get_novel_task_count() < self.swt_novel_threshold:
            return

        novel_tasks = self.long_memory.drain_novel_tasks()

        # SWT wake: the scheduler handles the actual weight updates.
        # We signal it by stepping through wake phases with the novel data.
        if hasattr(self.swt_scheduler, "step"):
            for _ in range(len(novel_tasks)):
                self.swt_scheduler.step()

    def reset_conversation(self):
        """Clear short-term memory between conversations."""
        self.short_memory.clear()

    def get_stats(self) -> dict:
        """Return agent performance statistics."""
        stats = {
            "patterns": {},
            "novel_task_count": self.long_memory.get_novel_task_count(),
        }
        for key, pattern in self.long_memory.patterns.items():
            stats["patterns"][key] = {
                "best_strategy": pattern.best_strategy,
                "success_rate": pattern.success_rate,
                "attempts": pattern.attempts,
                "avg_time_s": round(pattern.avg_time_s, 2),
            }
        return stats
