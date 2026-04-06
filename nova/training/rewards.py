"""Reward functions for GRPO training.

Three reward signals:
- MathReward: binary correctness + proximity partial credit
- CodeReward: sandbox execution with test cases
- GranularFormatReward: 0-0.7 scale checking reasoning structure markers

Combined reward aggregates all applicable signals per sample.
"""
from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


class MathReward:
    """Score math answers: binary correct (1.0) + proximity partial credit.

    Extracts the predicted number from "answer is <N>" patterns or falls back
    to the last number in the text. Partial credit scales linearly with distance.
    """

    def score(self, question: str, response: str, ground_truth) -> float:
        predicted = self._extract_answer(response)
        if predicted is None:
            return 0.0

        if self._exact_match(predicted, ground_truth):
            return 1.0

        return self._proximity_credit(predicted, ground_truth)

    def _extract_answer(self, text: str) -> str | None:
        # Try <answer>N</answer> format first
        m = re.search(r"<answer>\s*(-?\d+\.?\d*)\s*</answer>", text)
        if m:
            return m.group(1)

        # Try "answer is N"
        m = re.search(r"answer\s+is\s+(-?\d+\.?\d*)", text, re.IGNORECASE)
        if m:
            return m.group(1)

        # Try "= N"
        m = re.search(r"=\s*(-?\d+\.?\d*)", text)
        if m:
            return m.group(1)

        # Last number in text
        nums = re.findall(r"-?\d+\.?\d*", text)
        return nums[-1] if nums else None

    def _exact_match(self, predicted: str, truth) -> bool:
        try:
            return abs(float(predicted) - float(truth)) < 1e-4
        except (ValueError, TypeError):
            return False

    def _proximity_credit(self, predicted: str, truth) -> float:
        try:
            dist = abs(float(predicted) - float(truth))
            max_dist = abs(float(truth)) + 1
            return max(0.0, 0.5 * (1.0 - min(dist / max_dist, 1.0)))
        except (ValueError, TypeError):
            return 0.0


class CodeReward:
    """Score code solutions by extracting, executing in sandbox, and checking test cases.

    Extracts code from markdown fences or <answer> blocks, writes to a temp file,
    runs with a timeout, and checks expected outputs.
    """

    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout

    def score(self, question: str, response: str, test_cases: list[dict]) -> float:
        """Score based on test case pass rate.

        test_cases: [{"input": "...", "expected": "..."}, ...]
        """
        code = self._extract_code(response)
        if not code:
            return 0.0

        passed = 0
        for tc in test_cases:
            if self._run_test(code, tc["input"], tc["expected"]):
                passed += 1

        return passed / max(len(test_cases), 1)

    def _extract_code(self, text: str) -> str | None:
        # Try markdown code fence
        m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
        if m:
            return m.group(1).strip()

        # Try <answer> block
        m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if m:
            return m.group(1).strip()

        # If the whole response looks like code (has def/import/print)
        if any(kw in text for kw in ("def ", "import ", "print(")):
            return text.strip()

        return None

    def _run_test(self, code: str, test_input: str, expected: str) -> bool:
        full_code = code + "\n" + test_input
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(full_code)
                f.flush()
                result = subprocess.run(
                    ["python3", f.name],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
                Path(f.name).unlink(missing_ok=True)

            stdout = result.stdout.strip()
            return stdout == expected.strip()
        except (subprocess.TimeoutExpired, Exception):
            return False


class GranularFormatReward:
    """Multi-marker format reward on a 0-0.7 scale.

    Checks six independent signals for structured reasoning output:
    - "answer" keyword present (0.1)
    - "think"/"step" reasoning markers (0.1)
    - "need"/"first" planning markers (0.1)
    - "=" computation marker (0.1)
    - Good response length 15-200 words (0.1)
    - Answer is a number in "answer is N" pattern (0.2)
    """

    WEIGHTS = {
        "answer_keyword": 0.1,
        "reasoning_markers": 0.1,
        "planning_markers": 0.1,
        "equals_sign": 0.1,
        "good_length": 0.1,
        "answer_is_number": 0.2,
    }

    def score(self, text: str) -> float:
        total = 0.0
        low = text.lower()

        if "answer" in low:
            total += self.WEIGHTS["answer_keyword"]

        if any(w in low for w in ("think", "step", "<think>")):
            total += self.WEIGHTS["reasoning_markers"]

        if any(w in low for w in ("need", "first")):
            total += self.WEIGHTS["planning_markers"]

        if "=" in text:
            total += self.WEIGHTS["equals_sign"]

        word_count = len(text.split())
        if 15 <= word_count <= 200:
            total += self.WEIGHTS["good_length"]

        if re.search(r"answer\s+is\s+\d+", low) or re.search(r"<answer>\s*\d+", low):
            total += self.WEIGHTS["answer_is_number"]

        return total

    def detailed_score(self, text: str) -> dict[str, float]:
        """Return per-marker breakdown for analysis."""
        low = text.lower()
        return {
            "answer_keyword": self.WEIGHTS["answer_keyword"] if "answer" in low else 0.0,
            "reasoning_markers": self.WEIGHTS["reasoning_markers"] if any(w in low for w in ("think", "step", "<think>")) else 0.0,
            "planning_markers": self.WEIGHTS["planning_markers"] if any(w in low for w in ("need", "first")) else 0.0,
            "equals_sign": self.WEIGHTS["equals_sign"] if "=" in text else 0.0,
            "good_length": self.WEIGHTS["good_length"] if 15 <= len(text.split()) <= 200 else 0.0,
            "answer_is_number": self.WEIGHTS["answer_is_number"] if (re.search(r"answer\s+is\s+\d+", low) or re.search(r"<answer>\s*\d+", low)) else 0.0,
        }


def combined_reward(
    question: str,
    response: str,
    ground_truth=None,
    test_cases: list[dict] | None = None,
    math_weight: float = 0.6,
    format_weight: float = 0.3,
    code_weight: float = 0.1,
) -> float:
    """Aggregate math, format, and code rewards into a single scalar.

    Weights are normalized so they sum to the number of active reward types.
    Inactive reward types (missing ground_truth/test_cases) are excluded.
    """
    math_reward = MathReward()
    format_reward = GranularFormatReward()
    code_reward = CodeReward()

    total = 0.0
    weight_sum = 0.0

    # Format reward is always active
    total += format_weight * format_reward.score(response)
    weight_sum += format_weight

    # Math reward if ground truth provided
    if ground_truth is not None:
        total += math_weight * math_reward.score(question, response, ground_truth)
        weight_sum += math_weight

    # Code reward if test cases provided
    if test_cases:
        total += code_weight * code_reward.score(question, response, test_cases)
        weight_sum += code_weight

    return total / max(weight_sum, 1e-8)
