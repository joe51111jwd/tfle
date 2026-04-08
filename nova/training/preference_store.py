"""Persistent JSONL preference pair storage for Self-DPO.

Bracket matchups accumulate here during inference. The sleep phase
loads them to run DPO training, closing the think-train loop.

Store path defaults to ~/.nova/preference_pairs.jsonl.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_STORE_PATH = Path.home() / ".nova" / "preference_pairs.jsonl"

QUALITY_KEYWORDS = {
    "correct": "accuracy",
    "accurate": "accuracy",
    "wrong": "accuracy",
    "error": "accuracy",
    "mistake": "accuracy",
    "clear": "clarity",
    "readable": "clarity",
    "confusing": "clarity",
    "concise": "conciseness",
    "verbose": "conciseness",
    "brief": "conciseness",
    "wordy": "conciseness",
    "direct": "direct_answer",
    "straightforward": "direct_answer",
    "relevant": "relevance",
    "off-topic": "relevance",
    "complete": "completeness",
    "thorough": "completeness",
    "missing": "completeness",
    "logical": "reasoning",
    "reasoning": "reasoning",
    "step": "reasoning",
}


@dataclass
class Matchup:
    prompt: str
    chosen: str
    rejected: str
    reasoning: str
    quality_principle: str
    round_num: int
    timestamp: float = field(default_factory=time.time)


class PreferenceStore:
    """Persistent JSONL storage for bracket matchup preference pairs.

    Each line is a JSON object with prompt, chosen, rejected, reasoning,
    quality_principle, round_num, and timestamp. Append-only during
    inference; bulk-read during DPO training.
    """

    def __init__(self, store_path: str | Path | None = None):
        self.store_path = Path(store_path) if store_path else DEFAULT_STORE_PATH
        self.store_path.parent.mkdir(parents=True, exist_ok=True)

    def save_matchups(self, matchups: list[dict]) -> int:
        """Append matchups to the JSONL file.

        Each matchup dict should have: prompt, chosen, rejected, reasoning, round_num.
        Extracts quality_principle from reasoning automatically.
        """
        saved = 0
        with open(self.store_path, "a") as f:
            for m in matchups:
                reasoning = m.get("reasoning", "")
                record = {
                    "prompt": m["prompt"],
                    "chosen": m["chosen"],
                    "rejected": m["rejected"],
                    "reasoning": reasoning,
                    "quality_principle": self._extract_principle(reasoning),
                    "round_num": m.get("round_num", 0),
                    "timestamp": time.time(),
                }
                f.write(json.dumps(record) + "\n")
                saved += 1
        logger.info(f"Saved {saved} matchups to {self.store_path}")
        return saved

    def _extract_principle(self, reasoning: str) -> str:
        """Keyword-based labeling of the quality principle from judge reasoning."""
        if not reasoning:
            return "general"

        lower = reasoning.lower()
        principle_counts: dict[str, int] = {}
        for keyword, principle in QUALITY_KEYWORDS.items():
            if keyword in lower:
                principle_counts[principle] = principle_counts.get(principle, 0) + 1

        if not principle_counts:
            return "general"
        return max(principle_counts, key=principle_counts.get)

    def load_pairs(
        self,
        max_pairs: int | None = None,
        min_round: int = 0,
    ) -> list[dict]:
        """Load preference pairs for DPO training.

        Args:
            max_pairs: Maximum number of pairs to return (newest first).
            min_round: Only include pairs from this tournament round or later
                       (later rounds have higher-quality signal).
        """
        if not self.store_path.exists():
            return []

        pairs = []
        with open(self.store_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("round_num", 0) >= min_round:
                    pairs.append(record)

        # Newest first so max_pairs takes the most recent
        pairs.sort(key=lambda r: r.get("timestamp", 0), reverse=True)
        if max_pairs is not None:
            pairs = pairs[:max_pairs]
        return pairs

    def count(self) -> int:
        """Return total number of stored pairs."""
        if not self.store_path.exists():
            return 0
        n = 0
        with open(self.store_path) as f:
            for line in f:
                if line.strip():
                    n += 1
        return n

    def clear_old(self, days: int = 30) -> int:
        """Remove pairs older than `days` days. Returns count removed."""
        if not self.store_path.exists():
            return 0

        cutoff = time.time() - days * 86400
        kept = []
        removed = 0

        with open(self.store_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("timestamp", 0) >= cutoff:
                    kept.append(line)
                else:
                    removed += 1

        with open(self.store_path, "w") as f:
            for line in kept:
                f.write(line + "\n")

        logger.info(f"Cleared {removed} old pairs (>{days} days), kept {len(kept)}")
        return removed
