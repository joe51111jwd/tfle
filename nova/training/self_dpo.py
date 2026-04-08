"""Self-DPO learning loop for NOVA.

Closes the think-train loop: preference pairs accumulated during inference
(via bracket tournaments in the preference store) are used to fine-tune
the model via DPO during the sleep phase.

Later-round bracket pairs are weighted higher because finals signal
stronger preference than round 1.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass

import torch
import torch.nn as nn

from .dpo import DPOTrainer, DPOConfig
from .preference_store import PreferenceStore

logger = logging.getLogger(__name__)


@dataclass
class SelfDPOConfig:
    min_pairs: int = 100
    default_epochs: int = 1
    default_batch_size: int = 4
    default_lr: float = 5e-6
    beta: float = 0.1
    max_seq_len: int = 4096


class SelfDPO:
    """Self-improving DPO loop using accumulated preference pairs.

    Checks the preference store for sufficient pairs, weights later-round
    pairs higher (finals > semifinals > round 1), and runs DPO training.
    """

    def __init__(
        self,
        model: nn.Module,
        preference_store: PreferenceStore,
        config: SelfDPOConfig | None = None,
        device: torch.device | None = None,
        tokenizer=None,
    ):
        self.model = model
        self.preference_store = preference_store
        self.config = config or SelfDPOConfig()
        self.device = device or torch.device("cpu")
        self.tokenizer = tokenizer

    def should_train(self) -> bool:
        """Returns True when the preference store has enough pairs for training."""
        return self.preference_store.count() >= self.config.min_pairs

    def _weight_pairs_by_round(self, pairs: list[dict]) -> list[dict]:
        """Duplicate later-round pairs to upweight them in training.

        Round weighting: round N gets weight = N (so finals at round 4 appear
        4x as often as round 1 pairs in a training epoch).
        """
        weighted = []
        for pair in pairs:
            round_num = pair.get("round_num", 1)
            # clamp to reasonable range
            weight = max(1, min(round_num, 8))
            for _ in range(weight):
                weighted.append(pair)
        random.shuffle(weighted)
        return weighted

    def train(
        self,
        num_epochs: int = 1,
        batch_size: int = 4,
        lr: float = 5e-6,
    ) -> dict:
        """Run DPO training on accumulated preference pairs.

        Args:
            num_epochs: Number of passes over the data.
            batch_size: Pairs per training step.
            lr: Learning rate for DPO optimizer.

        Returns:
            Status dict with pairs_used, avg_loss, total_steps, etc.
        """
        pairs = self.preference_store.load_pairs()

        if len(pairs) < self.config.min_pairs:
            logger.info(
                f"Self-DPO skipped: {len(pairs)} pairs < {self.config.min_pairs} minimum"
            )
            return {
                "status": "skipped",
                "pairs_available": len(pairs),
                "min_required": self.config.min_pairs,
            }

        # weight later-round pairs higher
        weighted_pairs = self._weight_pairs_by_round(pairs)

        dpo_config = DPOConfig(
            beta=self.config.beta,
            lr=lr,
            max_seq_len=self.config.max_seq_len,
        )
        dpo_trainer = DPOTrainer(
            model=self.model,
            config=dpo_config,
            device=self.device,
            tokenizer=self.tokenizer,
        )

        # DPOTrainer.train handles epochs, batching, and gradient updates
        dpo_result = dpo_trainer.train(
            preference_pairs=weighted_pairs,
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=lr,
        )

        result = {
            "status": "completed",
            "pairs_used": len(pairs),
            "weighted_pairs": len(weighted_pairs),
            "num_epochs": num_epochs,
            "total_steps": dpo_result["total_steps"],
            "avg_loss": dpo_result["avg_loss"],
            "final_loss": dpo_result["final_loss"],
        }
        logger.info(f"Self-DPO complete: {result}")
        return result
