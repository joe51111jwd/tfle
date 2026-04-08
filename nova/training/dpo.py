"""DPO (Direct Preference Optimization) trainer for NOVA Self-DPO.

Bracket matchups from inference feed this trainer during the sleep phase.
The model learns from its own preference pairs — no external reward model.

Loss: -log sigmoid(beta * (log_ratio_chosen - log_ratio_rejected))
where log_ratio = log_pi(response|prompt) - log_pi_ref(response|prompt)
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .tokenizer_setup import get_tokenizer, encode
except ImportError:
    get_tokenizer = None
    encode = None

logger = logging.getLogger(__name__)


@dataclass
class DPOConfig:
    beta: float = 0.1
    lr: float = 5e-6
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    max_seq_len: int = 4096


class DPOTrainer:
    """Direct Preference Optimization on bracket-generated preference pairs.

    Uses a frozen snapshot of the current model as the reference policy.
    Beta controls how much the policy can deviate from the reference.
    """

    def __init__(
        self,
        model: nn.Module,
        config: DPOConfig | None = None,
        device: torch.device | None = None,
        tokenizer=None,
    ):
        self.model = model
        self.config = config or DPOConfig()
        self.device = device or next(model.parameters()).device
        self.tokenizer = tokenizer or (get_tokenizer() if get_tokenizer else None)
        self.ref_model: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None

    def prepare(self):
        """Snapshot current model as frozen reference and set up optimizer."""
        self.ref_model = copy.deepcopy(self.model)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        logger.info("DPO prepared: reference model frozen, optimizer ready")

    def _encode(self, text: str) -> list[int]:
        if self.tokenizer is not None:
            return self.tokenizer.encode(text)
        if encode is not None:
            return encode(text)
        return [ord(c) % 32000 for c in text.split()]

    def _response_logprob(
        self,
        model: nn.Module,
        token_ids: torch.Tensor,
        prompt_len: int,
    ) -> torch.Tensor:
        """Compute log probability of response tokens only (after prompt).

        Args:
            model: The model to evaluate.
            token_ids: Full sequence [prompt + response], shape (1, seq_len).
            prompt_len: Number of tokens in the prompt prefix.

        Returns:
            Scalar tensor: sum of log probs over response tokens.
        """
        use_amp = self.device.type == "cuda"
        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(token_ids)
        else:
            logits = model(token_ids)

        logits = logits.float()
        # Shift: predict token t from position t-1
        response_logits = logits[:, prompt_len - 1:-1, :]
        response_targets = token_ids[:, prompt_len:]

        log_probs = F.log_softmax(response_logits, dim=-1)
        token_log_probs = log_probs.gather(-1, response_targets.unsqueeze(-1)).squeeze(-1)

        # Mask out padding (token id 0)
        mask = (response_targets != 0).float()
        return (token_log_probs * mask).sum(dim=-1)

    def compute_loss(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
    ) -> torch.Tensor:
        """Compute DPO loss for a single preference pair.

        loss = -log sigmoid(beta * (log_ratio_chosen - log_ratio_rejected))
        where log_ratio = log_pi(y|x) - log_pi_ref(y|x)
        """
        if self.ref_model is None:
            raise RuntimeError("Call prepare() before compute_loss()")

        prompt_ids = self._encode(prompt)
        chosen_ids = self._encode(chosen)
        rejected_ids = self._encode(rejected)
        prompt_len = len(prompt_ids)

        # Build full sequences: prompt + response
        chosen_full = torch.tensor(
            [prompt_ids + chosen_ids], dtype=torch.long, device=self.device
        )
        rejected_full = torch.tensor(
            [prompt_ids + rejected_ids], dtype=torch.long, device=self.device
        )

        # Truncate to max_seq_len
        chosen_full = chosen_full[:, :self.config.max_seq_len]
        rejected_full = rejected_full[:, :self.config.max_seq_len]

        # Policy log probs
        chosen_policy_lp = self._response_logprob(self.model, chosen_full, prompt_len)
        rejected_policy_lp = self._response_logprob(self.model, rejected_full, prompt_len)

        # Reference log probs (frozen, no grad)
        with torch.no_grad():
            chosen_ref_lp = self._response_logprob(self.ref_model, chosen_full, prompt_len)
            rejected_ref_lp = self._response_logprob(self.ref_model, rejected_full, prompt_len)

        # Log ratios
        log_ratio_chosen = chosen_policy_lp - chosen_ref_lp
        log_ratio_rejected = rejected_policy_lp - rejected_ref_lp

        # DPO loss
        loss = -F.logsigmoid(self.config.beta * (log_ratio_chosen - log_ratio_rejected))
        return loss.mean()

    def train(
        self,
        preference_pairs: list[dict],
        num_epochs: int = 1,
        batch_size: int = 4,
        lr: float | None = None,
    ) -> dict:
        """Full DPO training loop over preference pairs.

        Args:
            preference_pairs: List of dicts with 'prompt', 'chosen', 'rejected'.
            num_epochs: Number of passes over the data.
            batch_size: Number of pairs per gradient step.
            lr: Override learning rate (uses config default if None).

        Returns:
            Training stats dict.
        """
        if self.ref_model is None:
            self.prepare()

        if lr is not None:
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

        self.model.train()
        total_loss = 0.0
        n_steps = 0
        step_losses = []

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_steps = 0

            for i in range(0, len(preference_pairs), batch_size):
                batch = preference_pairs[i:i + batch_size]
                batch_loss = torch.tensor(0.0, device=self.device)

                for pair in batch:
                    pair_loss = self.compute_loss(
                        pair["prompt"], pair["chosen"], pair["rejected"]
                    )
                    batch_loss = batch_loss + pair_loss

                batch_loss = batch_loss / len(batch)

                self.optimizer.zero_grad()
                batch_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )
                self.optimizer.step()

                loss_val = batch_loss.item()
                epoch_loss += loss_val
                total_loss += loss_val
                n_steps += 1
                epoch_steps += 1
                step_losses.append(loss_val)

            avg_epoch = epoch_loss / max(epoch_steps, 1)
            logger.info(f"DPO epoch {epoch + 1}/{num_epochs}: loss={avg_epoch:.4f}")

        avg_loss = total_loss / max(n_steps, 1)
        logger.info(f"DPO training complete: {n_steps} steps, avg_loss={avg_loss:.4f}")

        return {
            "total_steps": n_steps,
            "num_epochs": num_epochs,
            "avg_loss": avg_loss,
            "final_loss": step_losses[-1] if step_losses else 0.0,
            "n_pairs": len(preference_pairs),
        }
