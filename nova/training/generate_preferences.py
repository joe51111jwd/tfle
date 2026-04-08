"""Preference pair generation via student-teacher judging.

The student model generates two responses at different temperatures.
The teacher model (judge) picks the better one. Pairs are saved as
JSONL for DPO training.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    from .tokenizer_setup import get_tokenizer
except ImportError:
    get_tokenizer = None

logger = logging.getLogger(__name__)


class PreferenceGenerator:
    """Generate preference pairs by having a teacher judge student responses.

    For each prompt:
      1. Student generates response A (low temp) and response B (high temp)
      2. Teacher scores both and picks the winner
      3. The pair (chosen, rejected) is saved for DPO
    """

    def __init__(
        self,
        student_model: torch.nn.Module,
        teacher_model: torch.nn.Module,
        tokenizer=None,
        low_temp: float = 0.3,
        high_temp: float = 0.9,
        max_new_tokens: int = 256,
    ):
        self.student = student_model
        self.teacher = teacher_model
        self.tokenizer = tokenizer or (get_tokenizer() if get_tokenizer else None)
        self.low_temp = low_temp
        self.high_temp = high_temp
        self.max_new_tokens = max_new_tokens

    @torch.no_grad()
    def _generate_response(
        self,
        model: torch.nn.Module,
        prompt: str,
        temperature: float,
        max_new_tokens: int | None = None,
    ) -> str:
        """Autoregressive generation from a model."""
        max_new = max_new_tokens or self.max_new_tokens
        device = next(model.parameters()).device
        prompt_ids = self.tokenizer.encode(prompt)
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        for _ in range(max_new):
            logits = model(ids)
            next_logits = logits[:, -1, :].float() / max(temperature, 1e-8)
            probs = F.softmax(next_logits, dim=-1)
            tok = torch.multinomial(probs, 1)
            ids = torch.cat([ids, tok], dim=1)

        return self.tokenizer.decode(ids[0, len(prompt_ids):].tolist())

    @torch.no_grad()
    def _teacher_judge(self, prompt: str, response_a: str, response_b: str) -> dict:
        """Teacher model judges which response is better.

        Uses the teacher to score each response by computing average
        log-probability (how much the teacher "agrees" with the response).
        Higher score = teacher thinks it's a better response.
        """
        device = next(self.teacher.parameters()).device

        score_a = self._score_response(self.teacher, prompt, response_a, device)
        score_b = self._score_response(self.teacher, prompt, response_b, device)

        if score_a >= score_b:
            return {
                "chosen": response_a,
                "rejected": response_b,
                "chosen_score": score_a,
                "rejected_score": score_b,
                "reasoning": f"Response A scored {score_a:.4f} vs B {score_b:.4f}",
            }
        return {
            "chosen": response_b,
            "rejected": response_a,
            "chosen_score": score_b,
            "rejected_score": score_a,
            "reasoning": f"Response B scored {score_b:.4f} vs A {score_a:.4f}",
        }

    @torch.no_grad()
    def _score_response(
        self,
        model: torch.nn.Module,
        prompt: str,
        response: str,
        device: torch.device,
    ) -> float:
        """Score a response as mean log-prob under the model."""
        prompt_ids = self.tokenizer.encode(prompt)
        response_ids = self.tokenizer.encode(response)
        full_ids = torch.tensor(
            [prompt_ids + response_ids], dtype=torch.long, device=device
        )
        prompt_len = len(prompt_ids)

        logits = model(full_ids)
        log_probs = F.log_softmax(logits[:, prompt_len - 1:-1, :].float(), dim=-1)
        targets = full_ids[:, prompt_len:]
        token_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

        mask = (targets != 0).float()
        n_tokens = mask.sum().clamp(min=1)
        return (token_lp * mask).sum().item() / n_tokens.item()

    def generate_pair(self, prompt: str) -> dict:
        """Generate a single preference pair for a prompt.

        Two student responses at different temperatures, judged by teacher.
        """
        response_a = self._generate_response(self.student, prompt, self.low_temp)
        response_b = self._generate_response(self.student, prompt, self.high_temp)

        judgment = self._teacher_judge(prompt, response_a, response_b)
        return {
            "prompt": prompt,
            "chosen": judgment["chosen"],
            "rejected": judgment["rejected"],
            "chosen_score": judgment["chosen_score"],
            "rejected_score": judgment["rejected_score"],
            "reasoning": judgment["reasoning"],
        }

    def generate_dataset(
        self,
        prompts: list[str],
        output_path: str | Path,
    ) -> list[dict]:
        """Batch-generate preference pairs and save as JSONL.

        Args:
            prompts: List of prompt strings.
            output_path: Path to write the JSONL file.

        Returns:
            List of generated preference pair dicts.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        pairs = []
        with open(output_path, "w") as f:
            for i, prompt in enumerate(prompts):
                pair = self.generate_pair(prompt)
                f.write(json.dumps(pair) + "\n")
                pairs.append(pair)

                if (i + 1) % 10 == 0:
                    logger.info(f"Generated {i + 1}/{len(prompts)} preference pairs")

        logger.info(f"Saved {len(pairs)} preference pairs to {output_path}")
        return pairs
