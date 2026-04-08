"""Preference-based evaluation using a teacher model as judge.

The teacher scores model responses 1-10 via log-probability agreement.
This is the quality signal for tracking Self-DPO progress over time.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class PreferenceEvaluator:
    """Evaluate model response quality using a teacher model as judge.

    The teacher scores each response by computing its mean log-probability
    of the response tokens, then maps that to a 1-10 scale.
    """

    def __init__(
        self,
        student_model: torch.nn.Module,
        teacher_model: torch.nn.Module,
        tokenizer,
        max_new_tokens: int = 256,
        temperature: float = 0.3,
    ):
        self.student = student_model
        self.teacher = teacher_model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def _decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    @torch.no_grad()
    def _generate(self, prompt: str) -> str:
        """Generate a response from the student model."""
        device = next(self.student.parameters()).device
        ids = torch.tensor(
            [self._encode(prompt)], dtype=torch.long, device=device
        )
        prompt_len = ids.shape[1]

        for _ in range(self.max_new_tokens):
            logits = self.student(ids)
            next_logits = logits[:, -1, :].float() / max(self.temperature, 1e-8)
            probs = F.softmax(next_logits, dim=-1)
            tok = torch.multinomial(probs, 1)
            ids = torch.cat([ids, tok], dim=1)

        return self._decode(ids[0, prompt_len:].tolist())

    @torch.no_grad()
    def _teacher_score(self, prompt: str, response: str) -> float:
        """Score a response using teacher model log-probabilities.

        Returns a score from 1 to 10 based on mean log-prob.
        """
        device = next(self.teacher.parameters()).device
        prompt_ids = self._encode(prompt)
        response_ids = self._encode(response)
        full_ids = torch.tensor(
            [prompt_ids + response_ids], dtype=torch.long, device=device
        )
        prompt_len = len(prompt_ids)

        if len(response_ids) == 0:
            return 1.0

        logits = self.teacher(full_ids)
        log_probs = F.log_softmax(logits[:, prompt_len - 1:-1, :].float(), dim=-1)
        targets = full_ids[:, prompt_len:]
        token_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

        mask = (targets != 0).float()
        n_tokens = mask.sum().clamp(min=1)
        mean_lp = (token_lp * mask).sum().item() / n_tokens.item()

        # Map log-prob to 1-10 scale
        # Typical mean log-probs range from ~-10 (bad) to ~-1 (good)
        score = max(1.0, min(10.0, 10.0 + mean_lp))
        return round(score, 2)

    def evaluate_single(self, prompt: str) -> dict:
        """Generate a response and have the teacher score it 1-10."""
        response = self._generate(prompt)
        score = self._teacher_score(prompt, response)
        return {
            "prompt": prompt,
            "response": response,
            "score": score,
        }

    def run_eval_suite(
        self,
        prompts: list[str],
        output_path: str | Path | None = None,
    ) -> dict:
        """Evaluate on a full set of prompts.

        Args:
            prompts: List of evaluation prompts.
            output_path: Optional path to save results as JSON.

        Returns:
            Dict with average_score, per-prompt results, and timing.
        """
        t0 = time.time()
        results = []

        for i, prompt in enumerate(prompts):
            result = self.evaluate_single(prompt)
            results.append(result)

            if (i + 1) % 10 == 0:
                avg_so_far = sum(r["score"] for r in results) / len(results)
                logger.info(
                    f"Evaluated {i + 1}/{len(prompts)}, "
                    f"running avg: {avg_so_far:.2f}/10"
                )

        elapsed = time.time() - t0
        scores = [r["score"] for r in results]
        avg_score = sum(scores) / max(len(scores), 1)

        summary = {
            "average_score": round(avg_score, 2),
            "n_prompts": len(prompts),
            "elapsed_s": round(elapsed, 2),
            "min_score": min(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
            "results": results,
        }

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Eval results saved to {output_path}")

        logger.info(f"Preference eval: avg={avg_score:.2f}/10 over {len(prompts)} prompts")
        return summary
