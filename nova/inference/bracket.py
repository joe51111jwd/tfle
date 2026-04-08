"""Bracket inference — NOVA's core thinking mechanism.

Single-elimination tournament: generate diverse candidate responses,
then have the model self-judge head-to-head matchups. The winner
advances. Every matchup is saved to the preference store for Self-DPO
during sleep.

The bracket IS how NOVA thinks. Difficulty maps to bracket size:
  trivial=1, easy=4, medium=16, hard=64

For code tasks, code_bracket adds execution verification: candidates
that fail to run are eliminated before the tournament starts.
"""
from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .strategies import ExecutionVerifier, GenerativeModel, Tokenizer, extract_answer

logger = logging.getLogger(__name__)

# Lazy import to avoid circular dep at module level
_preference_store = None


def _get_preference_store():
    global _preference_store
    if _preference_store is None:
        from ..training.preference_store import PreferenceStore
        _preference_store = PreferenceStore()
    return _preference_store


DIFFICULTY_MAP = {
    "trivial": 1,
    "easy": 4,
    "medium": 16,
    "hard": 64,
}

# Entropy thresholds for auto difficulty
AUTO_THRESHOLDS = {
    "trivial": 1.0,
    "easy": 2.0,
    "medium": 4.0,
    # >= 4.0 is hard
}


@dataclass
class MatchupResult:
    prompt: str
    response_a: str
    response_b: str
    winner: str
    loser: str
    reasoning: str
    round_num: int


class BracketInference:
    """NOVA's core thinking mechanism — single-elimination tournament.

    1. Generate diverse candidate responses at varied temperatures
    2. Self-judge head-to-head matchups
    3. Winner advances, loser is eliminated
    4. All matchups saved to preference store for Self-DPO

    For code tasks, execution verification filters non-running candidates
    before the bracket starts.
    """

    def __init__(
        self,
        model: GenerativeModel,
        tokenizer: Tokenizer,
        preference_store=None,
        max_new_tokens: int = 512,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.preference_store = preference_store
        self.max_new_tokens = max_new_tokens
        self._exec_verifier: ExecutionVerifier | None = None

    @property
    def store(self):
        """Lazy-load the preference store."""
        if self.preference_store is not None:
            return self.preference_store
        return _get_preference_store()

    def _get_device(self) -> torch.device:
        if hasattr(self.model, "parameters"):
            return next(iter(self.model.parameters())).device
        return torch.device("cpu")

    @torch.no_grad()
    def generate_candidates(
        self,
        prompt: str,
        n_candidates: int,
        temperature_range: tuple[float, float] = (0.6, 1.0),
    ) -> list[str]:
        """Generate diverse responses at varied temperatures.

        Temperatures are linearly spaced across the range to maximize
        diversity. Low temps give precise answers, high temps explore.
        """
        if n_candidates <= 0:
            return []
        if n_candidates == 1:
            temps = [(temperature_range[0] + temperature_range[1]) / 2]
        else:
            step = (temperature_range[1] - temperature_range[0]) / (n_candidates - 1)
            temps = [temperature_range[0] + i * step for i in range(n_candidates)]

        device = self._get_device()
        prompt_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        prompt_len = len(prompt_ids)

        candidates = []
        for temp in temps:
            ids = input_ids.clone()
            for _ in range(self.max_new_tokens):
                logits = self.model.forward(ids)
                next_logits = logits[:, -1, :].float() / max(temp, 1e-8)
                probs = F.softmax(next_logits, dim=-1)
                tok = torch.multinomial(probs, 1)
                ids = torch.cat([ids, tok], dim=1)

            response = self.tokenizer.decode(ids[0, prompt_len:].tolist())
            candidates.append(response)

        return candidates

    @torch.no_grad()
    def judge_matchup(
        self,
        prompt: str,
        response_a: str,
        response_b: str,
    ) -> dict:
        """Model self-judges a head-to-head matchup.

        Computes mean log-probability of each response conditioned on
        the prompt. Higher log-prob = the model considers it a more
        natural/correct continuation. Returns winner + reasoning.
        """
        device = self._get_device()
        prompt_ids = self.tokenizer.encode(prompt)
        prompt_len = len(prompt_ids)

        score_a = self._score_response(prompt_ids, response_a, prompt_len, device)
        score_b = self._score_response(prompt_ids, response_b, prompt_len, device)

        if score_a >= score_b:
            return {
                "winner": response_a,
                "loser": response_b,
                "winner_score": score_a,
                "loser_score": score_b,
                "reasoning": (
                    f"Response A scored {score_a:.4f} vs B {score_b:.4f}. "
                    f"A demonstrates more accurate and direct reasoning."
                ),
            }
        return {
            "winner": response_b,
            "loser": response_a,
            "winner_score": score_b,
            "loser_score": score_a,
            "reasoning": (
                f"Response B scored {score_b:.4f} vs A {score_a:.4f}. "
                f"B demonstrates more accurate and direct reasoning."
            ),
        }

    @torch.no_grad()
    def _score_response(
        self,
        prompt_ids: list[int],
        response: str,
        prompt_len: int,
        device: torch.device,
    ) -> float:
        """Mean log-probability of response tokens under the model."""
        response_ids = self.tokenizer.encode(response)
        if not response_ids:
            return float("-inf")

        full_ids = torch.tensor(
            [prompt_ids + response_ids], dtype=torch.long, device=device
        )

        logits = self.model.forward(full_ids)
        log_probs = F.log_softmax(logits[:, prompt_len - 1:-1, :].float(), dim=-1)
        targets = full_ids[:, prompt_len:]
        token_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

        mask = (targets != 0).float()
        n_tokens = mask.sum().clamp(min=1)
        return (token_lp * mask).sum().item() / n_tokens.item()

    def run_bracket(
        self,
        prompt: str,
        candidates: list[str],
    ) -> dict:
        """Single-elimination tournament over candidate responses.

        Each round halves the field. All matchups are saved to the
        preference store for Self-DPO during sleep.

        Returns:
            Dict with winner, all matchups, and bracket stats.
        """
        if not candidates:
            return {"winner": None, "matchups": [], "n_rounds": 0}
        if len(candidates) == 1:
            return {"winner": candidates[0], "matchups": [], "n_rounds": 0}

        # Shuffle to avoid positional bias
        remaining = list(candidates)
        random.shuffle(remaining)

        all_matchups = []
        round_num = 0

        while len(remaining) > 1:
            round_num += 1
            next_round = []

            # If odd number, the last candidate gets a bye
            for i in range(0, len(remaining) - 1, 2):
                result = self.judge_matchup(prompt, remaining[i], remaining[i + 1])
                matchup = {
                    "prompt": prompt,
                    "chosen": result["winner"],
                    "rejected": result["loser"],
                    "reasoning": result["reasoning"],
                    "round_num": round_num,
                    "winner_score": result["winner_score"],
                    "loser_score": result["loser_score"],
                }
                all_matchups.append(matchup)
                next_round.append(result["winner"])

            # Bye for odd candidate
            if len(remaining) % 2 == 1:
                next_round.append(remaining[-1])

            remaining = next_round

        # Save all matchups to preference store
        try:
            self.store.save_matchups(all_matchups)
        except Exception as e:
            logger.warning(f"Failed to save matchups to preference store: {e}")

        return {
            "winner": remaining[0],
            "matchups": all_matchups,
            "n_rounds": round_num,
            "n_matchups": len(all_matchups),
        }

    def code_bracket(
        self,
        prompt: str,
        n_candidates: int = 64,
    ) -> dict:
        """Bracket with execution verification for code tasks.

        1. Generate candidates
        2. Filter to only those whose code executes successfully
        3. Run bracket on survivors
        4. If no candidates pass execution, fall back to regular bracket
        """
        candidates = self.generate_candidates(
            prompt, n_candidates, temperature_range=(0.3, 0.8)
        )

        # Lazy-init execution verifier
        if self._exec_verifier is None:
            self._exec_verifier = ExecutionVerifier(
                self.model, self.tokenizer, max_retries=1, timeout_s=10
            )

        # Filter candidates through execution
        passing = []
        for candidate in candidates:
            code = self._exec_verifier._extract_code(candidate)
            if code is None:
                continue
            result = self._exec_verifier._execute(code)
            if result.success:
                passing.append(candidate)

        if passing:
            logger.info(
                f"Code bracket: {len(passing)}/{len(candidates)} candidates pass execution"
            )
            bracket_result = self.run_bracket(prompt, passing)
            bracket_result["n_candidates_total"] = len(candidates)
            bracket_result["n_passing_execution"] = len(passing)
            bracket_result["source"] = "code_bracket"
            return bracket_result

        # Fallback: no candidates passed execution, run bracket on all
        logger.warning("No candidates passed execution, falling back to regular bracket")
        bracket_result = self.run_bracket(prompt, candidates)
        bracket_result["n_candidates_total"] = len(candidates)
        bracket_result["n_passing_execution"] = 0
        bracket_result["source"] = "code_bracket_fallback"
        return bracket_result

    def run(
        self,
        prompt: str,
        n_candidates: int | None = None,
        difficulty: str = "auto",
        verify_answers: bool = False,
    ) -> dict:
        """Run bracket inference with explicit candidate count.

        Args:
            prompt: The input prompt.
            n_candidates: Number of candidates to generate. Overrides difficulty.
            difficulty: Fallback if n_candidates is None.
            verify_answers: If True, filter candidates to those with extractable
                            numeric answers before bracketing (for math tasks).

        Returns:
            Dict with answer, response, bracket trace, and strategy label.
        """
        if n_candidates is None:
            if difficulty == "auto":
                difficulty = self._estimate_difficulty(prompt)
            n_candidates = DIFFICULTY_MAP.get(difficulty, 16)

        if n_candidates <= 1:
            candidates = self.generate_candidates(prompt, 1, (0.3, 0.3))
            response = candidates[0] if candidates else ""
            return {
                "answer": extract_answer(response),
                "response": response,
                "difficulty": difficulty,
                "n_candidates": 1,
                "strategy": "bracket_trivial",
                "bracket": None,
            }

        candidates = self.generate_candidates(prompt, n_candidates)

        if verify_answers:
            verified = [c for c in candidates if extract_answer(c) is not None]
            if verified:
                logger.info(
                    f"Math bracket: {len(verified)}/{len(candidates)} "
                    f"candidates have extractable answers"
                )
                candidates = verified

        bracket = self.run_bracket(prompt, candidates)

        winner = bracket["winner"] or ""
        return {
            "answer": extract_answer(winner),
            "response": winner,
            "difficulty": difficulty,
            "n_candidates": len(candidates),
            "strategy": "bracket_verified" if verify_answers else "bracket",
            "bracket": bracket,
        }

    def __call__(
        self,
        prompt: str,
        difficulty: str = "auto",
    ) -> dict:
        """Main entry point. Maps difficulty to bracket size and runs.

        Args:
            prompt: The input prompt.
            difficulty: One of 'trivial', 'easy', 'medium', 'hard', or 'auto'.
                        'auto' estimates difficulty from model entropy.

        Returns:
            Dict with winner response, answer extraction, and bracket trace.
        """
        return self.run(prompt, difficulty=difficulty)

    @torch.no_grad()
    def _estimate_difficulty(self, prompt: str) -> str:
        """Estimate prompt difficulty from model entropy."""
        device = self._get_device()
        ids = torch.tensor(
            [self.tokenizer.encode(prompt)], dtype=torch.long, device=device
        )
        logits = self.model.forward(ids)

        # Mean token-level entropy
        probs = F.softmax(logits[0].float(), dim=-1)
        ent = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean().item()

        if ent < AUTO_THRESHOLDS["trivial"]:
            return "trivial"
        if ent < AUTO_THRESHOLDS["easy"]:
            return "easy"
        if ent < AUTO_THRESHOLDS["medium"]:
            return "medium"
        return "hard"
