"""TFLE-GRPO integration: gradient-free RL with block-local TFLE.

Uses block-local TFLE with binary reward as the fitness signal instead of
perplexity. Group generation + scoring replaces the standard TFLE fitness
evaluation, connecting evolutionary weight search to reward-based learning.
"""
from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block_local_tfle import BlockLocalTFLE, BlockLocalConfig
from .rewards import MathReward, GranularFormatReward, combined_reward

logger = logging.getLogger(__name__)


@dataclass
class TFLEGRPOConfig:
    block_size: int = 64
    K: int = 512
    flip_rate: float = 0.05
    re_eval_tolerance: float = 0.005
    fisher_refresh_every: int = 1000
    fisher_boost_factor: float = 3.0
    group_size: int = 16
    max_response_len: int = 200
    max_seq_len: int = 4096
    temperature: float = 1.0
    questions_per_step: int = 4
    re_eval_questions: int = 4
    num_steps: int = 100
    eval_every: int = 10
    embed_lr: float = 1e-5


class BlockLocalTFLEGRPO:
    """TFLE-GRPO: evolutionary RL with block-local weight search.

    Instead of using cross-entropy loss as fitness, uses the mean reward
    from generating G responses to math questions. This connects TFLE's
    gradient-free search directly to the RL objective.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TFLEGRPOConfig,
        device: torch.device,
        tokenizer=None,
    ):
        self.model = model
        self.config = config
        self.device = device
        self.tokenizer = tokenizer

        # initialize block-local TFLE
        bl_config = BlockLocalConfig(
            block_size=config.block_size,
            K=config.K,
            flip_rate=config.flip_rate,
            re_eval_tolerance=config.re_eval_tolerance,
            fisher_refresh_every=config.fisher_refresh_every,
            fisher_boost_factor=config.fisher_boost_factor,
            embed_lr=config.embed_lr,
        )
        self.tfle = BlockLocalTFLE(model, bl_config, device)

        self.math_reward = MathReward()
        self.format_reward = GranularFormatReward()
        self.step_count = 0

    def _encode(self, text: str) -> list[int]:
        if self.tokenizer is not None:
            return self.tokenizer.encode(text)
        return [ord(c) % 32000 for c in text.split()]

    def _decode(self, ids: list[int]) -> str:
        if self.tokenizer is not None:
            return self.tokenizer.decode(ids)
        return " ".join(str(i) for i in ids)

    @torch.no_grad()
    def _generate_responses(
        self,
        prompt_ids: torch.Tensor,
        G: int | None = None,
    ) -> torch.Tensor:
        """Generate G responses from a single prompt."""
        G = G or self.config.group_size
        self.model.eval()

        ids = prompt_ids.expand(G, -1).clone()
        for _ in range(self.config.max_response_len):
            if ids.shape[1] >= self.config.max_seq_len:
                break
            use_amp = self.device.type == "cuda"
            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = self.model(ids)
            else:
                logits = self.model(ids)

            nxt = logits[:, -1, :].float() / max(self.config.temperature, 1e-8)
            probs = F.softmax(nxt, dim=-1)
            tok = torch.multinomial(probs, 1)
            ids = torch.cat([ids, tok], dim=1)

        return ids

    def fitness_fn(self, questions: list[dict]) -> float:
        """Compute reward-based fitness: mean reward over generated responses.

        This replaces the standard perplexity-based fitness in TFLE.
        """
        G = self.config.group_size
        total_reward = 0.0
        n = 0

        for qdata in questions:
            ids = self._encode(qdata["question"])
            prompt_ids = torch.tensor([ids], dtype=torch.long, device=self.device)

            seqs = self._generate_responses(prompt_ids, G=G)

            for g in range(G):
                prompt_len = len(ids)
                resp_tokens = seqs[g, prompt_len:].tolist()
                resp = self._decode([t for t in resp_tokens if t != 0])

                r = combined_reward(
                    qdata["question"], resp,
                    ground_truth=qdata.get("answer"),
                )
                total_reward += r
                n += 1

        return total_reward / max(n, 1)

    def step(self, questions: list[dict]) -> dict:
        """One TFLE-GRPO step.

        1. Select block (Fisher-weighted)
        2. Evaluate current fitness (reward-based)
        3. Generate K proposals for the block
        4. Evaluate best proposal on fresh questions
        5. Accept/reject with tolerance check
        6. Co-evolve embeddings
        """
        self.step_count += 1

        # select block
        block_idx = self.tfle._select_block()
        name, module, r_s, c_s, r_e, c_e = self.tfle.blocks[block_idx]

        # extract ternary weights
        w_ternary, alpha = self.tfle.get_ternary_weights(module)
        w_block = w_ternary[r_s:r_e, c_s:c_e].clone()

        # current fitness
        eval_qs = random.sample(questions, min(self.config.questions_per_step, len(questions)))
        current_fitness = self.fitness_fn(eval_qs)

        # generate proposals
        proposals = self.tfle._propose_block_flips(w_block, self.config.K)

        # evaluate proposals
        original_weight = module.weight.data.clone()
        best_k = -1
        best_fitness = current_fitness

        for k in range(self.config.K):
            w_copy = w_ternary.clone()
            w_copy[r_s:r_e, c_s:c_e] = proposals[k].float()
            self.tfle.set_ternary_weights(module, w_copy, alpha)

            fitness_k = self.fitness_fn(eval_qs)
            if fitness_k > best_fitness:
                best_fitness = fitness_k
                best_k = k

            module.weight.data.copy_(original_weight)

        # accept with re-eval on fresh questions
        accepted = False
        if best_k >= 0:
            w_best = w_ternary.clone()
            w_best[r_s:r_e, c_s:c_e] = proposals[best_k].float()
            self.tfle.set_ternary_weights(module, w_best, alpha)

            fresh_qs = random.sample(questions, min(self.config.re_eval_questions, len(questions)))
            fresh_fitness = self.fitness_fn(fresh_qs)

            if fresh_fitness >= current_fitness - self.config.re_eval_tolerance:
                accepted = True
                self.tfle.total_accepted += 1
            else:
                module.weight.data.copy_(original_weight)
                self.tfle.total_rejected += 1
        else:
            self.tfle.total_rejected += 1

        delta = best_fitness - current_fitness if accepted else 0.0
        return {
            "step": self.step_count,
            "accepted": accepted,
            "block_idx": block_idx,
            "block_name": name,
            "fitness_before": current_fitness,
            "fitness_after": best_fitness if accepted else current_fitness,
            "delta": delta,
            "acceptance_rate": self.tfle.acceptance_rate(),
        }

    def train(self, questions: list[dict]) -> dict:
        """Run full TFLE-GRPO training loop."""
        results = {"steps": []}

        for step in range(self.config.num_steps):
            metrics = self.step(questions)

            if step % self.config.eval_every == 0:
                logger.info(
                    f"TFLE-GRPO Step {step}/{self.config.num_steps} | "
                    f"Fit {metrics['fitness_after']:.4f} | "
                    f"AR {metrics['acceptance_rate']:.0%} | "
                    f"Block {metrics['block_name']}"
                )
                results["steps"].append(metrics)

        results["final_fitness"] = metrics["fitness_after"]
        results["acceptance_rate"] = self.tfle.acceptance_rate()
        return results
