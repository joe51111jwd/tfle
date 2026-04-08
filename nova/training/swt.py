"""Sleep-Wake Training for NOVA 2.4B continual learning.

Wake phase: block-local TFLE on new data
Sleep phase: consolidation with EWC (per-block Fisher matching block-local TFLE),
  replay buffer with surprise priority, CDLL-augmented fitness.

EWC prevents catastrophic forgetting by penalizing weight changes in proportion
to their Fisher Information — blocks that matter most for old tasks are protected.
"""
from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block_local_tfle import BlockLocalTFLE, BlockLocalConfig
from .self_dpo import SelfDPO
from .preference_store import PreferenceStore

logger = logging.getLogger(__name__)


@dataclass
class SWTConfig:
    block_size: int = 64
    K: int = 512
    flip_rate: float = 0.05
    re_eval_tolerance: float = 0.005
    fisher_refresh_every: int = 1000
    ewc_lambda: float = 5000.0
    sleep_every: int = 100
    sleep_steps: int = 50
    replay_buffer_size: int = 2048
    replay_priority: str = "surprise"
    embed_lr: float = 1e-5


# ── Replay Buffer ─────────────────────────────────────────────

class ReplayBuffer:
    """Experience replay with surprise-priority sampling.

    Stores (input, target, surprise_score) tuples. High-surprise experiences
    are sampled more frequently during sleep consolidation.
    """

    def __init__(self, max_size: int = 2048, device: torch.device | None = None):
        self.max_size = max_size
        self.device = device or torch.device("cpu")
        self.buffer: list[tuple[torch.Tensor, torch.Tensor, float]] = []

    def add(self, x: torch.Tensor, y: torch.Tensor, surprise: float):
        entry = (x.detach().cpu(), y.detach().cpu(), surprise)
        if len(self.buffer) >= self.max_size:
            # evict lowest-surprise entry
            min_idx = min(range(len(self.buffer)), key=lambda i: self.buffer[i][2])
            if surprise > self.buffer[min_idx][2]:
                self.buffer[min_idx] = entry
        else:
            self.buffer.append(entry)

    def sample(self, batch_size: int = 32) -> tuple[torch.Tensor, torch.Tensor] | None:
        if len(self.buffer) < batch_size:
            return None

        priorities = torch.tensor([e[2] for e in self.buffer], dtype=torch.float32)
        priorities = priorities.clamp(min=1e-10)
        probs = priorities / priorities.sum()
        indices = torch.multinomial(probs, min(batch_size, len(self.buffer)), replacement=False)

        x = torch.stack([self.buffer[i][0] for i in indices]).to(self.device)
        y = torch.stack([self.buffer[i][1] for i in indices]).to(self.device)
        return x, y

    def evict_oldest(self, fraction: float = 0.3):
        n_evict = int(len(self.buffer) * fraction)
        if n_evict > 0:
            self.buffer = self.buffer[n_evict:]

    def __len__(self) -> int:
        return len(self.buffer)


# ── EWC with per-block Fisher ─────────────────────────────────

class BlockEWC:
    """Elastic Weight Consolidation with per-block Fisher matching block-local TFLE.

    Fisher is computed per block (same decomposition as BlockLocalTFLE), so the
    regularization naturally aligns with the search structure.
    """

    def __init__(
        self,
        model: nn.Module,
        blocks: list[tuple[str, nn.Module, int, int, int, int]],
        ewc_lambda: float = 5000.0,
        device: torch.device | None = None,
    ):
        self.model = model
        self.blocks = blocks
        self.ewc_lambda = ewc_lambda
        self.device = device or torch.device("cpu")

        # stored reference weights and Fisher per block
        self.reference_weights: list[torch.Tensor] = []
        self.fisher_per_block: list[torch.Tensor] = []
        self._initialized = False

    def snapshot(self, dataloader, vocab_size: int, n_batches: int = 20):
        """Compute Fisher and store reference weights for EWC penalty."""
        self.reference_weights = []
        self.fisher_per_block = []

        # accumulate gradients for Fisher
        fisher_accum: dict[str, torch.Tensor] = {}
        for name, module, *_ in self.blocks:
            if name not in fisher_accum:
                fisher_accum[name] = torch.zeros_like(module.weight)

        self.model.train()
        n = 0
        for x, y in dataloader:
            if n >= n_batches:
                break
            x, y = x.to(self.device), y.to(self.device)
            self.model.zero_grad()
            logits = self.model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                y.reshape(-1),
                ignore_index=0,
            )
            loss.backward()

            for name, module, *_ in self.blocks:
                if module.weight.grad is not None:
                    fisher_accum[name] += module.weight.grad.data ** 2
            n += 1

        # normalize and extract per-block Fisher + reference weights
        for name, module, r_s, c_s, r_e, c_e in self.blocks:
            block_fisher = fisher_accum[name][r_s:r_e, c_s:c_e] / max(n, 1)
            self.fisher_per_block.append(block_fisher.detach().clone())

            with torch.no_grad():
                alpha = module.weight.abs().mean().clamp(min=1e-10)
                w_t = torch.clamp(torch.round(module.weight / alpha), -1, 1)
                self.reference_weights.append(w_t[r_s:r_e, c_s:c_e].detach().clone())

        self._initialized = True
        logger.info(f"EWC snapshot: {len(self.blocks)} blocks, {n} batches")

    def penalty(self, block_idx: int, current_block: torch.Tensor) -> float:
        """Compute EWC penalty for a single block.

        penalty = lambda/2 * sum(Fisher * (w - w_ref)^2)
        """
        if not self._initialized:
            return 0.0

        ref = self.reference_weights[block_idx].to(current_block.device)
        fisher = self.fisher_per_block[block_idx].to(current_block.device)
        diff_sq = (current_block.float() - ref.float()) ** 2
        return (self.ewc_lambda / 2.0) * (fisher * diff_sq).sum().item()


# ── Sleep-Wake Trainer ────────────────────────────────────────

class SleepWakeTrainer:
    """Sleep-Wake Training with block-local TFLE and EWC consolidation.

    Wake: block-local TFLE on streaming new tasks, fill replay buffer.
    Sleep (every N tasks): consolidation on replay buffer with EWC penalty
      and CDLL-augmented fitness.
    """

    def __init__(
        self,
        model: nn.Module,
        config: SWTConfig,
        device: torch.device,
        preference_store: PreferenceStore | None = None,
    ):
        self.model = model
        self.config = config
        self.device = device

        # block-local TFLE
        bl_config = BlockLocalConfig(
            block_size=config.block_size,
            K=config.K,
            flip_rate=config.flip_rate,
            re_eval_tolerance=config.re_eval_tolerance,
            fisher_refresh_every=config.fisher_refresh_every,
            embed_lr=config.embed_lr,
        )
        self.tfle = BlockLocalTFLE(model, bl_config, device)

        # replay buffer
        self.replay = ReplayBuffer(
            max_size=config.replay_buffer_size,
            device=device,
        )

        # EWC
        self.ewc = BlockEWC(
            model, self.tfle.blocks,
            ewc_lambda=config.ewc_lambda,
            device=device,
        )

        # Self-DPO (uses preference pairs from inference)
        self.preference_store = preference_store
        self.self_dpo = SelfDPO(
            model=model,
            preference_store=preference_store or PreferenceStore(),
            device=device,
        ) if preference_store is not None else None

        self.task_count = 0
        self.wake_steps = 0
        self.sleep_steps = 0

    def is_sleep_time(self) -> bool:
        return self.task_count > 0 and self.task_count % self.config.sleep_every == 0

    def wake_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_val: torch.Tensor,
        y_val: torch.Tensor,
        vocab_size: int,
    ) -> dict:
        """One wake-phase step: block-local TFLE on new data.

        Also stores experience in replay buffer with surprise score.
        """
        # compute surprise before update
        with torch.no_grad():
            self.model.eval()
            logits = self.model(x.to(self.device))
            surprise = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                y.to(self.device).reshape(-1),
                ignore_index=0,
            ).item()

        self.replay.add(x, y, surprise)

        # block-local TFLE step
        metrics = self.tfle.step(x, y, x_val, y_val, vocab_size)
        self.wake_steps += 1

        metrics["phase"] = "wake"
        metrics["surprise"] = surprise
        metrics["replay_size"] = len(self.replay)
        return metrics

    def _spot_check(self, vocab_size: int) -> float:
        """Run spot-check on replay buffer samples to measure accuracy.

        Uses 15 replay samples (proxy for 10 math + 5 code) to estimate
        whether the model degraded. Returns the average fitness score.
        """
        self.model.eval()
        n_checks = 15
        total_fitness = 0.0
        checked = 0

        for _ in range(n_checks):
            batch = self.replay.sample(batch_size=1)
            if batch is None:
                break
            x, y = batch
            fitness = self.tfle._evaluate_fitness(x, y, vocab_size)
            total_fitness += fitness
            checked += 1

        return total_fitness / max(checked, 1)

    def sleep_phase(
        self,
        vocab_size: int,
        dataloader=None,
    ) -> list[dict]:
        """Sleep phase: three-step consolidation.

        1. Memory consolidation (EWC + replay buffer)
        2. Self-DPO training (if sufficient preference pairs)
        3. TFLE refinement on replay buffer with EWC penalty

        After Self-DPO, a spot-check runs: if accuracy drops >5%,
        EWC lambda is increased to protect existing knowledge.
        """
        metrics_list = []

        # ── Step 1: Memory consolidation (EWC snapshot) ──
        if dataloader is not None:
            self.ewc.snapshot(dataloader, vocab_size)
        logger.info("Sleep step 1: EWC snapshot complete")

        # measure pre-DPO fitness for spot-check comparison
        pre_dpo_fitness = self._spot_check(vocab_size)

        # ── Step 2: Self-DPO training ──
        dpo_result = None
        if self.self_dpo is not None and self.self_dpo.should_train():
            logger.info("Sleep step 2: Running Self-DPO training")
            dpo_result = self.self_dpo.train(
                num_epochs=1, batch_size=4, lr=5e-6,
            )
            metrics_list.append({
                "phase": "sleep_dpo",
                "dpo_result": dpo_result,
            })

            # spot-check: did DPO hurt accuracy?
            post_dpo_fitness = self._spot_check(vocab_size)
            fitness_drop = pre_dpo_fitness - post_dpo_fitness
            relative_drop = abs(fitness_drop / min(pre_dpo_fitness, -1e-8))

            if relative_drop > 0.05:
                old_lambda = self.ewc.ewc_lambda
                self.ewc.ewc_lambda *= 1.5
                logger.warning(
                    f"Spot-check: fitness dropped {relative_drop:.1%} after DPO. "
                    f"Increasing EWC lambda {old_lambda:.0f} -> {self.ewc.ewc_lambda:.0f}"
                )
                # re-snapshot with stronger lambda
                if dataloader is not None:
                    self.ewc.snapshot(dataloader, vocab_size)
            else:
                logger.info(
                    f"Spot-check passed: fitness change {fitness_drop:.4f} "
                    f"({relative_drop:.1%})"
                )
        else:
            logger.info("Sleep step 2: Self-DPO skipped (insufficient pairs)")

        # ── Step 3: TFLE refinement on replay buffer with EWC ──
        logger.info("Sleep step 3: TFLE refinement with EWC penalty")
        for step in range(self.config.sleep_steps):
            batch = self.replay.sample(batch_size=32)
            if batch is None:
                break

            x, y = batch
            x, y = x.to(self.device), y.to(self.device)

            # select block
            block_idx = self.tfle._select_block()
            name, module, r_s, c_s, r_e, c_e = self.tfle.blocks[block_idx]

            w_ternary, alpha = self.tfle.get_ternary_weights(module)
            w_block = w_ternary[r_s:r_e, c_s:c_e].clone()

            # current fitness = task fitness - EWC penalty (CDLL-augmented)
            current_task_fitness = self.tfle._evaluate_fitness(x, y, vocab_size)
            current_ewc_penalty = self.ewc.penalty(block_idx, w_block)
            current_fitness = current_task_fitness - current_ewc_penalty

            # generate proposals
            proposals = self.tfle._propose_block_flips(w_block, self.config.K)
            original_weight = module.weight.data.clone()

            best_k = -1
            best_fitness = current_fitness

            for k in range(self.config.K):
                w_copy = w_ternary.clone()
                w_copy[r_s:r_e, c_s:c_e] = proposals[k].float()
                self.tfle.set_ternary_weights(module, w_copy, alpha)

                task_f = self.tfle._evaluate_fitness(x, y, vocab_size)
                ewc_p = self.ewc.penalty(block_idx, proposals[k].float())
                f_k = task_f - ewc_p

                if f_k > best_fitness:
                    best_fitness = f_k
                    best_k = k

                module.weight.data.copy_(original_weight)

            accepted = False
            if best_k >= 0:
                w_best = w_ternary.clone()
                w_best[r_s:r_e, c_s:c_e] = proposals[best_k].float()
                self.tfle.set_ternary_weights(module, w_best, alpha)
                accepted = True
                self.tfle.total_accepted += 1
            else:
                self.tfle.total_rejected += 1

            self.sleep_steps += 1
            metrics_list.append({
                "phase": "sleep_tfle",
                "step": step,
                "accepted": accepted,
                "block_idx": block_idx,
                "fitness_before": current_fitness,
                "fitness_after": best_fitness if accepted else current_fitness,
                "ewc_penalty": current_ewc_penalty,
            })

        # evict old experiences after sleep
        self.replay.evict_oldest(0.3)
        return metrics_list

    def step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_val: torch.Tensor,
        y_val: torch.Tensor,
        vocab_size: int,
        dataloader=None,
    ) -> dict | list[dict]:
        """One full step: wake or sleep depending on task count."""
        self.task_count += 1

        if self.is_sleep_time():
            logger.info(f"Entering sleep phase (task {self.task_count})")
            sleep_metrics = self.sleep_phase(vocab_size, dataloader)
            return sleep_metrics

        return self.wake_step(x, y, x_val, y_val, vocab_size)

    def train(
        self,
        dataloader,
        val_dataloader,
        vocab_size: int,
        num_steps: int = 1000,
        log_every: int = 50,
    ) -> dict:
        """Run full sleep-wake training loop."""
        results = {"wake_steps": [], "sleep_phases": []}
        train_iter = iter(dataloader)
        val_iter = iter(val_dataloader)

        for step in range(num_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(dataloader)
                x, y = next(train_iter)

            try:
                x_val, y_val = next(val_iter)
            except StopIteration:
                val_iter = iter(val_dataloader)
                x_val, y_val = next(val_iter)

            outcome = self.step(x, y, x_val, y_val, vocab_size, dataloader)

            if isinstance(outcome, list):
                results["sleep_phases"].append({
                    "task": self.task_count,
                    "steps": outcome,
                })
                if step % log_every == 0:
                    tfle_steps = [m for m in outcome if m.get("phase") == "sleep_tfle"]
                    n_accepted = sum(1 for m in tfle_steps if m.get("accepted"))
                    logger.info(
                        f"Sleep phase at task {self.task_count}: "
                        f"{n_accepted}/{len(tfle_steps)} TFLE accepted"
                    )
            else:
                if step % log_every == 0:
                    results["wake_steps"].append({"step": step, **outcome})
                    logger.info(
                        f"Wake step {step}/{num_steps} | "
                        f"Fit {outcome['fitness_after']:.4f} | "
                        f"AR {self.tfle.acceptance_rate():.0%} | "
                        f"Replay {len(self.replay)}"
                    )

        results["total_accepted"] = self.tfle.total_accepted
        results["total_rejected"] = self.tfle.total_rejected
        results["acceptance_rate"] = self.tfle.acceptance_rate()
        return results
