"""Dr. GRPO trainer for NOVA 2.4B.

Group Relative Policy Optimization with:
- Group size 16 responses per prompt
- Frozen reference model for KL divergence
- Batch generation (all G responses at once)
- Group-relative advantage normalization
- Clipped policy gradient with asymmetric ratio bounds
- Real math dataset support (GSM8K, MATH)
"""
from __future__ import annotations

import copy
import json
import logging
import math
import random
import re
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rewards import MathReward, GranularFormatReward, combined_reward

logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig:
    group_size: int = 16
    kl_coeff: float = 0.001
    clip_ratio: float = 10.0
    lr: float = 3e-6
    weight_decay: float = 0.01
    num_steps: int = 200
    batch_size: int = 4
    temperature: float = 1.0
    max_response_len: int = 200
    max_seq_len: int = 4096
    eval_every: int = 25
    grad_clip: float = 1.0


# ── Dataset loaders ───────────────────────────────────────────

def load_gsm8k(max_samples: int = 5000) -> list[dict]:
    """Load GSM8K math word problems from HuggingFace."""
    try:
        from datasets import load_dataset
        ds = load_dataset("gsm8k", "main", split="train")
        questions = []
        for ex in ds:
            if len(questions) >= max_samples:
                break
            answer_text = ex.get("answer", "")
            # GSM8K answer is after "####"
            m = re.search(r"####\s*(-?\d+\.?\d*)", answer_text)
            if m:
                questions.append({
                    "question": ex["question"],
                    "answer": m.group(1),
                    "source": "gsm8k",
                })
        logger.info(f"Loaded {len(questions)} GSM8K problems")
        return questions
    except Exception as e:
        logger.warning(f"Could not load GSM8K: {e}")
        return []


def load_math_dataset(max_samples: int = 5000) -> list[dict]:
    """Load MATH competition problems from HuggingFace."""
    try:
        from datasets import load_dataset
        ds = load_dataset("hendrycks/competition_math", split="train")
        questions = []
        for ex in ds:
            if len(questions) >= max_samples:
                break
            answer = ex.get("answer", "")
            if answer:
                questions.append({
                    "question": ex["problem"],
                    "answer": answer,
                    "source": "math",
                    "level": ex.get("level", ""),
                })
        logger.info(f"Loaded {len(questions)} MATH problems")
        return questions
    except Exception as e:
        logger.warning(f"Could not load MATH dataset: {e}")
        return []


def generate_synthetic_questions(n: int = 500) -> list[dict]:
    """Fallback: generate synthetic math questions."""
    random.seed(123)
    questions = []
    for _ in range(n):
        difficulty = random.choice(["easy", "medium", "hard"])
        if difficulty == "easy":
            a, b = random.randint(1, 100), random.randint(1, 100)
            op = random.choice(["+", "-", "*"])
            answer = eval(f"{a} {op} {b}")
            q = f"What is {a} {op} {b}?"
        elif difficulty == "medium":
            a, b, c = random.randint(1, 50), random.randint(1, 50), random.randint(2, 20)
            answer = (a + b) * c
            q = f"Add {a} and {b}, then multiply by {c}. What is the result?"
        else:
            items, price = random.randint(2, 20), random.randint(1, 10)
            answer = items * price
            q = f"You buy {items} items at ${price} each. What is the total cost?"
        questions.append({"question": q, "answer": answer, "source": "synthetic"})
    return questions


def load_grpo_questions(max_samples: int = 5000) -> list[dict]:
    """Load real math datasets, falling back to synthetic if unavailable."""
    questions = load_gsm8k(max_samples // 2)
    questions.extend(load_math_dataset(max_samples // 2))
    if not questions:
        questions = generate_synthetic_questions(max_samples)
    random.shuffle(questions)
    return questions[:max_samples]


# ── GRPO Trainer ──────────────────────────────────────────────

class GRPOTrainer:
    """Dr. GRPO: Group Relative Policy Optimization.

    For each prompt, generates G responses, computes group-relative advantages,
    and updates policy with clipped surrogate objective + KL penalty.
    """

    def __init__(
        self,
        model: nn.Module,
        config: GRPOConfig,
        device: torch.device,
        tokenizer=None,
    ):
        self.model = model
        self.config = config
        self.device = device
        self.tokenizer = tokenizer

        # Frozen reference model
        self.ref_model = copy.deepcopy(model)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        self.math_reward = MathReward()
        self.format_reward = GranularFormatReward()

    def _encode(self, text: str) -> list[int]:
        if self.tokenizer is not None:
            return self.tokenizer.encode(text)
        return [ord(c) % 32000 for c in text.split()]

    def _decode(self, ids: list[int]) -> str:
        if self.tokenizer is not None:
            return self.tokenizer.decode(ids)
        return " ".join(str(i) for i in ids)

    @torch.no_grad()
    def batch_generate(
        self,
        prompt_ids_list: list[torch.Tensor],
        G: int | None = None,
        max_new: int | None = None,
        temperature: float | None = None,
    ) -> tuple[list[torch.Tensor], list[int]]:
        """Generate G responses for each prompt, all batched together.

        Returns (list of [G, seq_len] tensors, list of prompt lengths).
        """
        G = G or self.config.group_size
        max_new = max_new or self.config.max_response_len
        temperature = temperature or self.config.temperature

        self.model.eval()
        n_prompts = len(prompt_ids_list)
        max_prompt = max(p.shape[1] for p in prompt_ids_list)
        total_seqs = n_prompts * G

        # right-align prompts, duplicate G times each
        batch = torch.zeros(total_seqs, max_prompt, dtype=torch.long, device=self.device)
        prompt_lens = []
        for i, p in enumerate(prompt_ids_list):
            plen = p.shape[1]
            for g in range(G):
                idx = i * G + g
                batch[idx, max_prompt - plen:] = p[0]
            prompt_lens.append(plen)

        ids = batch.clone()
        for _ in range(max_new):
            if ids.shape[1] >= self.config.max_seq_len:
                break
            use_amp = self.device.type == "cuda"
            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = self.model(ids)
            else:
                logits = self.model(ids)

            nxt = logits[:, -1, :].float() / max(temperature, 1e-8)
            probs = F.softmax(nxt, dim=-1)
            tok = torch.multinomial(probs, 1)
            ids = torch.cat([ids, tok], dim=1)

        results = []
        for i in range(n_prompts):
            group = ids[i * G: (i + 1) * G]
            results.append(group)

        return results, prompt_lens

    def compute_log_probs(
        self,
        model: nn.Module,
        sequences: torch.Tensor,
        prompt_len: int,
    ) -> torch.Tensor:
        """Compute sum of log-probs for generated tokens (after prompt)."""
        use_amp = self.device.type == "cuda"
        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(sequences)
        else:
            logits = model(sequences)

        logits = logits.float()
        gen_logits = logits[:, prompt_len - 1:-1, :]
        gen_targets = sequences[:, prompt_len:]
        log_p = F.log_softmax(gen_logits, dim=-1)
        token_lp = log_p.gather(-1, gen_targets.unsqueeze(-1)).squeeze(-1)
        mask = (gen_targets != 0).float()
        return (token_lp * mask).sum(dim=-1)

    def train_step(self, batch_questions: list[dict]) -> dict:
        """One GRPO step on a batch of questions."""
        self.model.train()
        G = self.config.group_size

        # tokenize prompts
        prompt_ids_list = []
        for qdata in batch_questions:
            ids = self._encode(qdata["question"])
            prompt_ids_list.append(torch.tensor([ids], dtype=torch.long, device=self.device))

        # generate all responses
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        all_seqs, prompt_lens = self.batch_generate(prompt_ids_list)

        total_policy_loss = torch.tensor(0.0, device=self.device)
        total_kl = torch.tensor(0.0, device=self.device)
        all_rewards = []
        n_correct = 0

        for i, qdata in enumerate(batch_questions):
            sequences = all_seqs[i]
            plen = prompt_lens[i]
            gt = qdata["answer"]

            # score each response
            rewards = []
            for g in range(G):
                resp_tokens = sequences[g, sequences.shape[1] - self.config.max_response_len:].tolist()
                resp = self._decode([t for t in resp_tokens if t != 0])
                r_math = self.math_reward.score(qdata["question"], resp, gt)
                r_fmt = self.format_reward.score(resp)
                rewards.append(r_math + r_fmt)
                if r_math >= 0.9:
                    n_correct += 1
            all_rewards.extend(rewards)

            rewards_t = torch.tensor(rewards, device=self.device, dtype=torch.float32)

            # group-relative advantages
            if rewards_t.std() > 1e-8:
                advantages = (rewards_t - rewards_t.mean()) / rewards_t.std()
            else:
                advantages = torch.zeros_like(rewards_t)

            # log-probs
            max_prompt = sequences.shape[1] - self.config.max_response_len
            actual_start = max(max_prompt, plen)

            policy_lp = self.compute_log_probs(self.model, sequences, actual_start)
            with torch.no_grad():
                ref_lp = self.compute_log_probs(self.ref_model, sequences, actual_start)

            # clipped policy gradient
            ratio = torch.exp(policy_lp - ref_lp.detach())
            clip_lo = 1.0 / self.config.clip_ratio
            clip_hi = self.config.clip_ratio
            clipped = torch.clamp(ratio, clip_lo, clip_hi)
            policy_loss = -torch.min(
                ratio * advantages.detach(),
                clipped * advantages.detach(),
            ).mean()

            kl = (ref_lp.detach() - policy_lp).mean()

            total_policy_loss = total_policy_loss + policy_loss
            total_kl = total_kl + kl

        # average over batch
        n_batch = max(len(batch_questions), 1)
        loss = total_policy_loss / n_batch + self.config.kl_coeff * (total_kl / n_batch)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()

        total_responses = len(batch_questions) * G
        return {
            "loss": loss.item(),
            "policy_loss": (total_policy_loss / n_batch).item(),
            "kl": (total_kl / n_batch).item(),
            "mean_reward": sum(all_rewards) / max(len(all_rewards), 1),
            "accuracy": n_correct / max(total_responses, 1),
        }

    def train(self, questions: list[dict] | None = None) -> dict:
        """Run full GRPO training loop."""
        if questions is None:
            questions = load_grpo_questions()

        results = {"config": vars(self.config), "steps": []}

        for step in range(self.config.num_steps):
            batch_qs = random.sample(questions, min(self.config.batch_size, len(questions)))
            metrics = self.train_step(batch_qs)

            if step % self.config.eval_every == 0:
                logger.info(
                    f"Step {step}/{self.config.num_steps} | "
                    f"R {metrics['mean_reward']:.3f} | "
                    f"Acc {metrics['accuracy']:.0%} | "
                    f"PL {metrics['policy_loss']:.4f} | "
                    f"KL {metrics['kl']:.4f}"
                )
                results["steps"].append({"step": step, **metrics})

        results["final_reward"] = metrics["mean_reward"]
        return results
