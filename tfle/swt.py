"""SWT: Sleep-Wake Training for continual learning.

Wake Phase: Fast TFLE updates from streaming data + store experience
Sleep Phase: Consolidation via replay, EWC, micro-critics, adversarial rounds
"""

from __future__ import annotations

from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TFLEConfig


class ReplayBuffer:
    """FOREVER-style replay buffer with surprise-priority sampling."""

    def __init__(self, max_size: int = 10000, device: torch.device | None = None):
        self.max_size = max_size
        self.device = device or torch.device("cpu")
        self.buffer: list[tuple[torch.Tensor, torch.Tensor, float, float]] = []
        self.priorities: list[float] = []

    def add(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        surprise: float,
        reward: float = 0.0,
    ):
        """Store an experience with surprise priority."""
        entry = (x.detach().cpu(), labels.detach().cpu(), surprise, reward)
        if len(self.buffer) >= self.max_size:
            min_idx = min(range(len(self.priorities)), key=lambda i: self.priorities[i])
            if surprise > self.priorities[min_idx]:
                self.buffer[min_idx] = entry
                self.priorities[min_idx] = surprise
        else:
            self.buffer.append(entry)
            self.priorities.append(surprise)

    def sample(self, batch_size: int = 64) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Sample weighted by surprise priority."""
        if len(self.buffer) < batch_size:
            return None

        probs = torch.tensor(self.priorities, dtype=torch.float32)
        probs = probs / (probs.sum() + 1e-10)
        indices = torch.multinomial(probs, min(batch_size, len(self.buffer)), replacement=False)

        x = torch.stack([self.buffer[i][0] for i in indices]).to(self.device)
        labels = torch.stack([self.buffer[i][1] for i in indices]).to(self.device)
        return x, labels

    def sample_hardest(self, batch_size: int = 32) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Sample the highest-surprise experiences."""
        if len(self.buffer) < batch_size:
            return None

        probs = torch.tensor(self.priorities, dtype=torch.float32)
        _, indices = probs.topk(min(batch_size, len(self.buffer)))

        x = torch.stack([self.buffer[i][0] for i in indices]).to(self.device)
        labels = torch.stack([self.buffer[i][1] for i in indices]).to(self.device)
        return x, labels

    def __len__(self):
        return len(self.buffer)

    def avg_surprise(self) -> float:
        if not self.priorities:
            return 0.0
        return sum(self.priorities) / len(self.priorities)


class MicroCritic(nn.Module):
    """Tiny adversarial network per layer for sleep consolidation.

    Learns to distinguish good representations (structured) from noise.
    Architecture: ~100-1000 parameters.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, device: torch.device | None = None):
        super().__init__()
        dev = device or torch.device("cpu")
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        ).to(dev)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        self.quality_threshold = 0.5
        self.device = dev

    def evaluate(self, activations: torch.Tensor) -> float:
        """Score representation quality (0=bad, 1=good)."""
        self.eval()
        with torch.no_grad():
            return self.net(activations.detach()).mean().item()

    def train_step(self, activations: torch.Tensor) -> float:
        """Train: real activations = positive, noise = negative."""
        self.train()
        noise = torch.randn_like(activations)

        real_scores = self.net(activations.detach())
        fake_scores = self.net(noise)

        loss = -torch.log(real_scores + 1e-8).mean() - torch.log(1 - fake_scores + 1e-8).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


class SleepWakeTrainer:
    """Orchestrates sleep-wake training cycles.

    Wake: fast TFLE update + store experience
    Sleep: replay + EWC + micro-critics + adversarial rounds
    """

    def __init__(self, model, config: TFLEConfig, device: torch.device | None = None):
        self.model = model
        self.config = config
        self.device = device or torch.device("cpu")

        self.replay_buffer = ReplayBuffer(
            max_size=config.swt_replay_buffer_size,
            device=self.device,
        )

        # EWC: Fisher information per layer
        self.fisher_info: dict[int, torch.Tensor] = {}
        self.optimal_weights: dict[int, torch.Tensor] = {}

        # Micro-critics: one per layer
        self.micro_critics = []
        for layer in model.layers:
            self.micro_critics.append(
                MicroCritic(
                    input_dim=layer.out_features,
                    hidden_dim=min(64, layer.out_features),
                    device=self.device,
                )
            )

        self.task_counter = 0

    def wake_step(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        temperature: float,
        task_reward: float = 0.0,
    ) -> list[dict]:
        """Wake phase: TFLE update + store experience.

        Returns training metrics from the TFLE step.
        """
        metrics = self.model.train_step(x, temperature, labels)

        # Compute surprise
        with torch.no_grad():
            eval_result = self.model.evaluate(x, labels)
            surprise = eval_result["loss"]

        self.replay_buffer.add(x, labels, surprise, task_reward)

        self.task_counter += 1
        if self.task_counter >= self.config.swt_frequency_tasks:
            self.sleep_phase(temperature)
            self.task_counter = 0

        return metrics

    def should_sleep(self) -> bool:
        return self.task_counter >= self.config.swt_frequency_tasks

    def sleep_phase(self, temperature: float = 0.01) -> dict:
        """Sleep phase: consolidation via replay + EWC + micro-critics.

        No new data — only replay buffer.
        """
        if len(self.replay_buffer) < 32:
            return {"replayed": 0, "skipped": True}

        # 1. Compute Fisher information
        self._compute_fisher()

        replayed = 0
        for _ in range(self.config.swt_consolidation_steps):
            batch = self.replay_buffer.sample(batch_size=64)
            if batch is None:
                break

            x, labels = batch

            # TFLE consolidation step with EWC penalty awareness
            # We run a normal train_step — the EWC penalty is applied
            # via modified acceptance (reject flips that hurt important weights)
            for layer_idx, layer in enumerate(self.model.layers):
                layer_temp = self.config.get_temperature_for_layer(temperature, layer_idx)
                combined_traces = layer._get_combined_traces()
                candidates = layer._select_candidates(combined_traces)
                proposed = layer._propose_flips(candidates)

                # Evaluate with EWC penalty
                old_weights = layer.weights.clone()

                # Loss before
                with torch.no_grad():
                    logits_before = self.model.forward(x)
                    loss_before = F.cross_entropy(logits_before, labels).item()

                # Loss after
                layer.weights = proposed.to(torch.int8)
                with torch.no_grad():
                    logits_after = self.model.forward(x)
                    loss_after = F.cross_entropy(logits_after, labels).item()
                layer.weights = old_weights

                # EWC penalty
                ewc_penalty = self._ewc_penalty(layer_idx, proposed)
                adjusted_loss_after = loss_after + ewc_penalty

                delta = loss_before - adjusted_loss_after
                accepted = layer._accept_or_reject(delta, layer_temp)

                if accepted:
                    layer.weights = proposed.to(torch.int8)

            # Train micro-critics
            with torch.no_grad():
                h = x
                for i, layer in enumerate(self.model.layers):
                    h = layer.forward(h)
                    self.micro_critics[i].train_step(h)
                    if i < len(self.model.layers) - 1:
                        h = F.relu(h)

            replayed += 1

        # Adversarial rounds: train on hardest examples
        adversarial_done = 0
        for _ in range(self.config.swt_adversarial_rounds):
            hard = self.replay_buffer.sample_hardest(batch_size=32)
            if hard is None:
                break
            hx, hlabels = hard
            self.model.train_step(hx, temperature * 0.1, hlabels)
            adversarial_done += 1

        # Update optimal weights for EWC
        for i, layer in enumerate(self.model.layers):
            self.optimal_weights[i] = layer.weights.clone()

        self.task_counter = 0

        return {
            "replayed": replayed,
            "adversarial_rounds": adversarial_done,
            "buffer_size": len(self.replay_buffer),
            "avg_surprise": self.replay_buffer.avg_surprise(),
        }

    def _compute_fisher(self):
        """Compute diagonal Fisher information via flip sensitivity."""
        for layer_idx, layer in enumerate(self.model.layers):
            fisher = torch.zeros_like(layer.weights, dtype=torch.float32, device=self.device)

            # Sample batches from replay
            for _ in range(min(10, len(self.replay_buffer))):
                batch = self.replay_buffer.sample(batch_size=32)
                if batch is None:
                    break
                x, labels = batch

                with torch.no_grad():
                    base_loss = F.cross_entropy(self.model.forward(x), labels).item()

                # Sample weights and measure sensitivity
                flat_w = layer.weights.flatten()
                n_sample = min(500, flat_w.numel())
                indices = torch.randperm(flat_w.numel(), device=self.device)[:n_sample]

                for idx in indices:
                    old_val = flat_w[idx].item()
                    new_val = int((old_val + 2) % 3 - 1)
                    flat_w[idx] = new_val
                    with torch.no_grad():
                        new_loss = F.cross_entropy(self.model.forward(x), labels).item()
                    fisher.flatten()[idx] += (new_loss - base_loss) ** 2
                    flat_w[idx] = old_val

            self.fisher_info[layer_idx] = fisher
            self.optimal_weights[layer_idx] = layer.weights.clone()

    def _ewc_penalty(self, layer_idx: int, proposed_weights: torch.Tensor) -> float:
        """Compute EWC penalty for proposed weights."""
        if layer_idx not in self.fisher_info:
            return 0.0

        fisher = self.fisher_info[layer_idx]
        optimal = self.optimal_weights[layer_idx]
        diff = (proposed_weights.float() - optimal.float()) ** 2
        penalty = (fisher * diff).sum().item() * self.config.swt_ewc_lambda
        return penalty
