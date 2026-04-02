"""SWT: Sleep-Wake Training for continual learning.

Wake Phase (900 steps): TFLE + CDLL on streaming data, fill replay buffer
Sleep Phase (100 steps): TFLE + micro-critic on replay buffer, consolidate

Two phases, two fitness signals:
- Wake: CDLL (information-theoretic compression of new data)
- Sleep: Micro-critic (representation quality against replay buffer)
"""

from __future__ import annotations

from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TFLEConfig


class ReplayBuffer:
    """FOREVER-style replay buffer with surprise-priority sampling.

    Stores (activations, layer_idx) snapshots for sleep consolidation.
    Evicts oldest fraction after each sleep phase.
    """

    def __init__(self, max_size: int = 1024, device: torch.device | None = None):
        self.max_size = max_size
        self.device = device or torch.device("cpu")
        self.buffer: list[tuple[torch.Tensor, torch.Tensor, float]] = []

    def add(self, x: torch.Tensor, labels: torch.Tensor, surprise: float = 1.0):
        """Store an experience."""
        entry = (x.detach().cpu(), labels.detach().cpu(), surprise)
        if len(self.buffer) >= self.max_size:
            min_idx = min(range(len(self.buffer)), key=lambda i: self.buffer[i][2])
            if surprise > self.buffer[min_idx][2]:
                self.buffer[min_idx] = entry
        else:
            self.buffer.append(entry)

    def sample(self, batch_size: int = 64) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Sample weighted by surprise."""
        if len(self.buffer) < batch_size:
            return None
        priorities = torch.tensor([e[2] for e in self.buffer], dtype=torch.float32)
        probs = priorities / (priorities.sum() + 1e-10)
        indices = torch.multinomial(probs, min(batch_size, len(self.buffer)), replacement=False)
        x = torch.stack([self.buffer[i][0] for i in indices]).to(self.device)
        labels = torch.stack([self.buffer[i][1] for i in indices]).to(self.device)
        return x, labels

    def evict_oldest(self, fraction: float = 0.5):
        """Evict oldest fraction of buffer after sleep."""
        n_evict = int(len(self.buffer) * fraction)
        if n_evict > 0:
            self.buffer = self.buffer[n_evict:]

    def __len__(self):
        return len(self.buffer)


class MicroCritic(nn.Module):
    """Tiny adversarial network per layer for sleep consolidation.

    Learns to score real activations high, noise low.
    Used as the fitness signal during sleep phase.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, device: torch.device | None = None):
        super().__init__()
        dev = device or torch.device("cpu")
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        ).to(dev)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
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

    def compute_fitness(self, activations: torch.Tensor) -> float:
        """Fitness = mean critic score. Used during sleep phase."""
        self.eval()
        with torch.no_grad():
            return self.net(activations.detach()).mean().item()


class SleepWakeScheduler:
    """Manages wake/sleep cycle for a TFLE model.

    Wake (wake_steps): train with CDLL fitness on streaming data
    Sleep (sleep_steps): train with micro-critic fitness on replay buffer

    The engine (gpu_engine.py) calls is_wake() to determine which fitness
    to use, and feeds data accordingly.
    """

    def __init__(self, model, config: TFLEConfig, device: torch.device | None = None):
        self.model = model
        self.config = config
        self.device = device or torch.device("cpu")

        self.replay_buffer = ReplayBuffer(
            max_size=config.swt_replay_buffer_size,
            device=self.device,
        )

        # Micro-critics: one per layer, on that layer's device
        self.micro_critics: list[MicroCritic] = []
        for i, layer in enumerate(model.layers):
            dev = model.device_map[i] if model.multi_gpu else self.device
            self.micro_critics.append(
                MicroCritic(
                    input_dim=layer.out_features,
                    hidden_dim=config.critic_hidden,
                    device=dev,
                )
            )

        self.step_in_cycle = 0
        self.total_wake_steps = config.wake_steps
        self.total_sleep_steps = config.sleep_steps
        self.cycle_length = self.total_wake_steps + self.total_sleep_steps

    def is_wake(self) -> bool:
        """True if in wake phase, False if in sleep phase."""
        return self.step_in_cycle < self.total_wake_steps

    def step(self):
        """Advance the wake/sleep cycle counter."""
        self.step_in_cycle = (self.step_in_cycle + 1) % self.cycle_length

    def on_wake_step(self, x: torch.Tensor, labels: torch.Tensor, layer_activations: list[torch.Tensor]):
        """Called after each wake training step. Stores experience and trains critics."""
        # Compute surprise from loss
        with torch.no_grad():
            eval_result = self.model.evaluate(x, labels)
            surprise = eval_result["loss"]
        self.replay_buffer.add(x, labels, surprise)

        # Train critics on current activations
        for i, act in enumerate(layer_activations):
            self.micro_critics[i].train_step(act)

    def on_sleep_end(self):
        """Called after sleep phase. Evict old experiences."""
        self.replay_buffer.evict_oldest(self.config.replay_evict_fraction)

    def get_sleep_batch(self, batch_size: int = 64) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Get a batch from the replay buffer for sleep training."""
        return self.replay_buffer.sample(batch_size)

    def get_critic_fitness(self, layer_idx: int, activations: torch.Tensor) -> float:
        """Get micro-critic fitness for a layer (used during sleep)."""
        return self.micro_critics[layer_idx].compute_fitness(activations)

    def get_critic_scores(self) -> list[float]:
        """Get all critic scores for dashboard display."""
        return [c.evaluate(torch.randn(32, c.net[0].in_features, device=c.device))
                for c in self.micro_critics]
