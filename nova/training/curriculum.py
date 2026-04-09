"""Sequence-length and batch-size curricula for NOVA pretraining.

Short sequences are cheap (attention is O(L^2)), so we start at 128 and ramp
up to the full 2048 context over the first 15% of training. This lets the
early-training steps fly through far more tokens per second than they would
at full context, and the ternary model mostly memorizes local structure in
those early steps anyway.

After the warmup finishes, an optional stochastic mode mixes in shorter
sequences (1024 / 1536) at low probability so the model stays robust to
length variation.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field


def get_current_seq_len(
    step: int,
    total_steps: int,
    start: int = 128,
    end: int = 2048,
    warmup_fraction: float = 0.15,
) -> int:
    """Linear sequence-length ramp with 128-token granularity.

    128 -> 2048 over the first `warmup_fraction` of training, then clamped
    at `end` for the remainder. Values are snapped to the nearest 128 so the
    data loader can pack cleanly.
    """
    if total_steps <= 0:
        return end
    if step < 0:
        return start

    warmup_steps = max(1, int(total_steps * warmup_fraction))
    if step >= warmup_steps:
        return end

    progress = step / warmup_steps
    current = int(start + progress * (end - start))
    current = (current // 128) * 128
    return max(start, min(current, end))


@dataclass
class SequenceLengthCurriculum:
    """Stateful wrapper over get_current_seq_len with optional stochastic mode.

    Attributes:
        total_steps: total training steps.
        start: minimum sequence length at step 0.
        end: maximum / target sequence length.
        warmup_fraction: fraction of training spent ramping.
        stochastic_seq_len: if True, sample from {1024, 1536, 2048} with
            probabilities {0.1, 0.2, 0.7} once warmup has finished.
        seed: RNG seed for the stochastic sampler (so curricula are reproducible
            across restarts).
    """

    total_steps: int
    start: int = 128
    end: int = 2048
    warmup_fraction: float = 0.15
    stochastic_seq_len: bool = False
    seed: int = 0

    current_step: int = 0
    _rng: random.Random = field(init=False, repr=False)

    _STOCHASTIC_LENGTHS: tuple[int, ...] = (1024, 1536, 2048)
    _STOCHASTIC_PROBS: tuple[float, ...] = (0.1, 0.2, 0.7)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    @property
    def warmup_steps(self) -> int:
        return max(1, int(self.total_steps * self.warmup_fraction))

    def is_warmup_done(self, step: int | None = None) -> bool:
        s = self.current_step if step is None else step
        return s >= self.warmup_steps

    def get_seq_len(self, step: int | None = None) -> int:
        s = self.current_step if step is None else step
        deterministic = get_current_seq_len(
            s,
            self.total_steps,
            start=self.start,
            end=self.end,
            warmup_fraction=self.warmup_fraction,
        )
        if not self.stochastic_seq_len or not self.is_warmup_done(s):
            return deterministic
        return self._rng.choices(
            self._STOCHASTIC_LENGTHS,
            weights=self._STOCHASTIC_PROBS,
            k=1,
        )[0]

    def step(self) -> int:
        """Advance one step and return the seq len for that step."""
        seq_len = self.get_seq_len(self.current_step)
        self.current_step += 1
        return seq_len

    def state_dict(self) -> dict:
        return {
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "start": self.start,
            "end": self.end,
            "warmup_fraction": self.warmup_fraction,
            "stochastic_seq_len": self.stochastic_seq_len,
            "seed": self.seed,
            "rng_state": self._rng.getstate(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.current_step = state["current_step"]
        self.total_steps = state["total_steps"]
        self.start = state["start"]
        self.end = state["end"]
        self.warmup_fraction = state["warmup_fraction"]
        self.stochastic_seq_len = state["stochastic_seq_len"]
        self.seed = state["seed"]
        if "rng_state" in state:
            self._rng.setstate(state["rng_state"])


@dataclass
class BatchSizeCurriculum:
    """Ramp per-GPU batch size from start to end over the first fraction.

    Default: bs=4 at step 0, bs=16 by step (5% of total). Rounds to the
    nearest power of 2 so data-loader code doesn't hiccup on odd sizes.
    """

    total_steps: int
    start: int = 4
    end: int = 16
    warmup_fraction: float = 0.05
    round_to_pow2: bool = True

    current_step: int = 0

    @property
    def warmup_steps(self) -> int:
        return max(1, int(self.total_steps * self.warmup_fraction))

    def get_batch_size(self, step: int | None = None) -> int:
        s = self.current_step if step is None else step
        if s < 0:
            return self.start
        if s >= self.warmup_steps:
            return self.end

        progress = s / self.warmup_steps
        bs = self.start + progress * (self.end - self.start)
        if self.round_to_pow2:
            return _round_to_pow2(bs, lo=self.start, hi=self.end)
        return max(self.start, min(int(bs), self.end))

    def step(self) -> int:
        bs = self.get_batch_size(self.current_step)
        self.current_step += 1
        return bs

    def state_dict(self) -> dict:
        return {
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "start": self.start,
            "end": self.end,
            "warmup_fraction": self.warmup_fraction,
            "round_to_pow2": self.round_to_pow2,
        }

    def load_state_dict(self, state: dict) -> None:
        self.current_step = state["current_step"]
        self.total_steps = state["total_steps"]
        self.start = state["start"]
        self.end = state["end"]
        self.warmup_fraction = state["warmup_fraction"]
        self.round_to_pow2 = state["round_to_pow2"]


def _round_to_pow2(value: float, lo: int, hi: int) -> int:
    if value <= lo:
        return lo
    if value >= hi:
        return hi

    candidates: list[int] = []
    p = 1
    while p <= hi:
        if p >= lo:
            candidates.append(p)
        p *= 2
    if hi not in candidates:
        candidates.append(hi)
    if lo not in candidates:
        candidates.append(lo)
    candidates.sort()

    best = candidates[0]
    best_diff = abs(value - best)
    for c in candidates[1:]:
        diff = abs(value - c)
        if diff < best_diff:
            best = c
            best_diff = diff
    return best
