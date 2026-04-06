"""Reasoning distillation pipeline for NOVA 2.4B.

Fine-tunes on DeepSeek-R1 reasoning traces (or synthetic fallback) using LoRA
on attention Q/K/V/O projections. Trains the model to produce structured
<think>...</think><answer>...</answer> reasoning chains.
"""
from __future__ import annotations

import json
import logging
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


# ── LoRA adapter ──────────────────────────────────────────────

class LoRALinear(nn.Module):
    """Low-rank adaptation wrapper around an existing linear layer."""

    def __init__(self, base_layer: nn.Module, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * (1.0 / math.sqrt(rank)))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        for p in self.base_layer.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        return base_out + lora_out


def apply_lora_to_attention(model: nn.Module, rank: int = 16, alpha: float = 32.0) -> list[LoRALinear]:
    """Apply LoRA to all attention Q/K/V/O projections in the model.

    Looks for modules named *q_proj*, *k_proj*, *v_proj*, *o_proj* that are
    linear layers (nn.Linear or BitLinear).
    """
    lora_modules = []
    target_names = ("q_proj", "k_proj", "v_proj", "o_proj")

    for name, module in list(model.named_modules()):
        if not any(t in name for t in target_names):
            continue
        if not hasattr(module, "in_features"):
            continue

        parent_name, attr_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model
        if parent_name:
            for part in parent_name.split("."):
                parent = getattr(parent, part)

        lora = LoRALinear(module, rank=rank, alpha=alpha)
        lora = lora.to(next(module.parameters()).device)
        setattr(parent, attr_name, lora)
        lora_modules.append(lora)

    logger.info(f"Applied LoRA (rank={rank}, alpha={alpha}) to {len(lora_modules)} projections")
    return lora_modules


# ── BPE tokenizer ─────────────────────────────────────────────

class BPETokenizer:
    """Wraps HuggingFace tokenizers BPE. Falls back to whitespace split."""

    def __init__(self, vocab_size: int = 32000, max_seq_len: int = 4096):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self._tokenizer = None
        self._word2idx: dict[str, int] = {}

        try:
            from tokenizers import Tokenizer, models, trainers, pre_tokenizers
            self._tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
            self._tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        except ImportError:
            logger.warning("tokenizers library not available, using whitespace fallback")

    def train_from_texts(self, texts: list[str]):
        if self._tokenizer is not None:
            from tokenizers import trainers, pre_tokenizers
            trainer = trainers.BpeTrainer(
                vocab_size=self.vocab_size,
                special_tokens=["<pad>", "<unk>", "<eos>", "<think>", "</think>", "<answer>", "</answer>"],
            )
            self._tokenizer.train_from_iterator(texts, trainer=trainer)
        else:
            self._build_word_vocab(texts)

    def _build_word_vocab(self, texts: list[str]):
        word_counts: dict[str, int] = {}
        for text in texts:
            for w in text.split():
                word_counts[w] = word_counts.get(w, 0) + 1
        sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])
        self._word2idx = {"<pad>": 0, "<unk>": 1, "<eos>": 2}
        for w, _ in sorted_words[:self.vocab_size - 3]:
            self._word2idx[w] = len(self._word2idx)

    def encode(self, text: str) -> list[int]:
        if self._tokenizer is not None:
            return self._tokenizer.encode(text).ids
        return [self._word2idx.get(w, 1) for w in text.split()]

    def decode(self, ids: list[int]) -> str:
        if self._tokenizer is not None:
            return self._tokenizer.decode(ids)
        idx2word = {v: k for k, v in self._word2idx.items()}
        return " ".join(idx2word.get(i, "<unk>") for i in ids if i not in (0,))

    @property
    def pad_id(self) -> int:
        return 0


# ── Dataset ───────────────────────────────────────────────────

class ReasoningTraceDataset(Dataset):
    """Dataset of <think>...</think><answer>...</answer> reasoning traces."""

    def __init__(self, traces: list[str], tokenizer: BPETokenizer, max_seq_len: int = 4096):
        self.samples: list[torch.Tensor] = []
        for text in traces:
            ids = tokenizer.encode(text)
            ids.append(2)  # <eos>
            if len(ids) > max_seq_len + 1:
                ids = ids[:max_seq_len + 1]
            while len(ids) < max_seq_len + 1:
                ids.append(tokenizer.pad_id)
            self.samples.append(torch.tensor(ids, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        t = self.samples[idx]
        return t[:-1], t[1:]


# ── Data loading ──────────────────────────────────────────────

def load_deepseek_traces(max_samples: int = 10000) -> list[str]:
    """Load DeepSeek-R1 distillation traces from HuggingFace.

    Falls back to synthetic traces if the dataset isn't available.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "deepseek-ai/DeepSeek-R1-Distill-SFT",
            split="train",
            streaming=True,
        )
        traces = []
        for i, example in enumerate(ds):
            if i >= max_samples:
                break
            messages = example.get("messages", example.get("conversations", []))
            assistant_text = ""
            for msg in messages:
                role = msg.get("role", msg.get("from", ""))
                content = msg.get("content", msg.get("value", ""))
                if role in ("assistant", "gpt"):
                    assistant_text = content
                    break
            if assistant_text:
                if "<think>" not in assistant_text:
                    assistant_text = f"<think>{assistant_text}</think><answer>{assistant_text.split('.')[-2].strip() if '.' in assistant_text else assistant_text}</answer>"
                traces.append(assistant_text)

        if traces:
            logger.info(f"Loaded {len(traces)} DeepSeek-R1 traces from HuggingFace")
            return traces
    except Exception as e:
        logger.warning(f"Could not load DeepSeek-R1 traces: {e}")

    logger.info("Falling back to synthetic reasoning traces")
    return generate_synthetic_traces(max_samples)


def generate_synthetic_traces(n: int = 10000) -> list[str]:
    """Generate synthetic math reasoning traces with <think>/<answer> format."""
    random.seed(42)
    traces = []

    for _ in range(n):
        r = random.random()
        if r < 0.4:
            a, b = random.randint(1, 100), random.randint(1, 100)
            op, verb = random.choice([("+", "add"), ("-", "subtract"), ("*", "multiply")])
            result = eval(f"{a} {op} {b}")
            text = (
                f"<think>I need to {verb} {a} and {b}. "
                f"Computing {a} {op} {b} step by step. "
                f"The result is {result}.</think>"
                f"<answer>{result}</answer>"
            )
        elif r < 0.7:
            a, b = random.randint(1, 50), random.randint(1, 50)
            c = random.randint(2, 20)
            s1 = a + b
            result = s1 * c
            text = (
                f"<think>First, I add {a} and {b} to get {s1}. "
                f"Then I multiply {s1} by {c}. "
                f"That gives {s1} * {c} = {result}.</think>"
                f"<answer>{result}</answer>"
            )
        else:
            items = random.randint(2, 20)
            price = random.randint(1, 10)
            total = items * price
            text = (
                f"<think>Each item costs {price} and there are {items} items. "
                f"I need to multiply {items} by {price} to get the total. "
                f"{items} * {price} = {total}.</think>"
                f"<answer>{total}</answer>"
            )
        traces.append(text)

    return traces


# ── Format compliance ─────────────────────────────────────────

def check_format_compliance(text: str) -> dict[str, bool]:
    """Check if output follows <think>...</think><answer>...</answer> format."""
    has_think_open = "<think>" in text
    has_think_close = "</think>" in text
    has_answer_open = "<answer>" in text
    has_answer_close = "</answer>" in text

    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)

    proper_order = False
    if think_match and answer_match:
        proper_order = think_match.end() <= answer_match.start()

    return {
        "has_think": has_think_open and has_think_close,
        "has_answer": has_answer_open and has_answer_close,
        "proper_order": proper_order,
        "fully_compliant": all([
            has_think_open, has_think_close,
            has_answer_open, has_answer_close,
            proper_order,
        ]),
    }


# ── Distillation Trainer ─────────────────────────────────────

@dataclass
class DistillationConfig:
    lr: float = 2e-5
    epochs: int = 3
    batch_size: int = 4
    max_seq_len: int = 4096
    warmup_steps: int = 100
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    lora_rank: int = 16
    lora_alpha: float = 32.0
    max_traces: int = 10000
    eval_fraction: float = 0.1


class DistillationTrainer:
    """Fine-tune NOVA on reasoning traces using LoRA on attention projections.

    Pipeline:
    1. Load DeepSeek-R1 traces (or synthetic fallback)
    2. Apply LoRA to attention Q/K/V/O
    3. Train with cross-entropy on full sequence (think + answer)
    4. Evaluate format compliance after each epoch
    """

    def __init__(
        self,
        model: nn.Module,
        config: DistillationConfig,
        device: torch.device,
        tokenizer: Optional[BPETokenizer] = None,
    ):
        self.model = model
        self.config = config
        self.device = device
        self.tokenizer = tokenizer or BPETokenizer(max_seq_len=config.max_seq_len)

        self.lora_modules = apply_lora_to_attention(
            model, rank=config.lora_rank, alpha=config.lora_alpha,
        )

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable: {trainable:,} / {total:,} ({trainable / total:.1%})")

        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

    def prepare_data(self, traces: list[str] | None = None) -> tuple[DataLoader, DataLoader]:
        """Load traces and create train/val dataloaders."""
        if traces is None:
            traces = load_deepseek_traces(self.config.max_traces)

        self.tokenizer.train_from_texts(traces)

        split = int(len(traces) * (1 - self.config.eval_fraction))
        train_ds = ReasoningTraceDataset(traces[:split], self.tokenizer, self.config.max_seq_len)
        val_ds = ReasoningTraceDataset(traces[split:], self.tokenizer, self.config.max_seq_len)

        train_dl = DataLoader(
            train_ds, batch_size=self.config.batch_size,
            shuffle=True, num_workers=2, pin_memory=True,
        )
        val_dl = DataLoader(
            val_ds, batch_size=self.config.batch_size,
            shuffle=False, num_workers=2, pin_memory=True,
        )

        logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
        return train_dl, val_dl

    def _get_vocab_size(self) -> int:
        """Infer vocab size from model's output layer."""
        for name, module in self.model.named_modules():
            if "lm_head" in name or "output" in name:
                if hasattr(module, "out_features"):
                    return module.out_features
        for p in self.model.parameters():
            pass
        return self.tokenizer.vocab_size

    def train(self, traces: list[str] | None = None) -> dict:
        """Run the full distillation pipeline."""
        train_dl, val_dl = self.prepare_data(traces)
        vocab_size = self._get_vocab_size()

        total_steps = self.config.epochs * len(train_dl)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps, eta_min=1e-6,
        )
        use_amp = self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda") if use_amp else None

        results = {"epochs": [], "config": vars(self.config)}
        step = 0

        for epoch in range(self.config.epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            for x, y in train_dl:
                x, y = x.to(self.device), y.to(self.device)

                # warmup
                if step < self.config.warmup_steps:
                    lr_now = self.config.lr * (step + 1) / self.config.warmup_steps
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = lr_now

                if use_amp:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        logits = self.model(x)
                        loss = F.cross_entropy(
                            logits.reshape(-1, vocab_size),
                            y.reshape(-1),
                            ignore_index=self.tokenizer.pad_id,
                        )
                    self.optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    logits = self.model(x)
                    loss = F.cross_entropy(
                        logits.reshape(-1, vocab_size),
                        y.reshape(-1),
                        ignore_index=self.tokenizer.pad_id,
                    )
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                    self.optimizer.step()

                if step >= self.config.warmup_steps:
                    scheduler.step()

                epoch_loss += loss.item()
                n_batches += 1
                step += 1

            # validation
            val_loss = self._evaluate(val_dl, vocab_size)
            avg_train = epoch_loss / max(n_batches, 1)

            # format compliance
            compliance = self._evaluate_format(n_samples=20)

            epoch_result = {
                "epoch": epoch + 1,
                "train_loss": avg_train,
                "val_loss": val_loss,
                "format_compliance": compliance,
            }
            results["epochs"].append(epoch_result)
            logger.info(
                f"Epoch {epoch + 1}: train={avg_train:.4f} val={val_loss:.4f} "
                f"compliance={compliance:.0%}"
            )

        results["final_compliance"] = self._evaluate_format(n_samples=50)
        return results

    @torch.no_grad()
    def _evaluate(self, val_dl: DataLoader, vocab_size: int) -> float:
        self.model.eval()
        total_loss, n = 0.0, 0
        for x, y in val_dl:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                y.reshape(-1),
                ignore_index=self.tokenizer.pad_id,
            )
            total_loss += loss.item()
            n += 1
        return total_loss / max(n, 1)

    @torch.no_grad()
    def _evaluate_format(self, n_samples: int = 20) -> float:
        """Generate samples and check format compliance."""
        self.model.eval()
        prompts = [
            "What is 5 + 3?",
            "What is 12 * 4?",
            "Add 10 and 20 then multiply by 3.",
        ]

        compliant = 0
        total = min(n_samples, len(prompts))

        for i in range(total):
            text = self._generate(prompts[i % len(prompts)])
            result = check_format_compliance(text)
            if result["fully_compliant"]:
                compliant += 1

        return compliant / max(total, 1)

    @torch.no_grad()
    def _generate(self, prompt: str, max_tokens: int = 200, temperature: float = 0.8) -> str:
        """Autoregressive generation from prompt."""
        self.model.eval()
        ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)

        for _ in range(max_tokens):
            if input_ids.shape[1] >= self.config.max_seq_len:
                break
            logits = self.model(input_ids)
            nxt = logits[0, -1, :].float()
            if temperature > 0:
                probs = F.softmax(nxt / temperature, dim=-1)
                tok = torch.multinomial(probs, 1)
            else:
                tok = nxt.argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, tok.unsqueeze(0)], dim=1)
            if tok.item() == 2:  # <eos>
                break

        return self.tokenizer.decode(input_ids[0].tolist())
