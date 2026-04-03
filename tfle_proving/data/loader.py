"""Data loading for character-level language modeling."""

from __future__ import annotations

import os
import urllib.request

import torch
from torch.utils.data import Dataset, DataLoader

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = os.path.join(os.path.dirname(__file__), "corpus")


def download_shakespeare() -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "tinyshakespeare.txt")
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.read()
    try:
        print(f"Downloading TinyShakespeare from {SHAKESPEARE_URL}...")
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)
        with open(path, "r") as f:
            text = f.read()
        print(f"Downloaded {len(text):,} characters.")
        return text
    except Exception as e:
        print(f"Download failed ({e}), using synthetic text.")
        return _synthetic_text()


def _synthetic_text() -> str:
    patterns = [
        "the cat sat on the mat. ",
        "to be or not to be that is the question. ",
        "all that glitters is not gold. ",
        "a rose by any other name would smell as sweet. ",
        "the quick brown fox jumps over the lazy dog. ",
        "now is the winter of our discontent. ",
    ]
    text = ""
    for _ in range(20000):
        for p in patterns:
            text += p
    return text


class CharDataset(Dataset):
    """Sliding-window character dataset. Each example: context_len chars -> next char."""

    def __init__(self, text: str, context_len: int = 128, stride: int = 1):
        self.data = torch.tensor(
            [ord(c) % 256 for c in text], dtype=torch.long
        )
        self.context_len = context_len
        self.stride = stride
        self.n_examples = (len(self.data) - context_len - 1) // stride + 1

    def __len__(self) -> int:
        return self.n_examples

    def __getitem__(self, idx: int):
        start = idx * self.stride
        x = self.data[start : start + self.context_len]
        y = self.data[start + self.context_len]
        return x, y


def create_char_dataloaders(
    context_len: int = 128,
    batch_size: int = 256,
    num_workers: int = 2,
    val_fraction: float = 0.1,
    stride: int = 3,
    text: str | None = None,
) -> tuple[DataLoader, DataLoader, int]:
    """Create train/val dataloaders for character-level LM.

    Returns (train_loader, val_loader, vocab_size=256).
    """
    if text is None:
        text = download_shakespeare()

    split = int(len(text) * (1 - val_fraction))
    train_text = text[:split]
    val_text = text[split:]

    train_ds = CharDataset(train_text, context_len, stride=stride)
    val_ds = CharDataset(val_text, context_len, stride=1)

    print(f"Train: {len(train_ds):,} examples, Val: {len(val_ds):,} examples")
    print(f"Context length: {context_len}, Vocab size: 256 (byte-level)")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader, 256
