"""Data loading for v2 experiments. Extends v1 with longer context support."""

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
        print(f"Downloading TinyShakespeare...")
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)
        with open(path, "r") as f:
            text = f.read()
        print(f"Downloaded {len(text):,} characters.")
        return text
    except Exception as e:
        print(f"Download failed ({e}), using synthetic text.")
        return _synthetic()


def _synthetic() -> str:
    patterns = [
        "the cat sat on the mat. ",
        "to be or not to be that is the question. ",
        "all that glitters is not gold. ",
        "a rose by any other name would smell as sweet. ",
    ]
    return "".join(patterns * 30000)


class CharDataset(Dataset):
    """Byte-level next-char prediction dataset."""

    def __init__(self, text: str, context_len: int = 256, stride: int = 3):
        self.data = torch.tensor([ord(c) % 256 for c in text], dtype=torch.long)
        self.context_len = context_len
        self.stride = stride
        self.n = (len(self.data) - context_len - 1) // stride + 1

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        s = idx * self.stride
        return self.data[s : s + self.context_len], self.data[s + self.context_len]


class SeqDataset(Dataset):
    """Sequence-level dataset: predict next token at EVERY position.

    Returns (input_seq, target_seq) where target[i] = input[i+1].
    Used for transformer training where we get loss from all positions.
    """

    def __init__(self, text: str, seq_len: int = 256, stride: int = 128):
        self.data = torch.tensor([ord(c) % 256 for c in text], dtype=torch.long)
        self.seq_len = seq_len
        self.stride = stride
        self.n = (len(self.data) - seq_len - 1) // stride + 1

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        s = idx * self.stride
        x = self.data[s : s + self.seq_len]
        y = self.data[s + 1 : s + self.seq_len + 1]
        return x, y


def create_seq_dataloaders(
    context_len: int = 256,
    batch_size: int = 64,
    num_workers: int = 2,
    val_fraction: float = 0.1,
    stride: int = 128,
    text: str | None = None,
) -> tuple[DataLoader, DataLoader, int]:
    """Create train/val loaders for sequence-level (transformer) training."""
    if text is None:
        text = download_shakespeare()
    split = int(len(text) * (1 - val_fraction))

    train_ds = SeqDataset(text[:split], context_len, stride)
    val_ds = SeqDataset(text[split:], context_len, stride=context_len)

    print(f"Seq dataset — Train: {len(train_ds):,}, Val: {len(val_ds):,}, "
          f"ctx={context_len}, vocab=256")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, 256


def create_char_dataloaders(
    context_len: int = 256,
    batch_size: int = 512,
    num_workers: int = 2,
    val_fraction: float = 0.1,
    stride: int = 3,
    text: str | None = None,
) -> tuple[DataLoader, DataLoader, int]:
    """Create train/val loaders for single-char prediction (MLP)."""
    if text is None:
        text = download_shakespeare()
    split = int(len(text) * (1 - val_fraction))

    train_ds = CharDataset(text[:split], context_len, stride)
    val_ds = CharDataset(text[split:], context_len, stride=1)

    print(f"Char dataset — Train: {len(train_ds):,}, Val: {len(val_ds):,}, "
          f"ctx={context_len}, vocab=256")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, 256
