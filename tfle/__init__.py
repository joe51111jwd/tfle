"""TFLE: Trit-Flip Local Evolution — gradient-free training for ternary neural networks."""

__version__ = "0.1.0"

from .cdll import CDLLFitness
from .config import resolve_device
from .layers import generate_k_proposals
from .local_heads import LocalClassifierHead
from .model import batched_task_loss_eval
from .swt import MicroCritic, ReplayBuffer, SleepWakeTrainer
from .training import make_dataloader
