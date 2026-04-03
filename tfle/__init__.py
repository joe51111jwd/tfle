"""TFLE: Trit-Flip Local Evolution — gradient-free training for ternary neural networks."""

__version__ = "0.1.0"

from .cdll import CDLLFitness
from .config import build_device_map, resolve_device
from .gpu_engine import SearchParallelEngine
from .layers import generate_k_proposals
from .local_heads import TernaryLocalHead
from .model import batched_task_loss_eval
from .swt import MicroCritic, ReplayBuffer, SleepWakeScheduler
from .training import make_dataloader
