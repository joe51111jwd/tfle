"""NOVA 2.4B post-training modules.

Modules:
  distill          - Reasoning distillation with LoRA
  rewards          - Math, code, and format reward functions
  grpo             - Dr. GRPO trainer
  block_local_tfle - Block-local TFLE (key innovation)
  tfle_grpo        - TFLE-GRPO integration
  swt              - Sleep-wake training for continual learning
"""

# Lazy imports to avoid circular deps and missing optional modules
try:
    from .tokenizer_setup import get_tokenizer, encode, decode, VOCAB_SIZE
except ImportError:
    pass

try:
    from .data_loader import PretrainDataLoader
except ImportError:
    pass

try:
    from .checkpoint import CheckpointManager, CheckpointState
except ImportError:
    pass
