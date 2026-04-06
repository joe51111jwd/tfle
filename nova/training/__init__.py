"""NOVA 2.4B post-training modules.

Modules:
  distill          - Reasoning distillation with LoRA
  rewards          - Math, code, and format reward functions
  grpo             - Dr. GRPO trainer
  block_local_tfle - Block-local TFLE (key innovation)
  tfle_grpo        - TFLE-GRPO integration
  swt              - Sleep-wake training for continual learning
  pretrain_stages  - 3-stage pretraining (synthetic, filtered, cooldown)
"""

# Lazy imports to avoid circular deps and missing optional modules
try:
    from .tokenizer_setup import get_tokenizer, encode, decode, VOCAB_SIZE
except ImportError:
    pass

try:
    from .data_loader import PretrainDataLoader, FilteredPretrainDataLoader
except ImportError:
    pass

try:
    from .checkpoint import CheckpointManager, CheckpointState
except ImportError:
    pass

try:
    from .pretrain_stages import (
        SyntheticCurriculumLoader,
        PerplexityFilter,
        STEStabilityMonitor,
        get_stage_config,
    )
except ImportError:
    pass
