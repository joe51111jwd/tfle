"""NOVA 2.4B post-training modules.

Modules:
  distill          - Reasoning distillation with LoRA
  rewards          - Math, code, and format reward functions
  grpo             - Dr. GRPO trainer
  block_local_tfle - Block-local TFLE (key innovation)
  tfle_grpo        - TFLE-GRPO integration
  swt              - Sleep-wake training for continual learning
  pretrain_stages  - 3-stage pretraining (synthetic, filtered, cooldown)
  dpo              - Direct Preference Optimization trainer
  self_dpo         - Self-DPO learning loop (preference pairs -> DPO)
  preference_store - Persistent JSONL storage for bracket matchups
  prune            - Model pruning (CONDITIONAL)
  ternarize        - Post-prune ternarization (CONDITIONAL)
  recover          - STE recovery training (CONDITIONAL)
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

try:
    from .dpo import DPOTrainer, DPOConfig
except ImportError:
    pass

try:
    from .preference_store import PreferenceStore
except ImportError:
    pass

try:
    from .generate_preferences import PreferenceGenerator
except ImportError:
    pass

try:
    from .self_dpo import SelfDPO, SelfDPOConfig
except ImportError:
    pass

# CONDITIONAL modules — only used if pruning tests pass
try:
    from .prune import ModelPruner, PruneConfig
except ImportError:
    pass

try:
    from .ternarize import Ternarizer
except ImportError:
    pass

try:
    from .recover import RecoveryTrainer, RecoveryConfig
except ImportError:
    pass
