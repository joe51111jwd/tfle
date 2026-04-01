"""Complete parameter configuration for TFLE algorithm."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class InitMethod(Enum):
    BALANCED_RANDOM = "balanced_random"
    SPARSE_RANDOM = "sparse_random"
    KAIMING_TERNARY = "kaiming_ternary"


class SelectionMethod(Enum):
    TRACE_WEIGHTED = "trace_weighted"
    UNIFORM_RANDOM = "uniform_random"
    GRADIENT_FREE_SALIENCY = "gradient_free_saliency"
    ANTI_CORRELATED = "anti_correlated"


class FlipDirectionBias(Enum):
    RANDOM = "random"
    CREDIT_BIASED = "credit_biased"
    TOWARD_ZERO = "toward_zero"
    AWAY_FROM_ZERO = "away_from_zero"


class BatchFlipMode(Enum):
    SIMULTANEOUS = "simultaneous"
    SEQUENTIAL = "sequential"
    GROUPED = "grouped"


class FlipRevertMode(Enum):
    IMMEDIATE_REVERT = "immediate_revert"
    COOLDOWN = "cooldown"


class TraceIncrement(Enum):
    BINARY = "binary"
    MAGNITUDE = "magnitude"
    SQUARED = "squared"


class ErrorSignalType(Enum):
    BINARY_CORRECT_INCORRECT = "binary_correct_incorrect"
    LOSS_MAGNITUDE = "loss_magnitude"
    CONFIDENCE_GAP = "confidence_gap"


class CreditNormalization(Enum):
    NONE = "none"
    LAYER_NORM = "layer_norm"
    RANK = "rank"


class CoolingSchedule(Enum):
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    COSINE = "cosine"
    ADAPTIVE = "adaptive"


class AcceptanceFunction(Enum):
    BOLTZMANN = "boltzmann"
    THRESHOLD = "threshold"
    METROPOLIS = "metropolis"


class FitnessType(Enum):
    CONTRASTIVE = "contrastive"
    PREDICTIVE = "predictive"
    COMPRESSION = "compression"
    HYBRID = "hybrid"
    TASK_LOSS = "task_loss"  # Use model-level cross-entropy as fitness signal


class CorruptionMethod(Enum):
    LABEL_SHUFFLE = "label_shuffle"
    GAUSSIAN_NOISE = "gaussian_noise"
    INPUT_MASK = "input_mask"
    FEATURE_PERMUTE = "feature_permute"
    MIXUP = "mixup"


class GoodnessMetric(Enum):
    SUM_OF_SQUARES = "sum_of_squares"
    MEAN_ACTIVATION = "mean_activation"
    MAX_ACTIVATION = "max_activation"
    ENTROPY = "entropy"


class FitnessBaseline(Enum):
    ABSOLUTE = "absolute"
    RELATIVE = "relative"


class CorruptionSchedule(Enum):
    FIXED = "fixed"
    INCREASING = "increasing"
    CURRICULUM = "curriculum"


class LayerTrainingOrder(Enum):
    PARALLEL = "parallel"
    BOTTOM_UP = "bottom_up"
    RANDOM = "random"


class DataStreaming(Enum):
    EPOCHS = "epochs"
    STREAMING = "streaming"


@dataclass
class TFLEConfig:
    """Complete configuration for the TFLE algorithm.

    All ~70 parameters from the spec with their documented defaults.
    """

    # --- 1. Weight Initialization ---
    init_method: InitMethod = InitMethod.BALANCED_RANDOM
    init_zero_bias: float = 0.5
    init_seed: Optional[int] = None

    # --- 2. Candidate Selection ---
    flip_rate: float = 0.03
    selection_method: SelectionMethod = SelectionMethod.TRACE_WEIGHTED
    trace_error_blend: float = 0.7
    protection_threshold: float = 0.8
    min_candidates_per_step: int = 10
    max_candidates_fraction: float = 0.10

    # --- 3. Flip Proposal ---
    flip_direction_bias: FlipDirectionBias = FlipDirectionBias.CREDIT_BIASED
    zero_gravity: float = 0.5
    batch_flip: BatchFlipMode = BatchFlipMode.SIMULTANEOUS
    flip_revert_on_reject: FlipRevertMode = FlipRevertMode.IMMEDIATE_REVERT
    cooldown_steps: int = 20

    # --- 4. Temporal Credit Assignment ---
    trace_decay: float = 0.95
    trace_increment: TraceIncrement = TraceIncrement.MAGNITUDE
    trace_activation_threshold: float = 0.0
    error_signal_type: ErrorSignalType = ErrorSignalType.BINARY_CORRECT_INCORRECT
    error_correlation_window: int = 100
    credit_normalization: CreditNormalization = CreditNormalization.LAYER_NORM
    separate_pos_neg_traces: bool = True

    # --- 5. Simulated Annealing ---
    initial_temperature: float = 10.0
    min_temperature: float = 0.05
    cooling_schedule: CoolingSchedule = CoolingSchedule.EXPONENTIAL
    cooling_rate: float = 0.9997
    reheat_on_plateau: bool = True
    plateau_window: int = 1000
    reheat_factor: float = 3.0
    acceptance_function: AcceptanceFunction = AcceptanceFunction.BOLTZMANN

    # --- 6. Local Fitness Function ---
    fitness_type: FitnessType = FitnessType.CONTRASTIVE
    fitness_eval_batch_size: int = 64
    corruption_method: CorruptionMethod = CorruptionMethod.LABEL_SHUFFLE
    corruption_strength: float = 0.5
    goodness_metric: GoodnessMetric = GoodnessMetric.SUM_OF_SQUARES
    fitness_baseline: FitnessBaseline = FitnessBaseline.RELATIVE
    fitness_ema_decay: float = 0.99

    # --- 7. Layer-Specific Scaling ---
    depth_scaled_flip_rate: bool = True
    flip_rate_depth_scale: float = 0.8
    depth_scaled_temperature: bool = True
    temperature_depth_scale: float = 0.7
    per_layer_fitness_type: bool = False

    # --- 8. Training Loop Control ---
    total_training_steps: int = 100_000
    eval_interval: int = 500
    early_stopping_patience: int = 10_000
    layer_training_order: LayerTrainingOrder = LayerTrainingOrder.PARALLEL
    steps_per_layer_per_cycle: int = 1
    data_streaming: DataStreaming = DataStreaming.EPOCHS
    shuffle_per_epoch: bool = True

    # --- 9. Memory and Compute ---
    trace_dtype: str = "float16"
    replay_buffer_size: int = 0
    checkpoint_interval: int = 5000
    max_memory_mb: Optional[int] = None
    gradient_free: bool = True

    # --- 10. Exploration and Diversity ---
    exploration_rate: float = 0.003
    exploration_decay: bool = True
    exploration_min: float = 0.0005
    diversity_penalty: bool = False
    weight_distribution_target: Optional[tuple[float, float, float]] = None
    tabu_list_size: int = 0

    # --- 11. Corruption Parameters ---
    num_negative_samples: int = 1
    corruption_schedule: CorruptionSchedule = CorruptionSchedule.FIXED
    hard_negative_mining: bool = False
    corruption_per_layer: str = "shared"

    # --- 12. Convergence Monitoring ---
    fitness_history_window: int = 200
    convergence_threshold: float = 0.0001
    oscillation_detection: bool = True
    oscillation_damping: float = 0.5
    log_weight_distribution: bool = True
    log_trace_statistics: bool = True
    log_acceptance_rate: bool = True

    # --- Model architecture (not in param tables but needed) ---
    layer_sizes: list[int] = field(default_factory=lambda: [784, 512, 256, 10])
    learning_rate_ste: float = 0.001  # for backprop baseline

    def get_flip_rate_for_layer(self, layer_idx: int) -> float:
        if self.depth_scaled_flip_rate:
            return self.flip_rate * (self.flip_rate_depth_scale ** layer_idx)
        return self.flip_rate

    def get_temperature_for_layer(self, base_temp: float, layer_idx: int) -> float:
        if self.depth_scaled_temperature:
            return base_temp * (self.temperature_depth_scale ** layer_idx)
        return base_temp
