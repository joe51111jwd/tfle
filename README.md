# TFLE: Trit-Flip Local Evolution

A gradient-free training algorithm for ternary neural networks.

## What is TFLE?

TFLE treats ternary weight training as what it actually is: a **combinatorial search problem on a discrete grid**, not continuous optimization. Instead of backpropagation with gradient descent, TFLE uses:

1. **Local evolutionary search** — propose random weight flips ({-1, 0, +1} transitions), evaluate with a local fitness function, keep improvements
2. **Temporal credit assignment** — inspired by neuroscience (STDP), tracks which weights were active during correct/incorrect predictions to guide which weights to flip
3. **Layer-independent training** — each layer trains with its own local fitness signal, eliminating the backward pass entirely

### Memory Advantage

| Component | Backprop (BitNet) | TFLE |
|---|---|---|
| Weights (100B params) | ~20 GB | ~20 GB |
| Gradients | ~200 GB (FP16) | **0 GB** |
| Optimizer state (Adam) | ~400 GB (FP32) | **0 GB** |
| Activation checkpoints | ~50-200 GB | **0 GB** |
| Temporal traces | 0 GB | ~12.5 GB |
| **Total** | **~670-1020 GB** | **~32.5 GB** |

## Installation

```bash
git clone https://github.com/joe51111jwd/tfle.git
cd tfle
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Requires Python 3.10+ and PyTorch 2.0+.

## Quick Start

```python
from tfle.config import TFLEConfig
from tfle.model import TFLEModel

# Create a ternary MLP
config = TFLEConfig(layer_sizes=[784, 512, 256, 10])
model = TFLEModel(config)

# Train with TFLE
from tfle.training import TFLETrainer
trainer = TFLETrainer(model, config, train_loader, val_loader)
result = trainer.train()
print(f"Accuracy: {result.final_accuracy:.4f}")
```

## Architecture

```
tfle/
├── config.py        # ~70 parameters as a typed dataclass
├── layers.py        # Ternary linear layers with TFLE training
├── conv_layers.py   # Ternary convolutional layers
├── model.py         # MLP model (Phase 1)
├── conv_model.py    # CNN model for CIFAR-10 (Phase 2)
├── transformer.py   # Ternary transformer (Phase 2)
├── fitness.py       # Contrastive + predictive coding fitness
├── corruption.py    # Data corruption strategies
├── annealing.py     # Temperature scheduling (simulated annealing)
├── monitor.py       # Convergence monitoring and diagnostics
├── training.py      # Training loop orchestration
├── baseline.py      # Backprop + STE baseline for comparison
├── benchmarks.py    # Benchmarking utilities
└── analysis.py      # Visualization and reporting
```

## Running Experiments

### Phase 1: MNIST Proof of Concept
```bash
python experiments/phase1_mnist.py
```
Trains a 3-layer ternary MLP on MNIST and compares against a backprop+STE baseline.

### Phase 2: CIFAR-10 and Transformer
```bash
python experiments/phase2_cifar10.py
python experiments/phase2_transformer.py
```

### Phase 3: Full Benchmarks
```bash
python experiments/phase3_benchmarks.py
```
Runs flip rate sensitivity, temperature schedule comparison, memory analysis, and convergence analysis with multiple seeds.

## Configuration

All ~70 parameters are documented in `TFLEConfig`:

```python
from tfle.config import TFLEConfig, InitMethod, CoolingSchedule

config = TFLEConfig(
    # Architecture
    layer_sizes=[784, 512, 256, 10],

    # Weight init
    init_method=InitMethod.BALANCED_RANDOM,
    init_zero_bias=0.5,

    # Candidate selection
    flip_rate=0.03,              # 3% of weights considered per step
    protection_threshold=0.8,    # protect top 20% success-correlated weights

    # Temporal credit
    trace_decay=0.95,
    separate_pos_neg_traces=True,

    # Simulated annealing
    initial_temperature=10.0,
    cooling_schedule=CoolingSchedule.EXPONENTIAL,
    cooling_rate=0.9997,
    reheat_on_plateau=True,

    # Training
    total_training_steps=100_000,
    eval_interval=500,
)
```

See `tfle/config.py` for the complete parameter reference with documentation.

## Testing

```bash
pytest tests/ -v
```

124 tests covering all components: config, layers, conv layers, model, training, fitness functions, annealing, monitoring, benchmarks, and analysis.

## Key Design Decisions

- **Weights stored as int8** — each ternary weight uses 1 byte, not 2 bits (simplicity over compression for the prototype)
- **Traces stored as float16** — 2 bytes per weight for temporal credit tracking
- **No GPU required** — Phase 1 runs on any laptop with 8GB+ RAM
- **Layer-independent training** — each layer optimizes its own local fitness, enabling future parallelization across hardware

## Status

- **Phase 1** (complete): MLP on MNIST with contrastive fitness
- **Phase 2** (complete): CNN on CIFAR-10, predictive coding fitness, ternary transformer
- **Phase 3** (complete): Benchmarks, analysis tools, visualization

## License

MIT

## Authors

James Camarota (Las Vegas, NV) in collaboration with Claude (Anthropic)
