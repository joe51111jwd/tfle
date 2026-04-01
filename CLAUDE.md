# CLAUDE.md — TFLE Project

## What This Is
Trit-Flip Local Evolution: gradient-free training for ternary neural networks.
Weights are {-1, 0, +1}. Training mutates weights and keeps changes that reduce loss.
No backpropagation, no gradients, no optimizer state.

## Key Commands
```bash
pip install torch torchvision tqdm pyyaml
python experiments/phase1_mnist.py          # Original experiment
python experiments/phase1_tuned.py          # Tuned task-loss fitness
python server/launch.py                     # Config-driven multi-GPU
python server/launch.py --gpu 0             # Single GPU
python server/launch.py --config my.yaml    # Custom config
```

## Algorithm Integrity Rules

**TFLE trains layers SEQUENTIALLY when using task-loss fitness.**
Each layer's accepted flips change the model, which affects the next layer's
fitness evaluation. Layers must be trained in order.

**Within-layer batched proposals (K>1) ARE safe and encouraged.**
Generating K flip proposals for the SAME layer and picking the best is just
smarter search. The traces learn from the single accept/reject outcome.
This is how Evolution Strategies work. Use K=32-256 for small models.

**Cross-layer parallel training IS safe with local fitness (CDLL/Mono-Forward).**
If each layer has its own fitness function that only depends on that layer's
input and output, all layers can train simultaneously. This is the path to
GPU saturation at scale. Build CDLL first.

**Cross-layer parallel training is NOT safe with task-loss fitness.**
Task-loss requires full-model forward passes. Layer 0's accepted changes
affect layer 1's fitness. Parallel here = evaluating against stale weights.

See `docs/NOVA_GPU_ACCELERATION_CC_OUTLINE.md` for the full implementation plan.

## GPU Usage

TFLE will NOT saturate a GPU on small models (<10M params). This is expected.

**Why:** Each forward pass on a 500K param model takes microseconds on GPU.
Python overhead (candidate selection, flip proposals, trace updates) takes
milliseconds. The GPU is idle 97% of the time waiting for Python.

**What helps:**
- Bigger batch sizes (512+) — makes each forward pass meatier
- Bigger models — more work per forward pass
- Moving weights to GPU — avoids CPU↔GPU data transfer
- Vectorized flip proposals — removed the Python for-loop

**What does NOT help with task-loss:**
- Parallelizing across layers (they're sequential with task-loss)

**What DOES help (and should be implemented):**
- K>1 within-layer proposals — same layer, pick best of K, GPU-parallel
- CDLL local fitness — enables cross-layer parallelism
- Cached prefix/suffix — only recompute the varying layer, not full model

**Expected GPU utilization by model size:**
- 500K params: ~5-10%
- 1.5M params: ~20-40%
- 10M+ params: ~60-80%
- 2.4B params (NOVA): ~90%+ (this is the target scale)

**Multi-GPU:** Use config.yaml to assign different experiments to different GPUs.
Each GPU runs its experiment list sequentially. Both GPUs run in parallel.
Use CUDA_VISIBLE_DEVICES or the --gpu flag.

## Current State

**Fitness function:** Task-loss (cross-entropy). The original contrastive fitness
was broken (10.31% on MNIST = random chance). Task-loss gave first-ever convergence
to 23.54% at 20K steps. Tuned cosine temperature + 100K steps running now.

**Key files:**
- `tfle/layers.py` — Core algorithm. `train_step()` is the main loop.
- `tfle/model.py` — Composes layers. `train_step()` passes task_loss_fn to layers.
- `tfle/config.py` — All 70+ parameters. FitnessType.TASK_LOSS is the fix.
- `tfle/training.py` — Training loop. Moves data to model device (GPU).
- `server/launch.py` — Config-driven multi-GPU launcher.
- `server/config.yaml` — Experiment definitions.

**What's broken:**
- `transformer.py` line ~223: uses torch.randn() instead of actual activations
- `fitness.py` has unused fitness functions (predictive, hybrid) — not wired in

## Hyperparameter Findings

| Parameter | Bad | Good | Why |
|-----------|-----|------|-----|
| Start temp | 0.5 | 0.10-0.20 | Too high = accept bad flips, too low = 0% acceptance |
| Temp schedule | exponential 0.9999 | cosine | Exponential barely decays in 100K steps |
| Flip rate | — | 0.01-0.03 | Lower for deeper layers (depth scaling) |
| Batch size | 64 | 512 | Larger = more stable loss estimate per forward pass |
| Reheat | off | on (window=3000, factor=2.5) | Escape local optima |

## Architecture Notes

- `[784, 256, 10]` — MNIST proof of concept (~200K params)
- `[784, 512, 256, 10]` — MNIST with more capacity (~530K params)
- `[3072, 512, 256, 10]` — CIFAR-10 (~1.7M params)
- All use ReLU activation (not ReLU² — that's for the full NOVA model)

## Related Projects

- `~/Projects/nova/` — The full NOVA model (2.4B params, hybrid Transformer-Mamba)
- This repo is the training algorithm. NOVA is the model it trains.
