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

**TFLE is one proposal per step. Do NOT batch proposals.**

The algorithm works like this:
1. Select candidates (weighted by temporal credit traces)
2. Propose ONE set of flips
3. Evaluate fitness before/after (two forward passes)
4. Accept or reject (simulated annealing)
5. Update traces based on the accept/reject outcome
6. Step N's traces inform step N+1's candidate selection

The traces learn from every single accept/reject decision sequentially.
This is what makes TFLE an intelligent evolutionary search, not random search.

**Batching multiple proposals breaks this.** If you evaluate 32 proposals at once
and pick the best, the traces don't learn from the 31 rejected ones. The search
degenerates to random search with selection — faster but dumber.

If you need speed, optimize the forward pass or write a CUDA kernel.
Do NOT parallelize the proposal-evaluate-accept loop.

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

**What does NOT help (and breaks the algorithm):**
- Batching multiple proposals per step
- Parallelizing the train_step across layers (they're sequential by design)

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
