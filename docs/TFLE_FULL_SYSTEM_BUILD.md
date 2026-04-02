# TFLE Full System Build — Claude Code Spec

> **Goal:** Build a proof-of-concept integrating all three algorithms (TFLE + CDLL + SWT) on CIFAR-100. Must scale to ANY number of GPUs and actually saturate them. The core learning algorithm stays identical regardless of GPU count or model size.

---

## The Scaling Idea: Search-Parallel TFLE

Standard distributed training splits gradients across GPUs. We have no gradients. TFLE is combinatorial search — propose flips, evaluate fitness, keep the best. The natural way to parallelize search is: **propose more candidates simultaneously, and evaluate each on massive batches.**

With 1 GPU, each layer proposes 1 set of weight flips per step. With N GPUs, each layer proposes N sets in parallel, evaluates all N, keeps the best. Same algorithm, same learning rule — just exploring N× more search space per step. More GPUs = better proposals per step, not just faster wall-clock.

**Why this saturates GPUs:** Each proposal evaluates CDLL fitness on a large batch (2048-4096 examples). That means each GPU is running full forward passes through its layer on thousands of examples per proposal. At 1M params that's moderate work; at 100M+ it's heavy compute. The key: batch size scales up to fill available VRAM, so the GPU always has real work to do.

**Scaling modes (auto-selected):**
```
GPUs ≤ num_layers:     1 GPU per layer, parallel across layers
GPUs > num_layers:     K = GPUs / num_layers proposals per layer
```

---

## Two Phases

### Phase 1: Verify (~1M params, fast)

Prove the full TFLE+CDLL+SWT stack works before burning GPU-hours.

**Model:** Ternary MLP, ~940K params:
```
[3072, 256, 256, 192, 128, 100]  →  786K + 65K + 49K + 25K + 13K = 938K params
```
**Fitness eval batch:** 512 examples. **Steps:** 50K. **Time:** minutes.
**Pass condition:** loss decreasing, accuracy >15% (random=1%), CDLL improving with depth.

### Phase 2: Saturate GPUs (~100M params)

Same architecture scaled up. This is where 5090s actually work.

**Model:** Ternary MLP, ~100M params:
```
[3072, 4096, 4096, 2048, 1024, 512, 100]
  3072×4096 = 12.6M
  4096×4096 = 16.8M
  4096×2048 = 8.4M
  ...
  TOTAL ≈ 100M params
```
**Fitness eval batch:** 4096 examples per proposal. At 100M params, each CDLL evaluation pushes ~400MB through each layer per proposal. With 2 proposals per layer across 2 GPUs, each GPU is doing real sustained matrix work.
**Steps:** 100K+. **Time:** hours.

**Memory at 100M:** weights (100MB) + dual traces (400MB) + activation buffers (~2GB at batch 4096) = ~2.5GB per GPU. Leaves 29GB headroom on each 5090 — so batch size can go even higher if utilization isn't saturated.

---

## Three Integrated Systems

### 1. TFLE — Weight Updates

Per layer, per step, per GPU in that layer's pool:
1. **Select candidates** — top 3% by error-trace, plus 0.3% random exploration
2. **Propose flips** — credit-biased direction (zero_gravity=0.5)
3. **Evaluate** — forward pass on FULL eval batch, compute CDLL fitness delta
4. **Return** proposal + fitness delta to layer coordinator

**Layer coordinator** picks best of K proposals, applies Boltzmann acceptance (init_temp=2.0, cosine cooling, reheat after 2000-step plateau). Broadcast winner. Update traces.

Depth scaling: deeper layers flip less (×0.8) and run cooler (×0.7).

### 2. CDLL — Fitness Function

Each layer's local fitness: `L = alpha * H(output) - beta * I(output; input)`

- **H(output):** histogram entropy of activations (32 bins). Lower = more compressed.
- **I(output; input):** trace of squared correlation matrix (variance-based MI proxy). Higher = more preserved.
- **alpha:** 0.3 (layer 0) → 0.8 (deepest hidden). Deeper = compress harder.
- **beta:** 1.0.

Each GPU evaluates CDLL on the SAME large batch but with DIFFERENT proposed flips. This is compute-heavy by design — big batch = accurate fitness AND full GPU utilization.

### 3. SWT — Sleep-Wake Schedule

**Wake (900 steps):** TFLE+CDLL with search-parallel. Each layer fills a replay buffer (1024 snapshots).

**Sleep (100 steps):** no new data, replay buffer only.
- Per-layer **micro-critic**: MLP (layer_width → 128 → 1, float32, trained with Adam lr=0.001)
- Critic learns to score real activations high, noise low
- TFLE flips evaluated by critic score instead of CDLL
- Sleep also runs search-parallel (K proposals per layer)
- Evict oldest 50% of replay buffer after sleep

---

## Implementation

**New files in `tfle/` project:**

| File | Purpose |
|---|---|
| `tfle/cdll_fitness.py` | `CDLLFitness` — entropy + MI on GPU, batch-parallel |
| `tfle/sleep_wake.py` | `MicroCritic`, `ReplayBuffer`, `SleepWakeScheduler` |
| `tfle/gpu_engine.py` | `SearchParallelEngine` — auto-detect GPUs, assign pools, coordinate proposals, broadcast, scale eval batch to fill VRAM |
| `experiments/full_system.py` | Runner: Phase 1 → Phase 2 pipeline with live dashboard |

**Extend existing:**
- `TFLELayer.to(device)` — move weights + traces to CUDA
- `TFLEConfig` — add `wake_steps`, `sleep_steps`, `critic_lr`, `fitness_eval_batch_size` (auto-scale to VRAM), `proposals_per_layer` (0=auto)

**SearchParallelEngine core loop:**
```python
engine = SearchParallelEngine(model, config)
# Auto: torch.cuda.device_count() GPUs
# Auto: fitness_eval_batch_size scaled to fill 80% of smallest GPU's free VRAM
# Each step:
#   1. Scatter eval batch to all GPUs
#   2. Each GPU: propose flips → forward pass on full batch → compute CDLL → return delta
#   3. Gather deltas, pick best, Boltzmann accept, broadcast
#   4. Update traces on primary GPU
```

**Dashboard:**
```
Phase 2 | GPUs: 2 | 100M params | Batch: 4096 | Pool: [2,2,2,1,1,1] proposals/layer
Step 5000 | Wake | Acc: 8.2% | CDLL: [-0.41,-0.33,-0.24,-0.17,-0.11,-0.06] | Temp: 1.71
  GPU0: 84% util | GPU1: 81% util | 3.2 steps/sec
Step 9000 | Sleep | Critics: [0.72,0.68,0.71,0.65,0.69,0.64] | Flips: 189/600
```

---

## Success Criteria

1. **Phase 1 passes** — loss decreasing, >15% accuracy at 1M params
2. **Phase 2 GPU saturation** — both 5090s >80% utilization sustained
3. **CDLL hierarchy** — deeper layers compress more (lower entropy, confirmed in logs)
4. **Sleep helps** — post-sleep accuracy ≥ pre-sleep
5. **Search-parallel advantage** — 2-GPU converges faster per-step than 1-GPU
6. **Phase 2 accuracy >20%** on CIFAR-100 at 100M params

After Phase 2 holds: scale to 1B+ params, same engine, same algorithm, more GPUs.
