# TFLE Updates — v2 Experiment Spec

**Author:** James Camarota
**Date:** April 2, 2026
**Project:** NOVA — Gradient-Free Ternary Neural Network Training
**Prerequisites:** Read TFLE_FINDINGS.md for context on what v1 proved and where it fell short.
**Hardware:** 2x RTX 5090 (32GB each, dedicated vast.ai instance)

---

## What This Document Is

A comprehensive spec for Claude Code (CC) to run the next round of TFLE experiments. Every improvement identified in the v1 analysis is here, with implementation details, pass/fail criteria, and GPU allocation. CC should build everything in `tfle/tfle_proving_v2/` without modifying any existing files in `tfle/` or `tfle/tfle_proving/`.

---

## GPU Allocation

**GPU 0 — Track A: Coherent Text**
Primary goal: get TFLE to produce recognizable English words using an attention-based model. This is the most important track.

**GPU 1 — Track B: Algorithmic Improvements**
Run 3-4 small experiments in parallel testing individual improvements against vanilla TFLE on the proven v1 MLP + Shakespeare setup. Winners get promoted to Track A.

**Overnight — Track C: Long-Horizon Stability**
200K-500K step comparison of best TFLE config vs properly-tuned STE. Runs on whichever GPU finishes first.

---

## Update 1: Re-Evaluation Threshold Calibration (CRITICAL — Do First)

### Problem

The 5090 run's re-eval gate rejected ALL ternary flips. The threshold was too strict — it required val_loss_after < val_loss_before with zero tolerance. Since individual flips produce tiny, noisy improvements that often fall within the noise margin on a different batch, everything was rejected.

### Solution

Tolerance-based threshold with cosine annealing:

```python
# Accept a flip if:
#   val_loss_after <= val_loss_before + tolerance
#
# tolerance schedule:
#   steps 0-5K:     tolerance = 0.1  (permissive — allow exploration)
#   steps 5K-20K:   tolerance = cosine_decay(0.1, 0.01)
#   steps 20K+:     tolerance = 0.01 (strict — only genuine improvements)
#
# This allows "neutral" flips early (which may enable later improvements)
# while becoming strict enough to prevent degradation in late training
```

Additionally, use a LARGER re-eval batch than the training batch:

```python
# Training batch: 512 examples
# Re-eval batch: 1024-2048 examples (more stable estimate)
# A flip that looks good on 512 training examples but bad on 2048 
# held-out examples is probably batch-overfitting
```

### Where to Implement

`tfle_v2/validation_gate.py` — standalone module that wraps the accept/reject decision.

### Test

Run the v1 tuned config (K=128, batch=512, 30K steps, Shakespeare) with the calibrated re-eval. Compare against:
- v1 without re-eval (ppl=79) — should be equal or better
- v1 with strict re-eval (ppl=29.4, zero ternary learning) — should show ternary flips being accepted

**Pass condition:** Some ternary flips accepted AND final ppl < 79.

---

## Update 2: Phased Embedding-Ternary Co-Evolution (CRITICAL — Track A)

### Problem

The ppl=29.4 result showed embeddings can learn through random ternary projections, but ternary layers contributed nothing. We need both to learn.

### Solution

Three-phase training schedule:

```python
# Phase 1: Embedding warmup (steps 0-5K)
#   - TFLE flips DISABLED (ternary layers frozen at random init)
#   - Embeddings train via backprop (lr=1e-3 with warmup)
#   - Purpose: embeddings learn to represent tokens through random ternary projections
#   - Expected: ppl drops to ~30 (matching the v3 5090 result)

# Phase 2: Gentle TFLE introduction (steps 5K-15K)  
#   - TFLE flips ENABLED with very conservative params:
#     flip_rate = 0.001 (touch very few weights)
#     K = 32 (modest search)
#     re-eval tolerance = 0.05 (permissive)
#   - Embedding lr decays to 1e-4
#   - Purpose: ternary layers start contributing real computation
#   - The key: embeddings continuously co-adapt to each accepted flip
#   - Expected: ppl drops below 30 (proving TFLE adds value over random)

# Phase 3: Full TFLE training (steps 15K+)
#   - Standard TFLE params: flip_rate=0.005, K=128
#   - Re-eval tolerance decays via cosine (0.05 → 0.01)
#   - Embedding lr decays to 1e-5
#   - Purpose: maximize ternary layer learning
#   - Expected: ppl continues improving, ternary layers carry real weight
```

### Why This Avoids the Handoff Problem

The handoff fails because STE creates rigid co-dependencies between layers, then TFLE breaks them. In co-evolution, the dependencies are NEVER rigid because:
- Ternary layers are always being slightly perturbed (TFLE flips)
- Embeddings are always slightly adjusting (backprop with small LR)
- The system is trained from the start to tolerate small ternary changes
- No sudden handoff — gradual introduction of TFLE over thousands of steps

### Where to Implement

`tfle_v2/improved_trainer.py` — the main training loop with phase scheduling.
`models/attention_lm.py` or `models/mlp_lm.py` — the model (try attention first, fall back to MLP).

### Test

Run on Shakespeare with the attention model (Update 3). Track these metrics at each phase transition:
- Number of ternary flips accepted per 1000 steps
- Ternary layer contribution: compare model output with learned ternary weights vs same model with re-randomized ternary weights. If ppl is lower with learned weights, TFLE is contributing.

**Pass condition:** ppl < 29.4 (beats the embeddings-only floor) with measurable ternary flip acceptance.

---

## Update 3: Attention Architecture (CRITICAL — Track A)

### Problem

The MLP flattens 128 characters into a single vector, destroying positional information. Both TFLE and STE produced gibberish regardless of loss. An architecture that preserves position is needed to produce coherent text.

### Solution

Tiny causal transformer:

```python
# Architecture:
#   vocab_size = 256 (byte-level)
#   d_model = 128
#   n_heads = 4 (32 dims per head)
#   n_layers = 2
#   d_ff = 512
#   context_len = 256
#   dropout = 0.0
#
# Ternary (trained by TFLE):
#   Q, K, V projections (128→128 each, per head) = 49K params × 2 blocks
#   Output projection (128→128) = 16K × 2 blocks
#   Feedforward (128→512, 512→128) = 131K × 2 blocks
#   Final output projection (128→256) = 32K
#   Total ternary: ~424K
#
# Float32 (trained by backprop or frozen):
#   Token embeddings: 256 × 128 = 32K
#   Positional embeddings: 256 × 128 = 32K
#   LayerNorm: ~512 params (4 LayerNorms × 128)
#   Total float: ~65K
#
# Grand total: ~490K params

# CRITICAL implementation details:
# 1. Causal mask: each position attends only to previous positions
# 2. Pre-LN: LayerNorm BEFORE attention and BEFORE feedforward
# 3. Each linear projection is a TFLELayer(in_features, out_features)
# 4. Attention computation (QK^T/sqrt(d), softmax, @V) has NO trainable params
# 5. LayerNorm params are float32, NOT ternary
```

### How TFLE Trains Attention

Each TFLE step:
1. Pick a layer (cycle through layers bottom-to-top)
2. Pick a projection within that layer (Q, K, V, O, FF1, FF2)
3. Propose K flip candidates using traces
4. For each candidate: run full forward pass through entire model, compute cross-entropy loss on all sequence positions
5. Accept/reject with re-eval gate
6. Move to next layer

This is the same task-loss fitness from v1, just applied to a model with attention instead of an MLP. The attention mechanism itself is parameter-free — TFLE only flips weights in the projection matrices.

### Where to Implement

`models/attention_lm.py` — the transformer model
`models/shared.py` — embeddings, positional encoding, causal mask

### Test

Train with phased co-evolution (Update 2) on Shakespeare, 30K+ steps.

**Pass/Fail:**

| Metric | FAIL | PASS | GOOD | GREAT |
|--------|------|------|------|-------|
| Loss | >4.5 | <4.0 | <3.5 | <3.0 |
| Perplexity | >90 | <55 | <33 | <20 |
| Text quality | Random bytes | Some real words | Mostly words | Coherent phrases |

**Text quality measurement:**

```python
# Generate 1000 chars from prompt "ROMEO:\n"
# Split on whitespace
# Check each token against English dictionary (top 10K words)
# Report: word_recognition_rate = real_words / total_tokens

# Also report character-level entropy:
#   Real English: ~4.0-4.5 bits/char
#   Random bytes: ~8.0 bits/char
#   Generated text entropy closer to 4.0 = more English-like
```

---

## Update 4: Tuned STE Baseline (Required for Credibility)

### Problem

The v1 STE baseline used lr=1e-3 with no scheduling — it peaked at ppl=39 then crashed to 145. TFLE "beating STE" was partly because STE was badly tuned.

### Solution

```python
# Properly tuned STE config:
optimizer = "AdamW"
learning_rate = 3e-4
lr_schedule = "cosine"      # decay to 1e-5
warmup_steps = 500          # linear warmup
weight_decay = 0.01         # L2 regularization
gradient_clip_norm = 1.0    # prevent gradient explosion
beta1 = 0.9
beta2 = 0.999
batch_size = 512            # match TFLE
eval_every = 500
```

### Where to Implement

`baselines/tuned_ste.py` — trainer with proper scheduling
`baselines/ste_config.py` — config dataclass

### Test

Run on the same architecture as Track A (attention model preferred, MLP fallback) for the same number of steps. The key comparison is the long-horizon stability test (Update 7).

---

## Update 5: Antithetic Flip Evaluation (Track B)

### Problem

Each proposed flip randomly picks a direction (e.g., 0→+1). Half the time the opposite (0→-1) would be better, wasting a step.

### Solution

```python
# For each candidate weight w at position (i,j):
#   current value: v
#   proposal A: flip to (v + 1) mod 3, mapped to {-1, 0, +1}
#   proposal B: flip to (v - 1) mod 3, mapped to {-1, 0, +1}
#   evaluate both via forward pass
#   keep whichever has lower loss
#
# For ternary {-1, 0, +1}:
#   If v = 0:  try +1 and -1
#   If v = +1: try 0 and -1
#   If v = -1: try 0 and +1
#
# This doubles forward passes per candidate but halves wasted steps
```

### Where to Implement

`tfle_v2/antithetic.py`

### Test

Run v1 MLP + Shakespeare setup, 20K steps. Control: vanilla TFLE (same K, batch, steps).

**Pass condition:** Lower final loss than vanilla at same step count, OR same final loss in fewer steps.

**Model size:** 2.56M params (same as v1). GPU 1, runs alongside other improvements.

---

## Update 6: Sparse Perturbation (Track B)

### Problem

Candidate selection uses error traces + random exploration. Research (Sparse MeZO) shows perturbing well-functioning weights wastes compute.

### Solution

```python
# Current selection: top 3% by error_trace + 0.3% random
# 
# New selection score:
#   score = error_trace - alpha * success_trace
#   where alpha = 0.5 (tunable)
#
# Interpretation:
#   High error_trace + low success_trace = "broken weight" → flip it
#   Low error_trace + high success_trace = "working weight" → protect it
#   High both = "controversial weight" → moderate priority
#   Low both = "inactive weight" → low priority
#
# Select top 3% by this combined score + 0.3% random
```

### Where to Implement

`tfle_v2/sparse_perturbation.py`

### Test

Same as Update 5: v1 MLP + Shakespeare, 20K steps, compare against vanilla.

**Pass condition:** Higher accuracy or lower loss at same step count.

---

## Update 7: Long-Horizon Stability Proof (Track C — Overnight)

### Problem

We claim TFLE is more stable than STE but only tested against an untuned STE for 20-30K steps.

### Solution

Run 200K-500K steps on the same architecture:
- **TFLE:** Best config from Track B improvements + co-evolution
- **STE:** Properly tuned (Update 4 config)
- **Architecture:** Attention model from Update 3 (or MLP fallback)
- **Data:** TinyShakespeare, same train/val split
- **Logging:** Every 500 steps — train loss, val loss, perplexity
- **Samples:** Every 5K steps — generate 500 chars from 5 prompts

### The Key Plot

```
X axis: training step (0 to 200K+)
Y axis: validation perplexity (log scale)
Blue line: TFLE
Red line: STE (tuned)

Looking for: the CROSSOVER POINT where TFLE permanently goes below STE
```

If TFLE crosses below tuned STE and stays there, that's the structural stability proof. If STE never degrades with proper tuning, TFLE's advantage is memory (1.1x vs 4x), not quality.

### Where to Implement

`experiments/track_c_long_horizon.py`

### Pass Condition

Either result is valuable:
- **TFLE crosses below tuned STE:** Publishable stability advantage
- **Tuned STE stays stable:** Means v1's instability was a tuning artifact; TFLE's advantage is memory efficiency

---

## Update 8: Overfitting Resistance Study (Track B/D)

### Problem

The 33x code overfitting result is one data point on one dataset. Need systematic evidence.

### Solution

Train both TFLE and STE on deliberately small datasets and measure generalization gap:

```python
# Dataset sizes: 100, 500, 1000, 5000, 10000 examples
# For each size:
#   1. Sample N examples from TinyShakespeare
#   2. Train TFLE for 20K steps
#   3. Train STE for 20K steps  
#   4. Measure: train_loss, val_loss, gap = val_loss - train_loss
#
# Plot: generalization_gap vs dataset_size for both methods
# Expected: TFLE gap stays small; STE gap explodes at small N
```

### Where to Implement

`experiments/overfitting_study.py`

### Pass Condition

TFLE maintains smaller generalization gap than STE across 3+ dataset sizes.

---

## Update 9: Rank-Based Fitness Shaping (Track B)

### Problem

Raw loss values are noisy and scale-dependent. A loss drop from 5.0→4.9 looks the same as 2.0→1.9 but the second is more meaningful.

### Solution

```python
# Given K proposals with losses [L_1, ..., L_K] + current loss L_0:
# 1. Rank all K+1 values (lower loss = better rank)
# 2. Compute NES-style utility:
#    u_i = max(0, log(K/2 + 1) - log(rank_i))
# 3. Normalize: u_i = u_i / sum(u)
# 4. Select the proposal with highest utility
# 5. Accept/reject using utility score, not raw loss delta
#
# This makes selection invariant to loss scale and eliminates outlier domination
```

### Where to Implement

`tfle_v2/fitness_shaping.py`

### Test

Same setup as Updates 5-6. Compare against vanilla.

**Pass condition:** Smoother loss curve and equal or better final loss.

---

## Update 10: Layer-Wise Sequential Handoff v2 (Track B)

### Problem

Multi-layer handoff degrades at +7.6% even with layer-wise training and re-eval.

### Solution

Enhanced layer-wise protocol with co-evolution:

```python
# 1. STE pretrain to convergence (loss ~3.0)
# 2. Freeze ALL layers
# 3. For sweep in range(3):  # 3 full sweeps
#     for layer_idx in range(num_layers):  # bottom to top
#       a. Unfreeze layer layer_idx only
#       b. Train TFLE on this layer for 2000 steps
#          - Re-eval gate active (tolerance 0.02)
#          - K=16 (small, focused search)
#          - Embedding continues co-adapting (lr=1e-5)
#       c. Freeze layer layer_idx
#     # After each sweep, evaluate full model
# 4. Final: output-layer-only fine-tuning for 2000 steps (proven to help)
```

### Why Multiple Sweeps

The first sweep adapts each layer to the original STE distribution below it. But adapting layer 0 changes the distribution for layer 1, which was adapted to the OLD layer 0. The second sweep re-adapts each layer to the post-first-sweep distributions. By sweep 3, the distributions should stabilize.

### Test

Run on the v1 architecture after STE pretrain. Compare degradation at each sweep.

**Pass condition:** Degradation decreases with each sweep. Ideal: <2% after 3 sweeps.

---

## Folder Structure

```
tfle/tfle_proving_v2/
├── README.md                         # Living results (update after each experiment)
├── models/
│   ├── attention_lm.py               # Tiny causal transformer (Update 3)
│   ├── mlp_lm.py                     # MLP from v1 (control)
│   └── shared.py                     # Embeddings, positional encoding, causal mask
├── data/
│   ├── loader.py                     # Text data loading, batching
│   └── tokenizer.py                  # Byte-level + optional BPE
├── tfle_v2/
│   ├── improved_trainer.py           # Phased co-evolution trainer (Update 2)
│   ├── validation_gate.py            # Calibrated re-eval (Update 1)
│   ├── antithetic.py                 # Antithetic flips (Update 5)
│   ├── sparse_perturbation.py        # Score-based selection (Update 6)
│   ├── fitness_shaping.py            # Rank-based utilities (Update 9)
│   └── fresh_batch.py                # Fresh batch per step
├── baselines/
│   ├── tuned_ste.py                  # Proper STE baseline (Update 4)
│   └── ste_config.py                 # STE hyperparameters
├── experiments/
│   ├── track_a_coherent_text.py      # GPU 0: attention + co-evolution
│   ├── track_b_improvements.py       # GPU 1: algorithmic improvements
│   ├── track_c_long_horizon.py       # Overnight: stability comparison
│   ├── overfitting_study.py          # Systematic overfitting test (Update 8)
│   ├── layerwise_handoff_v2.py       # Enhanced handoff (Update 10)
│   └── orchestrator.py               # Launches tracks on correct GPUs
├── results/                          # JSON + PNG plots (auto-populated)
└── checkpoints/                      # (auto-populated)
```

---

## Build Order for CC

### Phase 1: Foundation (do first, both GPUs idle until done)
1. Data loader (reuse v1, extend for longer context)
2. Validation gate with calibrated threshold (Update 1)
3. Attention model (Update 3)
4. Phased co-evolution trainer (Update 2)
5. Tuned STE baseline (Update 4)

### Phase 2: Launch parallel tracks
**GPU 0:** Track A — attention model + co-evolution + calibrated re-eval
**GPU 1:** Track B — antithetic, sparse perturbation, rank shaping, overfitting study (3-4 running simultaneously, each ~3GB VRAM)

### Phase 3: Combine and extend
- Promote Track B winners to Track A config
- Launch Track C overnight (long-horizon stability)
- If time: layer-wise handoff v2

### Code Rules
- Import from `tfle/` freely, never modify it
- Every experiment saves JSON results to `results/`
- Every experiment produces PNG loss curves
- Every text experiment generates samples at eval points
- Checkpoint every 5000 steps
- Print GPU utilization every 2000 steps

---

*Build it, tune it, run it overnight. The goal: coherent text from TFLE and a stability proof against tuned STE.*
