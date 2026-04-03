# TFLE Proving Grounds — Key Findings

**Author:** James Camarota
**Date:** April 2, 2026
**Project:** NOVA — Gradient-Free Ternary Neural Network Training
**Source Data:** TFLE Proving Grounds experiments (RTX 4070 + 2x RTX 5090, ~5 hours)

---

## Overview

The TFLE Proving Grounds were the first-ever experiments testing gradient-free ternary training on language tasks. Prior to this, TFLE had only been validated on MNIST digit classification (70.2% accuracy, 530K-param MLP). These experiments answered three questions: can TFLE learn text, can it take over from backprop, and how does it compare?

---

## Finding 1: TFLE Converges on Text From Scratch

TFLE trained a 2.56M-parameter ternary MLP on next-character prediction (TinyShakespeare, 256-class byte-level output) and drove loss from 32.0 to 4.37, achieving perplexity 79. This is below random chance (ln(256) = 5.55) and was still improving at termination.

This is the first published demonstration of gradient-free ternary convergence on any language task. No backpropagation, no gradients, no optimizer state — just evolutionary search over ternary weight flips guided by task-loss fitness and temporal credit traces.

**Key progression across tuning runs:**

| Run | K | Batch | Steps | Final PPL | Hardware |
|-----|---|-------|-------|-----------|----------|
| v1 (initial) | 64 | 256 | 20K | 198 | 4070 |
| v2 (tuned) | 128 | 512 | 30K | 79 | 4070 |
| v3 (5090) | 256 | 1024 | 28K | 29.4* | 5090 |

*See Finding 5 for the asterisk on ppl=29.4.

**What this means for NOVA:** Path A (TFLE pretrain from scratch) is plausible. From-scratch training avoids the cascading distribution shift that kills handoff because all layers adapt together from random initialization.

---

## Finding 2: TFLE Is More Stable Than STE at Long Horizons

STE (backprop with straight-through gradient estimator for ternary weights) peaked at perplexity 39 at step 1,500, then degraded to 145 by step 20,000. TFLE steadily improved over 30,000 steps with no degradation. At termination, TFLE (ppl=79) was 1.8x better than STE's final (ppl=145).

**Root cause of STE instability:** The straight-through estimator approximates the gradient of a non-differentiable quantization function. This approximation has inherent bias. Over many steps, the bias accumulates — weight updates push parameters in slightly wrong directions, and ternary quantization amplifies these errors. The model learns, overshoots, and degrades.

**Why TFLE doesn't have this problem:** TFLE's accept/reject mechanism is unbiased by construction. At low temperature, it literally cannot accept a weight flip that worsens the model. There's no gradient approximation to accumulate errors from. Each step either improves the model or leaves it unchanged.

**Caveat:** The STE baseline was not properly tuned (no LR schedule, no gradient clipping, no weight decay). A fair comparison requires tuned STE — see TFLE_UPDATES.md for the planned experiment.

---

## Finding 3: TFLE Resists Overfitting Better Than STE

On code prediction (Stage 3, Python corpus, char-level, 2.56M params), the results were dramatic:

| Method | Train Loss | Val Loss | Val PPL |
|--------|-----------|---------|---------|
| TFLE | — | — | 131 |
| STE | 0.07 | 8.4 | 4,435 |

STE memorized the training set completely (train loss near zero, validation loss 120x higher). TFLE maintained generalization — 33x better validation perplexity.

**Why this happens:** STE uses exact gradients to minimize training loss. On small datasets, this leads directly to memorization. TFLE's random perturbation search cannot find precise memorization configurations because it explores the weight space stochastically. Each accepted flip improves average performance, not performance on any specific example.

**Why this matters for NOVA:** GRPO uses binary reward signals (code compiles, math checks out) from a limited set of verification examples. If the training algorithm memorizes those examples instead of learning the underlying skill, the model generates code that passes training tests but fails on new problems. TFLE's overfitting resistance makes it naturally suited for GRPO on code.

---

## Finding 4: Multi-Layer Handoff Fails, Output-Layer Works

Five attempts at multi-layer STE→TFLE handoff all failed. Output-layer-only works:

| Config | Degradation |
|--------|-------------|
| All-layer, K=128, no reeval | +122% |
| All-layer, K=16, no reeval | +56% |
| All-layer, K=1, no reeval | +128% |
| All-layer, frozen embedding | +72.6% |
| All-layer, K=16, with reeval | +17.9% |
| Layer-wise, K=16, reeval | +7.6% |
| **Output only, K=1, no reeval** | **-4.5%** |
| **Output only, K=32, reeval** | **-7.0%** |

**Root cause:** Cascading input distribution shifts. Flipping layer 0's weights changes its output distribution. Layer 1 was trained to expect the old distribution. Layer 1 degrades, corrupting layer 2's input, compounding through the network.

The output layer has no downstream layers, so flips directly change predictions without cascading.

**Key trend:** Re-evaluation reduces degradation significantly (+72% → +17.9%). Layer-wise training reduces it further (+17.9% → +7.6%). These techniques combined with the phased co-evolution approach (see TFLE_EMBEDDING_COEVOLUTION.md) may solve the handoff problem entirely.

---

## Finding 5: Embedding Learning Through Random Ternary Projections

The ppl=29.4 result on the 5090 requires careful interpretation. Re-evaluation rejected ALL ternary weight flips. The ternary hidden layers remained at random initialization. The perplexity came entirely from float32 embeddings (trained via backprop) learning to encode tokens in a way that produces accurate predictions through random ternary projections.

**Why this is interesting, not just an artifact:**

Random projection theory (Johnson-Lindenstrauss) tells us that random matrices approximately preserve distances. The embeddings exploited this: tokens appearing in similar contexts mapped to nearby points in embedding space, such that random ternary projections preserved enough structure for ppl=29.4.

**What this means:** The ppl=29.4 from embeddings-only establishes a floor. If TFLE's ternary flips are genuinely learning, performance should go BELOW 29.4. Any improvement past this floor is direct evidence that TFLE contributes real computation beyond random projections. This enables a phased co-evolution training approach (see TFLE_EMBEDDING_COEVOLUTION.md).

---

## Finding 6: Re-Evaluation Is the Critical Anti-Overfitting Fix

Elite re-evaluation (check accepted flips on a fresh held-out batch before finalizing) was the single most impactful technique:

| Experiment | Without Reeval | With Reeval | Improvement |
|------------|---------------|-------------|-------------|
| All-layer handoff | +72-128% | +17.9% | 4-7x less degradation |
| Output-only handoff | -4.5% | -7.0% | 1.6x more improvement |

**Current problem:** Threshold was too strict on the 5090 (rejected ALL flips). Needs calibrated tolerance with annealing — see TFLE_UPDATES.md.

---

## Finding 7: Larger Vocabularies Are Harder

Stage 2 (768-class BPE vocab) failed for both TFLE and STE. Neither meaningfully learned. TFLE was slightly better (loss 23.8 vs 29.5). The architecture is the bottleneck — an MLP without positional information can't handle subword prediction. Attention is needed.

---

## Finding 8: Tuning Matters Enormously

| Parameter Change | PPL Impact |
|-----------------|------------|
| K: 64 → 128, Batch: 256 → 512 | 198 → 79 (2.5x better) |
| K: 128 → 256, Batch: 512 → 1024 | 79 → 29.4* |

K (proposals per step) and batch size are the two most important hyperparameters. More proposals = broader search. Larger batch = more reliable fitness signal.

---

## Summary: Proven vs Unproven

**Proven:**
- TFLE converges on text from scratch (first ever)
- TFLE more stable than (untuned) STE at long horizons
- TFLE resists overfitting better than STE on small datasets
- Output-layer handoff works (+7% improvement with reeval)
- Re-evaluation prevents batch overfitting (4-7x reduction)
- Embeddings can learn through random ternary projections (ppl=29.4)

**Unproven (needs v2 experiments):**
- TFLE stability against properly-tuned STE
- TFLE with attention producing coherent text
- Layer-wise handoff reaching zero degradation
- Phased embedding-ternary co-evolution
- TFLE scaling beyond 2.56M params on text
- Overfitting resistance as general property (one dataset tested)

---

*All experiments in `tfle/tfle_proving/`. Next steps in TFLE_UPDATES.md.*
