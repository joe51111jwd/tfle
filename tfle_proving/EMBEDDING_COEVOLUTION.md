# TFLE Embedding Co-Evolution Strategy

**Author:** James Camarota
**Date:** April 2, 2026
**Project:** NOVA — Gradient-Free Ternary Neural Network Training
**Context:** This document explains a training strategy discovered during the TFLE Proving Grounds experiments. Read TFLE_FINDINGS.md for the experimental results that led to this.

---

## The Discovery

During the TFLE v3 experiment on 2x RTX 5090, the re-evaluation gate was so strict that it rejected every single ternary weight flip over the entire 28K-step run. The ternary hidden layers stayed at their random initialization — completely untrained.

Despite this, the model achieved perplexity 29.4 on character-level Shakespeare. The float32 embedding layer (256 tokens → 32 dimensions, trained via backprop) learned to encode tokens in a way that produced accurate predictions when passed through random ternary projections.

This was initially treated as a failure (TFLE didn't learn) and an artifact (embeddings did all the work). But it reveals something genuinely useful about how embeddings and ternary layers interact, and it enables a new training strategy.

---

## Why Embeddings Can Learn Through Random Projections

### The Math

A random matrix R ∈ {-1, 0, +1}^{m×n} approximately preserves distances between points in high-dimensional space. This is related to the Johnson-Lindenstrauss lemma: for any set of points in high dimensions, there exists a projection to lower dimensions that approximately preserves all pairwise distances.

Ternary random matrices are particularly good at this because:
- Values are bounded ({-1, 0, +1}), preventing any single weight from dominating
- Zero weights act as natural feature selection (roughly 1/3 of weights are zero)
- The remaining ±1 weights compute sums and differences of input features

### What the Embeddings Learned

The embedding layer mapped each of the 256 byte tokens to a 32-dimensional vector. Through backprop training, it discovered an encoding where:

- Tokens that appear in similar contexts (e.g., lowercase letters in English words) were mapped to nearby points in embedding space
- Tokens that appear in different contexts (e.g., uppercase letters vs punctuation) were mapped to distant points
- The distances between embeddings were chosen such that the random ternary projection *preserved enough of this structure* to predict next tokens

In other words, the embeddings solved an optimization problem: "given these fixed random ternary weights, what token representations produce the best predictions?" The answer was a representation that encodes contextual similarity in a way that's robust to random linear transformations.

### Why ppl=29.4 Is Surprisingly Good

Random chance on 256-class prediction is perplexity 256 (loss = ln(256) = 5.55). Getting to 29.4 means the model is ~8.7x better than random, using embeddings alone with zero learned ternary computation. This suggests:

1. The embedding space has high capacity — 32 dimensions × 256 tokens = 8,192 learnable parameters is enough to encode substantial structure
2. Random ternary projections preserve more information than you'd expect
3. The bottleneck was never the ternary layers' values — it was the input representation

---

## The Co-Evolution Strategy

### Core Idea

Instead of training embeddings and ternary layers independently (embeddings via backprop, ternary via TFLE), train them as a co-evolving system where:

- **Embeddings adapt quickly** (gradient-based, small LR) to accommodate each ternary change
- **Ternary layers adapt slowly** (search-based, low flip rate) to improve on what random projections provide
- **Neither assumes the other is fixed** — the system is always in a state of mutual adaptation

### Why This Solves the Handoff Problem

The handoff fails because STE training creates rigid co-dependencies between layers. Embeddings learn representations optimized for specific ternary weight values. When TFLE flips those weights, the representations are wrong for the new values, and cascading distribution shifts destroy the model.

Co-evolution prevents this because:

1. **The system never has rigid dependencies.** From step 1, ternary weights are being perturbed (even if initially very gently). Embeddings learn to produce representations that work across a range of nearby ternary configurations, not just one specific configuration.

2. **Each ternary change is immediately compensated.** When a flip is accepted, the embedding layer (still training via backprop) adjusts within a few steps to accommodate the new ternary weight. The adjustment is small because only a few weights changed.

3. **No sudden handoff.** Instead of "train everything with STE, then switch to TFLE" (which creates a discontinuity), co-evolution is a smooth continuum from "mostly embedding learning" to "mostly ternary learning."

### The Three Phases

```
Phase 1: Embedding Warmup (steps 0 to 5K)
├── TFLE flips: DISABLED
├── Embedding LR: 1e-3 (with warmup)
├── What happens: Embeddings learn to represent tokens through random ternary projections
├── Expected perplexity: ~30 (matching the v3 5090 result)
└── Purpose: Establish a strong embedding foundation before introducing perturbations

Phase 2: Gentle TFLE Introduction (steps 5K to 15K)
├── TFLE flips: ENABLED (conservative)
│   ├── flip_rate: 0.001 (1/5 of normal)
│   ├── K: 32 (1/4 of normal)
│   └── re-eval tolerance: 0.05 (permissive)
├── Embedding LR: decays from 1e-3 to 1e-4
├── What happens: Ternary layers start contributing real computation
│   Each accepted flip slightly changes the ternary projections
│   Embeddings continuously adjust to the new projections
│   The system learns that ternary weights CAN change
├── Expected perplexity: drops below 30 (proving TFLE adds value)
└── Purpose: Break past the embeddings-only ceiling

Phase 3: Full TFLE Training (steps 15K onward)
├── TFLE flips: STANDARD
│   ├── flip_rate: 0.005
│   ├── K: 128
│   └── re-eval tolerance: cosine decay 0.05 → 0.01
├── Embedding LR: decays to 1e-5
├── What happens: Ternary layers carry substantial learned computation
│   Embeddings are fine-tuning rather than learning from scratch
│   The model's performance comes from both embeddings AND ternary layers
├── Expected perplexity: continues improving, target < 20
└── Purpose: Maximize the contribution of gradient-free ternary training
```

### How to Verify TFLE Is Actually Contributing

The ppl=29.4 floor from Phase 1 provides a natural control. At any point during Phase 2 or 3, you can check:

```python
# Test 1: Learned vs Random ternary weights
# Save current ternary weights
# Re-randomize all ternary weights (keep embeddings)
# Measure perplexity with random ternary weights
# Compare against perplexity with learned ternary weights
#
# If learned < random: TFLE is contributing
# If learned ≈ random: TFLE hasn't learned anything useful
# The gap between them measures TFLE's contribution

# Test 2: Flip acceptance rate
# Track how many flips are accepted per 1000 steps
# Phase 1: 0 (disabled)
# Phase 2: should be > 0 and gradually increasing
# Phase 3: should stabilize at 10-30%
# If acceptance is 0 in Phase 2/3: threshold is still too strict

# Test 3: Per-layer contribution
# For each ternary layer, replace its weights with random
# Measure perplexity impact
# Layers where randomization hurts most are contributing most
```

---

## How This Fits Into the NOVA Pipeline

### Current NOVA Training Pipeline (from NOVA_BUILD_PLAN.md)

```
1. STE pretrain on FineWeb-Edu + StarCoder
2. Distill reasoning from DeepSeek-R1
3. GRPO post-training for math/code reasoning
4. SWT continuous learning after deployment
```

### Where Co-Evolution Applies

**For NOVA-10M (74.5M params, currently training):**

The NOVA-10M STE pretrain is running right now on 4x RTX 5090. When it finishes, the co-evolution strategy provides a clean handoff path:

1. STE pretrain produces a model with trained embeddings + trained ternary weights
2. Instead of immediate TFLE handoff (which fails), use a modified Phase 2:
   - Keep the STE-trained embeddings (don't reset them)
   - Keep the STE-trained ternary weights (don't randomize them)
   - Start TFLE with very conservative params (flip_rate=0.001, K=16)
   - Let embeddings co-adapt at lr=1e-5
   - Gradually increase TFLE aggressiveness over 10K+ steps
3. This is NOT the same as the failed handoff experiments because:
   - The embedding layer continues adjusting (it was frozen in the failed experiments)
   - TFLE params start much gentler and ramp up (vs immediate standard TFLE)
   - Re-evaluation gate prevents batch-overfitting

**For NOVA-2.4B (future):**

Two options enabled by co-evolution:

**Option A — Pure TFLE pretrain with co-evolution:**
If the proving grounds show co-evolution produces coherent text at 2.56M params, scale it up. The memory advantage (1.1x vs 4x) means NOVA-2.4B could train on a single A100-80GB with TFLE where STE needs 4x A100s.

Phase 1 (embedding warmup) would be short relative to total training — maybe 1% of total steps. Phase 2 (gentle TFLE) would be another 5-10%. Phase 3 (full TFLE) would be the remaining 90%+.

**Option B — STE pretrain → co-evolution handoff → TFLE for GRPO:**
This is the safer path. STE does the heavy lifting of pretraining (proven technology). Co-evolution handles the transition to TFLE for the GRPO phase. The output-layer-only handoff (-7.0% improvement) is already proven. With co-evolution, multi-layer handoff may also become viable.

**For GRPO specifically:**

GRPO's binary reward signal ("code compiles" / "math is correct") is a natural TFLE fitness function. The co-evolution approach is particularly well-suited:

1. Start from STE-pretrained + distilled model (has knowledge and reasoning)
2. Phase 2: gently introduce TFLE on output projection with compiler rewards
3. Phase 3: expand to more layers if layer-wise handoff is proven
4. Embeddings continuously co-adapt to GRPO-driven ternary changes

The overfitting resistance finding (Finding 3) is especially relevant here — TFLE-based GRPO should generalize from limited test cases better than backprop-based GRPO.

**For SWT (Sleep-Wake Training):**

SWT's continuous learning after deployment is the ultimate application of co-evolution. The model:
- Receives task feedback (wake phase: did the code work?)
- Uses TFLE to flip weights based on this feedback
- Embeddings adjust to maintain coherence
- Sleep phase consolidates via replay buffer

This is co-evolution running indefinitely — exactly the "system trained to tolerate small ternary changes" that co-evolution produces.

---

## Connection to Research Literature

### Evolution Strategies at Scale (2025)

The EGGROLL paper showed that evolution strategies with rank-1 perturbations can train 14B-parameter LLMs, outperforming GRPO on reasoning tasks. Their key technique — antithetic sampling with centered rank transformation — is complementary to co-evolution. EGGROLL perturbs all weights simultaneously; co-evolution perturbs ternary weights while embeddings co-adapt.

### MeZO: Forward-Pass-Only Training (2023)

MeZO showed that zeroth-order optimization (no gradients, only forward passes) can fine-tune LLMs to within 1% of Adam on most tasks. MeZO uses full-parameter perturbation; TFLE uses targeted ternary flips. Both share the property that training memory ≈ inference memory. Co-evolution extends this by hybridizing gradient-based (embeddings) and gradient-free (ternary layers) optimization.

### Forward-Forward Algorithm Variants (2024-2026)

Mono-Forward, SCFF, and DeeperForward all use local learning objectives without backpropagation through the full network. Co-evolution is similar in spirit — embeddings use gradients locally while ternary layers use a non-gradient method. The key difference: co-evolution keeps task-loss as the objective (proven to work in TFLE), rather than local goodness/contrastive objectives (which failed in TFLE's early experiments).

---

## Summary

The embedding co-evolution strategy turns an apparent failure (TFLE v3 didn't learn any ternary weights) into a training methodology:

1. **Phase 1** lets embeddings find good representations through random ternary projections (ppl=29.4, proven)
2. **Phase 2** gently introduces TFLE flips while embeddings co-adapt (untested, expected to break past 29.4)
3. **Phase 3** runs full TFLE training with both embeddings and ternary layers contributing (untested, expected to reach <20 ppl)

The strategy avoids the handoff problem by never creating rigid layer co-dependencies, solves the re-eval threshold problem by establishing a measurable floor (29.4) against which TFLE's contribution can be tested, and provides a clean path from the current NOVA-10M STE pretrain to TFLE-based GRPO and continuous learning.

---

*Next step: implement and test in TFLE_UPDATES.md experiments.*
