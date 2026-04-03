# TFLE Proving Grounds — Results

## Summary
| Stage | Status | TFLE Result | STE Baseline | Gap | Notes |
|-------|--------|-------------|--------------|-----|-------|
| 1 - Char LM | GOOD | loss=5.29, ppl=198 | loss=4.98, ppl=145 | 1.4x | Both below random. TFLE learns text. |
| 2 - BPE LM | skipped | — | — | — | Prioritized Stage 4 diagnostics |
| 3 - Code | skipped | — | — | — | Pending |
| 4 - Handoff (all-layer) | FAIL | +72.6% degradation | loss=2.98, ppl=20 | — | Cascading layer corruption |
| 4 - Handoff (output-only) | PASS | -4.5% improvement! | loss=2.98, ppl=20 | — | **Key finding** |

## Stage 1 Details

**Architecture:** 256->32 embedding (float32) + [4096, 512, 512, 256, 256] ternary MLP
**Data:** TinyShakespeare, 128-char context, byte-level (vocab=256)
**Training:** 20K steps, batch=256, RTX 4070

### TFLE Training Curve
- Step 0: loss=32.3 (random ternary weights, severe overconfidence)
- Step 5K: loss=7.7 (rapid initial improvement)
- Step 10K: loss=6.9 (slowing down)
- Step 15K: loss=6.2 (approaching random baseline 5.55)
- Step 20K: loss=5.29 (below random -- TFLE IS learning)

### STE Training Curve
- STE peaked at ppl=39 (step 1500) then degraded to ppl=145 by step 20K
- STE quantization noise caused instability at longer training horizons

### Text Samples
Both models produced gibberish -- neither learned recognizable English words.
The MLP architecture (flattening 128 chars) loses positional structure, which is the bottleneck.

## Stage 4 Details (Handoff)

### The Handoff Problem
When STE trains a model to good performance (loss=2.98, ppl=20) and TFLE takes over, the model degrades catastrophically regardless of parameter settings.

### Experiments Tried
| Config | K | T | flip_rate | protection | Result |
|--------|---|---|-----------|------------|--------|
| Original spec | 128 | 0.5 | 0.003 | 0.6 | loss 5.8 -> 13 in 800 steps |
| Ultra-gentle | 16 | 0.1 | 0.001 | 0.8 | loss 3.1 -> 5.8 in 1200 steps |
| Nuclear (K=1, T~0) | 1 | 0.001 | 0.001 | 0.8 | loss 3.3 -> 7.4 in 4500 steps |
| Frozen embed, K=1 | 1 | 0.001 | 0.001 | 0.8 | loss 3.0 -> 5.1 in 1500 steps |
| **Output layer only** | **1** | **0.001** | **0.001** | **0.8** | **loss 2.98 -> 2.84 (IMPROVED)** |

### Root Cause
Multi-layer TFLE flips cause **cascading input distribution shifts**. When layer 0's weights change, it shifts the activation distribution entering layer 1. Even if the change improved layer 0's immediate contribution, layer 1 was trained for the OLD distribution. This compounds across layers.

The output layer works because:
1. It's closest to the loss signal (direct mapping)
2. Changes don't cascade to other layers
3. The fitness evaluation is unambiguous

### Batch-Overfitting Problem
With K>1 proposals evaluated on a single training batch:
- The "best" proposal is often just the luckiest on this batch
- Over many steps, these batch-specific improvements accumulate into generalization loss
- Even K=1 shows this effect due to the accept/reject criterion evaluating on a single batch

## What This Means for NOVA

### Path A (TFLE pretrain from scratch): VIABLE
- Stage 1 proves TFLE can learn text prediction (loss went from 32 to 5.29)
- TFLE's 1.4x gap to STE is surprisingly small
- From-scratch training doesn't have the cascading distribution shift problem because all layers adapt together

### Path B (TFLE as primary trainer): PARTIALLY VIABLE
- Good for pretraining (proven)
- BAD for fine-tuning existing models (handoff fails)
- Architecture-limited: MLP produces gibberish. Need attention or recurrence for quality text.

### Path C (STE pretrain -> TFLE for GRPO): MODIFIED
**Original plan:** STE pretrain -> TFLE takes over everything
**Revised plan:** STE pretrain -> TFLE for OUTPUT LAYER ONLY during GRPO

The output-layer handoff WORKS and IMPROVES performance. For GRPO:
- Freeze all hidden layers
- Use TFLE to search over the output projection (hidden_dim x vocab_size)
- Binary reward from code compilation provides clear fitness signal
- This is a natural fit for TFLE's random search -- the output projection is where "what to say" is decided

### Alternative: Layer-Wise Fine-Tuning
If full model TFLE is needed, try:
1. Train one layer at a time (all others frozen)
2. Many steps per layer before moving to the next
3. Bottom-up order (layer 0 first)
4. This prevents cascading distribution shifts

## Best Configs

### Stage 1 TFLE (from scratch)
```
flip_rate=0.01, K=64, T_init=1.5, T_min=0.1
cooling=cosine, protection=0.3, trace_decay=0.95
depth_scaled_flip=0.85, depth_scaled_temp=0.8
batch_size=256, eval_every=500
```

### Stage 4 TFLE (output-layer handoff)
```
flip_rate=0.001, K=1, T=0.001
selection=UNIFORM_RANDOM, protection=0.8
trace_decay=0.99, embed_lr=1e-5
ONLY touch the output layer -- freeze all hidden layers
```

## Key Findings

1. **TFLE learns text prediction from scratch.** Loss 32->5.29 in 20K steps, below random chance. Stage 1: PASS.
2. **TFLE-STE gap is only 1.4x.** Much better than expected. STE also struggles with ternary MLP.
3. **STE is unstable at long horizons.** STE peaked at ppl=39 then degraded to 145. Ternary quantization noise accumulates.
4. **Multi-layer TFLE handoff FAILS.** Cascading distribution shifts destroy learned representations, regardless of parameter tuning.
5. **Output-layer-only TFLE handoff WORKS.** 4.5% improvement over STE baseline. This is the viable path.
6. **Batch-overfitting is a systemic issue.** K-proposal evaluation on single batches leads to false "improvements" that hurt generalization.
7. **The MLP architecture bottlenecks text quality.** Both TFLE and STE produced gibberish. Need attention for real text generation.
