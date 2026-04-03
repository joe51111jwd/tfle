# TFLE Proving Grounds v2 — Results

## Experiments
| Track | GPU | Status | Description |
|-------|-----|--------|-------------|
| A — Coherent Text | 0 | running (step 9K/40K) | AttentionLM + phased co-evolution |
| B — Tuned STE | 1 | COMPLETE | Proper STE baseline: ppl=12.0, word_rate=62% |
| C — Long Horizon MLP | 1 | COMPLETE | Tuned STE degraded to loss=14.9 at 100K steps |

## Key Result: Co-Evolution Works

The phased co-evolution strategy from EMBEDDING_COEVOLUTION.md is confirmed:

| Step | Phase | PPL | TFLE Flips | Verified | Notes |
|------|-------|-----|-----------|----------|-------|
| 0 | P1 | overflow | 0 | — | Random init |
| 2500 | P1 | 29.0 | 0 | — | Embedding-only floor |
| 3000 | P2 | 28.0 | 8 | 8/8 | **TFLE flips turn on** |
| 4000 | P2 | 19.7 | 8 | 8/8 | Breaking through floor |
| 6000 | P2 | 16.7 | 6 | 6/6 | Steady descent |
| 8000 | P2 | 15.7 | 4 | 4/4 | Approaching STE |
| 9000 | P2 | 15.5 | 5 | 5/5 | Latest checkpoint |

- **Embedding-only floor: ppl=29.0** (Phase 1, random ternary projections)
- **With TFLE: ppl=15.5** and still dropping (Phase 2, verified flips)
- **Tuned STE target: ppl=12.0**
- **100% re-eval pass rate** — every accepted flip confirmed on fresh data

## Tuned STE Baseline (Track B)

| Metric | Value |
|--------|-------|
| Final PPL | 12.0 |
| Best loss | 2.487 |
| Word recognition rate | 62.4% |
| Char entropy | 4.62 bits (English ~4.0-4.5) |
| Architecture | Same attention model (256 vocab, d=128, 2 layers) |
| Training | AdamW, lr=3e-4, cosine schedule, warmup=500, grad_clip=1.0 |

Sample: `ROMEO:\nDen, HUK:\nThanond Yof,\nThanses whio ang fr this me s d...`

## Long-Horizon STE MLP (Track C)

Tuned STE on MLP (100K steps) degraded from best loss ~3.0 to loss=14.9.
**STE instability confirmed even with proper tuning** (AdamW, cosine LR, grad clip).
This validates v1 finding: ternary quantization noise accumulates with backprop over long horizons.

## Architecture

```
AttentionLM (2-layer causal transformer):
  Ternary (TFLE):  Q,K,V,O projections + FF layers + output head = 425,984 params
  Float32 (backprop): token+positional embeddings + LayerNorm = 99,584 params
  Total: ~526K params
```

## What's Running

Track A continues on GPU 0 (RTX 5090). Phase 3 (K=64, full TFLE) starts at step 10K.
Expected completion: ~2.5 hours from last checkpoint.
At completion: ternary contribution verification (learned vs random weights comparison).
