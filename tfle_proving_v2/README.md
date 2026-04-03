# TFLE Proving Grounds v2

Attention-based TFLE experiments with phased co-evolution training.

## What's Here

- `models/attention_lm.py` — Tiny causal transformer with ternary projections (~526K params)
- `tfle_v2/improved_trainer.py` — Phased co-evolution trainer (3 phases)
- `tfle_v2/validation_gate.py` — Calibrated re-evaluation gate
- `tfle_v2/antithetic.py` — Antithetic flip evaluation
- `tfle_v2/fitness_shaping.py` — NES rank-based utilities
- `baselines/tuned_ste.py` — Properly tuned STE baseline (AdamW, cosine LR, grad clip)
- `experiments/orchestrator.py` — Dual-GPU experiment launcher

## Results

See `RESULTS.md` for full experimental results.

## Quick Start

```bash
cd tfle/
python3 tfle_proving_v2/experiments/orchestrator.py
```

Requires 2x GPUs. GPU 0 runs Track A (co-evolution), GPU 1 runs Track B (tuned STE baseline).
