# CLAUDE.md — TFLE / NOVA Project

## CRITICAL: Read Before Doing Anything

**Read the full context first:** `~/Desktop/TFLE/context/SESSION_CONTEXT_April1_3.md`

**Read ALL files in:** `~/Desktop/TFLE/context/` — these contain every decision, experiment, failure, and discovery from development. Do not duplicate work. Do not repeat experiments.

**Read the gameplan:** `~/Desktop/TFLE/nova_gameplan.docx` — this is the original vision document. All decisions should be checked against it.

## Desktop File Organization (`~/Desktop/TFLE/`)

```
context/    — Session narratives, strategy docs, decision logs
              WHY things are the way they are. Written prose.
              Examples: SESSION_CONTEXT_*.md, NOVA_STATUS_AND_STRATEGY_*.md

results/    — Data, numbers, and reports with concrete metrics
              WHAT happened. JSON data files, benchmark results,
              test reports with tables/numbers/configs.
              Examples: best_config.json, results_phase1.json,
              NOVA_4x5090_RESULTS_April3.md

data/       — Raw datasets, training data files
experiments/— Experiment scripts (if not in the repo)
```

**Rule of thumb:** If it has training curves, configs, loss numbers, or GPU stats → `results/`. If it explains decisions, strategy, or session history → `context/`.

## What This Project Is

NOVA is a 2.4B parameter ternary neural network ({-1, 0, +1} weights) with a hybrid Transformer-Mamba architecture, trained gradient-free via TFLE (Trit-Flip Local Evolution). The goal: beat 10B-class models on reasoning and code tasks while fitting in 0.4GB and running on consumer hardware.

Three original algorithms form the stack:
- **TFLE** — gradient-free weight updates via evolutionary flip-and-evaluate
- **CDLL** — compression-driven layer learning (Information Bottleneck fitness)
- **SWT** — sleep-wake continual learning after deployment

## What's Proven (Do NOT Re-Test)

- TFLE works: 70% on MNIST, first-ever gradient-free ternary convergence
- Task-loss fitness is the engine (0.85 * task_loss + 0.15 * cdll_score)
- CDLL alone cannot classify — it's a 5% regularizer only
- LayerNorm after every ReLU is essential (fixes activation explosion → NaN)
- Layer-wise cycling is mandatory for models >1M params (flip one layer per step)
- K scales with model size: K=128 (530K) → K=384 (1.6M) → K=512+ (10M+)
- Cosine temperature annealing with reheat outperforms exponential
- Temporal credit traces outperform random selection
- NOVA-10M (74.5M params) architecture works and trains on real text (WikiText-103, loss 10.5→4.55)
- 4x RTX 5090 at 99% utilization with DataParallel

## What's NOT Proven (Needs Work)

- STE→TFLE handoff: DEGRADED from 95%→68%. Needs much gentler params.
- TFLE on language: all TFLE results are MNIST. Never tested on text.
- Conv model on CIFAR: exists but untested.
- Distillation + GRPO pipelines: code exists, not tested end-to-end.
- SWT in practice: built but not tested with real task sequences.
- Inference strategies: none tested on actual model.

## Key Commands

```bash
# Install
pip install torch torchvision tqdm pyyaml datasets tokenizers

# Run TFLE on MNIST (proven, 70%)
python experiments/phase1_tuned.py

# Run full system (CDLL + SWT + SearchParallelEngine)
python experiments/full_system.py

# Run on server (vast.ai)
python nova10m_pretrain.py  # or resume_4gpu.py for multi-GPU
```

## Architecture

```
NOVA-10M (74.5M params):
  12 layers: 9 Mamba + 3 Attention (MMMA MMMA MMM A)
  Hidden: 640, Vocab: 32K
  MoLoRA: 5 experts (math, code, planning, self_eval, tool_use), top-2
  All linear layers: BitLinear (ternary via absmean quantization)
  Activation: ReLU + LayerNorm (NOT ReLU² — that's for full 2.4B only)

NOVA-2.4B (target):
  32 layers: 24 Mamba + 8 Attention
  Hidden: 2560, Vocab: 128,256
  Same MoLoRA, BitLinear, ReLU² activation
```

## Fitness Function (Use This Always)

```python
fitness = 0.85 * task_loss_delta + 0.15 * cdll_score
```

- task_loss_delta = change in cross-entropy after flip (full model forward pass)
- cdll_score = compression metric (entropy + MI + class separation)
- Keep LayerNorm after every ReLU
- Local heads are monitoring only — don't use in fitness

## DO NOT Touch

- Ternary constraint ({-1, 0, +1})
- Trace system (success_traces, error_traces, selection formula)
- Annealing schedule (cosine with reheat)
- Accept/reject Boltzmann logic
- LayerNorm placement
- Best config values (cdll_w: 0.05, temp: 0.2, flip: 0.02, decay: 0.95, K: 128)

## Benchmark Targets (2.4B)

| Benchmark | Target | Rationale |
|-----------|--------|-----------|
| MATH-500 | 90%+ | DeepSeek-R1-Distill-1.5B hit 83.9% with less |
| GSM8K | 95%+ | Distillation + GRPO + 16-sample voting |
| HumanEval+ | 75%+ | Execution verify + retry |
| AIME 2024 | 45%+ | Tree search with ternary cost advantage |
| MMLU | 60-65% | Limited by training data (200B tokens) |

## File Structure

```
tfle/               — Core TFLE algorithm (PyTorch)
  tfle/             — layers, model, config, cdll, swt, local_heads, gpu_engine
  tests/            — 14 test files, 124+ tests
  experiments/      — MNIST, CIFAR, full system experiments
  server/           — vast.ai training scripts
  docs/             — specs and outlines

nova/               — Full NOVA system
  model_pt/         — PyTorch model (bitlinear, mamba, attention, hybrid, molora)
  training_pt/      — Training (tfle, cdll, swt, pretrain, distill, grpo, curiosity, competence)
  cuda/             — Custom CUDA kernels (ternary_matmul, absmax_quantize, trace_update)
```

## Git

- Repo: github.com/joe51111jwd/tfle
- Author: james camarota <atwbusinessjames@gmail.com>
- Always commit with meaningful messages explaining WHY
- Never commit credentials or API keys

## Server Access (vast.ai)

Instances change. Check the latest SSH string with the user. Previous instances:
- 2x 5090: ssh -p 53502 root@108.255.76.60 (may be expired)
- 4x 5090: ssh -p 18807 root@ssh9.vast.ai (may be expired)
- Always use: `-i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no`

## Priority Order (What To Do Next)

1. Fix STE→TFLE handoff (flip_rate=0.001, no Boltzmann on regressions)
2. Test TFLE on language (NOVA-10M on WikiText — the make-or-break experiment)
3. Phase 2: Reasoning distillation on pretrained NOVA-10M
4. Phase 3: Dr. GRPO with binary rewards
5. Fix overfitting (more data or regularization)
6. Scale to 2.4B (only after all above pass)
