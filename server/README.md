# TFLE Server Run — vast.ai 2x RTX 5090

## Quick Start

SSH into your vast.ai instance, then:

```bash
# 1. Setup (1 min)
cd /workspace
git clone https://github.com/joe51111jwd/tfle.git
cd tfle
pip install torch torchvision tqdm

# 2. Run (est. 30-60 min on 5090)
python server/run_100k.py

# 3. Or run in background
nohup python server/run_100k.py > run.log 2>&1 &
tail -f run.log
```

## What it does

- Trains TFLE with **task-loss fitness** (the fix) for 100K steps on MNIST
- Uses cosine temperature schedule (0.08 → 0.001)
- Runs STE baseline for comparison
- Saves results to `results_100k.json`
- Saves model checkpoint to `checkpoints/tfle_100k_final.pt`

## Important

The repo on GitHub may not have the task-loss fix yet. If `FitnessType.TASK_LOSS` 
doesn't exist, you need to push the local changes first:

```bash
# On your Mac, before renting the server:
cd ~/Projects/tfle
git add -A
git commit -m "Add task-loss fitness — first gradient-free ternary convergence"
git push
```

## Download results after

```bash
# From your Mac:
scp root@<vast-ip>:/workspace/tfle/results_100k.json ~/Projects/tfle/
scp root@<vast-ip>:/workspace/tfle/checkpoints/tfle_100k_final.pt ~/Projects/tfle/checkpoints/
```

## Expected results

| Metric | Previous (20K steps) | Target (100K steps) |
|--------|---------------------|---------------------|
| TFLE accuracy | 23.54% | >85% |
| STE baseline | 89.28% | ~95% |
| Gap | 65.7pp | <10pp |
