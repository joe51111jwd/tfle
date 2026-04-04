#!/usr/bin/env python3
"""Launch co-evolution (GPUs 0-1) and STE stability (GPUs 2-3) in parallel."""
import subprocess, sys, os

print("=" * 60)
print("LAUNCHING PARALLEL TESTS")
print("  Test 1: Co-Evolution on GPUs 0-1")
print("  Test 2: STE Stability on GPUs 2-3")
print("=" * 60)

env1 = {**os.environ, "CUDA_VISIBLE_DEVICES": "0,1"}
env2 = {**os.environ, "CUDA_VISIBLE_DEVICES": "2,3"}

p1 = subprocess.Popen(
    [sys.executable, "-u", "nova/test1_coevolution.py"],
    env=env1, cwd="/workspace/tfle",
    stdout=open("/workspace/tfle/nova/coevo_test1.log", "w"),
    stderr=subprocess.STDOUT,
)
print(f"  Test 1 PID: {p1.pid}")

p2 = subprocess.Popen(
    [sys.executable, "-u", "nova/test2_ste_stability.py"],
    env=env2, cwd="/workspace/tfle",
    stdout=open("/workspace/tfle/nova/coevo_test2.log", "w"),
    stderr=subprocess.STDOUT,
)
print(f"  Test 2 PID: {p2.pid}")

print("\nBoth running. Logs:")
print("  tail -f /workspace/tfle/nova/coevo_test1.log")
print("  tail -f /workspace/tfle/nova/coevo_test2.log")

r1 = p1.wait()
print(f"\nTest 1 finished (exit {r1})")
r2 = p2.wait()
print(f"Test 2 finished (exit {r2})")
print("DONE")
