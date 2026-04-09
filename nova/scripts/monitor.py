#!/usr/bin/env python3
"""
Live training monitor for NOVA Phase-2.
=======================================
Tails the training log produced by ``phase2_train_student.py`` and prints
a color-coded status line on each new ``[EVAL ...]`` entry. Also queries
nvidia-smi for live GPU utilization and tracks running cost against the
budget cap.

Usage:
    # continuous tail
    python nova/scripts/monitor.py --log_file /data/checkpoints/training.log

    # one-shot summary
    python nova/scripts/monitor.py --log_file /data/checkpoints/training.log --summary

Threshold colors:
    green   OK
    yellow  warning (throughput < 50K, GPU util < 80, gap > 0.30)
    red     critical (NaN/Inf, gap > 0.50, cost > 80% of budget)
"""
from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── Constants ───────────────────────────────────────────────────────

TOTAL_TOKENS: int = 18_000_000_000
THROUGHPUT_WARN_TOKPS: int = 50_000
GPU_UTIL_WARN_PCT: int = 80
GAP_WARN: float = 0.30
GAP_STOP: float = 0.50
COST_PER_HOUR_USD: float = 15.60
BUDGET_CAP_USD: float = 1000.0
TAIL_SLEEP_S: float = 0.5


# ── ANSI color helpers ──────────────────────────────────────────────


class Color:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"


def _use_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def colorize(text: str, color: str) -> str:
    if not _use_color():
        return text
    return f"{color}{text}{Color.RESET}"


def green(text: str) -> str:
    return colorize(text, Color.GREEN)


def yellow(text: str) -> str:
    return colorize(text, Color.YELLOW)


def red(text: str) -> str:
    return colorize(text, Color.RED)


def dim(text: str) -> str:
    return colorize(text, Color.DIM)


def bold(text: str) -> str:
    return colorize(text, Color.BOLD)


# ── Parsing ─────────────────────────────────────────────────────────


EVAL_RE = re.compile(
    r"\[EVAL\s+(?P<tokens>\d+)M\]\s+"
    r"train_loss=(?P<train>[-+]?\d+(?:\.\d+)?)\s+"
    r"eval_loss=(?P<eval>[-+]?\d+(?:\.\d+)?)\s+"
    r"gap=(?P<gap>[-+]?\d+(?:\.\d+)?)\s+"
    r"temp=(?P<temp>[-+]?\d+(?:\.\d+)?)\s+"
    r"seq_len=(?P<seq>\d+)\s+"
    r"tok/s=(?P<tps>\d+)K"
)

CRITICAL_RE = re.compile(r"CRITICAL", re.IGNORECASE)
NAN_RE = re.compile(r"(Non-finite loss|NaN|Inf)")
WARN_RE = re.compile(r"WARNING")


@dataclass
class MonitorState:
    tokens: int = 0
    train_loss: float = float("nan")
    eval_loss: float = float("nan")
    gap: float = float("nan")
    temperature: float = float("nan")
    seq_len: int = 0
    tok_s: int = 0
    start_time: float = field(default_factory=time.time)
    last_eval_time: float = field(default_factory=time.time)
    evals_seen: int = 0
    train_loss_history: list[float] = field(default_factory=list)
    eval_loss_history: list[float] = field(default_factory=list)
    nan_count: int = 0
    warn_count: int = 0

    def running_avg(self, values: list[float], n: int = 5) -> float:
        if not values:
            return float("nan")
        window = values[-n:]
        return sum(window) / len(window)


def parse_eval_line(line: str) -> Optional[dict]:
    m = EVAL_RE.search(line)
    if not m:
        return None
    return {
        "tokens": int(m.group("tokens")) * 1_000_000,
        "train_loss": float(m.group("train")),
        "eval_loss": float(m.group("eval")),
        "gap": float(m.group("gap")),
        "temperature": float(m.group("temp")),
        "seq_len": int(m.group("seq")),
        "tok_s": int(m.group("tps")) * 1000,
    }


# ── nvidia-smi ──────────────────────────────────────────────────────


def query_gpu_util() -> Optional[int]:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        utils = [int(x.strip()) for x in r.stdout.strip().split("\n") if x.strip()]
        if not utils:
            return None
        return sum(utils) // len(utils)
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return None


# ── Formatting ──────────────────────────────────────────────────────


def format_loss(name: str, value: float, history: list[float]) -> str:
    if value != value:  # NaN
        return f"{name}=n/a"
    avg = sum(history[-5:]) / max(len(history[-5:]), 1) if history else value
    label = f"{name}={value:.3f} (avg {avg:.3f})"
    return label


def format_gap(gap: float) -> str:
    if gap != gap:
        return dim("gap=n/a")
    text = f"gap={gap:+.3f}"
    if gap > GAP_STOP:
        return red(bold(text + " CRITICAL"))
    if gap > GAP_WARN:
        return yellow(text + " warn")
    return green(text)


def format_throughput(tok_s: int) -> str:
    text = f"{tok_s // 1000}K tok/s"
    if tok_s == 0:
        return dim(text)
    if tok_s < THROUGHPUT_WARN_TOKPS:
        return yellow(text + " slow")
    return green(text)


def format_gpu_util(util: Optional[int]) -> str:
    if util is None:
        return dim("GPU n/a")
    text = f"GPU {util}%"
    if util < GPU_UTIL_WARN_PCT:
        return yellow(text + " low")
    return green(text)


def format_cost(elapsed_h: float) -> str:
    cost = COST_PER_HOUR_USD * elapsed_h
    pct = cost / BUDGET_CAP_USD
    text = f"${cost:.2f}/{BUDGET_CAP_USD:.0f}"
    if pct > 0.80:
        return red(text + " !")
    if pct > 0.50:
        return yellow(text)
    return green(text)


def format_eta(state: MonitorState) -> str:
    if state.tok_s <= 0 or state.tokens <= 0:
        return dim("ETA n/a")
    remaining = max(TOTAL_TOKENS - state.tokens, 0)
    eta_s = remaining / max(state.tok_s, 1)
    hours = eta_s / 3600.0
    return green(f"ETA {hours:.1f}h")


def format_budget_eta(elapsed_h: float) -> str:
    if elapsed_h <= 0.001:
        return dim("budget n/a")
    burn = COST_PER_HOUR_USD
    remaining = BUDGET_CAP_USD - burn * elapsed_h
    if remaining <= 0:
        return red("BUDGET OVER")
    h_left = remaining / burn
    return green(f"budget {h_left:.1f}h left")


def format_progress(tokens: int) -> str:
    pct = tokens / max(TOTAL_TOKENS, 1)
    bars = 20
    filled = int(bars * pct)
    bar = "#" * filled + "-" * (bars - filled)
    return f"[{bar}] {tokens / 1e9:.2f}B/{TOTAL_TOKENS / 1e9:.0f}B ({pct:.1%})"


def render_status(state: MonitorState) -> str:
    elapsed_h = (time.time() - state.start_time) / 3600.0
    util = query_gpu_util()

    lines = [
        bold("NOVA Phase-2 Monitor"),
        format_progress(state.tokens),
        "  " + format_loss("train_loss", state.train_loss, state.train_loss_history)
        + "  " + format_loss("eval_loss", state.eval_loss, state.eval_loss_history)
        + "  " + format_gap(state.gap),
        "  " + format_throughput(state.tok_s)
        + "  " + format_gpu_util(util)
        + "  " + (
            f"temp={state.temperature:.1f}  seq_len={state.seq_len}"
            if state.temperature == state.temperature
            else dim("temp=n/a  seq_len=n/a")
        ),
        "  elapsed="
        + f"{elapsed_h:.2f}h  "
        + format_cost(elapsed_h)
        + "  "
        + format_eta(state)
        + "  "
        + format_budget_eta(elapsed_h),
    ]
    if state.nan_count or state.warn_count:
        flags = []
        if state.nan_count:
            flags.append(red(f"nan={state.nan_count}"))
        if state.warn_count:
            flags.append(yellow(f"warn={state.warn_count}"))
        lines.append("  " + "  ".join(flags))
    return "\n".join(lines)


# ── Tail driver ─────────────────────────────────────────────────────


def update_state_from_line(state: MonitorState, line: str) -> bool:
    """Update monitor state from a single log line. Returns True if interesting."""
    interesting = False

    parsed = parse_eval_line(line)
    if parsed is not None:
        state.tokens = parsed["tokens"]
        state.train_loss = parsed["train_loss"]
        state.eval_loss = parsed["eval_loss"]
        state.gap = parsed["gap"]
        state.temperature = parsed["temperature"]
        state.seq_len = parsed["seq_len"]
        state.tok_s = parsed["tok_s"]
        state.train_loss_history.append(parsed["train_loss"])
        state.eval_loss_history.append(parsed["eval_loss"])
        state.evals_seen += 1
        state.last_eval_time = time.time()
        interesting = True

    if NAN_RE.search(line):
        state.nan_count += 1
        interesting = True
    if CRITICAL_RE.search(line):
        state.nan_count += 1
        interesting = True
    if WARN_RE.search(line):
        state.warn_count += 1

    return interesting


def tail_follow(log_path: Path, poll_s: float = TAIL_SLEEP_S):
    """Generator that yields new lines appended to ``log_path`` (like tail -f).

    Handles the file not yet existing and log rotation (inode changes).
    """
    pos = 0
    inode = None
    fh = None
    while True:
        try:
            if not log_path.exists():
                time.sleep(poll_s)
                continue
            stat = log_path.stat()
            if fh is None or stat.st_ino != inode:
                if fh is not None:
                    fh.close()
                fh = log_path.open("r", encoding="utf-8", errors="replace")
                inode = stat.st_ino
                fh.seek(0, os.SEEK_END)
                pos = fh.tell()

            if stat.st_size < pos:
                fh.seek(0)
                pos = 0

            fh.seek(pos)
            new = fh.readline()
            while new:
                pos = fh.tell()
                yield new.rstrip("\n")
                new = fh.readline()

            time.sleep(poll_s)
        except (FileNotFoundError, OSError):
            time.sleep(poll_s)


def clear_screen() -> None:
    if _use_color():
        sys.stdout.write("\033[H\033[2J")
    else:
        sys.stdout.write("\n" * 2)
    sys.stdout.flush()


def print_status(state: MonitorState, clear: bool = True) -> None:
    if clear:
        clear_screen()
    sys.stdout.write(render_status(state))
    sys.stdout.write("\n")
    sys.stdout.flush()


# ── Summary mode ────────────────────────────────────────────────────


def run_summary(log_path: Path) -> int:
    if not log_path.exists():
        print(red(f"Log file not found: {log_path}"))
        return 1

    state = MonitorState()
    with log_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            update_state_from_line(state, line)

    print(render_status(state))
    return 0


# ── Follow mode ─────────────────────────────────────────────────────


def run_follow(log_path: Path) -> int:
    print(bold(f"NOVA monitor tailing {log_path}"))
    print(dim("press Ctrl-C to exit"))
    print()

    state = MonitorState()

    # Seed state from whatever already exists in the log so we don't start
    # empty if the training job is already running.
    if log_path.exists():
        with log_path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                update_state_from_line(state, line)
        print_status(state, clear=False)

    def _shutdown(signum, frame):  # noqa: ARG001
        print()
        print(dim("monitor stopped"))
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    last_refresh = time.time()
    for line in tail_follow(log_path):
        if update_state_from_line(state, line):
            print_status(state)
            last_refresh = time.time()
        elif time.time() - last_refresh > 10.0:
            print_status(state)
            last_refresh = time.time()

    return 0


# ── CLI ─────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live training monitor for NOVA Phase-2.")
    p.add_argument("--log_file", type=str, required=True)
    p.add_argument("--summary", action="store_true", help="print current state and exit")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    log_path = Path(args.log_file)
    if args.summary:
        return run_summary(log_path)
    return run_follow(log_path)


if __name__ == "__main__":
    sys.exit(main())
