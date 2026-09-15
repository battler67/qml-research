"""Run one CPU experiment with sampled memory, runtime, and concurrency guards.

This is a watchdog, not an OS allocation cap: a sudden allocation can exceed the
threshold between samples. Only the launched process and its children are stopped.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3


def stop_tree(process: psutil.Process) -> None:
    try:
        processes = process.children(recursive=True) + [process]
    except psutil.NoSuchProcess:
        return
    for child in reversed(processes):
        with contextlib.suppress(psutil.NoSuchProcess):
            child.kill()
    psutil.wait_procs(processes, timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--max-gib", type=float, default=6)
    parser.add_argument("--reserve-gib", type=float, default=8)
    parser.add_argument("--minutes", type=float, default=120)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or min(args.max_gib, args.reserve_gib, args.minutes) <= 0:
        parser.error("supply a command and positive limits")
    if not 1 <= args.threads <= 8:
        parser.error("threads must be between 1 and 8")
    if Path(args.name).name != args.name or args.name in {".", ".."}:
        parser.error("name must be a filename")
    output = ROOT / "results" / "laptop"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "STOP").exists():
        print("Not starting: results/laptop/STOP exists.", flush=True)
        return 2
    # The OS releases this lock even if the watchdog is interrupted.
    lock = (output / "active.lock").open("a+b")
    if os.name == "nt":
        import msvcrt

        lock.seek(0)
        lock.write(b"0")
        lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            print("Another guarded job is running.", flush=True)
            return 2
    else:
        import fcntl

        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print("Another guarded job is running.", flush=True)
            return 2
    available = psutil.virtual_memory().available / GIB
    if available < args.reserve_gib + 1:
        print(f"Not starting: only {available:.2f} GiB available; reserve plus 1 GiB required.")
        return 2
    env = os.environ.copy()
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        env[key] = str(args.threads)
    env.update(CUDA_VISIBLE_DEVICES="-1", MPLBACKEND="Agg", PYTHONUNBUFFERED="1")
    started = time.monotonic()
    summary = {
        "command": command,
        "max_gib": args.max_gib,
        "reserve_gib": args.reserve_gib,
        "runtime_limit_minutes": args.minutes,
        "peak_rss_gib": 0.0,
        "peak_private_gib": 0.0,
        "minimum_available_gib": available,
        "status": "running",
        "cpu_threads": args.threads,
        "gpu_enabled": False,
    }
    stamp = time.strftime("%Y%m%d-%H%M%S")
    stem = output / f"{args.name}-{stamp}"
    process = None
    child = None
    try:
        with stem.with_suffix(".log").open("w", encoding="utf-8") as log:
            options = (
                {"creationflags": subprocess.BELOW_NORMAL_PRIORITY_CLASS} if os.name == "nt" else {}
            )
            child = subprocess.Popen(
                command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, **options
            )
            process = psutil.Process(child.pid)
            summary["pid"] = child.pid
            print(f"Started PID {child.pid}; log: {stem.with_suffix('.log')}", flush=True)
            next_report = 0.0
            while child.poll() is None:
                rss = private = 0
                try:
                    members = [process] + process.children(recursive=True)
                except psutil.NoSuchProcess:
                    members = []
                for member in members:
                    try:
                        memory = member.memory_info()
                        rss += memory.rss
                        private += getattr(memory, "private", memory.rss)
                    except psutil.NoSuchProcess:
                        pass
                available = psutil.virtual_memory().available / GIB
                elapsed = time.monotonic() - started
                summary["peak_rss_gib"] = max(summary["peak_rss_gib"], rss / GIB)
                summary["peak_private_gib"] = max(summary["peak_private_gib"], private / GIB)
                summary["minimum_available_gib"] = min(summary["minimum_available_gib"], available)
                reason = None
                if (output / "STOP").exists():
                    reason = "user_stop"
                elif max(rss, private) / GIB > args.max_gib:
                    reason = "process_memory_limit"
                elif available < args.reserve_gib:
                    reason = "system_memory_reserve"
                elif elapsed > args.minutes * 60:
                    reason = "runtime_limit"
                if reason:
                    summary["status"] = reason
                    stop_tree(process)
                    break
                if elapsed >= next_report:
                    print(
                        f"{elapsed / 60:.1f} min | RSS {rss / GIB:.2f} GiB | "
                        f"available {available:.2f} GiB",
                        flush=True,
                    )
                    next_report = elapsed + 45
                time.sleep(0.25)
            summary["exit_code"] = child.wait(timeout=10)
            if summary["status"] == "running":
                summary["status"] = "completed" if child.returncode == 0 else "failed"
    except KeyboardInterrupt:
        summary["status"] = "interrupted"
    except Exception as exc:
        summary["status"] = "watchdog_error"
        summary["error"] = str(exc)
    finally:
        if process is not None:
            stop_tree(process)
        summary["elapsed_seconds"] = time.monotonic() - started
        stem.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        lock.close()
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
