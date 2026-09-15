"""Inspect, pause or resume the guarded innovation experiment on this laptop."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/innovation"
STOP = ROOT / "results/campaign/STOP"


def active():
    names = {"run_innovation.py", "run_campaign.py", "campaign_supervisor.py"}
    found = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            if (
                any(Path(a).name in names for a in (process.info["cmdline"] or [])[1:])
                and Path(process.cwd()).resolve() == ROOT
            ):
                found.append(process.pid)
        except (psutil.Error, OSError):
            continue
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["status", "pause", "resume"])
    parser.add_argument("--config", default="configs/innovation_confirm.json")
    parser.add_argument("--stage", choices=["select", "evaluate"], default="select")
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve()
    config = json.loads(config_path.read_text())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.action == "pause":
        STOP.parent.mkdir(parents=True, exist_ok=True)
        STOP.write_text("Pause requested at " + time.strftime("%Y-%m-%d %H:%M:%S"))
        print("Pause requested; the worker exits at a checkpoint/trial boundary.")
        return 0
    running = active()
    root = OUTPUT / config["name"]
    if args.action == "status":
        result = {
            "active_pids": running,
            "pause_requested": STOP.exists(),
            "selected_families": len(list(root.glob("seed-*/*/selected.json"))),
            "expected_selections": 16 * len(config["seeds"]),
            "test_records": len(list(root.glob("seed-*/*/test.json"))),
        }
        status = root / "status.json"
        if status.exists():
            result["worker_status"] = json.loads(status.read_text())
        progress = sorted(
            root.glob("seed-*/*/trials/*/progress.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if progress:
            state = json.loads(progress[0].read_text())
            result["latest_checkpoint"] = {
                "path": str(progress[0]),
                **{k: state[k] for k in ["epoch", "offset", "steps", "completed"]},
            }
        print(json.dumps(result, indent=2))
        return 0
    if running:
        print(f"Existing training processes {running}; not launching another worker.")
        return 2
    # Explicit resume authorizes clearing only the two documented local stop flags.
    for path in [STOP, ROOT / "results/laptop/STOP"]:
        path.unlink(missing_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    command = [
        sys.executable,
        str(ROOT / "scripts/laptop_guard.py"),
        "--name",
        "innovation-" + args.stage,
        "--threads",
        "4",
        "--max-gib",
        "8",
        "--reserve-gib",
        "6",
        "--minutes",
        "180",
        "--",
        sys.executable,
        str(ROOT / "scripts/run_innovation.py"),
        "--stage",
        args.stage,
        "--config",
        str(config_path),
    ]
    options = (
        {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP}
        if os.name == "nt"
        else {"start_new_session": True}
    )
    with (OUTPUT / f"guard-{stamp}.log").open("w") as stream:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=stream,
            close_fds=True,
            **options,
        )
    print(f"Guard started as PID {process.pid}; inspect status and results/laptop logs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
