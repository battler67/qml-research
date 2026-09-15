"""Pause, resume, or inspect persistent training without changing its scientific settings."""

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
OUTPUT = ROOT / "results/campaign"


def active():
    matches = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            arguments = process.info["cmdline"] or []
            relevant = any(
                Path(arg).name in {"run_campaign.py", "campaign_supervisor.py"}
                for arg in arguments[1:]
            )
            if relevant and Path(process.cwd()).resolve() == ROOT:
                matches.append(process.pid)
        except (psutil.Error, OSError):
            continue
    return matches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["status", "pause", "resume"])
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.action == "pause":
        (OUTPUT / "STOP").write_text("Pause requested at " + time.strftime("%Y-%m-%d %H:%M:%S"))
        print(
            "Pause requested. Neural batches/kernel chunks are checkpointed; "
            "an active SVM fit may finish first."
        )
        return 0
    running = active()
    if args.action == "status":
        print(
            json.dumps(
                {"active_pids": running, "pause_requested": (OUTPUT / "STOP").exists()}, indent=2
            )
        )
        config = json.loads((ROOT / "configs/resumable_campaign.json").read_text())
        path = OUTPUT / config["name"] / "status.json"
        if path.exists():
            status = json.loads(path.read_text())
            if status.get("status") == "running" and not running:
                status["status"] = "interrupted_or_exited; saved checkpoints remain resumable"
            print(json.dumps(status, indent=2))
        latest = sorted(
            (OUTPUT / config["name"]).glob("*/seed-*/*/trials/*/progress.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if latest:
            state = json.loads(latest[0].read_text())
            print(
                json.dumps(
                    {
                        "latest_training_checkpoint": str(latest[0]),
                        **{key: state[key] for key in ["epoch", "offset", "steps", "completed"]},
                    },
                    indent=2,
                )
            )
        return 0
    if running:
        print(f"A campaign is already running (PIDs {running}); not starting another.")
        return 2
    # Explicit resume is authorization to clear the two documented stop flags.
    for path in [OUTPUT / "STOP", ROOT / "results/laptop/STOP"]:
        if path.exists():
            path.unlink()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    options = (
        {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP}
        if os.name == "nt"
        else {"start_new_session": True}
    )
    with (
        (OUTPUT / f"console-{stamp}.log").open("w") as out,
        (OUTPUT / f"errors-{stamp}.log").open("w") as err,
    ):
        process = subprocess.Popen(
            [sys.executable, str(ROOT / "scripts/campaign_supervisor.py")],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=err,
            close_fds=True,
            **options,
        )
    (OUTPUT / "supervisor.pid").write_text(str(process.pid))
    print(
        f"Resumable campaign started, supervisor PID {process.pid}. "
        f"Log: {OUTPUT / f'console-{stamp}.log'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
