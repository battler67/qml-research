"""One guarded job at a time, reusing persistent campaign artifacts on restart."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from qml_research.campaign.storage import atomic_json

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/campaign"

# Finish broad full-data pilots, then all fixed repetitions; no test-driven grid changes.
JOBS = [
    ("kernel-pilots", "kernels", ["--seeds", "42", "--fold-limit", "1"], 240),
    (
        "neural-wdbc-pilot",
        "neural",
        ["--datasets", "wdbc", "--seeds", "42", "--fold-limit", "1"],
        240,
    ),
    ("all-kernels", "kernels", [], 720),
    ("all-neural", "neural", [], 1440),
]


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    journal = {
        "started": time.time(),
        "jobs": [],
        "pid": os.getpid(),
        "status": "running",
    }
    path = OUTPUT / "supervisor.json"
    for name, stage, extra, minutes in JOBS:
        if (OUTPUT / "STOP").exists():
            journal["status"] = "paused"
            break
        journal["active_job"] = name
        atomic_json(path, journal)
        command = [
            sys.executable,
            str(ROOT / "scripts/laptop_guard.py"),
            "--name",
            name,
            "--threads",
            "4",
            "--max-gib",
            "8",
            "--reserve-gib",
            "6",
            "--minutes",
            str(minutes),
            "--",
            sys.executable,
            str(ROOT / "scripts/run_campaign.py"),
            "--stage",
            stage,
            *extra,
        ]
        result = subprocess.run(command, cwd=ROOT, check=False)
        journal["jobs"].append({"name": name, "exit_code": result.returncode})
        if result.returncode:
            journal["status"] = "stopped"
            break
    else:
        journal.update(status="completed", active_job=None)
    atomic_json(path, journal)
    return 0 if journal["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
