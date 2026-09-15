"""Sequential laptop experiments; every worker is launched through laptop_guard."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOBS = {
    "smoke": [("smoke-phase1", 15), ("smoke-ehr", 30), ("smoke-qcnn", 30)],
    "train": [("train-ehr", 180), ("train-qcnn", 240), ("train-wdbc", 180)],
}


def worker(name: str) -> dict:
    if name in {"smoke-phase1", "train-wdbc"}:
        from qml_research.config import load_config
        from qml_research.runner import run_experiments

        path = (
            "configs/smoke.yaml" if name == "smoke-phase1" else "configs/laptop_wdbc_training.yaml"
        )
        config = load_config(ROOT / path)
        if name == "smoke-phase1":
            # Iris has only 100 rows; a full-data run otherwise hits the runner's
            # 200-row timing prerequisite and skips both quantum families.
            config = replace(
                config,
                name="laptop_iris_smoke",
                sample_sizes=(50,),
                output_dir="results/laptop/phase1",
            )
        return run_experiments(config)
    if name in {"smoke-ehr", "train-ehr"}:
        from qml_research.ehr_ihd.cli import dispatch
        from qml_research.ehr_ihd.config import load_ihd_config

        filename = "uci_smoke.yaml" if name == "smoke-ehr" else "laptop_24gb_guarded.yaml"
        config_path = ROOT / "experiments/ehr_ihd_qml/configs" / filename
        if name == "smoke-ehr":
            config = load_ihd_config(config_path)
            for summary_path in sorted(config.output_dir.glob("*/summary.json"), reverse=True):
                resolved = summary_path.parent / "resolved_config.yaml"
                if (
                    resolved.exists()
                    and load_ihd_config(resolved).fingerprint() == config.fingerprint()
                ):
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    if summary.get("status") == "completed":
                        return {**summary, "reused_existing_smoke": True}
        return dispatch(["run", "--config", str(config_path)])
    from qml_research.qcnn.cli import dispatch

    args = ["run", "--profile", "smoke" if name == "smoke-qcnn" else "full_dataset"]
    args += ["--set", "seeds=[42]", "--set", "num_workers=0", "--set", "image_batch_size=16"]
    # Includes the native CNN and controls without large pretrained-weight downloads.
    args += [
        "--set",
        "models=[qcnn,logistic,linear_svm,rbf_svm,random_forest,reduced_mlp,small_cnn]",
    ]
    if name == "train-qcnn":
        args += ["--set", "bootstrap_iterations=2000", "--confirm-expensive"]
    return dispatch(args)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=JOBS, default="smoke")
    parser.add_argument("--worker", choices=[name for jobs in JOBS.values() for name, _ in jobs])
    args = parser.parse_args()
    if args.worker:
        result = worker(args.worker)
        print(json.dumps(result, indent=2, default=str), flush=True)
        unsuccessful = (
            result.get("failed", 0)
            or result.get("status") == "failed"
            or result.get("runtime_limit_reached", False)
        )
        return 1 if unsuccessful else 0
    output = ROOT / "results/laptop"
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / f"{args.stage}-suite.json"
    if args.stage == "train":
        smoke_path = output / "smoke-suite.json"
        if not smoke_path.exists() or json.loads(smoke_path.read_text())["status"] != "completed":
            print("Complete the guarded smoke suite before starting full training.")
            return 2
    status = {
        "stage": args.stage,
        "status": "running",
        "jobs": [],
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    for name, minutes in JOBS[args.stage]:
        status["active_job"] = name
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        command = [
            sys.executable,
            str(ROOT / "scripts/laptop_guard.py"),
            "--name",
            name,
            "--minutes",
            str(minutes),
            "--",
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            name,
        ]
        result = subprocess.run(command, cwd=ROOT, check=False)
        status["jobs"].append({"name": name, "exit_code": result.returncode})
        if result.returncode:
            status["status"] = "stopped"
            status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
            return result.returncode
    status.update(status="completed", active_job=None)
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
