"""Reproducible innovation experiment: smoke first, locked holdout evaluated separately."""

import argparse
import json
from pathlib import Path

from qml_research.campaign.storage import Paused
from qml_research.innovation.data import download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=["download", "select", "evaluate", "report"], required=True
    )
    parser.add_argument("--config", default="configs/innovation_smoke.json")
    args = parser.parse_args()
    if args.stage == "download":
        download()
        return 0
    config = json.loads(Path(args.config).read_text())
    from qml_research.innovation.experiment import evaluate, select
    from qml_research.innovation.report import report

    try:
        if args.stage == "select":
            select(config)
        if args.stage == "evaluate":
            evaluate(config)
        report(config)
    except (Paused, KeyboardInterrupt):
        print("Paused. Keep checkpoints and rerun the same command.", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
