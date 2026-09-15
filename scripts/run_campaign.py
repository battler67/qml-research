"""Entry point for a checkpointed campaign; wrap with laptop_guard.py."""

import argparse
import json
from pathlib import Path

from qml_research.campaign.runner import run
from qml_research.campaign.storage import ROOT

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", type=Path, default=ROOT / "configs/resumable_campaign.json")
parser.add_argument("--stage", choices=["kernels", "neural", "report", "audit"], default="kernels")
parser.add_argument("--datasets", nargs="+")
parser.add_argument("--seeds", nargs="+", type=int)
parser.add_argument("--fold-limit", type=int)
args = parser.parse_args()
config = json.loads(args.config.read_text())
if args.stage == "report":
    from qml_research.campaign.report import report

    print(report(config))
elif args.stage == "audit":
    from qml_research.campaign.data import load_dataset

    for name in args.datasets or config["datasets"]:
        data = load_dataset(name)
        print(name, len(data.y), data.digest)
else:
    raise SystemExit(
        run(
            config,
            stage=args.stage,
            datasets=args.datasets,
            seeds=args.seeds,
            fold_limit=args.fold_limit,
        )
    )
