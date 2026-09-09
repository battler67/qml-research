"""CLI dispatch for the EHR IHD experiment."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from qml_research.ehr_ihd.config import load_ihd_config
from qml_research.ehr_ihd.predict import predict_record
from qml_research.ehr_ihd.runner import run_uci_smoke


def dispatch(argv: Sequence[str] | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser(prog="qml-research ehr-ihd")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--run-id")
    predict = commands.add_parser("predict")
    predict.add_argument("--run-dir", type=Path, required=True)
    predict.add_argument("--record", type=Path, required=True)
    predict.add_argument("--explain", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "run":
        return run_uci_smoke(load_ihd_config(args.config), run_id=args.run_id)
    record = json.loads(args.record.read_text(encoding="utf-8"))
    return predict_record(args.run_dir, record, explain=args.explain)
