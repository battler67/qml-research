"""Command-line interface for datasets, experiments, figures, and reports."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from qml_research.config import load_config
from qml_research.data import load_dataset
from qml_research.reports import generate_reports
from qml_research.runner import run_experiments
from qml_research.visualization import generate_all_plots


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qml-research",
        description="Reproducible classical-versus-QML biomedical experiments",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    data = commands.add_parser("data", help="dataset provenance and availability")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    verify = data_commands.add_parser("verify", help="load and validate authoritative datasets")
    verify.add_argument(
        "--datasets",
        nargs="+",
        default=["iris", "wdbc", "heart"],
        choices=["iris", "wdbc", "heart"],
    )
    run = commands.add_parser("run", help="execute a YAML experiment matrix")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--force", action="store_true", help="recompute completed run hashes")
    plot = commands.add_parser("plot", help="regenerate all available figures")
    plot.add_argument("--results", default="results/raw", type=Path)
    plot.add_argument("--output", default="results/figures", type=Path)
    report = commands.add_parser("report", help="regenerate normalized tables and reports")
    report.add_argument("--results", default="results/raw", type=Path)
    report.add_argument("--tables", default="results/tables", type=Path)
    report.add_argument("--research", default="research", type=Path)
    return parser


def _verify(names: list[str]) -> dict[str, object]:
    evidence: dict[str, object] = {}
    for name in names:
        bundle = load_dataset(name)
        values, counts = np.unique(bundle.y, return_counts=True)
        evidence[name] = {
            "samples": len(bundle.X),
            "features": bundle.X.shape[1],
            "class_counts": {
                str(int(value)): int(count) for value, count in zip(values, counts, strict=True)
            },
            "positive_class": bundle.positive_class,
            "source": bundle.provenance.get("source"),
            "checksum": bundle.provenance.get("sha256"),
        }
    return evidence


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "data":
        result = _verify(args.datasets)
    elif args.command == "run":
        result = run_experiments(load_config(args.config), force=args.force)
    elif args.command == "plot":
        result = {"figures": [str(path) for path in generate_all_plots(args.results, args.output)]}
    elif args.command == "report":
        result = {
            key: str(path)
            for key, path in generate_reports(args.results, args.tables, args.research).items()
        }
    else:  # pragma: no cover - argparse enforces a known command
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
