"""CLI entry points for the QCNN experiment."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from qml_research.qcnn.config import PROFILE_NAMES, load_qcnn_config
from qml_research.qcnn.runner import (
    generate_report,
    inspect_run,
    predict_image,
    prepare_or_verify_data,
    run_experiment,
    tune_qcnn,
    verify_research,
)


def _profile_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", choices=PROFILE_NAMES, default="laptop_8gb")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override any validated YAML field; repeat for multiple values",
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="qml-research qcnn")
    commands = root.add_subparsers(dest="command", required=True)

    data = commands.add_parser("data", help="prepare or verify an authoritative dataset")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    for name in ("prepare", "verify"):
        child = data_commands.add_parser(name)
        _profile_arguments(child)

    research = commands.add_parser("research", help="verify research prerequisite documents")
    research.add_argument("research_command", choices=["verify"])

    run = commands.add_parser("run", help="run a profile benchmark")
    _profile_arguments(run)
    run.add_argument("--download", action="store_true")
    run.add_argument("--force", action="store_true")
    run.add_argument("--confirm-expensive", action="store_true")
    run.add_argument(
        "--checkpoint",
        type=Path,
        help="explicit QCNN checkpoint for a noisy-simulation profile",
    )

    tune = commands.add_parser("tune", help="validation-only seeded QCNN search")
    _profile_arguments(tune)
    tune.add_argument("--confirm-expensive", action="store_true")

    estimate = commands.add_parser("estimate", help="show resources without running")
    _profile_arguments(estimate)

    evaluate = commands.add_parser("evaluate", help="inspect one completed run")
    evaluate.add_argument("--run-id", required=True)
    evaluate.add_argument("--output", default="results/qcnn_breast_cancer")

    predict = commands.add_parser("predict", help="predict one image with a QCNN checkpoint")
    predict.add_argument("--image", required=True, type=Path)
    predict.add_argument("--checkpoint", required=True, type=Path)

    report = commands.add_parser("report", help="regenerate aggregate reports")
    report.add_argument("--output", default="results/qcnn_breast_cancer")
    return root


def dispatch(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parser().parse_args(argv)
    if args.command == "research":
        return verify_research()
    if args.command == "data":
        config = load_qcnn_config(args.profile, args.overrides)
        return prepare_or_verify_data(config, download=args.data_command == "prepare")
    if args.command == "run":
        config = load_qcnn_config(args.profile, args.overrides)
        return run_experiment(
            config,
            download=args.download,
            force=args.force,
            confirm_expensive=args.confirm_expensive,
            checkpoint=args.checkpoint,
        )
    if args.command == "tune":
        config = load_qcnn_config(args.profile, args.overrides)
        return tune_qcnn(config, confirm_expensive=args.confirm_expensive)
    if args.command == "estimate":
        config = load_qcnn_config(args.profile, args.overrides)
        return {"config_fingerprint": config.fingerprint(), **config.estimated_resources()}
    if args.command == "evaluate":
        return inspect_run(args.output, args.run_id)
    if args.command == "predict":
        return predict_image(args.checkpoint, args.image)
    if args.command == "report":
        return generate_report(args.output)
    raise AssertionError(args.command)


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(dispatch(argv), indent=2, sort_keys=True))
    return 0
