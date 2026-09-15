"""Regenerate complete and positive-difference tables without selecting on test data."""

from __future__ import annotations

# ruff: noqa: E501 -- Markdown prose and table templates.
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from qml_research.campaign.storage import OUTPUT, ROOT, atomic_json, replace_file


def report(config: dict) -> Path:
    root = OUTPUT / config["name"]
    records = [json.loads(path.read_text()) for path in root.glob("*/seed-*/**/result.json")]
    rows = []
    for record in records:
        rows.append(
            {
                **{
                    key: record[key]
                    for key in [
                        "dataset",
                        "seed",
                        "fold",
                        "family",
                        "kind",
                        "train_rows",
                        "validation_rows",
                        "test_rows",
                        "tuning_trials",
                    ]
                },
                **record["test"],
                "validation_auroc": record["validation"]["auroc"],
            }
        )
    table = pd.DataFrame(rows)
    root.mkdir(parents=True, exist_ok=True)
    table.to_csv(root / "all_results.csv", index=False)
    text = [
        "# QML models with better measured metrics",
        "",
        "Generated from saved held-out predictions and validation-only model selection. This file includes negative results; it is not a promise of quantum advantage.",
        "",
        "## Evidence standard",
        "",
        "Each model family selects hyperparameters by validation AUROC, and chooses a threshold on validation data. Test labels are not used for selection. The comparator is the classical family selected by validation AUROC on the same split. All classical test results are also retained below to expose stronger alternatives.",
        "",
        "A positive mean difference is exploratory. A stronger signal requires all 3 seeds × 5 outer folds, mean balanced-accuracy gain ≥0.02, no mean sensitivity loss >0.02, and a multiplicity-adjusted corrected repeated-CV interval above zero. The correction uses test/train size and a Bonferroni adjustment over the planned quantum-family/dataset comparisons. It remains an approximate within-dataset analysis, not external clinical validation.",
        "",
        "Angle kernels are separable and have an efficient classical product-cosine implementation. Projected quantum kernels have a classical RBF head on simulated quantum features. No simulator result here establishes computational quantum advantage.",
        "",
        "Kaggle Framingham is a 4,238-row third-party mirror with unverifiable follow-up/censoring and unspecified source license; it is not the official BioLINCC longitudinal cohort. Pima is a limited demographic benchmark. BreastMNIST uses its official partitions; repeated seeds share the same test images.",
        "",
    ]
    comparisons = []
    if len(table):
        classical = table[table.kind == "classical"]
        for (dataset, family), quantum in table[table.kind == "quantum"].groupby(
            ["dataset", "family"]
        ):
            paired = []
            for _, q in quantum.iterrows():
                candidates = classical[
                    (classical.dataset == dataset)
                    & (classical.seed == q.seed)
                    & (classical.fold == q.fold)
                ]
                required = {
                    name for name in config["kernel_families"] if name.startswith("classical_")
                }
                if not required.issubset(set(candidates.family)):
                    continue
                best = candidates.sort_values(
                    ["validation_auroc", "family"], ascending=[False, True]
                ).iloc[0]
                paired.append(
                    {
                        "seed": int(q.seed),
                        "fold": int(q.fold),
                        "reference": best.family,
                        "ba": float(q.balanced_accuracy - best.balanced_accuracy),
                        "auc": float(q.auroc - best.auroc),
                        "ap": float(q.auprc - best.auprc),
                        "sensitivity": float(q.sensitivity - best.sensitivity),
                        "ratio": float(q.test_rows / q.train_rows),
                    }
                )
            if not paired:
                continue
            pairs = pd.DataFrame(paired)
            n = len(pairs)
            mean = float(pairs.ba.mean())
            lower = upper = None
            if n > 1:
                variance = float(pairs.ba.var(ddof=1))
                planned = max(
                    1, 5 * len(config["datasets"]) + int("breastmnist" in config["datasets"])
                )
                error = float(t.ppf(1 - 0.05 / (2 * planned), n - 1)) * np.sqrt(
                    (1 / n + pairs.ratio.mean()) * variance
                )
                lower, upper = float(mean - error), float(mean + error)
            complete = n == len(config["seeds"]) * config["outer_folds"] and set(pairs.seed) == set(
                config["seeds"]
            )
            qualifies = bool(
                complete
                and mean >= 0.02
                and lower is not None
                and lower > 0
                and pairs.sensitivity.mean() >= -0.02
            )
            comparisons.append(
                {
                    "dataset": dataset,
                    "family": family,
                    "paired_splits": n,
                    "delta_balanced_accuracy": mean,
                    "delta_auroc": float(pairs.auc.mean()),
                    "delta_auprc": float(pairs.ap.mean()),
                    "delta_sensitivity": float(pairs.sensitivity.mean()),
                    "corrected_interval": [lower, upper],
                    "complete_repeated_cv": complete,
                    "meets_stronger_signal_rule": qualifies,
                    "comparisons": paired,
                }
            )
        text += [
            "## Models with positive point estimates",
            "",
            "These rows improved at least one mean metric against the validation-selected classical comparator. They are candidates, not established advantages.",
            "",
            "| Dataset | QML family | Paired splits | Δ balanced accuracy | Δ AUROC | Δ AUPRC | Stronger signal rule |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
        positives = [
            r
            for r in comparisons
            if max(r["delta_balanced_accuracy"], r["delta_auroc"], r["delta_auprc"]) > 0
        ]
        for r in positives:
            text.append(
                f"| {r['dataset']} | {r['family']} | {r['paired_splits']} | {r['delta_balanced_accuracy']:+.4f} | {r['delta_auroc']:+.4f} | {r['delta_auprc']:+.4f} | {'Yes' if r['meets_stronger_signal_rule'] else 'No'} |"
            )
        if not positives:
            text.append("| No positive paired result yet | — | — | — | — | — | No |")
        text += [
            "",
            "## All completed family results",
            "",
            "Means below include every completed split for that family. Unequal counts indicate incomplete coverage and must not be treated as matched comparisons.",
            "",
            "| Dataset | Family | Splits | Mean balanced accuracy | Mean AUROC | Mean AUPRC |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for (dataset, family), group in table.groupby(["dataset", "family"]):
            text.append(
                f"| {dataset} | {family} | {len(group)} | {group.balanced_accuracy.mean():.4f} | {group.auroc.mean():.4f} | {group.auprc.mean():.4f} |"
            )
    else:
        text += [
            "No completed tuned comparisons yet. Historical smoke scores are not substituted for new evidence.",
            "",
        ]
    winners = [r for r in comparisons if r["meets_stronger_signal_rule"]]
    text += [
        "",
        "## Current conclusion",
        "",
        f"{len(winners)} comparison(s) meet the stronger signal rule. "
        + (
            "No robust predictive advantage has been demonstrated in this campaign yet."
            if not winners
            else "These are within-dataset predictive signals, not computational advantage or external validation."
        ),
        "",
        "Raw records, trial warnings/failures, exact split indices, selected specifications, and predictions are under `results/campaign/"
        + config["name"]
        + "/`. Resume and protocol details: [campaign guide](../docs/RESUMABLE_CAMPAIGN.md).",
        "",
    ]
    failures = list(root.glob("*/seed-*/**/failure.json"))
    text += [
        f"Saved failed-trial records: {len(failures)}. Completed family/split records: {len(records)}.",
        "",
    ]
    atomic_json(root / "paired_comparisons.json", comparisons)
    destination = ROOT / "research/QML_MODELS_WITH_BETTER_METRICS.md"
    temporary = destination.with_suffix(".md.tmp")
    temporary.write_text("\n".join(text), encoding="utf-8")
    replace_file(temporary, destination)
    return destination
