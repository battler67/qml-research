"""Paired patient bootstrap conditional on the fitted seeds; never independent seed CIs."""

from __future__ import annotations

import json

import numpy as np
from scipy.stats import binomtest, rankdata

from qml_research.campaign.storage import atomic_json, replace_file
from qml_research.innovation.data import OUTPUT
from qml_research.innovation.experiment import FAMILIES, VARIANTS

METRICS = ["auroc", "auprc", "sensitivity", "specificity"]


def measures(y, score, prediction):
    positive = y == 1
    npos, nneg = positive.sum(), (~positive).sum()
    auc = (rankdata(score)[positive].sum() - npos * (npos + 1) / 2) / (npos * nneg)
    order = np.argsort(-score, kind="stable")
    truth, ordered = y[order], score[order]
    # Average precision with tied scores grouped exactly as sklearn does.
    ends = np.r_[np.flatnonzero(np.diff(ordered)), len(y) - 1]
    tp = truth.cumsum()[ends]
    recall = tp / npos
    ap = np.sum(np.diff(np.r_[0.0, recall]) * tp / (ends + 1))
    return np.array([auc, ap, prediction[positive].mean(), 1 - prediction[~positive].mean()])


def intervals(y, scores, predictions, draws, seed=915):
    """Arrays have [family, initialization, patient] axes; resample patient axis jointly."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == label) for label in [0, 1]]
    point = np.array(
        [
            [measures(y, s, p) for s, p in zip(ss, pp, strict=True)]
            for ss, pp in zip(scores, predictions, strict=True)
        ]
    )
    samples = np.empty((draws, len(scores), len(METRICS)))
    for draw in range(draws):
        indices = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        for family, (ss, pp) in enumerate(zip(scores, predictions, strict=True)):
            samples[draw, family] = np.mean(
                [measures(y[indices], s[indices], p[indices]) for s, p in zip(ss, pp, strict=True)],
                axis=0,
            )
    return point, samples


def rate_intervals(y, predictions, alpha=0.05):
    """Conservative interval for mean seed rates, without assuming independent seeds.

    Bonferroni gives simultaneous per-seed Clopper-Pearson coverage. Averaging
    those endpoints covers the mean rate whenever all seed intervals cover.
    """
    bounds = []
    for label in [1, 0]:
        mask = y == label
        per_seed = [
            binomtest(int((p[mask] == label).sum()), int(mask.sum())).proportion_ci(
                confidence_level=1 - alpha / len(predictions), method="exact"
            )
            for p in predictions
        ]
        bounds.append(np.mean([[ci.low, ci.high] for ci in per_seed], axis=0))
    return np.array(bounds).T


def report(config):
    root = OUTPUT / config["name"]
    root.mkdir(parents=True, exist_ok=True)
    results = root / "results.json"
    lines = [f"# {config['name']}", ""]
    if not results.exists():
        paths = list(root.glob("seed-*/*/selected.json"))
        lines += [
            f"Completed validation selections: {len(paths)}. Test set remains locked.",
            "",
            "| Seed | Family | Validation ROC-AUC | Trials | Tuning seconds |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
        for path in sorted(paths):
            r = json.loads(path.read_text())
            lines.append(
                f"| {path.parent.parent.name} | {path.parent.name} | "
                f"{r['validation']['auroc']:.4f} | {r['trial_count']} | "
                f"{r['total_tuning_seconds']:.2f} |"
            )
    else:
        records = json.loads(results.read_text())
        lookup = {(r["seed"], r["family"]): r for r in records}
        loaded = {}
        for family in FAMILIES:
            loaded[family] = [
                np.load(root / f"seed-{seed}" / family / "test_predictions.npz")
                for seed in config["seeds"]
            ]
        first = loaded[FAMILIES[0]][0]
        for runs in loaded.values():
            for run in runs:
                if not np.array_equal(first["indices"], run["indices"]) or not np.array_equal(
                    first["y"], run["y"]
                ):
                    raise ValueError("Predictions do not share identical test patients")
        refs = [lookup[seed, FAMILIES[0]]["reference"] for seed in config["seeds"]]
        names = FAMILIES + ["validation_selected_classical"]
        all_scores = [[r["score"] for r in loaded[f]] for f in FAMILIES]
        all_predictions = [[r["prediction"] for r in loaded[f]] for f in FAMILIES]
        all_scores.append([loaded[ref][i]["score"] for i, ref in enumerate(refs)])
        all_predictions.append([loaded[ref][i]["prediction"] for i, ref in enumerate(refs)])
        point, samples = intervals(
            first["y"], np.array(all_scores), np.array(all_predictions), config["bootstrap_draws"]
        )
        summary = {}
        lines += [
            f"Locked test patients: {len(first['y'])}; seeds: {config['seeds']}.",
            "",
            "Values are mean per-seed metrics. ROC-AUC/PR-AUC intervals use a stratified "
            "paired patient bootstrap. Sensitivity/specificity use conservative binomial "
            "intervals: mean endpoints of seed-wise Clopper-Pearson intervals with "
            "Bonferroni coverage across seeds. No seed independence is assumed. Intervals "
            "are conditional on these fitted models, not independent cohort or seed evidence.",
            "",
            "PR-AUC here means average precision. Thresholds use validation Youden J.",
            "",
            "| Family | ROC-AUC [95% CI] | PR-AUC [95% CI] | Sensitivity [95% CI] | "
            "Specificity [95% CI] | Tuning s/seed | Peak RSS GiB |",
            "| --- | --- | --- | --- | --- | ---: | ---: |",
        ]
        for i, family in enumerate(names):
            values = point[i].mean(axis=0)
            low, high = np.quantile(samples[:, i], [0.025, 0.975], axis=0)
            low[2:], high[2:] = rate_intervals(first["y"], all_predictions[i])
            summary[family] = {
                metric: {"mean": float(values[j]), "ci95": [float(low[j]), float(high[j])]}
                for j, metric in enumerate(METRICS)
            }
            costs = [lookup[seed, family] for seed in config["seeds"]] if family in FAMILIES else []
            seconds = (
                np.mean([r["selected"]["total_tuning_seconds"] for r in costs]) if costs else 0.0
            )
            peak = max([r["selected"]["peak_rss_gib"] for r in costs], default=0.0)
            summary[family]["tuning_seconds_per_seed"] = float(seconds)
            summary[family]["peak_rss_gib"] = peak
            cells = [
                f"{v:.4f} [{lo:.4f}, {hi:.4f}]" for v, lo, hi in zip(values, low, high, strict=True)
            ]
            lines.append(f"| {family} | " + " | ".join(cells) + f" | {seconds:.2f} | {peak:.3f} |")
        qi = names.index("residual_quantum")
        comparisons = {}
        references = ["validation_selected_classical"] + [
            "residual_" + v for v in VARIANTS if v != "quantum"
        ]
        lines += [
            "",
            "## Prespecified paired comparisons",
            "",
            "Positive differences favor the proposed quantum residual. Six ROC-AUC comparisons "
            "use Bonferroni-adjusted intervals; other metric intervals are descriptive 95%.",
            "",
            "| Comparator | ROC-AUC difference [adjusted CI] | Sensitivity difference [95% CI] |",
            "| --- | --- | --- |",
        ]
        for family in references:
            i = names.index(family)
            delta = point[qi].mean(axis=0) - point[i].mean(axis=0)
            bootstrap = samples[:, qi] - samples[:, i]
            lower, upper = np.quantile(
                bootstrap[:, 0], [0.05 / (2 * len(references)), 1 - 0.05 / (2 * len(references))]
            )
            lo, hi = np.quantile(bootstrap, [0.025, 0.975], axis=0)
            comparisons[family] = {
                "delta": dict(zip(METRICS, delta.tolist(), strict=True)),
                "auroc_adjusted_ci": [float(lower), float(upper)],
                "descriptive_ci95": {
                    m: [float(lo[j]), float(hi[j])] for j, m in enumerate(METRICS)
                },
            }
            lines.append(
                f"| {family} | {delta[0]:+.4f} [{lower:+.4f}, {upper:+.4f}] | "
                f"{delta[2]:+.4f} [{lo[2]:+.4f}, {hi[2]:+.4f}] |"
            )
        baseline = comparisons["validation_selected_classical"]
        evidence = (
            baseline["delta"]["auroc"] >= 0.01
            and baseline["auroc_adjusted_ci"][0] > 0
            and baseline["descriptive_ci95"]["sensitivity"][0] >= -0.02
        )
        lines += [
            "",
            "## Conclusion",
            "",
            (
                "The prespecified predictive signal rule was met on this small holdout."
                if evidence
                else "The prespecified predictive signal rule was not met; "
                "no reliable improvement is established."
            ),
            "This rule requires ROC-AUC gain >=0.01, adjusted interval above zero, and "
            "sensitivity-difference lower bound >=-0.02. "
            "It is not a computational advantage claim.",
            "",
            f"Validation-selected classical families by seed: {refs}.",
            "Costs include all tuning trials; shared preprocessing/one fixed logistic-anchor fit "
            "is recorded separately per seed. Peak RSS includes the interpreter and libraries. "
            "The selected-reference row has no standalone cost "
            "(zero is a placeholder, not free training).",
            "",
        ]
        atomic_json(
            root / "summary.json",
            {
                "families": summary,
                "comparisons": comparisons,
                "predictive_signal_rule_met": bool(evidence),
                "bootstrap_draws": config["bootstrap_draws"],
            },
        )
    destination = root / "REPORT.md"
    temp = destination.with_suffix(".tmp")
    temp.write_text("\n".join(lines), encoding="utf-8")
    replace_file(temp, destination)
    print(f"Report: {destination}", flush=True)
    return destination
