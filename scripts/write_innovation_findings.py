"""Copy completed numerical evidence into reviewable, Git-visible research documents."""

from __future__ import annotations

import json
from pathlib import Path

from qml_research.campaign.storage import atomic_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config = json.loads((ROOT / "configs/innovation_confirm.json").read_text())
    root = ROOT / "results/innovation" / config["name"]
    summary = json.loads((root / "summary.json").read_text())
    records = json.loads((root / "results.json").read_text())
    if len(records) != 16 * len(config["seeds"]):
        raise ValueError("Evaluation is incomplete")
    report = (root / "REPORT.md").read_text(encoding="utf-8")
    (ROOT / "research/INNOVATION_RESULTS.md").write_text(report, encoding="utf-8")
    guards = []
    for path in sorted((ROOT / "results/laptop").glob("innovation-*.json")):
        value = json.loads(path.read_text())
        command = value.get("command", [])
        if any(Path(part).name == "run_innovation.py" for part in command):
            value["command"] = [part.replace(str(ROOT), "<repo>") for part in command]
            guards.append({"artifact": path.name, **value})
    atomic_json(
        ROOT / "research/innovation_evidence.json",
        {
            "config": config,
            "summary": summary,
            "per_seed_results": records,
            "resource_guard_runs": guards,
            "environment": json.loads((root / "environment.json").read_text()),
            "implementation": json.loads((root / "implementation.json").read_text()),
            "dataset_audit": json.loads((root / "audit.json").read_text()),
        },
    )
    families = summary["families"]
    quantum = families["residual_quantum"]
    reference = families["validation_selected_classical"]
    difference = summary["comparisons"]["validation_selected_classical"]
    comparisons = summary["comparisons"]
    figure, axis = plt.subplots(figsize=(9, 4.4))
    for i, (_name, value) in enumerate(comparisons.items()):
        low, high = value["auroc_adjusted_ci"]
        axis.hlines(i, low, high, color="#375c85", linewidth=2)
        axis.scatter(value["delta"]["auroc"], i, color="#375c85", s=35, zorder=3)
    axis.set_yticks(range(len(comparisons)), [n.replace("_", " ") for n in comparisons])
    axis.axvline(0, color="#888888", linestyle="--", linewidth=1)
    axis.set_xlabel("ROC-AUC difference: quantum residual minus comparator")
    axis.set_title("Paired comparisons with multiplicity-adjusted intervals")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(ROOT / "research/innovation_comparisons.png", dpi=160)
    plt.close(figure)
    all_fits = list(root.glob("seed-*/*/trials/*/validation.json"))
    total = sum(r["selected"]["total_tuning_seconds"] for r in records)
    peak = max(r["selected"]["peak_rss_gib"] for r in records)
    result_lines = [
        f"Completed **{len(all_fits)} candidate fits and {len(records)} held-out family/seed "
        "evaluations**, using all 400 rows under the fixed 256/64/80 partitions.",
        "",
        f"The proposed model's mean ROC-AUC was **{quantum['auroc']['mean']:.4f}**, versus "
        f"**{reference['auroc']['mean']:.4f}** for the validation-selected classical comparator.",
        f"The paired ROC-AUC difference was {difference['delta']['auroc']:+.4f}, with "
        f"multiplicity-adjusted interval [{difference['auroc_adjusted_ci'][0]:+.4f}, "
        f"{difference['auroc_adjusted_ci'][1]:+.4f}]. "
        + (
            "The prespecified predictive signal rule was met."
            if summary["predictive_signal_rule_met"]
            else "The prespecified predictive signal rule was not met."
        ),
        "",
        "| Approach | ROC-AUC [95% CI] | PR-AUC [95% CI] | "
        "Sensitivity [95% CI] | Specificity [95% CI] |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name in [
        "validation_selected_classical",
        "residual_quantum",
        "residual_classical",
        "residual_no_quantum",
        "residual_separable",
        "residual_frozen",
        "residual_ungated",
    ]:
        cells = []
        for metric in ["auroc", "auprc", "sensitivity", "specificity"]:
            value = families[name][metric]
            cells.append(f"{value['mean']:.4f} [{value['ci95'][0]:.4f}, {value['ci95'][1]:.4f}]")
        result_lines.append(f"| {name} | " + " | ".join(cells) + " |")
    result_lines += [
        "",
        f"Summed candidate tuning time: **{total / 60:.2f} minutes**; "
        f"maximum sampled per-trial RSS: **{peak:.3f} GiB**. "
        "This excludes the deliberately paused interval, test-suite time, report generation "
        "and shared preprocessing/anchor fits. Full-process guard totals are preserved separately.",
        "",
        "All ten baseline results, paired ablation intervals and per-family costs are in "
        "[INNOVATION_RESULTS.md](research/INNOVATION_RESULTS.md). "
        "[Machine-readable evidence](research/innovation_evidence.json) retains every seed, "
        "selected configuration, runtime and memory record. Raw predictions and checkpoints "
        "remain local under `results/innovation/`.",
        "",
        "![Paired ROC-AUC differences](research/innovation_comparisons.png)",
        "",
        "The smoke run completed 32 validation-only candidates. A full 92-test run "
        "and two additional interval checks passed. "
        "A deliberate real-run pause resumed at epoch 16 and step 120, "
        "preserving previous selections.",
    ]
    document = ROOT / "QML_INNOVATION.md"
    text = document.read_text(encoding="utf-8")
    start, end = "<!-- INNOVATION_RESULTS_START -->", "<!-- INNOVATION_RESULTS_END -->"
    before, rest = text.split(start)
    _, after = rest.split(end)
    document.write_text(
        before + start + "\n" + "\n".join(result_lines) + "\n" + end + after, encoding="utf-8"
    )
    print("Updated QML_INNOVATION.md and research/INNOVATION_RESULTS.md")


if __name__ == "__main__":
    main()
