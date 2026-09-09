# QML Research Phase 1

This repository evaluates small biomedical quantum-machine-learning experiments
without assuming that quantum models should win. It compares PennyLane quantum
kernels and a shallow variational quantum classifier (VQC) with four classical
baselines on identical folds and leakage-safe features.

The workspace is intentionally independent from Quantum Helix Lab. Its YAML
configuration and JSON result interfaces are designed so selected Phase 2 models
can later be integrated without importing research-only orchestration.

## Scientific boundaries

- Binary Iris is an implementation sanity check, not medical evidence.
- Wisconsin Diagnostic Breast Cancer (WDBC) is a diagnostic breast-mass dataset,
  not longitudinal or pre-symptomatic early-detection data.
- Cleveland Heart Disease is converted to disease presence (`num > 0`) and is not
  a clinical deployment study.
- All data are public and non-identifiable. No real-QPU job is submitted in Phase 1.
- A favorable simulator result does not establish quantum advantage, clinical
  utility, speedup, or reduced end-to-end computation.

## Setup (Windows PowerShell)

For a fresh clone on the 24 GB laptop, follow
[the laptop handoff](docs/LAPTOP_24GB_SETUP.md). The complete experiment summary
and configuration limitations are in
[the experiment roadmap](specs/QML_EXPERIMENTS_AND_24GB_ROADMAP.md).

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements-lock.txt` records the exact Python 3.12 environment used for the
Phase 1 verification run. Use it when exact dependency-version reproduction is
more important than resolving the newest compatible packages.

## Commands

```powershell
# Verify loaders and provenance; Heart Disease requires network access once.
.\.venv\Scripts\python.exe -m qml_research data verify

# Fast end-to-end run.
.\.venv\Scripts\python.exe -m qml_research run --config configs/smoke.yaml

# Research checkpoints (progressively more expensive).
.\.venv\Scripts\python.exe -m qml_research run --config configs/iris_sanity.yaml
.\.venv\Scripts\python.exe -m qml_research run --config configs/wdbc_checkpoint.yaml
.\.venv\Scripts\python.exe -m qml_research run --config configs/heart_checkpoint.yaml
.\.venv\Scripts\python.exe -m qml_research run --config configs/wdbc_feature_screen_checkpoint.yaml
.\.venv\Scripts\python.exe -m qml_research run --config configs/wdbc_benchmark.yaml
.\.venv\Scripts\python.exe -m qml_research run --config configs/heart_benchmark.yaml
.\.venv\Scripts\python.exe -m qml_research run --config configs/noise_study.yaml

# Regenerate every available table, plot, and result report.
.\.venv\Scripts\python.exe -m qml_research plot --results results/raw
.\.venv\Scripts\python.exe -m qml_research report --results results/raw
```

Runs are identified by a stable hash of their scientific configuration and fold.
Already completed records are reused unless `--force` is supplied. Failures are
serialized instead of disappearing from the evidence trail.

Each YAML config declares dataset/sample sizes, folds, seeds, reduction methods,
feature counts, model families, feature maps, VQC layers/steps/batch size, optimizer,
learning rate, backend, shots, finite-shot count, noise strength/conditions, and the
runtime ceiling. Phase 1 validates `adam` and `default.qubit`; depolarizing kernels
switch explicitly to `default.mixed`. Full-data QML is skipped with a serialized
reason unless a matching 200-sample timing projection stays within the configured
per-family runtime gate.

## Layout

```text
configs/                 Reproducible experiment matrices
data/                    Provenance notes; downloaded data stay ignored
experiments/             Small direct reproductions and checkpoint entrypoints
notebooks/               Explanatory notebooks that import package code
research/                Literature, protocol, results, gaps, recommendation
results/raw/             One JSON record per fold/model plus matrix artifacts
results/reproductions/   Direct tutorial reproductions kept outside run aggregation
results/tables/          Regenerated normalized and summary CSV tables
results/figures/         Regenerated plots and circuit diagrams
specs/                   Branch, implementation, and verification record
src/qml_research/        Reusable loaders, models, evaluation, plotting, CLI
tests/                   Fast correctness and scientific-integrity checks
```

## Verification

```powershell
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest -m "not slow"
```

See `research/experiment-protocol.md` for the pre-registered comparison rules and
`research/qml-mvp-recommendation.md` for the evidence-gated Phase 2 decision.

## QCNN breast-ultrasound experiment

The dedicated `feature/qcnn-breast-cancer` experiment adds a genuine hierarchical
four-/eight-qubit QCNN for BreastMNIST malignant-versus-normal/benign image
classification. It includes leakage-matched classical baselines, a native CNN,
pinned ImageNet transfer-learning benchmarks, and WDBC as a diagnostic tabular
control.

Install the optional stack with `pip install -r requirements-qcnn.txt` (or the
exact verified `requirements-qcnn-lock.txt`), then start with:

```powershell
qml-research qcnn research verify
qml-research qcnn data prepare --profile smoke
qml-research qcnn run --profile smoke
```

All training sizes and hyperparameters are under
`experiments/qcnn_breast_cancer/configs/`. See its `docs/reproduction.md` for laptop,
full-dataset, high-memory, tuning, noise, inference, and future-hardware commands.
