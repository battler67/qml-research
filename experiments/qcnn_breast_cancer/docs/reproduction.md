# Reproduction and configuration guide

## Install

Python 3.12 is required.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-qcnn.txt
```

Use `requirements-qcnn-lock.txt` instead when reproducing the exact verified CPU
environment. The shorter `requirements-qcnn.txt` follows compatible dependency
ranges and is more suitable when selecting a CUDA-specific Torch build.

The pinned default is CPU-compatible PyTorch. On a CUDA machine, use the installer command produced by the official PyTorch selector for the required CUDA version, then reinstall this project with `pip install -e ".[dev,qcnn]"`. Record any changed Torch build in the run metadata and do not compare timing across different builds as if hardware were matched.

## Functionality-first commands

```powershell
.venv\Scripts\qml-research.exe qcnn research verify
.venv\Scripts\qml-research.exe qcnn data prepare --profile smoke
.venv\Scripts\qml-research.exe qcnn estimate --profile smoke
.venv\Scripts\qml-research.exe qcnn run --profile smoke
.venv\Scripts\qml-research.exe qcnn run --profile laptop_8gb --confirm-expensive
.venv\Scripts\qml-research.exe qcnn run --profile full_dataset --confirm-expensive
.venv\Scripts\qml-research.exe qcnn run --profile high_memory --confirm-expensive
.venv\Scripts\qml-research.exe qcnn tune --profile laptop_8gb --confirm-expensive
.venv\Scripts\qml-research.exe qcnn run --profile noisy_simulation `
  --checkpoint results\qcnn_breast_cancer\runs\RUN_ID\checkpoint.pt `
  --confirm-expensive
.venv\Scripts\qml-research.exe qcnn run --profile real_hardware_ready --confirm-expensive
```

Run WDBC as a controlled diagnostic tabular experiment:

```powershell
.venv\Scripts\qml-research.exe qcnn run --profile smoke `
  --set dataset=wdbc --set reducer=pca --set train_size=32 `
  --set validation_size=16 --set test_size=32
```

## Changing sample sizes and hyperparameters

Permanent profile changes belong in `configs/base.yaml` or a profile overlay. For one run, repeat `--set`:

```powershell
.venv\Scripts\qml-research.exe qcnn estimate --profile laptop_8gb `
  --set train_size=128 --set epochs=12 --set learning_rate=0.003 `
  --set seeds=[7,17]
```

Supported configuration includes dataset and split limits, class-balanced subsampling, reducer/features/qubits/stages, model list, backend/differentiation/shots/noise, optimizer settings, workers, seeds/folds, early stopping, thresholds, tuning space, cache/output locations, and expensive-run limits. Invalid combinations fail before training.

Official BreastMNIST validation and test partitions are never replaced. Integer limits select deterministic stratified subsets within each official partition. WDBC uses repeated stratified outer folds and a train-only inner validation split.

The tuning command uses the first configured seed as a fixed search seed and ranks
candidates on validation AUROC only; it records `test_set_evaluated: false`. Apply
the selected values with `--set` in a subsequent benchmark so every configured
seed is evaluated once on the untouched test split.

## Artifacts

Each deterministic run ID has its own folder below `results/qcnn_breast_cancer/runs/`:

- `record.json`: metrics, resources, environment, and scientific boundary
- `resolved_config.json`: effective profile after CLI overrides
- `split_manifest.json`: exact original indices and class counts
- `predictions.csv`: validation/test scores, thresholds, and decisions
- `history.json`: losses, AUROC, gradients, and circuit executions
- `checkpoint.pt`: model state and inference metadata
- `circuit.txt`: device-level QCNN circuit
- `figures/`: training, ROC/PR, confusion, confidence/calibration, PCA, and architecture plots

Regenerate aggregate reports with `qml-research qcnn report`. Downloaded data, checkpoints, caches, and generated binary results are ignored by Git. On a larger machine, clone the repository, recreate the environment, run `data prepare`, and select `high_memory`; no code changes are needed.

## Image inference

```powershell
.venv\Scripts\qml-research.exe qcnn predict `
  --image path\to\ultrasound.png `
  --checkpoint results\qcnn_breast_cancer\runs\RUN_ID\checkpoint.pt
```

Only spatial-pooling QCNN checkpoints accept standalone images. Output is an experimental malignant probability, never a medical diagnosis.

## Interpretation

- Compare QCNN first with models using the exact same reduced features.
- Treat native CNN/ImageNet results as practical full-image references, not capacity-matched controls.
- Keep published MedMNIST results separate from reproduced local results.
- Highlight malignant sensitivity and false negatives alongside AUROC and balanced accuracy.
- Do not claim quantum advantage without the preregistered paired-confidence criterion.
