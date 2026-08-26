# QCNN breast-cancer classification implementation record

Date: 2026-08-26
Branch: `feature/qcnn-breast-cancer`
Baseline: `77c82e9 chore: checkpoint qml research phase 1`

## Objective

Add a reproducible, image-input breast-cancer classification experiment built around a genuine hierarchical quantum convolutional neural network (QCNN). Benchmark it against matched-feature classical models, a native-image CNN, and pinned ImageNet transfer-learning models. Preserve the existing Phase 1 QSVM/VQC implementation.

The primary dataset is BreastMNIST ultrasound imagery with malignant disease remapped to positive label `1`. WDBC is a secondary diagnostic tabular control. This software is for research and education, not clinical use.

## Approved implementation decisions

- Keep the new work self-contained under `experiments/qcnn_breast_cancer/`, with integration through the existing `qml-research` CLI.
- Use a Cong-style inverse-MERA hierarchy with shared 15-parameter two-qubit convolution blocks and three-parameter unitary pooling blocks.
- Use four qubits and a `4 -> 2 -> 1` hierarchy by default. Reserve eight qubits and `8 -> 4 -> 2 -> 1` for the high-memory profile.
- Primary image reduction is deterministic adaptive average pooling to a `2 x 2` or `2 x 4` spatial grid followed by angle encoding. PCA is a train-fitted ablation.
- Train with analytic simulation first. Finite-shot and depolarizing-noise experiments evaluate a locked ideal checkpoint and never imply real-hardware execution.
- Benchmark on identical split manifests. Keep reproduced results separate from literature results.
- Implement and pass the smoke profile before any tuning or large benchmark.
- Do not implement QViT. Document QRNN feasibility without adding it to the experiment.
- Require explicit `--confirm-expensive` for estimated runs above 25,000 quantum forwards, two hours, eight qubits, or a noisy sweep.
- A quantum-advantage statement requires a preregistered metric improvement whose paired 95% confidence interval excludes zero. Parameter count or simulator accuracy alone is not quantum advantage.

## Planned public interface

```text
qml-research qcnn data prepare|verify
qml-research qcnn research verify
qml-research qcnn run --profile PROFILE [--set KEY=VALUE ...]
qml-research qcnn tune --profile PROFILE
qml-research qcnn evaluate --run-id RUN_ID
qml-research qcnn predict --image PATH --checkpoint PATH
qml-research qcnn report
```

Profiles and all tunable values live in YAML under `experiments/qcnn_breast_cancer/configs/`.

## Implementation log

- 2026-08-26: Inspected all five supplied papers and the full `prompt2.md`; checked authoritative MedMNIST, UCI WDBC, foundational QCNN, PennyLane, TorchVision, and QML benchmarking sources.
- 2026-08-26: Verified the pre-feature Phase 1 checkout with 18 passing tests, Ruff, a credential scan, and a large-file scan.
- 2026-08-26: Created baseline commit `77c82e9` on `qml-research-phase1` and branched `feature/qcnn-breast-cancer`.
- 2026-08-26: Added validated YAML profiles, deterministic official-split data loaders, the 4/8-qubit hierarchical QCNN, matched reduced-feature models, native CNN, random/pretrained ResNet-18, pretrained MobileNetV3-Small, and WDBC controls.
- 2026-08-26: Added early stopping, checkpoints, stable run identities, system/package metadata, exact indices, predictions, bootstrap intervals, quantum resources/gradients, reports, plots, tuning, noise evaluation, image prediction, and a no-submit hardware bundle.
- 2026-08-26: Added exact and range-based dependency files plus clone-to-larger-machine reproduction instructions. Generated datasets, checkpoints, figures, and reports remain ignored by Git.

## Verification record

### Automated gates

- Imported `torch 2.13.0+cpu`, `torchvision 0.28.0+cpu`, `PennyLane 0.45.1`, and `MedMNIST 3.0.2`; CUDA was unavailable on the verification laptop.
- `python -m ruff check .`: passed.
- `python -m ruff format --check .`: passed.
- `python -m compileall -q src tests`: passed.
- `python -m pytest -m "not slow" -q`: 37 passed in 98.69 seconds after all regressions were added.
- Synthetic orchestrator smoke: completed QCNN, logistic, and small-CNN runs with predictions, checkpoints, plots, and aggregate CSVs.
- BreastMNIST download: official archive MD5 `750601b1f35ba3300ea97c75c52ff8f6` verified; smoke subsets were 32 train, 16 validation, and 32 test without crossing official partitions.
- Standalone prediction: a saved QCNN checkpoint accepted a PNG and returned probability, threshold, class, run ID, and a non-diagnostic warning.
- Validation-only tuning smoke: two one-epoch trials completed without evaluating the test set. The selected smoke candidate used batch 16, initialization scale 0.01, learning rate 0.01, and weight decay 0.0001 with validation AUROC 0.625.
- Noise path: locked BreastMNIST checkpoint evaluated at 128 shots and depolarizing probability 0.01 for 32 recorded circuit executions.
- Hardware path: produced a four-qubit, 1,024-shot circuit/resource bundle with status `ready_not_submitted`; no provider or QPU was contacted.

### Reproduced functionality benchmark

These results are a one-seed smoke check, not a tuned study: 32/16/32 BreastMNIST images, two QCNN epochs, one image-model epoch, and validation-selected thresholds. They establish end-to-end comparability only.

| Model | Input | AUROC | Balanced accuracy | Training seconds |
|---|---|---:|---:|---:|
| QCNN | Same four pooled features as reduced baselines | 0.348 | 0.510 | 43.77 |
| Logistic regression | Same four pooled features | 0.657 | 0.507 | 0.04 |
| Reduced MLP | Same four pooled features | 0.671 | 0.703 | 1.45 |
| Small CNN | Native 28x28 image | 0.725 | 0.575 | 1.67 |
| ResNet-18 ImageNet V1 | Resized/normalized 224x224 image | 0.633 | 0.534 | 17.68 |
| MobileNetV3-Small ImageNet V1 | Resized/normalized 224x224 image | 0.816 | 0.556 | 1.69 |

The QCNN recorded 36 quantum parameters plus affine scale/bias, 144 circuit executions, device-level depth 38, 94 gates, and 24 two-qubit gates on `lightning.qubit`. The two-fold 16/8/16 WDBC control also completed: mean QCNN AUROC 0.575 versus logistic 0.967. Dataset-qualified plots prevent BreastMNIST and WDBC metrics from being averaged together.

### Interpretation and remaining work

- The smoke run shows no QCNN predictive advantage and no computational quantum advantage. It is too small for model ranking, and the QCNN was materially slower than the matched classical baselines.
- Hyperparameter tuning began only after functionality passed. The two-trial smoke validates test-set isolation but is not a final selection; the next research run is the configured larger validation-only search, followed by the `laptop_8gb` or `full_dataset` repeated-seed benchmark on the higher-resource machine.
- Confidence intervals from one smoke seed describe test-sample resampling only; advantage testing requires paired repeated seeds/splits and a preregistered difference interval excluding zero.
- ImageNet models use original-image information and are practical references, not capacity- or representation-matched controls. Published paper numbers remain separate from locally reproduced outputs.
- All outputs are research-only. BreastMNIST is small and WDBC is diagnostic FNA data; neither establishes screening utility or clinical validity.
