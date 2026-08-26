# QML Research Phase 1 Implementation Record

## Objective

Build a standalone, reproducible research repository for fair classical-versus-
quantum machine-learning experiments on small biomedical datasets. The work must
identify a scientifically defensible QML configuration for a later Quantum Helix
Lab integration without modifying that application in Phase 1.

## Branch and scope

- Repository: `Rit_Grover_project/qml-research`
- Branch: `qml-research-phase1`
- Created with: `git init -b main`, then `git checkout -b qml-research-phase1`
- Runtime: isolated Python 3.12 environment
- In scope: Iris sanity reproduction, WDBC, Cleveland Heart Disease, four
  classical baselines, PennyLane quantum kernel, PennyLane VQC, controlled noise,
  full metric/resource reporting, plots, literature review, and MVP recommendation.
- Out of scope: Quantum Helix Lab UI/API integration, real-QPU execution, private
  or identifiable data, MedMNIST, large genomics, and Parkinson's experiments.
- Repository remains local; no remote, publication, or provider credentials.

## Scientific rules

- Disease is the positive class: malignant WDBC samples and heart-disease presence
  map to `1`.
- Imputation, encoding, scaling, PCA, and selection are fitted on training data
  only. Matched models share samples, folds, transformed features, and seeds.
- Primary comparison metric is balanced accuracy; sensitivity is the safety metric.
- Accuracy alone, simulator wall-clock time, or one favorable seed cannot establish
  quantum advantage.
- Failures, PSD violations, convergence issues, skipped resource-heavy runs, and
  differences between ideal, finite-shot, noisy, and hardware execution are explicit.
- Raw machine-readable results are the source for all tables and figures.

## Planned checkpoints

1. Repository/environment scaffold, literature strategy, and dataset verification.
2. Leakage-aware Iris reproduction, quantum-kernel validation, VQC validation, and
   classical sanity baselines.
3. WDBC matched-feature benchmark, metrics, resources, and required plots.
4. Heart Disease benchmark, controlled kernel noise study, and scaling analysis.
5. Literature review, results interpretation, MVP recommendation, tests, and
   reproducibility audit.

## Verification log

Verification completed on 2026-08-26. All result statements below come from the
serialized records and regenerated tables, not hardcoded report values.

### Checkpoint 1: repository, environment, literature strategy, and data

Completed:

- Initialized the standalone repository and created `qml-research-phase1` with
  `git checkout -b qml-research-phase1`. No remote was configured.
- Created the Python 3.12.7 `.venv`, installed the declared dependencies, and
  captured the resolved environment in `requirements-lock.txt`.
- Implemented typed Iris, WDBC, and official UCI Heart Disease loaders with
  explicit positive-class semantics and provenance.
- Verified Iris as 100 x 4 with 50/50 classes, WDBC as 569 x 30 with 212 malignant
  positives, and Heart as 303 x 13 with 139 disease-present positives.
- Cached Heart Disease with SHA-256
  `6843d7c31598a05caa6e0968d6e95d803af235a048f69d5e8f7dd3a80808747e`.
- Defined a 15-source literature strategy spanning official tutorials,
  peer-reviewed biomedical studies, foundational QML theory, noise/trainability,
  and fair benchmarking.

Commands:

```powershell
git init -b main
git checkout -b qml-research-phase1
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m qml_research data verify --datasets iris wdbc heart
```

Observation and risk: the Heart download required one approved network call; the
cached CSV and provenance manifest make subsequent runs local. All datasets are
public and non-identifiable, but none represents prospective clinical validation.

### Checkpoint 2: Iris reproduction and implementation validation

Completed:

- Reproduced the binary-Iris PennyLane kernel workflow and separately evaluated
  the corrected split-before-scaling workflow.
- Both tutorial-order and fold-local variants produced 1.000 accuracy on the fixed
  25-sample test set with 4,725 circuit executions. This is recorded only as an
  implementation sanity result.
- The smoke matrix completed 10/10 records without failures, covering four full-
  and matched-feature classical baselines, Angle QKSVM, and a short VQC run.
- Tests validate kernel self-similarity, symmetry, dimensions, PSD reporting,
  execution counting, prediction shape, VQC history, and device-level resources.

Commands:

```powershell
.\.venv\Scripts\python.exe experiments\reproduce_pennylane_kernel.py
.\.venv\Scripts\python.exe -m qml_research run --config configs\smoke.yaml --force
.\.venv\Scripts\python.exe -m pytest tests\test_quantum_kernel.py tests\test_vqc.py
```

Observation and risk: perfect Iris performance is neither biomedical evidence nor
quantum advantage. VQC uses parameter-shift differentiation and therefore required
far more executions than a kernel prediction on these small examples.

### Checkpoint 3: WDBC, matched features, metrics, and scaling

Completed:

- `wdbc_phase1_checkpoint` completed 30/30 records at 50, 100, and 200 samples in
  298.9 seconds using one fixed fold and seed 42.
- `wdbc_feature_kernel_checkpoint` completed 48/48 records in 514.4 seconds across
  PCA/mutual information, 2/4/6/8 features, Angle/IQP kernels, and four matched
  classical baselines.
- The strongest observed QML checkpoint used AngleEmbedding plus two PCA features:
  balanced accuracy 0.9375, sensitivity 0.8750, specificity 1.0000, AUROC 0.9896.
  Matched logistic regression produced the same balanced accuracy, sensitivity,
  specificity, and AUROC on that exact 100-sample fold.
- Angle QKSVM execution counts grew from 1,220 at 50 samples to 4,840 at 100 and
  19,280 at 200. The 100-step, four-qubit VQC used about 40,000 QNode executions per
  checkpoint run and did not converge to competitive WDBC performance.

Commands:

```powershell
.\.venv\Scripts\python.exe -m qml_research run --config configs\wdbc_checkpoint.yaml
.\.venv\Scripts\python.exe -m qml_research run --config configs\wdbc_feature_screen_checkpoint.yaml
```

Observation and risk: the two-feature choice is exploratory and based on one fold.
The pre-registered `wdbc_screen.yaml` and `wdbc_benchmark.yaml` matrices preserve the
full five-fold/three-seed protocol; those combinations were not silently claimed as
executed.

### Checkpoint 4: Heart, noise, and scaling

Completed:

- `heart_phase1_checkpoint` completed 30/30 records at 50, 100, and 200 samples
  without model failures. Its measured wall time was 1,936.7 seconds.
- Heart results were unstable across size: Angle QKSVM balanced accuracy was 0.7000,
  0.5505, and 0.7197 at 50, 100, and 200 samples. The VQC fell to sensitivity 0.0556
  at 200 samples, reinforcing that a favorable small split is not reliable evidence.
- The fixed WDBC noise study completed ideal, 1,024-shot, and 1% depolarizing
  conditions in 40.0 seconds, each with 4,840 circuit executions.
- Classification metrics happened to remain unchanged on this fold (balanced
  accuracy 0.8958; sensitivity 0.8750; specificity 0.9167; AUROC 0.9896), but the
  mean absolute Gram-matrix delta was 0.00575 for finite shots and 0.00993 for
  depolarizing noise. Both noisy matrices had 20 negative eigenvalues; this numerical
  violation is reported and is not silently repaired.

Commands:

```powershell
.\.venv\Scripts\python.exe -m qml_research run --config configs\heart_checkpoint.yaml
.\.venv\Scripts\python.exe -m qml_research run --config configs\noise_study.yaml
```

Observation and risk: unchanged classifications on one small fold do not establish
noise robustness. No real-QPU job was submitted, and simulator time is not a QPU
speedup measurement.

### Checkpoint 5: reports, recommendation, and reproducibility audit

Completed:

- The literature matrix contains 15 sources and 21 fields, including separate
  author claims, demonstrated evidence, project interpretation, reproducibility,
  limitations, and advantage assessment.
- Generated 121 completed model records, zero failed model records, normalized and
  summary tables, exhaustive matched deltas, deterministic 2,000-resample intervals,
  direct noise-kernel deltas, and all 11 required figures.
- Generated `experiment-results.md` and an evidence-gated MVP recommendation from
  saved JSON records. The recommendation is WDBC + Angle QKSVM + two PCA features,
  compared against all four classical baselines on `default.qubit`.
- The recommendation explicitly says quantum advantage was not demonstrated and
  postpones real hardware, clinical claims, imaging, genomics, and web integration.
- Ruff formatting/lint and all 18 routine tests passed, including end-to-end runner,
  plot, and report regeneration tests.
- Rechecked `quantum-helix-lab-full`; its pre-existing modified/untracked files were
  unchanged by this work. No file in that nested application was edited.

Commands:

```powershell
.\.venv\Scripts\python.exe -m qml_research report
.\.venv\Scripts\python.exe -m qml_research plot
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest -m "not slow"
git -c safe.directory=C:/Users/seera/Desktop/Exploration/Rit_Grover_project/quantum-helix-lab-full status --short --branch
```

Final verification results: 40 Python files formatted, Ruff clean, 18 tests passed
in 11.74 seconds, 121 completed raw model records, zero failed model records, and 11
PNG figures.

## Remaining limitations and Phase 2 gate

- Biomedical checkpoint evidence is limited to seed 42 and fold 0. Means,
  standard deviations, and t-interval columns are generated, but confirmatory
  uncertainty requires executing the pre-registered three-seed/five-fold configs.
- WDBC is diagnostic breast-mass classification, not pre-symptomatic screening.
- Heart is a small public dataset with no external validation; it is only a
  cross-dataset engineering check.
- Prediction-level bootstrap intervals are descriptive and do not remove fold or
  model-selection dependence.
- Real-hardware work is restricted to a future frozen, small inference experiment
  behind an explicit credential, cost, and confirmation gate.
