# QML MVP Recommendation

## Evidence status

The strongest observed checkpoint QML balanced accuracy is 0.938 for QKSVM with Angle encoding and 2 PCA features. The strongest matched classical balanced accuracy is 0.938 for logistic_regression on the same subset, fold and reduced features.

**Quantum advantage has not been demonstrated.** Phase 1 uses classically simulated,
small-data circuits and does not establish computational speedup, clinical benefit,
or superiority beyond the tested folds.

## Current integration recommendation

- Primary disease dataset: WDBC for a transparent diagnostic demonstration; retain
  Heart Disease as a cross-dataset robustness check.
- Candidate QML path: QKSVM with Angle encoding and 2 PCA features.
- Classical references: logistic regression, linear SVM, RBF SVM, and random forest,
  including full-feature results alongside quantum-matched features.
- Simulator: analytic PennyLane `default.qubit`; demonstrate finite-shot and simple
  depolarizing sensitivity separately and label it clearly.
- Phase 2 hardware experiment: freeze preprocessing, the trained model/configuration,
  and a very small inference batch; submit only after an explicit credential and cost
  confirmation gate. Do not perform real-hardware training.

## Quantum Helix Lab integration boundary

Integrate a QML experiment/result view backed by the stable YAML/JSON schema, dataset
provenance, matched classical comparison, resource telemetry, and scientific warnings.
Keep deterministic mutation/search analysis distinct from learned disease prediction.

Postpone clinical uploads, MedMNIST, gene-expression data, automated medical advice,
real-QPU training, and any "early detection" or "quantum advantage" claim.

## Claims we can safely make

- Phase 1 implements reproducible, leakage-aware quantum-kernel and VQC experiments
  with four classical baselines on public Iris, WDBC, and Heart Disease data.
- Selected exploratory configuration: QKSVM with Angle encoding and 2 PCA features.
- On its fixed WDBC fold, that configuration reached the same balanced accuracy as
  its strongest matched classical comparator.
- Finite shots and depolarizing noise changed the saved Gram matrices even where the
  small fold's predicted classes happened to remain unchanged.
- Circuit executions and device-level resource counts are measured and exposed for
  audit; they are not presented as hardware speedups.

## Claims we must avoid

- Quantum advantage, computational speedup, clinical superiority, or deployment
  readiness.
- Pre-symptomatic or longitudinal "early detection" from WDBC.
- Noise robustness from one fixed fold, or real-hardware performance from simulation.
- Generalization beyond the public benchmark populations or medical advice for an
  individual patient.

## Exact Phase 2 prerequisites

1. Complete all pre-registered seeds/folds or document the resource-limited subset.
2. Review the selected configuration against paired balanced accuracy, sensitivity,
   uncertainty, circuit counts, and noise degradation, not accuracy alone.
3. Freeze a versioned inference contract and reproduce it from a clean environment.
4. Add the adapter to Quantum Helix Lab on a separate integration branch while
   preserving authentication, provider confirmation, and scientific-boundary UI.
