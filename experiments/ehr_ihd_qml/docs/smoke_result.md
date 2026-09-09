# Verified UCI smoke result

Run ID: `uci-smoke-20260829T194302Z-3b55c98a85f0`

The generated evidence remains under `results/ehr_ihd_qml/<run_id>/` and is ignored by Git because
it contains fitted state and patient-level split/prediction artifacts. This committed summary traces
the verified aggregate results without committing those artifacts.

## Data and selection

- Source: official UCI Heart Disease dataset 45 via `ucimlrepo`, DOI `10.24432/C52P4X`.
- Cached SHA-256: `6843d7c31598a05caa6e0968d6e95d803af235a048f69d5e8f7dd3a80808747e`.
- Cohort: 303 Cleveland rows; 164 negative and 139 positive; no duplicate rows.
- Missing values: `ca` 4 and `thal` 2; all other smoke fields 0.
- Split: train 212/97 positive, validation 45/21 positive, test 46/21 positive.
- Smoke four-field consensus: `thal`, `cp`, `thalach`, `ca`.
- Train-only paper-like ten-field consensus (selection only, not a ten-qubit performance run):
  `thal`, `thalach`, `cp`, `ca`, `exang`, `oldpeak`, `chol`, `sex`, `age`, `restecg`.
- mRMR status: mutual-information fallback because no maintained mRMR package is installed.

## Held-out aggregate metrics

All probabilities use validation-only Platt calibration and all thresholds are selected on validation
data. Values below are descriptive for one seed and one 46-person test split.

| Model | AUROC | AUPRC | Balanced accuracy | Brier | ECE | Train seconds | Circuit/kernel evaluations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic | 0.909 | 0.875 | 0.845 | 0.133 | 0.138 | 0.007 | 0 |
| Linear SVM | 0.910 | 0.878 | 0.865 | 0.132 | 0.163 | 0.049 | 0 |
| RBF SVM | 0.897 | 0.896 | 0.852 | 0.154 | 0.150 | 0.004 | 0 |
| Random forest | 0.899 | 0.894 | 0.736 | 0.173 | 0.164 | 0.649 | 0 |
| Classical MLP | 0.619 | 0.663 | 0.574 | 0.247 | 0.011 | 0.054 | 0 |
| Pauli OQSVM | 0.775 | 0.739 | 0.650 | 0.211 | 0.128 | 65.011 | 5,825 |
| Preferred BCE HQMLP | 0.621 | 0.583 | 0.572 | 0.235 | 0.054 | 3.888 | 376 |
| Paper-loss HQMLP | 0.619 | 0.558 | 0.572 | 0.239 | 0.041 | 4.592 | 376 |

The Pauli OQSVM 95% patient-bootstrap intervals were AUROC 0.602-0.890 and AUPRC 0.505-0.896
(200 valid resamples). Preferred HQMLP intervals were AUROC 0.456-0.771 and AUPRC 0.409-0.765.
Paper-loss HQMLP intervals were AUROC 0.448-0.779 and AUPRC 0.384-0.758. These wide intervals
reinforce that the smoke run cannot establish model superiority.

## Exact quantum configurations and cost

- OQSVM: 4 selected fields/qubits, Pauli Z/ZZ feature map, two repetitions, linear entanglement,
  exact PennyLane `default.qubit` statevector, no shots, validation-AUPRC-selected `C=0.1`, 50
  balanced training-only rows,
  1,275 unique training pairs and 5,825 total train/validation/test kernel evaluations. Circuit depth
  was 38 with 68 gates, including 24 two-qubit gates. The support-vector counts were 23/19 and
  kernel-target alignment was 0.151.
- Preferred HQMLP: four-field classical projection, 4 qubits, one `RY/RZ` plus linear-CNOT layer,
  BCE logits, Adam 0.01, weight decay 0.0001, three epochs, 33 trainable parameters (8 quantum),
  circuit depth 6, and 376 recorded total forward evaluations after final inference/explanation.
- Paper-inspired HQMLP: the same input scope with `Rot`/CNOT ansatz, sigmoid MSE ablation, Adam
  0.01, three epochs, 37 parameters (12 quantum), circuit depth 5, and 376 recorded evaluations.

The validation-only QSVM tuning compared `C=0.1` (validation AUPRC 0.800) with `C=1.0` (0.778),
recorded `test_set_evaluated: false`, and froze `C=0.1` before test evaluation.

Total end-to-end runtime was 165.77 seconds. Peak recorded process RSS was about 461 MB. The
reloadable logistic prediction example completed with a calibrated probability and research warning.

## Interpretation

RBF-SVM had the highest descriptive test AUPRC. Both quantum approaches were below the strongest
classical models, and the quantum methods required much more computation. This bounded result is
evidence against quantum predictive utility in this particular smoke configuration. It is not proof
that no QML configuration can help. It provides no computational quantum advantage evidence because
the circuits ran on an ideal classical statevector simulator, the dataset and split are small, only
one seed was used, and matched repeated scaling/hardware experiments were not performed.

The Framingham track was not executed because the verified official teaching dataset was absent. No
Framingham cohort count, censoring count, selected feature, or performance metric was fabricated.
