# Result record schema

Each `record.json` contains:

- identity: schema version, run ID, experiment, profile, dataset, model, seed, fold;
- reproducibility: resolved configuration, dataset/split manifest, Git state, Python/platform/hardware and package versions;
- evaluation: selected-threshold and default-0.5 metrics, ROC/PR arrays, confusion matrix, false positives/negatives, and bootstrap intervals;
- training: best epoch, stopping reason, duration, prediction duration, peak traced memory, and history artifact;
- preprocessing: reducer, number of features, training-fit count, and PCA variance when applicable;
- resources: backend, shots, differentiation, qubits, depth, gates, two-qubit gates, parameter counts, and circuit executions;
- boundary: explicit research-only and non-advantage language.

`predictions.csv` contains official dataset index, disease-positive label, malignant score, selected threshold, and prediction. Aggregate reports use locally completed records only; literature values remain separate.
