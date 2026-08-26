# Pre-registered Phase 1 Experiment Protocol

## Research questions and hypotheses

1. Can state-overlap quantum kernels and shallow VQCs be reproduced with auditable
   circuit counts? **Expected:** yes on binary Iris.
2. Do they improve biomedical classification over matched classical baselines?
   **Null:** paired balanced-accuracy differences include zero or are inconsistent.
3. How do feature count, reduction, dataset size and noise affect performance and
   resources? **Expected:** cost rises rapidly; no direction of accuracy effect is
   assumed.
4. Is one configuration sufficiently reproducible and compact for an SIH MVP?
   **Decision:** balanced accuracy, sensitivity, uncertainty, stability and resources
   jointly determine feasibility.

## Datasets and labels

- Iris: first two classes, 100 samples, four numeric features; sanity only.
- WDBC: 569 samples, 30 numeric features; malignant remapped to positive `1`.
- UCI Cleveland Heart Disease: 303 samples, 13 mixed features; target values 1-4
  mapped to disease-present `1`, target 0 to `0`; median/mode imputation is fold-local.

Public authoritative loaders and provenance are mandatory. WDBC must never be called
pre-symptomatic early detection.

## Splitting and preprocessing

- Stratified shuffled five-fold CV with seeds 42, 123 and 2026.
- Sample-size experiments use deterministic stratified subsets of 50, 100, 200 and
  full data. Quantum and matched classical models reuse exact subset/fold indices.
- Numerical pipeline: median imputation, standard scaling.
- Categorical pipeline: most-frequent imputation, one-hot encoding.
- Reduction: PCA or seeded mutual information selecting 2, 4, 6 or 8 dimensions.
- Reduced features are training-fitted to `[-pi, pi]` for angle compatibility; matched
  classical models use exactly those values.
- Full classical references use all fold-local preprocessed features.

No imputer, encoder, scaler, PCA, selector or threshold is fitted on validation data.

## Models and fixed hyperparameters

- Logistic regression: `C=1`, balanced class weights, liblinear, 2,000 iterations.
- Linear SVM: `C=1`, balanced class weights.
- RBF SVM: `C=1`, `gamma=scale`, balanced class weights.
- Random forest: 500 trees, square-root feature sampling, balanced class weights.
- QKSVM: raw precomputed fidelity Gram matrix, `C=1`, balanced class weights;
  AngleEmbedding baseline and one-repeat IQPEmbedding comparison.
- VQC: Y-angle embedding, one/two strongly-entangling layers, Pauli-Z output plus
  bias, binary cross-entropy, Adam learning rate 0.05, 100 steps, batches up to 16,
  parameter-shift gradients and seeded initializations.

Hyperparameters are pre-registered rather than chosen on test folds. WDBC screening
is explicitly exploratory. Ties use mean balanced accuracy, then sensitivity, then
lower execution/runtime cost. Confirmatory results remain labelled after selection.

## Execution sequence and resource gate

1. Reproduce the published Iris tutorial behavior, then report only a corrected
   fold-local version in comparisons.
2. Screen WDBC on 100 samples, seed 42, across feature/reduction/map/depth settings.
3. Run the selected QKSVM/VQC and all baselines at increasing dataset sizes on all
   seeds/folds.
4. Transfer the selected configuration to Heart Disease and repeat matched tests.
5. Run a fixed 100-sample WDBC fold under analytic ideal, 1,024-shot, and 1,024-shot
   1% single-qubit depolarizing conditions. Retrain the SVM per measured kernel.

Runs stop visibly when the configured wall-time budget is reached. Full-data QML is
attempted only when extrapolation from the 200-sample configuration is no more than
eight hours per model family. Classical full-data results still run. Any reduction is
reported with the same classical subset, never compared to a different split.

## Outcomes

Primary metric: balanced accuracy. Safety metric: sensitivity. Secondary metrics:
accuracy, precision, specificity, F1, MCC, AUROC, AUPRC, confusion matrix, training
time and prediction time. Quantum telemetry: qubits, circuit depth, total/two-qubit
gates, trainable parameters, shots, backend and actual executions.

Report means, sample standard deviations and descriptive 95% t-intervals across
available fold records. For a claim that one model is meaningfully better, require:

- at least three seeds with the pre-registered folds;
- a paired balanced-accuracy improvement of at least 0.02;
- a 95% paired interval excluding zero;
- no material sensitivity reduction; and
- no cherry-picked initialization or hidden failure.

Because repeated-CV folds are correlated and cohorts are small, even this criterion
supports an experimental result—not clinical efficacy or asymptotic quantum advantage.

## Failure and integrity policy

- Serialize exceptions, non-convergence, PSD violations, time limits and skipped runs.
- Do not repair indefinite kernels without a separately named sensitivity experiment.
- Do not tune on test folds, select one seed, or compare different data.
- Do not call simulator time QPU speed, fewer parameters lower total computation, or
  an accuracy difference quantum advantage.
- All tables and figures must regenerate from raw JSON/NPZ evidence and configurations.

