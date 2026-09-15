# Dataset validation and QML campaign rationale

Audit date: 10 September 2026. This is exploratory benchmarking, not a clinical
prediction product. Per-file download manifests and per-dataset machine-readable
audits are retained in `data/raw/` and `results/campaign/audits/` respectively.

## Downloaded Kaggle datasets

| Dataset | Rows / features | Class counts | Validation and limitations |
| --- | --- | --- | --- |
| Framingham mirror | 4,238 / 15 | 3,594 negative; 644 positive | Expected 16-column schema including target; binary label; no duplicate complete rows; missingness retained for train-only imputation. Participant identity, outcome timing and censoring cannot be independently established. |
| Pima diabetes | 768 / 8 | 500 negative; 268 positive | Headerless CSV parsed explicitly. Every numeric value, label and row order matches the independent OpenML reference. No duplicate complete rows. Restricted demographic population. |

Framingham was downloaded from the public
[Kaggle dataset listing](https://www.kaggle.com/datasets/dileep070/heart-disease-prediction-using-logistic-regression)
using its public API. The downloaded file's SHA-256 is
`40921748dbdf6587045b9822414199a1ae3349358b63b975ab2208aa3cccc45b`.
The API reports an **unknown license**; metadata does not establish official cohort
provenance. Education has 105 missing values, cigarettes/day 29, BP medication 53,
total cholesterol 50, BMI 19, heart rate 1, and glucose 388. Missing predictors are
imputed within training folds; rows are not dropped to improve apparent scores.
The `TenYearCHD` column is evaluated as supplied, without inventing follow-up dates.

This mirror is deliberately named `framingham_kaggle`, not the repository's official
Framingham longitudinal teaching-cohort task. [BioLINCC teaching datasets](https://biolincc.nhlbi.nih.gov/teaching/)
require a request and are anonymized teaching resources with stated research-use
limitations. No official teaching data or participant linkage was obtained here.
Consequently, mirror results cannot validate the pre-existing longitudinal IHD
endpoint or establish clinical usefulness.

The original `uciml/pima-indians-diabetes-database` download returned HTTP 403.
The public [Kaggle CSV mirror](https://www.kaggle.com/datasets/kumargh/pimaindiansdiabetescsv)
was downloaded instead; Kaggle reports CC0. Its SHA-256 is
`c1e0f7d59d2e366cff1ab9fee1796e1e381ea5da64c6d96ef4baabf1eda5f238`.
It was cross-checked against [OpenML dataset 37](https://www.openml.org/api/v1/json/data/37)
and its [ARFF reference](https://openml.org/data/v1/download/37/diabetes.arff).
The reference and comparison result are saved beside the CSV. Zero glucose,
blood pressure, skin thickness, insulin and BMI are treated as missing under a
fixed preprocessing rule. Zero pregnancies is valid. This dataset describes adult
women of Pima heritage and a diabetes-status label; it is not a newly verified
prospective risk cohort.

Downloads can be retrieved through
`https://www.kaggle.com/api/v1/datasets/download/<owner>/<slug>` using the owner/slug
in the above links. Retain the existing manifests: an upstream version change must
produce a new dataset identity and campaign, rather than silently replacing inputs.
Raw CSV files are not committed or redistributed. File hashes are checked before
campaign use. Schema/hash checks establish reproducibility, not clinical validity.

## Existing authoritative benchmark data

WDBC uses the 569-row
[UCI diagnostic breast cancer dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic).
Cleveland uses the 303-row processed Cleveland subset of
[UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease), with disease
grades binarized and nominal features encoded separately. These are diagnostic
benchmarks, not a substitute for longitudinal cardiovascular follow-up.
BreastMNIST uses all 780 images with the published 546/78/156 partitions from
[MedMNIST](https://medmnist.com/). Its label is reversed consistently so malignant
is positive. Small samples and benchmark reuse limit generalization claims.

## Research-informed model changes

The design follows the need for strong classical comparisons and entanglement
ablations highlighted by [Bowles, Ahmed and Schuld (2024)](https://arxiv.org/abs/2403.07059).
That benchmark reports frequent classical wins; therefore this campaign includes
full-feature and PCA-matched RBF SVMs, logistic regression, forests, boosting,
MLPs and a native-image CNN. It also tests an unentangled reuploading ablation.

Input bandwidth is tuned on validation data for fidelity and projected kernels.
Projected features comprise local X/Y/Z expectations followed by a classical RBF
head, motivated by the construction explained in
[PennyLane's projected-kernel tutorial](https://pennylane.ai/demos/tutorial_huang_geometric_kernel_difference).
This is a model-design rationale, not evidence that projection guarantees a gain.
Quantum states are computed once per input and reused to form exact overlaps,
avoiding repeated pairwise circuit evaluation. Tests compare cached matrices and
projected expectations against direct circuits to numerical precision.

The new VQC repeatedly encodes inputs with trainable scales and shallow rotations,
inspired by [Pérez-Salinas et al., data re-uploading](https://quantum-journal.org/papers/q-2020-02-06-226/).
Both entangled and separable versions use the same tuning grid. The hybrid MLP's
quantum layer now batches circuit inputs; its outputs and gradients were checked
against the previous per-sample computation. This improves execution efficiency
without changing the mathematical model.

The report uses paired outer-fold differences and an approximate corrected
resampled interval inspired by [Nadeau and Bengio](https://papers.neurips.cc/paper_files/paper/1999/hash/7d12b66d3df6af8d429c1a357d8b9e1a-Abstract.html),
plus a multiplicity adjustment. Repeated folds share data, so ordinary independent
sample confidence intervals would be misleading. Even the corrected analysis is
within-dataset evidence; external validation remains absent.

Checkpointing includes optimizer and model state as described by
[PyTorch's checkpoint guidance](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html),
plus RNG, batch order/cursor and early-stopping state. Automated interruption tests
verify bitwise-identical final weights and equal histories after resuming classical
MLP/CNN, hybrid QMLP, VQC and QCNN runs in this fixed CPU environment.

## Evidence available now

See the continuously regenerated
[measured-results file](QML_MODELS_WITH_BETTER_METRICS.md), which includes negative
results and counts. The first WDBC split gave Angle-kernel AUROC 0.999669 versus
0.998677 for the validation-selected PCA-RBF comparator, with identical balanced
accuracy 0.976190. A difference on one small test partition is not a robust advantage.
The Angle kernel also equals a classical product-of-cosines kernel, so its predictive
score cannot establish a uniquely quantum benefit.

All 80 repository tests passed after integrating resume support (103.41 seconds;
guarded peak RSS 0.574 GiB). Search budgets and resource settings are documented in
the [campaign guide](../docs/RESUMABLE_CAMPAIGN.md). No metrics are guaranteed, and
no failed or unfavorable family is intentionally omitted from completed results.
