Continue working in my existing QML research repository. Create a separate experiment for ischemic/coronary heart disease classification and early risk prediction using the optimized QSVM and hybrid QMLP architectures inspired by the following paper:

- [Quantum Machine Learning Applied to Electronic Healthcare Records for Ischemic Heart Disease Classification](https://hcisj.com/data/file/article/2023020001/13-06.pdf)
- DOI: `10.22967/HCIS.2023.13.006`

Use the research decisions and resources supplied in this prompt. Do not repeat a broad literature search. You may consult the linked papers, official dataset documentation and installed library documentation only when an implementation detail requires clarification.

Build this as an isolated experiment:

```text
experiments/ehr_ihd_qml/
```

Do not overwrite or mix its results with my previous:

- QCNN breast-cancer experiment;
- EHR AKI experiment;
- EHR mortality experiment;
- genomic experiments;
- earlier QSVM/QMLP results.

Reuse tested utilities from earlier experiments where appropriate, including:

- QSVM and quantum-kernel code;
- MLP/QMLP layers;
- metrics;
- plotting;
- explainability;
- configuration loading;
- caching;
- checkpointing;
- reproducibility;
- runtime and memory measurement.

Do not blindly copy an entire previous experiment. Keep the datasets, labels, preprocessing, configurations, results and documentation specific to this heart-disease experiment.

Before modifying files:

1. Inspect the current repository and branch.
2. Check for uncommitted user changes.
3. Locate existing QSVM, SVM, MLP and QMLP code.
4. Check Qiskit, PennyLane, PyTorch and scikit-learn versions.
5. Inspect CPU, RAM and disk availability.
6. Preserve all existing work.
7. Do not push remotely.
8. Start with a bounded smoke run suitable for an 8 GB RAM laptop.

# 1. Experiment objective

Implement two clearly separated tracks.

## Track A: Paper-inspired heart-disease classification

Use the UCI Heart Disease dataset to classify whether heart disease is already present.

```text
Clinical measurements and diagnostic-test results
    → preprocessing
    → feature selection
    → classical models
    → optimized QSVM
    → hybrid QMLP
    → heart-disease presence classification
```

This track is an immediate public-data reproduction and functionality benchmark.

It must be called:

> Heart-disease presence classification

Do not call it early disease prediction because the label describes disease already present during evaluation.

## Track B: Genuine early coronary heart-disease prediction

Use longitudinal Framingham data to predict a first coronary heart-disease event among participants who are free of CHD at the index examination.

```text
Disease-free participant at index exam
    → baseline or longitudinal risk factors
    → 10-year follow-up
    → predict first coronary event
```

This track must be called:

> Ten-year incident coronary heart-disease risk prediction

The main scientific question is:

> Can optimized QSVM and hybrid QMLP models predict future coronary heart disease as well as or better than strong feature-matched classical models?

The project must never describe a risk prediction as:

- a confirmed diagnosis;
- treatment advice;
- a clinically validated screening system;
- a replacement for ECG, angiography or clinical examination;
- evidence of quantum advantage without supporting experiments.

# 2. Supplied research basis

Create:

```text
experiments/ehr_ihd_qml/docs/research_basis.md
```

Summarize the following research without conducting another broad literature search.

## 2.1 What ischemic heart disease means

Ischemic heart disease, also called coronary heart disease or coronary artery disease in many contexts, occurs when the heart muscle receives insufficient blood and oxygen because the coronary arteries are narrowed or blocked.

Relevant clinical outcomes include:

- stable or unstable angina;
- coronary insufficiency;
- myocardial infarction;
- fatal coronary heart disease.

Do not combine unrelated conditions such as:

- congenital heart disease;
- valvular disease;
- arrhythmia;
- myocarditis;
- heart failure from any cause.

If a dataset uses the broader phrase “heart disease,” document the exact outcome represented by that dataset instead of silently calling every positive case ischemic heart disease.

## 2.2 Primary QML paper

Study and reproduce the relevant design from:

- [Maheshwari et al., 2023](https://hcisj.com/data/file/article/2023020001/13-06.pdf)

The paper used a private EHR cohort from patients presenting with chest pain at Basurto University Hospital.

Important design details:

- approximately 2.2 million initial EHR records were considered;
- 43,835 chest-pain patients were identified;
- 33,372 records remained after exclusions;
- approximately 16.1% developed coronary artery disease;
- the original representation contained 82 variables;
- 42 variables remained after cleaning;
- feature selection used:
  - logistic-regression RFE;
  - random forest;
  - mRMR;
  - an intersection/consensus of selected features;

- 10 final features were used;
- the OQSVM used a Pauli feature map;
- feature-map repetitions were reported as two;
- statevector simulation was used;
- the HQMLP used classical layers, quantum encoding, parameterized rotations, entanglement, Adam and MSE;
- an 80/20 train/test split was reported;
- the data was balanced and normalized.

Important limitations:

- the private dataset is unavailable for direct reproduction;
- some sample counts in the paper appear inconsistent;
- validation-based hyperparameter selection is insufficiently described;
- feature-selection leakage cannot be ruled out;
- balancing appears to occur before or around splitting;
- accuracy is emphasized more than calibration or AUPRC;
- the model was evaluated on an ideal statevector simulator;
- classical SVM and MLP results were at least as good as, or better than, the OQSVM and HQMLP results;
- the paper does not establish computational quantum advantage.

Use its architecture as inspiration, but improve the experimental methodology.

## 2.3 Longitudinal cardiovascular prediction

Use:

- [Learning from Longitudinal EHR and Genetic Data to Improve Cardiovascular Event Prediction](https://www.nature.com/articles/s41598-018-36745-x)

Important findings:

- longitudinal EHR data can improve 10-year cardiovascular-event prediction;
- time-aware features can add value beyond a single snapshot;
- important features included age, blood pressure, cholesterol, BMI, creatinine, glucose and medication exposure;
- AUROC alone is insufficient; AUPRC also matters;
- temporal features must be constructed without using future outcome data.

## 2.4 EHR coronary-disease prediction

Use:

- [Coronary Heart Disease Prediction From EHRs](https://medinform.jmir.org/2020/7/e17257)

Document that EHR-based coronary prediction usually requires:

- an explicit index date;
- exclusion of prevalent disease;
- a future prediction horizon;
- patient-level separation;
- careful management of censoring;
- clinically meaningful calibration and decision thresholds.

## 2.5 QML evidence

Use:

- [Quantum kernels for EHR classification](https://research.ibm.com/publications/quantum-kernels-for-electronic-health-records-classification)
- [Clinical data classification with NISQ computers](https://www.nature.com/articles/s41598-022-05971-9)
- [Systematic review of QML for digital health](https://www.nature.com/articles/s41746-025-01597-z)
- [Better Than Classical? Benchmarking QML](https://arxiv.org/abs/2403.07059)

The current evidence does not consistently show that QML outperforms classical models on health data.

Therefore:

- do not assume QSVM is better than RBF-SVM;
- do not assume QMLP is better than classical MLP;
- compare identical patients and features;
- measure runtime and circuit cost;
- use multiple seeds;
- report uncertainty;
- accept negative results.

# 3. Dataset strategy

## 3.1 Track A: UCI Heart Disease

Use the official dataset:

- [UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart%2Bdisease)

Do not substitute an unverified Kaggle copy when the official source is available.

The dataset contains 76 documented attributes, but the standard experiments use 13 predictors plus the outcome.

Use these standard variables:

| Field      | Meaning                            |
| ---------- | ---------------------------------- |
| `age`      | Age                                |
| `sex`      | Sex                                |
| `cp`       | Chest-pain type                    |
| `trestbps` | Resting blood pressure             |
| `chol`     | Serum cholesterol                  |
| `fbs`      | Fasting blood sugar indicator      |
| `restecg`  | Resting ECG result                 |
| `thalach`  | Maximum heart rate achieved        |
| `exang`    | Exercise-induced angina            |
| `oldpeak`  | ST depression from exercise        |
| `slope`    | ST-segment slope                   |
| `ca`       | Number of major vessels visualized |
| `thal`     | Thalassemia/test category          |
| `num`      | Heart-disease diagnostic outcome   |

Primary binary label:

```text
num == 0       → no heart disease detected
num in 1..4    → heart disease present
```

Also preserve the original `0–4` severity label for optional multiclass analysis, but do not make multiclass prediction part of the smoke acceptance criteria.

Start with the Cleveland subset because it is the most established version.

Optionally add:

- Hungarian;
- Switzerland;
- Long Beach VA.

If multiple sites are combined:

- add a dataset-source field for auditing;
- do not use the site ID as a predictor by default;
- report site-specific missingness;
- perform leave-one-site-out validation where feasible;
- do not silently merge incompatible encodings.

The UCI track is a diagnostic classification benchmark, not longitudinal EHR prediction.

## 3.2 Track B: Framingham longitudinal teaching data

Use the official NHLBI/BioLINCC teaching dataset:

- [Teaching-dataset access page](https://biolincc.nhlbi.nih.gov/teaching/)
- [Framingham longitudinal documentation](https://biolincc.nhlbi.nih.gov/media/teachingstudies/FHS_Teaching_Longitudinal_Data_Documentation_2021a.pdf)
- [Framingham cohort page](https://biolincc.nhlbi.nih.gov/studies/framcohort/)

The teaching dataset contains:

- 4,434 participants;
- 11,627 examination rows;
- up to three clinic examinations;
- examinations approximately six years apart;
- up to 24 years of follow-up;
- adjudicated coronary and cardiovascular outcomes.

The teaching dataset is anonymized for instruction and is explicitly unsuitable for publication-quality research. It is acceptable for:

- learning;
- reproducibility demonstrations;
- hackathon experiments;
- pipeline validation.

Document this limitation prominently.

Do not use unofficial `framingham.csv` files unless their provenance, schema and label definitions are verified against official documentation.

If the official teaching dataset is not yet locally available:

1. Implement and test the loader against the documented schema.
2. Provide the expected local path.
3. Provide instructions for requesting/downloading it.
4. Fully execute the UCI smoke track.
5. Do not fabricate Framingham results.
6. Clearly report that the early-prediction track is awaiting data.

## 3.3 Future real-EHR adapter

Prepare an adapter specification for future MIMIC-IV experimentation using:

- ICD-10 `I20–I25`;
- ICD-9 `410–414`.

However, do not use MIMIC-IV as the primary incident-CHD dataset in this task because:

- it is hospital-focused rather than a community screening cohort;
- diagnoses may be assigned at discharge;
- diagnosis timestamps may not represent true disease onset;
- out-of-network events are not observable;
- many patients are already symptomatic or critically ill.

Do not use MIMIC-IV-Ext Cardiac Disease as a standalone binary diagnostic dataset because it primarily contains positive cardiac cases and includes information from the complete diagnostic process, creating control-selection and leakage problems.

# 4. Track A task definition

## UCI diagnostic task

Predict whether the UCI `num` field is zero or greater than zero.

Do not use `num` as an input.

Use a deterministic train/validation/test split:

```text
70% training
15% validation
15% testing
```

Because the dataset is small, additionally run repeated stratified cross-validation on the development data.

Recommended:

```yaml
outer_repeats: 5
outer_folds: 5
inner_folds: 3
```

Keep a fixed untouched test set for the final smoke report.

All operations must occur inside the correct training fold:

- imputation;
- encoding;
- scaling;
- feature selection;
- balancing;
- hyperparameter tuning.

If multiple UCI sites are used, include a leave-one-site-out experiment as a separate generalization analysis.

# 5. Track B early-prediction task

## 5.1 Primary endpoint

Use baseline examination `PERIOD == 1`.

Include only participants with:

```text
PREVCHD == 0
```

Primary outcome:

> Any first coronary heart-disease event within 10 years.

Use:

```text
ANYCHD
TIMECHD
```

The `ANYCHD` outcome includes:

- angina pectoris;
- myocardial infarction;
- coronary insufficiency/unstable angina;
- fatal coronary heart disease.

Construct:

```text
positive:
    ANYCHD == 1
    and TIMECHD <= 3652.5 days

negative:
    no ANYCHD event before 3652.5 days
    and follow-up/censor time >= 3652.5 days

censored/unknown:
    no observed event
    and follow-up ended before 3652.5 days
```

Exclude censored/unknown participants from the binary 10-year experiment.

Do not convert insufficient follow-up into a negative label.

## 5.2 Secondary endpoint

Use:

```text
MI_FCHD
TIMEMIFC
```

Predict:

> Hospitalized myocardial infarction or fatal coronary heart disease within 10 years.

This is a more specific ischemic endpoint but will have fewer positive events.

## 5.3 Longitudinal landmark experiment

Create an optional longitudinal experiment using the second examination as the index.

Include participants who:

- attended Period 1 and Period 2;
- remained free of prevalent CHD at Period 2;
- have adequate future follow-up after the Period 2 examination.

Construct features from Periods 1 and 2:

- first value;
- current value;
- absolute change;
- percentage change;
- slope per year;
- medication-status change;
- smoking-status change;
- diabetes-status change;
- hypertension-status change.

Calculate outcome time relative to Period 2:

```text
relative_event_time = TIMECHD - TIME_at_period_2
```

Do not use Period 3 measurements to predict an event occurring after Period 2.

This longitudinal track is the strongest early-prediction demonstration because it shows how changing risk factors influence future model risk.

# 6. Leakage prevention

## UCI leakage

Avoid:

- fitting imputers on the complete dataset;
- scaling before splitting;
- using feature selection on test data;
- choosing hyperparameters from test accuracy;
- balancing validation or test sets;
- manually choosing the “best” random split.

## Framingham leakage

Never use these as predictors:

- `ANYCHD`;
- `ANGINA`;
- `HOSPMI`;
- `MI_FCHD`;
- `CVD`;
- `DEATH`;
- `TIMECHD`;
- `TIMEAP`;
- `TIMEMI`;
- `TIMEMIFC`;
- `TIMECVD`;
- `TIMEDTH`;
- future examination values;
- disease-prevalence values measured after the index date;
- variables derived from future follow-up;
- `glucoseyear6` in the Period 1 baseline experiment;
- simulated fields such as `nhosp`.

Prevalent disease fields may be used only for cohort exclusion unless their use as predictors is clinically and temporally justified.

Add automated feature-name and timestamp leakage checks.

# 7. Data validation

For each dataset, generate:

- source URL;
- dataset version/date;
- checksum where available;
- raw row count;
- patient count;
- duplicate rows;
- duplicate patient IDs;
- missingness by feature;
- class distribution;
- categorical values;
- numeric ranges;
- implausible values;
- split counts;
- outcome/censoring counts.

For UCI:

- parse `?` as missing;
- check `ca` and `thal` carefully;
- do not drop all incomplete records automatically;
- compare complete-case and imputed analyses.

For Framingham:

- treat periods, negative sentinel values and documented missing markers correctly;
- confirm one row per participant per period;
- confirm time increases across periods;
- validate that prevalent disease is consistent with event time;
- check the relationship between `ANYCHD` and `TIMECHD`;
- record early censoring;
- confirm event times are after the selected index examination.

Create:

```text
experiments/ehr_ihd_qml/docs/data_dictionary.md
```

# 8. Preprocessing

Build scikit-learn-compatible pipelines.

## Numerical variables

Support:

- median imputation;
- missing-value indicators;
- standard scaling;
- robust scaling;
- min-max scaling to `[0,1]`;
- quantum angular scaling to `[0, π]` or another documented circuit range.

## Categorical variables

Use:

- explicit categorical typing;
- one-hot encoding for classical models;
- compact ordinal or learned encoding for quantum models only when scientifically defensible;
- unknown-category handling.

Do not impose ordinal meaning on nominal fields such as chest-pain category without documenting the effect.

Save:

- fitted imputer;
- category mapping;
- scaler;
- feature selector;
- final feature order.

# 9. Paper-inspired feature selection

Implement the paper’s wrapper/filter approach more rigorously.

Use:

1. Logistic-regression RFE.
2. Random-forest importance.
3. mRMR when safely available.
4. Mutual information as a fallback.
5. Consensus/intersection selection.

For each method:

- fit using training data only;
- rank features;
- save ranking and scores;
- calculate stability across folds/seeds.

Consensus options:

```yaml
consensus_method:
  - strict_intersection
  - rank_average
  - majority_top_k
```

Select:

```yaml
n_quantum_features: [4, 6, 8, 10]
```

The paper-like configuration must use 10 selected features.

If strict intersection returns fewer than the required features, fill the remaining positions using deterministic average rank and document this.

Also compare:

- no feature selection;
- PCA;
- clinically fixed risk-factor subset.

The classical controls must receive the same selected features as the quantum models.

# 10. Classical model suite

Train:

- dummy classifier;
- logistic regression;
- linear SVM;
- RBF SVM;
- K-nearest neighbours;
- Gaussian Naive Bayes;
- decision tree;
- random forest;
- Extra Trees;
- histogram gradient boosting;
- XGBoost or LightGBM if already installed;
- classical MLP.

For Framingham, add:

- Cox proportional-hazards model where dependencies permit;
- optional random survival forest;
- a simple conventional risk-factor logistic model.

Do not label a custom logistic model as the official Framingham Risk Score unless its exact published equation, endpoint and required predictors are implemented.

Use class weighting where appropriate.

# 11. Optimized QSVM/OQSVM

Implement the paper-inspired OQSVM using the maintained Qiskit Machine Learning API available in the repository environment.

Prefer:

- `FidelityStatevectorKernel` for exact local simulation when supported;
- `FidelityQuantumKernel` for shot/noise experiments;
- `QSVC` or `SVC(kernel="precomputed")`;
- optional `PegasosQSVC`.

Paper-like configuration:

```yaml
model: oqsvm
quantum_features: 10
feature_map: pauli
feature_map_reps: 2
backend: statevector
shots: null
```

Support:

```yaml
feature_map:
  - pauli
  - zz

paulis:
  - ["Z"]
  - ["Z", "ZZ"]
  - ["X", "Y", "Z"]
  - ["X", "Y", "Z", "ZZ"]

reps: [1, 2, 3]

entanglement:
  - linear
  - circular
  - full

C: [0.1, 1.0, 10.0]
```

Test the paper-inspired rotations, but do not claim an exact reproduction when the original gate sequence is underspecified.

Optimize using validation AUPRC or validation AUROC, never the test set.

Support:

- kernel centering;
- kernel normalization;
- PSD correction;
- kernel-target alignment;
- optional trainable quantum kernel when affordable.

Efficiency requirements:

- compute unique symmetric kernel entries only;
- cache using data, split, feature and circuit hashes;
- save train and test kernels separately;
- resume partial safe computations;
- record kernel dimensions;
- record unique kernel evaluations;
- estimate circuit calls before training;
- enforce configurable cost limits.

Report:

```text
n_train(n_train+1)/2
```

as the approximate number of unique training-kernel pairs.

# 12. Hybrid QMLP/HQMLP

Implement two variants.

## 12.1 Paper-faithful ablation

Approximate the paper’s reported architecture:

```text
10 selected features
    → classical linear layer
    → ReLU
    → optional pooling
    → fully connected projection
    → dropout
    → quantum encoding
    → U3/general rotations
    → entangling gates
    → measurement
    → classical output
```

Use:

- Pauli or angle encoding;
- U3 or equivalent `Rot` gates;
- CNOT entanglement;
- Adam;
- learning rate near `0.01`;
- MSE as a paper-reproduction ablation.

Because max-pooling a tabular feature vector is ordering-dependent, keep pooling optional and label it clearly as a paper-inspired ablation.

## 12.2 Scientifically preferred HQMLP

Use:

```text
Selected tabular features
    → small classical projection
    → normalized quantum latent vector
    → trainable variational circuit
    → expectation values
    → classical sigmoid/logit head
```

Support:

- 4, 6, 8 and 10 qubits;
- angle encoding;
- Pauli encoding;
- `RY`, `RZ` and general rotations;
- linear and circular entanglement;
- 1–3 quantum layers;
- multiple expectation-value outputs;
- Adam;
- early stopping;
- weight decay;
- dropout;
- class weights;
- checkpointing;
- resumable execution.

Preferred loss:

```yaml
loss: bce_logits
```

Paper ablation:

```yaml
loss: mse_paper_ablation
```

Record:

- classical parameters;
- quantum parameters;
- total trainable parameters;
- circuit depth;
- one-qubit gates;
- entangling gates;
- gradient method;
- forward circuit evaluations;
- gradient evaluations;
- optimizer steps;
- training time;
- inference time;
- peak memory.

# 13. Fair comparisons

Maintain separate tables.

## Reduced feature-matched comparison

Use identical participants and identical selected features for:

- logistic regression;
- linear SVM;
- RBF SVM;
- random forest;
- classical MLP;
- OQSVM;
- HQMLP.

## Full-feature practical comparison

Use all valid features for:

- logistic regression;
- RBF SVM;
- random forest;
- Extra Trees;
- gradient boosting;
- classical MLP.

Do not imply the reduced quantum model is better than a full-feature classical model unless the comparison is clearly labelled as input-unmatched.

For every matched comparison, preserve:

- same split;
- same feature values;
- same natural validation/test prevalence;
- same metric definitions.

# 14. Class imbalance

Do not balance validation or test data.

Preferred options:

- class-weighted loss;
- class-weighted SVM;
- validation-selected threshold;
- train-only random undersampling for expensive QSVM experiments.

If the QSVM uses a capped balanced training subset:

- select it only from training data;
- preserve natural validation/test prevalence;
- save chosen participant IDs;
- train matched classical controls on exactly the same subset;
- report both original and sampled distributions.

Do not apply SMOTE before splitting.

# 15. Probability and calibration

SVM decision scores are not automatically calibrated probabilities.

Support:

- Platt scaling;
- isotonic regression;
- temperature scaling for neural logits where appropriate.

Fit calibration using validation data only.

Report uncalibrated decision score and calibrated probability separately.

If the validation set is too small for reliable calibration, state that clearly.

# 16. Evaluation metrics

For every model, calculate:

- dataset track;
- prediction type;
- input representation;
- patient count;
- positive count;
- prevalence;
- feature count;
- trainable parameter count;
- accuracy;
- balanced accuracy;
- AUROC;
- AUPRC;
- precision;
- recall/sensitivity;
- specificity;
- NPV;
- F1;
- F2;
- Matthews correlation coefficient;
- Brier score;
- expected calibration error;
- selected threshold;
- training time;
- inference time;
- peak memory;
- seed;
- circuit/kernel evaluations.

For the Framingham survival analysis, additionally calculate:

- Harrell’s C-index;
- time-dependent 10-year AUROC when feasible;
- integrated Brier score when feasible.

Use AUPRC and AUROC together.

Select thresholds using validation data only.

Support:

```yaml
threshold_strategy:
  - youden_j
  - max_f1
  - max_f2
  - sensitivity_at_least_0_80
```

For a screening-style report, emphasize sensitivity, NPV, PPV and calibration.

Calculate 95% patient-level bootstrap confidence intervals.

Do not claim one model is better when intervals substantially overlap without an appropriate paired comparison.

# 17. Explainability

Explainability is required for classical, QSVM and QMLP predictions.

These explanations describe model behaviour and do not establish clinical causality.

## Classical models

Implement:

- logistic coefficients;
- tree-based feature importance;
- permutation importance;
- SHAP for supported tree models;
- partial-dependence plots for a few stable continuous features.

## QSVM

Implement:

- permutation importance;
- feature ablation;
- local perturbation;
- decision score;
- distance from decision boundary;
- support-vector count by class;
- closest support vectors;
- quantum-kernel similarity to representative training participants;
- kernel-target alignment;
- optional small-sample KernelSHAP.

Do not present the closest support vector as a clinically identical patient.

## QMLP

Implement:

- permutation importance;
- feature ablation;
- local perturbation;
- input gradients;
- integrated gradients when supported;
- quantum expectation values;
- quantum-parameter sensitivity;
- stability across seeds.

For selected participants, produce:

```text
Feature
Original value
Training reference range
Missingness status
Model sensitivity/importance
Direction of score change
```

Generate examples for:

- true positive;
- true negative;
- false positive;
- false negative.

For Framingham longitudinal prediction, show how the risk score changes from Period 1 to Period 2 without claiming that the difference is causal.

# 18. Prediction interface

Create a prediction interface accepting a validated JSON record.

UCI example:

```json
{
  "age": 58,
  "sex": 1,
  "cp": 2,
  "trestbps": 140,
  "chol": 240,
  "fbs": 0,
  "restecg": 1,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 1.2,
  "slope": 1,
  "ca": 0,
  "thal": 2
}
```

Framingham example:

```json
{
  "age": 52,
  "sex": 1,
  "sysbp": 138,
  "diabp": 84,
  "totchol": 235,
  "bmi": 27.1,
  "cursmoke": 1,
  "cigpday": 10,
  "diabetes": 0,
  "bpmeds": 0,
  "heartrte": 76,
  "glucose": 89
}
```

The interface must:

- validate all required fields;
- validate units and ranges;
- reject outcome fields;
- load the complete fitted pipeline;
- preserve feature order;
- produce model score;
- produce calibrated probability when calibration exists;
- apply the validation-selected threshold;
- return model name and run ID;
- return prediction task and time horizon;
- report missing inputs;
- optionally produce an explanation;
- display a research-only warning.

Framingham output example:

```json
{
  "prediction_task": "incident_chd_within_10_years",
  "risk_probability": 0.18,
  "threshold": 0.14,
  "predicted_high_risk": true,
  "model": "hqmlp",
  "dataset": "framingham_teaching",
  "warning": "Research demonstration only; not a diagnosis or treatment recommendation."
}
```

Do not call an uncalibrated quantum output a probability.

# 19. Hyperparameter YAML files

Create:

```text
experiments/ehr_ihd_qml/configs/uci_smoke.yaml
experiments/ehr_ihd_qml/configs/framingham_early_chd.yaml
experiments/ehr_ihd_qml/configs/paper_replication.yaml
experiments/ehr_ihd_qml/configs/laptop_8gb.yaml
experiments/ehr_ihd_qml/configs/workstation_24gb.yaml
experiments/ehr_ihd_qml/configs/tuning.yaml
```

## UCI smoke

Use:

```yaml
experiment:
  track: uci_diagnostic
  mode: smoke
  seed: 42

data:
  source: uci_official
  cohort: cleveland
  target: num_binary

resources:
  memory_limit_gb: 5.5
  max_runtime_minutes: 30
  max_kernel_evaluations: 20000
  bounded_parallelism: true

quantum:
  features: 4
  layers: 1
  backend: statevector
  shots: null
  real_hardware_submit: false

training:
  qmlp_epochs: 3
  tuning_trials: 2
  bootstrap_iterations: 200
```

## Paper replication

Use:

```yaml
experiment:
  track: paper_replication

feature_selection:
  method: consensus
  n_features: 10

qsvm:
  feature_map: pauli
  reps: 2
  backend: statevector

qmlp:
  quantum_features: 10
  ansatz: u3_cnot
  optimizer: adam
  learning_rate: 0.01
  loss: mse_paper_ablation
```

Also run the preferred BCE QMLP as a separate model.

## Framingham early CHD

Use:

```yaml
experiment:
  track: framingham_early_prediction

cohort:
  index_period: 1
  exclude_prevalent_chd: true

outcome:
  endpoint: anychd
  horizon_days: 3652.5
  censoring: exclude_incomplete_binary

feature_selection:
  method: consensus
  n_features: 10

model_selection:
  primary_metric: validation_auprc
```

## Eight-GB profile

Use:

- 4–8 quantum features;
- one or two quantum layers;
- bounded QSVM subset;
- 2–3 seeds;
- 15–30 QMLP epochs;
- early stopping;
- cached kernels;
- 1,000 bootstrap iterations;
- bounded parallelism;
- memory safety below physical RAM.

## Twenty-four-GB profile

Use:

- 4, 6, 8 and 10 quantum features;
- larger QSVM sample grids;
- 3–5 seeds;
- 30–60 QMLP epochs;
- more tuning trials;
- optional shot/noise experiments;
- 2,000–10,000 bootstrap iterations;
- resumable execution.

Do not assume 24 GB makes a full quantum-kernel matrix cheap.

## Tuning file

Use:

```yaml
search:
  strategy: random
  primary_objective: validation_auprc
  secondary_objective: validation_auroc
  max_trials: 18
  timeout_minutes: 120
  seed: 42

feature_selection:
  method:
    - mutual_info
    - rfe_logistic
    - consensus
    - clinical_fixed
  n_features: [4, 6, 8, 10]

qsvm:
  feature_map: [pauli, zz]
  paulis:
    - ["Z", "ZZ"]
    - ["X", "Y", "Z"]
    - ["X", "Y", "Z", "ZZ"]
  reps: [1, 2, 3]
  entanglement: [linear, circular]
  C: [0.1, 1.0, 10.0]
  psd_correction: [true, false]
  kernel_centering: [true, false]

qmlp:
  quantum_features: [4, 6, 8, 10]
  quantum_layers: [1, 2, 3]
  encoding: [angle, pauli]
  ansatz: [ry_rz_cnot, u3_cnot]
  entanglement: [linear, circular]
  learning_rate: [0.001, 0.003, 0.01]
  weight_decay: [0.0, 0.00001, 0.0001]
  batch_size: [8, 16, 32]
  dropout: [0.0, 0.1, 0.2]
  init_scale: [0.01, 0.05, 0.1]
  loss: [bce_logits, mse_paper_ablation]
```

Validate incompatible combinations and provide useful errors.

# 20. Quantum scaling and advantage analysis

Run a bounded grid:

```text
Training patients: 50, 100, 200, 300, 500
Features/qubits:   4, 6, 8, 10
```

Only run larger configurations after estimation.

Compare:

- QSVM with RBF-SVM;
- QSVM with linear SVM;
- QMLP with classical MLP;
- reduced quantum models with reduced classical models;
- reduced models with full-feature classical models.

Measure:

- AUROC;
- AUPRC;
- calibration;
- training time;
- inference time;
- peak memory;
- circuit/kernel evaluations;
- circuit depth;
- trainable parameters;
- variation across seeds.

Document:

```text
QSVM training pairs ≈ n(n+1)/2
```

Distinguish:

1. Predictive utility.
2. Sample efficiency.
3. Runtime/memory utility.
4. Hardware utility.
5. Asymptotic quantum advantage.

State explicitly:

- statevector simulation cannot establish computational quantum advantage;
- better accuracy on one small split is not quantum advantage;
- any advantage claim requires strong matched baselines;
- feature-selection improvements must also benefit classical controls;
- differences must be stable across seeds;
- uncertainty must be reported;
- negative findings are valid.

Create:

```text
experiments/ehr_ihd_qml/docs/quantum_advantage_limits.md
```

# 21. Notebook

Create:

```text
notebooks/ehr_ihd_qml_walkthrough.ipynb
```

It must call reusable project modules and run from top to bottom in UCI smoke mode.

Include:

1. Ischemic-heart-disease explanation.
2. Classification versus early prediction.
3. Dataset provenance.
4. Data-quality report.
5. Label construction.
6. Leakage prevention.
7. Feature selection.
8. Classical baselines.
9. OQSVM circuit.
10. HQMLP circuit.
11. Training.
12. Validation thresholding.
13. Final test metrics.
14. Calibration.
15. QSVM explanation.
16. QMLP explanation.
17. Example patient prediction.
18. Runtime and kernel/circuit counts.
19. Comparison with the paper.
20. Quantum-advantage conclusion.
21. Clinical limitations.

Show prominently:

```text
RESEARCH/EDUCATIONAL EXPERIMENT — NOT A MEDICAL DIAGNOSIS
```

If Framingham data is available, include the early-prediction results in a second notebook section.

# 22. Testing

Add tests for:

- UCI official download/loading;
- UCI `?` missing-value parsing;
- binary label conversion;
- Framingham period selection;
- prevalent CHD exclusion;
- 10-year endpoint construction;
- censored-case handling;
- Period 2 landmark time calculation;
- future-feature exclusion;
- patient-level split isolation;
- deterministic stratification;
- train-only imputation;
- train-only encoding;
- train-only scaling;
- train-only feature selection;
- consensus feature ranking;
- feature-order preservation;
- 4-, 6-, 8- and 10-feature quantum inputs;
- kernel symmetry;
- kernel-cache consistency;
- QSVM save/load;
- QMLP forward/backward pass;
- QMLP save/load;
- calibration save/load;
- prediction input validation;
- explanation output schema;
- YAML parsing and validation;
- result-directory non-overwrite behaviour.

Run:

- new unit tests;
- UCI smoke integration test;
- existing project tests;
- Python compilation;
- linting;
- formatting checks;
- dependency checks;
- secret scan;
- large-file check;
- notebook structural validation;
- automated notebook smoke execution.

# 23. Output structure

Use:

```text
results/ehr_ihd_qml/<run_id>/
```

Save:

```text
resolved_config.yaml
environment.json
dataset_metadata.json
cohort_report.json
split_manifest.json
censoring_report.json
feature_dictionary.json
feature_rankings.json
selected_features.json
preprocessing/
models/
predictions/
metrics/
calibration/
plots/
explanations/
scaling/
logs/
summary.md
```

Protect earlier valid runs from overwriting.

Do not commit:

- downloaded datasets;
- large kernel matrices;
- model checkpoints;
- unnecessary patient-level data;
- notebook caches.

# 24. Documentation

Create:

```text
experiments/ehr_ihd_qml/docs/heart_disease_experiment.md
```

Keep the main explanation approximately 1,500–2,000 words excluding generated tables.

Explain:

- what ischemic heart disease is;
- classification versus early prediction;
- what the UCI label means;
- what the Framingham endpoints mean;
- why prevalent CHD is excluded;
- how censoring is handled;
- how raw features become model inputs;
- how feature selection works;
- how QSVM computes a quantum kernel;
- how QMLP combines classical and quantum layers;
- how calibration works;
- how a patient prediction is explained;
- what dataset and portion were used;
- exact sample counts;
- resource usage;
- what was learned;
- whether quantum utility was observed;
- why this is not a clinical diagnosis.

Include a compact Mermaid pipeline diagram.

Create:

```text
experiments/ehr_ihd_qml/docs/paper_comparison.md
```

Include:

| Component | Original paper         | This experiment             | Reason for difference  |
| --------- | ---------------------- | --------------------------- | ---------------------- |
| Dataset   | Private Basurto EHR    | UCI/Framingham              | Reproducibility        |
| Features  | 42 → 10                | Dataset-specific → 10       | Paper-inspired         |
| Selection | RFE/RF/mRMR            | Train-only nested selection | Leakage prevention     |
| QSVM      | Pauli, two repetitions | Reproduced plus tuning      | Validation             |
| QMLP loss | MSE                    | BCE plus MSE ablation       | Binary classification  |
| Split     | 80/20                  | Train/validation/test       | Proper model selection |
| Metrics   | Mainly accuracy        | AUROC, AUPRC, calibration   | Clinical relevance     |
| Hardware  | Statevector            | Statevector, optional noise | Resource transparency  |

Also create:

```text
experiments/ehr_ihd_qml/CHANGELOG.md
```

Document:

```text
File | Change | Reason | Effect on existing experiments
```

# 25. Acceptance criteria

The task is complete when:

- the experiment is isolated;
- official UCI ingestion works;
- UCI smoke mode completes;
- missing values are handled without leakage;
- classical baselines train;
- paper-inspired OQSVM trains;
- paper-inspired and preferred HQMLP variants train;
- validation-based tuning works;
- models reload successfully;
- a held-out patient can be predicted;
- QSVM explainability works;
- QMLP explainability works;
- real metrics are generated;
- runtime, memory and circuit counts are reported;
- the notebook completes in smoke mode;
- documentation exists;
- tests pass;
- no unsupported clinical or quantum claim is made.

For the Framingham track:

- if official data is available, execute the 10-year early-prediction smoke experiment;
- if it is unavailable, finish the loader, schema tests, label tests, configuration and instructions;
- do not fabricate Framingham results;
- clearly mark the track as awaiting dataset access.

# 26. Final response required from Codex

At completion, report:

1. What was implemented.
2. Files created and modified.
3. Dataset sources and versions.
4. UCI cohort and split counts.
5. Framingham availability and execution status.
6. Framingham cohort and censoring counts if executed.
7. Missingness handling.
8. Feature-selection results.
9. Exact 10 selected features.
10. Exact OQSVM configuration.
11. Exact HQMLP configurations.
12. Classical metrics.
13. Quantum metrics.
14. Calibration metrics.
15. Confidence intervals.
16. Runtime and memory.
17. Kernel and circuit evaluations.
18. Prediction-interface status.
19. Explainability artifacts.
20. Notebook execution status.
21. Tests and quality checks.
22. Comparison with the source paper.
23. Maximum safe 8 GB configuration.
24. Exact 24 GB longer-run command.
25. Evidence for or against quantum predictive utility.
26. Why computational quantum advantage was or was not demonstrated.
27. Known clinical and dataset limitations.
28. Whether anything was committed or pushed.

Do not invent results, feature rankings, sample counts or performance. If blocked, report the exact blocker and preserve all completed work.
