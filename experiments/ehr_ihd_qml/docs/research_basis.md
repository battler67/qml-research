# Research basis

## Scope and terminology

Ischemic heart disease is reduced delivery of blood and oxygen to heart muscle because coronary
arteries are narrowed or blocked. Coronary heart disease and coronary artery disease are often used
for overlapping clinical concepts. Relevant outcomes include angina, coronary insufficiency,
myocardial infarction, and fatal coronary disease. This experiment does not silently combine these
with congenital or valvular disease, arrhythmia, myocarditis, or heart failure from any cause.

The UCI outcome is a diagnostic heart-disease label recorded during clinical evaluation. It is
therefore reported only as **heart-disease presence classification**. The Framingham endpoint is a
future first coronary event in participants free of prevalent CHD at baseline and is reported as
**ten-year incident coronary heart-disease risk prediction**.

## Primary paper-inspired design

Maheshwari et al. (2023), DOI `10.22967/HCIS.2023.13.006`, studied a private chest-pain EHR cohort
from Basurto University Hospital. About 2.2 million initial records were considered, 43,835
chest-pain patients were identified, and 33,372 records reportedly remained after exclusions, with
about 16.1% coronary artery disease. Cleaning reduced 82 variables to 42. Logistic-regression RFE,
random-forest importance, and mRMR were combined to obtain ten final fields. The optimized QSVM
used a Pauli feature map with two repetitions on a statevector simulator. The hybrid QMLP combined
classical layers, quantum encoding, trainable rotations, entanglement, Adam, and MSE.

Direct reproduction is impossible because the private cohort is unavailable. Reported counts are
not fully consistent, validation-based selection is underspecified, feature-selection leakage cannot
be excluded, and balancing appears to occur before or around splitting. Calibration and AUPRC are
not central results. Classical SVM and MLP comparisons were at least as good as, or better than,
the reported optimized quantum models. This repository therefore reproduces ideas, not an exact
private-data result: train-only selection, untouched natural-prevalence validation/test sets,
validation-only calibration and threshold selection, and both MSE and preferred BCE variants.

## Longitudinal prediction

The supplied research identifies explicit index dates, exclusion of prevalent disease, a future
horizon, patient-level separation, censoring, calibration, and decision thresholds as necessary for
credible coronary prediction. Time-aware features can add value but may use only measurements
known at the landmark. The Framingham baseline endpoint uses `PERIOD == 1`, `PREVCHD == 0`, and
`ANYCHD/TIMECHD`; insufficient follow-up is unknown rather than negative. An optional Period 2
landmark can use prior/current values and changes, but never Period 3 measurements for an event
after Period 2.

## Current QML evidence and claim boundary

The supplied 2025 systematic-review PDF screened digital-health QML evidence and found no
consistent performance differential supporting empirical quantum utility. It reports 169 eligible
studies, excludes 123 for insufficient rigor, and finds only 16 studies considering realistic
hardware or noisy conditions; 13 included hardware demonstrations. It also emphasizes encoding
costs, noise, restrictive hardware assumptions, and frequent reliance on ideal simulation.

Accordingly, this experiment does not assume that QSVM beats RBF-SVM or that QMLP beats a
classical MLP. Models receive matched patients and selected fields, runtime and circuit work are
recorded, uncertainty is reported, and negative results are accepted. A statevector run can show
functionality and finite-sample predictive behaviour; it cannot establish computational quantum
advantage, clinical validation, or replacement of clinical examination, ECG, or angiography.

## Sources supplied for this task

- Maheshwari et al., 2023: <https://hcisj.com/data/file/article/2023020001/13-06.pdf>
- Longitudinal EHR/genetic prediction: <https://www.nature.com/articles/s41598-018-36745-x>
- EHR coronary prediction: <https://medinform.jmir.org/2020/7/e17257>
- Quantum kernels for EHR: <https://research.ibm.com/publications/quantum-kernels-for-electronic-health-records-classification>
- Clinical NISQ classification: <https://www.nature.com/articles/s41598-022-05971-9>
- Supplied systematic review: DOI `10.1038/s41746-025-01597-z`
- QML benchmarking: <https://arxiv.org/abs/2403.07059>
