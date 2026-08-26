# Phase 1 Research Gaps

## Evidence gaps

- Few digital-health studies combine strong tuned classical baselines, identical
  preprocessing, repeated cross-validation, uncertainty, noisy execution and public
  code. The systematic review found only a small hardware/noisy subset.
- Published small-data wins often conflate quantum contribution with classical
  compression, feature selection, optimizer choice or asymmetric tuning.
- External clinical validation, calibration, subgroup analysis, prospective testing
  and decision-curve analysis are generally missing. None are supplied by Phase 1.
- Natural biomedical data with a useful, classically difficult quantum feature map
  has not been identified.

## Technical gaps

- Angle embedding is transparent but separable; IQP encoding is more structured but
  may concentrate and remains classically simulable at Phase 1 scale.
- Quadratic kernel evaluation and parameter-shift VQC training prevent naive scaling.
- A single depolarizing parameter is a sensitivity test, not a calibrated IBM noise
  model. Readout, routing, coherent errors and drift remain unmodeled.
- Kernel PSD repair, mitigation and trainable kernels require separate controlled
  studies; silently applying them would alter the Phase 1 question.
- VQC optimizer and initialization sensitivity needs more trials than one best seed.

## Biomedical gaps

- WDBC is diagnostic breast-mass morphology, not screening or pre-symptomatic data.
- Cleveland Heart Disease is small, historical and not an external modern cohort.
- The MVP has no longitudinal outcome, survival, multimodal, genomic-expression,
  imaging or patient-grouped repeated-measure evaluation.
- Clinical costs, calibration and false-negative thresholds are not modeled.

## Phase 2 questions

1. Can the frozen Phase 1 feature map be reproduced on a small IBM inference batch
   without invalidating the classifier through noise or transpilation?
2. Does a trainable or domain-informed kernel improve paired performance without
   exploding circuit calls or introducing selection bias?
3. Can an external public cohort support a true transportability test?
4. Which result/provenance fields should become the stable Quantum Helix Lab API?
5. Can the demo communicate uncertainty and scientific boundaries clearly enough
   that judges cannot mistake a simulator experiment for clinical or quantum advantage?

