# EHR IHD QML experiment implementation record

## Scope

- Branch: `feature/ehr-ihd-qml`
- Workspace boundary: only the standalone `qml-research` repository.
- Governing inputs: `research/prompt3.md` and
  `reasearch-papers-docs/prompt3-research-doc.pdf`.
- Experiment root: `experiments/ehr_ihd_qml/`.
- Results root: `results/ehr_ihd_qml/<run_id>/`, with non-overwrite protection.
- No remote push and no modification of the QCNN or earlier Phase 1 result trees.

## Scientific contract

- Track A is **heart-disease presence classification** on the official UCI Cleveland data.
- Track B is **ten-year incident coronary heart-disease risk prediction** from official
  Framingham teaching data, only when the data is locally available and provenance is verified.
- All imputation, scaling, encoding, feature selection, calibration, and threshold selection are
  fitted without test-set access.
- Statevector results are functionality evidence, not computational quantum advantage or clinical
  validation. Negative or inconclusive results remain valid results.
- The supplied systematic review is used to set evidence and claim boundaries; no broad new
  literature search is part of this change.

## Planned implementation

1. Add experiment-specific data loaders and Framingham cohort/endpoint construction.
2. Add train-only preprocessing, consensus feature selection, matched classical models,
   PennyLane quantum-kernel SVM, and hybrid quantum/classical MLP variants.
3. Add thresholding, calibration, metrics, bootstrap intervals, explainability, prediction input
   validation, environment/resource tracking, and protected artifacts.
4. Add the six requested YAML configurations, documentation, MIMIC-IV adapter specification,
   walkthrough notebook, CLI entry point, and focused tests.
5. Execute a bounded UCI smoke run and preserve real outputs. Do not fabricate Framingham results.

## Environment observed before editing

- Python: 3.12.7 in `.venv` (project requires Python 3.12).
- PennyLane: 0.45.1; Torch: 2.13.0; scikit-learn: 1.9.0.
- Qiskit and Qiskit Machine Learning are not installed in the project virtual environment.
- Logical CPUs: 8; physical memory: about 8.25 GB; available memory at inspection: about 1.32 GB.
- Disk free at inspection: about 74.4 GB.

## Verification record

- Branch created: `feature/ehr-ihd-qml`; no push performed.
- Focused implementation tests: 20 passed before the smoke run.
- Final UCI smoke: `uci-smoke-20260829T194302Z-3b55c98a85f0`, completed in 165.77 seconds.
- Cohort/split: 303 total; train 212, validation 45, test 46.
- Smoke selected fields: `thal`, `cp`, `thalach`, `ca`.
- Models completed: five classical, Pauli OQSVM, preferred BCE HQMLP, paper-loss HQMLP.
- QSVM validation tuning selected `C=0.1` by AUPRC before any test evaluation.
- Generated artifacts: 56 files, about 2.01 MB, in the ignored protected run directory.
- Framingham: official teaching file unavailable; loader/tests/config/docs completed, no result made up.
- Project suite: 60 passed; notebook reusable-cell smoke execution included.
- Ruff lint: clean; Ruff format check: 93 files already formatted.
- Python compilation: clean for `src` and `tests`.
- Dependency check: no broken requirements.
- Secret-shaped scan of new implementation/docs/tests: zero matches.
- New implementation files over 5 MB: zero; supplied PDF is 1,268,832 bytes.
- `git diff --check`: clean apart from informational Windows LF-to-CRLF warnings.
- Commit/push: neither performed.
