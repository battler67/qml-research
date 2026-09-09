# Change log

| File | Change | Reason | Effect on existing experiments |
| --- | --- | --- | --- |
| `src/qml_research/ehr_ihd/` | New isolated data, feature, model, evaluation, prediction, runner, and CLI package | Implement both heart-disease tracks | None; additive package |
| `experiments/ehr_ihd_qml/configs/` | Six bounded/tuning profiles | Reproducible resource control | None |
| `experiments/ehr_ihd_qml/docs/` | Research, data, methodology, and limitation records | Trace scientific decisions | None |
| `tests/test_ehr_ihd_*.py` | Dataset, leakage, configuration, quantum, and interface tests | Regression protection | Adds test coverage only |
| `notebooks/ehr_ihd_qml_walkthrough.ipynb` | Reusable-module walkthrough | Reproducible educational narrative | None |
| `src/qml_research/cli.py` | Added `ehr-ihd` subcommand | Discoverable execution | Existing commands unchanged |
| `.gitignore` | Ignore generated IHD runs/checkpoints | Avoid patient-level/generated artifacts | Existing rules retained |
