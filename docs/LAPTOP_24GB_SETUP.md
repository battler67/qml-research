# Run on the 24 GB laptop

For the prepared laptop checkout, use the [guarded run commands](LAPTOP_GUARDED_RUNS.md).
They use the isolated `.venv`, leave an 8 GiB system-memory reserve, and stop jobs
on sampled memory or runtime limits. The direct commands below do not provide
that watchdog.

Install Git and Python 3.12. Authenticate GitHub with an account that has access
to the private repository, then run these commands in PowerShell:

```powershell
git clone --branch feature/ehr-ihd-qml https://github.com/battler67/qml-research.git
cd qml-research
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-qcnn-lock.txt
.\.venv\Scripts\python.exe -m pip install -e ".[dev,qcnn]"
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m qml_research ehr-ihd run --config experiments/ehr_ihd_qml/configs/uci_smoke.yaml
```

Run commands from the cloned `qml-research` folder. The full optional dependency
stack includes Torch, needed by the EHR hybrid model as well as the QCNN. Initial
installation and dataset downloads require internet access.

After the smoke run completes, start the larger EHR profile:

```powershell
.\.venv\Scripts\python.exe -m qml_research ehr-ihd run --config experiments/ehr_ihd_qml/configs/workstation_24gb.yaml
```

This profile requests ten quantum features, three layers, and 60 hybrid-model
epochs. It is a configured experiment, not a measured 24 GB benchmark. The runner
uses `experiment.seed`; `training.seeds` does not automatically run five repeats.
Memory/runtime settings are not a complete watchdog. Run one experiment at a time
and monitor the laptop. See the [roadmap](../specs/QML_EXPERIMENTS_AND_24GB_ROADMAP.md)
for measured results, comparison limitations, and the recommended experiment order.

EHR output is created under `results/ehr_ihd_qml/<run_id>/`. Those generated model
and run directories, downloaded datasets, and `.venv` are ignored by Git and must
be regenerated on the laptop. Existing Phase 1 evidence and written smoke reports
are included in the repository. Framingham requires separately obtained official
teaching data; it is not bundled.

For the separate breast-ultrasound QCNN experiment, follow
[its reproduction guide](../experiments/qcnn_breast_cancer/docs/reproduction.md).
