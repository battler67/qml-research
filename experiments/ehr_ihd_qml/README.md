# EHR ischemic/coronary heart-disease QML experiment

This experiment is isolated from every earlier benchmark. Run the bounded UCI functionality profile:

```powershell
qml-research ehr-ihd run --config experiments/ehr_ihd_qml/configs/uci_smoke.yaml
```

Outputs are written to a protected `results/ehr_ihd_qml/<run_id>/` directory. Existing run
directories are never overwritten. The Framingham track remains unavailable until the official
NHLBI/BioLINCC teaching file is placed at the documented local path.

**RESEARCH/EDUCATIONAL EXPERIMENT - NOT A MEDICAL DIAGNOSIS**
