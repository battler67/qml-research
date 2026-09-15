# Guarded runs on this laptop

For periodic batch checkpoints, use the newer [resumable campaign](RESUMABLE_CAMPAIGN.md)
or the current [innovation experiment](../QML_INNOVATION.md). The earlier training
queue documented below is retained as historical evidence.

The checkout is `feature/ehr-ihd-qml` at `26b3f27`. Python 3.12 and all
dependencies are isolated in `.tools/` and `.venv/`; the existing Anaconda
environments are unchanged. The resolved versions and hardware observation are
saved in `results/laptop/environment-lock.txt` and `hardware.json`.

## Initial verification, 10 September 2026

GitHub authentication and checkout succeeded. UCI Cleveland and BreastMNIST were
downloaded from their official sources and verified. All three guarded smoke
experiments completed: 10 Iris model/fold records, eight EHR models, and seven
BreastMNIST models, with no failures or skipped quantum models.

| Measured run | Elapsed | Peak process-tree RSS | Lowest available system RAM |
| --- | ---: | ---: | ---: |
| Iris smoke, including process startup | 32.6 s | 0.235 GiB | 12.71 GiB |
| EHR notebook verification, including its smoke run | 316.0 s | 0.468 GiB | 12.30 GiB |
| BreastMNIST smoke, including evaluation/reports | 394.5 s | 0.537 GiB | 12.13 GiB |

The EHR experiment itself took 291.2 seconds. QCNN fitting itself took 33.6
seconds for two epochs; report generation accounted for much of the remaining
time. Bootstrap evaluation now computes only its requested statistic, preserving
the original draws and thresholds. Five regression cases compare its intervals
against the original complete-metric calculation, including tied scores and
threshold equality. Historical smoke timings above precede this optimization.

Watchdog checks exercised successful exit, a deliberately tiny memory threshold,
runtime expiry, refusal under insufficient available RAM, and rejection of a
concurrent guarded worker. Resource readings are sampled measurements, not
allocation guarantees. Smoke scores establish functionality only; full-data
training status is recorded separately in `results/laptop/train-suite.json`.

Final verification passed: **65 tests**, Ruff lint, formatting (99 files), package
compatibility, and `git diff --check`. The final test suite took 77 seconds and
peaked at 0.596 GiB process-tree RSS. The background training console is
`results/laptop/training-console.log`; its launcher PID is recorded in
`results/laptop/training-suite.pid`. Use the STOP file described below to stop
the active model and remaining queue.

The machine has 23.64 GiB system RAM and an RTX 5050 Laptop GPU with 8 GiB VRAM.
These runs use CPU simulation, two BLAS/OpenMP threads, below-normal process
priority, and no GPU. The current EHR implementation uses `default.qubit`.

## Run and stop

From the repository root in PowerShell:

```powershell
# Short real-data runs across Phase 1, EHR, and QCNN; stops on failure.
.venv/Scripts/python.exe scripts/laptop_suite.py --stage smoke

# Requires a completed smoke suite. Runs each experiment sequentially.
.venv/Scripts/python.exe scripts/laptop_suite.py --stage train

# Read current status and recent resource measurements.
Get-Content results/laptop/train-suite.json
Get-ChildItem results/laptop/*.log | Sort-Object LastWriteTime -Descending

# Stop the active guarded worker and prevent subsequent jobs from starting.
Set-Content results/laptop/STOP stop

# Only when ready to allow training again:
Remove-Item -LiteralPath results/laptop/STOP
```

The watchdog samples the worker and its children every 0.25 seconds. It stops
them if aggregate RSS or private memory exceeds 6 GiB, available system RAM
falls below 8 GiB, or the job's runtime budget expires. It refuses to start
without at least 9 GiB available. An OS file lock prevents two guarded workers
from running together. Closing the foreground run with Ctrl+C also stops its
active worker. No scheduled task or startup service is installed.

These are sampled thresholds, **not a hard OS memory-allocation cap**. A sudden
allocation can cross a threshold before the watchdog reacts; unrelated programs
can also consume memory. Keep the watchdog alive while its worker runs. The
`STOP` file is the preferred way to stop a background suite. Interrupted models
may lose the current training progress; checkpoints are saved when each model
completes, not after every epoch. Completed QCNN and Phase 1 records are reused
on rerun. EHR creates a new run directory and retrains.

## Training scope

1. **Heart disease:** all available training rows from the 303-row Cleveland
   dataset (212 train / 45 validation / 46 test at seed 42), four selected
   features/qubits, two hybrid layers, a 60-epoch ceiling, and patience 10.
   Both hybrid variants and seven classical controls run. Pauli QSVM selects
   `C` from `[0.01, 0.1, 1, 10]` on validation AUPRC. Runtime ceiling: 3 hours.
2. **BreastMNIST:** complete official 546/78/156 partitions, four-qubit QCNN,
   five reduced classical controls and a native CNN, seed 42, up to 60 epochs,
   patience 10, and zero data-loader workers. Runtime ceiling: 4 hours. The
   repository's conservative estimate is 3.14 hours for quantum forwards alone;
   this is not a measured laptop runtime. Pretrained image models and eight-qubit
   circuits are outside this initial queue.
3. **WDBC:** 200-row timing runs followed by full-data runs, five folds, seed 42,
   two PCA features, Angle QKSVM, two-layer VQC with 300 optimizer steps, and
   classical controls. Full-data QML still obeys the existing timing gate.
   Runtime ceiling: 3 hours for the entire WDBC job.

This is substantial bounded training, not proof of convergence or optimal
hyperparameters. Early stopping restores the best validation-loss state. Test
results do not guide tuning. One seed does not establish training stability.
The guard stops the remaining queue on a worker failure or resource stop.

## Outputs

- `results/laptop/*-suite.json`: current/completed jobs.
- `results/laptop/*.json`: sampled resource peaks, exit status, runtime, hardware.
- `results/laptop/*.log`: worker logs and final command results.
- `results/laptop/phase1/`: WDBC and Iris model/fold records.
- `results/ehr_ihd_qml/`: EHR models, calibration, predictions, and metrics.
- `results/qcnn_breast_cancer/`: QCNN/CNN checkpoints, histories, and reports.

The inherited EHR runner names its directories and summaries `uci-smoke` even
for larger configurations. Use `resolved_config.yaml` and training tracking to
identify the actual run budget.

To run another command through the same guard:

```powershell
.venv/Scripts/python.exe scripts/laptop_guard.py --name my-run --minutes 60 -- .venv/Scripts/python.exe -m qml_research run --config configs/iris_sanity.yaml
```
