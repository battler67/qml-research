# Resume the laptop QML campaign

**Current task (15 September):** the separate innovation experiment supersedes the
older queue for the active work. Read [QML_INNOVATION.md](QML_INNOVATION.md) and use
`scripts/innovation_control.py status`, `pause`, or `resume`. Do not start the old
campaign at the same time. Its saved artifacts remain intact.

Run these commands in PowerShell from `C:\Users\SRI RAMANA\qml-research`.

```powershell
.venv/Scripts/python.exe scripts/campaign_control.py status
.venv/Scripts/python.exe scripts/campaign_control.py pause
.venv/Scripts/python.exe scripts/campaign_control.py resume
```

`pause` requests a stop at the next checkpoint/trial boundary. Wait until `status`
reports no active campaign processes before shutting down. `resume` clears the pause
flags and starts one hidden, guarded supervisor. It reuses completed trials and
resumes neural models at their saved mini-batch cursor. It refuses to start a second
active campaign. Closing Codex alone may leave the background process running;
check status first in the next session.

**Do not delete `results/campaign`, `.venv`, `.tools`, or the downloaded datasets.**
Checkpoints live locally and are deliberately excluded from Git. Git alone is not
a checkpoint backup. Keep the same environment, code, dataset and configuration
when resuming. A forced shutdown can lose the currently executing batch or SVM fit;
the previous successfully saved checkpoint remains usable.

The fixed campaign uses four CPU threads, one worker at a time, an 8 GiB process-tree
memory threshold and a 6 GiB free-system-memory reserve. The sampled watchdog stops
its own worker if either threshold is breached; it cannot guarantee against every
transient allocation. GPU training is disabled. The laptop remains available for
other work, although training still uses CPU, power and disk writes.

- [Automatically updated measured comparisons](research/QML_MODELS_WITH_BETTER_METRICS.md)
- [Training protocol, budgets and checkpoint details](docs/RESUMABLE_CAMPAIGN.md)
- [Dataset validity and research rationale](research/CAMPAIGN_DATA_AND_METHODS.md)
- Fixed search specification: `configs/resumable_campaign.json`
- Live worker state: `results/campaign/biomedical_qml_tuning_v1/status.json`
- Queue state: `results/campaign/supervisor.json`
- Resource logs and final measured peaks: `results/laptop/`

For a future Codex session: read this file and the campaign guide, check running
processes and saved status, then resume the existing campaign. Do not launch the
older non-resumable `laptop_suite.py train` queue or change grids based on test scores.
