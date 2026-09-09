from __future__ import annotations

import json
from pathlib import Path


def test_notebook_structure_and_smoke_execution(monkeypatch) -> None:
    notebook_path = Path("notebooks/ehr_ihd_qml_walkthrough.ipynb").resolve()
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) >= 10
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "RESEARCH/EDUCATIONAL EXPERIMENT" in source
    assert "run_uci_smoke" in source
    namespace = {"display": lambda value: value}
    monkeypatch.chdir(notebook_path.parent)
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), str(notebook_path), "exec"), namespace)
    assert namespace["summary"]["status"] == "completed"
    assert namespace["prediction"]["warning"].startswith("Research demonstration")
