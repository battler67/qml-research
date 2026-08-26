# Notebooks

Core algorithms and experiment orchestration live in `src/qml_research` and
`experiments/`. A notebook may import those modules for judge-facing explanation,
but it must not duplicate preprocessing, model, metric, or result logic.

The script `experiments/reproduce_pennylane_kernel.py` is the executable Phase 1
replacement for a stateful tutorial notebook. It preserves the published tutorial
ordering as a labelled reference and separately runs the leakage-corrected version.

