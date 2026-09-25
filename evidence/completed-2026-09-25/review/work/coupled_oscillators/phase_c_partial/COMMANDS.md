# Executed Phase C commands

Working directory: `/Users/omer.karaduman/Work/Projects/JEPA-Anything-experiments`.
These are scientific, aggregation, and validation commands; routine read-only
source/result inspections are omitted. Each scientific output path was new when
executed. Reproduction must use new paths, including updated qualification/panel
arguments. Existing output directories are deliberately rejected.

```bash
# Before changes: 30 oscillator tests passed.
.venv/bin/python -m pytest experiments/coupled_oscillators/tests -q

# Observation qualification and controls, selected successively from results.
.venv/bin/python -m experiments.coupled_oscillators.phase_c_partial --qualify --output work/coupled_oscillators/phase_c_partial/c0
.venv/bin/python -m experiments.coupled_oscillators.phase_c_partial --compression-control --output work/coupled_oscillators/phase_c_partial/c0-compression

# Initial information-boundary/equivalence tests: 8 passed at that point.
.venv/bin/python -m pytest experiments/coupled_oscillators/tests/test_partial.py -q

.venv/bin/python -m experiments.coupled_oscillators.phase_c_partial --neural --qualification work/coupled_oscillators/phase_c_partial/c0 --output work/coupled_oscillators/phase_c_partial/c1
.venv/bin/python -m experiments.coupled_oscillators.phase_c_controls --ridge --output work/coupled_oscillators/phase_c_partial/c0-ridge
.venv/bin/python -m experiments.coupled_oscillators.phase_c_partial --duration --qualification work/coupled_oscillators/phase_c_partial/c0 --output work/coupled_oscillators/phase_c_partial/c2-duration
.venv/bin/python -m experiments.coupled_oscillators.phase_c_controls --diagnose --panel work/coupled_oscillators/phase_c_partial/c2-duration --output work/coupled_oscillators/phase_c_partial/c3-diagnostics

# Read-only aggregation/plotting. Repeated after fixing an up-to-6D plot label.
PYTHONPATH=. MPLCONFIGDIR=/private/tmp/coupled-oscillators-mpl .venv/bin/python work/coupled_oscillators/phase_c_partial/analyze.py
PYTHONPATH=. MPLCONFIGDIR=/private/tmp/coupled-oscillators-mpl .venv/bin/python work/coupled_oscillators/phase_c_partial/analyze.py > work/coupled_oscillators/phase_c_partial/aggregation.log

# Final deterministic repository checks after adding the two final boundary tests.
make check PYTHON=.venv/bin/python > /private/tmp/jepa-phase-c-check.log 2>&1

.venv/bin/ruff check --config jepa-anything-core/pyproject.toml experiments/coupled_oscillators/partial_data.py experiments/coupled_oscillators/partial_baselines.py experiments/coupled_oscillators/partial_models.py experiments/coupled_oscillators/partial_probes.py experiments/coupled_oscillators/phase_c_partial.py experiments/coupled_oscillators/phase_c_controls.py experiments/coupled_oscillators/tests/test_partial.py
.venv/bin/ruff format --check --config jepa-anything-core/pyproject.toml experiments/coupled_oscillators/partial_data.py experiments/coupled_oscillators/partial_baselines.py experiments/coupled_oscillators/partial_models.py experiments/coupled_oscillators/partial_probes.py experiments/coupled_oscillators/phase_c_partial.py experiments/coupled_oscillators/phase_c_controls.py experiments/coupled_oscillators/tests/test_partial.py
git diff --check
```

Ruff formatting was also run on the new local modules during implementation.
Final whitespace checks additionally used `git diff --no-index --check /dev/null
FILE` for each new source/report/test file, since this checkout's entire experiment
directory is untracked. All passed. `repository-check.log` preserves exact final
test output: **66 core passed; 33 Skill passed plus 2 subtests; 40 oscillators passed**.

The preexisting `.gitignore`, `Makefile`, and root `pyproject.toml` modifications
were not changed during this session. No upstream core or Atlas files were edited.
`session_manifest.json` records final source hashes and the protected-artifact
verification, separately from immutable per-panel protocols/results.
