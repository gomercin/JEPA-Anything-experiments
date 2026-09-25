# Phase B session commands

Executed from `/Users/omer.karaduman/Work/Projects/JEPA-Anything-experiments`.
File inspection (`rg`, `sed`, `git status`) and read-only result extraction also
occurred. These are the exact experiment, analysis and validation commands.

```bash
.venv/bin/python -m pytest experiments/coupled_oscillators/tests -q
.venv/bin/python -m experiments.coupled_oscillators.phase_b --qualify --output work/coupled_oscillators/phase_b/b0
.venv/bin/python -m experiments.coupled_oscillators.phase_b --qualify --substeps 8 --output work/coupled_oscillators/phase_b/b0-refined
.venv/bin/python -m pytest experiments/coupled_oscillators/tests/test_nonlinear.py -q
.venv/bin/python -m experiments.coupled_oscillators.phase_b --panel --qualification work/coupled_oscillators/phase_b/b0-refined --output work/coupled_oscillators/phase_b/b1
.venv/bin/python -m experiments.coupled_oscillators.phase_b_followup --panel work/coupled_oscillators/phase_b/b1 --output work/coupled_oscillators/phase_b/b2-controls
.venv/bin/python -m experiments.coupled_oscillators.phase_b_followup --transform-control --panel work/coupled_oscillators/phase_b/b2-controls --output work/coupled_oscillators/phase_b/b3-transform
make check PYTHON=.venv/bin/python
PYTHONPATH=. MPLCONFIGDIR=/private/tmp/coupled-oscillators-mpl .venv/bin/python work/coupled_oscillators/phase_b/analyze.py
.venv/bin/ruff check --config jepa-anything-core/pyproject.toml experiments/coupled_oscillators/nonlinear.py experiments/coupled_oscillators/phase_b.py experiments/coupled_oscillators/phase_b_followup.py experiments/coupled_oscillators/tests/test_nonlinear.py
.venv/bin/ruff format --check --config jepa-anything-core/pyproject.toml experiments/coupled_oscillators/nonlinear.py experiments/coupled_oscillators/phase_b.py experiments/coupled_oscillators/phase_b_followup.py experiments/coupled_oscillators/tests/test_nonlinear.py
git diff --check
```

The first B0 invocation used the then-default **4 substeps**, recorded in each
simulator record and preserved source snapshot. It failed the declared integration
gate (three moderate seeds), before any neural model ran. It is retained intact.
To reproduce that qualification using current code, explicitly pass `--substeps 4`
and a fresh directory. The current default is 8. All successful run directories
refuse overwrite; use fresh paths to reproduce.

The initial pre-change oscillator tests passed 21/21. The first new-test run had
5 passes and a test-only JSON comparison failure (core audit tuples serialize to
lists); the assertion was narrowed to numeric metrics/model states and the six
then-present tests passed. Three more transform/folding cases bring the final new
cases to nine. Final `make check`: 66 core, 33 skill plus 2 subtests, 30 oscillator,
all passing; design, recipe, manifest, JSON and compile checks also pass.

Ruff formatting/checking was also run during implementation; line-length/format
issues were fixed. The first plotting-script launch without `PYTHONPATH=.` failed
to import the local experiment package; the exact corrected command above succeeds.
Neither development failure produced scientific model results.

All Phase A, interrogation and frozen-encoder artifact hashes were checked before
and after these commands. `analysis.json` and `session_manifest.json` record the
50-file preservation check and final provenance. Plotting only reads completed
results; it neither trains nor selects checkpoints.
