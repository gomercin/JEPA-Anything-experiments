# Executed Phase D commands

Working directory: `/Users/omer.karaduman/Work/Projects/JEPA-Anything-experiments`.
Commands below are scientific, aggregation, and validation commands; routine
read-only source/result inspections are omitted. For reproduction, select new
output paths and update the dependent input paths. Existing directories are
rejected to preserve completed and failed evidence.

```bash
# Baseline before changes: 40 oscillator tests passed.
.venv/bin/python -m pytest experiments/coupled_oscillators/tests -q

# Noise/causality tests, initially three, then four, finally six test cases.
.venv/bin/python -m pytest experiments/coupled_oscillators/tests/test_noise.py -q

.venv/bin/python -m experiments.coupled_oscillators.phase_d_noise --qualify --output work/coupled_oscillators/phase_d_noise/d0
.venv/bin/python -m experiments.coupled_oscillators.phase_d_noise --filters --qualification work/coupled_oscillators/phase_d_noise/d0 --output work/coupled_oscillators/phase_d_noise/d1-filters
.venv/bin/python -m experiments.coupled_oscillators.phase_d_neural --qualification work/coupled_oscillators/phase_d_noise/d0 --output work/coupled_oscillators/phase_d_noise/d2-neural

# Failed before model results: LBFGS could not flatten a noncontiguous gradient.
.venv/bin/python -m experiments.coupled_oscillators.phase_d_refine --filters work/coupled_oscillators/phase_d_noise/d1-filters --output work/coupled_oscillators/phase_d_noise/d3-output-error

# Retry after making local optimizer parameters contiguous; same scientific settings.
.venv/bin/python -m experiments.coupled_oscillators.phase_d_refine --filters work/coupled_oscillators/phase_d_noise/d1-filters --output work/coupled_oscillators/phase_d_noise/d3-output-error-contiguous
.venv/bin/python -m experiments.coupled_oscillators.phase_d_neural --duration --qualification work/coupled_oscillators/phase_d_noise/d0 --output work/coupled_oscillators/phase_d_noise/d4-duration
.venv/bin/python -m experiments.coupled_oscillators.phase_d_control --neural work/coupled_oscillators/phase_d_noise/d2-neural --output work/coupled_oscillators/phase_d_noise/d5-normalization
.venv/bin/python -m experiments.coupled_oscillators.phase_d_refine --iterations 240 --filters work/coupled_oscillators/phase_d_noise/d1-filters --output work/coupled_oscillators/phase_d_noise/d6-output-error-duration

# Read-only analysis; writes derived review files, never raw panel results.
PYTHONPATH=. MPLCONFIGDIR=/private/tmp/coupled-oscillators-mpl .venv/bin/python work/coupled_oscillators/phase_d_noise/analyze.py

make check PYTHON=.venv/bin/python > /private/tmp/jepa-phase-d-check.log 2>&1
.venv/bin/ruff format --config jepa-anything-core/pyproject.toml experiments/coupled_oscillators/noise_*.py experiments/coupled_oscillators/phase_d_*.py experiments/coupled_oscillators/tests/test_noise.py
.venv/bin/ruff check --config jepa-anything-core/pyproject.toml experiments/coupled_oscillators/noise_*.py experiments/coupled_oscillators/phase_d_*.py experiments/coupled_oscillators/tests/test_noise.py
.venv/bin/ruff format --check --config jepa-anything-core/pyproject.toml experiments/coupled_oscillators/noise_*.py experiments/coupled_oscillators/phase_d_*.py experiments/coupled_oscillators/tests/test_noise.py
git diff --check
```

Ruff import/format fixes were applied during implementation. The final tests
pass: **66 core; 33 Skill plus 2 subtests; 46 oscillators**. Exact output is in
`repository-check.log`. New untracked files additionally passed whitespace
checks using `git diff --no-index --check /dev/null FILE`.

The layout failure and its source snapshot remain in `d3-output-error/`.
The 80-iteration refinement produced worse forecasts and remains in its own
completed panel. The 240-iteration replay preserves its entire earlier loss
prefix exactly. All twelve neural duration replays reproduce their 1,000-step
checkpoints exactly. No outcome was replaced.

Files added under `experiments/coupled_oscillators/`:

- `noise_data.py`: seeded sensor noise and restricted learner settings.
- `noise_baselines.py`: causal polynomial endpoint state.
- `noise_filters.py`: observation-only identification, analytic tangent map,
  windowed EKF, explicitly separate oracle equations/variance.
- `noise_evaluate.py`: separate noisy/clean metrics, evaluator-only probes and
  exact-state transition diagnostics.
- `phase_d_noise.py`: noise qualification and filtering panels.
- `phase_d_neural.py`: existing JEPA adapters under noisy training.
- `phase_d_refine.py`: targeted noisy-output identification refinement.
- `phase_d_control.py`: matched-normalization control.
- `tests/test_noise.py`: six instrument correctness tests.
- `PHASE_D_NOISE.md`: scientific report and limitations.

Changed `README.md` to link the report. Existing Phase A–C source/result reports
and all 249 earlier generated artifacts remain unchanged. The preexisting root
`.gitignore`, `Makefile`, and `pyproject.toml` changes were not touched. No core
or Atlas changes. `session_manifest.json` records hashes and final verification.
