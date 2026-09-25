# Session commands

Named panels, executed in order (qualification-02 intentionally failed and is retained):

```bash
.venv/bin/python -m experiments.mediated_patterns.explore --qualify --output work/mediated_patterns/qualification-02
.venv/bin/python -m experiments.mediated_patterns.explore --qualify --output work/mediated_patterns/qualification-03
.venv/bin/python -m experiments.mediated_patterns.explore --panel --qualification work/mediated_patterns/qualification-03 --output work/mediated_patterns/development-01
.venv/bin/python -m experiments.mediated_patterns.explore --receiver-control --output work/mediated_patterns/receiver-control-01
.venv/bin/python -m experiments.mediated_patterns.outward_response --development work/mediated_patterns/development-01 --output work/mediated_patterns/outward-development-01
.venv/bin/python -m experiments.mediated_patterns.explore --fresh --qualification work/mediated_patterns/qualification-03 --model work/mediated_patterns/development-01/response_model.json --output work/mediated_patterns/fresh-01
.venv/bin/python -m experiments.mediated_patterns.outward_response --evaluate work/mediated_patterns/fresh-01 --model work/mediated_patterns/outward-development-01/model.json --output work/mediated_patterns/outward-fresh-01
.venv/bin/python -m experiments.mediated_patterns.analyze --root work/mediated_patterns --output work/mediated_patterns/analysis-01
.venv/bin/python -m experiments.mediated_patterns.explore --quick --output work/mediated_patterns/quick-01
.venv/bin/python -m pytest experiments/mediated_patterns/tests -q
make check PYTHON=.venv/bin/python
.venv/bin/python -m ruff check --config jepa-anything-core/pyproject.toml experiments/mediated_patterns
.venv/bin/python -m ruff format --check --config jepa-anything-core/pyproject.toml experiments/mediated_patterns
git diff --check
```

The first four pilot panels and qualification-01 were inline `.venv/bin/python -` scripts. Their complete configs and numeric outputs are retained in protocol/results JSON and NPZ; pilot source copies are explicitly retrospective. Exact inline shell bodies were not separately archived. The later readout-diagnostic-01 (PCHIP on original training only) and fresh-refinement-01 (seed101, d25, a=-.075, N1024, dt=.00625) were also inline diagnostic scripts; their contracts and complete results are saved. No command in this session changes Atlas or trains a neural model.

New untracked files also passed individual `git diff --no-index --check /dev/null FILE` checks. Read-only aggregation and source inspection commands are not counted as scientific runs.
