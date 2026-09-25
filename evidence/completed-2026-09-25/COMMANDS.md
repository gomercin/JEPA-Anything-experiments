# Preservation commands, not experiment commands

Executed from the repository root unless explicitly noted. No scientific runner
was launched; historical reproduction commands reside unchanged in the archives.
Temporary audit/output directories are substituted below by `$AUDIT`; the local
execution used `/private/tmp/jepa-preserve-sjuk7oqw`. This is publication-time
bookkeeping, not a historical experiment environment.

```bash
pwd
git status --short
git branch --show-current
git rev-parse HEAD
git remote -v
gh repo view gomercin/JEPA-Anything-experiments --json nameWithOwner,url,visibility,defaultBranchRef
gh pr list --repo gomercin/JEPA-Anything-experiments --state all --limit 50 --json number,title,state,headRefName,baseRefName,url
gh release list --repo gomercin/JEPA-Anything-experiments --limit 20
git fetch origin
git ls-remote --heads https://github.com/gomercin/JEPA-Anything-experiments.git
git diff HEAD..origin/main --stat
git switch -c codex/preserve-completed-experiments
.venv/bin/python evidence/completed-2026-09-25/package.py --output "$AUDIT/assets"
OPENBLAS_NUM_THREADS=1 .venv/bin/python evidence/completed-2026-09-25/saved_data_smoke.py --root .
OPENBLAS_NUM_THREADS=1 make check PYTHON=.venv/bin/python
.venv/bin/ruff check --config jepa-anything-core/pyproject.toml experiments
.venv/bin/ruff format --check --config jepa-anything-core/pyproject.toml experiments
.venv/bin/ruff check --config jepa-anything-core/pyproject.toml experiments --output-format json
.venv/bin/ruff format --config jepa-anything-core/pyproject.toml evidence/completed-2026-09-25/{package,restore,saved_data_smoke}.py evidence/completed-2026-09-25/tests
.venv/bin/ruff check --config jepa-anything-core/pyproject.toml evidence/completed-2026-09-25/{package,restore,saved_data_smoke}.py evidence/completed-2026-09-25/tests
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest evidence/completed-2026-09-25/tests -q
git diff --check
```

The initial sandboxed fetch could not write `.git/FETCH_HEAD`; sandboxed Git
network access also failed DNS resolution. Authorized escalated fetch/ls-remote
then succeeded. No history or worktree content was reset. The first lint pass on
new packaging code exposed line-length formatting issues, corrected only in new
publication tooling. The strict check on historical source retained 58 style
findings and 19 would-reformat files; those sources were not changed.

Read-only Python inspections also inventoried/hashes files, scanned included text
for credential signatures, checked original source-hash availability, indexed
recorded result/model identities, and copied existing small review artifacts.
Their derived inventories are committed here. Original artifact paths/hashes,
failed panels and exposure statuses were not edited. Archive restoration checks
below verify bytes and serialization, not scientific accuracy.
