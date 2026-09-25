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

## Publication and independent retrieval

These operations use only the explicitly authorized owner repository. The new
tag is fixed; neither scientific assets nor history were overwritten.

```bash
git add -- .gitignore Makefile pyproject.toml .github/workflows/ci.yml README.md experiments/coupled_oscillators experiments/mediated_patterns experiments/requirements.txt evidence/completed-2026-09-25
git add -- .gitattributes Makefile evidence/completed-2026-09-25/README.md evidence/completed-2026-09-25/checks.json
git diff --cached --check
git commit -m "Preserve completed oscillator and mediated-pattern experiments with evidence inventory"
git tag -a evidence-completed-2026-09-25 e17fe41d07a03d3a7fe67955054112773e8344be -m "Completed exploratory research evidence; unmerged and unreviewed preservation snapshot"
git push --set-upstream https://github.com/gomercin/JEPA-Anything-experiments.git codex/preserve-completed-experiments
git push https://github.com/gomercin/JEPA-Anything-experiments.git refs/tags/evidence-completed-2026-09-25
gh release create evidence-completed-2026-09-25 "$AUDIT/assets/coupled_oscillators.tar.gz" "$AUDIT/assets/mediated_patterns.tar.gz" --repo gomercin/JEPA-Anything-experiments --verify-tag --prerelease --latest=false --title "Research evidence snapshot — completed exploratory work; unmerged/unreviewed" --notes-file "$AUDIT/release-notes.md"
gh pr create --repo gomercin/JEPA-Anything-experiments --base main --head codex/preserve-completed-experiments --draft --title "Preserve completed oscillator and mediated-pattern experiments" --body-file "$AUDIT/pr-body.md"
git clone --depth 1 --single-branch --branch codex/preserve-completed-experiments https://github.com/gomercin/JEPA-Anything-experiments.git "$AUDIT/remote-checkout"
git -C "$AUDIT/remote-checkout" rev-parse HEAD
# First download attempt failed local Python certificate validation; no bypass.
.venv/bin/python -I "$AUDIT/remote-checkout/evidence/completed-2026-09-25/restore.py" --destination "$AUDIT/remote-checkout"
# Downloaded from GitHub, not copied from the local build archives; large asset timed out.
gh release download evidence-completed-2026-09-25 --repo gomercin/JEPA-Anything-experiments --pattern '*.tar.gz' --dir "$AUDIT/remote-assets"
# Resumed the remote partial file; full archive SHA-256 subsequently verified.
curl --fail --location --retry 2 --retry-delay 2 --connect-timeout 30 --max-time 600 --continue-at - --output "$AUDIT/remote-assets/mediated_patterns.tar.gz" https://github.com/gomercin/JEPA-Anything-experiments/releases/download/evidence-completed-2026-09-25/mediated_patterns.tar.gz
.venv/bin/python -I "$AUDIT/remote-checkout/evidence/completed-2026-09-25/restore.py" --archives "$AUDIT/remote-assets" --destination "$AUDIT/remote-checkout"
OPENBLAS_NUM_THREADS=1 .venv/bin/python -I "$AUDIT/remote-checkout/evidence/completed-2026-09-25/saved_data_smoke.py" --root "$AUDIT/remote-checkout"
.venv/bin/python -m pytest experiments/mediated_patterns/tests/test_fields.py::test_prior_artifact_preservation -q
```

A new oscillator README appendix was removed after read-only checking showed it
would conflict with the historical exact-preservation fixture. The original bytes
were restored; no frozen manifest or test was changed. Root/field handoffs contain
the publication pointers. The focused preservation regression passed.

Final publication bookkeeping uses a second documentation commit and explicit
branch push, followed by a fetch/checkout of that exact head in the independent
clone. The externally pinned final head and final command/check receipts are in
`remote-verification.json` on the release, avoiding a self-referential commit hash.
