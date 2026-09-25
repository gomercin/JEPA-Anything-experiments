# Completed exploratory work — preservation snapshot

Research evidence snapshot; **unmerged/unreviewed**, not a production release or
independent reproduction of the science. This index preserves the completed
oscillator and mediated-pattern lineages, including failures. No new scientific
panel, recalibration, threshold change, or organization-conditioned-response
experiment is part of this publication.

Destination: [gomercin/JEPA-Anything-experiments](https://github.com/gomercin/JEPA-Anything-experiments),
public repository, default branch `main`. Preservation branch:
`codex/preserve-completed-experiments`. The immutable evidence tag is
`evidence-completed-2026-09-25`. [Draft PR #1](https://github.com/gomercin/JEPA-Anything-experiments/pull/1) is open
and unmerged. Source snapshot: `e17fe41d07a03d3a7fe67955054112773e8344be`.
Publication identity and verification are recorded
in [publication.json](publication.json); the tag identifies the source-snapshot
commit, while the later PR head adds the publication handoff. Neither is claimed
to be the historical execution revision of dirty/uncommitted runs.

## What is preserved

[artifacts.json](artifacts.json) enumerates **every one of the 2,250 non-cache
files** present in the two completed `work/` trees: original relative path, byte
count, SHA-256, archive/member, and nearby provenance records. Context links are
not a claim that all dependencies were recorded. There is no winners-only filter.
The archive also contains numerical failures, rejected repairs, exposed fresh
panels, preparation states, checkpoints, traces, source snapshots and commands.
[duplicate-bytes.json](duplicate-bytes.json) identifies identical bytes at multiple
original paths; those paths are all retained, not omitted. No local original was
deleted.

| Asset | Compressed bytes | Retrieval |
|---|---:|---|
| Oscillator lineage | 39,152,049 | [coupled_oscillators.tar.gz](https://github.com/gomercin/JEPA-Anything-experiments/releases/download/evidence-completed-2026-09-25/coupled_oscillators.tar.gz) |
| Field lineage | 306,879,896 | [mediated_patterns.tar.gz](https://github.com/gomercin/JEPA-Anything-experiments/releases/download/evidence-completed-2026-09-25/mediated_patterns.tar.gz) |

Total: **346,031,945 bytes (330.002 MiB)**, below the 500 MiB artifact cap.
Unpacked original data: 1,009,289,031 bytes. Archive hashes are in the manifest;
member hashes are distinct and not self-referential. The tag/release assets must
not be overwritten. [Release page](https://github.com/gomercin/JEPA-Anything-experiments/releases/tag/evidence-completed-2026-09-25).
Small byte-identical tables, freeze records and images are also in
[review/](review/) for browser inspection, with mappings in
[review-copies.json](review-copies.json). They are duplicates, not new outcomes.
Generated tracked evidence remains below 25 MiB.

## Reports, models and commands

[catalog.json](catalog.json) maps each report to its complete package, recorded
commands, protocols, result tables and exact frozen-model/configuration hashes.
Restore the original paths to follow historical local links or recorded commands.
**Those commands run science; restoration and the smoke check below do not.**

| Completed stage | Report | Restored package / identity |
|---|---|---|
| Linear oscillator calibration | [Phase A](../../experiments/coupled_oscillators/README.md) | `work/coupled_oscillators/phase-a-quick/`; `protocol.json`, results and matrices; no trained checkpoint was retained by that protocol |
| Optimization interrogation | [Interrogation](../../experiments/coupled_oscillators/INTERROGATION.md) | `work/coupled_oscillators/interrogation/`; executed runner and saved diagnostics |
| Frozen encoder controls | [Frozen encoder](../../experiments/coupled_oscillators/FROZEN_ENCODER.md) | `work/coupled_oscillators/frozen_encoder/`; per-panel executed runner, protocols, states/results |
| Nonlinear fully observed | [Phase B](../../experiments/coupled_oscillators/PHASE_B.md) | `work/coupled_oscillators/phase_b/`; includes failed `b0` numerical qualification and refined qualification |
| Deterministic partial history | [Phase C](../../experiments/coupled_oscillators/PHASE_C_PARTIAL.md) | `work/coupled_oscillators/phase_c_partial/`; delay/neural controls and diagnostics |
| Noisy state estimation | [Phase D](../../experiments/coupled_oscillators/PHASE_D_NOISE.md) | `work/coupled_oscillators/phase_d_noise/`; includes failed output-error invocation and adverse refinement |
| Field world / terminal maps | [Findings](../../experiments/mediated_patterns/FINDINGS.md) | `work/mediated_patterns/`; original pilots, qualifications, development/fresh/outward/replay packages; `outward-development-01/model.json` |
| Evolving response model | [Recursive response](../../experiments/mediated_patterns/RECURSIVE_RESPONSE.md) | `work/mediated_patterns/recursive_response/`; `frozen-01/kernel.json`, `state-uncorrected.json`, failed susceptibility correction |
| Cross-pulse repair | [Cross-pulse repair](../../experiments/mediated_patterns/CROSS_PULSE_REPAIR.md) | `work/mediated_patterns/cross_pulse_repair/`; both freezes, failed first fresh panel, selected `frozen-02/model.json` |
| Spatial AB reduction | [Two-way hybrid](../../experiments/mediated_patterns/TWO_WAY_HYBRID.md) | `work/mediated_patterns/two_way_hybrid/`; AB20 `frozen-01/model.npz`, failed lower orders |
| Frozen AB repeated use | [Repeated hybrid](../../experiments/mediated_patterns/REPEATED_HYBRID.md) | `work/mediated_patterns/repeated_hybrid/`; same AB20 model, its own freeze, controls, numerical checks and checkpoints |
| C qualification / composition | [Dual reduction](../../experiments/mediated_patterns/DUAL_REDUCTION.md) | `work/mediated_patterns/dual_reduction/`; C24 `frozen-01/model.npz`, C32 `frozen-02/model.npz`, both fresh panels, `checkpoint-02/state.npz` |

The selected AB20 hash is
`a128228fd027986ed17fbc1802f02a9d1820f7dffd215aa46fa6211c4c04bb6e`.
Original C24:
`232cdd3477d7e80f61decf18e5b0354f12f05c253de1c23f234d06c9c2729d35`.
Selected C32:
`d81f276cade69d40ef8b215ed9e38a225296061dfc74cf2d5b8d14615afdc7b1`.

## Scientific boundaries retained

- Oscillator A–D found ordinary identification, delay-state construction and
  filtering sufficient for these tasks. No distinctive OPF benefit was
  established; this is not general neural inferiority.
- Terminal maps, evolving pulse-response models, and spatial participant
  reductions are different constructions, not one unchanged model.
- Frozen AB20 passed the reported single and repeated interaction tests,
  including the selected return contrast. Interior state was not refreshed.
- **Original AB + C24** means FULL_AB+C24 (first-panel FR). It failed the same
  two late C-response windows as AB20+C24. The second panel did not record
  FULL_AB+C24. The first freeze did not establish failure arising only when
  individually passing replacements were composed.
- C32 was a composition-informed C-only enrichment; AB20 stayed frozen. The
  repaired AB20/C32 passed 12 new schedules from six preparations/three seeds,
  over 80–109 units, including the return contrast and resolved windows.
  This is not first-freeze independent-composition success.
- The environment remains resolved. Dense dual reduction was about 1.66 times
  slower than full reference. Early/sub-floor fidelity and broader regimes
  remain uncertified. No successor experiment was run during preservation.

## Provenance and gaps

[provenance-gaps.json](provenance-gaps.json) checks the 1,764 explicit source-hash
references found in per-panel protocols against bytes actually available here.
1,762 match; the two unmatched references name **one older oscillator README
version**, SHA `5ca62c3da07e74d801a0dc0af8eec283d03df6fe6381c9228b000617b06e11b4`.
Those exact historical prose bytes were not found. No original source is
fabricated to fill the gap. All runtime-source hashes in that checked set match
available bytes; this does not certify completeness of unrecorded provenance.

Early field pilots explicitly retain `retrospective_sources`; that label is
preserved, not promoted to contemporaneous execution capture. Some historical
records contain only partial environment/version information. The
[environment inventory](environment-current.json) is **current at preservation**,
not a reconstructed historical environment. Existing run versions, commands,
source snapshots, hashes and thresholds remain unaltered. Five analysis scripts
that previously existed only in ignored `work/` are additionally copied verbatim
into [historical_analysis/](historical_analysis/), indexed by
[historical-analysis.json](historical-analysis.json). Their original restored
paths remain the durable entry points for historical analysis commands; no
analysis was rerun for publication. Historical analysis scripts can overwrite
their own derived reports, so use a disposable restored copy if deliberately
rerunning them.

Only [replaceable caches](excluded-caches.json) were omitted from the extant work
trees. No extant substantive run file is intentionally local-only, and none is
classified merely as reproducible-but-unarchived. Historical temporary plotting
caches are replaceable; old `/tmp` command-output log names are not available as
original files, while the saved stage-local check logs are archived. Their exact
identity to vanished temporary logs is not claimed. No saved next-experiment
prompt or implementation was found in the scoped inventory. No unrelated
conversation export, external document, environment or private project data is
included. [Disclosure review](disclosure-review.json) records the public scope;
historical local cwd/interpreter strings remain in original records, not in new
installation instructions.

[Preserved source hashes](preserved-source.json) identify the original local
experiment files before publication handoffs. The root `.gitignore`, Makefile
and pytest-discovery edits predated preservation and are included as relevant
integration, not discarded as unrelated work. Scientific Python implementations,
tests, frozen reports and all work artifacts remain byte-identical. New changes
are publication tooling, living handoffs and test/dependency integration only.
Original licenses and citations remain in place: [repository license](../../LICENSE)
and [core license](../../jepa-anything-core/LICENSE).

## Explicit restoration and saved-data inspection

Cloning or importing the repository does not download evidence. With Python
**3.11 or newer**, choose a fresh destination (or a fresh clone with no `work/`
results). Never point restoration at the original populated lab checkout:

```bash
python3 evidence/completed-2026-09-25/restore.py --destination /chosen/empty/evidence-root
python3 evidence/completed-2026-09-25/restore.py --destination /chosen/empty/evidence-root --verify-only
```

The script downloads the two declared release assets into temporary storage,
verifies archive sizes and SHA-256, rejects unsafe paths, symlinks, duplicate or
unexpected members, refuses existing result files, restores original relative
paths, and verifies every member. A failed partial restore stays visible and is
never silently reused. Allow roughly 1.4 GB free space for download plus restore.
The direct Python downloader could not find a trusted issuer certificate in the
preservation host environment. TLS verification was not disabled. The documented
GitHub downloader was used; after a partial-download timeout, TLS-verified curl
HTTP range resume completed the large asset and its full hash matched. These
transport failures did not change any evidence bytes.

To predownload with authenticated GitHub tooling, into a **new** directory:

```bash
gh release download evidence-completed-2026-09-25 --repo gomercin/JEPA-Anything-experiments --pattern '*.tar.gz' --dir /chosen/new/assets
python3 evidence/completed-2026-09-25/restore.py --archives /chosen/new/assets --destination /chosen/empty/evidence-root
```

NumPy is sufficient for the read-only model/checkpoint/table smoke check:

```bash
python3 -I evidence/completed-2026-09-25/saved_data_smoke.py --root /chosen/empty/evidence-root
```

It opens saved AB20/C32 bases, their freeze configuration, the checkpoint and the
recorded aggregate table. It imports no experiment code, fits nothing, advances
no state, and does not claim to reproduce the science. Saved arrays are opened
with `allow_pickle=False`. For later code review, dependencies are declared in
[experiments/requirements.txt](../../experiments/requirements.txt), separately
from the unchanged core package. No installation or experiment runs implicitly:

```bash
python3 -m pip install -e './jepa-anything-core[dev]' -r experiments/requirements.txt
OPENBLAS_NUM_THREADS=1 make check PYTHON=python3
```

`make check` includes bounded deterministic unit checks, including tiny synthetic
optimizer/state-update fixtures; **it excludes the existing test that launches
the full Phase A `--quick` scientific panel**. Direct unfiltered pytest discovery
still contains that explicit historical panel test. No existing experiment
`quick`, `fresh`, `qualify`, train or fit runner is invoked by preservation/CI.

## Review status and local checks

[checks.json](checks.json) and [checks.log](checks.log) record checks performed
now, separately from historical test reports. The fast suite passed 202 tests
and 2 subtests, with the panel test deselected. Design/JSON/compile/structural
recipe/manifest checks passed. New archive tooling passed lint and formatting.
Applying the upstream core's stricter Ruff configuration to all historical
experiment source reports **58 existing style findings** (49 E501, 8 B007,
1 E731) in nine files; formatting would change 19 files. These findings are
recorded in [lint-findings.json](lint-findings.json), left unchanged rather than
rewriting executed science source. They remain review issues, not claimed fixes
or scientific failures. The two historical SVG review copies retain their
2,120 path-data trailing-space lines under a scoped `.gitattributes` whitespace
exemption; source/documentation checks remain enabled. Historical statements about then-run checks remain
literal. This PR is not labelled ready to merge.

## Independent remote verification

A fresh HTTPS clone at the source-snapshot commit was obtained from GitHub, not
from a local worktree. Both assets were downloaded from the published release
into a separate directory. All archive hashes and 2,250 restored member hashes
matched; 1,009,289,031 original bytes were restored. The fresh-checkout smoke
script loaded AB20/C32, their configuration and the time-6 checkpoint, and read
the saved 12/12 RR table. It used Python `-I`, stdlib and NumPy; no simulator or
editable experiment module was imported. This verifies persistence and saved
serialization, not scientific replication.

The final documentation head is pinned outside its own Git content in the
[immutable remote verification receipt](https://github.com/gomercin/JEPA-Anything-experiments/releases/download/evidence-completed-2026-09-25/remote-verification.json)
and the PR body. The receipt is a small additional release asset, separate from
the two scientific archives and their manifest. No existing asset or tag is moved.

## Pending Atlas ownership and next task

The exact historical pending paragraph remains in
[DUAL_REDUCTION.md](../../experiments/mediated_patterns/DUAL_REDUCTION.md#stop-and-pending-atlas-delta).
Its ownership remains **Effective Motif Dynamics / persistence-to-composability**;
Atlas is unchanged. Remote evidence pointers supplement that pending record,
without rewriting its execution-time language. The AB20/C32 cycle stays closed.
The separately chosen organization-conditioned-response investigation is
**unexecuted by this task**. It must use a separate branch and not append new
science to this preservation PR. See the living lab handoff and publication
receipt for the verified starting revision.
