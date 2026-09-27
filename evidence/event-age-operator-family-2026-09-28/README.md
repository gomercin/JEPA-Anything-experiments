# Event-age operator family: portable evidence

**FIXED-AGE MAPS WORK BUT SHARED FAMILY FAILS.** The [findings](../../experiments/mediated_patterns/EVENT_AGE_OPERATOR_FAMILY.md)
retain a bounded development interpolation limit, not a hidden-state claim.
Five center-only quadratic maps at ages10/15/18/20/30 each pass480 grouped gates
using common rank4 response bases. No basis expansion was needed. The age-blind
pooled control fails; quadratic age context improves it; one fixed-knot natural
spline improves it further but still fails38/1440 complete-age-holdout gates.
All large R gates pass. The prospective new age24 and three new preparations
remain untouched. Retained G and repeated-input stages were not run.

| Best tested shared family | Held15 D1 | Held18 D1 | Held20 D1 |
|---|---:|---:|---:|
| Quadratic age |57.33%|42.98%|72.65%|
| Natural spline |15.49%|11.47%|18.99%|

Whole preparations and entire ages are excluded from each relevant fit, center
scaling and response SVD. The spline passes grouped fitting when all ages are
present (worst D1 6.974%); that is not interpolation qualification. Failed ridge
variants and all fitted folds remain available. No one-hot timing ID or response
lookup is used by the shared runtime. Known age is explicit scheduling context.

The worst failure is the exact state/trio already refined prospectively at age20:
20101/odd04,+.02,late mass. Error3.856e-12, signal2.030e-11, bound2.277e-13;
error is16.93 times the empirical matched-history bound. Direct D1 refinement
and larger uncorrelated absolute-trajectory differences remain separate. Bounds
refer to future branches from the declared current snapshot, not a continuum
certificate for its entire preparation history. No threshold was relaxed.

This package contains **996 new members**, including every new matched branch,
current-state table, reused-source identity, numerical refinement, failed and
successful fit, fold record, prediction, score, figure, command and cost receipt.
Historical archives/models are not republished. The one reporting-import syntax
failure and its corrected successor remain separate. Scientific cost is
1029.16 CPU seconds (17.15 minutes), including failures and inspection allowance.

The disjoint package has two parts:

- Git regression archive:8,734,551 bytes, SHA256 `7bae5dfe0652784dd1799903bc32a65e3c6668dd37da4b54bc8aad567e821166`.
- Unique [non-latest release supplement](https://github.com/gomercin/JEPA-Anything-experiments/releases/tag/event-age-operator-family-20260928-9a96632):31,281,667 bytes, SHA256 `4be427380956b7ecad8db80a71bd1a4a30417e7536a2de6c6444f359f5b19384`.

`artifacts.json` records every member's location, size and hash. The supplement
holds remaining new branch arrays and fitted fold coefficients. The local subset
contains full grouping metadata, score tables, numerical pilot/refinement,
selected replay folds, training/current-state tables and figures. Fast CI uses
only that subset and does not download assets or run scientific panels.

```bash
# Local regression subset; no network or scientific simulation:
python evidence/event-age-operator-family-2026-09-28/restore.py --fixtures --verify-only
python -m pytest evidence/event-age-operator-family-2026-09-28/tests -q

# Complete new package only; downloads its one checksum-verified supplement:
python evidence/event-age-operator-family-2026-09-28/restore.py --verify-only
python evidence/event-age-operator-family-2026-09-28/restore.py --destination /private/tmp/operator-evidence-new
```

Restoration rejects overwrites, symlink targets, traversal, extra members, size
and hash mismatches. `--archives` accepts a predownloaded directory containing
both named archives. All files restore under
`work/mediated_patterns/event_age_operator_family/`. [Exact commands](commands.txt)
include source revisions; [critical review](review.md) distinguishes scientific
limits, information boundaries and publication checks.

Figures separate [operator changes at identical centers](figures/operator-age-curves.png),
[common-basis coefficient trajectories](figures/coefficient-age-trajectories.png),
[held-age gate failures](figures/held-age-errors.png), and
[large R versus smaller D1 in the adverse case](figures/adverse-held-age.png).
Coefficients have no physical-mode interpretation. No fresh timing success,
new G qualification, autonomous memory, repeated composition, minimal state,
unique storage, arbitrary-stream closure or successor is claimed. All prior
successes and failures remain unchanged. PR comments carry reviewed-head and
actual merge confirmations rather than a metadata-only follow-up PR.

Local checks pass338 tests and2 subtests (one inherited skip, one historical
scientific-CLI deselection). The192-member regression subset and all996 complete
members verify locally. Remote retrieval and actual reviewed-head merge status
are recorded in the linked PR comments after publication.
