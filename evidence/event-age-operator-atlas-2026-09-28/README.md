# Finite-resolution event-age operator atlas

**LOCAL OPERATOR ATLAS SUFFICIENT.** [The findings](../../experiments/mediated_patterns/EVENT_AGE_OPERATOR_ATLAS.md)
qualify a bounded context-indexed response interface, without hidden memory.
Adding ages 14, 17 and 25 makes a local quadratic pass complete-age/grouped
holdouts at 15/18/20 and a prospective fresh age-24 test on three new preparations.
Current A/B/C centers, conditioning amplitude and known age are the only inputs.

| Held age | Coarse quadratic D1 | Refined quadratic D1 |
|---|---:|---:|
|15|23.00%|1.93%|
|18|22.95%|3.22%|
|20|16.83%|5.87%|

Every comparison excludes the held age and preparation from fitting, scaling and
basis construction. Coarse/refined models have identical shared coordinates and
identical coefficients at their common nodes. New fixed-age maps all qualify.
Linear refinement improves all three ages but fails age 20; that candidate stays
visible. PCHIP and a fourth added age were not run.

The eight-node frozen atlas passes all 240 distinct fresh snapshot gates at
unseen age 24 and trained control 25, both signs. Worst age-24 R/D1 is
0.104%/1.423%. All forecasts precede nonlinear future access. The worst D1 signal
is 4.320e-11, residual 6.146e-13, conservative floor 2.349e-13. Matched half-dt
and double-N checks at development 14/25 and fresh 24 preserve the pass; direct
contrast and larger absolute-trajectory errors remain separate.

Unchanged G passes a subsequent **exposed** t50-only diagnostic, not a new fresh
retained qualification. The requested historical repeated check removes the old
isolated age-15 failure. Common-baseline independent addition retains one timing
D12 miss (13.49%; true single-event addition 13.34%), whose excess over 10% is
below the inherited numerical bound. Literal addition has ten timing D12 misses
because it also retains disagreement between predicted zero-event contexts.
Resolved K12 remains inaccurate. No composition fit or new repeated panel.

## Portable package

The package contains **817 new members**, 72,411,895 uncompressed bytes. It includes
all new branches, calibrated and failed operators, folds, forecasts/seals,
refinements, figures, source identities and CPU receipts. A small combined
current/target evaluation table supports standalone replay of the new comparisons;
old field archives are not duplicated. Source/report revision:
`989360432269593d74190248481a108cd2f34404`.

The two parts are disjoint:

- Git regression archive: 395 members, 15,732,160 bytes, SHA256
  `c60bfb1e72bee8c9d7f1cd3ed62a3198808fe2e60fe4557da8d60aff87d036b4`.
- [Unique non-latest supplement](https://github.com/gomercin/JEPA-Anything-experiments/releases/tag/event-age-operator-atlas-20260928-9893604):
  422 members, 20,879,081 bytes, SHA256
  `f56c3998cdca7c2594afb900bb566e5181b55bf6227746690ab7eacfffa15972`.

`artifacts.json` specifies every member and archive checksum. The supplement
contains remaining new development branches and detailed curvature arrays.
All fitted operators and replay tables are local for fast CI. Restoration rejects
overwrites, symlinks, traversal, extra members and size/hash mismatches.

```bash
# Local replay only; no scientific simulation or network:
python evidence/event-age-operator-atlas-2026-09-28/restore.py --fixtures --verify-only
python -m pytest evidence/event-age-operator-atlas-2026-09-28/tests -q

# Verify/download only this new package:
python evidence/event-age-operator-atlas-2026-09-28/restore.py --verify-only
python evidence/event-age-operator-atlas-2026-09-28/restore.py --destination /private/tmp/atlas-evidence-new
```

`--archives` accepts a directory with both predownloaded named archives.
[Exact commands](commands.txt) identify every execution revision. Reproduction
of the scientific stages needs the inherited repeated/event-age/aliasing/operator
inputs through their existing indexes; replay of this package does not rerun them.

Scientific cost: **1,026.70 CPU seconds / 17.11 minutes**, including the inspection
allowance and all fits, diagnostics and refinements. Editing/tests/publication
are separate. The atlas stores 3,218 active fit/basis/scaling values plus node
ages and other stated constants; it has no evolving response memory. Current
field acquisition, context, buffers, G and multi-query costs are explicit.

## Figures and review

- [Coarse versus refined errors](figures/resolution-comparison.png)
- [Prediction-space curvature](figures/operator-curvature.png)
- [Fresh R and D1 separately](figures/fresh-response.png)
- [Pre-event retained geometry](figures/retained-geometry.png)
- [Exposed R11, D12, D2|1 and K12](figures/exposed-repeated.png)

[Critical review](review.md) records information boundaries, fixes and claim limits.
`details-01/fresh-by-sign.json` supplies explicit per-sign maxima; the earlier
report's deduplicated positive baseline entry is not a zero-error claim.
No historical artifact, failure, route or checkpoint was changed. PR comments
record exact reviewed heads, remote retrieval, hosted CI and confirmed merges.

Local `make check` passes **358 tests and two subtests**, with one inherited
session-local evidence skip and one historical scientific CLI deselection.
The six new portable checks replay grouping, identical atlas coordinates,
forecasts, scores, numerical floors, fresh seals and restoration safety.
