# Explicit critical self-review

Review scope: actual diff from lab base `7275c360a3caf2ae4f82ba0a283fdc520b92af58`,
new scientific modules/tests, living note and incremental evidence. This is an
agent self-review, not independent human approval. Final reviewed-head and
current-base checks are recorded on the PR after all fixes and fast checks.

- **Target-age leakage:** each interpolation fold excludes the entire held age
  and held preparation. Shared representation indices are separately recorded.
  The common scaling/SVD come from coarse training nodes only; added nodes cannot
  alter those coordinates. Existing-node coefficients match exactly across densities.
- **Node provenance:** only ages 14/17/25 were added, on twelve exposed groups.
  Old source arrays were checksum-verified. Fresh 24 was absent from all fitting,
  scaling, basis and interpolation selection. Calibration and interpolation claims
  are kept separate; the direct fixed-node tests do not qualify omitted contexts.
- **Local rule:** a pre-outcome review caught the held-20 nearest-node tie that
  could choose an all-left stencil. It was fixed before fitting to require the
  bracketing pair plus closest third node. Deterministic selection is tested.
  Linear and quadratic direct-curve/coefficient interpolation agree to roundoff.
- **Physical semantics and algebra:** inherited fixed additive profile, extractor,
  simulator, matched probe/sham boundaries and C readouts are reused unchanged.
  Every R is its own probe-minus-sham response; D1 subtracts independent predicted
  R values. No true baseline, history label or future response enters inference.
- **Fresh seals:** the actual fresh runner is tested with future access refusing
  to run until all predictions and their checksums exist. Three new groups,
  both signs, held24 and control25 pass without refitting. Frozen runtime/model
  hashes and execution revisions are retained. Later changes affect reporting
  and accounting, not the frozen interpolation, coefficients or validation path.
- **Numerics:** half-dt/double-N matched pairs at 14/25/24; five-times R refinement
  bounds, arithmetic floors and triangle propagation to D1. Direct D and larger
  absolute-Y discrepancies are separate. No denominator or acceptance threshold
  changed. The repeated 13.49% miss is retained with its sub-floor threshold margin.
- **Conditional boundaries:** G is evaluated only after snapshot success and
  explicitly labeled exposed. Its predictions use t50 fields only and are sealed
  before evaluator later centers. Repeated checks use exposed ±.02 cancellation
  prefixes only, no fit. Literal and common-predicted-baseline addition remain
  distinct; evaluator true addition and K12 are not deployable corrections.
- **Reporting fix:** per-sign use of a cross-sign deduplication helper made the
  positive-only R_without summary empty. The original report output is preserved;
  `details-01/fresh-by-sign.json` gives explicit maxima for both signs. This is
  a presentation correction; saved predictions, full score tables and gates agree.
- **Storage/scope:** eight separately calibrated matrices, shared basis/scaling,
  node contexts, buffers, field acquisition, G and four-query repeated costs are
  counted. No autonomous memory, minimality, physical-mode discovery, universal
  smoothness, arbitrary timing/streams or accurate nonlinear composition claim.
- **Preservation/publication:** old reports/models/archives are untouched. New Git
  evidence remains below 25 MiB; a unique prerelease/non-latest supplement holds
  only additional new material. Full remote retrieval is checked before merge.
  Checkpoint/output creation and restoration retain overwrite/symlink protections.

Validation includes synthetic linear and curved context fixtures, response
arithmetic, grouped/full-age exclusion, common coordinates, exact-node behavior,
serialization, actual prediction sealing, retained access separation and portable
score/numerical reconstruction. CI runs fast tests only, never the scientific panel.
No automatic successor or new Atlas route is created.

Local validation: `make check` passes 358 tests plus two subtests; the inherited
session-local evidence skip and scientific CLI deselection remain. Scoped Ruff
and `git diff --check` pass. All six new portable replay tests pass. No material
code/claim blocker remains from this self-review. Hosted CI and remote retrieval
are checked on the final published head, rather than inferred from local tests.
