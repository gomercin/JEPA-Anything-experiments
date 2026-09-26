# Critical review of the scoped implementation and claims

This is an explicit self-review of the actual diff, not an independent scientific
replication or a second reviewer. Review base: inspected main
`6e0eb17135341706e9e96ed3ab42a7c28a560ad6`. Final reviewed head and merge guard are
recorded in the PR comments after all fixes and checks.

- **Information boundary.** The pure model accepts a descriptor and amplitude.
  It never imports a solver, schedule, history label or future response. Fitting
  uses only the caller's training rows. No true R_without is supplied at inference.
  The snapshot test explicitly charges a t72 scan. The later retained test saves
  every forecast from t50 centers before constructing any later field array.
  Its G and response coefficients are unchanged from the preceding frozen models.
- **Pair selection.** The pure selector has no response argument. All candidate
  indices, descriptor/row hashes, metric, uncertainty and near flags are sealed
  before references. An optional overlap nomination uses present data only.
  Full fields appear solely as evaluator context. Selection is replay-tested.
- **Matching interpretation.** The screen includes numerical error and a factor-5
  local fitted sensitivity allowance. It is empirical, not a mathematical
  Lipschitz certificate. No pair survives it, so no robust aliasing or physical
  state insufficiency is claimed. Successful grouped/fresh center-only prediction
  supplies the positive approximation result. No response-informed state search
  or perturbation was needed.
- **Event and branch semantics.** Reuses the exact fixed additive source and
  original instrument. The signed overlap/source identity is independently tested.
  All responses subtract their own sham, on compatible boundaries; D1 then
  subtracts the independently predicted common response. The final +.02 probe
  remains fixed. No physical trajectory is normalized afterward.
- **Exposure and capacity.** All histories of a preparation stay grouped. Means,
  scales, SVD bases and regression coefficients are rebuilt inside each fold.
  The quadratic fits all three current centers; ridge selection is finite and
  train-only. High-ridge failures remain. The RBF and extra-descriptor ladder
  stop once centers pass. The 20101 former adverse case is exposed development,
  with held-preparation scoring, not reused as fresh. Two disjoint fresh seed
  sets serve the separate snapshot and retention contracts.
- **Numerical qualification.** Both closest-pair states are refined at half dt
  and double N from the supplied t72 initial fields. Matched per-R error bounds
  propagate conservatively into D1 and pair differences. Direct cancellation
  estimates and larger absolute-Y bounds remain separate. Floors were frozen
  before either fresh panel. Sub-floor *residuals* are distinguished from
  sub-floor *signals*. Extraction interpolation is not misrepresented as full
  preparation-history convergence.
- **State accounting.** Three centers and one G counter; no new physical scalar.
  The fixed-age output map stores 1,538 active numbers, plus G's 37 and amplitude
  normalization. It is not a three-number complete program or an evolving
  post-event response state. Saving an output trace costs 322 numbers. The
  diagnostic six-feature extractor is kept separate from selected three-feature
  inference.
- **Self-containment and safe outputs.** Exclusive output/serialization helpers
  and the active-run lock retain inherited protections. Six checkpoint resumes
  are tested in isolated processes without scientific solvers or field arrays.
  Portable evidence restoration preflights all files, hashes and collisions.
  CI executes fast synthetic/replay checks only, never the scientific runner.
- **Claims and historical continuity.** The result is fixed-age approximation
  sufficiency, with bounded once-measured pre-event propagation. It does not
  qualify new ages, arbitrary streams, post-event geometry, physical storage,
  minimality or Markov closure. The previous fifteen-state timing failure,
  repeated failures/K12 limits, composition-informed correction and all geometry
  failures remain unchanged. No repeated experiment or successor was launched.

Focused engineering correction during review: one synthetic test initially used
bit equality for decimal floating-point scale construction; it now checks the
specified tolerance at machine precision. No scientific threshold, reference,
model or outcome was changed. The retained figure uses the same snapshot-map
plotter; its README caption explicitly identifies G-propagated inputs.

Local full-suite results, remote retrieval and hosted CI are recorded alongside
the final reviewed head in the PR. Merge is conditional on their completion and
a live unchanged head/base check, without protection bypass.
