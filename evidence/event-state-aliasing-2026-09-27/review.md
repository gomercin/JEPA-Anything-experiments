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
model or outcome was changed. Portable replay then caught a copied restore-prefix
suffix; the new helper now accepts exactly event_state_aliasing/, with the same
exclusive path protections. The replay assertion was also corrected to compare selector-owned keys while
checking the separately added present-field diagnostic metadata independently.
All replay tests were rerun. The retained figure uses the same snapshot-map
plotter; its README caption explicitly identifies G-propagated inputs.

Local full-suite results, remote retrieval and hosted CI are recorded alongside
the final reviewed head in the PR. Merge is conditional on their completion and
a live unchanged head/base check, without protection bypass.

Hosted Linux CI at initial head 45547d7 exposed cross-platform last-bit differences
in the matching norm and response matrix products (maximum response difference
1.06e-22). The portability assertions now use relative 1e-12 / absolute 1e-22
response tolerance, many orders below the scientific floors; selector numeric
values use relative 1e-14. Hashes, pair order, near flags and pass decisions stay
exact. Each isolated process additionally compares resumed and uninterrupted G
states bit-exactly on its own platform. Frozen arrays, coefficients, scientific
metrics and archive bytes were not changed. Affected replay tests were rechecked
locally and exact-head hosted CI was required again.

The next hosted run isolated one remaining harness error: its supposed
same-platform comparison still began at a macOS-generated midpoint checkpoint.
The corrected test generates and serializes a new local midpoint for bit-exact
segmented/uninterrupted comparison, while separately checking the published
checkpoint against local advancement at relative 1e-14 tolerance. The observed
center difference was 1.73e-18. This changes no runtime, scientific artifact or
acceptance threshold; the entire evidence suite was rerun.
