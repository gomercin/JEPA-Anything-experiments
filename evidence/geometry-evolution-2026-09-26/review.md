# Critical review of the geometry continuation

Review scope: the actual new updater, acquisition/evaluator, fitting and scoring
code; the frozen F dependency; all new evidence; report/README claims; fast tests
and safe restoration. This is an explicit self-review, not a claim of an
independent reviewer or a second account. Scientific sources were frozen at
`cbabf57`; report/tests at `2c42405` do not alter them.

## Information boundaries and dynamics

- `geometry_model.py` imports only NumPy and the old solver-free response module.
  Its velocity depends on three numeric offsets and shared static coefficients.
  It has no clock, field, callback, ID or history argument. RK4 is the same rule
  at each step. The step count does not enter the derivative. Checkpoint hashes
  bind the static updater, and resumed/segmented trajectories agree.
- The exact previous F artifact/hash and all previous scientific Python files
  are unchanged. Scaling/basis/coefficients are loaded, not reconstructed or
  refitted. F always receives predicted centers for deployment. Exact later
  centers enter only the named evaluator diagnostic.
- Full-field center rates use the actual fixed-window chain rule and Fourier
  equation RHS. They are training targets/diagnostics only. No initial rate is
  retained in the primary state. The fixed anchors and unwrapped local coordinate
  agree with the actual original extractor; windows do not cross the boundary.
- Each preparation's histories share ancestry only in the evaluator. Every
  response prediction is independent. Paired subtraction occurs in scoring.
  Probe/no-probe branches originate from the same later state. The evaluator
  finishes unforced origins before probing, so an early probe cannot contaminate
  a later origin. Tests independently recover initial branch readouts from saved
  actual fields and recover all target responses from their own sham subtraction.
- Seals at 50 and 60 precede their respective future continuations. The separate
  initialization at 60 never resets the 50-origin prediction. A mocked acquisition
  test enforces that order; a fresh-process inference test blocks solver/evaluator
  imports and runs both resumed G and the serialized frozen F.

## Selection, exposure and metrics

- Seven complete preparation groups define development folds; all writes and
  times of each group remain together. Mean/scales and G coefficients are fit
  within each fold. F was already frozen from the prior task. 147 rate rows across
  21 trajectories are not 147 independent preparations. Interior 80/85/90/95 samples
  are stored evaluator evidence but excluded from fitting, scaling and selection.
- The quadratic extension follows a recorded affine fractional-motion miss.
  Both ridges are retained. Selection uses held-out free-rollout coordinate RMS,
  never an attempt to move centers to offset F's response error. Fresh 14101/2/3
  outcomes did not trigger a model repair. Target 90 is an interpolation check,
  not an unseen history family or a new physical envelope.
- Both R and Delta_R use their own RMS denominators, separate mass/moment and
  whole/late windows. No stationary background or write offset dilutes errors.
  Absolute residuals/maxima and per-preparation/worst scores are stored. Saved
  arithmetic is independently recomputed by tests without running simulations.
- Signed update/readout residuals reproduce total residuals exactly locally.
  Cross terms are kept, including their signs; no RMSE addition or causal share
  is asserted. Exact-current F remains accurate in the new sampled envelope.
- The shared numerical floor is recomputed from half-dt/double-N histories, from
  the same prepared initial array through write/wait/probe. Checks cover a new
  development time, the worst fresh contrast and the weak-motion failure. The
  response floor happens to remain 1e-12; it is not blindly inherited. Coordinate
  uncertainty has a separate rule. No confidence interval or uniform rigorous
  error bound is claimed.

## Findings and interpretation

The report explicitly says the joint criterion does **not** pass. All420 response
and 54 absolute coordinate gates pass, but two of 54 fractional-motion gates fail:
the same nearly stationary A trajectory at two origins. The failure is numerically
resolved, survives halved update stepping, and has a systematic local rate bias.
Quadratic improves physical coordinates; affine gives slightly better response
scores through error cancellation and remains a useful cheaper readout control.
Neither result proves descriptor sufficiency, minimality, storage localization or
closure under interventions. No extra state or automatic successor is inferred.
This preserves the old snapshot success and finite-retention/other historical
results. The cost includes F's 732 learned scalars, G's 36 and the full-field initial
scan, plus reference/rate/fitting work; it is not a three-number system-cost claim.

## Blockers found, fixed and re-reviewed

1. The first copied restore wrapper had the wrong incremental prefix
   `geometry_evolution_transmission/`. Both the manifest test and verify-only
   command rejected it. Corrected only the new wrapper to `geometry_evolution/`;
   all 429 members then verify and exclusive restoration tests pass. Historical
   helpers and evidence were not changed.
2. Strengthened the boundary tests to cover quadratic checkpoint/resume and
   train-only processing as well as affine, and to execute F in the solver-blocked
   fresh process. Added saved-origin readout checks and independent floor/gate/
   signed-cross-term recomputation. These changes do not alter frozen predictions.
3. Kept the remaining fractional-motion failure prominent in the report lead,
   tables and dedicated close-up, rather than calling every criterion passed.
   Kept both the failed optional diagnostic import and its conservative CPU charge.

Validation: `make check` passes 274 tests plus two subtests, with the existing one
session-manifest skip and one historical scientific-panel deselection. New
publication tests pass 5/5. `git diff --check` and safe verify-only pass. Hosted CI,
remote retrieval, reviewed-head/base identity and actual merges are checked
separately; final metadata is recorded in PR comments. No unresolved material
code, boundary, score or claim blocker was found in this reviewed scope.
