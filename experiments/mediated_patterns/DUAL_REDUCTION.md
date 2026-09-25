# Dual reduction through the resolved shared medium

**A C-only enrichment produced a working connected model, but the first independently frozen attempt did not pass every fresh case.** C24 was selected and qualified with full AB before any RR predictions. On its first fresh panel, RR had 7 passes, 2 late-response failures and 3 unresolved return cases. After those outcomes, C32 was selected, requalified in FR and frozen. AB20 remained unchanged. The adapted RR passed **12/12 new schedules from six preparations and three new seeds**, including the selected controlled return contrast. This is composition-informed adaptation, not an unqualified success of the first independent freeze.

![Selected model and unchanged comparators](../../work/mediated_patterns/dual_reduction/analysis-02/responses.png)

The representative trace is seed 5011, C39, with pulses at 0/11/29. FF is the full reference; RF reduces AB; FR reduces C32; RR reduces both. Gray `RR_original` uses frozen C24 on the same fresh preparation and inputs. Near-overlap of total curves is supplemented by signed errors and the much smaller controlled contrast below.

![Selected return contrast](../../work/mediated_patterns/dual_reduction/analysis-02/return_contrast.png)

Gray shading marks instantaneous signals below 2e-7; acceptance uses window RMS. Early/cancelled sub-floor relative fidelity remains uncertified. These scores concern the retained discrete field problem, not equally precise continuum error guarantees.

## Construction, ownership and access

The SH35-plus-finite-mediator equations and all coefficients are unchanged:

```text
u_t = r*u - (1 + d_xx)^2*u + b*u^3 - u^5 + g*m*u
tau*m_t = D*m_xx - m + s*u^2
r=-.67, b=2, g=.05, s=1, D=64, tau=10
L=192, N=768, dx=.25, dt=.00625; periodic; float64 ETDRK4
```

See README/FINDINGS for the base-model sources and the status of our mediator extension. No oscillator panels, neural models or new physical laws were introduced.

| Owner | Evolving state | Static initial information | Exchange |
|---|---|---|---|
| AB | frozen 20 coordinates replacing 248 interior u values, x in[-34, 28) | unchanged 248x20 basis; 248-value initial template | current mediator feedback and full projected fourth-order operator; emits its own reconstructed u/source |
| C32 |32 coordinates replacing 72 interior u values, x in[C-9, C+9) |72x32 basis; 72-value initial template; fixed preparation anchor | same physical channels; relative pulse applied to its own reconstructed u then projected |
| Environment |448 exterior u values and all 768 mediator values | initial exterior/mediator snapshot, fixed periodic mesh | owns transport, self-feedback and all exterior tails |

C's region contains the radius 8 stimulus and smooth readout, plus one unit of padding. Its tails remain resolved: 0.68–0.87% of initial u-squared energy in C's nearest-center periodic cell lies outside the reduced interior. This is reduction of participant **interiors**, not removal of every field associated with C. It is not a universally small communication interface. The C diagnostic forces have 64 coordinates (mediator/direct force projections); computing them still uses the shared spatial environment. AB retains its 40 force coordinates.

Let P embed separate AB and C bases and the exterior identity. No joint basis is learned. With initial interior template T, the same known equations are projected:

```text
u = T + P*q
q_t = P.T*L_u*P*q + P.T*L_u*T + P.T*(b*u^3-u^5+g*m*u)
m_t = L_m*m + s*u^2/tau
```

The coupled linear operator is diagonalized as a numerical coordinate change for ETDRK4. AB's self-operator, basis, nonlinear rule, source and calibration are unchanged; environmental cross terms are assembled from the supplied operator. There is no invented boundary clamp, omitted direct-field path, independently summed nonlinear source, duplicated mediator or fitted sensor-to-actuator gain. A full-rank C partition matches monolithic readouts within 1.31e-11; the AB-only generalized assembly matches the old solver exactly in the pilot.

One initial snapshot is allowed. Thereafter the primary solver retains only the two local coordinate vectors, resolved environment, static templates/bases and clock. No fine interior residual evolves, no truth centers are fetched, and there is no reset or output assimilation. The jump is the original relative operator `u_after=(1+a*w)*u_before`, reconstructed and projected; it is not an additive force or a future truth-derived increment. The sensor and pulse anchor remain fixed. The maximum fresh C32 pulse mass projection error is 1.49e-8; initial reconstruction error is at most 2.23e-14. No true reference is restarted from reconstructed fields.

## Selection, failure and repair

**Observation:** compact C candidates could look accurate over a whole trace yet fail late windows. Alternatives included pulse projection, numerical exchange, spatial approximation, projected evolution and feedback amplification.

**Tests:** C's POD was built only from existing full-AB/full-C development trajectories: seeds 71/82, C38/42, C pulses+/-0.1 and one A pulse+.05. Drift, stimulus response, selected return response and actual initial jump directions were normalized as declared in `calibration-01/fit.json`. Operators were derived from the equations; no transition/readout was fitted. Selection used closed-loop FR, not RR.

- C8 failed all 3 initial cases: peak C-response error reached 0.00255 against the unchanged 0.001 absolute gate.
- C16 reduced that transient error but failed late relative gates in 4 cases; late absolute errors were only about 1e-7 to 9e-7. These failures remain saved.
- C24 had no accuracy-gate failures in 4 FR cases. Three had resolved qualification; the distant equal-reversal return RMS 1.94e-7 was below the existing 2e-7 floor. That case was never counted as passed.
- C24 froze before RR. Two development placements had no accuracy-gate failures, with the same weak case unresolved.
- First fresh panel: RF: 9 passes/3 unresolved/0 failures; FR: 5/3/4; RR: 7/3/2. Failures were late B/C response gates, not return-contrast gates. Two FR errors partly cancelled in RR. Thus this was not a demonstrated case of separately accurate participants failing solely through composition.

**Discriminators:** in an exposed C41 triple, best-in-C24-basis late C-readout RMSE was 1.37e-11 while evolved FR error was 7.77e-8. Halving dt left its 1.1145% late NRMSE essentially unchanged. The event mass projection error was also far smaller than the whole transient errors. These observations weaken readout expressivity alone and time integration as explanations. They support inaccurate projected evolution as the practical problem; they do not prove state insufficiency or identify missing physical memory.

**Repair:** retain 32 rather than 24 columns from the **same original C POD**. No new calibration trajectories were fitted, no AB coefficient changed, and no combined residual was fitted. This enriches the projected spatial evolution. It is explicitly composition-informed because the first RR/fresh outcomes were already exposed. C32 passed FR on original single/reversal regressions and exposed C41/C39 triples, then froze before a new untouched panel. It later also repaired both original RR failures in separate regression runs. No minimum dimension is established; intermediate orders were not searched.

## Untouched validation and controlled return fidelity

Final seeds 5011/5022/5033, C39/41: six whole preparations, two related schedules each. Odd/even seeds use preparation ages 20/60. Training used neither these preparations nor their outcomes. The second panel's schedules were fixed before access:

```text
two:   (0,+.080), (12,-.070)                 horizon 80
three: (0,-.075), (11,+.070), (29,-.080)      horizon 109
```

The first panel's seeds 4011/4022/4033 and different 10/9/27 schedules became development evidence after failure. The second panel is fresh within-family transfer after that exposure, not blind discovery of a new family. There are three paired seeds, not 12 independent preparations.

Each FF/RF/FR/RR run has its own sham, stimulated, interrupted stimulated and interrupted sham branches. The existing AB-region intervention replaces only mediator feedback in `g*m*u` with that solver's own sham stage values. The direct pattern pathway remains. Primary prediction never receives replay. The contrast is `(stimulated-sham)-(cut_stimulated-cut_sham)` in each configuration. No-stimulus matching is exact in the tested hybrids.

Inherited gates stay unchanged in whole/post-event/final windows: B NRMSE10% and max 2e-5; C NRMSE1% and max 0.001; resolved return NRMSE10% and max 1e-6, with RMS floor 2e-7. Early5 relative scores are not certified. Additional initial/local/jump gates are preserved in both freeze files. Original observed scores were never substituted for gates.

| Configuration | Passed / 12 | Worst B trace NRMSE | Worst C trace NRMSE | Worst return trace NRMSE | Worst resolved return window |
|---|---:|---:|---:|---:|---:|
| RF: AB20 + full C | 12 | 0.443135% | 7.6211e-05% | 0.666233% | 1.5303% |
| FR: full AB + C32 | 12 | 0.00053294% | 2.6686e-05% | 0.000376621% | 0.0013044% |
| RR: AB20 + C32 | 12 | 0.443081% | 8.01793e-05% | 0.665983% | 1.529% |
| Comparator: AB20 + C24 | 10 | 0.4407% | 0.00552243% | 0.621025% | 1.29821% |

All selected-model whole and last-return signals are resolved. Some early/cancelled windows remain unresolved in every case. C32 does **not** improve every scalar: C24 sometimes partially cancels AB's return error. RR32 instead tracks RF very closely, consistent with the remaining return error being dominated by unchanged AB. Across the final panel, signed non-additivity `RR-RF-FR+FF` has maximum 2.66e-11 for C response and 6.29e-13 for its return contrast. This is approximation-error bookkeeping, not a new physical interaction or additive causal budget.

| Preparation | RR two-event return NRMSE | RR three-event return NRMSE | RR worst resolved window | C24 failures on these schedules |
|---|---:|---:|---:|---|
| s5011-c39 | 0.3924% | 0.4722% | 1.0696% | none |
| s5011-c41 | 0.0647% | 0.0521% | 0.0807% | none |
| s5022-c39 | 0.5603% | 0.6660% | 1.5290% | three |
| s5022-c41 | 0.0756% | 0.0556% | 0.0970% | three |
| s5033-c39 | 0.2443% | 0.2934% | 0.6456% | none |
| s5033-c41 | 0.0412% | 0.0312% | 0.0435% | none |

The second-event-minus-first-only continuation also passes all resolved gates; RR32 return NRMSE is 0.299% after event 2. A no-return control would have tiny whole C-response errors (0.000178–0.002997%), but it omits the selected contrast and is not a valid return-fidelity substitute.

## Numerical scope, coverage and limitations

The original mesh/domain refinements remain applicable to the retained family and the identical development reversal. New full-rank interface and whole-quartet timestep tests qualify the new assembly. On the repaired fresh C41 triple, dt/2 changes actual return contrast by at most 5.51e-14 across FF/FR/RR; the conservative paired response-subtraction bound is 6.71e-8. Earlier new-panel qualification had bound 8.05e-8. Both remain below the frozen 2e-7 floor. These are empirical refinement diagnostics, not rigorous continuum error bounds or mesh-independent convergence of a POD basis.

Fresh C center drift is at most 0.0447 and width change 0.0815. RR AB separation changes by at most 0.0606; no recentering enters prediction. The fixed geometry is supported over these schedules, not arbitrary motion. The extra C-field RMS criterion, initial projection criterion, sham-background criterion and event projection criterion all pass. Maximum RR C sham mass discrepancy is 1.47e-5 (gate 0.005), reported separately from response subtraction.

C's incident-force diagnostics extend outside training coverage: maximum marginal excursion 4.01 training spans, standardized centered joint nearest-neighbor distance 5.37, and rate ratio 6.86. At least one marginal channel is outside at all samples in the worst cases. These diagnostics use training-only scaling and are not calibrated error bounds. Nothing is clipped; successful tests support only these trajectories.

## State, storage, access and cost

| Configuration | Physical evolving scalar count | Participant interior coordinates | Resolved exterior u | Mediator |
|---|---:|---|---:|---:|
| FF |1536|248 AB +72 C grid values|448|768|
| RF |1308|20 AB +72 C grid values|448|768|
| FR32 |1496|248 AB grid +32 C|448|768|
| RR32 |1268|20 AB +32 C|448|768|

The RR implementation stores 500 real coupled pattern coordinates (a numerical rotation of the 52 interior plus 448 exterior coordinates), 385 complex mediator Fourier values, and a clock. Its physical mediator has 768 degrees of freedom; the FFT payload has 770 real slots. There is no growing history buffer. The bounded external schedule is separate input bookkeeping.

Static data include 4, 960 AB basis coefficients, 2, 304 C basis coefficients, 320 initial template values, and derived dense caches: a768x500 lift, 500x500 rotation, 500 eigenvalues, 500 constant-force values and ETD coefficient arrays. Total directly owned NumPy arrays in one RR instance occupy 5, 217, 384 bytes, including 10, 160 evolving payload bytes; temporary stages/Python/runtime overhead are extra. Four branches are used for controlled evaluation, while unrestricted deployment needs one. The complete compressed checkpoint is 69, 972 bytes; it contains no future field/replay arrays. A fresh-process restart at time 6, before the input at 11, reproduces the uninterrupted RR109-unit trace **exactly**, with truth construction/stepping disabled; it also matches the saved primary RR trace exactly.

Initialization reads one 1536-scalar field snapshot. Retained templates and initial environment are charged above. Basis construction used prior full-field snapshots; operators are calculated, not learned from future validation. Each update reconstructs a full profile for nonlinear quadrature, projects it, and advances the explicit mediator. This is not a computational speedup: final-panel mean instrumented runtime ratios versus FF are RF1.667, FR1.875 and RR1.663. The total primary world is neither a52-state model nor cheaper merely because its interiors are smaller.

Timed new simulation/fitting sections, including failed candidates, refinements, regression, checkpoint and quick reproduction: **32.080 CPU minutes; 32.133 summed wall-timer minutes**, below the 45 CPU-minute cap. These are sums of recorded section timers, not end-to-end elapsed session time; editing, parsing/reporting/plotting and routine checks are outside them. The per-run caps were 120 CPU seconds, 180 wall seconds and 768MiB process RSS after the pilot. `budget.json` and `budget-audit.json` preserve the accounting. No divergence was clipped or omitted.

## Reproduction, checks and provenance

Quick demonstration using the selected frozen C32 (use a new output directory):

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.mediated_patterns.dual_reduction quick --freeze frozen-02 --output work/mediated_patterns/dual_reduction/quick-review-01
```

It runs the full reference, all three configurations and the old RR24 comparator for one repeated sequence. The executed `quick-01` reproduces its corresponding saved final-panel fields/readouts exactly. Scientific panels are deliberately outside CI.

Panel reproduction commands:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.mediated_patterns.dual_reduction fresh --freeze frozen-01 --output work/mediated_patterns/dual_reduction/reproduce-first
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.mediated_patterns.dual_reduction fresh --freeze frozen-02 --output work/mediated_patterns/dual_reduction/reproduce-selected
```

`commands.txt` lists the executed scientific/analysis invocations and checks. Each panel retains its pre-execution protocol, source snapshot and hashes. Frozen models and all arrays/traces are under `work/mediated_patterns/dual_reduction/`; analysis JSON contains per-case, absolute, normalized, window, background, geometry and coverage metrics. The original first panel, C8/C16 failures, C24 first freeze, and regression failures are preserved, not relabelled as fresh successes. The contemporaneous protocol/amendments are retained in `protocol-history.md`.

Files added: `dual_pair.py`, `dual_reduction.py`, `dual_analysis.py`, `tests/test_dual.py`, this report. The experiment README receives only an additive handoff. Tests cover complete-rank consistency, unchanged AB assembly, source ownership, absent evolving interior residuals, projected relative events, causal scheduling, own-sham matching, no truth/extractor access on restart, split separation, strict serialization and overwrite refusal. **194 tests passed plus 2 subtests**: 49 field-experiment, 66 core, 33 skill, 46 oscillator. `make check`, Ruff and `git diff --check` pass. Tests assert implementation boundaries, not scientific superiority.

Local branch `codex/compose-reduced-participants`, base Git revision `c6e6c88f3ef75a4ce7acd660d6fa5779d995512c`. Sources remain local/uncommitted reviewable additions, not pushed or merged. AB model SHA256 `a128228fd027986ed17fbc1802f02a9d1820f7dffd215aa46fa6211c4c04bb6e`. Original C24 `232cdd3477d7e80f61decf18e5b0354f12f05c253de1c23f234d06c9c2729d35`; selected C32 `d81f276cade69d40ef8b215ed9e38a225296061dfc74cf2d5b8d14615afdc7b1`. `final-manifest.json` records current source/model hashes. All1, 651 starting protected files are checked; the sole authorized change is the README append, whose original bytes are retained and verified as an unchanged prefix. Earlier scientific sources, frozen models, reports and outputs are byte-identical. Unrelated pre-existing `.gitignore`, Makefile and pyproject edits are preserved. No core redesign, remote writes, Atlas edits or merges occurred.

## Stop and pending Atlas delta

Stop here. The missing capability is a validated rule for predicting reliability under incident histories/preparations outside this envelope; neither a successful finite test nor a larger basis supplies it. No mediator reduction, higher-level compression or successor experiment is selected.

Pending Atlas delta, **recorded here only** for the existing Effective Motif Dynamics and persistence-to-composability records:

> Lab-local dual-reduction exploration implemented and executed locally on base c6e6c88f3ef75a4ce7acd660d6fa5779d995512c, with uncommitted execution snapshots and hashes. AB remained frozen; C24 was separately qualified on development. Initial frozen composition had 7 passing, 2 failing and 3 unresolved schedules on six fresh preparations; local C failures also appeared in FR. Composition-informed C-only enrichment to 32 modes, without new basis data or joint fitting, was requalified and passed 12 new schedules from six preparations across three seeds. Supported readouts are B source-region mass response, C mass response and the selected AB-feedback return contrast under the declared two/three-event schedules to 80/109 time units. Worst RR return trace/window NRMSE: 0.666%/1.529%; early sub-floor fidelity remains unresolved. Timed science cost: 32.080 CPU minutes; RR is about 1.66x slower than FF. Shared environment remains resolved. No claim of automatically discovered units, universal closure, or next-level compression. No successor selected unless explicitly stated by the owner.

**Strongest supported statement:** a conventional C-only spatial enrichment, while leaving AB's successful model frozen, supports their connected repeated response over the new tested preparations and schedules without interior refresh. The first independent freeze's failure remains part of that result.
