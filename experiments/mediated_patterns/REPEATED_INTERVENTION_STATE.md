# Two conditioning events, then a conditional probe

**Result: unchanged nine-state reuse is not sufficient.** A composition-informed
extension with two squared-amplitude summaries passes the response gates at the
tested60/80/90 timing on three new preparations. It fails the withheld60/75/90
schedule, already in the second-only control. Thus the complete prospective
fresh contract **fails**: bounded approximate addition works at the sampled
ages, but interpolation of the individual response does not. No further repair
or successor is selected. The physical geometry limits remain separate.


Living record, opened before outcome access, 2026-09-26. Base main is
`6ee60082aada2e368d8d79960fe928f43d13426a`; no later continuation was present.
Question: does the previously selected nine-scalar retained state combine two
actual additive A events and predict the conditional response to a third +.02 A
probe, without another measurement or reset?

## Prospective contract

Use unchanged `intervention_model.State`, directly through a new scheduler.
Original selected `transient4-readout` identity:
`167c1229f39a2f29fbda0c502c7fc434e2f6d83e42b42ef62e66c7eae54cd76c`.
Original `frozen-01/models.json` SHA256:
`9524c674fb9247b97284fc2bd8fa8082c5dc953d989f1db3e44f68a101e9d56a`.
The artifact is in `evidence/intervention-aware-state-2026-09-26/data/work/mediated_patterns/intervention_aware_state/`.
All old E/G/F and nine-state code, coefficients, evidence and failures remain unchanged.

Measure three centers once at t=50, initialize four deformation and two response
coordinates to zero (no NEW conditioning since initialization, not equilibrated
fields). Keep one state through both events. Runtime has nine scientific scalars,
a step counter, 1,636 active coefficient/basis/scaling values, two known time and
amplitude slots and the final boundary. Labels, fields, rates, exact future centers,
shams and twin responses are evaluator-only. Unit-step G/transient advancement;
`response()` at 90 does not first apply the diagnostic as `event()`.

Development uses exposed seeds 16101–16103, none/odd04 preparations and saved
late states. Conditioning at 60,80 and probe at90: (+.01,+.01), (+.02,-.02),
(-.02,+.02). The second-only event stays at80. Three genuinely new seeds
18101–18103 are reserved. Fresh testing repeats these schedules and withholds
one timing schedule: (+.02,-.02) at60,75, probe90. No Cartesian timing sweep.
All descendants remain grouped by seed; no sequence-informed fit unless the
unchanged panel fails. At most two small diagnosis-driven repair families.

Eight logical branches per schedule share the initial field and event boundaries.
Prefix order is 00,10,01,11, each with probe-absent/present branches. R_ij subtracts
its own continuation. D12=R11-R00; D2|1=R11-R10; K12=R11-R10-R01+R00.
Compare ignore-both, last-only, independent single-event prediction addition and
the unchanged combined state. True single-event addition is evaluator-only.
Check the signed error decomposition, including cross terms for squared errors.

Keep mass and signed moment separate, whole h=0..80 and late40..80. R targets
are 2% relative RMS; resolved D1,D2,D12,D2|1 targets10%. K is reported separately,
without a nonlinearity gate. Unresolved signals are not relative-accuracy passes.
Record RMS and maximum absolute errors, magnitudes, signed peak and its time.
A secondary cancellation diagnostic uses a fixed development single-event RMS
scale and never replaces the original denominator.

Numerics: refine the full prepared/write/wait/eight-branch history of seed16101
odd04 for the cancellation schedule using dt/2 and 2N separately. Five times
RMS refinement differences of each matched probe-minus-sham R form per-prefix
uncertainties; propagate them by the triangle inequality through each actual
contrast. Take maxima over refinements/windows with a 64-epsilon absolute-C
arithmetic floor. Also publish absolute branchwise refinement and its more
pessimistic uncorrelated bound, and direct contrast refinement; do not substitute
the direct cancellation estimate into acceptance. These are matched-history
numerical estimates, not rigorous PDE error bounds. Freeze floors before fresh.
Keep physical absolute tolerances [5e-5,2e-5,2e-6], report induced motion separately.
Qualify separated-pattern mass/width/drift and source/work for every event;
equal input norm does not imply equal work. Never normalize trajectories.

Cheap checks precede simulation: identical additive profiles compose and opposite
zero-gap pulses cancel to roundoff; model accumulator cross terms are inherited
predictions, not evidence of physical non-additivity. Model zero-gap discrepancies
are diagnostics outside the finite-spacing acceptance envelope.

Save all independently initialized prefix forecasts and hashes before any future
reference arrays. Checkpoints two steps after both events must resume in a fresh
solver-free process from coefficients, retained state and schedule only. Check
segmentation, zero events, causal timing and retention of auxiliary coordinates.

Science budget: 1,800 aggregate CPU seconds including failed attempts and a
30-second inspection/startup allowance; development stops at1,200. Pilot first;
per section120CPU/180wall seconds, RSS1GiB. Fresh reserve roughly600CPU seconds.
Editing, fast software checks and publication are separate. No larger amplitudes,
new world, architecture search, geometry repair campaign or automatic successor.

Prior boundaries persist: one-event fresh336 distinct gates passed, grouped
development contrast14.123% missed; physical B/C induced motion31.2%/100% missed;
unforced A fractional motion41.8%/25.1% missed. This study does not establish a
full absolute trace, unique storage, minimal state, arbitrary streams or general
recursive composability.

## Development observations and decisions

The pilot used seed16101/odd04 and all three sign schedules. Its unchanged-state
worst R11 error was0.288%, D12 9.732%, D2|1 7.493%; independent addition was
0.284%,4.990%,4.663%. Ignore-both passed the large R while missing the event
contrasts100%. The model's K12 relative error was about3.6; this is separate from
the main gates. No repair was justified by the pilot's main gates.

Full-history half-dt/double-N qualification used the opposite-sign pilot. Matched
D12 floors are[2.737e-13,3.881e-13] whole and[3.472e-13,5.074e-13] late;
D2|1 floors[2.644e-13,4.160e-13] and[3.415e-13,5.444e-13]. K12 floors
[5.229e-13,8.119e-13] and[6.550e-13,1.064e-12]. These do not come from old
floors or the smaller direct contrast differences. Common absolute branch
quadrature shifts are much larger: the uncorrelated eight-Y K bound reaches
[1.352e-5,7.875e-7] whole. Numerical resolution here is conditional on the
matched probe/sham instrument; it is not a rigorous independent-branch PDE bound.
Both bounds and every underlying branch are published. Opposite-sign mass K12
resolves in the pilot; moment and same-sign K12 are below these matched floors.

The additive field zero-gap identities hold within2.23e-16. The fitted event
maps do not satisfy exact zero-gap closure: the cancelling pair leaves up to
1.78e-4 in a fitted deformation coordinate and about2.74e-12 response RMS.
The p,d accumulator cancels exactly. This is an approximation limitation outside
the declared finite event gaps, not a repaired input semantics or a new gate.

Pilot cost33.16CPU seconds; numerical qualification50.95. The isolated-runtime
software fixture initially omitted the static measurement utility imported by
F; adding that pure NumPy module fixed the fixture. No solver or field access
was added. The seven instrument tests then passed. Full development retains all
six exposed seed/history starts and does not fit any new coefficients.

Full exposed development reveals five unchanged response-contrast gate misses:
worst D12 11.237%, D2|1 13.741%, while isolated D1/D2 stay below1.874%/4.804%.
Independent addition stays below7.128%/5.707%. This points to the excessive
combined quadratic readout contribution, not an event-timing defect or a need to
alter physical centers. One repair family is declared before fitting: multiply
both existing quadratic readout coefficient blocks by one scalar alpha. Fit
alpha by ordinary least squares to resolved whole-window mass K12 on training
seeds, and evaluate each held preparation including every descendant. No new
basis, variable, event semantics, G, J, or physical trajectory. Compare alpha0
(linear readout) and alpha1 (frozen original) as endpoints, retain all failures,
and replay the saved one-event panel to detect loss. This is explicitly
composition-informed even if the nine-dimensional runtime remains unchanged.

The first repair is retained as **failed**: alpha=-0.174855 (held-seed fits
-0.158260,-0.186470,-0.180527), four selected and four grouped main-gate misses,
and one old one-event miss (10.512%). Opposite-sign p=0 makes the combined
quadratic features exactly zero regardless of alpha; changing their coefficients
cannot restore that lost individual-event contribution. CLI dispatch initially
rejected the new stage before any fit; the parser was fixed and that startup
attempt is covered by the30CPU-second allowance.

Second/final repair family, declared before evaluation: retain q+=b² and e+=b²
at each event, e decaying by exp(-1/20) each unit step, with q constant. Use
[p,d,q,e] in the original readout coefficients instead of[p,d,p²,p*d]. This is
ordinary diagonal event addition within a fixed-size state, not physical energy.
Both summaries initialize to zero, require no additional acquisition, and do
not change geometry, G, J or the four deformation coordinates. It adds two
scientific scalars (eleven total) but fits no new coefficient. For one isolated
event q=p² and e=p*d analytically; verify the saved panel numerically. This is
composition-informed and no longer unchanged nine-state reuse. No third repair.

The second family passes all development main gates: R11 .4055%, D12 6.8338%,
D2|1 5.3469%, with no saved one-event gate lost and exactly zero maximum change
to those old one-event forecasts. Select this eleven-scalar separate-square
candidate for fresh access. Its K12 is still inaccurate (worst raw relative1.48),
so no preservation of nonlinear combination is claimed. This construction is
useful approximate addition; the original and independent-addition comparators
remain alongside it. No third family, further fitting, or fresh-triggered repair
is authorized by this record. New seeds and the single withheld timing remain
18101–18103 and60/75/90(+.02,-.02). Freeze hashes, preprocessing, schedules,
floors, scores and both original/selected models before their preparation.

## Fresh result: restricted composition success, timing failure

Selected model identity is
`3e799415e73b6d34e1c3bba54eec66ee4b29bade544114d1cf7cbb96c61572f5`;
freeze SHA256 is
`a165613f51f66bc6a96fa9a512fe730264650280bfe61b18c3fb2bd5ac043438`.
The freeze and fresh protocol use source revision`d440951`. Six supplied starts
(none/odd04 for each of18101–18103) generate18 standard and6 withheld-timing
schedule cases. These are **three independent preparation groups**, not24.
There are192 logical full-field branches, with168 computed branches after safe
sham/single-prefix reuse. The final +.02 diagnostic always follows both slots;
its full80-unit conditional response is forecast without advancing the reduced
state through that final probe.

Worst relative RMS errors below take the maximum over whole/late windows and
all corresponding preparations. Each entry is **mass / signed moment**.

| Predictor, 60/80/90 | R11 | D12 | D2 given1 |
|---|---:|---:|---:|
| Original nine-state | .334% / .404% | 11.155% / 12.453% | 7.455% / 12.828% |
| Independent single-event addition | .331% / .400% | 7.919% / 7.667% | 5.193% / 4.204% |
| Selected eleven-state | .332% / .402% | 7.498% / 7.045% | 4.806% / 4.117% |

The selected candidate passes all528 distinct main scalar gates at standard
timing (576 entries include repeated shams). Original reuse misses six standard
gates. Ignoring both events predicts zero for every event contrast and misses
those resolved signals100%, despite passing every large-R gate (worst R11 .434%
including timing). Keeping only the last event also loses the earlier effect.
The three-path ordinary addition is a useful comparator, and the selected state
mostly implements the same approximate account more compactly.

| Withheld60/75/90 | R11 | D12 | D2 given1 | isolated D2 |
|---|---:|---:|---:|---:|
| Original nine-state | .350% / .429% | 239.536% / 384.349% | 130.020% / 108.450% | 122.322% / 100.444% |
| Independent addition | .348% / .425% | 221.501% / 350.262% | 122.222% / 100.388% | 122.322% / 100.444% |
| Selected eleven-state | .349% / .427% | 229.131% / 365.072% | 124.766% / 103.128% | 122.322% / 100.444% |

All24 timing entries for each of D2,D12,D2|1 fail for the selected model:
**72 failures across648 distinct fresh main gates** overall (768 entries).
The same second-only predictions fail at their original75 boundary. The first
control stays within1.447% mass/1.176% moment. This identifies a single-event
response-age interpolation limitation in the retained readout; it does not
uniquely identify an accumulation failure, missing physical storage, or a
necessity for more state. The new15-unit gap lies between individually sampled
10/30, but interpolation was not previously qualified. The larger response still
passes, so total-R accuracy alone would have hidden this failure.

For example, seed18102/odd04 timing D2 mass has true signed peak+1.642e-10 at
h80, while the prediction's largest peak is-1.659e-10 at h36.5. Selected D2|1
RMS error is8.619e-11 against a6.908e-11 signal. The worst selected D12 moment
relative error occurs at18101/none, late window: error4.480e-11 versus signal
1.227e-11. Cancellation amplifies that relative error, but the fixed development
single-event normalization still gives23.58%; it is not a harmless zero divided
by zero. Frozen normalizations and thresholds are unchanged.

### Absolute scales and non-additivity

At standard timing R11 RMS is2.84–3.36e-7 mass and9.02e-8–1.34e-7 moment.
Selected maximum RMS errors are1.116e-9/5.398e-10; maximum pointwise errors
1.748e-9/8.720e-10. The smaller D12 RMS spans7.19e-11–3.70e-10 mass and
3.04e-11–2.05e-10 moment; its maximum RMS errors are1.129e-11/4.210e-12.
D2|1 maximum RMS errors are1.587e-11/6.208e-12. At withheld timing the selected
D12 and D2|1 errors reach about1.04e-10 mass/4.67e-11 moment, well above the
matched numerical floors. Complete maxima, signed peaks, times, fixed-scale
residuals and every window are in the indexed score records.

K12 is much smaller: standard mass RMS1.91e-13–3.07e-12, moment9.78e-14–1.63e-12.
Only34/72 standard K entries resolve against the frozen floors;16/24 timing
entries resolve. The selected candidate's **resolved** K errors reach98.7%
mass/56.9% moment at standard timing and160.6%/90.4% at withheld timing.
Neither unchanged reuse, exact additivity (K=0), nor the repair earns accurate
nonlinear combination. Unresolved entries carry no relative-accuracy pass.

Evaluator-only true single-event addition incurs at most2.04%/3.11% D12 error
at standard timing and1.08%/1.62% D2|1 error. At timing75 its D12 moment error
can reach13.34% because cancellation exposes even this small K; D2|1 remains
within1.12%/1.08%. The much larger deployable timing misses mainly accompany the
single-event response error. The signed three-term identity holds to2.07e-25;
its saved Gram matrices are sums of temporal products, not an invalid sum of RMS
errors. No evaluator-only additive answer enters a primary prediction.

The adverse timing history18102/odd04 was additionally refined from its original
prepared/write boundary at half dt and double N. Propagated matched D2,D12,D2|1
estimates remain below2.34e-13 mass/4.67e-13 moment; direct contrast changes are
also retained. This confirms the large miss under the matched instrument. It
does not replace the prospectively frozen floors or recertify any fresh failure.

### Physical regime and geometry are separate

All reference branches remain separated: minimum center gap23.9708; windowed
mass7.9634–8.1926; width2.8909–2.9100. Event formula residual is at most2.35e-16,
with exactly zero immediate B/C jumps. Maximum immediate A jump is2.603e-4.
Full-field norm at event boundaries is5.7313–5.7432; source-integral jumps range
-.112685 to+.113481 and energy-functional increments8.396e-5–3.404e-4. Individual
input norms are.01/.02; total absolute input is.02 or.04. Equal norm is not equal
source increment/work, and trajectories were not normalized. Full per-event
moments, norms, actual jumps, and every regime screen are retained.

The two selected extra coordinates change **only the readout**. Both models have
the same true/predicted physical-center comparison. New-prefix maximum absolute
A/B/C errors are[6.960e-5,4.487e-5,2.539e-6], exceeding the unchanged
[5e-5,2e-5,2e-6] tolerances. Worst errors relative to peak event-induced motion
are15.5%/46.4%/100.0%; the tiny C motion must also be read with its resolution
limits. New response success does not certify geometry. Earlier unforced A
41.8%/25.1% fractional misses, one-event B/C motion limitations, and the prior
grouped readout miss remain unchanged and visible.

### Retained state, access and cost

Selected state is **eleven scientific scalars plus one integer counter**: three
centers, four deformation coordinates, p,d,q,e. q and e accumulate b², with e
using the same supplied20-unit decay. Initial acquisition is still one current
(2,768) field snapshot passed to E, retaining only three measured centers and
eight declared zero auxiliaries. No new field-derived initial quantity and no
runtime field/rate/twin access is added. q/e are response summaries, not uniquely
identified physical memory or energy. Across the fresh prefixes p,d stay within
[-1,1], q reaches2 and e1.47237. Deformation-coordinate extrema are in
`state-ranges.json`; they are not clipped to old single-event ranges.

Static active storage remains1,636 coefficient/basis/scaling values: G36, J14,
transient60, duplicated excitation-J14, F732 and readout780. Two4×161 temporal
bases are included. The original serialized diagnostic metadata also remains:
1,918 numeric values, plus the new extension-name string; it is not an eleven-
number complete program. Fixed anchors3, output times161, amplitude scale.02,
unit step and decay20 are explicit. The driver has two(time,amplitude) event slots
and a final boundary; each response produces322 values. Stored development
trajectories and plots are evaluator evidence, not retained inference history.

Each step keeps the original four10×3 G evaluations, two3×4 products, one4×4
product and d decay, adding one e decay. Each event keeps the original two8-term
feature maps, A-only kick and4×8 excitation, adds b² and two accumulations.
Readout retains10×8 and16×8 coefficient products and rank4 reconstructions,
substituting q,e for the two products. No field reconstruction/PDE is hidden.
For R11 the selected path advances40 unit steps; independent addition uses three
nine-state paths (120 steps, three readouts,27 scientific floats/three counters
if concurrent). Full prefix diagnostics use four independent paths:44 selected
or36 original floats/four counters if concurrent, plus four322-value outputs.
Sequential evaluation can reuse the runtime storage; it does not remove those
extra inference operations or output buffers.

All384 checkpoints—both models, all prefixes, after both events—resume exactly
in a fresh process containing only copied inference modules, coefficients,
retained state and schedules. Solver imports and npz/npy reads are blocked.
Segmented advancement, zero-event identity, accumulator persistence and causal
independence from future events are tested. This is numerical/serialization
self-containment, separate from physical accuracy.

Science used720.44CPU seconds (**12.01CPU minutes**), including failed fits,
refinements, audits, summaries and a30-second startup/inspection allowance.
Fresh preparation/reference cost340.67CPU seconds; largest section16.53CPU/
16.94wall seconds, below120/180 limits and1GiB RSS guard. The aggregate cap was
1,800CPU seconds, with development capped at1,200. Software checks/publication
are separate. Remaining budget does not authorize a third repair family.

[Portable evidence and exact commands](../../evidence/repeated-intervention-state-2026-09-26/README.md)
include all failures, seals, models, eight-branch arrays and numerical checks.
[True/predicted standard geometry](../../evidence/repeated-intervention-state-2026-09-26/figures/geometry-cancel.png),
[standard R11/D12/D2|1/K12](../../evidence/repeated-intervention-state-2026-09-26/figures/response-cancel.png),
and [withheld timing failure](../../evidence/repeated-intervention-state-2026-09-26/figures/response-timing.png)
show geometry, common response and small contrasts separately. No full absolute
trace, practical switch, minimal state, arbitrary input stream or recursive
composability follows from this restricted result.
