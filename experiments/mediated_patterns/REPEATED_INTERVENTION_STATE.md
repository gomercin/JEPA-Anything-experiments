# Two conditioning events, then a conditional probe

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
