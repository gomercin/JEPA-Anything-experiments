# Intervention-aware retained state

Living record, 2026-09-26; base 36c25209d8d64bcdb9a10b7bc89cc2a8c387a13b.
Question: can a once-measured compact description take an actual additive A event,
retain its consequences and predict the conditional response of a later +.02 A
probe at C without any field access after initialization?

## Prospective contract

Preserve original E (`present_state_model.extract`, geometry), G (geometry frozen
quadratic-1e-06, affine control), and F (present-state frozen geometry model).
No old source or evidence changes. The earlier response pass and geometry
conjunction failure (A fractional-motion misses 41.8%/25.1%) remain unchanged.
No old pulse-response or spatial-hybrid substitute.

Same SH35/mediator laws, N=768, dt=.00625, length192, anchors[-28,-4,28].
Initialize at write-history t0=50. Pilot conditioning t1=60, a=+.02 and -.02;
alternative final +.02 probes at t2=70/90, h=0:.5:80. q is the original compact
laboratory-fixed unit-L2 source. No recentering, normalization or multiplicative
pulse. At t1 u+=a*q, m unchanged. E uses ell=x-anchor on windows not crossing
the periodic seam and uniform dx. For each window expand (u+a*q)^2 to test
z+=(M*z+2*a*I1+a²*Q1)/(M+2*a*I0+a²*Q0). All moments at t1 are evaluator-only.

Record true jumps, norm, energy/source increments, sampled regime, geometry and
conditional C mass/signed-moment responses. Y00/Y10/Y01/Y11 use identical time
steps/event boundaries. R_after=Y11-Y10, R_without=Y01-Y00, D=R_after-R_without.
Each reduced history initializes independently from its own z0. Ignore-event,
state-independent amplitude-polynomial kick, state-dependent local kick, and
F(exact current centers) are retained. Fit kicks to actual physical jumps only.
Zero amplitude is identity; immediate B/C kick is identically zero.

Primary targets: <=2% relative RMS of each R and <=10% of resolved D, separately
mass/moment, whole0:80 and late40:80. Keep absolute/max errors, signs/peak timing
and unresolved labels. Estimate numerical floors from new full four-branch
half-dt and double-N histories from the same prepared initial field: five times
the summed separate R refinement RMS per readout/window (no inherited 1e-12).
Minimum floor only float64 subtraction roundoff at measured absolute magnitude.
Physical coordinate tolerances to freeze after development resolution/sensitivity.
Keep old geometry scores as immutable prior results.

Development: exposed seeds8101,10101,10102,10103, initially none/odd04 histories;
all descendants group by initial preparation. Pilot8101 odd04. Saved old t50
states are read-only. No old seed is fresh. Small ordinary jump fits first;
at most two diagnosis-driven repair families, recorded before running them.
At least three fresh seeds16101/16102/16103, with one withheld intermediate
conditioning amplitude +.01 (same times/gaps), after model/schedule/metric freeze.
No fresh outcomes enter fitting or scaling. Predictions and retained geometry
must be written before post-t0 reference fields are generated. No true jump
reset, future response, sham, age, seed, write label or field solve in inference.
Checkpoint after the event, fresh-process solver/file-denied resume; segmented
advance and zero-event equivalence are independent of scientific accuracy.

1800 aggregate CPU seconds for scientific preparation/simulation/fitting/analysis,
including failures and 30 seconds startup/inspection allowance; development cap
1200, reserve600 fresh. One scientific process; pilot provisional per-section
120 CPU/180 wall seconds, 1GiB RSS; tighten/confirm from measured pilot. Exclusive
stage directories/outputs and source hashes, no overwrite/symlink following.
Publish substantive failures and evidence within25MiB Git (larger unique archive
only if needed), verify remote retrieval, critical review/fix, guarded merge.
Update only existing Atlas Effective Motif Dynamics and persistence homes.
No arbitrary-stream, absolute two-pulse trace, practical switch, minimality or
unique-storage claim; stop after this bounded task, without an automatic successor.

## Pilot and next discriminator

`pilot-01` costs26.347483 CPU seconds. Exact center jump identity agrees within
5.3e-17; B/C direct jumps are zero. +.02 shifts A by−1.24292e-4 but at gap10
its induced shift is+1.94780e-4; true-post-jump G still errs by−3.18022e-4.
The event excites a transient rather than merely setting a new unforced center.
Input norm .02 has source increments +.113485/−.112685 and energy increments
.000340383/.000323757, so opposite/equal-norm events do not mean equal work.
D/R is .021–.049% mass and .035–.044% moment. F at exact post-event centers has
D errors21.7%/17.6% at gap30 and281.5%/456.8% at gap10, even though R errors
are below .24%. This is a frozen-readout limitation, not a proof of missing
geometry information. Refinement must establish whether this small D resolves.

Keep the schedule/amplitudes unchanged. Complete four-preparation development,
fit the two specified J controls under unchanged G/F, and refine the entire
four-branch history at half dt/double N. Confirm pilot per-section limits120 CPU/
180 wall seconds/1GiB; measured single probes cost3.97 seconds and continuations
.75. Expected development plus numerical costs<350 CPU seconds. Before fresh,
allow two compact repair families only: (1) ordinary low-order linear transient
realization of event-induced physical center motion, with zero initialized
*incremental event state*, not assumed equilibrium; (2) low-rank response
correction driven by that evolving transient, identically zero before events.
Choose small orders by whole-preparation held-out free rollouts, preserve all
candidates, and charge every retained mode/coefficient. No absolute age input.

`development-01` failed during import before reference work (unmatched parenthesis);
its receipt is retained. Corrected before rerun in a new exclusive directory.
Transient family specification: discrete unit-time ERA orders4/6 fitted to all
0..30 event-induced center curves, train-only standardized [a,a*z,a²,a²*z]
excitation. Enforce the initial output as the fitted A-only jump, B/C exactly0.
The center law is now separately named/extended: advance G on z-Cr, r+=Ar,
then restore Cr to the physical centers. This has3+order evolving scientific
scalars plus the step counter, not three dimensions. No reference trajectory is
retained. Zero r means no *new conditioning event*, not an equilibrated old write.
Readout family: four-mode SVD correction, linear in retained transient, fitted
to development D minus the independently predicted F contrast. It is identically
zero when r=0 and keeps all original no-event predictions byte-equivalent.

`plain-fit-01`: all grouped R gates pass (worst .418%) but local kick+G/F fails
D badly; ignore predicts exactly zero D (100% relative when resolved).
`repair-fit-01`: order6 reduces worst physical A/B/C errors over0..30 to
[3.331e-5,1.398e-5,8.102e-7], versus order4[1.116e-4,3.584e-5,8.103e-7].
Its linear readout correction has worst D error161.5%, at a small2.34e-12
moment contrast. Retain it. Within the same second repair family, test one
state-dependent readout: transient coordinates times[1, standardized current
centers], with fixed ridge1e-4 and training-only scales. This tests ordinary
readout curvature/interactions rather than changing physical coordinates or
adding extraction. No new repair family, amplitude, timing or sensor search.
