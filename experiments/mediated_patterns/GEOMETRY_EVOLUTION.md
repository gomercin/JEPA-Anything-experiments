# Geometry evolution through an unobserved interval

Living experiment record, 2026-09-26. Base is merged `e2cba645`.
Question: can three measured fixed-window A/B/C center offsets advance themselves
through an unforced interval, sufficiently for the **unchanged** snapshot-response
model to predict a later .02 fixed-A probe and its history contrast? Does updating
earn anything beyond remembering the last measurements?

## Prospective access, comparisons and resources

The original extractor, field laws, preparation recipe, physical anchors and frozen
rank-4 quadratic response model F stay unchanged. Its authoritative artifact is
`evidence/present-state-transmission-2026-09-26/data/work/mediated_patterns/present_state_transmission/frozen-01/models.json`,
key `geometry`, file SHA256 `f0bc4ced6914bc43148d615912619370465ce48332a140dd41af52b7b85bb74e`.
The primary evolving state is exactly three current offsets; no field, twin,
seed, write label, absolute age, future center or response is an inference input.
A single shared autonomous vector field is integrated with fixed-step RK4.
The integrator's elapsed step count is bookkeeping, never a vector-field input.
Each history starts independently. Possible delayed probes branch independently
from their actual unprobed reference origins; they never affect later origins.

Begin with persistence, a shared mean drift, and affine laws with ridge 1e-6 or
1e-3. Preprocessing is training-only. Select on complete-preparation held-out
**free rollouts**, retaining direct coordinate loss; do not move coordinates to
compensate F's error. Quadratic or one compact additional current measurement
requires a recorded discrepancy/justification first. No neural fit or broad sweep.

All earlier seeds, including 12101/12102/12103, are exposed development evidence.
Inspect original saved unprobed center traces first. Check the center-rate identity
using the actual unwrapped local coordinates within fixed C2 windows, uniform
periodic-grid quadrature and the implemented Fourier field derivative. A short
step is a diagnostic, not a source of inference features. Strong correlations
between geometry, shape, mediator and age do not prove global closure.

Development samples natural unforced trajectories at 5-unit intervals from 50
through 100, and actual nonlinear responses at 60 and 75, reusing saved responses
at 50/100. These are grouped descendants, not independent preparations. Start
with existing none/odd histories plus already qualified mixed/negative histories.
Use t0=50 and shifted t0=60, with delays to 75/100. Reserve new preparations
14101/14102/14103 and an unobserved target time 90 (from both initialization
boundaries) for final evaluation. No reference sample at 90 enters development
selection. Choose/freeze the final history subset on development evidence.

Keep separate C mass/moment errors over all 161 h=0..80 samples and h=40..80.
R must meet <=2% relative RMS and resolved Delta_R <=10% in both windows, for
each preparation/history. Save absolute residuals, maxima and per-case scores.
Save predictions before opening future reference data. Compare persistence,
updated F, and evaluator-only F(exact later centers). Retain signed update and
readout residuals and their mean-square cross term; RMSE terms do not add.
Set physical coordinate tolerances before fresh access from development motion,
F sensitivity and matched refinement. Sub-floor motion/contrast is unresolved.

Refine a representative development history/time and the worst relevant final
contrast, from the same prepared initial state through write/wait/probe, using
half dt and separately double N. Response floors are shared per readout/window:
max(1e-12, five times the largest sum of both history response refinement RMS
errors). Coordinate uncertainty is five times the largest absolute matched
center discrepancy, with a 1e-10 roundoff minimum. These numerical estimates
are not confidence intervals. F is never retuned, even if its envelope ends.

1800 aggregate CPU seconds for science, including failures and a 30-second
inspection/startup allowance; development cap 1200, roughly 600 reserved fresh.
Pilot section caps: 120 CPU/180 wall seconds and 1 GiB RSS, checked within loops.
Only one scientific process at a time. Output stages and files are exclusive,
with source hashes and failed receipts. Editing, fast tests and publication are
separate. No old evidence duplication or mutation. New Git evidence <=25 MiB.

Freeze G, F identity, stepping, coefficients, tolerances, split and time panel
before fresh preparation. Check solver-free checkpoint/resume and segmented
updates. A useful outcome is bounded unforced propagation for delayed readout;
it is not updating through probes, repeated-input closure, minimality, physical
storage localization or universal dynamics. Publish/review/merge this scope,
then append a small result to the two existing Atlas homes. No automatic successor.

## Pilot and development decisions

`pilot-01` (e22078f) uses the four original saved histories and costs .066 CPU
seconds. Persistence from 50 to 100 has worst R/Delta_R errors 5.386%/29.742%,
so updating could matter. Maximum physical A/B/C drift is .005452/.008444/.003294.
The chain-rule rate agrees with one reference step to about 1e-9 units/time;
the finite-step difference has the expected small integration-interval bias.

Before any update fitting: acquire development geometry at 5-unit spacing but
**exclude interior times 80/85/90/95 from fitting, scaling and rollout selection**.
Use 50/55/60/65/70/75/100 only. The final target at 90 tests interpolation inside
that gap, with new preparations; it is not a new dynamical regime. The saved
interior development samples are evaluator-only and are not consulted in model
selection. F adequacy is checked at actual 60/75 states as well as saved 50/100.

`development-01` costs 192.597 CPU seconds: 21 natural history continuations
from seven exposed complete preparations; 50-unit unforced continuations cost
about 1.24 seconds each and independent two-branch probe evaluations about 3.95.
All sampled histories/probes qualify. The 120 CPU/180 wall/1 GiB section bounds
have ample pilot margin. Saved wait-100 checkpoint agreement is checked per case.

`fit-01` selects affine ridge 1e-6 by grouped free-rollout physical-coordinate RMS.
Maximum held-out A/B/C errors are .0001070/.0001559/.00001686. The standardized
fit has condition 1.635. Worst delayed R/Delta_R errors are .5003%/1.8732%; exact
current F gives .5115%/2.0295%. Persistence gives 8.828%/33.752%, shared constant
drift 6.205%/32.963%. The alternative affine ridge .001 is nearly tied. This is
not a need for quadratic dynamics or additional fields; no state extension is
attempted. The response scores use a previously fitted F, so this development
panel is not fresh validation of F. New intermediate snapshots pass its diagnostic.

Before refinement/fresh access, choose maximum physical coordinate tolerances
[A,B,C]=[2e-4,2e-4,5e-5], plus <=10% RMS error relative to each coordinate's
resolved actual movement over each preparation/history/origin's selected delays.
The maximum development relative F sensitivities per unit are [6.020,.366,.136]
for mass and [10.381,1.041,.236] for moment. The sum of sensitivity times these
coordinate tolerances is about .128%/.230%, leaving most of the 2% R allowance
for the unchanged F. This is a local development error budget, not a global
contrast guarantee; Delta_R is scored independently. Refine seed8101 mixed
(.28,.28) and its unwritten branch through times50/60/75 and probe at75, then
freeze. Final histories are none, odd+.4 and already qualified heldmix(.2,-.2)
for seeds14101/14102/14103. The history family itself is exposed; time90 inside
(75,100) is the withheld interpolation interval. No fresh-driven repair is planned.

**Development discrepancy and repair opportunity (before outcomes).** The
absolute coordinate gates are adequate for affine rollouts, but the additional
10% relative-motion gate is not: maximum A relative-motion RMS error is 46.8%
from50 and68.7% from60, in an almost stationary A trajectory. B/C are below
2.1%/.61%. Keep this affine limitation. Try one modest quadratic autonomous law
with the same two ridge choices and the same grouped rollout selection. It
keeps exactly three measured/evolving coordinates; it changes the approximation
class, from12 to30 rate coefficients (plus6 scaling numbers). Preserve both
baselines and affine, and retain all gates. This is a targeted curvature check,
not evidence of a missing physical variable or a reason to extend the field regime.
