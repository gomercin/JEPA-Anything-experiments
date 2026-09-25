# Source-to-receiver relay: living bounded investigation

Started 2026-09-25 from merged integration bf21ecbe9e940c0ab0c2adce2d994ff732bc50f5.
The earlier reports, AB20/C32 models and evidence remain unchanged. This new
input/output question does not reinterpret the C-probe selectivity negative.

## Before development

Question: with A=-28 and C=28 fixed, does moving B from 0 to -4 change C's
response to a fixed additive A pulse, and how much depends on induced mediator
feedback at B? Only these two supplied arrangements are considered. Nominal
neighbor gaps are 28/28 and 24/32; periodic alternate AC path is 136. Both
neighbor gaps must qualify on the actual three-pattern evolution. No binding
or native formation is presumed. No reduced model is loaded.

The ordinary screened mediator has frequency-domain spatial exponent
kappa(p)=sqrt((1+tau*p)/D). A point-relay approximation gives
exp(-kappa*dAB)*exp(-kappa*dBC)=exp(-kappa*56) for BOTH placements,
including delays in this homogeneous propagator. Reflection-related placements
would be a weak contrast too. Differences can arise from finite profiles,
B's susceptibility and evolving background, direct oscillatory u tails, and
loops. The far-field stationary SH roots satisfy (1+lambda^2)^2=r;
for r=-.67 the positive decay rate is sqrt((sqrt(1.67)-1)/2)=0.3823,
shorter than the mediator length 8, but not an excuse to delete direct coupling.
Diffusive impulse arrival over 56 has a kernel peak near 33 time units
(from t^2/tau+t/2-d^2*tau/(4D)=0). Start with horizon 80; no early below-floor
trace will be called screening. The trajectory-dependent ETDRK4 tangent, with
its own sham as privileged diagnostic information, is the stronger ordinary
calculation. It is not a deployable compact predictor.

Preparation: original coefficients, L=192, N=768, dt=.00625; independently
relax three seeded components for 150, reuse their exact fields across paired
arrangements, existing .02 smooth residual recipe, 40 units free joint
relaxation. Record actual masses, centers, widths, peaks, source, energy and
C mediator background through the sham and at every branch endpoint. Envelope:
three nonvanishing separated patterns, both measured center gaps >22, centers
within .5 of assembly anchors and drift <.5 through horizon 80. Violation is
reported as failed qualification, not silently excluded.

One source direction suffices for this gain/pathway question: radius-8 compact
bump*cos(x-A), unit spatial L2, additive u += epsilon*q at t=0, unchanged m.
Primary epsilon=.02, paired physical inputs; +/- and half sizes qualify
linearity, not independent channels. Read M and signed moment P of u^2 at
fixed A/C anchors and the B assembly anchor; mean mediator is diagnostic.
The input never touches B/C. Record actual field norm, source jump and Lyapunov
energy change; equal norm is not equal work. No rescaling after outcomes.

Selected control: suppress induced mediator feedback ONLY in one fixed smooth
union around B=0 and B=-4, core radius 8, taper to 12. Same mask for both
arrangements, support [-16,12], disjoint from source and C sensors. Four own-sham
ETDRK4-stage mediator fields replace m by (1-w)m+w*m_sham in g*m*u.
The tangent removes g*w*u0*delta_m stage by stage, retaining g*m0*delta_u,
induced u^2 source, mediator propagation and all direct u pathways. This is a
feedback cut, not a source cut or complete decomposition. For each organization
F=stim-full_sham, K=cut_stim-cut_sham, Q_B=F-K; compare F AND Q_B between
organizations. Monitor A and its exterior outgoing m at fixed A+12. Any A
change makes Q_B loop-wide pathway dependence, not a one-pass relay coefficient.
A translated radius-8/12 mask is one bounded geometry diagnostic if a candidate
appears; no mask will be selected for a larger effect.

Development seed 8101 only; fresh whole preparation seeds 9101/9102/9103.
First save tangent-only predictions for both placements, then nonlinear signed
and half-size validation. Tangent gate: 2% C-response RMS-normalized error or
absolute floor, declining on halving unless floor-limited. Do not pool M and P
without fixed scales. No practical capability threshold: report absolute changes
and fractions of total transmission; there is no external receiver task.

Resolution: same prepared fields at half dt and separately double N, Fourier
interpolating the latter. Per-readout Q floor = max(1e-11,5*max over arrangements
and refinements of [RMS(F_fine-F)+RMS(K_fine-K)]); organization-difference floor
uses the SUM over both arrangements of those branch errors before multiplying
by 5. Retain direct Q and delta-Q refinement too, without replacing the
conservative rule with favorable cancellation. Freeze these shared floors and
any common gain/delay fit (all samples, fixed development readout scales) before
fresh access. No continuum theorem or universal no-go inference.

If the candidate is resolved, predict withheld amplitude .03 from the saved
own-sham tangent BEFORE its nonlinear reference, on all three fresh seeds.
This tests one amplitude transfer dimension; .03 is never development input.
Prediction gates 2% for F and 10% for resolved Q, with absolute residuals.
No mixtures, threshold invention, broad placement sweep, fields or coefficient
changes. One mechanism-motivated alternative is allowed only if this contrast
fails the initial cheap calculation; exposed cases remain development.

Budget 1800 aggregate process CPU seconds, 600 reserved for fresh work. Pilot
cost first; 120 CPU/180 wall seconds per preparation or response; development
hard stop at cumulative 1200. Failures charged and preserved. Source commit and
hash manifest per run; unique non-overwriting directories; no historical archive
duplication. Stop after one explanation and qualified fresh check or a bounded
negligible/ordinary-account outcome. No automatic scientific successor.

## Development decision

The tangent-only pilot predicts C mass RMS 1.7293e-7 / 2.9442e-7 and selected
feedback contribution RMS 1.1222e-8 / 1.6992e-8 for B=0/-4. The Q_B change has
mass/moment RMS 1.0856e-8 / 7.5707e-9. Both triples retain measured gaps >23.9
and maximum drift <.014 over 80. Retain this contrast, with no alternative
placement or new field. Pilot cost 27.619 CPU seconds; projection 579.831.
Nonlinear signed .02 checks subsequently give 0.85-1.40% C tangent error;
halving gives 0.43-0.70%. This supports using the tangent for the smaller signal,
subject to refinement. At withheld .03 the 2% prediction gate may be marginal;
retain both the amplitude and gate and record any failure, rather than tuning.

## Completed result — responsive neighborhood dependence, with a prediction limit

**Moving only B changes forward transmission, including a resolved contribution
that requires induced mediator feedback in B's neighborhood.** The stronger
ordinary interacting tangent explains the primary small-input comparison.
The point-source two-leg attenuation cancellation is insufficient here.
This is prepared, finite-horizon, loop-wide pathway dependence, not a new channel
or a separately identified one-pass B-to-C coefficient.

Across all three fresh seeds, B=-4 raises C mass-response RMS by **1.701–1.711x**
and signed-moment RMS by **2.408–2.444x**, relative to B=0 at epsilon=.02.
Absolute C responses are small: mass RMS 1.701e-7–2.945e-7. The selected
feedback contribution has mass RMS 1.085e-8–1.699e-8 (5.77–6.42% of each total
response RMS). These are response magnitudes, not energy amplification or
independent causal fractions. The response is not described by a common gain
alone: the development-fitted gain/delay across mass and moment is
1.9636263 / -0.35 time units, but leaves fresh mass/moment errors of
19.33–19.58% / 21.80–23.18%. This is response reshaping for ONE source direction;
no input selectivity or new capability was tested.

| Fresh seed | RMS change in C mass F | RMS change in C mass Q_B | RMS change in C moment Q_B |
|---|---:|---:|---:|
| 9101 | 1.217712e-7 | 1.086614e-8 | 7.585339e-9 |
| 9102 | 1.251087e-7 | 1.096110e-8 | 7.711077e-9 |
| 9103 | 1.223106e-7 | 1.091984e-8 | 7.690779e-9 |

Both Q_B itself and its BETWEEN-organization change resolve against the frozen
1e-11 RMS C floors for mass/moment. The branchwise conservative sums were
computed before taking the floor minimum; no favorable contrast cancellation
replaced them. Half-dt and double-grid checks include both arrangements and
the smaller Q/delta-Q quantities. These are empirical numerical estimates on
one development preparation, not uniform continuum or statistical bounds.
Three independent seed preparations, each paired across organizations, are
three independent units; their multiple branches/readouts are related.

The translated-mask diagnostic changes delta-Q mass/moment RMS by 1.04%/3.54%
and individual Q traces by 1.96–6.10%, far smaller than the retained contrast.
Its differences are numerically real: the exact amount remains mask-dependent.
The primary common mask never moved. This local neighborhood includes tails;
it is not an ontologically exact boundary of B.

### Mechanism and attribution

The original `g*m*u` feedback is evaluated at each actual ETDRK4 stage. The cut
substitutes the OWN sham stage value of m only in its fixed neighborhood.
Full/cut shams agree to at most 1.59e-15 in fresh fields. Thus baseline B,
its source and static environmental conditioning remain; no B removal is used.
The induced source `2*s*u0*delta_u/tau`, mediator propagation and direct
pattern-field tails all remain. The event derivative is identity in state
and `(q,0)` in amplitude. The tangent differentiates all four actual ETDRK4
stages, including the cut, using the existing checked implementation unchanged.

At epsilon=.02 the tangent's fresh C error is 0.849–1.374% for F and
0.930–1.319% for Q_B. It predicts the smaller organization difference to
0.312–0.508% for F and 0.982–1.028% for Q_B. It requires the freely evolving
full sham; no autonomous/compact deployment claim follows. The finite profiles,
conditioned susceptibilities, direct oscillatory tails and loops are included
by ordinary equation-derived response theory. The experiment does not isolate
these remaining terms into exclusive fractions.

A's Q_B mass response is resolved too (3.687e-8–3.263e-7 RMS), despite being
only 0.00035–0.00310% of its much larger initial response. The fixed outgoing
mediator sensor at A+12 changes under the cut (Q RMS about 8.95e-8–3.78e-7;
F RMS about 5.18e-5–5.77e-5). Therefore this intervention changes the loop,
including later A sourcing. It cannot be interpreted as an isolated one-pass
relay gain. B's local mass response is 8.39e-6–4.40e-5 RMS; its selected
feedback contribution is 6.74e-6–1.54e-5. Responsive B-neighborhood dynamics do
causal work in the declared counterfactual, without explaining all transmission.

Most organization dependence survives the cut: C mass difference K is
1.142e-7–1.175e-7 RMS, compared with F 1.218e-7–1.251e-7. Static conditioning,
changed source dynamics and direct propagation remain available explanations
for this larger remainder; static conditioning alone was not identified.
Neither ordinary point propagation alone nor a fitted common gain/delay is
adequate, while the interacting tangent is accurate in its tested small-input
range. The Q gain/delay diagnostic reaches the -8 delay search boundary and
fits poorly; it does not exclude arbitrary delays or richer filters.

### Actual preparations and input accounting

All six fresh arrangements retain both neighbor gaps above 23.959 and every
recorded branch centroid within .02513 of its assembly anchor. Maximum drift
from the prepared state is .01392, well inside the .5 envelope. Centers for
all sampled branches are recovered exactly from the fixed mass/moment
readouts; widths are checked through shams and branch endpoints. No true future
center enters an actuator or sensor. No merger, disappearance or pinning occurs
in the tested observations; long-term binding is unearned.

Means at the prepared boundary, ordered A/B/C:

| Quantity | B=0 | B=-4 |
|---|---|---|
| Centers | -28.00402 / .00412 / 27.99732 | -27.97757 / -4.01333 / 27.98815 |
| Widths | 2.88898 / 2.88884 / 2.88895 | 2.90529 / 2.90720 / 2.89086 |
| Peaks | 1.32071 / 1.32127 / 1.32066 | 1.32218 / 1.32252 / 1.32047 |
| Weighted masses (local s=1 source integrals) | 8.00620 / 8.01562 / 8.00612 | 8.07366 / 8.08696 / 8.00984 |
| Whole-domain mediator source | 25.10416 | 25.33809 |
| C background mean mediator | .353505 | .346399 |
| Lyapunov energy | 1.903094 | 1.893857 |

Equal inventory did not produce equal physical states. At epsilon=.02 actual
field L2 is .02 to roundoff; source jumps are .112968–.112971 versus
.113461–.113465 and energy jumps .000332218–.000332231 versus
.000339874–.000339930. The roughly .44% source-jump difference and 2.3% work
difference are retained. No trajectory or readout was rescaled to remove them.
These small initial differences do not by themselves identify the full dynamic
cause of the roughly 70% C response increase. The matched cut establishes only
its declared pathway dependence, not a source-strength-normalized coefficient.

### Fresh consequence and adverse result

Before each fresh nonlinear reference, the own-sham tangent was saved and
predicted the withheld amplitude .03 by multiplication by 1.5. All six Q_B
predictions pass the frozen 10% gate (1.38–1.97% errors). But the total F
prediction passes the 2% gate only for B=-4: B=0 fails on all three seeds,
with mass/moment errors of 2.020–2.052%. B=-4 errors are 1.263–1.370%.
Largest F absolute RMS residual is 5.99e-9 (mass) / 1.83e-9 (moment);
Q_B residual is at most 3.61e-10 / 1.17e-10. These exceed numerical floors.

**The larger-amplitude total-response transfer is a partial failure**, consistent
with the development amplitude trend; it is not quietly rounded down to a pass.
The main .02 mechanism comparison passes. No threshold, source direction,
readout, mask or preparation was retuned, no additional fresh panel was consumed,
and no claimed nonlinear mechanism is inferred from this small truncation error.
The initial homogeneous point-relay simplification and the frozen gain/delay
fit are retained inadequate candidates. There was no outcome-bearing simulator
repair; the pre-freeze analysis was strengthened to check every sampled branch
centroid from its already saved readouts. Existing checkpoint safety fixes and
all previous scientific source/report/evidence bytes remain untouched.

### Scope, cost, reproduction and publication

Supplied: two fields and their laws/coefficients, three independently seeded
localized components, translations, preparation ages/residuals, finite domain,
input, fixed measurements and diagnostic intervention. Unearned: spontaneous
interface production, binding, new microscopic laws, exclusive mediation,
one-pass relay identification, input selectivity, neural advantage, practical
receiver utility, universal closure, automatic unit discovery or a new reduction.
The earlier AB20/C32 success is adapted composition; the C-probe selectivity
negative remains unchanged. Their reports were inspected, not rerun here.

Timed simulation/preparation/tangent/fitting/analysis used **357.039 CPU seconds
(5.951 minutes)**: pilot 27.619, development 47.953, refinement 84.019, mask
23.185, freeze .071, fresh 173.171, analysis 1.021. A conservative 15-second
allowance for brief untimed numerical inspections gives **372.039 seconds
(6.201/30 CPU minutes)** charged. Fresh reserved 600 seconds; largest task
23.999 seconds. Refinement and mask processes briefly overlapped; their full
process CPU times are both charged in the summed receipts. Fast regression,
editing and publication checks are separate. No infrastructure purchase or
dependency download occurred. No remaining budget must be spent.

Exact executed stage commands and revisions are in the
[portable evidence](../../evidence/source-receiver-relay-2026-09-25/README.md),
with original commands under its `data/work/mediated_patterns/source_receiver_relay/`.
Run from the repository root with the existing Python environment and
`OPENBLAS_NUM_THREADS=1`. `source_receiver_relay` stages were `pilot`, `develop`,
`refine`, `mask`, `fresh`; `relay_analysis` stages were `freeze`, `analyze`.
Every stage used a new directory. Execution revisions were `3d63b96` (pilot,
development), `3b9e2cf` (refinement), `74a001b` (mask), and `02760ba`
(freeze, fresh, analysis). Source hashes and configs are in each protocol.
All substantive new data fit in Git; no old archive is republished.

Software review and hosted status are recorded in the evidence index/receipt
and current README handoff. Scientific disposition: **resolved bounded pathway
and organization dependence; larger-amplitude F approximation partly fails**.
Stop here with no automatic scientific successor. A future one-pass attribution
or new interface-production question would require a distinct decision.
