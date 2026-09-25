# Organization-conditioned response

## Prospective contract — 2026-09-25, before new field runs

Question: with fixed microscopic laws and three supplied persistent patches,
does moving A change C's signed response operator beyond one common gain/delay?
The completed AB20/C32 result and all historical failures remain frozen.
This investigation starts at `1c555d5aab826418c81e0a9005a85200e3f8fd65` on
`codex/organization-conditioned-response`; no remote write or Atlas edit.

- Organizations: A=-10 or -18, B=14, C=39 on periodic L=192. AB gaps 24/32;
  BC=25; AC=49/57 (opposite periodic paths 143/135). These are physical
  preparations, with the existing three-patch recipe, not relabelings. Each
  seed's isolated components are reused across its two arrangements; independent
  150-unit relaxation, seeded 1% variation, existing .02 assembly residuals,
  then 40 units of unconstrained joint relaxation. No pins or recentering.
- CPU float64 full-field reference, N=768, dt=.00625, original r=-.67, b=2,
  g=.05, s=1, D=64, tau=10. Horizon 80, samples every .5 including t=0.
  Preparation drift is measured; use the actual sham trajectory for the tangent
  in all cases, without assuming a stationary background or autonomous reduction.
- NEW additive diagnostic events: `u+ = u- + epsilon*q_j`, `m+ = m-`,
  epsilon=.02, at t=0 only. Fixed C-centered compact radius-8 bump times
  cos(x-C) (amplitude direction) or sin(x-C) (displacement direction), each
  normalized to unit spatial L2. Support, phases and normalization never use
  evolving or prepared fields. Equal L2 is not equal injected Lyapunov energy;
  save each actual norm, peak, source increment and energy jump.
- Fixed smooth C readouts: existing mass M=integral(w*u^2) and one complementary
  signed moment P=integral(w*(x-C)/8*u^2). Also save existing local mediator mean
  and component descriptors for diagnosis. Mass alone is symmetry-degenerate for
  an odd probe about a symmetric C. That analytic failed design is retained here;
  P resolves odd response without selecting a sensor after outcomes. Positive
  and negative versions are linearization checks, not independent probes.
- Derive and implement the tangent of the actual four-stage ETDRK4 map, including
  source `2*s*u0*delta_u/tau` and feedback `g*m0*delta_u+g*u0*delta_m`.
  Check against positive/negative .02 and .01 full-field responses; the tangent
  must predict each signed trace within 2% weighted NRMSE (or absolute floor),
  with reduced small-signal error on halving unless already numerically limited.
  For the historical multiplicative event, `D_state E=diag(1+a*b,1)` and
  `D_a E=(b*u-,0)`; it is not the additive event used here.
- Cut: identical fixed union of radius-8/core, radius-12/taper masks centered at
  -18,-10,14 for BOTH organizations. Replay own sham mediator only in `g*m*u`.
  Thus remove `g*w_AB*u0*delta_m` from the tangent; retain background feedback,
  perturbed pattern sourcing, mediator transport and direct SH coupling.
  Four matched branches give F=stim-sham, K=cut_stim-cut_sham, Q=F-K. The
  intervention is externally enforced and does not isolate every AB pathway.
- Comparator, fitted separately for F, K, Q: A=-18 traces = g times shifted
  A=-10 traces, with ONE signed gain and common delay across both probes and
  both readouts. Delay grid [-8,8] in .025 units, linear interpolation, zero
  before the event, constant endpoint extension after 80. Evaluate ALL original
  samples; never crop mismatches. Also retain gain-only and identity comparisons.
  Geometry changes diffusive attenuation and SH tails, so beating this fit does
  not beat conventional propagation. The equation-derived time-varying tangent
  is the stronger ordinary interacting explanation; no per-case fitted filter.
- Calibration: one numerical quartet per organization/probe, dt/2 and N doubled
  separately, from the same (Fourier-interpolated for mesh) prepared state.
  Per-readout absolute floor=max(1e-10, 5 times the largest conservative RMS sum
  of separate full/cut response refinement errors). Report direct Q refinement
  too. Floors are empirical, not rigorous continuum bounds. The larger floor
  across all calibration cases is shared by organizations and preparations.
- Development weights: each readout divided by its pooled development RMS over
  organizations/probes/times, bounded by its floor; fixed separately for F/K/Q.
  Resolved signal requires RMS>floor. A practical comparator failure requires
  weighted NRMSE>5% AND some per-readout residual RMS>5*floor, reproducibly across
  all three fresh seeds. Report smaller resolved differences without upgrading
  them to practical selectivity. No mandatory reversal or new reachability claim.
- Exposure: development seeds 6101/6102; entire preparations withheld at seeds
  7101/7102/7103. Two organizations per seed are paired, probes are related.
  Freeze choices, numerical floors, weights, fitted gain/delay and source hashes
  before fresh access. Fresh uses the same two probes plus the single withheld
  mixture `(q_even+q_odd)/sqrt(2)` if selectivity is a candidate. The ordinary
  tangent predicts this mixture before its full-field trajectory is run; this
  is a prospective trajectory prediction from linearity, not a new identity.
  Prediction gate: 2% weighted error for total response; 10% for resolved Q,
  with absolute error/floor reported. If gain/delay suffices, fresh-check the
  two probes and stop; no mixture or successor required.
- Budget: hard 1,800 aggregate process CPU seconds for new simulation/calculation/
  fitting, including failures; reserve at least 600 for untouched evaluation.
  Time a pilot first. Per preparation/response task stop: 120 CPU seconds or
  180 wall seconds. No paid compute, new training, field family, or broad sweep.
  One observability revision (above) and at most one diagnosis-driven follow-up
  are allowed; never change thresholds after outcomes. All stages write new
  directories under `work/mediated_patterns/organization_response/`.

Primary-source use is confined to Raja et al., [Eq. (1) and the pattern-dependent
linearization in Eq. (7)](https://arxiv.org/html/2303.00798v1). Their base operator
supplies `r-(1+partial_xx)^2+3*b*u0^2-5*u0^4`; the mediator blocks follow from our
code and are not validated by that donor paper. No eigenspectrum is needed.

Atlas continuity: current roadmap delegates local work to labs. Existing
Effective Motif Dynamics and persistence-to-composability nodes distinguish
mediated response, a useful interface and production of that interface. This
experiment supplies organizations and local laws; it cannot establish native
formation, binding, physical mode uniqueness, general closure, new global
reachability, or Preserver's separate organization-fluid project.

## Result — closed after the fresh check

**No practical selectivity at the prospectively declared 5% criterion.** Moving
A does produce small, numerically resolved changes in C's response shape. A
single gain accounts for the two-probe/two-readout operator to 0.084–0.107%
weighted error on three fresh seeds; the best common delay is zero. The selected
AB-feedback return is resolved, but its change between organizations and the
residual of its gain fit are below the conservative numerical floor. The total
response's small shape differences survive that feedback cut almost unchanged.

Thus **exact scalar-gain equivalence is not established**: total-response fit
residuals are above numerical error. The useful negative is bounded by the
declared practical criterion. The equation-derived, trajectory-dependent
linearized interacting model explains the tested response and its small
organization dependence. No new response kind, reachability, or special
nonlinear mechanism is established. No refinement, new family, mixture panel,
or automatic successor was selected after the negative fresh check.

### What physically changed

Only A's assembly position changed. All three independently prepared component
fields, B/C assembly positions, random residual draws, coefficients, domain,
preparation age, actuator and sensors were paired across the organizations.
The mediator attenuation scale remains 8, while the AB/AC distances change by
8; both the diffusive path and oscillatory SH tails are ordinary candidate
mechanisms. BC's nominal distance is fixed, but the subsequent freely evolved
B/C positions and environments are allowed to differ.

Means over the three **fresh** seeds at the end of joint preparation:

| Measured quantity | AB gap 24 | AB gap 32 |
|---|---:|---:|
| A/B/C centers | -9.96648 / 14.02019 / 38.95672 | -17.98342 / 14.03742 / 38.95734 |
| A/B/C widths | 2.90527 / 2.92148 / 2.90501 | 2.89082 / 2.90656 / 2.90469 |
| A/B/C peaks | 1.32198 / 1.32359 / 1.32169 | 1.32012 / 1.32228 / 1.32165 |
| A/B/C weighted masses | 8.07443 / 8.15069 / 8.07438 | 8.00967 / 8.08578 / 8.07261 |
| C smooth mean mediator | 0.367467 | 0.366298 |
| Total mediator source, `s*integral(u²)` | 25.53869 | 25.34573 |
| Coupled Lyapunov energy | 1.885684 | 1.895787 |

The farther organization supplies about 0.756% less total source, and C's
mediator background is about 0.318% lower. No source or state normalization was
applied. Equal component count is not equal source or environmental conditioning.
All observed sham trajectories and stimulated endpoints retain three separated
patches; no candidate disappearance or merger was encountered. Maximum fresh
sham centroid drift is 0.04066; stimulated endpoint drift is 0.04073. Maximum
sham/endpoint width changes are 0.001335/0.001334. These small finite-horizon
drifts neither establish binding nor justify an exactly stationary linearization.

### Inputs, observability and the equation-derived explanation

The even carrier probes an amplitude direction; the odd carrier probes a
displacement direction, without translating or recentering the actual pattern.
Both have the same compact support and instantaneous envelope, and actual L2
jumps equal .02 to float64 roundoff. Their maximum field increments are
0.0101188 and 0.00966597. They are independent channels; signs and half sizes
were used only to test linearity. Mass is even and the signed moment is odd
about the fixed C anchor, so the complementary moment avoids an analytically
known mass-only blind direction. No outcome-based sensor change was made.

Across fresh preparations, even-probe energy injection is 3.3944e-4–3.4023e-4;
odd-probe injection is 4.9212e-5–4.9485e-5. Equal L2 therefore does **not** mean
equal work across probes. The source jumps also differ: even 0.113356–0.113423;
odd -0.002993 to +0.000195. The latter varies with the prepared phase relative
to the fixed actuator. Within each paired organization comparison the physical
additive input is identical; changed work or readout action comes from the
changed prepared state. All individual values are retained, including the odd
source jump's sign change. This is not a sign reversal of response preference.

For `u0(t),m0(t)` on the freely evolving own-sham trajectory, the implemented
linearization is

```text
delta_u_dot = [r-(1+partial_xx)^2+3*b*u0^2-5*u0^4+g*m0] delta_u
              + g*u0*delta_m
delta_m_dot = [(D*partial_xx-1)/tau] delta_m + (2*s*u0/tau) delta_u
delta_(M,P) = 2*integral((w,w*(x-C)/8)*u0*delta_u)
```

The additive event has state derivative identity and input derivative `(q_j,0)`.
The historical relative event instead has the two derivatives stated in the
prospective contract. The new code differentiates every ETDRK4 stage, rather
than inserting a continuous-time Jacobian into a different integrator. A finite
difference of the actual one-step map passes a focused test. The unperturbed
trajectory is an explicit diagnostic input; this is not an autonomous reduced
model, and the frozen AB20/C32 models were never loaded or retrained.

At epsilon=.02, worst development full-response tangent error is 1.0471%
over signed probes; at .01 it is 0.5202%. The odd-probe maximum falls from
0.3495% to 0.1747%. Positive and negative cases show the expected approximately
first-order relative error decrease on halving. All pass the unchanged 2%
small-signal criterion. Fresh paired-probe tangent errors are at most 0.7622%
for total response and 0.5280% for resolved return. A descriptive additional
check of the **difference between organizations** gives total-response tangent
errors of 0.850–1.038% of that smaller difference; this was not a new fitted
model or a new acceptance gate. Return-difference relative scores are retained
but not interpreted because that target is below the floor.

### Frozen comparator and fresh prediction

The fit uses development seeds 6101/6102, then freezes before any 7101/7102/7103
outcome. There are six fresh preparations paired by three seeds, with two
related probes per preparation: **12 positive-input full-field responses**,
plus their matched control branches, not twelve independent preparations.
Every score includes all 161 samples from 0 to 80. No early samples were cropped.

Frozen gap-24 → gap-32 parameters are:

| Response operator | Common signed gain | Common delay |
|---|---:|---:|
| Full | 1.000678686 | 0 |
| Feedback cut | 1.000679344 | 0 |
| Selected return | 0.985183438 | 0 |

Allowing delay did not improve the gain-only comparator. No richer fitted filter
was introduced; conventional spatial propagation remains inside the explicitly
interacting tangent model.

| Fresh seed | Full fit error | Cut fit error | Return fit error | Practical selectivity criterion |
|---|---:|---:|---:|---|
| 7101 | 0.10005% | 0.10014% | 0.75455% | not met |
| 7102 | 0.10657% | 0.10665% | 0.82348% | not met |
| 7103 | 0.08379% | 0.08390% | 0.57809% | not met |

These are weighted RMS errors with the development-frozen per-readout scales;
the denominator is signed response RMS, not the large background field.
Unfitted identity errors were already only 0.105–0.133% for full response and
1.552–1.678% for return. The numerical floor is shared across preparations:
5.66087e-8 for mass and 5.23294e-8 for moment.

Full-fit residual RMS is 6.56e-6–7.36e-6 in mass and 2.79e-6–4.01e-6 in moment:
these weak departures from uniform gain **are resolved**. Return-fit residual
RMS is at most 4.94e-9, below the floor. Its sub-percent score is a descriptive
discrete-reference number, not independently certified continuum fidelity at
that precision. The 5% criterion was fixed before outcomes and never relaxed.

The prospective negative check predicts the new gap-32 response from the paired
gap-24 response using those fixed parameters. Its reference trace is an allowed
comparison input, not a compact deployment interface. A withheld probe mixture
was conditional on a practical selectivity candidate; the condition was false,
so no mixture was run or claimed. The same probe family across fresh whole
preparations is the scope of this result.

![Total C response, fixed gain and fixed apparatus](../../work/mediated_patterns/organization_response/analysis-01/full.png)

### Causal contrast and its limit

The full and cut shams agree within 1.31e-15 in the fresh physical fields. The
fixed spatial intervention removes induced mediator-to-pattern feedback in the
declared AB neighborhood, leaving `g*m0*delta_u`, perturbed source, transport,
direct SH coupling and C's local laws intact. It is not a pattern clamp and not
a natural alternative preparation. The exact subtraction is
`Q=(stim-sham)-(cut_stim-cut_sham)`; t=0 return is zero.

Q itself is resolved and has different time shapes for the two probes. However,
the **change in Q between organizations** has RMS only 1.48e-8–1.59e-8 in mass
and 6.02e-9–6.85e-9 in moment, below the conservative floor. Those numbers are
also exactly the change introduced by this cut into the organization contrast.
The full organization contrast is much larger: 4.68e-6–5.74e-6 in mass and
4.76e-6–6.04e-6 in moment. It survives this intervention essentially unchanged.

![Selected AB-feedback contribution, separately scaled](../../work/mediated_patterns/organization_response/analysis-01/return.png)

![Organization differences and the ordinary tangent prediction](../../work/mediated_patterns/organization_response/review-01/organization_difference.png)

The measured C background/source changes and the surviving cut response support
static preconditioning as an available conventional explanation. They **do not
establish that static conditioning alone explains everything**: direct pattern
influence and its responsive sourcing remain in the cut. There is no resolved
organization-dependent contribution specifically attributable to the interrupted
mediator feedback under this contract. A fitted return variable or the odd
probe's persistent moment is not evidence of an ontological memory component.

### Numerical uncertainty, contrary cases and costs

Each organization and both positive probes were checked at dt/2 and at doubled
N, separately, from the identical prepared state (Fourier-interpolated only for
the spatial diagnostic). The conservative sum of full/cut refinement RMS errors
sets the floors by the predeclared factor-five rule, taking the worst individual
probe. Direct Q refinement is far smaller: largest sampled absolute differences
are of order 1e-12. The wider conservative bound is retained rather than replaced
by that favorable cancellation. These are empirical discretization checks for
this periodic initial-value problem, not rigorous PDE bounds or a new domain
independence claim. Early Q below the floor is not called suppression.

The analytic mass-only design was rejected before simulation; its observability
failure remains in the contract. No organization or probe was discarded after
outcomes. No numerical, tangent, practical-selectivity, or held-out gate was
retuned. Resolved small total-response residuals, unresolved changes in return,
source imbalance, probe work imbalance and the control's remaining pathways are
all contrary to an inflated claim of pure gain, exclusive feedback, or new
response opportunities.

Measured process CPU seconds, with per-task stops enforced inside the loops:

| Stage | CPU seconds |
|---|---:|
| Development including both preparations and signed/half-size checks | 89.084 |
| Time and mesh qualification | 79.435 |
| Freeze and fit | 0.139 |
| Six fresh preparations and their response/control/tangent runs | 90.515 |
| Saved-data analysis and plots | 1.035 |
| **Timed total** | **260.208 (4.34 minutes)** |

The first preparation plus response cost 32.938 seconds and projected 494.068
seconds, below the cap; 600 seconds were reserved for fresh evaluation. Largest
observed response task was about 25.85 CPU seconds, under the 120-second task
stop. Final focused tests used a further 0.64 CPU seconds. A conservative
10-second budget allowance covers the first untimed instrument check and brief
interactive saved-data calculations. Total budget charge is therefore 270.85
seconds (4.51 minutes), below 1,800. Editing, reading and human-facing tool wait
time are not simulation CPU. No paid computation, neural training or reduction
fit occurred.

### Review and reproduction

New implementation is one focused runner,
`experiments/mediated_patterns/organization_response.py`, with ten focused tests
in `tests/test_organization_response.py`. Tests cover fixed physical anchors,
actual L2/action, parity observability, both event/readout derivatives, ETDRK4
linearization, source-preserving feedback interruption, matched shams, response
subtraction, common gain/delay with an unchanged time window, absolute floors,
preparation splits, non-overwriting output and resource stops.

Final verification: **10 tests passed**; focused Ruff checks/format checks and
`git diff --check` passed. Historical scientific panels and preservation audits
were not rerun. This is instrument validation, not implementation certification
or a statistical proof of the scientific interpretation.

All records are local under `work/mediated_patterns/organization_response/`.
`frozen-01/freeze.json` holds the numerical floors, fitted comparator, split,
source identities and development/refinement hashes. Each execution directory
has its pre-run contract, command, source snapshot and CPU receipt. Initial
fields, all branch readouts, tangent traces, individual stimulus work and
component descriptors remain in the development/fresh JSON/NPZ files.
`analysis-01/summary.json` contains the fresh scores;
`review-01/details.json` contains the additional descriptive calculations.
`analysis-01/report_calculations.py` preserves those read-only calculations.
The review receipt records the current hashes and commands.

Run from the repository root with one BLAS/OMP thread and `.venv/bin/python`.
The exact five staged commands are saved in `review-checks-01/commands.txt`.
Every output directory must be new. Reusing these seeds is reproduction of
exposed evidence, not another fresh test. Expensive stages remain outside CI.

Current execution uses Python 3.14.6, NumPy 2.5.3, SciPy 1.18.1 on macOS arm64.
The simulator, measurements and original preparation source were checked against
the requested starting commit for this investigation's identity; no archive was
restored and no broader preservation verification was repeated. Changes remain
local and reviewable on `codex/organization-conditioned-response`, with no push,
PR #1 update, tag change, archive change, merge, or Atlas write. The historic
AB20/C32 result, its slower implementation, C24 failure interpretations and
earlier pending Atlas paragraph remain unchanged.

### Exact pending Atlas paragraph — record only, no Atlas edit

> Lab-local organization-conditioned-response exploration on base
> 1c555d5aab826418c81e0a9005a85200e3f8fd65 supplied three prepared SH35 patterns,
> the existing diffusive mediator/source/feedback laws, two fixed additive C
> probes and fixed mass/signed-moment readouts. Moving A changed AB separation
> from 24 to 32 while nominal BC remained 25; prepared source strength and C's
> background also changed. On six fresh preparations across three paired seeds,
> a development-frozen common gain with zero delay described total response to
> 0.084–0.107% weighted error. Small departures from exact gain are numerically
> resolved but below the prospective 5% practical criterion. The selected
> own-sham AB mediator-feedback cut preserves the unperturbed trajectory and
> leaves those small total-response differences almost unchanged; the change
> between organizations in its controlled return contrast is below the numerical
> floor. The equation-derived trajectory-dependent tangent predicts the tested
> response and supports an ordinary conditioned interacting explanation, without
> separating static conditioning from all remaining direct pathways. Evidence is
> locally executed and reviewable, unpushed and unmerged, separate from the
> remotely preserved completed AB20/C32 result in draft PR #1. This concerns an
> 80-unit, two-probe response panel, not native formation, binding, physical mode
> uniqueness, arbitrary-input closure, new global reachability or a new Atlas
> project. No practical selective-response capability or automatic successor is
> selected; certifying an interface and explaining its production remain distinct.
