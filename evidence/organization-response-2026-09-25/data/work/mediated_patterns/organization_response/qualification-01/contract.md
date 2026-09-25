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

## Execution and result

Pending. Append the measured outcome, contrary cases, costs, checks, figure and
exact pending Atlas paragraph here; do not rewrite this prospective contract.
