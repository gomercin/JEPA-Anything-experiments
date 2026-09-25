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
