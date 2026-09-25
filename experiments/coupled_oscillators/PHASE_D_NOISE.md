# Phase D: state estimation from noisy positions

Exploratory session, 2026-09-19. The closest outcome is
**CONVENTIONAL_FILTERING_SUFFICIENT for the tested point-prediction task**,
with **LEARNED_OPF_ADDS_LITTLE** and **PREDICTIVE_SUFFICIENCY_UNCERTIFIED**.
Noise makes the excellent Phase C delay solution fail. An observation-identified
mechanical model plus causal filtering recovers much of the oracle reference's
accuracy. The tested neural states do not improve on that conventional estimator.
Uncertainty calibration and amplitude transfer remain imperfect.

This is a bounded investigation of one engineered oscillator family, not a
claim that conventional filters solve arbitrary noisy representation learning.
No earlier interpretation was revised, no factors were interpreted, and no
harder world was started.

## Contract and scope

- Same alpha=0.5 world, dt=0.1, RK4 with eight simulator substeps. The alpha=0
  world supplies a separate Kalman-style calibration. No new physical dynamics,
  missing sensors, colored noise, drifting sensors, or process noise.
- Measurements are q1,q2,q3 plus independent Gaussian noise. The prospective
  ladder was 0%, 5%, 15% of RMS position standard deviation, approximately
  infinite, 26.0, and 16.5 dB signal/noise ratios. Noise generation calibrates
  scale from simulator-side training positions; **that scale and true sigma
  never enter learned models or their normalization**. Low sigma is
  0.0289–0.0300; moderate sigma is 0.0866–0.0899 across the three seeds.
- Seeds 11,22,33, 48 train/16 test trajectories, 65 samples each. Whole-trajectory
  split and forecast origins match Phase C. H=2,4,8 conventional screening;
  H8 primary filtering/neural comparison, with H2 controls. Evaluation uses
  544 origins per seed and horizons 1,4,8,16. SD means sample SD across paired
  data/model seeds, not a confidence interval.
- Main training targets are noisy observations or embeddings/statistics derived
  from noisy observations. Clean positions and all physical states are evaluator
  only. Neural constructors receive a restricted settings object containing no
  simulator coefficients, noise level, or pulse information.
- Noise seeds are dataset seed+60000, then offsets 0/1/2 for train/test/shift,
  and 10..13 for pulse branches. Held-out noise uses fresh draws at offset+1000
  and sigma multiplied by 1.5. Standardized draws are paired across ladder levels.
- Amplitude shift multiplies initial conditions by 1.5 while retaining the same
  absolute sensor noise: its SNR therefore changes. Unknown pulse: +0.2 to p1
  at t=24, then eight post-pulse measurements before forecasting. No action,
  timing, or pulse label enters a model. Base/pulse branches have independent
  measurement noise, so response-difference error includes both inference errors.
- All compared forecasts are online: only the declared H past/current samples
  enter each estimate, and no measurements enter the subsequent autonomous rollout.
  Filters reset at each context start; they do not receive longer history than
  neural models. Training-only centered smoothing is explicitly declared below.

## 1. What does noise break?

**Observation.** Noise-free H2 exactly reproduces Phase C. Noise severely damages
the noise-trained cubic delay transition, even at the lower level.

| Noise/signal SD | H2 clean h16 MSE | H4 clean h16 MSE | H8 clean h16 MSE |
|---|---:|---:|---:|
| 0 | 1.92e-8±4.6e-9 | 2.21e-5±4.4e-6 | 0.00153±0.00033 |
| 0.05 | 0.454±0.036 | 0.0466±0.0016 | 0.0391±0.0043 |
| 0.15 | 0.783±0.078 | 0.350±0.037 | 0.373±0.031 |

**Competing explanations:** noisy initialization, damaged regression, unstable
recurrence, and loss of useful directions during six-component PCA compression.
**Test:** feed clean evaluator histories through the noise-trained model without
refitting. Moderate-noise H2 changes only from 0.783 to 0.782; H8 from 0.373 to
0.375. Thus noisy initialization alone is not the explanation. Naive noisy
transition identification is also damaged. Longer history helps at low noise
but does not guarantee a useful compressed state. Some amplitude-shift polynomial
rollouts overflow; those outcomes remain explicit null metrics and failure counts,
not clipped predictions or dropped seeds.

## 2. What is the strongest conventional estimator?

We successively tested three learned conventional approaches and a separate oracle:

1. The original SVD/cubic delay-state regression, retrained on noisy positions.
2. Causal local-polynomial endpoint filtering of H samples into position/slope,
   followed by cubic state-transition regression. This uses no future test data.
   It did not solve noisy recurrence; moderate H8 h16 MSE is 0.701±0.026.
3. An **identified mechanical model plus a windowed EKF**. Acceleration is fitted
   as a complete cubic in three position coordinates plus linear velocity terms:
   69 global coefficients. Initial fitting uses centered degree-five, 11-sample
   polynomial derivatives of **noisy training trajectories only**. Measurement
   variance is estimated from third differences, using the white-noise identity
   Var(delta-cubed noise)=20 sigma-squared. No true noise variance or dynamics
   coefficients are supplied. Test filtering uses past/current samples only.
4. `ORACLE_EKF` knows true dynamics and training measurement variance. The linear
   `ORACLE_KALMAN` calibration uses the same estimation mechanics. These are
   privileged references, **not learned-method scores or rigorous Bayes bounds**.
   They retain the same weak initialization prior and H-limited history.

The first identified model left a dynamics gap. A targeted **output-error
refinement** jointly fitted those 69 coefficients and unknown initial states for
four 16-step training blocks per trajectory, using noisy position error only.
The 1,152 fitted training-start coordinates are nuisance parameters, not hidden
labels or parameters used at test time. LBFGS used lr=0.5, a fixed 1e-6 coefficient
penalty around its initializer, and no coefficient/grid search. An 80-iteration
fit remained underconverged and worsened prediction; a single extension to 240
iterations improved it substantially. Both results are preserved.

**Best explanation:** conventional identification and online state estimation
are effective for this family. The final estimator is not an offline test
smoother. It maintains a 6D mean and a covariance during assimilation; point
rollout then propagates the mean, while uncertainty diagnostics also propagate
covariance. Do not call its entire belief state six-dimensional.

**Important limitation:** the conventional model has an explicit mechanical prior
q'=v and cubic(q)+linear(v) acceleration. It receives no true coefficients, but
this is a stronger structural prior and a different identification objective
from a generic one-step JEPA MLP. The comparison does not establish that every
neural state-estimation method is inferior.

## 3. Is the main error state estimation or dynamics?

**Observation/test.** Start only the identified transition from exact evaluator
physical state, bypassing the filter. Its ordinary h16 error falls from 0.00993
before refinement to 0.00242 afterward. The corresponding complete online pipeline
improves from 0.0148 to 0.00630. Direct current-position estimation errors stay
around 0.003, close to the oracle's value.

**Best explanation:** noise damaged dynamics identification, and reducing that
error improves forecasting without a large change in current-state estimation.
The remaining gap includes model bias and state uncertainty. These errors are
coupled; the diagnostic is not an exact additive decomposition. Oracle equations
from exact state give about 1.4e-15 error, checking integration consistency.

Longer context is genuine repeated evidence under noise: at moderate noise,
the oracle's nonlinear h16 MSE drops from roughly 0.120 at H2 to 0.00334 at H8.
The linear calibration likewise drops from about 0.209 to 0.00653. This makes an
observation-process-is-hopeless explanation implausible at the chosen levels.

## 4. Do neural latent models improve state estimation?

Phase C's flattened-history encoder and actual core APIs were reused: H8 gives
24 inputs, a 32-unit GELU hidden layer, d=6, K=3,r=2, and the same capacity-matched
latent predictors. Standard/fixed have 1,441 trainable parameters; OPF/ordinary
transform have 1,477. Predictors alone match at 422. Adam lr=0.003, batch=128,
EMA=0.99, Gram weight=0.05, activity weights=0.1 and minimum SD=0.1 are unchanged.
The objective remains noisy next-observation MSE plus core JEPA latent loss.

Screening covered both noise levels, H2 moderate-noise Standard, and no-noise H8
Standard. Declining training losses justified one 3,000-step moderate-noise
extension; all twelve 1,000-step checkpoint prefixes reproduced exactly.

Mean±sample SD, moderate noise. Neural rows use 3,000 steps and native coordinates.

| Method, H8 unless indicated | Clean h1 | Noisy h1 | Clean h16 | Noisy h16 |
|---|---:|---:|---:|---:|
| Naive cubic delay H2 | 0.0160±0.00034 | 0.0236±0.00056 | 0.783±0.078 | 0.790±0.077 |
| Initial identified EKF | 0.00430±0.00045 | 0.0121±0.00099 | 0.0148±0.00050 | 0.0225±0.00085 |
| Output-error identified EKF | 0.00407±0.00044 | 0.0118±0.00099 | 0.00630±0.00058 | 0.0143±0.00064 |
| ORACLE EKF, privileged | 0.00406±0.00044 | 0.0118±0.00099 | 0.00334±0.00030 | 0.0111±0.00017 |
| Standard JEPA | 0.00644±0.00040 | 0.0143±0.0012 | 0.393±0.060 | 0.402±0.058 |
| Fixed factorized | 0.00644±0.00040 | 0.0143±0.0012 | 0.393±0.060 | 0.402±0.058 |
| Learned soft OPF | 0.00664±0.00057 | 0.0143±0.0011 | 0.459±0.088 | 0.464±0.086 |
| Ordinary output transform | 0.00604±0.00052 | 0.0139±0.0014 | 0.293±0.11 | 0.300±0.12 |

The average moderate sensor variance is about 0.00778. Noisy-target MSE includes
that irreducible future measurement component; clean-truth MSE does not. Both
are recorded independently. One-step differences are small relative to the
large autonomous-rollout gap. No generic neural advantage over competent
identified filtering was found under this training recipe.

## 5. Does fixed factorization help?

Native Standard/fixed predictions and objectives agree to roundoff, as expected
from the identity basis and identically initialized stacked heads. There is no
additional predictive benefit here. An ordinary 36-parameter output transform
modestly improves neural rollout. No basis-learning explanation is needed.

**Unexpected observation:** noise improves neural long-rollout error relative
to noiseless training while worsening one-step accuracy. Is that just different
normalization, noisy test inputs, or altered learned dynamics? In the additional
control, noiseless and moderate-noise Standard runs begin with **identical weights
and normalization buffers**, all normalization fitted from noisy training data.
Evaluation uses the same clean histories and clean futures:

| Training/control, 1,000 steps | Clean-context h16 MSE |
|---|---:|
| No noise, original normalization | 2.12±1.3 |
| No noise, shared noisy-data normalization | 4.84±1.4 |
| Moderate noise, same shared normalization | 0.608±0.21 |

This weakens normalization-only and test-input-noise explanations. Sampled
transition Jacobians are generally less expansive under moderate-noise training.
Regularization/shrinkage of learned dynamics is plausible; the test does not
separate every optimizer and conditional-prediction effect. These forecasts
remain poor compared with filtering, so this is not a denoising victory.

## 6. Does learned OPF add anything?

No reproducible external gain survives the fixed and ordinary-transform controls.
At 3,000 steps the OPF h16 errors are 0.549,0.454,0.373, compared with ordinary
transform errors 0.421,0.260,0.199. Native frame was retained; there was no OPF
advantage warranting another full coordinate study or factor interpretation.
This does not repeal the earlier Adam coordinate-sensitivity findings.

All latent covariance matrices remain full rank. Final encoder linear condition
numbers are approximately 1.6–2.7, so the earlier extreme encoder-conditioning
pathology is absent. OPF basis conditions are 17.4,22.5,1.08, respectively; all
fail the strict orthogonality audit. Geometry varies considerably without a
distinct predictive benefit. Core geometry/activity/correlation and synthesis
diagnostics are saved, but no factor meaning or ontology is inferred.

## 7. Does performance transfer?

| Method | Amplitude-shift clean h16 | Pulsed clean h16 | Unseen-noise clean h16 |
|---|---:|---:|---:|
| Initial identified EKF | 0.0426±0.0036 | 0.0150±0.0044 | 0.0192±0.00065 |
| Output-error identified EKF | 0.0199±0.0037 | 0.00635±0.0015 | 0.0101±0.0011 |
| ORACLE EKF, privileged | 0.00346±0.00053 | 0.00355±0.00074 | 0.00684±0.00071 |
| Standard/fixed | 1.21±0.26 | 0.313±0.11 | 0.397±0.070 |
| Learned OPF | 1.11±0.18 | 0.471±0.13 | 0.479±0.10 |
| Ordinary transform | 0.998±0.40 | 0.233±0.083 | 0.302±0.11 |

No OPF advantage persists across these transfer measures. The refined filter's amplitude-shift gap to
the oracle remains material, so it does not match oracle dynamics everywhere.
In unseen-noise evaluation even the oracle reference retains its **training** R;
only its equations remain oracle-correct for that mismatch condition.

The refined filter's pulse-response-difference MSE is 0.00678±0.0010, versus
0.00627±0.00081 for the oracle. Independent branch noise can dominate a small
response. A small difference error alone is insufficient: a predictor that
suppresses both trajectories can appear good on that metric. The full pulsed
trajectory errors are therefore reported alongside it. This is post-pulse
prediction from observed consequences, not causal discovery.

One isolated favorable OPF comparison is retained: its mean response-difference
error is 0.0207 versus 0.0288 for the ordinary transform. But its full pulsed
trajectory error is substantially worse, and both response errors exceed the
identified filter's. This does not establish a robust learned-basis benefit.

## 8. What hidden information is encoded, and what about uncertainty?

Evaluator-only affine probes are fitted after main training with encoders frozen:

| Representation | Current clean position MSE | Hidden velocity MSE |
|---|---:|---:|
| Naive delay H2 | 0.00510±0.00026 | 0.518±0.019 |
| Output-error EKF mean | 0.00300±0.00029 | 0.0157±0.0019 |
| ORACLE EKF mean | 0.00301±0.00030 | 0.0152±0.0016 |
| Standard/fixed neural | 0.0101±0.0024 | 0.148±0.017 |
| Learned OPF | 0.0108±0.0027 | 0.259±0.10 |
| Ordinary transform | 0.00938±0.0031 | 0.140±0.053 |

Neural histories contain more linearly recoverable velocity information than
naive noisy delays, but the conventional filter recovers substantially more.
A linear probe is a restricted readout, not a proof about all information in z.

**Uncertainty limitation:** filter covariances propagate linearized measurement
uncertainty with zero process noise and no parameter uncertainty. Nominal 95%
clean-position marginal coverage at h16 is 54.2% for the initial identifier,
81.5% after output-error refinement, and 94.9% for the oracle. Noisy-target
coverage is 81.4%,91.3%,95.1%, respectively. Under unseen noise the refined
filter's clean coverage falls to 72.2%. Good point prediction does not establish
calibrated epistemic uncertainty. Position error/covariance need not increase
monotonically with horizon in an oscillatory system.

## 9. Is predictive sufficiency established?

No. The confounded Phase C extra-history probe was not repeated. Accurate
conventional mean forecasts do not prove that its mean alone replaces the
posterior distribution or every older measurement. Poor neural rollout also
does not prove its latent lacks information: learned transition/readout quality
can limit prediction. The outcome remains **PREDICTIVE_SUFFICIENCY_UNCERTIFIED**.
No further probe was needed to decide whether OPF had an external benefit here.

## 10. Stop decision and next question

Stop: a learned conventional dynamics model plus causal filtering handles this
noisy mechanical family far better than the tested neural recipes, including
held-out noise, amplitudes, and post-pulse prediction. No distinctive learned
factor benefit motivates a harder world. The main unresolved neural issue is
still how to train accurate autonomous transitions, not factor semantics.

Before another world, recursive/multi-step neural objectives would need their
own qualification if neural comparison is pursued. A genuinely different later
question could concern state estimation when the assumed mechanical model is
misspecified. That requires a separate contract; no such extension was implemented.

Boundaries remain: denoising/state estimation is not physical or causal discovery;
filter covariance is not automatically calibrated uncertainty; factorization
benefit is not learned-basis benefit; noisy-training regularization is not an
observer-emergence result. No Atlas or upstream core files changed.

## Artifacts, commands, and validation

New outputs: `work/coupled_oscillators/phase_d_noise/`. Seven completed panels:
45 D0 records; 30 filtering/calibration records; 30 initial neural fits;
3 initial output-error refinements; 12 neural duration replays; 3 normalization
controls; 3 longer output-error fits. Recorded panel wall times total about
243 seconds, excluding startup, development, validation, and plotting.

The first refinement launch failed before producing model results because
PyTorch LBFGS flattened a noncontiguous least-squares parameter gradient with
`view`. The failed protocol/source snapshot and `failure.json` remain in
`d3-output-error/`. Making the experiment-local parameter contiguous fixed it;
the retry used a new directory. There was no core bug or physics change.

`analysis.json` and `tables.md` contain aggregates and individual seed values;
`prediction.png` and `controls.png` show useful comparisons. Per-panel JSON
contains noise seeds/hashes, configs, splits, fit coefficients, neural checkpoints,
traces, clean/noisy metrics, probes, and covariance diagnostics. `COMMANDS.md`
lists exact scientific and validation commands. Existing output paths are
rejected. Raw completed outputs are not overwritten by reporting.

All **249 earlier A–C artifact files** were hash-checked unchanged. The preexisting
root `.gitignore`, `Makefile`, and `pyproject.toml` edits were not changed.

Final `make check PYTHON=.venv/bin/python` passed: **66 core tests; 33 Skill tests
plus 2 subtests; 46 oscillator tests**. Six new Phase D tests cover deterministic
noise/config reconstruction, noisy targets and truth boundaries, causal filtering,
unknown pulse, analytic filter Jacobian/covariance, explicit oracle labels,
the LBFGS layout regression, coefficient reconstruction, short OPF training,
and strict finite JSON. Scientific nonfinite prediction failures are retained
as null metrics and failure counts. Ruff and whitespace checks pass.
