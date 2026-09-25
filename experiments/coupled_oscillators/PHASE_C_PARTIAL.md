# Phase C: noiseless partial observation

Exploratory session, 2026-09-19. **Ordinary delay identification is sufficient for
the tested POSITIONS problem.** Two position samples support an accurate
six-dimensional autonomous predictor without access to simulator velocities.
The learned JEPA states improve on memoryless one-step prediction, but their
recursive dynamics remain poor. Learned soft OPF adds no predictive benefit in
this setup. This is not a general negative result about neural state estimation.

The useful outcome labels are `DELAY_REGRESSION_SUFFICIENT` and, for the tested
neural rollout objective, unstable/inaccurate learned dynamics. A compact
**conventional delay state** was demonstrated; a sufficient learned neural state
was not certified. No factor-interpretation analysis is warranted.

## Contract and information boundary

- Same three oscillators, Duffing coefficient alpha=0.5; alpha=0 is the linear
  observability calibration. Sampling dt=0.1, RK4 with eight substeps, reusing
  the qualified Phase B simulator. Stiffness, damping, initial-state generation,
  and numerical integration are unchanged.
- Seeds 11, 22, 33 jointly vary trajectories, initialization, and minibatches.
  Each seed has 48 training and 16 held-out trajectories, each with 65 samples.
  Split shuffle uses seed+10000; minibatches use seed+20000. No trajectory spans
  train and test. Amplitude-shift test multiplies held-out initial conditions by
  1.5. This is exploratory reuse of test panels, not fresh confirmation.
- POSITIONS observes q1,q2,q3. LOCAL observes q1,q2. History H=1,2,4,8 is flattened
  after an invertible observation-only transform: current positions plus all
  adjacent finite differences divided by dt. These are not simulator velocities.
  Scaling and SVD/PCA fits use training observations only.
- Main training functions receive only observations. Targets are future
  observations or EMA embeddings of the next allowed observation history.
  True simulator states are kept in a separate evaluator object, never passed
  into main models, normalizers, losses, or fitting functions. Hidden-state
  probes run after main training and did not select architectures or budgets.
- All histories use the same forecast origins: training t=15..63 (2,352 pairs),
  evaluation t=15..48 (544 origins), with horizons 1,4,8,16. Errors are physical
  **observed-position** MSE, not hidden-state or latent MSE. LOCAL errors average
  two coordinates; POSITIONS errors average three, so compare within regimes.
- The pulse contract is **unknown intervention, observed consequences**. At
  simulator t=24, p1 receives +0.2. No action or pulse label enters a learner.
  Forecasts start eight samples later, using only post-pulse position history.
  Thus this tests assimilation of consequences, not anticipation of an unseen
  pulse. Both pulsed trajectory error and pulsed-minus-base response error are
  recorded, also under amplitude shift.
- Everything is CPU float64, one thread, deterministic PyTorch operations.
  All earlier Phase A/interrogation/frozen-encoder/Phase B artifacts (151 files)
  were hash-checked unchanged before/after every panel and at session completion.

## 1. What observation histories are sufficient?

**Observation.** POSITIONS H1 has linear observability rank 3; H2 reaches rank 6
with condition number 20.0. Linear delay regression at H2 predicts linear-world
16-step observations at roughly 1e-27 MSE. In the nonlinear world, a cubic delay
transition at H2 reduces 16-step MSE from about 0.99 at H1 to 1.92e-8.

| Observation | Linear rank at H=1,2,4,8 | Full-rank condition at H4 / H8 |
|---|---|---:|
| POSITIONS | 3,6,6,6 | 8.93 / 4.29 |
| LOCAL | 2,4,6,6 | 43,888 / 3,164 |

**Competing explanations and tests.** H1 failure could reflect missing temporal
information or a weak predictor. Linear rank, H1/H2 delay models, and later
probe controls distinguish them: history supplies essential information. The
nonlinear conclusion is empirical over the sampled envelope, not a global
observability theorem.

LOCAL is harder. Cubic H2 gives ordinary h16 MSE 0.00266±0.00049 but is rank
deficient in the linear reference. H4 is linearly full rank yet poorly
conditioned. Its nonlinear polynomial recurrences can fail catastrophically.
This is **not evidence that LOCAL is structurally non-identifiable**. No LOCAL
neural panel or MINIMAL/noise/dimension sweep was launched.

## 2. What conventional method is strongest?

The strongest small comparator forms six whitened linear coordinates from
the latest two 3D position samples, fits a complete cubic transition by SVD
least squares, and fits a linear observation readout. It recursively updates
only six numbers; no older history or hidden state is supplied after the origin.
There are 84 cubic features including the constant, 504 transition coefficients,
and 21 observation-readout coefficients. The encoder is an invertible fitted
coordinate change of the six delay values. It is not a full N4SID implementation
or a discovery of physical coordinates. Direct cubic autoregression at H2 is
functionally equivalent to this predictor to numerical precision.

| POSITIONS history | Linear AR h16 | SVD + cubic state h16 |
|---|---:|---:|
| H1 | 0.991±0.045 | 0.991±0.067 |
| H2 | 0.0110±0.0021 | 1.92e-8±4.6e-9 |
| H4 | 0.0116±0.0017 | 2.21e-5±4.4e-6 |
| H8 | 0.00109±0.00036 | 0.00153±0.00033 |

**Unexpected result and discriminating tests.** Longer history made some fitted
models worse, despite containing the two useful samples. Is six-component PCA
discarding information, or is redundant delay fitting unstable? Keeping every
H4 component and fitting full cubic features produced extraordinarily small
one-step errors but nonfinite h16 predictions for all three POSITIONS seeds.
The scaled feature matrices had condition numbers around 2e11–4e11. Sampled
history-update Jacobian radii were 32–38. One fixed ridge coefficient (1e-6,
mean-loss scaling) made recurrences worse; no coefficient search followed.
Using only the most recent two samples from the same H4/H8 inputs restored the
excellent H2 result.

**Best explanation.** Extra history is not intrinsically harmful or missing
information. Redundant delay coordinates, compression choice, and an unstable
estimated recurrence are the problem. PCA is variance-based, not guaranteed
to preserve predictive information. Full cubic features do not fix recurrence
conditioning. Nonfinite outcomes remain explicit JSON null metrics with failure
counts; they were never clipped or removed from aggregates.

## 3. Does a learned latent state help?

The neural panel uses POSITIONS H4 (12 observed values compressed to d=6), a
12→32 GELU→6 history encoder, a core capacity-matched 6→32→6 predictor, and a
6→3 linear visible readout. K=3,r=2. The target history encoder is EMA, momentum
0.99. Loss is normalized next-observation MSE plus core JEPA latent loss;
soft OPF additionally uses the core Gram/activity objective. Gram weight=0.05,
activity weights=0.1, minimum standard deviation=0.1. Adam lr=0.003, batch=128.
Variance regularization is evaluated in the canonical latent frame so that it
does not introduce a separate coordinate-control confound.

Native Standard/fixed have 1,057 learned parameters; learned OPF and ordinary
output transform have 1,093. Core predictor counts match exactly at 422. These
are JEPA-style models with an explicit observable supervision term, not a claim
to test every possible JEPA training recipe. The predictor class is shared; OPF
changes the loss geometry as well as adding a basis. Its synthesis uses the
core pseudoinverse, not a manually assumed transpose.

At 1,000 steps Standard H4 one-step MSE is about 0.00069 versus 0.00847 for H1:
the encoder uses history. But autonomous rollout remains poor. Declining loss
motivated exactly one duration extension to 3,000 steps, with every 1,000-step
checkpoint reproduced **exactly**, including weights and diagnostics.

Means ± sample SD over three seeds; native neural results below use 3,000 steps.

| Method | h1 | h4 | h8 | h16 | amplitude-shift h16 |
|---|---:|---:|---:|---:|---:|
| Conventional 6D delay state | 8.35e-12±9.4e-13 | 6.59e-10±9e-11 | 5.13e-9±7.6e-10 | 1.92e-8±4.6e-9 | 5.50e-7±1.1e-7 |
| Standard JEPA | 1.45e-4±6.4e-5 | 0.0313±0.0054 | 0.144±0.077 | 0.889±0.44 | 2.98±1.4 |
| Fixed factorized | 1.45e-4±6.4e-5 | 0.0313±0.0054 | 0.144±0.077 | 0.889±0.44 | 2.98±1.4 |
| Learned soft OPF | 2.99e-4±3.2e-5 | 1.16±0.40 | 18.7±28 | 8.11e3±1.4e4 | 1.48e4±2.6e4 |
| Ordinary output transform | 1.33e-4±5.6e-5 | 0.0249±0.0086 | 0.106±0.059 | 0.640±0.36 | 1.88±0.46 |

**Best explanation and limits.** Temporal information is available and used,
but these one-step-trained latent transitions do not yield accurate autonomous
dynamics. This weakens an information-absence explanation. It does not rule out
better neural training, a different objective, or more data. A compact state
need not be neural: the six-dimensional conventional state already succeeds.

## 4. Does fixed factorized output help?

Native Standard and fixed factorized runs agree to roundoff at every saved
checkpoint. The fixed identity basis, matched stacked heads, and inactive
variance-floor penalties make their native objectives equivalent here. This is
an expected control result, not independent replicated evidence for a different
architecture. Rotated runs differ under Adam, but fixed factorization does not
produce a useful accurate state. See the coordinate table below.

## 5. Does learning the OPF basis add anything?

No predictive gain was found. OPF h16 errors at 3,000 steps are **24,297.5, 5.91,
12.35** for seeds 11,22,33: the mean is dominated by seed 11, but all three are
worse than their matched Standard and ordinary-transform runs.

**Competing explanations and tests.** Encoder collapse, target-frame mismatch,
undertraining, and unstable learned transitions were considered. Encoder final
linear condition numbers are 2.53–3.42 for OPF, rather than the earlier Phase A
pathology near 519. Latent covariances remain full rank. OPF analysis-map
condition numbers are 12.1–18.5, orthogonality errors about 0.40, and sampled
transition Jacobian radii reach 1.20–1.69. Factor coordinates remain active;
cross-factor correlation reaches 0.69–0.85. Exact pseudoinverse round trips have
errors around 1e-15. Thus synthesis correctness does not imply stable recurrence.

A train-only affine adapter from target-encoder to online-encoder coordinates
reduces Standard mean h16 from 0.889 to 0.656 and OPF from 8,105 to 4,861, still
poor. Re-encoding predicted observation histories makes both much worse.
This weakens **EMA mismatch alone** as an explanation. It supports unstable
one-step-trained recurrence, with additional OPF conditioning problems, but does
not identify a unique cause. Local Jacobian samples are diagnostics, not a
global stability proof. No Gram sweep or alternative OPF rescue was attempted.

## 6. Does the latent replace history?

Encoders were frozen. Matched nominal-width linear heads fit complete cubic
features of whitened z plus (a) zero slots, (b) eight real older observations, or
(c) same-time older observations from a different trajectory. All targets are
allowed future positions. Train/test trajectories remain disjoint. Heads have
the same nominal coefficient count; zero slots have lower effective rank, so
this is not a perfect effective-capacity match.

| Frozen state | z-only h16 | z + older history h16 | z + shuffled history h16 |
|---|---:|---:|---:|
| Standard, 3,000 steps | 0.0330±0.043 | 0.00147±0.00051 | 0.0366±0.048 |
| Learned OPF, 3,000 steps | 0.0758±0.012 | 0.00198±0.00052 | 0.0799±0.011 |
| Ordinary transform, 3,000 steps | 0.0395±0.034 | 0.00148±0.00059 | 0.0442±0.038 |
| Conventional two-sample state | 0.000777±0.00036 | 0.000165±0.000080 | 0.000853±0.00039 |

**Critical confound.** Older history helps the polynomial probe even for the
conventional state whose recursive h16 error is only 1.92e-8. A direct cubic map
over 16 steps is a different function class from 16 compositions of a cubic
transition. Older history can supply an easier nonlinear readout of information
already available in z. A degree-five probe worsened held-out error (z-only mean
0.0306), indicating finite-data/readout estimation problems rather than fixing
the test. No degree or regularizer search was performed.

Calibration works in the linear world: H1 z-only mean error 0.317 falls to
2.3e-29 with older history; H2 is already about 4e-30 and gains nothing.
Therefore the probe detects genuine missing history in a known case, but its
nonlinear result **cannot certify conditional insufficiency**. The learned
latents have not earned task-relative sufficiency; neither has this finite
probe proved that their missing information, rather than readout limitations,
causes failure. Do not equate direct-probe improvement with causal state discovery.

## 7. What hidden information is encoded?

Evaluator-only affine probes from frozen z to full physical x yield these
ordinary-test velocity MSEs:

| State | Velocity MSE |
|---|---:|
| H1 Standard, 1,000 steps | 0.782±0.074 |
| H4 Standard, 1,000 steps | 0.142±0.082 |
| H4 Standard, 3,000 steps | 0.197±0.198 |
| H4 OPF, 3,000 steps | 0.211±0.027 |
| H4 ordinary transform, 3,000 steps | 0.181±0.146 |
| Conventional two-sample state | 1.04e-4±1.9e-5 |

History supplies linearly recoverable momentum-related information. It is not
reliably retained/improved by longer one-step training: Standard seed 11's probe
worsens from 0.212 to 0.423 while its prediction training loss improves. This is
an exploratory warning about the objective, not evidence of a discovered or
uniquely identified velocity coordinate. A nonlinear probe could differ.

## 8. Does the inferred state transfer?

The conventional state retains low amplitude-shift error (5.50e-7 at h16).
Its ordinary pulsed-trajectory h16 MSE is 1.83e-8±1.1e-8, and response-difference
MSE is 1.36e-9±1.0e-9. Shifted pulsed-trajectory error is 3.64e-7±2.2e-7.

Neural pulse results remain poor: ordinary pulsed-trajectory h16 means are
0.977 Standard/fixed, 8,918 OPF, and 0.565 ordinary transform. Response-difference
means are 0.0271, 199, and 0.0107 respectively. There is no OPF transfer benefit
to interpret. Forecasting consequences after observation is not predicting an
unknown intervention before it occurs, and is not causal discovery.

## 9. Is the representation stable or functionally equivalent?

One fixed seeded orthogonal latent rotation (2718) was tested with functionally
matched initialization. Initial observable forecasts agree within 1.1e-16;
initial losses agree within roundoff. Comparison uses decoded common physical
observations, not raw weight entries.

| 1,000-step model | Native h16 | Rotated h16 | Paired physical-function MSE at h16 |
|---|---:|---:|---:|
| Standard | 1.50±0.12 | 2.06±0.40 | 2.53±0.75 |
| Fixed factorized | 1.50±0.12 | 1.73±0.24 | 2.23±0.51 |
| Learned OPF | 1.62e5±1.6e5 | 6.69e5±1.1e6 | 1.08e6±1.3e6 |
| Ordinary transform | 1.68±0.40 | 2.05±0.52 | 2.41±0.79 |

Paired function differences use the same first origin in each test trajectory;
the main MSE columns use all 544 origins. Coordinate/optimization sensitivity
remains material, while conventional prediction is already accurate. The learned
encoder reintroduces that confound; fixed output bases alone cannot remove it.
No physical subspace stability claim is justified, and no factor matrices were
interpreted as modes. SGD and alternate rotations were not rerun because no
candidate learned advantage survived the simpler controls.

## 10. What should the next different experiment ask?

Stop here: competent delay identification already resolves noiseless POSITIONS
within the tested envelope. Do not escalate difficulty to obtain an OPF gain.

If a separate session is authorized, the clearest new question is whether a
compact predictive state remains useful under **measurement noise**, where
two-sample finite differences become fragile and uncertainty/state estimation
matter. It would need a conventional smoothing/state-estimation comparator and
a calibrated sufficiency test. Before making neural representation claims,
recursive or multi-step observation training also needs qualification. Neither
noise nor a new training objective was implemented here.

Boundaries: information in history is not efficient learnability; lower error
is not ontology; factorization is not a causal decomposition; a compact delay
state is not unique; a useful nonlinear probe is not a physical state certificate.
No Atlas, core, earlier results, additional fields, or active-observer work changed.

## Evidence, reproduction, and checks

Outputs are under `work/coupled_oscillators/phase_c_partial/`, separate from every
earlier phase. Six completed panels contain configs, seeds, split IDs, rotation,
source snapshots/hashes, parameter counts, fitted conventional coefficients,
neural checkpoints at 0/250/1000/3000 as applicable, diagnostics, and strict JSON.
`analysis.json`, `tables.md`, `COMMANDS.md`, `summary.md`, and two PNG plots provide
the review surface. Raw panel files were not replaced by the aggregation script.

Run commands from the repository root, with **new** output directories; existing
directories are rejected. The executed command list is in `COMMANDS.md`.

```bash
.venv/bin/python -m experiments.coupled_oscillators.phase_c_partial --qualify --output work/coupled_oscillators/phase_c_partial/c0
.venv/bin/python -m experiments.coupled_oscillators.phase_c_partial --compression-control --output work/coupled_oscillators/phase_c_partial/c0-compression
.venv/bin/python -m experiments.coupled_oscillators.phase_c_partial --neural --qualification work/coupled_oscillators/phase_c_partial/c0 --output work/coupled_oscillators/phase_c_partial/c1
.venv/bin/python -m experiments.coupled_oscillators.phase_c_controls --ridge --output work/coupled_oscillators/phase_c_partial/c0-ridge
.venv/bin/python -m experiments.coupled_oscillators.phase_c_partial --duration --qualification work/coupled_oscillators/phase_c_partial/c0 --output work/coupled_oscillators/phase_c_partial/c2-duration
.venv/bin/python -m experiments.coupled_oscillators.phase_c_controls --diagnose --panel work/coupled_oscillators/phase_c_partial/c2-duration --output work/coupled_oscillators/phase_c_partial/c3-diagnostics
make check PYTHON=.venv/bin/python
```

Recorded panel runtimes total about 63 seconds, excluding process startup,
development, plotting, and validation. There were 27 initial neural fits and
12 native duration replays, with three paired seeds throughout. Conventional
panels contain 96 qualification, 12 compression-control, and 6 ridge records;
feedback/probe diagnostics reuse trained weights rather than training new main
models. Scope deliberately excludes noise, MINIMAL, a latent-dimension sweep,
LOCAL neural models, full subspace-ID packages, and factor interpretation.

Validation: `make check PYTHON=.venv/bin/python` passed all deterministic checks:
66 core tests, 33 Skill tests plus 2 subtests, and 40 oscillator tests. Ten new
partial-observation test cases cover the observation/hidden-state boundary,
history/trajectory timing, pulse availability, initial rotation equivalence,
OPF backward pass, frozen probes, deterministic short training, EMA-feedback
placement, strict JSON, and explicit nonfinite-result handling. Ruff and whitespace
checks also pass. There are no assertions that a model must win scientifically.
