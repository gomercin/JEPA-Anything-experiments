# Phase B: nonlinear oscillator interrogation — exploratory

**A is the closest description: this nonlinear family is still handled very
accurately by ordinary system identification.** Cubic transition regression
handles interpolation, pulses and the declared amplitude shift. The small neural
advantage from learned OPF is also reproduced by an ordinary learned output
transform. Fixed factorization already explains the reduced coordinate sensitivity.
There is no evidence here that a learned predictive decomposition is necessary.

This is an exposed-seed exploratory session, not a preregistered benchmark or a
proof about other worlds. Phase A and frozen-encoder findings stand unchanged.
All 50 files in the three protected artifact directories remain byte-identical.
No core code, PARTIAL observation, adaptive basis, larger system or Atlas changes.

## Instrument and successive decisions

The extension stays in the original dimensionless `[q1,q2,q3,p1,p2,p3]` world:

```text
dq/dt = p
dp/dt = -S q - 0.03 p - alpha * q^3
S = diag(1.0,1.4,2.0) + 0.2 * chain_laplacian
E = (p.T p + q.T S q)/2 + alpha * sum(q_i^4)/4
dE/dt = -0.03 * p.T p
```

`DuffingSimulator` subclasses the original simulator locally. At **alpha=0**, it
delegates to the original matrix-exponential flow, giving bitwise identical Phase A
trajectories. For positive alpha it uses RK4 inside each original `dt=0.1` interval.
The inherited stiffness eigenvectors remain a reference; there is no nonlinear
state-independent transition matrix. Calling `transition()` for positive alpha
explicitly fails rather than returning a misleading linear oracle.

| Panel | Question / decision | Scope |
|---|---|---|
| `b0` | Qualify alpha=0/.1/.5 before neural results; four substeps fail the numerical gate | 3 seeds, simulator + linear ID only; failed evidence retained |
| `b0-refined` | Double integration resolution, keep strengths and gate unchanged | 8 versus 16 substeps; all gates pass |
| `b1` | Does learned OPF add prediction beyond fixed parameterization and cubic ID? | 54 neural fits: 3 strengths × 3 seeds × 3 models × 2 frames, 1,000 steps; 18 conventional fits |
| `b2-controls` | Is conventional success coordinate/numerical privilege? Are neural models still learning? | Rotated cubic fits, finer truth, mode-coordinate linear fits; 9 moderate/native neural fits to 3,000 steps |
| `b3-transform` | Does the small longer-run OPF gain require OPF? | 3 ordinary output-transform fits to 3,000 steps; exact OPF-to-MLP folding check |

The candidate strengths .1 and .5 were declared before training. Their accepted
linear-error separation and stable energy motivated retaining them. The integration
gate was maximum state difference below `1e-5` between chosen and doubled substeps,
over 64 intervals, including train/test, amplitude-shift and post-pulse states.
Four substeps missed it in all three moderate seeds: maximum `1.38e-5`. Eight
substeps passed: worst `8.08e-7` moderate, `5.45e-8` weak. Refinement reduces errors
by roughly 16–17×, consistent with fourth-order integration. All sampled energies
decrease; no nonlinearity was chosen using a neural outcome.

The 66 neural fits use 90,000 optimizer updates **including replayed prefixes**.
The three comparison/control panels took about 50.5 seconds in total on this CPU,
excluding qualification and tests. No learning-rate or architecture sweep occurred.

## Data, controls and conventional baseline

Seeds 11/22/33 jointly control initial conditions, model initialization and minibatch
order. Each strength reuses the same Gaussian initial draws (SD .7), original
48/16 trajectory split and 64 transitions per trajectory. Split RNG is seed+10000;
batch RNG is seed+20000. Training has 3,072 transition pairs per seed. No normalization,
noise, augmentation, validation selection or early stopping is added.

The amplitude shift multiplies **held-out** initial states by 1.5 and resimulates
them with the same dynamics. This changes the initial-condition distribution,
not alpha or coupling. Gaussian supports overlap: it is not a disjoint-support
OOD claim. In the moderate regime, 10.0–25.9% of shifted trajectory states exceed
their seed's maximum training-state norm; none of the ordinary test states do.
Maximum observed norms: 3.60 train, 3.36 interpolation, 5.44 shifted. Maximum
energies: 6.71 / 5.97 / 15.76 respectively. The weak regime's corresponding
maximum energies are 5.35 / 5.31 / 12.41. Full distributions and coordinate-box
exceedance fractions are saved in B0.

All neural models use the existing frozen isometry, d=6, K=3, r=2, and a shared
6→32→6 GELU predictor from core capacity-matching tools. Native frame is I; rotated
frame is the existing seeded orthogonal R2718. Initialization transforms input
weights, output weights or the factor basis exactly as in the frozen-encoder study.
Initial paired physical predictions agree within `1.12e-15`. RAW/MIXED-to-latent
equivalence is gated again; encoders and targets stay fixed, with zero drift.
The target states are detached by core. EMA arithmetic is unnecessary on fixed
buffers. Physical decoding uses the exact isometric inverse, with no learned readout.

Adam remains .003 with default moments/epsilon and no weight decay. The same
examples and batches are used for all neural variants. OPF uses the actual core
soft-Gram machinery, weight .05, activity weights .1 and standard-deviation floor
.1. Factor targets retain basis gradients, and synthesis uses the core pseudoinverse.
Fixed OPF uses B=I natively and B=R.T in rotated coordinates. No Gram tuning or new
SGD study was needed. The common native standard/fixed result is an expected
functional equivalence, not independent model evidence.

| Method | Fitted/trainable parameters | Information / constraints |
|---|---:|---|
| Affine linear ID | 42 | Same physical train transition pairs; unrestricted SVD least squares |
| Cubic polynomial ID | 504 | All 84 monomials through degree 3, including constant; train-only RMS column scaling; SVD least squares |
| Standard / fixed factorized | 422 | Same core shared trunk and output capacity |
| Learned OPF | 458 | Adds 36 basis parameters and the existing composite objective |
| Standard + output transform | 458 | Adds identity-initialized bias-free 6×6 output map; ordinary prediction loss |

Polynomial ID receives **no simulator coefficients, derivatives, energies or
normal modes**. Feature degree is motivated by the cubic force, not selected by
model results. A sampled cubic vector field's flow is not exactly a cubic polynomial;
this baseline is an approximation, not a simulator oracle. All 84 design columns
have full rank; scaled design condition numbers are 9.42–13.46. Compute budgets
and fitting procedures across conventional and neural models are not matched.

## 1. Where does the linear description fracture?

Physical h16 MSE, mean ± sample SD over the three paired seeds:

| alpha | Linear ID, interpolation | Linear ID, amplitude shift | Cubic ID, interpolation | Cubic ID, amplitude shift |
|---:|---:|---:|---:|---:|
| 0 | 1.61e-28 ± 1.70e-28 | 3.59e-28 ± 3.84e-28 | 6.40e-28 ± 1.47e-28 | 3.68e-27 ± 1.47e-27 |
| .1 | .000826 ± .000295 | .01439 ± .00433 | 3.41e-10 ± 1.95e-10 | 2.11e-8 ± 1.50e-8 |
| .5 | .01219 ± .00249 | .2161 ± .0432 | 7.40e-8 ± 1.49e-8 | 9.41e-6 ± 2.40e-6 |

**Observation:** the linear transition model progressively fails, especially at
larger amplitudes. **Competing explanations:** nonlinear dynamics, discretization
error, or an unfavorable coordinate basis. **Tests:** integration refinement and
unrestricted linear fits in normal-mode coordinates. The modal and physical linear
predictions agree within `2.33e-14`. **Best explanation:** a linear transition is
insufficient. Full normal-mode coordinates still contain all information; changing
coordinates does not remove the nonlinear residual. **Limit:** only this sampled
amplitude envelope, coupling and 1.6-time-unit forecast horizon are covered.

## 2. Is ordinary nonlinear identification sufficient?

**Observation:** cubic ID reduces moderate h16 error by about 165,000× in
interpolation and 23,000× under the amplitude shift relative to linear ID. Its
shifted h16 RMSE is about .0031 in the original dimensionless state units.
**Competing explanations:** polynomial structure, privileged physical axes, or
numerical artifacts. **Tests:** fit complete cubic features in rotated coordinates,
decode physically, and score the original predictor against finer simulator truth.
Rotated/native polynomial rollouts differ by at most `1.11e-12`; all prediction and
pulse metrics change by less than **0.0068%** under finer truth. **Best explanation:**
ordinary nonlinear transition identification accounts for this fracture.
**Limit:** noiseless full state, ample samples for 84 features and an appropriate
polynomial family; this does not establish performance with unknown sensors or noise.

## 3. Does fixed factorization help? Does learning its basis help?

Moderate regime at 1,000 steps, native coordinates, physical MSE:

| Model | h1 | h4 | h8 | h16 |
|---|---:|---:|---:|---:|
| Linear ID | .000221 ± .000059 | .00274 ± .00071 | .00654 ± .00145 | .01219 ± .00249 |
| Cubic ID | 1.67e-9 ± 3.20e-10 | 1.86e-8 ± 2.42e-9 | 4.00e-8 ± 6.07e-9 | 7.40e-8 ± 1.49e-8 |
| Standard / fixed | .000538 ± .000102 | .00644 ± .00115 | .01729 ± .00308 | .03659 ± .00392 |
| Learned OPF | .000515 ± .000099 | .00614 ± .00108 | .01648 ± .00277 | .03522 ± .00362 |

**Observation:** native fixed factorization matches standard to roundoff. Learned
OPF has a small gain. At 1,000 steps its moderate h16 improvement over fixed is
2.8/4.4/3.9% by seed natively and 2.9/1.6/3.1% when rotated. But amplitude-shift
improvements are mixed: -2.9/+2.6/+2.5% native; -6.2/+1.1/+3.3% rotated.

**Competing explanations:** learned decomposition, extra transform parameters,
changed loss geometry, or unfinished optimization. **Tests:** training duration,
an ordinary parameter-matched output transform, and exact synthesis folding.

Moderate/native h16 error after the duration/control runs:

| Model / steps | Interpolation | Amplitude shift | Pulse response, interpolation | Pulse response, shifted |
|---|---:|---:|---:|---:|
| Standard / fixed, 1,000 | .03659 ± .00392 | .3243 ± .0515 | .001729 ± .000676 | .004559 ± .00134 |
| OPF, 1,000 | .03522 ± .00362 | .3208 ± .0422 | .001679 ± .000697 | .004266 ± .000973 |
| Standard / fixed, 3,000 | .01892 ± .00337 | .2704 ± .0437 | .001110 ± .000521 | .003641 ± .000720 |
| OPF, 3,000 | .01780 ± .00311 | .2570 ± .0432 | .001109 ± .000556 | .003384 ± .000454 |
| Ordinary output transform, 3,000 | .01665 ± .00206 | .2566 ± .0388 | .001088 ± .000528 | .003336 ± .000388 |

The longer runs reproduce their entire recorded 1,000-step metric prefix exactly.
At 3,000 steps OPF improves shifted rollout in all three seeds versus fixed.
The ordinary output transform beats OPF on interpolation for every seed, has
nearly identical mean shifted rollout, and mixed shifted seed results. Its map
condition numbers are 1.21–1.24. Thus a learned OPF basis is not needed to reproduce
the modest gain. Loss and optimization geometry differ between these controls;
this is not an isolated estimate of the Gram term's contribution.

**Structural check:** all core heads see the same shared hidden state. Synthesis
is a state-independent linear map. Its action can be folded into a standard output
layer (and the latent rotation into the input layer). Using **core compose** on unit
factor coordinates, the trained 458-parameter OPF predictors become ordinary
422-parameter MLPs with identical physical rollouts within `6.22e-15`. The function
class does not expand. Learning a basis can still change the training path or
regularization; this equality does not say those effects can never be useful.

**Best explanation:** unfinished learning plus output reparameterization explains
the observed small neural gains without evidence for distinctive predictive
structure. **Limit:** no claim that 3,000 steps reaches convergence. The final
extra-transform comparison is moderate/native only; it does not establish equality
for every optimizer, coordinate frame or nonlinear strength.

## 4. Do the gains survive coordinate and intervention controls?

Native-versus-rotated **physical prediction disagreement**, h16 at 1,000 steps
(this is not prediction error against truth):

| Regime | Standard | Fixed basis | Learned OPF |
|---|---:|---:|---:|
| Weak | .01414 ± .00219 | .005982 ± .001151 | .006296 ± .002171 |
| Moderate | .01706 ± .00100 | .008295 ± .002066 | .008565 ± .003054 |

**Observation:** reduced coordinate sensitivity survives, but fixed OPF explains
it at least as well on average. For moderate/rotated coordinates, mean h16 truth
errors are .03764 standard, .03839 fixed, .03739 learned: lower coordinate
disagreement does not imply better prediction. **Competing explanations:** changed
physics or input information versus Adam/output parameterization. **Tests:** frozen
isometry gates, matched initial functions, fixed counter-rotation, and complete
polynomial coordinate controls. **Best explanation:** the previous optimization
effect persists; no new nonlinear coordinate mechanism is needed. **Limit:** one
fixed rotation and three seeds; SGD is not re-investigated.

The original held-out **+0.2 p1 pulse at t=8** is retained for every test trajectory,
in both amplitude conditions. There are no training pulses. Evaluate both the
post-pulse trajectory and its difference from the unpulsed trajectory. Moderate
cubic h16 pulse-response MSE is `6.23e-9 ± 5.73e-9` in interpolation and
`5.91e-8 ± 3.12e-8` after amplitude shift; post-pulse trajectory MSE means are
`8.34e-8` and `4.12e-6`. These also survive finer truth. Ordinary identification
therefore predicts the changed response. This is not causal discovery.

## 5. Conditioning, rollout and interpretation gate

**Observation:** the 1,000-step nonlinear OPF analysis maps have condition numbers
1.0013–1.0036 and active coordinates, but all 12 fail the unchanged strict basis
orthogonality tolerance (1e-10). Maximum round-trip NMSE is below 2.81e-29;
cross-factor correlations range roughly .136–.395. Encoder conditions remain 1.
These are diagnostics, not performance certificates or independence claims.

Moderate neural probe Jacobian spectral radii reach about 1.043 at 1,000 steps.
The cubic predictor's probe radii are about .9985. The neural rollouts in this
panel do not show the previous explosive norm pathology over 16 steps. Saved
true/predicted norm traces and first envelope-exit steps distinguish ordinary
shifted inputs from predictor-created growth. Jacobian probes cover the first
four test trajectories at t=0 and t=8; a local radius is not a stability proof.
The strong improvement with duration weakens a representational-ceiling claim.

No factor-to-mode matching, factor ablation or cross-seed subspace interpretation
was launched. The ordinary transform control and polynomial baseline remove the
predictive reason to interpret the learned basis. Anonymous labels remain
pc_000/pc_001/pc_002. No assertion of reproducible physical structure is warranted.

## Stop, limits and one possible next question

This session meets **CONVENTIONAL_NONLINEAR_MODEL_SUFFICIENT** within the declared
family and envelope. Fixed factorization explains most coordinate-stability benefit;
ordinary output reparameterization reproduces the small learned-basis prediction
gain. This is closest to **A**, with optimization evidence related to **B/D**, not
evidence for C or a regime-specific physical representation (E).

The initial integration gate failure was fixed by numerical refinement and retained
as evidence. No upstream core, split, synthesis, physical decoding or pulse-truth
bug was found. A nonlinear oracle test specifically prevents accidentally using
Phase A pulse truth in this extension. The common frozen evaluator is reused;
the old frozen study's checkpoint helper is not used because it constructs a
linear simulator internally.

All seeds are exposed, test metrics informed follow-up choices, horizons are short,
and data are noiseless. Conventional and neural training are not compute-matched.
Small standard-versus-OPF differences should not be generalized. The amplitude
shift is harder for all methods; that fact alone does not imply a state-dependent
predictive representation.

No more strength, width, seed or Gram searches are justified by these results.
If a later decision opens a new question, the most informative one is whether
decomposition helps **under a separately controlled nonlinear observation map**,
where low-order polynomials in sensor coordinates are no longer tailored to the
vector field. That would be a different observation/representation experiment,
with new identifiability controls; it is not implemented here.

```text
nonlinear prediction benefit != physical ontology
learned decomposition != causal decomposition
reproducible subspace != fundamental mode
held-out perturbation prediction != causal discovery
factorization benefit != learned-representation benefit
learned-representation benefit != emergent higher-level object
failure of a linear transition != failure of conventional representations
frozen isometric encoder != realistic observation process
```

## Reproduction and checks

From the repository root, use fresh output directories (existing ones are refused):

```bash
.venv/bin/python -m experiments.coupled_oscillators.phase_b --qualify --substeps 8 --output work/coupled_oscillators/phase_b/NEW-b0
.venv/bin/python -m experiments.coupled_oscillators.phase_b --panel --qualification work/coupled_oscillators/phase_b/NEW-b0 --output work/coupled_oscillators/phase_b/NEW-b1
.venv/bin/python -m experiments.coupled_oscillators.phase_b_followup --panel work/coupled_oscillators/phase_b/NEW-b1 --output work/coupled_oscillators/phase_b/NEW-b2
.venv/bin/python -m experiments.coupled_oscillators.phase_b_followup --transform-control --panel work/coupled_oscillators/phase_b/NEW-b2 --output work/coupled_oscillators/phase_b/NEW-b3
make check PYTHON=.venv/bin/python
```

Each panel saves its protocol before execution, seeds, source hashes and runner
snapshots, metrics, diagnostics, model states and matrices. Comparison results are
strict finite JSON with SHA256 digests. B1 also emits a complete human-readable
mean/SD table. Source/command/result provenance and a reporting-only plotting script
are under `work/coupled_oscillators/phase_b/`; exact session commands are in
`COMMANDS.md` there. Plots require optional matplotlib; training does not.

Validation: **66 core tests; 33 skill tests and 2 subtests; 30 oscillator tests**,
all passing, plus design/recipe/manifest/JSON/compile checks and Ruff checks/format.
Nine new cases cover alpha-zero identity, nonlinear determinism/energy/convergence,
polynomial identification, nonlinear pulse truth, finite frozen training records,
matched extra-transform initialization, and folding in both coordinate frames.
Tests never assert a scientific model ranking.
