# Frozen-encoder investigation — exploratory

The frozen physical frame removes the RAW/MIXED discrepancy. Rotating latent
coordinates still changes Adam's learned physical function; SGD remains equivalent.
OPF is less sensitive here, but a fixed counter-rotated basis captures essentially
the same effect without learned basis parameters. Output parameterization is a
major confound in attributing this difference to learned factor structure.

These are exposed-seed exploratory observations, not preregistered results. The
original Phase A and interrogation directories remain byte-identical. No nonlinear
Phase B, PARTIAL observations, Atlas changes, core changes or hyperparameter search
were performed. Conventional linear identification still solves this world: the
original RAW/MIXED one-step MSE means are 9.78e-31 / 2.13e-30.

## Instrument

Keep d=6, K=3, r=2, and use W=I6. No latent expansion or rank-deficient covariance
is needed. For latent frame F (I or R) and observation matrix Q (I or the original
MIXED transform), the frozen encoder is F Q.T. The physical input takes the full
sensor path x -> Qx -> FQ.T Qx. Online and target encoders are distinct modules
with identical fixed buffers and zero parameters. The core detaches target states.
EMA arithmetic is skipped because the encoders never change.

Predicted latent states are decoded using the exact inverse F.T, with no fitted
readout. Rollout feeds back physical predictions through the sensor and encoder.
The existing evaluator supplies physical MSE at 1/4/8/16 and the fixed +0.2 p1 pulse.
Standard has 422 trainable predictor parameters, learned OPF 458 including its 36
analysis entries, and fixed OPF 422. Predictor width, examples, batches and initial
physical functions match. Functionally matched rotation initialization is:

```text
Standard: A1' = A1 R.T; A2' = R A2; b2' = R b2
OPF:      A1' = A1 R.T; B' = B R.T; factor heads unchanged
```

The hidden coordinates stay fixed. The core Gram objective is invariant to this
right rotation. Factor labels remain pc_000, pc_001, pc_002; no mode analysis is
performed. Frozen encoder variance penalties cannot update the encoder.

C0 gates all seeds and both latent frames before any panel training:

| Engineering check | Maximum observed |
|---|---:|
| RAW/MIXED latent absolute difference | 1.11e-15 |
| Relative Frobenius difference | 2.77e-16 |
| Isometry error | 2.22e-16 |
| Encoder singular values / condition | all 1 / 1 to roundoff |
| Encoder drift / online-target difference throughout | exactly 0 / 0 |

Latent covariance spectra, inverse reconstruction and norm distributions also
pass the declared 1e-12 gate. Full values are saved. No failed gate was bypassed.
C1 independently trained RAW and MIXED: maximum paired physical prediction
difference across recorded steps/horizons is 4.01e-13.

## Successive panels and decisions

Each directory under `work/coupled_oscillators/frozen_encoder/` contains the protocol
written before training, C0 checks, source hashes, exact executed runner, per-seed
metrics and model states, paired common-frame differences, diagnostics and digest.
Checkpoints are reporting points, not candidates for choosing a winning model.

| Directory | Discriminating question | Fits / maximum steps |
|---|---|---:|
| c1 | Does freezing the physical frame remove RAW/MIXED differences? | 12 / 1,000 |
| c2-c3 | Do Adam/SGD preserve equivalence after latent rotation? | 24 / 1,000 |
| sgd-duration | Is unexpected SGD rollout growth transient at the unchanged rate? | 6 / 5,000 |
| alternate-rotation | Does reduced OPF sensitivity recur on another fixed rotation? | 12 / 1,000 |
| gram-controls | Is it Gram pressure, or the choice of output coordinates? | 12 / 1,000 |
| fixed-alternate | Does the fixed-output-coordinate explanation recur? | 6 / 1,000 |
| qr-native | Does the earlier QR pathology remain with an isometric encoder? | 3 / 1,000 |

Total: **75 small fits, 99,000 optimizer updates**, about 48 seconds summed panel
wall time. Repeated native controls are included. Seeds remain 11/22/33 with the
original 48/16 trajectories and batch order. Rotation seeds 2718 and 31415 were
fixed before their respective panels. Most checkpoints are 0/1/25/100/250/500/1000;
the extended SGD panel uses 0/250/1000/2000/5000.

Adam uses the existing rate .003 and default moments/epsilon. Vanilla SGD uses a
fixed .03, no momentum, chosen before outcomes and unchanged in the extension.
Neither has weight decay. Within each optimizer, rates match across models/frames.
These are not tuned rates or a convergence-matched optimizer performance contest.

## Prediction versus coordinate sensitivity

Physical h16 MSE at 1,000 steps, mean ± sample SD over three seeds:

| Optimizer/model | Native | Rotation 2718 | Rotation 31415 |
|---|---:|---:|---:|
| Adam standard | .01903 ± .00107 | .02217 ± .00193 | .02343 ± .00171 |
| Adam soft OPF | .01815 ± .00160 | .01996 ± .00350 | .01839 ± .00219 |
| Adam fixed basis | .01903 ± .00107 | .02089 ± .00492 | .01950 ± .00262 |
| Adam Gram-free OPF | .01942 ± .00307 | .01979 ± .00445 | not run |
| SGD standard | 64.49 ± 100.9 | same to roundoff | not run |
| SGD soft OPF | 26.87 ± 40.5 | same to roundoff | not run |

Native Adam standard/OPF one-step errors are .0002637/.0002422; pulse-response h16
errors are .001027/.0009917. These are modest differences, not discovery evidence.
The JSON retains every horizon and pulse metric per seed, with sample SDs in the
generated summaries.

The paired sensitivity metric below is MSE **between native and rotated physical
predictions**, not prediction error against truth. Fixed probes use every test
trajectory at t=0 and t=8, recursively continued. Parameter comparisons also map
weights back into a common frame; raw matrix entries are not treated as evidence.

| Adam model | Paired h16 function MSE, rotation 2718 | Rotation 31415 |
|---|---:|---:|
| Standard | .01439 ± .00227 | .01748 ± .00297 |
| Learned soft OPF | .006154 ± .00242 | .005933 ± .00181 |
| Fixed counter-rotated basis | .005778 ± .00152 | .006623 ± .00163 |
| Gram-free OPF | .009399 ± .00555 | not run |

Both soft and fixed OPF have smaller function differences than standard for each
of the six seed/rotation pairs. SGD h16 paired MSE is only ~1e-27 to 1e-25; its
maximum absolute difference is 3.41e-12, even including large unstable predictions.
One-step differences and common-frame training trajectories remain near roundoff.

The fixed control uses core OPF with frozen B=R.T, so heads see the same output
targets across frames. Native fixed OPF is equivalent to standard JEPA. Rotated
fixed OPF rotates input weights but keeps its learned output weights in canonical
coordinates; standard rotates both. This changes the parameter axes receiving
elementwise Adam updates without adding parameters or changing the function class.
It reproduces nearly the full sensitivity reduction. Learned factors, 36 extra
entries and Gram pressure are therefore **not necessary** for most of that effect
in this instrument. This does not isolate every interaction in learned OPF.

## Gram, conditioning and unexpected failures

At 250, soft/Gram-free OPF h16 errors are .164/.354 native and .199/.549 rotated.
At 1,000 they are .0182/.0194 and .0200/.0198. No coefficient search was performed.
Removing Gram changes transient convergence and permits a less orthogonal, more
correlated map, without causing catastrophic synthesis failure here. At 1,000,
soft-map conditions are 1.0012–1.0024; Gram-free values are 1.44–2.48. Soft maximum
cross correlations span .162–.398 across both rotation panels, versus .357–.616
for Gram-free in its panel. Coordinates remain active. Strict orthogonality audit
failures and accurate pseudoinverse reconstruction are retained as diagnostics.

SGD's coordinate equivalence does not protect against poor recursive predictions.
Seed 22 dominates the 1,000-step error. Continuing all three native seeds at the
unchanged rate reduces standard/OPF mean h16 MSE from 64.49/26.87 to .4536/.5034 at
5,000, and one-step MSE from .01147/.01807 to .002025/.002135. These are still
imperfect endpoints, not established convergence.

Post-hoc inspection reconstructs saved seed-22 models without further fitting.
For standard at 1,000, the worst test rollout starts at norm 2.07 (maximum training
norm 3.12) and reaches norm 160.24. Its local physical Jacobian spectral radii are
1.39 at that start and 1.37 at the predicted endpoint; the true transition radius
is .99850. At 5,000 the worst predicted norm is 7.41. OPF shows an analogous
116.25 -> 7.87 reduction. Jacobians alone are not global stability proofs, but the
actual trajectories locate recursive expansion in the predictor. These selected
origins are descriptive failure analysis, not fresh holdouts.

The old learned-encoder QR seed-33 h16 error at 250 was 5661.89. The native frozen
QR control gives .2433 at 250 and .01986 at 1,000, with encoder and basis conditions
1. Removing learned encoder conditioning eliminates that observed pathology here,
but does not eliminate Adam coordinate sensitivity or every rollout instability.

## Debugging/research map

| Observation | Candidate explanations and tests | Weakened / ruled out | Still plausible; confidence and limits |
|---|---|---|---|
| Frozen RAW/MIXED discrepancy disappears | C0 independent sensor paths; C1 matched training | Changed dynamics, information loss, sensor numerical error here | Earlier learned initialization/conditioning/update paths matter; strong evidence for these fixed paths |
| Adam changes physical functions under latent rotation | Matched first/output/basis maps; common-frame trajectories | Encoder learning is not required; literal matrix differences are not the evidence | Elementwise adaptive updates depend on parameter coordinates; strong controlled evidence, two rotations |
| SGD remains equivalent | Same transformed initialization, batches and Euclidean parameter maps | Rotation sensitivity is not inevitable in this objective | SGD respects these symmetries; no implication of convergence or quality |
| OPF appears less sensitive | Three seeds, alternate rotation, fixed-basis control | Learned factorization, extra entries and Gram are unnecessary for most of this effect | Output parameterization explains much of it; other problems/interactions remain untested |
| Gram helps early, small late gaps | Change only Gram weight; inspect traces and synthesis | Universal Gram-based prediction advantage unsupported | Regularization alters convergence/map conditioning; exposed small panel, no significance claims |
| Frozen QR avoids old anomaly; SGD can still fail | QR control, longer SGD, saved-model Jacobians/rollouts | Encoder conditioning does not explain every instability | Predictor underfitting and expansive extrapolation remain; stable neural dynamics were not imposed |

No simulator, inversion, synthesis, split or rollout bug was found. The newly
isolated confound is attributing a choice of output parameter coordinates to
learned factor structure. An initial serialization test compared core-report tuples
with JSON lists; only that test assertion was corrected before the panels. No
scientific outcome was altered or discarded.

The coordinate/optimizer question is understood sufficiently to stop. Phase B is
cleaner to formulate: ask whether a factorized predictive objective adds held-out
prediction/rollout benefit in a harder system after controlling encoder conditioning,
output parameterization and training adequacy. Include the fixed-basis control.
No harder system was implemented, and no physical discovery is claimed.

```text
coordinate-sensitive optimization != different underlying dynamics
optimizer preference != dynamically privileged representation
orthogonal factorization != physical factorization
lower rollout error != physical discovery
coordinate-invariant predictor != unique representation
frozen isometric encoder != realistic observation process
```

## Files, commands and verification

Added `frozen_encoder.py`, `tests/test_frozen_encoder.py`, this note, and a README
entry. Previous experiment/core implementations were not changed. The output root
contains read-only `analyze_session.py`, `analysis.json`, and `mechanisms.png`.
Analysis verifies every panel digest/executed source, finite results, and all hashes
in the protected prior artifact directories.

Exact panel commands (repository root):

```bash
.venv/bin/python -m experiments.coupled_oscillators.frozen_encoder --panel physical --output-dir work/coupled_oscillators/frozen_encoder/c1
.venv/bin/python -m experiments.coupled_oscillators.frozen_encoder --panel rotation --output-dir work/coupled_oscillators/frozen_encoder/c2-c3
.venv/bin/python -m experiments.coupled_oscillators.frozen_encoder --panel sgd_duration --output-dir work/coupled_oscillators/frozen_encoder/sgd-duration
.venv/bin/python -m experiments.coupled_oscillators.frozen_encoder --panel rotation_adam --rotation-seed 31415 --output-dir work/coupled_oscillators/frozen_encoder/alternate-rotation
.venv/bin/python -m experiments.coupled_oscillators.frozen_encoder --panel gram --output-dir work/coupled_oscillators/frozen_encoder/gram-controls
.venv/bin/python -m experiments.coupled_oscillators.frozen_encoder --panel fixed_alternate --rotation-seed 31415 --output-dir work/coupled_oscillators/frozen_encoder/fixed-alternate
.venv/bin/python -m experiments.coupled_oscillators.frozen_encoder --panel qr_native --output-dir work/coupled_oscillators/frozen_encoder/qr-native
MPLCONFIGDIR=/private/tmp/coupled-oscillators-mpl .venv/bin/python work/coupled_oscillators/frozen_encoder/analyze_session.py
```

Validation commands:

```bash
.venv/bin/python -m pytest experiments/coupled_oscillators/tests -q
.venv/bin/python -m pytest experiments/coupled_oscillators/tests/test_frozen_encoder.py -q
.venv/bin/python -m ruff check --config jepa-anything-core/pyproject.toml experiments/coupled_oscillators
.venv/bin/python -m ruff format --config jepa-anything-core/pyproject.toml --check experiments/coupled_oscillators
make check PYTHON=.venv/bin/python
git diff --check
```

**66 core tests, 33 task-design tests plus 2 subtests, and 21 oscillator tests
(8 new) passed.** Lint/format, structural/design/manifest and whitespace checks
pass. New tests cover isometry, sensor equivalence, inverse mapping, matched
initial functions, frozen buffers, paired SGD equivalence and finite JSON. They
do not assert scientific rankings.
