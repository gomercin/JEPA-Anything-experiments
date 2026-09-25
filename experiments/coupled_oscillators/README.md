# Coupled oscillators: one Phase A integration experiment

The original Phase A protocol below remains unchanged. Subsequent exploratory
studies are documented in [INTERROGATION.md](INTERROGATION.md),
[FROZEN_ENCODER.md](FROZEN_ENCODER.md), [PHASE_B.md](PHASE_B.md), and
[PHASE_C_PARTIAL.md](PHASE_C_PARTIAL.md), and [PHASE_D_NOISE.md](PHASE_D_NOISE.md).
These studies use separate local modules
and preserve earlier artifacts. Ordinary cubic identification is sufficient in
the fully observed nonlinear regime; the partial-observation study finds that
two position samples support an accurate conventional six-dimensional delay state.
With noisy positions, identified mechanical dynamics plus causal filtering
outperform the tested neural models; uncertainty calibration remains imperfect.

This is a small instrument check of learned predictive representations in a
known linear system. It compares external prediction and recursive rollout
under RAW and MIXED observations. It does not establish architectural advantage,
factor semantics, or any broad theory. Matching or losing to ordinary linear
regression is a valid result; in this noiseless linear world, regression should
be close to numerical precision.

## Run

Use the repository's prepared environment, from its root:

```bash
source .venv/bin/activate
python experiments/coupled_oscillators/run_experiment.py --quick
python experiments/coupled_oscillators/run_experiment.py --full
python -m pytest experiments/coupled_oscillators/tests -q
make check
```

Only PyTorch and the installed core are needed at runtime. Tests additionally
use pytest. No downloaded data, plotting packages or accelerators are needed.
`--quick` is the development/CI-sized instrument panel, not a converged benchmark.
`--full` is a longer fixed budget, not a different scientific protocol. The first
deliverable executes quick only; full is available but not automatically run.

Generated files go to a new timestamped directory under the ignored
`work/coupled_oscillators/`. `--output-dir PATH` selects a **new** directory;
existing directories are rejected to preserve earlier outcomes. Outputs:

- `protocol.json`: configuration, matrices, seed roles, code hashes, git revision
  and working-tree status, command and environment, written **before training**.
- `results.json`: every seed/model/condition, train/test IDs and data hash,
  parameters, fit and training diagnostics, metrics, and aggregate mean/sample SD.
  Completed records are saved incrementally; `status=running` means incomplete.
- `summary.md`: compact prediction and OPF diagnostic tables.
- `results.sha256`: digest of the final result artifact.

The source hashes include local uncommitted experiment code and the actual core
sources; a git commit alone is not claimed to identify uncommitted code. Seeded
CPU float64 operations use one thread and deterministic PyTorch algorithms.
Exact last bits and timing may vary with PyTorch/platform versions. No model
checkpoint is required; fixed seeds, inputs, and source hashes support rerunning.

## Fixed protocol (declared before examining outcomes)

State is `[q1,q2,q3,p1,p2,p3]`, with unit masses and dimensionless coordinates:

```text
dq/dt = p
dp/dt = -S q - 0.03 p
S = diag(1.0, 1.4, 2.0) + 0.2 * [[1,-1,0],[-1,2,-1],[0,-1,1]]
dt = 0.1
```

Positive stiffness and positive damping make this stable. Sampling uses the
matrix exponential of the 6D generator, with no numerical integration step
error beyond floating-point roundoff. The JSON includes generator, transition,
stiffness eigenvalues, and the known 2D modal position/velocity planes.
Initial coordinates are independent Gaussian draws with SD 0.7. There is no
process noise, observation noise, normalization or augmentation.

| Setting | quick | full |
|---|---:|---:|
| Paired dataset/model seeds | 11, 22, 33 | 11, 22, 33 |
| Train/test trajectories per seed | 48 / 16 | 256 / 64 |
| Transitions per trajectory | 64 | 128 |
| Neural optimizer steps | 250 | 2000 |
| Minibatch size | 128 | 128 |

Each trajectory includes its initial state. Whole trajectory IDs are shuffled
using `seed+10000` and split before time-step pairs are formed. All methods and
both conditions share the same physical trajectories and split within a seed.
The three seeds vary data, initialization, and minibatches jointly; their SD is
not a confidence interval or an estimate from three datasets with a fixed model.
There is no validation tuning, early stopping, or selection: evaluate the final
fixed step. Test data enter only reporting. Minibatch sampling uses `seed+20000`.

RAW uses identity observations. MIXED uses `y = M x`, where `M` is a fixed
orthogonal (therefore invertible) signed-QR transform from seed 1729. It is reused
across all seeds and saved in JSON. Orthogonality additionally preserves Euclidean
MSE across the two conditions; coordinate scaling is not an extra confound.
The transform is never selected or tuned against outcomes.

## Four approaches and common readout

All neural methods use a learned affine 6→6 encoder and an EMA target encoder
(momentum 0.99). Width `d=6`, `K=3`, `r=2` is sufficient for the full physical
state; there is no latent expansion. A linear encoder makes the exploratory
physical-coordinate subspace comparison exact and auditable. It also means
this panel cannot test the benefits of a nonlinear observation encoder.

1. **Standard JEPA:** the core `StandardJEPABaseline` with latent MSE and the
   core online encoder variance penalty.
2. **Unconstrained multi-head:** the core `UnconstrainedMultiHeadJEPABaseline`,
   with three independent two-wide outputs and the same loss. The core's shared
   trunk plus output grouping is algebraically equivalent to standard JEPA.
   Encoder/trunk/output weights are copied to match initial functions. Their
   results should agree up to numerical effects; this is a useful control, not
   two structurally different model families.
3. **OPF:** the same core multi-head predictor and EMA lifecycle plus a learned
   core `OrthogonalFactorProjection`, initialized to identity in `soft_gram`
   mode. Training calls `jepa_anything_objective`: factor MSE + 0.05 projector
   Gram + 0.1 target-factor activity + 0.1 online encoder variance. Both activity
   floors are 0.1. Target encoder states are detached **before** OPF projection;
   the projector receives target-side prediction and activity gradients. Factor
   predictions are synthesized with the core Moore–Penrose inverse, never by
   treating flattened coordinates as the reconstructed state.
4. **Linear transition:** unrestricted affine state-transition least squares,
   fitted directly on all train observation pairs using float64 SVD (`gelsd`),
   with no regularization or intentional bottleneck. It never sees oracle modes.

Neural predictors come from `build_capacity_matched_predictors`: 6→32→6,
GELU, two affine layers, exactly **422 predictor parameters**. All three models
use Adam at 0.003, default betas/epsilon, no weight decay, and the same batches.
The standard/multi-head models have **464 trainable neural parameters**; OPF
has **500**, including its 36 basis entries. Each also has 42 frozen EMA entries
and a separate 42-coefficient fitted readout. Linear transition has 42 fitted
coefficients. The core capacity report covers predictors only, with tolerance
zero. Full model capacity is **not** equal, and FLOPs are unmeasured/null; no
compute-equivalence claim is made. This compares OPF's objective package, not
the isolated causal contribution of its Gram term.

After neural training, fit an affine map from the **frozen target encoder's
train target states** to their observations using the same SVD least squares.
This supervised readout never updates the representation and never uses test
data. Save its rank, singular values, condition number, and train reconstruction
MSE. This distinguishes latent predictive learning from observable evaluation.
The readout may be ill-conditioned if a representation collapses; retain those
diagnostics rather than silently replacing it with an oracle inverse.

At inference, predict target latent state, decode to observations, and re-encode
the prediction for the next step. This is an observation-feedback rollout, not
teacher forcing or uninterrupted latent-space iteration.

## Metrics, perturbation and diagnostics

Report **endpoint** MSE at horizons 1, 4, 8 and 16. Each averages all six
observable coordinates, test trajectories, and the same origins `t=0..T-16`.
`one_step_mse` is the horizon-1 entry. Equal trajectory lengths give every
trajectory equal weight. Means and sample SD (`ddof=1`) are across the three
paired seeds; all individual values remain available.

The held-out perturbation is fixed: at time index **8** in **every test
trajectory**, add **+0.2 to p1**, then simulate 16 more steps. Train trajectories
contain no pulses. Models receive the post-pulse observed state. At every
horizon report both post-pulse trajectory MSE and response MSE: predicted
`pulsed - unpulsed` trajectory minus the simulator's corresponding difference.
The second metric prevents ordinary trajectory error from concealing response
error. In the noiseless linear setting, this is a simple superposition check;
it is prediction under a declared simulator intervention, not causal discovery.

The test-set core OPF audit records basis rank, conditioning, Gram/completeness
errors, pseudoinverse round-trip and direct transpose-synthesis errors,
coordinate-wise activity, and cross-factor correlations. Core dtype-aware
tolerances are unchanged; correlation threshold is 0.2. Soft Gram training need
not meet the strict orthonormality tolerance. Audit `passed=false` is preserved
and does not mean the execution failed. For the JEPA baselines, anonymous latent
groups receive activity/correlation diagnostics, without an orthogonality claim.

For **Phase A only**, map each OPF analysis block back to physical coordinates
as `B_k W_target M`. Compare its row span with every known 2D mode plane using
principal angles and projection overlap `||U_learned.T U_mode||_F² / 2`. Record
the full 3×3 matrices, numerical ranks, and the best one-to-one permutation
(enumerating six possibilities). Internal rotations, sign flips and permutations
do not count as failures. Rank loss is recorded and angles are null for deficient
blocks, rather than completing a spurious 2D basis. The complete 6D span is never
used as evidence of alignment. These are analysis sensitivities in the declared
Euclidean `[q,p]` metric, not reconstructed physical factor components.
Best-permutation overlap is exploratory and optimistically matched, with no
calibrated significance threshold or random-subspace reference. Describe it as
degree of learned predictive subspace alignment with known dynamical subspaces,
never “factor 1 discovered mode X.”

## Limits and stopping point

```text
a useful representation != a privileged physical ontology
orthogonal learned factors != statistically independent causes
good prediction after an intervention != causal discovery
MIXED coordinates != changed dynamics
PARTIAL observation != mere coordinate change
```

PARTIAL was intentionally absent from Phase A: it required a separately declared
sensor history, predictor inputs and fairness protocol. That separate study is
now documented in [PHASE_C_PARTIAL.md](PHASE_C_PARTIAL.md). Phase B was deferred at the
original Phase A stopping point. The subsequently authorized
[Duffing study](PHASE_B.md) uses the same `simulate(initial, steps)` /
`trajectories(...)` interface in an experiment-local subclass, qualifies integration
before model comparisons, and stores separate exploratory results. The original
Phase A runner still runs only its declared linear experiment.

Tests check determinism, stable flow, trajectory separation, observation
invertibility and equal-information transforms, shapes/factorization, live OPF
gradients and geometry audits, oracle rollout/pulse scoring, subspace invariance,
and an actual three-seed quick CLI run producing finite metrics and JSON. They
never require OPF to beat another method.

## Exploratory Phase A interrogation

The separate [interrogation note](INTERROGATION.md) records exposed-seed debugging
of training duration, encoder conditioning, coordinate sensitivity, extra transform
parameters, orthogonality enforcement, and mode-alignment nulls. This is exploratory,
not a replacement for the original Phase A results or a preregistered benchmark.
Run its bounded panel with a new output directory:

```bash
python -m experiments.coupled_oscillators.interrogate \
  --output-dir work/coupled_oscillators/interrogation/NEW_RUN
```

It reuses the original dataset, core model/loss/audit APIs and external evaluator.
It records fixed checkpoints and verifies that the saved `phase-a-quick` artifacts
are unchanged. It does not select a winning checkpoint or start Phase B.

The subsequent [frozen-encoder note](FROZEN_ENCODER.md) isolates sensor equivalence,
latent rotation, optimizer behavior and output parameterization using a fixed
isometric physical frame. Its small named panels are invoked separately with
`python -m experiments.coupled_oscillators.frozen_encoder --panel physical --output-dir NEW_PATH`.
All outputs are exploratory and separate from earlier runs. The note records
successive controls, exact commands, adverse SGD rollout behavior and the
fixed-basis explanation for much of the apparent OPF coordinate-sensitivity benefit.
