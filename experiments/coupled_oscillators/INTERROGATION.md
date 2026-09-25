# Phase A instrument interrogation — exploratory

This is exposed-seed debugging, not a preregistered benchmark or evidence for a
physical ontology. No Phase B dynamics or Atlas changes were made. The original
`work/coupled_oscillators/phase-a-quick/` files remain byte-identical. Competent
linear identification still solves this noiseless linear world: the original
RAW/MIXED one-step MSE means are 9.78e-31 / 2.13e-30, respectively. It was neither
weakened nor replaced by a neural comparator.

## What ran

One bounded panel in `work/coupled_oscillators/interrogation/panel-01/`:

- Seeds 11, 22, 33; the same 48/16 train/test trajectories, 64 transitions,
  dt=0.1, RAW/MIXED matrices, minibatches, Adam settings, EMA and readout as Phase A.
- Five variants × two conditions × three seeds = **30 fits**, each continued to
  1,000 steps. Checkpoints 0, 25, 100, 250, 500 and 1,000 are reporting points,
  not candidates for checkpoint selection. All states and per-seed metrics remain
  in JSON. No 5,000-step run or hyperparameter search was performed.
- **12 paired 25-step probes** (24 short model fits): standard/OPF × three seeds
  × Adam/SGD, each comparing RAW/MIXED under physically equivalent initialization.
  SGD uses the same untuned learning rate solely to check coordinate equivalence.
- **1,024 seeded orientation draws** (seed 314159), reused for null diagnostics.
  A complete Haar-orthogonal basis is a reference; a common right rotation of each
  actual physical analysis map is the more relevant null, preserving its singular
  values, inter-factor geometry and within-factor geometry.
- Total: 30,600 optimizer updates; panel wall time 13.67 s on this CPU. No neural
  training during post-run analysis or plotting. Every panel result is exploratory.

`standard` reuses the existing standard baseline. `output_transform` adds an
identity-initialized, bias-free 6×6 matrix after its predictor and retains standard
JEPA loss. It has 500 trainable parameters, matching OPF (versus 464 for standard),
plus the same separate 42-coefficient readout. This matrix is algebraically
absorbable into the existing output layer: it tests redundant transform parameters
and optimization, not increased functional expressiveness. OPF instead learns a
target analysis map and changes the factor-space error metric. Those roles are
not matched by a parameter-count control.

`opf_no_gram` changes only Gram weight from .05 to zero; factor activity remains.
`opf_qr` uses the core's QR retraction after every optimizer step, retaining the
other objective terms. As documented by the core, retraction does not transform
Adam's moments. Its behavior is not a general test of every orthogonal optimizer.

No frozen random/mode basis, normalization, or factor deletion panel was needed
to answer the live questions. Standard and multi-head are already algebraically
equivalent here, and every head consumes the whole latent state; head grouping
alone does not impose separate modal dynamics.

## Coordinate audit and optimizer probe

MIXED is already orthogonal: determinant -1.0000000000000004, rank 6, singular
values all 1 to roundoff, condition number 1.0000000000000004, maximum error in
`M.T M - I` 2.22e-16. Neither condition uses normalization. Across the three
training datasets, maximum per-sample norm difference is 8.88e-16. Covariance
eigenvalues and condition numbers also agree, not just total variance.

For seed 11, before/after coordinate variances are:

| Coordinate | RAW | MIXED |
|---|---:|---:|
| 1 | .4307 | .4068 |
| 2 | .3954 | .6151 |
| 3 | .3093 | .3736 |
| 4 | .5066 | .5165 |
| 5 | .6852 | .6427 |
| 6 | .6570 | .4296 |

Both covariance condition numbers are 2.619909; norm min/quartiles/max are
[.5096, 1.2716, 1.6527, 2.0077, 3.0795]. JSON contains the full audit for all seeds.
Mixing redistributes coordinate variances but does not improve full-data spectral
conditioning or scale. Normalization was not added to solve a nonexistent global
scale change; diagonal scaling could still change the optimizer's behavior.

Default Phase A shares *numerical* encoder initialization across conditions.
`W x` and `W M x` are different initial physical functions. For the diagnostic
probe, set both online and EMA MIXED weights to `W M.T`, making the initial
physical functions agree. At step 25, maximum prediction differences are:

| Optimizer | Range across standard/OPF and three seeds |
|---|---:|
| SGD | 1.11e-16 – 3.05e-16 |
| Adam | .0462 – .0670 |

This directly demonstrates coordinate sensitivity in elementwise Adam updates,
even after matching initial functions. It does **not** establish that Adam is
inferior or quantify how much of the full-run performance gap comes from Adam
versus initial orientation. At 1,000 steps the original aggregate RAW/MIXED gap
mostly vanishes anyway.

## Prediction and duration

Mean ± sample SD over the three paired seeds; lower is better. These are exposed,
small-sample comparisons, not significance claims. The full generated summary
also includes one-step and pulse-response errors, with every horizon in JSON.

| Condition | Variant | h16 MSE, 250 steps | h16 MSE, 1,000 steps |
|---|---|---:|---:|
| RAW | Standard | 1.093 ± .182 | .04748 ± .0382 |
| RAW | Extra output transform | 1.992 ± 1.30 | .08649 ± .109 |
| RAW | OPF soft Gram | .9676 ± .461 | .03713 ± .0251 |
| RAW | OPF no Gram | 1.456 ± .430 | .05060 ± .0472 |
| RAW | OPF QR | 1888 ± 3270 | .03525 ± .0170 |
| MIXED | Standard | 1.166 ± .934 | .03700 ± .0134 |
| MIXED | Extra output transform | 1.112 ± .846 | .03390 ± .0140 |
| MIXED | OPF soft Gram | .8763 ± .555 | .03434 ± .00657 |
| MIXED | OPF no Gram | 5.356 ± 3.32 | .1527 ± .174 |
| MIXED | OPF QR | 1.578 ± 1.06 | .03129 ± .00828 |

At 1,000 steps, MIXED one-step MSE is .0003074 ± .0000416 for standard versus
.0003064 ± .0000304 for soft OPF. Their pulse-response h16 errors are also close:
.001306 ± .0000711 versus .001329 ± .000211. A stable large OPF advantage is not
supported. More training is a much larger effect than the quick-run architecture
differences, though 1,000 steps is not established convergence.

Removing Gram gives the best MIXED one-step error at 250 (.006129) but a much
worse h16 error (5.356) and response error (.5524). One-step success cannot stand
in for rollout stability. The extra-transform control does not recreate OPF's
early RAW behavior, but is competitive on MIXED by 1,000 steps. Thus neither
"36 parameters explain everything" nor "only OPF can help" is established.

## Unexpected finding: the encoder, not just the OPF map, needs auditing

The target encoder is affine, and its train-only fitted readout reconstructs
targets essentially exactly. That does not prevent it from amplifying predictive
errors along weak encoder directions. We decompose held-out one-step residuals
using the target encoder SVD: each direction contributes latent residual energy
divided by squared singular value (and by 6) to observable MSE. The sum agrees
with the external metric; the maximum absolute residual over all checkpoints is
2.39e-12, including enormous early errors.

RAW seed 33 at step 250:

| Variant | Encoder condition | Observable / latent MSE | Error in weakest direction | h16 MSE |
|---|---:|---:|---:|---:|
| Standard | 37.29 | 244.3× | 98.98% | 1.125 |
| Soft OPF | 38.52 | 237.0× | 98.91% | 1.498 |
| QR OPF | 519.13 | 14918× | 99.98% | 5661.89 |

The QR basis is perfectly conditioned and passes orthogonality audits, yet the
encoder has a nearly lost direction. At step 1,000 its encoder condition is 3.26
and h16 error .0512. The adverse 250-step result is retained, including its large
effect on the across-seed mean. The plotted seed was selected **after inspection**
to explain this failure, not presented as a representative seed.

Coordinate-wise activity floors do not bound encoder singular values: all output
coordinates can vary while their joint representation is nearly redundant.
This is a limitation of the current objective/readout instrument, not evidence of
a broken simulator, mixed-coordinate inversion, core synthesis, or test split.
No implementation correctness bug was found in those paths. The conditioned
readout is exposing an actual predictive weakness, not simply a bad least-squares
fit. Very early rollouts can be much worse still; all checkpoint values are saved.

## Geometry, modes and representation stability

Soft OPF begins exactly orthogonal, moves away during learning, then becomes
closer: max entrywise orthogonality error .00209–.00487 at 250 versus
.000493–.000985 at 1,000. These remain above the unchanged float64 tolerance
1e-10. Its map condition is 1.0033–1.0081 at 250 and 1.0007–1.0016 at 1,000.
Transpose-synthesis NMSE shrinks from 3.78e-6–1.75e-5 to 3.01e-7–1.33e-6.
Operational pseudoinverse synthesis remains accurate. Strict audit failure is
not the same thing as unusable synthesis. QR passes all six basis audits at both
checkpoints, without preventing the encoder failure above.

All soft-OPF coordinates remain active. Cross-factor correlations are .599–.761
at 250 and .507–.697 at 1,000. Orthogonality does not imply statistical independence.

Mode stiffness eigenvalues [1.13613, 1.77533, 2.28854] are distinct. Comparisons
already allow arbitrary rotations within each position/velocity mode plane and
best one-to-one permutations. The full Haar reference has mean best-matched
overlap .4805 and approximately 5th/95th percentiles .401/.605. More appropriately,
randomly rotating each actual learned physical analysis map gives:

| Soft OPF checkpoint | Observed mode overlap range | Fraction of null at least observed |
|---|---:|---:|
| Initialization | .395–.476 | .295–.889 |
| 250 | .379–.438 | .642–.967 |
| 1,000 | .384–.512 | .203–.969 |

These are descriptive reference fractions, not calibrated scientific p-values.
There is no above-null alignment signal here. Prediction can improve greatly
without recovery of privileged physical subspaces.

After mapping MIXED rows back with `B W_target M`, best matched RAW/MIXED overlaps
at 1,000 are .451, .438, .434 (seeds 11/22/33). They are far from subspace identity
(1.0); no literal matrix-entry comparison was used. Different physical analysis
subspaces coexist with similar predictive scores. No general convergence toward
one coordinate-invariant modal decomposition is demonstrated.

## Debugging map and next discriminating experiment

| Observation | Strengthened explanation | Weakened / unresolved |
|---|---|---|
| Quick OPF differs from JEPA | Transient optimization and learned error metric matter | Large durable advantage; extra parameter count alone is insufficient |
| MIXED differs from RAW | Different physical initialization and Adam coordinate sensitivity | Global scale/information/conditioning change ruled out; their relative contributions remain unresolved |
| Low latent loss, bad physical prediction | Weak encoder singular directions amplify errors | Active coordinates and a good OPF map are insufficient safeguards |
| QR passes geometry but can predict badly | Geometry and predictive conditioning are separate | Exact basis orthogonality alone is insufficient |
| Mode overlap around .4 | Ordinary orientation under suitable nulls | Privileged dynamical factor discovery unsupported |

**One next experiment, not executed:** freeze a full-rank isometric encoder in a
common physical frame (identity for RAW, inverse mixing for MIXED), then compare
standard and soft-OPF prediction at the same adequate training budget. This
removes learned encoder conditioning and the arbitrary initial physical frame,
while retaining the difference in the predictor/OPF objective. It would clarify
whether any residual advantage survives once this identified weakness is removed.
It changes the representation-learning question, so its result must be labeled
accordingly. Stop here rather than tuning activity floors, adding a new panel,
or using nonlinearity to manufacture separation.

Phase B should eventually ask about predictive and rollout benefit after these
conditioning and optimization controls, not whether arbitrary active factors
"discover" physical modes. Useful predictive transforms, independent causes,
stable subspaces, and prediction under a pulse remain distinct claims.

## Files, artifacts and verification

Added `interrogate.py`, `tests/test_interrogate.py`, and this note; appended the
README entry. No Phase A implementation or upstream core code was changed.
Existing dirty Makefile/pytest/ignore changes belonged to the previous phase.

The panel directory contains `protocol.json`, `results.json`, `summary.md`,
`results.sha256`, post-run `analysis.json`, `conditioning.png`, and the read-only
`analyze_panel.py`. `interrogate.executed.py` preserves the exact executed runner;
the source subsequently received only a formatting fix around a string literal.
The report/README were written after outcomes and are not claimed as preregistered.

Panel result SHA-256:
`ad02c458a7438e2cf86d93d94045315469b8a3f169de42f2a7a25f29bff67963`.
Original result SHA-256 remains:
`ca08b0e1c3572278a8ac68c9cf6cac4b66d405eceea04882e5e9e51ab90d07bf`.
All four original artifact hashes are checked. Reproduced original standard/OPF
h16 metrics differ by exactly zero for all seeds and conditions.

Commands executed from the repository root (formatting/lint fixes preceded the
final passing checks):

```bash
.venv/bin/python -m pytest experiments/coupled_oscillators/tests -q
.venv/bin/python -m pytest experiments/coupled_oscillators/tests/test_interrogate.py -q
.venv/bin/python -m experiments.coupled_oscillators.interrogate --output-dir work/coupled_oscillators/interrogation/panel-01
MPLCONFIGDIR=/private/tmp/coupled-oscillators-mpl .venv/bin/python work/coupled_oscillators/interrogation/panel-01/analyze_panel.py
.venv/bin/python -m ruff check --config jepa-anything-core/pyproject.toml experiments/coupled_oscillators
.venv/bin/python -m ruff format --config jepa-anything-core/pyproject.toml --check experiments/coupled_oscillators
make check PYTHON=.venv/bin/python
git diff --check
```

Results: 66 core tests; 33 task-design tests plus 2 subtests; **13 oscillator tests**
(5 new). All pass, as do lint/format and existing structural/manifest checks.
Tests cover physical basis pullback, deterministic rank-aware nulls, exact
transform parameter counts and initial function, trace noninterference with
training, singular-direction error accounting, and SGD rotation equivalence.
No test requires a scientific performance ordering. The post-run plot script had
an iterator-consumption error in a min/max helper on its first attempt; that was
fixed without rerunning training or modifying any result artifact.
