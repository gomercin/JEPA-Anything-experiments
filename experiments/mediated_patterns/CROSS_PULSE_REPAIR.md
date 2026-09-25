# Short-gap response repair — exploratory findings

The repaired, fixed-size model passes the unchanged response gates on all 63
fresh sequences from nine preparations using three new seeds. It retains the
original ten response modes and adds two decaying summaries of commanded inputs.
Better single-pulse approximation and a separation-dependent cross-pulse event
correction are both needed. It never refreshes its state from the fields.

This extends a specified exterior-response interface. It does not identify a
physical memory variable, predict arbitrary fields, or certify the receiver's
mediator-specific contribution. The first repair failed a fresh triple panel;
that failure and all original artifacts remain available.

![Fresh response predictions](../../work/mediated_patterns/cross_pulse_repair/analysis-03/fresh_sequences.png)

The examples above include the final candidate's worst reversal and worst
triple by whole-trace NRMSE. Black is the independent field reference. Vertical
lines mark inputs. The lower panels expose the small early signal and its
approximation artifacts, rather than averaging them into late accuracy.

## Fixed task, provenance, and gates

Base revision: `c6e6c88f3ef75a4ce7acd660d6fa5779d995512c`, with pre-existing local
experiment work. Each outcome-bearing panel saved its own source snapshot,
source hashes, command, and protocol before execution. Original reports and
frozen models were not edited. This note supplements `RECURSIVE_RESPONSE.md`.

The equations, float64 Fourier ETDRK4 solver, preparation, pulse support, and
sensor are unchanged:

```
u_t = r*u - (1 + d_xx)^2*u + b*u^3 - u^5 + g*m*u
tau*m_t = D*d_xx*m - m + s*u^2
r=-.67, b=2, g=.05, s=1, D=64, tau=10
periodic L=128, N=512, dt=.0125
```

Two independently relaxed patches are assembled at nominal anchors ±d/2,
with seeded residual .02, then freely evolve for age 20 (odd seeds) or age 60
(even seeds). A commanded pulse is `u <- (1+a*w)*u`, where the smooth compact
window has radius 8 at the fixed left anchor. It changes no mediator values
instantaneously. The smooth radius 2 exterior mediator sensor stays at the
right anchor+12. There is no future centroid tracking.

The predictor gets one initial field scan for measured separation, static
coefficients, a clock, and commanded amplitudes/times. It predicts the signed
exterior pulse-minus-own-sham response through time 100, sampled every .25.
The sham, current fields, evolving separation, and actual microscopic pulse
increments are evaluator-only. No reset, output assimilation, or field refresh
occurs between inputs. Pulses must arrive on the .25 model grid. One event per
tick is supported; there is no unbounded internal event list.

The original gates are copied verbatim into both new freezes:

- Whole-trace and after-last-pulse NRMSE each at most 5%, if the corresponding
  response RMS exceeds the fixed `1e-7` resolution floor.
- Maximum absolute error at most`2e-6` in those windows.
- First time 0–5 maximum absolute error at most`2e-8`.
- Below the floor, relative accuracy remains unresolved. Historical terminal
  horizon gates are separate and have not been redefined.

## What caused or amplified the failure?

**Observation.** At seed 411/separation 29, same-sign pulses have a larger
absolute error than opposite signs but a lower response-normalized error.
Cancellation is a real amplifier. It does not explain all the error.

**Competing explanations.** Small denominators; nonlinear cross-pulse response;
single-pulse/kernel fitting or preparation age; finite-state realization error;
event semantics; numerical cancellation between four trajectories.

**Tests.** Four matched runs from exactly the same initial fields used both event
boundaries even when a pulse had zero amplitude. The second-only pulse occurred
at its original absolute time. The following identity held to zero reported
floating-point residual:

```
r12 - recursive = c12 + (r1 + r2 - kernel) + (kernel - recursive)
c12 = y12 - y1 - y2 + y0
```

| Pulses / gap | Response RMS | Cross RMS | Single-fit RMS | Realization RMS | Total error RMS | Original NRMSE |
|---|---:|---:|---:|---:|---:|---:|
| -.075,+.075 /3 | 3.04e-6 | 7.74e-8 | 1.05e-7 | 2.69e-9 | 1.79e-7 | 5.88% |
| +.075,-.075 /3 | 3.09e-6 | 1.03e-7 | 1.06e-7 | 5.99e-9 | 2.07e-7 | 6.72% |
| +.075,+.075 /3 | 1.51e-5 | 2.53e-8 | 2.85e-7 | 7.07e-8 | 3.10e-7 | 2.05% |
| -.075,+.075 /15 | 7.87e-6 | 1.15e-9 | 1.88e-7 | 1.22e-8 | 1.90e-7 | 2.42% |

For gap 3 reversals, the combined response RMS is only 17.0–17.3% of the sum
of individual RMS values. Against the separately frozen development single-pulse
scale `1.59105e-5`, errors are 1.12% and 1.30%; same-sign error is 1.95%. These
are supplemental diagnostics: the original failed gates remain failures.

The signed cross and single-fit errors reinforce one another in the reversal
cases. Their RMS values are **not additive contributions**. The complete
mean-product matrices, signed traces, pre/post-second-input metrics, and identity
checks are in `analysis-03/summary.json` and `diagnosis-01/results.json`.

![Matched quartet error decomposition](../../work/mediated_patterns/cross_pulse_repair/analysis-03/error_decomposition.png)

**Current conclusion.** Cancellation, resolved non-superposition, and imperfect
single-response fitting all matter. The original ten-mode realization is not
the main short-gap limitation. Increasing its order was not the selected repair.
The cross response is task-specific nonlinear response, not an identified
mediator-memory measurement.

## Pulse semantics and numerical qualification

The implemented pulse satisfies, without intervening evolution,
`J_b(J_a(u)) = [1+(a+b)w+ab*w^2]u`. Opposite amplitudes are not inverse
operations. The composition identity error was`2.22e-16`; ±.1 pulses left a
maximum u difference`0.01321`. A scalar inverse amplitude also fails for the
nonuniform window. These checks use no long simulation.

An explicitly changed-input diagnostic replaced only the second jump by an
additive increment computed from that time's sham field. This privileged
reference was used only in the evaluator. At gap 3, cross RMS increased from
`7.74e-8` to`1.31e-7` for negative-positive order and from`1.03e-7` to`1.56e-7`
for positive-negative order. Thus pulse composition alone does not explain the
exterior error; nonlinear evolution remains important. This control does not
uniquely partition pulse and dynamics contributions, and did not replace the
primary task.

Both gap 3 reversal quartets were refined together at dt/2, N1024 with dt/2,
and L256/N1024 with dt/2. A new ±.1 gap 2 quartet and close triple received the
same targeted checks after the final freeze. For the new quartet:

| Change | Max difference in c 12 | Conservative sum of three paired-response differences | Pair response max difference | Triple response max difference |
|---|---:|---:|---:|---:|
| dt/2 | 1.45e-13 | 2.47e-11 | 2.72e-12 | 7.62e-12 |
| N1024,dt/2 | 3.84e-12 | 2.37e-10 | 4.80e-11 | 1.51e-11 |
| L256,N1024,dt/2 | 1.57e-9 | 1.10e-7 | 1.75e-8 | 6.39e-9 |

There is a qualification nuance: summing errors of the four *raw sensor*
traces produces a much looser bound, up to`3.74e-5` for domain change, because
background changes are common to the quartet. We retain that adverse bound,
then compare each pulse-minus-own-sham response and conservatively sum those
three differences. At fixed domain this bound resolves c 12 by a large margin.
The domain bound is less decisive despite the small observed change in c 12.
These are targeted convergence diagnostics, not rigorous PDE error bounds or
a claim that boundary effects vanish. No scientific threshold was widened.

## Repairs attempted, including failures

1. **Existing-state matrix event correction:** added `a*M*z` with 100 fitted
   coefficients, keeping the baseline first pulse and unforced evolution.
   It improved fitted short cases but failed exposed regression: worst primary
  13.32%, old triple 20.42%, and short 12.37%. An unconstrained fit to accessible
   response-state directions generalized poorly. This failure is retained in
   `event-linear-01` and `event-linear-regression-01`.

2. **Geometry-independent cross injection:** two decaying command summaries,
   four mixed-product gains and two decay times; original A/B/C unchanged.
   It differs from the previously failed effective-amplitude susceptibility
   correction: it injects a cross-only response, without squaring an altered
   amplitude and accidentally adding other terms. It passed all 27 fresh short
   reversals and all 27 regression sequences, but failed all 9 close triples
   (worst 10.52%). That complete first fresh set, seeds 741/852/963, became
   exposed development evidence; it was not reused as final validation.

3. **One targeted follow-up:** eight-run triple inclusion-exclusion at exposed
   separations 27 and 25 separated single, pair, and third-order effects. At 27,
   single-fit error was`1.48e-7` RMS. At 25, true pair-sum RMS was`8.01e-8`, but
   the candidate injected`4.08e-7`. Third-order RMS was smaller (`5.81e-9`
   and`1.89e-8`), though not zero. Cross-response gain needed geometry dependence;
   simply adding a generic third-input memory was unsupported. A cheap cubic-only
   amplitude scratch check also did not resolve the exposed triple behavior;
   it was not selected. Its exploratory summary, rather than a frozen fitted
   model, is retained here.

The selected repair improves the single-pulse fit across amplitude and geometry
and conditions the cross correction on the already-retained initial separation.
It uses 36 single-pulse traces (15 reused,21 new) at four amplitudes
`[-.1,-.05,.05,.1]`, at nominal separations 24 through 32. Quartic amplitude
coefficients are projected onto the **same** ten-mode A/C realization. A cubic
spline between nine measured separation knots is evaluated only at initialization.
This is a small stored geometry table, not ten independently fitted horizon maps.

The cross fit uses 25 matched quartets plus six pair contrasts extracted from
the two exposed triple diagnostics. It does not fit the triple target. Two
decay times are fitted from one fixed initialization with bounds[.25,10],
not a grid search; the resulting times are 1.1521 and .9520. Their physical
interpretation is unspecified. This successful repair demonstrates a useful
approximation; it does not prove that the old state lacked information.

## What the evolving model retains

Let `z` contain the ten original response modes, `w1,w2` be command summaries,
and `d` the fixed initial measured separation. At an input of amplitude a:

```
c = beta(d) dot [a*w1, a^2*w1, a*w2, a^2*w2]
w1 <- w1+a; w2 <- w2+a^2
retain pending a,c until the next .25 tick
z_next = A*z + B(d)*[a,a^2,a^3,a^4] + B(d)[:,0]*c
w_i,next = exp(-.25/tau_i)*w_i
response = C*z
```

The event does not instantaneously change the exterior output. With no pending
input, z advances by the unchanged A. Cross injection is zero at the first
event from the zero response state. The improved B changes the first-pulse
approximation explicitly. Twelve cross gains use three supplied separation
features: a constant and the existing oscillatory SH-tail/mediator-scale ratios.
They are static geometry functions, not observations of future fields.

| Resource | Original recurrence | Selected repair |
|---|---:|---:|
| Response variables | 10 | 10 |
| Extra evolving command summaries | 0 | 2 |
| Pending input scalars | 2 | 2 |
| Clock; fixed initial separation | 1;1 | 1;1 |
| Independent numeric model coefficients / knots | 210 | 493 |
| Cached initialization matrix | included in 210 | 40 additional values |
| Internal growing pulse-history buffer | none | none |

The 493 count is A100+C10+B-knot 360+9 knot locations+12 gains+2 decay times.
The fixed .25 step and code constants are additional static settings. A temporary
spline calculation at initialization is not retained. The model JSON is 12,480
bytes; a live checkpoint is 12,780 bytes, including state and static tables.
The original class also carries an unused susceptibility slot and its constants.
These are not active original response variables. The full field state is 1,024
float64 scalars; storage reduction is modest once coefficients and caches count,
even though the evolving state is small.

The convenience batch rollout accepts a schedule; the actual event interface
needs only each input when it arrives. No unknown future input changes earlier
predictions. The original table kernel continues to replay past pulses and is
not credited as a fixed-size state model.

## Fresh validation under unchanged gates

The final model hash is
`4b94038cd857d3109f52a9dacacd48a8d17afb82cc4d61c383b240e327ba7d0a`.
It and the second protocol were frozen before seeds 1101/1202/1303 were run.
Nine whole preparations used intermediate nominal separations 25.5/27.5/30.5.
Seven schedules per preparation are related cases, not 63 independent preparations.

- Reversals: `(-.085,+.085)` at 0/2.75; `(+.095,-.095)` at 0/3.25;
  `(-.1,+.1)` at 0/2.
- Successful-regime regression: the original 6/15/30 waits and amplitudes.
- Withheld close triple: `(+.08,-.09,+.06)` at 0/2.25/7. Both gaps are shorter
  than the first failed triple's 0/2.5/7.5; inputs were not shrunk to obtain a pass.

Each cell below is pooled NRMSE / worst run NRMSE / runs passing **all** gates.

| Model | Short reversals,27 | Wait 6/15/30 regression,27 | Withheld triple,9 |
|---|---|---|---|
| Unchanged kernel | 5.79% /10.35% /9 | 1.30% /1.97% /27 | 6.06% /9.86% /3 |
| Unchanged recurrence | 5.79% /10.35% /9 | 1.31% /1.97% /27 | 6.17% /10.11% /3 |
| Better single fit only | 5.01% /9.70% /18 | .634% /1.225% /27 | 10.58% /16.73% /0 |
| **Selected repair** | **1.36% /2.10% /27** | **.623% /1.205% /27** | **3.72% /4.56% /9** |

The single-fit-only coefficients were saved before final evaluation. Their
post-evaluation ablation is diagnostic, not a separately selected validation
winner. It shows that improving the single fit alone is insufficient.

Worst selected-model whole-trace NRMSE per preparation:

| Seed / separation | Short | Regression | Triple |
|---|---:|---:|---:|
| 1101 /25.5 | 2.100% | .631% | 3.761% |
| 1101 /27.5 | .761% | .237% | 3.290% |
| 1101 /30.5 | .907% | .121% | 1.058% |
| 1202 /25.5 | 2.041% | 1.205% | 4.564% |
| 1202 /27.5 | .726% | .335% | 3.453% |
| 1202 /30.5 | 1.016% | .288% | .939% |
| 1303 /25.5 | 2.073% | .900% | 4.156% |
| 1303 /27.5 | .715% | .455% | 3.572% |
| 1303 /30.5 | 1.050% | .349% | .928% |

All 27 reversal traces improve against both original comparators. Against the
original recurrence,24/27 regression and 7/9 triple traces improve. Three
regression and two triple traces become somewhat worse but remain within their
original gates; no claim of per-run uniform improvement is made. After-last
worst scores are 2.100%,1.205%,4.564%, respectively. Worst absolute errors are
`3.56e-7`, `6.42e-7`, and`3.13e-7`. The old exposed 45-case panel also passes,
with worst primary .474%, old triple .639%, and old wait 3 challenge 2.712%.

A checkpoint immediately after the second input in the worst fresh triple
serialized only the reduced state and permitted static data. A fresh instance
continued through the third input with exactly zero difference from uninterrupted
and saved prediction. It passed against the independent field trace at 4.564%.
This is a self-containment and accuracy check for that sequence, not arbitrary
microscopic closure.

## Remaining qualification limits

**Early response:** first 0–5 signal RMS never exceeds`1.60e-8`, below the
unchanged`1e-7` floor. Maximum early error is`1.67e-8`, passing the absolute
gate. Relative early accuracy remains uncertified; the original early terminal
failure was not erased.

**Geometry and persistence:** maximum measured separation change over the fresh
100-unit runs is .0992, less than .4% of 25.5. Both measured patches remain;
mass ratios to each run's initial post-first-pulse measurement range .730–1.363
across input transients. Fixed initial geometry is a qualified approximation
for these sequences, not binding or exact stationarity.

**Pathway fidelity:** on the exposed seed 411/d 29 replay records, repaired total
error is .79–1.92% of the receiver-feedback contrast for the three regression
schedules,6.53% for the old wait 3 challenge, and 9.34% for the old triple. These
are encouraging scale comparisons on exposed evidence only. No reduced model
predicts the full-minus-replay counterfactual, so mediator-pathway preservation
is **not certified**. Replay arrays never enter prediction. The receiver-mass
regression failure remains a separate historical result.

**Envelope:** the fresh evidence covers the stated seeds, preparations,
intermediate separations, amplitudes, on-grid waits, and 100-unit horizon. It
does not certify every intervening wait, arbitrary long histories, pulses
closer than 2, off-grid inputs, arbitrary shapes, boundary-free behavior, or
third-participant back-reaction. The weakest accepted triple is fairly close
to the 5% gate. No further complexity was added after the second fresh panel.

## Cost, reproduction, files, and checks

Measured new simulation/preparation cost, including failed candidates' data,
both fresh panels, refinements, and the quick repeat:382.906 CPU seconds
(6.38 CPU-minutes),384.772 summed wall seconds. Recorded fitting/evaluation
inside the three fit commands: .980 CPU seconds,1.440 wall seconds. Additional
saved prediction timers for fresh/quick panels total 1.71 CPU seconds. Editing,
plotting, scratch analysis, and routine checks are separate; these are process
timers rather than the session's elapsed duration. CPU/float64 throughout.

Representative measured costs (100 repeats, host-specific): initial full-field
scan about .10ms; static interpolation about .054ms; autonomous 100-unit/400-step
forecast about 3.2ms; one retained-state tick about 7microseconds. Initial
extraction scans the 1,024-scalar field once. Training uses full simulation
traces; deployment does not. No formal speedup claim conflates the shared-sham
multi-run simulation timing with one standalone forecast.

Run from the repository root. These output names are intentionally new;
commands refuse to overwrite an existing directory.

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.mediated_patterns.cross_pulse quick --data work/mediated_patterns/cross_pulse_repair/frozen-02 --output work/mediated_patterns/cross_pulse_repair/quick-repeat
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.mediated_patterns.cross_pulse validate --data work/mediated_patterns/cross_pulse_repair/frozen-02 --output work/mediated_patterns/cross_pulse_repair/panel-repeat
```

The executed quick command used`quick-02`: about 7.1 CPU seconds, seven schedules
from one preparation. Fields and all predictions exactly matched the
corresponding frozen fresh records. Repeating the panel demonstrates
reproducibility, not another untouched validation set.

All exact scientific commands, including failed attempts, are recorded in
`work/mediated_patterns/cross_pulse_repair/analysis-03/commands.txt`. Each panel's
`protocol.json` additionally records literal argv and the source snapshot used.
The sequence is diagnose → refine → development → failed matrix fit/regression
→ cross fit/regression → pulse audit → freeze/first validation → two triple
diagnostics → single refinement → conditioned fit → second freeze/validation
→ numerical qualification → historical regression → quick → read-only analysis.
Analysis 02 is preserved; analysis 03 only repairs a long figure title.

New source files only:

- `event_response.py`: failed matrix candidate, selected command-summary model,
  and bounded fitting routines; no simulator imports.
- `cross_pulse.py`: matched controls, frozen protocols, qualification and runners.
- `cross_analysis.py`: read-only summaries, cost/access accounting, checkpoint
  demonstration and plots.
- `tests/test_cross_pulse.py`:10 correctness tests including pulse composition,
  residual identity, causality, no-field checkpointing, interpolation bounds,
  run separation, strict serialization and non-overwrite behavior.
- `CROSS_PULSE_REPAIR.md`: this report.

Checks executed:

```bash
make check PYTHON=.venv/bin/python
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest experiments/mediated_patterns/tests -q
.venv/bin/ruff check experiments/mediated_patterns/cross_pulse.py experiments/mediated_patterns/event_response.py experiments/mediated_patterns/cross_analysis.py experiments/mediated_patterns/tests/test_cross_pulse.py
.venv/bin/ruff format --check experiments/mediated_patterns/cross_pulse.py experiments/mediated_patterns/event_response.py experiments/mediated_patterns/cross_analysis.py experiments/mediated_patterns/tests/test_cross_pulse.py
git diff --check
```

Repository checks passed:66 core tests;33 skill tests plus 2 subtests;46
oscillator tests; design, recipe, manifest, JSON and compile checks. All 30 field
tests passed, including 10 new tests. Source lint/format and whitespace checks
passed. SHA256 verification found 877/877 protected pre-existing files unchanged.
No field laws, upstream code, historical report, original model, or earlier
result artifact was modified. No divergence was clipped or run omitted.

Machine-readable primary evidence: `fresh-02/results.json`, `fresh-02/summary.json`,
`frozen-02/model.json`, `frozen-02/freeze.json`, `qualification-02/refinement.json`,
and `analysis-03/summary.json`. The latter includes every paired run, preparation
summary, error-product matrix, access/resource accounting, and checkpoint check.

**Supported statement:** within this prepared family and 100-unit input contract,
an ordinary fixed-size recurrence can carry the effect of earlier inputs into
closely spaced later responses, with the original gates retained and no field
refresh. Better response fitting plus two decaying input summaries suffice on
the tested fresh sequences. A physical memory identification, universal state
sufficiency, and mediator-pathway closure remain unestablished.
