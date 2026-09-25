# A self-contained exterior-response model

Exploratory session, 2026-09-22. A ten-variable linear response realization,
with a quadratic pulse input map, carries previous responses into subsequent
inputs without reopening fields. On fresh preparations it predicts the specified
exterior signal with 2.06% pooled trace NRMSE for two pulses and 1.66% for one
withheld three-pulse schedule. Closely spaced sign reversals remain a tested
limitation. This is a perturbation-response model of this prepared family,
not microscopic closure, binding, or a model of arbitrary arrangements.

The representative figure is
`work/mediated_patterns/recursive_response/report-02/response_sequences.png`.
It includes signed full-field, kernel and recursive predictions, pulse times,
residuals, the receiver-replay contrast, and a separately magnified early window.

## Contract and preserved evidence

All earlier files and artifacts were retained. The pre-session SHA-256 inventory
contains 663 files; all remained byte-identical. No original source, core code,
Atlas record, field law, historical result, or historical gate was changed.
The checkout's original `.gitignore`, `Makefile`, and `pyproject.toml` changes
were already present. Base revision is
`c6e6c88f3ef75a4ce7acd660d6fa5779d995512c`; experiment files are local/untracked.
Each new panel saves the executed source snapshot and hashes, command, and
configuration. Later edits add reporting, input validation, and formatting;
they do not refit the frozen model. Source snapshots preserve execution provenance.

The existing equations remain

```
u_t = r u - (1 + d_xx)^2 u + b u^3 - u^5 + g m u
tau m_t = D m_xx - m + s u^2
```

Here `r=-.67, b=2, g=.05, s=1, D=64, tau=10`, periodic `L=128`,
`N=512`, `dt=.0125`, float64 Fourier ETDRK4. The SH35 base-model references and
the status of our mediator extension remain as documented in README/FINDINGS.
No new physical mechanism is borrowed or claimed in this session.

The event is exactly the existing relative-amplitude operator:

```
u(x,t+) = u(x,t-) * (1 + a b(x-x_A)),  m(x,t+) = m(x,t-)
b(d) = exp(1 - 1/(1-(d/8)^2)) if |d|<8; otherwise 0.
```

It depends on instantaneous `u`. A declared amplitude is **not** an additive
field increment. The reduced model receives only the declared amplitude/time;
its polynomial event response approximates this physical operation in the
tested regime. No field-derived future pulse increment is supplied.

Anchors stay at nominal `x_A=-d/2, x_B=d/2`. The unchanged smooth mediator
sensor is centered at `x_B+12`, radius 2. Neither pulse nor sensor follows a
future centroid. Targets are signed sensor values minus each preparation's own
no-pulse evolution. Only the evaluator runs that sham; the predictor receives
neither its trajectory nor the background signal. It predicts deviations only.

Initialization permits one field scan to extract the existing weighted-centroid
separation. That scalar is then held fixed. Other recorded masses, widths,
centers, full/sham readouts, and replay data are evaluator diagnostics, never
model inputs. There are no later state refreshes, resets after pulses, true
separation updates, teacher forcing, or time-varying reference replay.

## Models and development decisions

The kernel is fitted to **interacting-pair** single-pulse traces, not the old
independent-component sum. We reused ten saved initial fields (seeds 11/22,
nominal separations 24/26/28/30/32). New traces cover 0–100 at spacing .25 for
amplitudes -.1, +.05, +.1. The .05 trace reproduces the old saved fields' exterior
readout to `8.33e-17` over their shared 0–50 interval. Old terminal maps have
only three horizons and cannot supply the missing trace or a sequence forecast.

The kernel uses the existing five equation-informed separation features `f(s)`:
three screened-mediator decay/polynomial terms and two oscillatory SH-tail terms.
Its input vector is `phi(s,a)=[a f(s), a^2 f(s)]`. Each sampled lag has ten
coefficients. `K(lag<0)=0`; calls beyond its 100-unit stored horizon raise an
error, rather than silently truncating the tail. The sequence baseline sums
delayed kernels. It stores the response table and past pulse history.

**Observation:** linear amplitude dependence has 17.85% single-pulse development
NRMSE; quadratic dependence reduces it to 0.904%. Sign reversal is not a simple
sign flip. **Test:** paired signs and amplitude scaling, same preparations.
**Conclusion:** quadratic input dependence is warranted here. No odd-symmetry
or monotone-response prior was imposed.

The evolving model is an ordinary discrete ERA/SVD realization of the fitted
kernel's block Hankel matrices, with one shared transition and readout:

```
z_next = A z + B phi(s,a) at the first sample after an event
z_next = A z otherwise
predicted exterior deviation = C z
```

The instantaneous event only fills a two-scalar pending-input buffer, so `m`
does not jump at the pulse. At the next .25-unit tick, the buffer drives `B`
and clears. `z0=0`, reflecting a perturbation about the prepared sham, not a
zero physical field. Events/updates must be on the declared grids; simultaneous
unordered events and off-grid updates are rejected. Signed/competing outputs
are allowed. There is no stability constraint or clipping. The selected `A`
has spectral radius `0.999616`; accuracy beyond time 100 was not tested.

**Observation:** four and six response states approximate the kernel poorly.
Single-pulse NRMSE is 7.53% / 5.31% / 1.71% for orders 4 / 6 / 8. A targeted
order-10 follow-up gives 0.924%, close to the kernel's 0.904%.
**Explanation/test:** kernel approximation order, not evidence of missing
physical state. The order-10 realization was frozen before fresh evaluation.

**Observation:** at separation28, exact measured single-pulse superposition
has about 9.05% error for `+.1, -.1` separated by 2 units, but below 0.1% for
the tested waits10/40. **Competing explanations:** kernel fitting, reference
aging, numerical error, state-dependent pulse action or pattern relaxation.
**Tests:** compare complete sequences with *true timed single pulses*; refine
numerics; reverse pulse order; add waits5. Aging changes single responses by at
most `1.07e-8` in the diagnostic, versus `1.06e-6` short-sequence residual.
The short-pulse residual survives replacing fitted kernels with measured ones.
It is therefore not predominantly kernel interpolation or slow reference drift.

One attempted repair retained a scalar susceptibility `w`, relaxing exponentially
and incremented by each pulse. It used
`a_effective=a[1+w(c0+c1*a)]`. A least-squares development fit found
`tau_w=1.436, c0=-.437, c1=6.054`, but worsened worst trace error from 10.24%
to 10.90%. `repair-01` preserves this failed model and every score. It was
not selected. This does not prove that one additional variable is insufficient;
the chosen event-map form may simply be inappropriate. The exact division
between multiplicative pulse semantics and fast field deformation remains
unresolved. We stopped increasing complexity and tested a relaxation-limited
interface, with a shorter-wait challenge kept visible.

## Numerical qualification and frozen fresh panel

Development sequence waits were 2,10,40, with both signs; the targeted follow-up
added reversed order at2 and opposite signs at5. The new 100-unit two-pulse
qualification used separations24/28 and independently prepared refined fields.

| Change | Maximum exterior-response difference |
|---|---:|
| dt .0125 → .00625 | 4.70e-11 |
| N512 → 1024, dt .00625 | 3.01e-10 |
| L128 → 256 at matched spatial spacing, dt .00625 | 7.87e-8 |
| Three-pulse follow-up: time / space / domain | 3.73e-12 / 4.29e-11 / 9.96e-9 |

The three-pulse qualification is a numerical follow-up, not model refitting.
The domain effect is periodic-image sensitivity, not integrator truncation.
All errors used to diagnose close-pulse failure exceed these uncertainties.
No instability was clipped or discarded. This extends qualification to these
sequences/horizons; it does not certify arbitrary input streams.

`frozen-01/freeze.json` contains the unchanged model hashes, complete fresh
schedules, readout, split, and gates **before** new preparations were run:

- seeds411/522/633; intermediate separations25/29/31; nine complete preparations;
  two isolated 150-unit relaxations, residual .02 at assembly, joint age20 for
  odd seeds and60 for even seeds, matching the prior preparation family;
- primary two-pulse schedules: `(+.075,+.075)` at0/6,
  `(-.075,+.075)` at0/15, `(+.05,-.1)` at0/30;
- withheld three-pulse schedule: `+.06, -.075, +.05` at0/7/23;
- challenge outside proposed wait>=5 envelope: `-.075,+.075` at0/3;
- horizon100, .25 sampling, no resets; same preparation across its schedules
  is **not** counted as independent evidence;
- per-run whole-trace **and after-last-input** NRMSE <=5%, max absolute <=2e-6;
  first0–5 absolute error <=2e-8, reported separately;
- absolute readout resolution floor1e-7, rounded above the development domain
  discrepancy. Below that signal RMS, relative accuracy is uncertified. This
  is a conservative task measurement contract, not a claim that early solver
  roundoff is as large as1e-7. Historical 5/20/50 gates remain unchanged.

No fresh outcomes changed the models or gates. The proposed envelope is a
tested within-family approximation, not a guarantee for every wait>=5 or all
preparations/amplitudes in an interval.

## Fresh results

NRMSE uses the signed **response**, not the large mediator background.
Zero-response prediction has 100% NRMSE. Each row retains absolute errors;
failed short-wait runs remain in pooled and worst-case scores.

| Fresh set | Runs | Kernel pooled NRMSE | Recursive pooled NRMSE | Recursive worst run | Pass count |
|---|---:|---:|---:|---:|---:|
| Two pulses, waits6/15/30 | 27 | 2.051% | 2.063% | 2.843% | 27/27 |
| Withheld three-pulse schedule | 9 | 1.700% | 1.663% | 3.384% | 9/9 |
| Wait3 challenge | 9 | 3.374% | 3.369% | 6.028% | 3/9 |

Recursive maximum absolute errors are `1.808e-6`, `2.583e-7`, and `4.243e-7`,
respectively. The challenge fails relative accuracy at separations29/31 for
all three seeds, despite small absolute residuals; this failure is not averaged
away. Across the three preparation-seed groups, primary pooled NRMSE is
`2.131%, 1.955%, 2.099%` (mean2.062%, sample SD0.094 percentage points).
Three-pulse seed values are `1.627%, 1.643%, 1.717%`.

Primary errors at 5/20/50 **after the last pulse** have pooled NRMSE
`1.18%, 2.01%, 1.97%`; maxima `4.87e-7, 1.63e-6, 3.80e-7`.
These are continued sequence outputs, not restarted single-pulse maps.
Dominant signed peak times agree to the .25-unit sample grid for all primary
and three-pulse cases (challenge maximum discrepancy .25). Late residual and
peak values for every run are saved; no exponential decay law is asserted.

The first0–5 signal RMS is at most `2.09e-8` and recursive absolute error at
most `1.70e-8`. The absolute early gate passes, but **early relative accuracy
remains uncertified** under the frozen1e-7 readout floor. The plot exposes the
small nonphysical oscillations of the finite-order approximation there.
Historical early terminal failure has not been redefined or overturned.

Throughout the fresh horizon, measured center drift is at most .04384 and
separation drift .08443 (about .34% of separation25). Two measured patches
remain present; terminal masses relative to each sham are within1.30e-6.
Static geometry is an adequate approximation for this exterior task over100,
not a claim that separation is exactly constant or that the pair is bound.

**Current explanation:** quadratic single-pulse response plus delayed
superposition already explains the supported sequence behavior. The compact
recurrence mostly realizes that response kernel. It does not need a learned
ontology or a new interaction mechanism. The existing kernel encodes the
interacting arrangement; successful superposition does not negate interaction.

## Receiver pathway and remaining limits

The seed411/separation29 sequences include the existing receiver-local
stage-matched mediator replay. Sham RK stage fields are streamed only to the
diagnostic simulation; they never enter the primary predictor. The earlier
roundoff-matched sham control and its interpretation are retained.

| Schedule | Contrast RMS / total response RMS | Model-error RMS / contrast RMS |
|---|---:|---:|
| Same-sign, wait6 | 6.52% | 28.97% |
| Opposite-sign, wait15 | 6.22% | 38.87% |
| Unequal amplitudes, wait30 | 6.56% | 19.32% |
| Three pulses | 5.48% | 58.13% |
| Wait3 challenge | 6.33% | 92.93% |

This is full-minus-targeted-replay, not a universal additive channel
decomposition. Direct pattern-field interaction is still present. These RMS
figures differ from the historical4.2% point comparison because geometry,
inputs and temporal aggregation differ. Model errors are smaller than the
contrast on the primary cases but remain substantial on that scale. No model
of the controlled contrast was fitted, so **preservation of the receiver
pathway is not established**. Exterior success is not receiver-mass closure;
the historical receiver regression failure remains intact.

## State, access and cost

| Resource | Recursive model | Causal kernel |
|---|---|---|
| Live response state | 10 floats | Past pulse times/amplitudes |
| Event bookkeeping | 2 pending input floats; clock | Growing event list |
| Initial physical information | One measured separation | Same |
| Stored response coefficients | A100 + B100 + C10 = 210 floats | 401 lags ×10 = 4010 floats |
| Coefficient binary storage | 1680 bytes | 32080 bytes |
| Saved JSON | 5113 bytes | 96085 bytes |
| 100-unit forecast, representative CPU time | 4.67 ms | .0304 ms |

The recurrence uses 12 live response/event floats plus clock and static
geometry; the reusable implementation also serializes the unused, zero
susceptibility slot and three constants for the rejected repair. These are
accounted for rather than hidden. Static separation-feature formulas and the
sample interval are also part of the model. There is no response table, pulse
history, or future field inside the deployment checkpoint. The full physical
state alone has1024 floats/8192 bytes, excluding solver workspace and
coefficients. Dynamic-state compression and total storage are different claims.

Initial field scanning costs about33 microseconds in this environment and
touches the full initial grid; it is not free. Subsequent state maintenance
averages about12 microseconds per .25-unit tick in this Python implementation,
with no field reads. The vectorized table kernel is faster for this short
offline forecast, but requires larger storage and growing input history.
Neither timing is a broad algorithmic speed claim.

The recorded development fitting/scoring of six degree/order candidates cost
2.21 CPU seconds; the failed event repair cost .34 seconds including its
order-10 realization and scoring. New field simulation, evaluator sampling and
preparation, including pilot, all refinements, fresh panel and quick repeat,
cost **260.75 CPU seconds (4.35 minutes)**, well below the30-minute budget.
No paid compute, downloads, neural training or remote writes were used.

At t15 immediately after an event, serializing only model parameters,
separation, clock, response variables and pending inputs, then loading a fresh
predictor reproduces the remaining rollout exactly (`max difference=0`).
The test also blocks simulator/extractor calls after initialization. This
establishes implementation self-containment, not universal predictive accuracy.

## Reproduction and checks

Run from the repository root. Outputs must be new directories. The quick
demonstration regenerates one prepared field and all five frozen schedules in
about10 CPU seconds, including receiver replay. Its five records exactly
match those in the saved panel. The known short-wait failure is printed too.

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.mediated_patterns.recursive_response quick \
  --data work/mediated_patterns/recursive_response/frozen-01 \
  --output work/mediated_patterns/recursive_response/quick-review
```

These are the exact scientific commands executed in this session (with the
shared prefix factored into `R` below for legibility). Outputs include source
snapshots and their actual argv. `report-01` used overly narrow plot axes;
`report-02` fixes that visual-only issue and preserves the first rendering.
An initial engineering test required exact zero after an FFT round trip; the
observed6.93e-20 was correctly treated as roundoff, using tolerance1e-18.
Neither issue changed equations, predictions, gates or scientific outcomes.

```bash
R=work/mediated_patterns/recursive_response
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.mediated_patterns.recursive_response pilot --output "$R/pilot-01"
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.mediated_patterns.recursive_response development --output "$R/development-01"
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.mediated_patterns.recursive_response qualify --output "$R/qualification-01"
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.mediated_patterns.recursive_response fit --data "$R/development-01" --output "$R/fit-01"
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.mediated_patterns.recursive_response diagnose --output "$R/diagnostic-01"
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.mediated_patterns.recursive_response extra --output "$R/diagnostic-02"
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.mediated_patterns.recursive_response repair --data "$R/development-01" --extra "$R/diagnostic-02" --output "$R/repair-01"
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.mediated_patterns.recursive_response freeze --data "$R/repair-01" --output "$R/frozen-01"
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.mediated_patterns.recursive_response fresh --data "$R/frozen-01" --output "$R/fresh-01"
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.mediated_patterns.recursive_response qualify-three --output "$R/qualification-three-01"
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.mediated_patterns.recursive_response report --data "$R/fresh-01" --output "$R/report-01"
MPLCONFIGDIR=/tmp/jepa-response-matplotlib OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.mediated_patterns.recursive_response report --data "$R/fresh-01" --output "$R/report-02"
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.mediated_patterns.recursive_response quick --data "$R/frozen-01" --output "$R/quick-01"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 make check PYTHON=.venv/bin/python
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest experiments/mediated_patterns/tests -q
.venv/bin/ruff check experiments/mediated_patterns/recursive_response.py experiments/mediated_patterns/response_state.py experiments/mediated_patterns/tests/test_recursive_response.py
.venv/bin/ruff format --check experiments/mediated_patterns/recursive_response.py experiments/mediated_patterns/response_state.py experiments/mediated_patterns/tests/test_recursive_response.py
git diff --check
```

The saved commands intentionally refuse overwrites. To repeat the complete
panel, use a different `R`, preserving its internal dependencies. No previous
oscillator panels need rerunning. The normal repository check passed:66 core,
33 skill tests plus2 subtests,46 oscillator tests, design/recipe/manifest/JSON/
compile checks. Field tests passed **20/20**, including9 new tests covering
causality, exact event support/timing, deterministic traces, permitted
initialization, whole-preparation separation, checkpoint/resumption without
field access, output non-overwrite, and explicit divergence failure. Ruff and
diff whitespace checks passed.

Added files only: `response_state.py`, `recursive_response.py`,
`tests/test_recursive_response.py`, and this note. Generated evidence lives
under `work/mediated_patterns/recursive_response/`: raw signed traces,
source snapshots, frozen coefficients/protocol, failed fits, per-run errors,
seed scores, numerical checks, replay contrasts, checkpoint and visual.
`protected.json` and `session_manifest.json` provide preservation/provenance.

The supported achievement is a small, self-contained **input-driven exterior
response model** for this prepared arrangement family and tested time/input
envelope. Its principal missing capability is reliable prediction of closely
spaced sign-reversing pulses. Arbitrary microstate closure, accurate early
relative response, receiver-pathway preservation, very long input streams,
and third-participant back-reaction remain unestablished.
