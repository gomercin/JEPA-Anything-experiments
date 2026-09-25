# Repeated interaction through the frozen spatial pair

Exploratory local investigation, 2026-09-24. **The unchanged twenty-mode pair passed the resolved-target criteria in all 24 fresh repeated-input runs from six preparations across three seeds.** Worst whole-trace return NRMSE was 0.668%; worst resolved post-event/relaxation-window return NRMSE was 1.315%. No repair or new fit was needed. Early and cancelled sub-floor windows remain uncertified. This is reuse of the spatial pair from TWO_WAY_HYBRID.md, not the earlier ten-mode pulse-response map.

![Repeated total/source responses and residuals](../../work/mediated_patterns/repeated_hybrid/analysis-01/repeated_responses.png)

![The much smaller controlled return response](../../work/mediated_patterns/repeated_hybrid/analysis-01/repeated_return.png)

## Contract and provenance

The original basis and solver are unchanged. The model SHA256 remains `a128228fd027986ed17fbc1802f02a9d1820f7dffd215aa46fa6211c4c04bb6e`. The new code supplies exact event scheduling, measurement and evaluator branches around the existing equations. There is no new fitted coefficient, basis extraction, physical-memory variable, state reset, output assimilation, or AB field refresh.

The protected starting manifest hashes 1,434 historical files, including the prior hybrid source, frozen model, qualification panels and failed orders. The prior single-stimulus case s2011-c39 was rerun first: both reference and hybrid absolute traces matched the saved record exactly. The new event runner's first-only continuation provides a separate regression check against that case.

AB still owns twenty evolving coordinates and a permitted, once-extracted static template on [-34,28). The 520 exterior pattern values and entire 768-point mediator remain resolved. The direct fourth-order pattern-field pathway, one mediator source per grid point, and current-environment mediator feedback retain their previous ownership. Global projected ETDRK4 exchange is unchanged. Reconstructed AB profiles are quadrature evaluations of the retained coordinates; no independent fine-grid residual advances.

The laws remain `u_t = r*u-(1+d_xx)^2*u+b*u^3-u^5+g*m*u` and `tau*m_t=D*m_xx-m+s*u^2`, with r=-0.67, b=2, g=.05, s=1, D=64 and tau=10. The primary grid remains periodic L=192, N=768, dt=.00625, float64. Initial templates, pulse supports and all sensors stay fixed at their preparation anchors; measured future centroids never enter either predictor or input placement.

Each declared event is the existing instantaneous relative pulse `u <- (1+a*w(x-C))*u`, with compact radius-8 support and no mediator jump. Samples at an event time are taken after the jump. Event times must lie exactly on the timestep grid; unsupported off-grid times are rejected, not rounded into a different experiment. Omitted events leave the same fixed timestep partition. Positive and negative pulses are not inverses: without evolution, equal opposite events leave `-a^2*w^2*u`. A unit test checks that algebra. This session introduces no continuous forcing law or per-step repeated jump disguised as a finite-duration input.

## Schedules chosen from the field response

The saved C39 single response has B's maximum near time 7 and C's return maximum at 23.5. At time 8, B retains 98.9% of its peak while C's local response is only 1.39% of its initial value. At time 60, B retains 0.848% of its peak, but the return signal still retains 20.0%. Thus a locally recovered C does not imply a relaxed coupled arrangement.

Development reused the exposed preparation s2011-c39:

| Schedule | Events (time, amplitude) | Horizon |
|---|---|---:|
| Overlapping same sign | (0,+.075), (8,+.075) | 80 |
| Overlapping reversal | (0,+.075), (8,-.075) | 80 |
| Substantially relaxed reference | (0,+.075), (60,+.075) | 140 |

The 60-unit gap is explicitly only comparatively relaxed; its residual return is not zero. The 140-unit run extends the prior 80-unit evidence and leaves 80 units after its second event. A matched first-event-only run continues the original preparation with a zero event at time 8; it is not a newly prepared single-pulse trace.

The untouched panel was frozen after development and numerical qualification. It uses three new seeds 3011/3022/3033, each with C at 39 and 41: six preparations. Preparation is unchanged (independent 150-unit patch relaxation, assembled residuals, then joint age 20 for odd seeds or 60 for the even seed). Each preparation receives all four schedules below, making 24 related runs, not 24 independent preparations.

| Frozen schedule | Events (time, amplitude) | Horizon |
|---|---|---:|
| Overlapping same sign | (0,+.080), (10,+.065) | 80 |
| Reversed input order | (0,-.080), (10,+.065) | 80 |
| Longer-gap reversal | (0,+.075), (55,-.075) | 135 |
| Irregular triple | (0,+.075), (9,-.065), (27,+.080) | 107 |

Individual amplitudes remain within the earlier +/-0.1 range. Cumulative absolute commanded amplitudes are .145, .145, .150 and .220 respectively. These totals describe commands, not an equivalent physical impulse; each jump acts on the instantaneous evolving field. No schedule is replaced after observing its score.

## Readouts, control and frozen criteria

For each preparation, reference and hybrid each evolve four branches: sham, stimulated, stimulated with AB mediator-feedback replay, and sham with that replay. Only the mediator entering `g*m*u` near AB is replayed from that solver's own sham at the four ETD stages, using the original smooth mask. The direct pattern-field pathway and mediator source/transport continue. This is an evaluator-only intervention; the primary hybrid receives none of the replay or sham arrays.

We calculate `delta_full = full_stimulated-full_sham`, `delta_cut = cut_stimulated-cut_sham`, and `return = delta_full-delta_cut`. This subtracts each control's own baseline explicitly. The result is the same selected AB mediator-feedback-dependent contrast as before, not a decomposition of all back-reaction. B mass and C mass use the original smooth fixed windows, and response is never divided by the stationary background.

The original gates remain: B response NRMSE <=10% and maximum error <=2e-5; C response NRMSE <=1% and maximum error <=1e-3; resolved C return NRMSE <=10% and maximum error <=1e-6. The return RMS resolution floor remains 2e-7. Whole-trace scores are supplemented by fixed windows: the first 20 units after each event; from ten units after the last event to the end; and the final 20 units. Windows may overlap other events and always use the same initial sham, without resetting the baseline. The first five units after each event are separately reported using absolute errors; their relative accuracy is not an acceptance claim.

Below the return floor, relative accuracy is explicitly unresolved rather than converted to a pass. Overall resolved-task qualification additionally requires resolved whole-trace and last-return windows. Every unresolved interval remains listed. A fixed single-event scale, taken from the previously exposed s2011-c39 0–80 response, supplements all scores to expose cancellation; it never replaces the original normalization or gates.

## Development diagnosis and numerical qualification

All three unchanged-model development runs passed the resolved gates. Whole-trace return NRMSE was 0.286%, 0.322%, and 0.538%; the largest resolved return-window error was 1.534% in the final relaxation of the 140-unit run. No repair was selected or fitted.

The extra effect of event two was isolated by subtracting the matched first-only continuation. After the overlapping second event, incremental return NRMSE was 0.307% for same-sign and 0.299% for reversal. This demonstrates prediction of the additional response, not merely retention of event one's larger signal. It is an incremental response, not a four-run nonlinear cross-pulse identification. The reversal reduces the total-response denominator; signed residuals and fixed-scale errors are retained alongside NRMSE.

Competing explanations for an apparently accurate response included numerical cancellation, accurate C-only relaxation, and a genuine prediction of AB-dependent return. The selected contrast is evaluated in both models, so the result cannot be earned by matching C's isolated relaxation alone. Whole-quartet refinement to dt=.003125 changed the C return by at most 4.55e-13 in reference and 5.58e-13 in hybrid for the two targeted development cases. Conservative sums of the separately baseline-subtracted C-response errors were at most 7.60e-8.

Doubling spatial resolution together with halving dt in the reversal changed the return contrast by 6.74e-12; its conservative paired bound was 1.379e-7, below the unchanged 2e-7 floor. These empirical refinement measures are not rigorous continuum error bounds. Early/cancelled intervals below the floor remain unresolved even when the discrete reference and hybrid agree closely.

For the 140-unit reference, enlarging the domain to L=256 at the same dx changed C response by 3.76e-9 and the selected return by 1.37e-13. This diagnostic zero-extends the same interior preparation, whose mediator is about 5.08e-4 at the old endpoints; it is not an exactly identical periodic initial-value problem. The small measured effect supports the local boundary/image approximation over this horizon, without claiming exact boundary independence.

The original twenty-mode approximation was therefore carried forward unchanged. The earlier failed four/eight-mode models and their diagnosis remain historical evidence, not new failures or new comparisons in this session.

After the fresh panel, the triple with the largest declared whole-return error (s3022-c39) was checked at half timestep in both solvers. Reference/hybrid return differences were 4.37e-13/4.94e-13 and conservative paired bounds were 8.215e-8. This confirms the repeated-event numerical check without changing the model, floor, gates or exposed outcomes. The no-stimulus replay mismatch over all fresh runs was at most 5.33e-15 in the reference and exactly zero in the hybrid.

## Fresh outcomes and their limits

| Frozen schedule | Runs passing resolved task | Worst B whole NRMSE | Worst C whole NRMSE | Worst return whole NRMSE | Worst resolved return-window NRMSE |
|---|---:|---:|---:|---:|---:|
| Same sign, gap 10 | 6/6 | 0.385% | 0.0000899% | 0.501% | 0.819% |
| Reversal, gap 10 | 6/6 | 0.419% | 0.0000334% | 0.433% | 0.566% |
| Reversal, gap 55 | 6/6 | 0.393% | 0.000118% | 0.668% | 0.874% |
| Irregular triple, gaps 9/18 | 6/6 | 0.456% | 0.0000497% | 0.645% | 1.315% |

Worst absolute trace errors were 7.887e-6 for B, 3.842e-7 for C, and 6.154e-8 for the selected C return contrast. These are errors against the retained discrete reference, not equally precise continuum-error claims. Per-preparation scores, absolute RMSE, fixed-single-scale errors, signed traces and every window are preserved in `analysis-01/per_preparation.md`, `analysis-01/summary.json` and `fresh-01/results.json`.

All 24 whole-trace and last-return windows were resolved. Across the overlapping diagnostic/acceptance windows, 37 return windows were below the floor: each initial five-unit interval, plus 13 cancelled/weak first-20 or final-relaxation intervals. Those 37 scores are not independent cases. In particular, final reversal tails at C39 and C41, and several short C41 windows, do not earn relative-fidelity claims. They are retained with absolute errors rather than silently treated as relative passes. The figures show the largest whole-return-error case, s3022-c39 with the 55-unit reversal; the shaded scale is the frozen resolution floor, not a pointwise confidence band.

**Observation:** later return responses are accurately predicted while total C response is even easier. **Competing explanations:** an accurate local C relaxation might disguise missing return feedback, or small denominators could manufacture a relative success/failure. **Tests:** both solvers predict their own matched controlled contrast; the development first-only subtraction isolates event two; both relative and fixed-scale errors are retained. **Conclusion:** the unchanged pair predicts the specified return pathway during these repeated interactions. A comparator using the feedback-cut C response already achieves total-C NRMSE <=0.00483%, well inside the 1% total-response gate, but predicts zero of the selected contrast (100% contrast NRMSE). Thus the controlled-contrast test, not C relaxation accuracy, carries the two-way claim. This remains specific to the declared AB mediator-feedback intervention and does not remove or identify every direct return pathway.

No reference merger, extinction, divergence or escape from the measured three-region geometry occurred. Across fresh runs, minimum region masses were approximately [8.006, 8.018, 6.957], maximum measured width change 0.0815, maximum center drift 0.0652 and maximum AB separation change 0.07446 (0.266% of nominal separation 28). These output-only measurements support fixed geometry over these trajectories; they were not used to recenter or correct the reduced state.

**Observation:** excitation extends beyond calibration while scores still pass. **Alternatives:** channel coverage could be an error indicator, geometry drift could invalidate the static frame, or the projection could simply remain useful over these trajectories. **Tests:** the exact frozen model is evaluated with channel/rate monitoring and reference-only geometry diagnostics. **Conclusion:** no model failure requiring a repair was found. The largest joint/marginal excursion occurs in the largest whole-return-error case, but its error still passes; the largest channel rate occurs in another case. These associations do not identify a causal error mechanism or certify a broad extrapolation region. No best-in-basis or teacher-forced rescue was needed, and no additional modes or physical memory were inferred.

There are no new failed scientific candidates or selected repairs to hide. Numerical-resolution limits and channel excursions remain adverse qualifications. The sole code correction was diagnostic duration bookkeeping, described below. No field law, interface, basis, transition, pulse semantics, readout or gate was changed.

## Channel monitoring and implementation self-containment

The new runner saves current mediator/direct-pattern generalized forces, their sampled rates, reduced coordinates and output-only shape measurements. None is fed back as a correction. Calibration monitoring uses only the old full-field calibration snapshots: channel minima/maxima, maximum sampled rate, and a joint nearest-neighbor distance. For the joint diagnostic, each trajectory is centered at its own initial channel vector, each channel is divided by its training standard deviation, and Euclidean distance to the standardized training cloud is divided by sqrt(40). Its training nonself nearest-neighbor maximum is 0.0757. It is an excitation/coverage descriptor, not an error estimate or proof that a state is absent from the basis.

Every fresh run lies outside at least one marginal calibration interval for the sampled duration. Maximum marginal excursion is 9.182 training spans; the largest centered standardized joint distance is 21.09. The largest sampled force rate is 3.457 times the corresponding training maximum. These are actual un-clipped excursions. The model evaluates projected equations, not an interpolated gain table with a guaranteed calibration box. Its fresh success covers the recorded sequences, not arbitrary points at those distances.

The finite scheduler holds at most three declared `(time,amplitude)` pairs. It does not retain an expanding event history or future field trace. A restart at time 6, while the first response is active and before the events at 9 and 27, serializes the complete hybrid plus the declared schedule and cursor. A newly constructed predictor reproduces uninterrupted state and saved primary readouts exactly. Accuracy is assessed separately against field truth; restart equality itself is an engineering property.

The AB dimension, basis, template and dense operator caches are unchanged: twenty AB coordinates, 520 exterior u values, 768 physical mediator values, 4,960 spatial-basis coefficients and a 248-value static AB template. The whole hybrid has 1,308 physical dynamic scalars, not twenty. FFT redundancy makes its stored dynamic representation 1,310 float64 equivalents plus the clock. It still reconstructs a profile on the full quadrature grid; its roughly 5.78 MB of retained arrays include large dense matrices. It is not a total-memory reduction or a demonstrated speedup.

Logged simulation/preparation/restart CPU time is 787.918 seconds (13.13 CPU minutes); summed measured wall time is 789.704 seconds. This includes original-case reproduction, all development controls, refinements, 24 fresh sequences, the repeated quick demonstration and restart comparison. There is no new model-fitting cost. Construction of the training-only coverage diagnostic, plotting, editing and routine checks are outside this simulation ledger; the wall-time sum is not total session elapsed time. Per-run preparation, initialization and integration times are saved. For the four-branch fresh evaluations, hybrid/reference CPU ratios range from 1.68 to 1.85. The initial full snapshot remains a charged one-time access; no reduced-state advantage is converted into a speedup claim.

New scheduling storage consists of up to six command scalars, a derived six-value tick/amplitude cache, a tick counter, event cursor and fixed C anchor. The complete measured restart is 51,625 bytes of hybrid state/static data plus 342 bytes of scheduling metadata. The checkpoint contains no removed evolving AB grid, future replay, or reference trajectory. Logged diagnostic histories and evaluator sham/control branches are additional experimental storage and work, not predictor inputs. Primary stepping does not call the truth solver or measurement extractor; tests disable those dependencies after initialization.

One bookkeeping correction was made in the new diagnostic code: the first development snapshot counted every sampled endpoint as a full half-unit of excursion duration (80.5 for an 80-unit all-outside trace). Fresh monitoring uses trapezoidal integration over the actual sample times. The original development output is retained with that limitation. This changes no field evolution, model, criterion or scientific response score.

## Reproduction and checks

Run from the repository root, always choosing a new output directory:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.mediated_patterns.repeated_hybrid quick --output work/mediated_patterns/repeated_hybrid/quick-reproduction
```

This reproduces the frozen irregular three-event case s3011-c39, including reference and hybrid matched controls. It is a reproduction of exposed evidence, not new validation. The full frozen panel can be reproduced with:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.mediated_patterns.repeated_hybrid fresh --output work/mediated_patterns/repeated_hybrid/fresh-reproduction
```

The staged commands are `development`, `qualify`, `freeze`, `fresh`, `checkpoint`, and the post-freeze `fresh_refine` diagnostic. They intentionally use the retained local baseline and frozen artifacts, rather than implementing a general orchestration system. Exact simulation/analysis commands are in `work/mediated_patterns/repeated_hybrid/analysis-01/commands.txt`, with execution-source snapshots in each panel. The analysis was run with `MPLCONFIGDIR=/tmp/jepa-repeated-matplotlib` and `OPENBLAS_NUM_THREADS=1`; an unwritable optional font-cache warning did not prevent rendering, and both plots were inspected. The quick triple reproduced the corresponding saved reference and hybrid traces exactly.

Only new source files are added: `hybrid_schedule.py`, `repeated_hybrid.py`, `repeated_analysis.py`, `tests/test_repeated_hybrid.py`, and this report. The original `hybrid_pair.py`, frozen basis, field equations and previous reports are protected unchanged.

Checks executed:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest experiments/mediated_patterns/tests/test_repeated_hybrid.py -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest experiments/mediated_patterns/tests -q
OPENBLAS_NUM_THREADS=1 make check PYTHON=.venv/bin/python
.venv/bin/ruff check experiments/mediated_patterns/hybrid_schedule.py experiments/mediated_patterns/repeated_hybrid.py experiments/mediated_patterns/repeated_analysis.py experiments/mediated_patterns/tests/test_repeated_hybrid.py
.venv/bin/ruff format --check experiments/mediated_patterns/hybrid_schedule.py experiments/mediated_patterns/repeated_hybrid.py experiments/mediated_patterns/repeated_analysis.py experiments/mediated_patterns/tests/test_repeated_hybrid.py
git diff --check
```

The six new tests cover exact causal event timing/support, relative-pulse non-inversion, each control's own sham subtraction, zero-event partition/sham consistency, checkpoint self-containment with forbidden truth access, preparation split, non-overwriting output and strict nonfinite JSON rejection. They do not assert model superiority. The field suite has 43 passing tests; existing checks have 66 core, 33 skill plus two subtests, and 46 oscillator tests: 188 distinct tests plus two subtests. Recipe/design/manifest/compile checks also pass.

Machine-readable primary results, freeze, numerical checks, restart and reporting summary are under the new root in `fresh-01/results.json`, `frozen-01/freeze.json`, `qualification-01/`, `fresh-refinement-01/`, `checkpoint-01/` and `analysis-01/summary.json`. All 1,434 starting files were rehashed without a change. `final_manifest.json` records final source/model hashes, strict JSON checks, preservation and test counts. Base Git revision remains `c6e6c88f3ef75a4ce7acd660d6fa5779d995512c`; the per-panel source hashes identify the executed local additions. Later report/diagnostic additions do not rewrite those execution snapshots.

## Supported scope

The same frozen spatial reduction continues participating after its resolved neighbor is disturbed again: overlapping same-sign and reversed-sign pairs, longer-gap reversals, and one irregular triple per preparation. Fresh horizons are 80, 107 and 135 units, with an additional 140-unit development qualification. It advances continuously, using its own environment and without reopening AB's microscopic evolution.

No learned memory correction, recalibration or enrichment was required. This does not establish minimality, unique physical state, binding, arbitrary continuous stimulation, an unlimited event train, or universal closure. Sub-floor early/cancelled intervals remain uncertified. The result is unchanged-model reuse for the declared finite event family and resolved return task, with the original dense computational costs still present.
