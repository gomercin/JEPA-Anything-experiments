# A reduced pair coupled to a resolved third pattern

Exploratory local result, 2026-09-22. **A shared twenty-coordinate spatial model of AB replaced its 248 evolving pattern-field values and predicted a selected return influence on C in all six fresh preparations.** The largest return-contrast trace NRMSE was 0.570%. The shared mediator and exterior pattern field remain resolved. This is a newly constructed reduced subsystem, not direct reuse of the earlier pulse-response model, and it is not a speedup.

![Full field and hybrid, including the smaller return contrast](../../work/mediated_patterns/two_way_hybrid/analysis-01/two_way_response.png)

The figure shows the worst fresh return-contrast case, seed 2022, C at 39. The initial relative pulse is applied at time zero. C's large local relaxation, B's outgoing-source response, the much smaller controlled return contrast, and signed residuals are shown separately. Near coincidence of total-response curves alone would not demonstrate two-way accuracy.

## What was preserved and what changed

The equations, relative pulse operator, and previous evidence are unchanged. The earlier repaired model's 63/63 sequence result, failures, gates, ten response modes, two command summaries, 493 coefficients/knots and 40-value cache remain historical pulse-task results. Its exterior output includes transport and feedback, so injecting that output as a source into another mediator solver would double-count physics. None of its fitted dynamics or outputs is used here.

The new model uses a spatial Galerkin projection of the supplied equations. Only its spatial basis is calibrated. No transition matrix, scalar sensor-to-actuator gain, physical-memory interpretation, or neural model is fitted. The initial AB field is retained once as a static reference template; it is never advanced independently or refreshed.

The protected manifest covers 1,199 earlier source and artifact files. All remain byte-identical. No upstream core, Atlas, old report, old frozen model, or earlier output was edited. Base Git revision is `c6e6c88f3ef75a4ce7acd660d6fa5779d995512c`; experiment sources are local additions, so the per-panel source snapshots and hashes are the execution provenance.

## Equations, preparation, and physical connection

The retained local laws are

```text
u_t = r*u - (1 + d_xx)^2*u + b*u^3 - u^5 + g*m*u
tau*m_t = D*m_xx - m + s*u^2
r=-0.67, b=2, g=0.05, s=1, D=64, tau=10.
```

The SH35 base-model references and status of our mediator extension remain as documented in README.md and FINDINGS.md. There is no additional drive or controller in this extension. Boundaries are periodic. The larger domain is L=192, N=768, dx=0.25. The retained integration/exchange step is 0.00625 using float64 Fourier collocation and the existing Cox–Matthews ETDRK4 construction. Development began at 0.0125; qualification motivated halving it before final evaluation.

A and B are nominally at -14 and +14. Development C locations were 38 and 42, giving B–C gaps 24 and 28. Each component is independently prepared for 150 time units, with seeds `seed + 100*j` and initial variation 0.01. Their translated fields are added, small seeded shape residuals and a mediator residual are added, and the entire triple evolves freely for a declared preparation age. Translation occurs only during preparation. No pattern is pinned, recentered, or clamped during measurement. The zero background is counted once. Fresh post-preparation centers and shapes are recorded rather than assumed equal to the nominal anchors.

A single compact relative pulse at time zero acts on C:

```text
u_after(x) = (1 + a*w(x-C))*u_before(x)
```

The existing smooth support has radius 8; m is unchanged. After the pulse there are no external inputs. C affects AB through the evolving fields, and AB's evolving source and pattern tails affect C. All pulse supports in the final panel lie outside the removed AB region. Sensors are fixed at the preparation anchors, using the existing smooth radius-8, flat-core-5 window. Primary observables are weighted masses `integral(w*u^2)` at B and C. Local mediator means and A mass are also saved.

All fresh triples remain separated over the 80-unit measurement horizon, eight mediator relaxation times. The maximum observed center drift is 0.03962 and AB separation change is 0.04501, about 0.16% of its nominal separation. This qualifies fixed geometry locally, not binding, indefinitely stable organization, or the former 500-unit claim for the new triple.

## Ownership and access ledger

| Quantity | Owner and access |
|---|---|
| AB pattern field on [-34,28), 248 grid points | Removed as independent dynamic state; reconstructed as T+Bz |
| AB state z | 20 evolving coordinates, initialized to zero around the permitted initial template T |
| Exterior pattern field, including C and exterior tails | 520 evolving resolved values |
| Mediator, including the region inside AB | One shared 768-point field, evolved by the environment |
| Incident mediator channel | 20 generalized forces `g*B.T@(m_AB*u_AB)` from current hybrid state |
| Incident direct-pattern channel | 20 generalized forces `B.T@(L_u*embedded_u_exterior)_AB` |
| Outgoing pair description | 20 coordinates plus fixed basis/template, used to reconstruct source and pattern-tail contributions |
| Initial access | One complete initial snapshot; retain only T, initial exterior u, initial m, static geometry and basis |
| Later access | Hybrid state and supplied equations only; no truth AB, true centers, saved future inputs, or output assimilation |

The mediator source is `s*(T+Bz)^2/tau` inside AB and `s*u_exterior^2/tau` outside. These occupy disjoint rows and enter exactly once. Self-feedback is supplied by the one shared mediator; no response-kernel feedback is added.

Let P embed the pair basis and the exterior identity. With q=(z,u_exterior), the implementation advances

```text
u = T + P*q
q_t = P.T*L_u*P*q + P.T*L_u*T + P.T*(b*u^3-u^5+g*m*u)
m_t = L_m*m + s*u^2/tau.
```

The symmetric projected SH operator is diagonalized for ETDRK4. Its full cross terms remain, including the fourth-order direct pattern pathway. This is a global periodic trial-space projection, not a domain split with an invented single-value boundary condition. With a complete AB basis it is an orthogonal change of coordinates of the original spatial discretization. Linear exchange and nonlinear feedback are evaluated at integration stages, without a fitted delay or gain.

The implementation integrates the coupled trial coordinates jointly. The channel formulas describe the physical exchange; they are not a separate lagged co-simulation API. Spatial reconstructions for nonlinear quadrature are permitted and charged. There is no independently evolving grid residual inside AB, including at intermediate stages.

## Qualification and the controlled return signal

**Observation:** C has a measurable response dependent on AB's mediator feedback, but this is tiny beside C's own pulse relaxation. Candidate explanations were a genuine returning influence, numerical subtraction error, or baseline changes introduced by the control.

**Test:** four simultaneous branches start from identical prepared fields: sham, pulse, pulse with AB mediator-feedback replay, and sham with that same replay. In the replay branches, only the m entering `g*m*u` around AB is replaced by the sham's matching four ETD stage values. The mask is one within distance 8 of A/B and tapers smoothly to zero by 12. Pattern evolution, the direct pathway, shared mediator transport/source, and C's equations continue. The unperturbed control agrees to 3.55e-15 in the full solver and exactly in the hybrid.

The selected contrast is **full pulsed C mass minus AB-feedback-replayed pulsed C mass**. It is a pathway-specific counterfactual, not an additive decomposition of every return pathway. The hybrid predicts this contrast using its own control and its own sham. Those branches are evaluator diagnostics; the primary hybrid step receives neither sham nor replay. The reference simulator is never queried by the primary hybrid.

The fully resolved version of the same connection reproduced the monolithic ten-unit reference: maximum raw observable difference 5.01e-11, C pulse-response difference 3.45e-12, and C return-contrast difference 8.89e-15. This weakened interface bookkeeping, missing fourth-order conditions, and source double-counting as explanations for later low-rank failures.

Targeted whole-quartet checks varied timestep, doubled N, and enlarged the domain to L=256 at the same dx. At the development placement, actual return-contrast differences were at most 1.56e-11 and B-response differences at most 3.58e-10. However, a conservative sum of the separate pulse-minus-sham refinement errors was much larger: 6.4–7.34e-7 at the original timestep, because C's large early local transient is subtracted. This adverse qualification is retained.

After halving the retained timestep, refinement to 0.003125 gave an actual return difference of 5.04e-13, but a conservative paired bound of 1.105e-7. **Before fresh outcomes**, the prospective signal-resolution floor was raised from 1e-8 to 2e-7; all accuracy gates stayed unchanged. A post-freeze check of the weaker fresh case, seed 2022/C41, found timestep/spatial return differences no larger than 7.54e-13, conservative whole-trace paired bounds no larger than 1.143e-7, and bounds after time 10 below 4.21e-9. No model or gate was changed afterward.

These are empirical refinement diagnostics, not rigorous PDE error bounds. The reported small model errors are errors against the retained discrete reference; they should not be read as equally precise continuum accuracy. Initial 0–5 return RMS is at most 2.44e-8, below the floor: early relative accuracy remains unresolved. The late return signal is resolved. Spatial refinement qualifies the field reference; it does not establish uniform mesh-independent convergence of the learned basis.

## Calibration, failed reductions, and repair

Training used four whole preparations: seeds 71/82 at C=38/42, with joint ages 40/60 respectively. Five schedules supplied full AB snapshots: C pulses +0.1 for seed 71 and -0.1 for seed 82, plus an A pulse +0.05 at seed 71/C38. This is declared training access. Each block of sham drift, pulse-minus-sham response, and replay-minus-sham response was normalized by its own Frobenius norm before a shared SVD, so static background did not dominate the basis. Related schedules were not counted as independent preparations.

**Observation:** the first compact models badly missed B's response. Missing channels, feedback amplification, and inadequate spatial approximation were competing explanations. Full-rank consistency had already passed. Increasing the spatial order repaired the response without changing equations or adding a new memory law:

| Pair order | Worst B NRMSE | Worst B absolute error | Worst C return NRMSE | Development result |
|---:|---:|---:|---:|---|
| 4 | 83.07% | 1.31e-3 | 39.12% | failed |
| 8 | 63.24% | 1.07e-3 | 41.04% | failed |
| 12 | 4.10% | 7.58e-5 | 0.412% | failed B maximum gate |
| 16 | 1.22% | 2.74e-5 | 0.061% | failed B maximum gate |
| 20 | 0.238% | 5.33e-6 | 0.028% | passed |

Orders 4/8/12 used two positive-pulse development cases. The two justified four-mode extensions used all four C-stimulus cases, including the negative inputs, at the retained timestep. Thus rows are not identical panels. Every failure and source snapshot remains saved; no divergent runs were hidden or clipped.

A later diagnostic supplied exact reference exterior u and mediator at each RK stage to the four- and twenty-mode pair blocks. Four-mode B error remained 80.91%; even projecting instantaneous truth into that subspace gave 65.60% error. Twenty-mode driven B error was 0.194%, with instantaneous projection error 0.075%. True inputs did not rescue the poor basis. The current explanation is inadequate spatial approximation at low rank, with little evidence of large closed-loop error amplification in this tested regime. This does not identify minimal physical state or prove twenty is the smallest possible model.

## Frozen fresh evaluation

The basis, equations, extraction, readouts, timestep, schedules, envelope diagnostic and gates were frozen before six new preparations: seeds 2011, 2022 and 2033, each at C=39 with pulse +0.075 and C=41 with pulse -0.08. Odd seeds used joint age 20, the even seed age 60. Both locations and amplitudes were withheld. Location and sign are paired, so their effects cannot be separated by this small panel.

Frozen gates over the sampled 0–80 trace are: C response NRMSE <=1% and maximum error <=1e-3; B response NRMSE <=10% and maximum error <=2e-5; resolved C return NRMSE <=10% and maximum error <=1e-6. NRMSE divides by the RMS of the signed response, never the background. The return RMS must exceed 2e-7 for a relative claim. Historical pulse-only gates were not reused or edited.

| Seed | C | Pulse | B response NRMSE | C response NRMSE | C return NRMSE | Result |
|---:|---:|---:|---:|---:|---:|---|
| 2011 | 39 | +0.075 | 0.274% | 0.0000332% | 0.266% | pass |
| 2011 | 41 | -0.080 | 0.301% | 0.000000546% | 0.0428% | pass |
| 2022 | 39 | +0.075 | 0.319% | 0.0000762% | 0.570% | pass |
| 2022 | 41 | -0.080 | 0.382% | 0.000000852% | 0.0481% | pass |
| 2033 | 39 | +0.075 | 0.278% | 0.0000390% | 0.306% | pass |
| 2033 | 41 | -0.080 | 0.318% | 0.000000605% | 0.0453% | pass |

Worst absolute errors: B response 7.293e-6, C response 1.428e-7, selected C return contrast 3.603e-8. Return RMS ranges from 3.640e-7 to 3.375e-6. No fresh refitting or case-specific response correction occurred.

**Crucial comparator:** the AB-feedback-replay control already predicts total C response within 0.00358% NRMSE. At the total-output tolerance, returning feedback is unnecessary. Its zero prediction of the selected return contrast has 100% NRMSE, whereas the hybrid explicitly predicts that controlled difference. The qualified two-way result rests on this second comparison. Direct interaction remains in both control and full model; the control is not an assertion that all return pathways were removed.

**Envelope caveat:** every fresh run exceeded some calibration channel extrema. Raw excursions reached 6.82 training spans. Checking force changes relative to their initial values did not remove the issue: centered excursions reached 7.78 spans, up to 0.00497 absolute, on most sample times. No clipping occurred. Because channel forces are evaluated from the equations rather than a fitted lookup, these cases still ran and passed, but they demonstrate only the recorded extrapolation, not a generally qualified channel box. Static offsets alone do not explain the excursions. A broader input family remains untested.

## Self-containment and actual costs

A complete hybrid checkpoint at time 20 in the worst return case contains only config, basis, static AB template, transformed coordinates of `(z,u_exterior)`, mediator spectrum and clock. Loading into a fresh instance and continuing reproduces uninterrupted state and readouts exactly on this platform. That uninterrupted run also matches the saved fresh hybrid exactly and has the independently reported field-reference error. No future reference/replay array or evolving AB microstate is serialized. The test suite disables the full simulator and measurement extractor after initialization.

| Resource | Retained amount |
|---|---:|
| Pair dynamic coordinates | 20 instead of 248 AB grid values |
| Resolved exterior u | 520 values |
| Shared mediator | 768 physical values |
| Total physical dynamic scalars | 1,308 instead of 1,536, about 14.8% fewer |
| Actual dynamic float64 equivalents | 1,310 plus clock, including FFT redundancy |
| Pair basis | 4,960 coefficients |
| Per-case static AB template | 248 values |
| Incoming generalized forces / outgoing coordinates | 40 / 20 |
| Complete serialized hybrid checkpoint | 51,635 bytes |
| Actual retained arrays, including dense operator caches | 5,777,256 bytes |

The dense lift and diagonalization cache dominate storage. Nonlinear updates reconstruct the spatial profile and use full-grid quadrature and FFTs. There is no ever-growing input history, but this is not a small whole-world simulator or a total-memory reduction.

Measured costs on this local CPU with one BLAS thread: POD fit 0.045 CPU seconds; one hybrid initialization including extraction/matrix setup 0.0244 seconds; average update 0.000175 seconds. The initialization measurement bundles extraction and operator setup, so it does not isolate a field-read bandwidth cost. Four diagnostic branches over 80 units took about 9.7 CPU seconds for the hybrid versus 5.4–5.6 for the field reference. The hybrid is roughly 1.8 times slower here. Forecasting cost, preparation cost, fitting cost and evaluator branch costs are separately recorded; the one-time static initialization is not free.

Logged new simulation/fitting/restart work totals 488.94 CPU seconds (8.15 CPU minutes), summed measured wall time 489.28 seconds. This includes failed orders, refinements, and the repeated quick demonstration. It excludes editing, plotting, routine checks and uninstrumented reporting work; it is not total session elapsed time. A 1,000-tick cost microbenchmark is separate and earns no longer-horizon scientific claim.

## Files, reproduction, and checks

Added source files only:

- `hybrid_pair.py`: projected equation solver and self-contained checkpoint.
- `two_way_hybrid.py`: reference, qualification, calibration, staged repairs, freeze, fresh and quick runners.
- `hybrid_analysis.py`: comparison, envelope diagnostic, restart, cost ledger and figure.
- `tests/test_hybrid.py`: seven instrument tests.
- `TWO_WAY_HYBRID.md`: this report.

Outputs are under `work/mediated_patterns/two_way_hybrid/`. The principal records are `frozen-01/freeze.json`, `frozen-01/model.npz`, `fresh-01/results.json`, and `analysis-01/summary.json`. Model SHA256 is `a128228fd027986ed17fbc1802f02a9d1820f7dffd215aa46fa6211c4c04bb6e`. Each panel snapshots its execution sources before simulation. `analysis-01/commands.txt` records the exact scientific commands, including all failed orders, open-loop checks and refinements. Final hashes/check results are in `final_manifest.json`.

Run from the repository root. The quick demonstration takes about 23 CPU seconds including preparation and both sets of controls:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.mediated_patterns.two_way_hybrid quick --data work/mediated_patterns/two_way_hybrid/frozen-01 --output work/mediated_patterns/two_way_hybrid/quick-reproduction
```

Reproduce the six-case panel with the unchanged frozen model:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.mediated_patterns.two_way_hybrid fresh --data work/mediated_patterns/two_way_hybrid/frozen-01 --output work/mediated_patterns/two_way_hybrid/fresh-reproduction
```

These reruns reproduce already exposed cases; they are not additional held-out evidence. All output directories must be new. The staged runner intentionally uses saved local pilot/calibration artifacts for several diagnostics; it is not a generalized experiment framework. The exact executed commands file documents those dependencies. The saved quick run reproduced its corresponding fresh reference and hybrid traces bit-for-bit.

Checks executed:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest experiments/mediated_patterns/tests/test_hybrid.py -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest experiments/mediated_patterns/tests -q
make check PYTHON=.venv/bin/python
.venv/bin/ruff check experiments/mediated_patterns/hybrid_pair.py experiments/mediated_patterns/two_way_hybrid.py experiments/mediated_patterns/hybrid_analysis.py experiments/mediated_patterns/tests/test_hybrid.py
.venv/bin/ruff format --check experiments/mediated_patterns/hybrid_pair.py experiments/mediated_patterns/two_way_hybrid.py experiments/mediated_patterns/hybrid_analysis.py experiments/mediated_patterns/tests/test_hybrid.py
git diff --check
```

Results: 7 new tests; 37 field tests including those seven; 66 core, 33 skill plus two subtests, and 46 oscillator tests. Thus 182 distinct tests plus two subtests passed. Repository recipe/design/manifest/JSON/compile checks also passed. Tests cover full-rank consistency, source ownership, feedback units, pulse support and causality, no dynamic AB residual, forbidden truth access, deterministic complete restart, preparation split, non-overwriting output, sham-preserving replay and explicit nonfinite failure. They do not assert scientific superiority. Strict JSON validation and the final 1,199-file preservation check are recorded separately.

## Supported result and remaining capability

The supplied ingredients are local field laws, prepared patterns, initial templates, a calibrated shared spatial basis, the initial pulse, fixed readouts and a declared replay intervention. The dynamics generate subsequent C-to-AB influence, evolving AB source/tails, and a returning C response. The hybrid predicts both the main responses and the specified AB mediator-feedback-dependent contrast across the six fresh preparations without advancing or refreshing AB's removed fine-scale state.

The concrete missing capability is **reuse under sustained or repeated stimulation of C**, where the pair could leave this small-deformation envelope. This session does not establish that capability, arbitrary preparations, full pathway fidelity, unchanged old pulse-model reuse, minimality, binding, unique physical state, or universal closure. It establishes a conventional twenty-coordinate pair reduction coupled in both directions to a resolved environment and third pattern over the tested 80-unit, single-stimulus family.
