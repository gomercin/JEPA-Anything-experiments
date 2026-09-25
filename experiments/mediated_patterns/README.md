# Mediated localized patterns

A small, exploratory field toy, independent of JEPA/OPF training. See
[FINDINGS.md](FINDINGS.md) for the measured results and limits. Earlier oscillator
source and evidence are preserved. Nothing in Atlas is changed.

## Equations and provenance

On a periodic one-dimensional domain, two real fields evolve everywhere by the
same local differential equations:

```text
u_t     = r u - (1 + ∂xx)^2 u + b u^3 - u^5 + g m u
τ m_t   = D ∂xx m - m + s u^2
```

The base cubic–quintic Swift–Hohenberg equation is Eq. (1), with epsilon=0, in
[Raja et al., arXiv:2303.00798](https://arxiv.org/html/2303.00798v1).
Its b=2 localized-pattern regime near r=-0.675 motivates our r=-0.67 starting
point. We do not reproduce that paper's traveling-pattern or collision panel.
Our local `g m u` feedback and `s u²` source are an **engineered extension**.
The auxiliary diffusion/relaxation construction is related to the finite-rate
system RD_delta in
[Ishii and Tanaka, arXiv:2504.15180](https://arxiv.org/html/2504.15180v2).
We retain finite tau; we do not use its singular-limit theorem to certify this
nonlinear extension. The stationary mediator's length scale is sqrt(D)=8.
[Gelens et al.](https://arxiv.org/html/1305.6804v1), Eqs. (2)–(3), was considered
as a front-pinning alternative; that family was not implemented or tested.

Default parameters: r=-0.67, b=2, g=.05, s=1, D=64, tau=10; domain length 128,
512 grid points, dt=.0125. The wavelength selected by the linear SH operator is
near 2π. We use Fourier collocation and ETDRK4 (32-point contour coefficients).
The FFT implements derivatives of a **local PDE** with periodic boundaries;
it is not an object-based all-to-all interaction rule. Spatial refinement
qualifies nonlinear aliasing at the retained resolution. No field clipping,
recentring, object registry, bonds, pair forces, or stochastic ongoing forcing.

This is overdamped order-parameter relaxation, not a conservative material model.
For positive g,s its Lyapunov functional is

```text
E = ∫[-r u²/2 + ((1+∂xx)u)²/2 - b u⁴/4 + u⁶/6
       -g m u²/2 + g(m² + D m_x²)/(4s)] dx
u_t = -δE/δu;   m_t = -(2s/(gτ)) δE/δm.
```

There is no time-dependent drive or energy injection after preparation except
the declared instantaneous pulse. The coefficients supply bistability and
homogeneous local gain/saturation. Persistence here is stationary organization,
not an energy-consuming oscillator or a claim about thermodynamic emergence.

## Preparation and measurement

An initial patch is `1.3 cos(x) exp(-(x/5)^8)`, with a seeded 1% smooth shape
variation, and initially zero mediator. Each component evolves alone for 150
time units (15 mediator relaxation times). Two independently prepared fields
are translated **only during assembly**, then superposed (background u=m=0).
Their joint preparation evolves freely. No pins are ever imposed.

Qualification uses joint age 100 and then another 500 time units (50 tau).
Development/fresh response runs use age 20 or 60 and a seeded .02 smooth
preparation disturbance of u and m, removed before that free relaxation.
These small residuals deliberately test different microscopic preparations.
Two patches need not form a bound state to be a maintained interacting arrangement.

Measurements are separate from dynamics. Labels A/B identify fixed windows only.
A C2 window equals one within 5 units of a nominal center, tapers to zero at 8,
and defines mass `∫w u² dx`, center, width, peak, and weighted mediator. These
are descriptive observables; mass is not a conserved physical mass. The early
hard-window qualification failed and is retained, not silently replaced.

At the prediction boundary the input is an instantaneous **relative amplitude**
pulse to u near A: `u <- u[1 + a b(x-x_A)]`, where
`b(d)=exp(1-1/(1-(d/8)^2))` for |d|<8 and zero elsewhere. It never directly
reaches B. Outcomes subtract each condition's own sham. Primary horizon is
20 time units; 5 and 50 are diagnostic horizons.

Two outputs are retained: B's weighted pattern mass, and mean mediator in a
smooth sensor window centered 12 units beyond B (support radius 2). The latter
is a literal exterior field response. The sensor neither sources nor reacts;
it is **not** a third dynamically responding participant.

The primary field laws are homogeneous. The explicit causal diagnostic replays
an unperturbed mediator's four integrator-stage values. Global replay removes
induced mediator feedback; a more targeted fixed spatial replay at B preserves
A's self-feedback and the direct u pathway. This is an open-loop counterfactual,
not the main simulation. Its own sham agrees with full evolution to roundoff.

## Compact description and evidence split

Ordinary terminal regression retains measured separation plus known pulse
amplitude. Ten features combine exponential mediator-length terms, oscillatory
SH-tail terms, and linear/quadratic pulse amplitude. Constants are derived from
the supplied equations; coefficients are fitted to development full-field runs.
The two terms do not independently identify causal contributions; the replay
control supplies that evidence. Independent isolated-component predictions
remain a weaker baseline, with their zero background counted once.

Development: complete preparations from seeds 11/22, separations 24/26/28/30/32,
pulses .05/.1/-.1. Fresh within-family checks: seeds 101/202/303, separations
25/27/29/31, pulses +/-.075. No shared preparation or adjacent-snapshot split.
The fits, readouts, horizons and tolerances freeze before fresh runs: h20 NRMSE
≤.15 and worst absolute error ≤2e-5 for B mass, ≤2e-6 for exterior mediator.
These are engineered toy tolerances, not universal scientific thresholds.
Near-coarse-state diagnostic: separation difference ≤.05 length units, while
relative whole-field RMS differs by >1e-5. No microstate-sufficiency theorem.

Prediction evaluates a small regression after one field-based extraction.
It does **not** advance separation, amplitudes, or mediator state recursively.
Extraction scans the field; regression evaluation does not. Do not repeatedly
refresh descriptors and call that autonomous reduced dynamics.

## Commands

Run from the repository root. Existing output directories are refused.
The prepared environment supplies NumPy, matplotlib and pytest.

```bash
# Development/CI-like demonstration: persistence, mediator, response/replay, dt check
.venv/bin/python -m experiments.mediated_patterns.explore --quick --output work/mediated_patterns/NEW-quick

# Full numerical qualification and saved exploratory panel
.venv/bin/python -m experiments.mediated_patterns.explore --qualify --output work/mediated_patterns/NEW-qualification
.venv/bin/python -m experiments.mediated_patterns.explore --panel --qualification work/mediated_patterns/NEW-qualification --output work/mediated_patterns/NEW-development
.venv/bin/python -m experiments.mediated_patterns.outward_response --development work/mediated_patterns/NEW-development --output work/mediated_patterns/NEW-outward-development
.venv/bin/python -m experiments.mediated_patterns.explore --fresh --qualification work/mediated_patterns/NEW-qualification --model work/mediated_patterns/NEW-development/response_model.json --output work/mediated_patterns/NEW-fresh
.venv/bin/python -m experiments.mediated_patterns.outward_response --evaluate work/mediated_patterns/NEW-fresh --model work/mediated_patterns/NEW-outward-development/model.json --output work/mediated_patterns/NEW-outward-fresh
.venv/bin/python -m experiments.mediated_patterns.explore --receiver-control --output work/mediated_patterns/NEW-receiver-control
.venv/bin/python -m pytest experiments/mediated_patterns/tests -q
make check PYTHON=.venv/bin/python
```

Exploratory outcomes do not establish physical ontology, causal decomposition,
spontaneous formation, universal closure, or reusable third-participant interfaces.
An ordinary mechanism and a useful ordinary compact predictor are positive results.


## Dual participant reduction: local handoff

See [DUAL_REDUCTION.md](DUAL_REDUCTION.md). The first independently frozen C24 model had late-response failures in fresh FR/RR tests. A composition-informed C-only enrichment to 32 modes, using the original C calibration basis and leaving AB20 unchanged, passed 12 new repeated-input schedules from six preparations/three seeds. Worst selected return trace/window NRMSE is 0.666%/1.529%; early sub-floor fidelity remains unresolved. The shared mediator and exterior/tails remain resolved, and dense RR remains slower than FF. Both the first failure and adapted success are retained under `work/mediated_patterns/dual_reduction/`.

The selected model initializes once and resumes exactly without truth access. This is adapted composition over a bounded envelope, not joint fitting, first-freeze universal success, automatic unit discovery or whole-world compression. Stop here; no successor is selected. The report contains the exact pending Atlas delta; Atlas was not edited.

Saved-result interpretation check: **Original AB + C24** means **FULL_AB+C24** (FR in `fresh-01`), not AB20+C24. It failed the same final-relaxation C-response windows as AB20+C24 in `s4011-c41-three` and `s4033-c41-three`: respectively 1.1145%/1.0991% versus 1.0876%/1.0642% NRMSE, against the unchanged 1% gate. This does not establish a failure caused only by composing two individually passing replacements. In `fresh-02`, FR uses C32; Original AB + C24 was not recorded, so its performance on that panel's AB20+C24 failure windows remains unresolved.

Dual-reduction cycle completed locally: successful composition-informed C-only repair, not first-freeze independent-composition success. AB20/C32 retained as the frozen result. No successor selected.

## Integration review status

Implementation review and integration are recorded in [PR #1](https://github.com/gomercin/JEPA-Anything-experiments/pull/1)
and the dependent [PR #2](https://github.com/gomercin/JEPA-Anything-experiments/pull/2).
Their discussions record exact reviewed revisions, checks and confirmed merge status.
The publication-time draft/unreviewed statements below and in frozen reports are
historical. Integration does not independently reproduce the scientific panels or
expand their stated limits. Atlas remains unchanged; no research continuation is
part of this review.

Review found a checkpoint file-safety defect: a suffixless save could overwrite
an existing `.npz`, and a dangling destination symlink could be followed. Current
`PairHybrid.save` and `DualHybrid.save` now create the actual NumPy destination
exclusively. Only serialization changed; stepping, initialization, bases, event
maps, readouts and saved evidence did not. Six file-only regressions cover both
classes. Historical execution identity continues to use the archived source and
pinned publication revisions, not the corrected live files.

## Remote preservation handoff (publication-time record)

The completed field lineage and oscillator lineage are remotely preserved on
`codex/preserve-completed-experiments` in `gomercin/JEPA-Anything-experiments`.
See the [portable evidence index](../../evidence/completed-2026-09-25/README.md)
and [publication receipt](../../evidence/completed-2026-09-25/publication.json)
for the source-snapshot revision, review PR, checksum-pinned evidence tag/assets,
retrieval verification, and provenance gaps. Publication is not merge approval.
The original pending Atlas paragraph stays unchanged in DUAL_REDUCTION.md;
these remote pointers belong with the existing Effective Motif Dynamics and
persistence-to-composability ownership. Atlas remains pending and unedited.

The closed AB20/C32 cycle remains frozen. The separately chosen
organization-conditioned-response investigation is unexecuted by this task and
must start on a separate branch from the verified preservation head. No science
is to be appended to the preservation PR.

Publication verified: [draft PR #1](https://github.com/gomercin/JEPA-Anything-experiments/pull/1) is open and unmerged. Source-snapshot commit: `e17fe41d07a03d3a7fe67955054112773e8344be`; evidence tag: `evidence-completed-2026-09-25`. Both remote archives (346,031,945 bytes) were restored into a fresh GitHub clone: all 2,250 member hashes matched, and saved AB20/C32 models, checkpoint and table loaded without simulation. All extant substantive run files are archived; one older oscillator README version is unavailable, and historical provenance/style limits are listed in the index. The exact final pushed head is pinned in the [remote verification receipt](https://github.com/gomercin/JEPA-Anything-experiments/releases/download/evidence-completed-2026-09-25/remote-verification.json) and PR body. That head adds publication bookkeeping to the source snapshot; neither revision is retrospectively assigned to old runs.

Safe next-task start: fetch this preservation branch, verify its head against the receipt, then `git switch -c codex/organization-conditioned-response <verified-final-head>`. While preservation is unmerged, the next branch depends on this PR; keep its scientific changes separate and adjust its PR base when appropriate. No branch for that next task was created here.

## Organization-conditioned response: completed local investigation

The separately authorized investigation is complete on
`codex/organization-conditioned-response`, starting at
`1c555d5aab826418c81e0a9005a85200e3f8fd65`. See
[ORGANIZATION_RESPONSE.md](ORGANIZATION_RESPONSE.md) for its prospective contract,
actual response plots, numerical qualification and exact pending Atlas paragraph.

Moving A to change AB separation from 24 to 32, with B/C and two additive C probes
fixed, produced small resolved response-shape changes. No practical selectivity
met the declared 5% criterion on three fresh preparation seeds: a frozen common
gain with zero delay left only 0.084–0.107% total-response error. The selected
AB-feedback return is resolved, but its change across organizations is below the
conservative numerical floor. The small total-response differences survive the
feedback cut, with changed static backgrounds and remaining direct pathways;
pure gain equivalence and exclusive static causation are not claimed.

Timed work used 4.34 CPU minutes; ten focused instrument tests pass. The new
full-field experiment used no frozen reduction. AB20/C32, prior failures and
preserved evidence remain unchanged; no science was added to PR #1. No push,
merge, Atlas edit, or successor was selected. The preservation handoff above is
historical; this task used the owner's supplied verified head and local evidence
without restoring archives or repeating the preservation audit.

### Organization-response integration review

[PR #2](https://github.com/gomercin/JEPA-Anything-experiments/pull/2) records the
implementation review, exact revisions/checks and confirmed integration status.
The frozen report, thresholds and evidence remain unchanged. Saved-data score
verification supports the reported fit errors, not exact response equality or
resolved organization-dependent return selectivity. This task integrates the
completed result and stops; Atlas is pending and no successor is selected.

### Incremental publication handoff (publication-time record)

The completed organization-conditioned-response record is preserved separately
through the [portable incremental evidence index](../../evidence/organization-response-2026-09-25/README.md).
It includes all 163 new work files (8,853,052 bytes) in Git, with explicit
checksum-verified, non-overwriting restoration and saved-data-only inspection.
The index's publication receipt and separate draft PR record the source snapshot,
final head and dependency on still-unmerged preservation PR #1. This publication
adds no new science and changes no conclusion, criterion, floor or exposure split.
The original report and pending Atlas paragraph remain intact. Earlier archives,
PR #1 and AB20/C32 are unchanged; no Atlas edit, merge or successor is selected.

Publication verified in [draft PR #2](https://github.com/gomercin/JEPA-Anything-experiments/pull/2),
targeting `codex/preserve-completed-experiments`. The source snapshot is
`f1b6f2c62000e4eb415711653ab7ba973f8ab762`; a fresh GitHub clone and separate
restoration matched all 163 member hashes and loaded the saved configuration,
gain fits, floors, preparations and table without science or experiment imports.
See the [retrieval receipt](../../evidence/organization-response-2026-09-25/remote-verification.json)
and PR body for the distinct final pushed head. All new evidence is in Git;
no new release asset or inherited-archive download was needed. No extant
substantive new evidence remains local-only.
