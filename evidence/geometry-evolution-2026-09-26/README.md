# Geometry evolution evidence

This package records the bounded unforced continuation of the existing three-center
response description. [GEOMETRY_EVOLUTION.md](../../experiments/mediated_patterns/GEOMETRY_EVOLUTION.md)
owns the prospective decisions, quantitative outcome and limits. The response
predictor and all earlier evidence remain unchanged.

The selected quadratic three-coordinate updater beats persistence and passes
all delayed-response gates on three new preparations, including target90 inside
the withheld75→100 interval. It passes every absolute coordinate tolerance but
fails the strict fractional-motion gate for a nearly stationary A center in one
preparation. The complete joint criterion therefore does not pass. Affine also
predicts delayed responses usefully, with worse small-motion geometry. No state
extension, response refit or post-fresh model repair was performed.

- [Final per-case scores, geometry gates and signed-error decomposition](data/work/mediated_patterns/geometry_evolution/analysis-final/summary.json)
- [Actual/predicted responses and history contrasts](data/work/mediated_patterns/geometry_evolution/analysis-final/responses.png), [physical center drift](data/work/mediated_patterns/geometry_evolution/analysis-final/geometry.png), and [weak-motion failure](data/work/mediated_patterns/geometry_evolution/diagnostic-02/weak-motion.png)
- [Freeze](data/work/mediated_patterns/geometry_evolution/frozen-01/freeze.json), [updaters](data/work/mediated_patterns/geometry_evolution/frozen-01/models.json), and unchanged F at the checksum-recorded path inside the existing present-state package
- [Initial affine comparison](data/work/mediated_patterns/geometry_evolution/fit-01/fit.json), [quadratic repair](data/work/mediated_patterns/geometry_evolution/fit-02/fit.json), and [retained development relative-motion miss](data/work/mediated_patterns/geometry_evolution/frozen-01/development-geometry.json)
- [Prediction seal at 50](data/work/mediated_patterns/geometry_evolution/fresh-01/seal-50.json) and [independent initialization seal at 60](data/work/mediated_patterns/geometry_evolution/fresh-01/seal-60.json)
- [Development refinement](data/work/mediated_patterns/geometry_evolution/dev-refine-01/refinement.json), [worst contrast refinement](data/work/mediated_patterns/geometry_evolution/fresh-refine-01/refinement.json), and [weak-motion refinement](data/work/mediated_patterns/geometry_evolution/fresh-motion-refine-01/refinement.json)
- [Post-freeze diagnosis](data/work/mediated_patterns/geometry_evolution/diagnostic-02/diagnostic.json), [failed diagnostic import](data/work/mediated_patterns/geometry_evolution/diagnostic-01/failure.txt), [cost ledger](data/work/mediated_patterns/geometry_evolution/budget-total.json), [inference accounting](data/work/mediated_patterns/geometry_evolution/analysis-final/accounting.json), [runtime/configuration](data/work/mediated_patterns/geometry_evolution/runtime.json), and [exact commands](data/work/mediated_patterns/geometry_evolution/commands.txt)
- [Member checksums](artifacts.json), [critical review](review.md), and [remote retrieval receipt](publication.json)

Schema: `development-01/rows.json` orders 21 history trajectories across seven
preparations. `trajectories.npz` stores times 50..100 every 5 units, `z`(21,11,3),
instantaneous full-field `rates`, and diagnostic 15-feature rows. Interior times
80/85/90/95 are excluded from G fitting, scaling and selection. The 84 development
responses use actual origins50/60/75/100; existing50/100 targets are reused with
input hashes. Only new intermediate fields are copied; older archives are not.

`fresh-01/rows.json` orders 9 histories, grouped by the three preparations.
`initial-descriptors-{50,60}.npz` contains only numeric three-center rows.
Corresponding boundary fields are separate evaluator objects. For origin50,
forecast arrays have axes(history,4 delays,h=161,output=2), at targets 60/75/90/100;
origin60 has three delays, to 75/90/100. `geometry-*` has(history,delay,center).
Each NPZ has one array per sealed updater/control. Forecasts are uninterrupted;
the origin60 observation never resets the origin50 continuation. JSON checkpoints
contain three centers, an integer step counter and a model hash, without fields.

Only after each seal does the evaluator generate its future fields. Saved
`evaluator-states.npz` and `evaluator-geometry.npz` cover 60/75/90/100. Each
`reference-i-time.npz` stores absolute(time,sham/probe,A/B/C,mass/moment/mediator)
readouts and C response = own probe minus own sham. Evaluation probes are separate
possible experiments and never alter later origins. `analysis-final/rows.json`
indexes 63 predictions and 42 paired contrasts, sharing 36 nonlinear responses;
these are descendants of three independent preparations. The NPZ of signed
predictions keeps update/readout residuals, true geometry, predictions and truth
separate. All seeds and outcomes are now exposed.

The archive includes all substantive new stages, including unsuccessful models
and the failed diagnostic startup. Scientific sources are unchanged after freeze;
protocols retain per-file hashes and execution revisions. The test/report commit
is `2c42405`. The archive is below 25 MiB; no release asset or inherited archive
is duplicated. Costs total 567.012545 CPU seconds:536.012545 measured, one
conservatively charged failed startup second, and 30 seconds for startup/inspection.
The inspection failure in the console summary is retained separately.

Verify or restore without importing scientific code or executing a field run:

```bash
python3 -I evidence/geometry-evolution-2026-09-26/restore.py --verify-only
python3 -I evidence/geometry-evolution-2026-09-26/restore.py --destination /chosen/new-empty-directory
```

Restoration reuses the existing path/symlink/exclusive-copy protections. Hashes
make bytes inspectable, not scientifically correct or platform-immutable. The
critical review checks information flow and claims separately from checksums and
CI. Final merge metadata belongs in PR comments, not a metadata-only successor PR.
