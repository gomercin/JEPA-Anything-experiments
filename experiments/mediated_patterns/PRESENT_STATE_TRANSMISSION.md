# Present-state transmission

Living note, 2026-09-26. Base: merged `34c9b333057f3a059b53798674baa7e7e3da6ae2`.
The question is whether a small measurement of one CURRENT configuration predicts
its conditional fixed-A `.02` probe response and the independently predicted
written-minus-unwritten contrast. This is response prediction, not autonomous
state evolution, storage localization, minimality, or a new Atlas route.

## Prospective contract

Keep the SH35/mediator laws, assembly, anchors A=-28/B=-4/C=28, radius-8 compact
write support, waits 50/100, and 80-unit response horizon. Use C weighted u-squared
mass and signed moment, sampled every .5 units. A prediction receives only a
serialized extraction from a single `(2,N)` current field, static fitted model,
the declared probe and forecast times since that probe. No seed, case, filename,
write label/amplitude, wait, twin, future sham, trajectory or response enters it.
Checkpoint loading selects only `checkpoint_times` and `checkpoints`, never the
convenient future arrays in an old history file. Evaluator targets are separate.

Initial comparisons: development mean; B center only; all three centers relative
to fixed anchors; geometry plus three masses and widths; then those nine plus
three mediator means and three signed mediator spatial moments. Fixed windows
are the existing C2 windows. Mass is weighted u-squared, not conserved material.
Extraction scans the current arrays once conceptually, with multiple vector
reductions; it is not a local observer. No spatial template, FFT/rate or solver is
needed initially. Gaps alone would omit actuator alignment.

Fit a shared rank-4 temporal SVD basis, separately scaled readouts, and linear
or quadratic standardized features with ridge 1e-6 or 1e-3. All preprocessing,
basis, selection and regularization are fitted inside complete-preparation
leave-one-group-out folds. Include a fixed weight-10 difference loss on
development pairs to prevent common-response bias; inference is independent.
Retain evaluator-only temporal projection errors. If rank 4 fails the accuracy
targets by itself, inspect rank 6 before attributing failure to descriptors.
Do not interpret ill-conditioned feature ablation as a necessity result.

The old seeds 8101/10101/10102/10103 are exposed development data. Begin with
their saved odd-.4 and unwritten checkpoints/targets, and inspect feature
variation/conditioning before selecting a model. If they are correlated as
expected, a small extension on old preparations 8101/10101 will use negative
odd -.4 and a mixed odd .28/even .28 write (norm below .4), at the same waits.
Each must pass the existing three-pattern bounds throughout its history and
sampled probe continuation. All descendants of a preparation stay in one fold.
No thousands-of-snapshots sample inflation. Reserve fresh seeds 12101/12102/12103;
choose one withheld history after development and before any fresh preparation.

Use all 161 times and a fixed late window h=40..80, with each output separate.
Report RMS residual, maximum absolute residual, response/contrast RMS and
relative RMS, per preparation and worst cases. Targets are <=2% for each R and
<=10% for each resolved Delta_R in BOTH windows, no tuning after fresh access.
No division by write-only offsets/backgrounds. State-blind Delta_R=0 scores
100% on a resolved contrast. Sub-floor contrasts are unresolved, never passed.
For new histories refine a representative development contrast and the final
worst relative-error contrast from the same initial prepared state, at half dt
and doubled N, through write/wait/probe. Shared per-readout/window floor is
max(1e-12, five times the largest SUM of the two separate response refinement
RMS errors), across those checks. The fixed rule can raise the floor after final
refinement but cannot alter the error gates. Prior numerical evidence applies
only to unchanged cases. No Delta_Q/pathway claim is sought.

Freeze selected extraction/model/coefficients/criteria/history family before
fresh access. Persist descriptor-only predictions before nonlinear evaluation.
Permit up to two development repairs; if fresh results motivate repair that
panel becomes exposed and a separate untouched check is required. Stop when a
simple useful description succeeds; retain negative/limited outcomes too.

## Resources and evidence

1800 aggregate CPU seconds, development ceiling 1200 and roughly 600 reserved
for fresh evaluation/refinement. Meter preparation, simulation, fitting, plots
and analysis, including failures; reserve an additional 30 seconds for small
read-only inspection processes. Initial section bounds 120 CPU/180 wall seconds
and 1 GiB resident memory, to be checked against a pilot. Ordinary editing/tests
and publication are separate. One scientific process at a time. No paid compute,
large downloads, neural fitting or broad sweeps. Output directories and NPZ/JSON
files are exclusive. New evidence only; old portable packages remain read-only.
Publish all substantive new records, failures, commands and hashes in <=25 MiB
Git evidence, or a unique <=500 MiB compressed asset if needed. Review the actual
diff and merge with a reviewed-head guard; update the two existing Atlas homes.

## Development record

`old-fit-01` at ed3aeb5 used the 16 old late states from four preparations.
Three-center quadratic ridge (.001) gave grouped worst R/Delta_R errors
0.2754%/1.0634%; B-only gave 11.258%/34.458%, and the mean's Delta_R error is
100%. The rank-4 temporal projection's worst whole-response error was .0954%.
Geometry's weighted design condition was 66.3, versus 2.16e5 for the best
nine-variable shape fit and 3.28e9 for the fifteen-variable mediator fit.
These are approximation/conditioning comparisons, not information theorems.
The four preparation groups remain the effective replication count.

The planned extension ran next: two old initial preparations, negative odd -.4
and mixed odd .28/even .28 at waits 50/100. All histories/probes passed the regime
screen. Each two-branch response took about 4 CPU seconds; the original section
bounds (120 CPU seconds, 180 wall seconds, 1 GiB) have ample margin. The later
source cleanup adds deterministic scoring/freezing and separate inference-cost
receipts; it changes no physical law or prior evidence.

`extended-fit-01` retains all 24 states and all four preparation groups. The
three-center quadratic fit's worst total error is 2.892% (negative write,
seed 10101, wait 100, late signed moment), while its worst contrast error is
5.070%. Rank-4 projection is at most .139% on whole traces. Nine-variable shape
is marginal at 2.020% total/4.214% contrast; fifteen-variable mediator is
1.180%/1.655%, but weighted design conditions are 1.03e5 and 9.93e6 respectively.
No descriptor-insufficiency or mediator-necessity conclusion follows.

**Development repair 1, before its outcomes:** the current weight-10 paired loss
may trade total-response accuracy for already-adequate contrast accuracy. Repeat
the same finite fitting choices with pair weight 3, keeping the old fits. This
is a response-model repair with unchanged inputs/temporal rank, not a new state
claim or relaxed accuracy criterion. If needed, one compact descriptor repair
will add the current A probe/source overlap (integral of u times the fixed
unit-L2 source profile): it directly controls the instantaneous u-squared source
increment under the additive probe and distinguishes amplitude/phase at its
fixed actuator. This is a same-snapshot scalar, not response spectroscopy.

**Repair outcome and final selection.** `balanced-fit-01` uses exactly the same
linear/quadratic/ridge comparisons with paired-loss weight 3. Three-center
quadratic ridge 1e-6 now gives worst grouped total/contrast errors
1.4866%/5.1658%, with weighted design condition 29.07. Shape and mediator give
1.1633%/4.6743% and .9015%/4.5511%; their conditions remain 4.91e4/8.25e6.
Geometry and the best linear shape fit both use 80 response coefficients;
mediator uses 128. Thus the apparent earlier need for more inputs was partly a
fitting tradeoff. Select geometry, stop adding variables, and do not execute
the optional source-overlap repair. No current-feature collision or storage
necessity is established. Retain the failed weight-10 geometry fit.

`dev-refine-01` repeats the mixed history from seed 8101's same initial array,
with half dt and doubled N through both waits and both independent probe
branches. The declared shared floor remains 1e-12 for both readouts/windows.
This is newly checked numerical evidence, not inherited certification.
Science charged so far: 119.581 CPU seconds plus the fixed 30-second allowance.

**Fresh freeze, before preparation access.** Keep rank 4, the geometry quadratic
map/ridge 1e-6/pair weight 3 and the saved coefficients. Forecast independently
from three offsets, with no field or metadata queries. Fresh seeds are
12101/12102/12103; all have unwritten, odd +.4, and withheld odd +.2/even -.2
histories at waits 50/100. The withheld mixture has opposite even deformation
and intermediate odd displacement; it challenges training-history dependence,
though all states remain within the same prepared three-pattern regime. The
mean and B-only controls are also sealed. All panel predictions must be saved
before the first nonlinear reference outcome. The 2%/10% gates, h40..80 window,
floor rule, and final worst-contrast refinement stay fixed. No fresh-driven
repair is planned if this check fails; retain that domain limit.
