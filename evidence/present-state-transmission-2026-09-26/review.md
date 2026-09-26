# Critical review

This is a substantive second pass by the implementing agent, not independent
replication or external approval. The reviewed head and actual remote checks
are recorded in the PR. No subagent or outside reviewer is represented.

1. **Snapshot boundary.** `present_state_model` imports NumPy and measurement
   windows, never the solver. Extraction accepts one current array; the loader
   selects checkpoint times/states without reading future sham/readout members.
   The selected inference API rejects anything except three numeric descriptors,
   fixed probe and declared forecast times. Paired and batch-independent inference
   tests, poisoned future/twin arrays and blocked solver imports test this flow.
   Seed/wait/history labels reside only in evaluator rows and grouping/pair logic.
2. **Conditional response.** Each evaluator starts sham and .02-probe branches
   from the identical current state. The saved reference hash matches the selected
   current checkpoint. R subtracts that state's sham; Delta_R subtracts independently
   predicted responses. Saved-data tests independently recompute branch subtraction,
   every score/denominator/gate and the shared floor. No write-only offset enters
   a denominator. Both mass and signed moment use all 161 times and h40..80.
3. **Preparation grouping and fitting.** Four development preparation groups keep
   every descendant in one fold. Means/scales, temporal SVD and ridge fitting are
   recomputed inside each fold. The two varied-write preparations increase history
   support, not seed replication. Each descriptor set receives the same finite
   model choices. Geometry's selected ten polynomial terms and rank-4 two-output
   coefficients have a well-conditioned weighted design (29.07). Shape/mediator
   conditioning and capacity limits are explicit. A paired-loss rebalance was
   chosen from development discrepancy and retained alongside the initial miss.
4. **Fresh exposure.** `75544ab` seals coefficients, criteria and the opposite-even
   intermediate history before new seeds. All descriptor-only predictions are
   written before the first reference loop, not merely before scoring. Source,
   model, descriptor and prediction hashes agree. Saved predictions reproduce
   from serialized inputs alone. No model or scientific Python source was changed
   after fresh access; the only later lab code addition was two boundary tests.
5. **Numerical resolution.** Half dt and double N repeat a representative new
   history and the final worst contrast through both waits/probes, from the same
   prepared fields. Floors use five times the sum of independent response errors,
   not favorable contrast cancellation. The maximum branch sum is 1.2532e-13;
   the shared 1e-12 guard remains conservative. Sub-floor contrasts would be marked
   unresolved. Primary success does not make claims about Delta_Q or early
   pointwise relative accuracy.
6. **Claims and adverse evidence.** Three centers are adequate in this envelope;
   B-only failure and the initial geometry miss remain. The rank-4 approximation
   to the development-mean baseline predicts zero contrast and scores 100% error.
   Larger descriptor fits are development comparisons, not fresh necessity tests.
   A fit repair succeeding without extra inputs directly limits interpretation of
   the earlier miss. No collision witness, minimality, exclusive storage, autonomous
   closure, sequential intervention or universal history transfer is claimed.
7. **Cost and safety.** All new stages have receipts; total charge includes the
   failed accounting import and an explicit startup/inspection allowance. The
   fresh/refinement reserve is respected. Reference simulation uses the existing
   CPU/wall/memory guards and serial lock. NPZ and JSON destinations are exclusive.
   Acquisition, static basis/coefficients, transient scans and forecast costs are
   separate. Small descriptor count is not equated to small serialized storage.
8. **Publication scope.** Only new evidence is copied under the portable convention.
   The original report, sources and prior packages are outside the diff. Restore
   uses existing validated helpers; new tests check hashes and non-overwrite.
   Checkpoint history order is explicitly documented through `rows.json`, avoiding
   ambiguity from alphabetically serialized JSON keys. No credentials, private
   correspondence, unrelated data, paid compute or overwritten assets are present.

The actual/predicted figure was visually inspected. No material scientific blocker
remains. Local fast tests, remote retrieval, hosted CI and guarded merge are
separate requirements; publication.json and PR comments carry their actual status.

Local validation completed: `make check` passes 257 tests plus two subtests, with
the existing session-manifest skip and scientific-panel deselection. Scoped Ruff,
whitespace and all 213 evidence hashes pass. The 12 focused instrument/publication
tests recompute sealed inference and scores without field evolution.

Fresh GitHub retrieval at `3567e995c4b273021ff616970283d8c09135304a` verified and
restored all 213 members, checked 288 source hashes across ten protocols and
loaded 115 finite NPZ files without importing scientific modules or rerunning
science. Both hosted checks passed at that source/evidence head. The final delta
adds only this verification receipt and publication pointers; recheck its exact
head/base and hosted status before guarded merge.

Final CI found a publication-test portability defect at `3303ab2`: one Linux
runner recomputed five descriptor values with maximum difference 1.11e-16,
while an exact-array assertion required bit identity. The saved arrays/model
hashes and prediction replay all passed. The test now allows 1e-13 relative /
1e-15 absolute roundoff for re-extraction only; current-state and model hashes
remain exact. No predictor, evidence bytes, floor, scientific gate or outcome
changed. The affected 12 focused tests pass locally; both hosted checks must
pass at the new reviewed head before merge. The original failure remains in
[run 36256491548](https://github.com/gomercin/JEPA-Anything-experiments/actions/runs/36256491548).
