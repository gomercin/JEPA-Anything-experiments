# Critical review of the history experiment

Review is a direct critical pass by the implementing agent, not an independent
scientific replication or external approval. Final results/check status and the
reviewed Git head are recorded in the PR and publication receipt.

- **Four histories:** initial complete arrays are shared exactly. Only the
  written history receives the compact finite write. Probe branches start from
  the appropriate saved states at 10, 50 or 100; no new preparation/time reset
  substitutes for the aged unwritten state. Separate sham subtraction removes
  additive write output but does not assume away nonlinear interaction.
- **Event/frame boundaries:** all event times are exact grid points; omitted
  events follow compatible fixed-step partitions. B write support radius 8,
  fixed A probe support radius 8 and fixed C readouts are disjoint. Diagnostic
  centers/projections never feed dynamics. The explicit absolute-time matching
  guard was added and tested before fresh evaluation.
- **Tangent/control:** the reused tangent differentiates the actual ETDRK4
  stages. Each written/unwritten branch supplies its own future unprobed sham.
  The B feedback cut starts only at the probe, retains its history's background,
  source law and direct coupling, and uses the same physical mask. It tests
  expression, not storage or writing. A loop response prevents one-pass claims.
- **Numerics:** half timestep and doubled mesh redo the finite write, entire
  waiting interval and probe, beginning from the same prepared state (Fourier
  interpolation for mesh refinement). The shared floor sums individual response
  errors across both histories/full/cut before multiplying by five, with a
  prospectively stated 1e-12 lower bound. Favorable contrast cancellation does
  not set the floor. State differences/refinement errors are retained separately.
- **Explanations:** translation-like field differences, small remaining shape
  changes and continuously sourced mediator differences are ordinary present
  state. The exact homogeneous mediator operator isolates inherited residue
  conditionally on the recorded coupled histories; it does not furnish causal
  fractions. There is no unique localization of storage to B, new law, learned
  actuator, practical threshold, permanent memory or autonomous predictor.
- **Information boundary:** only seed 8101 was used for choice/qualification.
  Source hashes, write, waits, floors and interpretation/prediction criteria
  freeze before seeds 10101–10103. All delays are paired observations within a
  preparation, not extra independent seeds. Tangents are persisted before each
  fresh nonlinear reference. Descriptive output gains are not held-out fits.
- **Safety/provenance:** unique output directories and files reject existing
  targets and symlinks. Scientific stages are serialized with an exclusive lock,
  enforce CPU/wall/memory stops, and retain receipts on failure. Scientific
  source must be committed/tracked. Existing checkpoint safety and all earlier
  scientific sources, reports, gains, arrays, releases and source snapshots are
  unchanged. Public evidence is scoped to this cycle.

The initial even candidate and all development choices remain visible. No
outcome-driven repair or extra write search was needed. Focused changes before
fresh testing improved absolute-time checking and output/provenance safety;
these did not change the physical events or outcome arrays.

## Completed local checks

`make check` passed: 245 tests plus two subtests, including the prior checkpoint
safety regressions and new publication checks. One session-local manifest check
is intentionally skipped in an isolated checkout; the historical oscillator
scientific-panel test remains excluded from CI. Scoped Ruff and whitespace
checks pass. All 149 new member hashes verify locally; new-package tests also
verify exact aged sham continuations, subtraction, frozen-copy equality and
prediction-before-reference ordering. The actual figure was visually inspected.
No material scientific or software blocker remains in this reviewed scope.
Remote retrieval, hosted checks and actual merge remain separate requirements.

Fresh remote retrieval at `0f4fd6b09b1e93ddd9589c53deaf012e12e928ec` verified
149 member/restoration hashes, 179 source hashes across seven protocols, and
66 finite NPZ files. No scientific module was imported or field advanced.
Both hosted checks passed at that head. The remaining diff is publication
metadata and links only; the final head/base and hosted checks must still be
checked immediately before guarded merge. No old package was republished.
