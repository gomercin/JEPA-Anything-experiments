# Critical pass: source-to-receiver relay

This is the implementing agent's explicit critical review, not an independent
reviewer or replication. Reviewed the diff against merged bf21ecb, the actual
ETDRK4/control routines and new runner/analysis/tests, the saved development and
fresh arrays, conservative refinement calculations, mask diagnostic, source/work
accounting and final report. The final PR head is pinned in the publication
receipt and PR description; merge uses a head-SHA guard.

Findings and dispositions:

1. **Attribution could have overclaimed a one-pass relay.** A responds to the
   B-neighborhood cut too. Report labels Q_B as loop-wide pathway dependence;
   no source replay was used to claim isolation. Direct u propagation, source
   changes and static conditioning remain in K. This is an interpretation bound,
   not a hidden total causal decomposition.
2. **Prepared endpoints alone were insufficient drift checks.** Before the
   freeze, analysis was strengthened to derive every saved branch centroid from
   its fixed M/P readouts. All fresh cases pass; a regression catches transient
   excursions even when endpoints look safe. No field run changed or was rerun.
3. **The prospective larger-amplitude F prediction partly fails.** All three
   B=0 cases exceed 2%; preserve the failed gate. Q_B predictions pass their
   distinct 10% gate. Report does not promote the whole prediction panel to PASS.
4. **Mask and comparator limitations matter.** The shared primary mask removes
   boundary movement as an explanation. Translated-mask delta-Q sensitivity is
   still 1.04%/3.54%; report retains this. The Q common-delay fit reaches -8,
   so no universal gain/delay exclusion is claimed. One input direction does
   not establish source selectivity.
5. **Conservative resolution and pairing:** floors use separate F/K errors and
   sum both organizations for delta-Q; direct cancellation is diagnostic only.
   All arrangements/adverse cases remain. Same component arrays/residual draws
   are paired, while actual states, work and source differ and are reported.
   The floor is empirically calibrated on one development preparation.
6. **Provenance and budget:** clean committed scientific sources before every
   stage, with exact revision/hash protocols. No snapshot is retrospectively
   assigned a later revision. Aggregate CPU receipts include the brief overlap
   of two diagnostic processes; neither is omitted. 15 seconds are additionally
   charged for brief untimed numerical inspection. Fresh choices were frozen
   before any 9101/9102/9103 outcomes.
7. **Preservation and software:** old solver/control/reduction/report/evidence
   bytes and checkpoint overwrite/symlink fixes are unchanged. New output
   directories reject reuse and symlinks. Evidence restores through the existing
   safe-copy helpers, scopes paths to this cycle and refuses overwrites.

Validation: `make check` passed: 66 core, 33 skill (+2 subtests), 45 fast
oscillator, 69 field, 22 evidence tests. One original oscillator scientific
panel was deselected; one session-local historical field-manifest check was
skipped in the isolated checkout. Neither is called reproduced evidence.
Focused Ruff/format checks and `git diff --check` passed. Five new instrument
checks and two new publication checks are included in these totals. No old
scientific panel, training, frozen model, or historical archive was rerun.

Disposition: no material software or evidence blocker found. Scientific result
is bounded positive pathway/organization dependence with a partial approximation
failure; merge readiness does not turn that failure into success. No successor.
