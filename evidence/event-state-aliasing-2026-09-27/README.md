# Fixed-age event-state aliasing: portable evidence

**CENTER-ONLY APPROXIMATION SUFFICIENT** in the fixed t72 event / t90 probe assay.
The [findings](../../experiments/mediated_patterns/EVENT_STATE_ALIASING.md) preserve
all previous failures. A competent quadratic on three current centers passes
960 grouped development gates, then 120 fresh snapshot gates. The unchanged G
plus the same response map passes another 120 gates on three further untouched
preparations measured only at t50. No extra descriptor or event-memory state is
added. This does not qualify age transfer or repeated-event composition.

The five response-blind selected near pairs do not support an aliasing witness:
the response differences are accounted for by resolved center differences under
the declared sensitivity screen. Overlap, mediator and mass are diagnostic
measurements only. Ridge .1 fails two grouped gates and is retained; the selected
ridge is 1e-6. No RBF or feature ladder was needed.

This incremental package includes new current fields/descriptors, pair seals,
all new future branches, matched half-dt/double-N checks, all candidate fits,
train-only fold metadata, fresh predictions/seals, source identities, costs,
scores, checkpoints and figures. Historical models and archives are referenced
by their original identities. The two development `initial-50.npz` copy-only
outputs are omitted: their exact historical sources, row indices and hashes are
in `inputs.json` and `rows.json`; all new t72 and genuinely fresh fields remain.
Derived matched training targets retain reused old responses with source hashes.

```bash
python evidence/event-state-aliasing-2026-09-27/restore.py --verify-only
python evidence/event-state-aliasing-2026-09-27/restore.py --destination /private/tmp/aliasing-evidence-new
python -m pytest evidence/event-state-aliasing-2026-09-27/tests -q
```

The restored root is `work/mediated_patterns/event_state_aliasing/`.
[Exact commands](commands.txt) include every scientific execution revision.
Earlier exposed input restores are documented there and in their existing
indexes; no historical panel needs rerunning. Safe restoration validates every
member and rejects overwrite, symlink, traversal and hash mismatches. Tests do
not run field simulations. [Critical review](review.md) covers information flow,
matching, approximation, numeric resolution, exposure and claim boundaries.

- [Fresh R and D1](figures/fresh-responses.png): actual pre-event centers, three new preparations; solid reference and dashed prediction.
- [All selected matched pairs](figures/matched-pair-responses.png): response differences and preparation-held-out prediction differences.
- [Pre-event geometry](figures/pre-event-geometry.png): true versus G-propagated center motion in the separate once-measured fresh panel.
- [Retained-input R and D1](figures/retained-responses.png): the same snapshot response map evaluated on G-propagated centers, on three further new preparations.

Fresh D1 signal RMS spans roughly 9e-12–3e-10; numerical floors are
2.9e-13–4.5e-13. Every fresh D1 residual RMS is below those empirical floors;
reported sub-percent residuals do not establish that continuum precision. R and
D1 are scored separately, with no unresolved signals counted as passes.
Scientific cost is 674.72 CPU seconds (11.25 minutes), including a 30-second
inspection allowance, below the 30-minute cap. The selected map has 1,538 active
fitted/basis/scaling values; unchanged G adds 37 and amplitude normalization one.
It retains three centers plus a counter, with a declared amplitude input and
322 output values per materialized conditional-response forecast. No full Y
trace, post-event physical geometry or multi-event update is claimed.

See `artifacts.json` for archive/member checksums and the exact publication
revision. PR comments record reviewed-head checks and actual merge confirmations;
no metadata-only successor PR is used.

Package: 453 members, 23,680,179 uncompressed bytes; 10,823,446 compressed bytes.
Archive SHA256 `d24b4f2004ae444ce7a9b345f5018f958525438423a7fb3c66893d9dc4501ce3`.
Local `make check` passes 324 tests and 2 subtests, retaining one inherited skip
and one historical scientific-CLI deselection. This includes eight new synthetic
instrument tests and five new portable evidence tests; six isolated checkpoint
processes reproduce saved predictions exactly. Archive verification restores all
453 members without executing experiments. The generated Git evidence is below
25 MiB.
