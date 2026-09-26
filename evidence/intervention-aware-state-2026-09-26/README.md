# Intervention-aware state: portable evidence

The [findings](../../experiments/mediated_patterns/INTERVENTION_AWARE_STATE.md)
report a bounded nine-scalar response-prediction success after one actual local
conditioning event. Original three-center G+J+F fails the event-contrast target;
the selected extension passes fresh response gates while retaining physical
geometry limits and development failures. [Critical review](review.md) is explicit.

`artifacts.json` indexes only new evidence under
`data/work/mediated_patterns/intervention_aware_state/`. Old evidence is neither
copied nor modified. The package is Git-resident, below25MiB. No release asset,
tag replacement or latest archive is involved. Each stage protocol records its
source commit and source hashes; `commands.txt` lists the exact invocations.
Use that source commit when reproducing a historical failed/earlier fit. The
final runner's repair-fit stage implements the final readout form.

Verify without simulation:

```bash
python evidence/intervention-aware-state-2026-09-26/restore.py --verify-only
python -m pytest evidence/intervention-aware-state-2026-09-26/tests -q
```

Restore exclusively into a new destination:

```bash
python evidence/intervention-aware-state-2026-09-26/restore.py --destination /tmp/intervention-evidence-new
```

The runtime audit copies only the solver-free runtime modules, static model and
retained checkpoint into an isolated directory. Scientific imports and field
array reads are unavailable. All24 selected fresh checkpoints reproduce the
uninterrupted predictions exactly. The evidence tests validate every score and
four-branch target arithmetically, and do not launch scientific panels in CI.

Authoritative inherited artifacts remain:

- E: `experiments/mediated_patterns/present_state_model.py`, `extract(..., "geometry")`.
- G: `evidence/geometry-evolution-2026-09-26/data/work/mediated_patterns/geometry_evolution/frozen-01/models.json`, SHA256 `4b42006337f4e27c1149211e724ffeef022e9b6391c46aee27d96ecb9c038477`; quadratic selected, affine control.
- F: `evidence/present-state-transmission-2026-09-26/data/work/mediated_patterns/present_state_transmission/frozen-01/models.json`, SHA256 `f0bc4ced6914bc43148d615912619370465ce48332a140dd41af52b7b85bb74e`; geometry model, fixed+.02 probe only.

The initial seeds8101/10101/10102/10103 and all their descendants are development.
Fresh16101/16102/16103 are grouped by initial preparation. The withheld variation
is conditioning amplitude+.01 at unchanged times/gaps.432 scored entries include
repeated shams; there are336 distinct scalar gates and three independent seeds.
No failed panel is fresh again. No further scientific successor is selected.
