# Present-state transmission evidence

This package preserves only the new snapshot-prediction cycle. The
[living findings note](../../experiments/mediated_patterns/PRESENT_STATE_TRANSMISSION.md)
owns the question, prospective choices, failures, final result and limits.
Earlier portable packages, field laws, reductions and historical conclusions
remain unchanged.

- [All fresh scores and worst cases](data/work/mediated_patterns/present_state_transmission/analysis-final/summary.json)
- [Actual/predicted R and Delta_R figure](data/work/mediated_patterns/present_state_transmission/analysis-final/predictions.png)
- [Frozen access contract](data/work/mediated_patterns/present_state_transmission/frozen-01/freeze.json) and [serialized models](data/work/mediated_patterns/present_state_transmission/frozen-01/models.json)
- [Initial fit](data/work/mediated_patterns/present_state_transmission/old-fit-01/fit.json), [changed-history miss](data/work/mediated_patterns/present_state_transmission/extended-fit-01/fit.json), and [loss-balancing repair](data/work/mediated_patterns/present_state_transmission/balanced-fit-01/fit.json)
- [Development refinement](data/work/mediated_patterns/present_state_transmission/dev-refine-01/refinement.json) and [worst fresh contrast refinement](data/work/mediated_patterns/present_state_transmission/fresh-refine-01/refinement.json)
- [Prediction seal](data/work/mediated_patterns/present_state_transmission/fresh-01/prediction-seal.json), [cost ledger](data/work/mediated_patterns/present_state_transmission/budget-total.json), [accounting](data/work/mediated_patterns/present_state_transmission/accounting-02/accounting.json), and [retained failed import](data/work/mediated_patterns/present_state_transmission/accounting-01/failure.txt)
- [Exact commands](data/work/mediated_patterns/present_state_transmission/commands.txt), [213 member hashes](artifacts.json), [critical review](review.md), and [publication receipt](publication.json)

The package has 13,214,589 substantive data bytes, below 25 MiB. It includes
the new preparations/checkpoints, descriptor and target objects separately,
all 17 candidates in each of three development fitting rounds, all sealed
fresh predictions and matched nonlinear response branches, both refinement
families, analysis scripts/receipts and failures. No inherited evidence tree
or release asset is duplicated. `old-fit-01` records checksum references into
the existing history package; old target curves used as training supervision
also appear in compact derived training matrices.

Schema: each panel's `rows.json` orders descriptor/target/prediction rows.
Checkpoint axes are `(wait, history, field, grid)`, with waits `[50,100]` and
history order given by **first appearance in `rows.json` for that preparation**.
Do not infer an axis order from sorted JSON object keys. `families` maps names
to `[odd,even]` amplitudes for evaluator bookkeeping; it is never a predictor
input. A `reference-i.npz` stores `(time, sham/probe, A/B/C, mass/moment/mediator)`
absolute readouts and its own C response. Current checkpoints are `(2,N)`;
descriptors are 15 available measurements in the documented order, with the
selected model receiving only the first three. `predictions.npz` contains no
reference outcomes. Test seeds are now exposed; do not reuse them as unseen
validation of a later repair.

Execution revisions and per-file source hashes are in protocols. In particular,
fresh prediction used `75544ab`; `49a0e4e` adds boundary tests only before final
refinement. Frozen scientific Python bytes are unchanged after fresh access.
The accounting script's `metered_cpu_seconds` aggregate includes a conservative
two-second failed-startup charge: strictly metered work is 350.904366 seconds,
with that charge plus a 30-second allowance giving 382.904366 seconds total.

Use the existing safe-copy helpers without importing scientific modules:

```bash
python3 -I evidence/present-state-transmission-2026-09-26/restore.py --verify-only
python3 -I evidence/present-state-transmission-2026-09-26/restore.py --destination /chosen/new-empty-directory
```

Restoration rejects traversal, symlinks and existing destinations. Git and
checksums make bytes inspectable, not platform-immutable. Remote retrieval and
confirmed merge state are recorded separately; final merge metadata belongs in
the PR comments, without a metadata-only successor PR.
