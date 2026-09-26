# Critical review of the actual implementation and evidence

The review covers the diff against `241e232723871515c70189fa130903aca4549d27`,
the frozen execution at `ab6c6ad`, the portable data and the final claims.

- **Causal flow:** the primary `State.response()` accepts no arguments. Its step
  counter is unused by the readout; the scheduler alone dispatches events. State
  advancement multiplies retained response coordinates by fixed stable factors.
  Event excitation uses the known amplitude and inherited local geometry event
  rule. The original response memory is removed from the geometry submodel,
  avoiding hidden duplicate state. Current predicted centers are legitimate
  state. The exact-center override exists only as a separate evaluator call.
  No reference field/rate/response is passed into the primary runtime.
- **Timing and physical events:** the fixed unit-L2 laboratory source adds to u,
  leaves m untouched, and has disjoint B/C support. Four matched branches use
  the same integration boundaries. Age30 uses the original first slot; other
  ages use a zero first slot and the actual second slot. The diagnostic+.02
  probe is never applied to reduced state before reading its own response.
- **Arithmetic:** each R is probe minus its matching sham. D1 is the difference
  of independently initialized R forecasts. Tests replay every saved score and
  verify exact array subtraction. Duplicate R_without rows across signs are
  deduplicated in the360-gate count. Below-floor comparisons return unresolved,
  never a success. Absolute errors, maxima and peak timing remain visible.
- **Model selection and exposure:** all descendants stay with their preparation
  seed. Each grouped fit rebuilds centering, scale, temporal SVD and coefficients
  from training preparations only. Age18 appears in no fitting, normalization,
  mode/basis selection or candidate selection. The first flexible and conditioned
  fits remain failed evidence. The second bounded family preserves the inherited
  endpoint readout and adds a stable nullspace correction. Its constants and
  coefficients were frozen before20101–20103 outcomes. There was no fitting to
  paired-event outcomes or fresh-triggered parameter search.
- **Inherited distinctions:** original9 and11 are identical for a single event;
  algebra and saved development forecasts verify this. The new single-event
  linear realization of squared excitation is not evidence for repeated closure.
  Historical models, reports, numerical rules and failures are unchanged.
- **Numerics:** development age20 and adverse fresh age18 are refined from their
  saved prepared initial fields through the original write and event/probe
  history. Per-R matched uncertainties are conservatively triangle-propagated
  into D1. Direct D1 refinement is not substituted. Raw uncorrelated absolute-Y
  bounds, much larger than matched-response bounds, are retained and disclosed.
  The adverse floor does not change the frozen gates. The failure is resolved
  under both matched estimates, without claiming continuum or absolute-Y accuracy.
- **Interpretation:** the selected fresh panel fails despite strong improvement.
  Exact centers repair the adverse positive event but not the negative event,
  and impair some endpoint contrasts. The result implicates a coupled fitted
  timing/readout/geometry representation, not a proof of state insufficiency or
  identifiable physical storage. Explicit-age interpolation also fails; therefore
  this does not cleanly isolate causal-realization failure. Repeated evaluation
  is blocked by the explicit single-event decision, so no new R11/D12/D2|1/K12
  claim is made. All historical geometry and repeated-event limits remain.
- **Engineering and cost:** serialized state, static coefficient/basis costs,
  schedule/output buffers and multi-path operations are counted.72 checkpoints
  resume exactly in an isolated process. Safe outputs and grouped/information
  boundaries have fast fixtures. The science ledger includes failed fits and
  adverse qualification; routine tests and publication are separate.

One review clarification was material to the claims: exact centers improve the
withheld-age positive case while worsening global endpoint metrics. Reporting
only the global worst would incorrectly imply that geometry never matters.
The findings now report both facts. The normalized center singular values also
prevent the initial conditioning hypothesis from being stated as an established
cause. No scientific gate, frozen coefficient, future prediction or old evidence
was changed during review. Remote retrieval, CI and reviewed-head merge checks
are recorded in the PR rather than inferred from this local review.

Focused engineering fix: the fresh runner now verifies the privileged diagnostic
artifact hash as well as the primary model and sources. The published execution
already used the matching artifact (checked against its frozen hash), so no
prediction, reference, gate or scientific result changes. The historical
execution revision remains the one recorded in each protocol.
