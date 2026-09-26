# Critical review of the intervention experiment

Scope: actual diff from main36c25209, new compact runtime, opt-in evaluator,
all development failures, frozen fresh predictions, scores and portable evidence.
This is an explicit self-review, not an independent human review.

## Information and physical boundaries

- E is the original three weighted centers at t50. Predictor code imports only
  NumPy and solver-free geometry/snapshot modules. Labels, seed and old-write age
  exist only in evaluator grouping. No field-derived t1 moments, true jump,
  current t2 centers, actual no-event response, reference cache or PDE enters it.
- The primary two histories initialize separately. Each runs continuously from
  t50. All predictions/checkpoints are written and hashed before post-t50 fields.
  True-center F and true-jump G are explicitly evaluator diagnostics.
- q is the existing fixed periodic A profile, additive in u, unit L2, disjoint
  from B/C windows. m remains unchanged. Expansion of u² is tested using the
  implemented window/local ell/dx. It is not proof of reduced-state inadequacy.
- Each gap branches an alternative probe from the conditioning-only trajectory.
  Y00/Y10/Y01/Y11 use matching fixed integration boundaries. Targets are
  Y11-Y10 and Y01-Y00; the first event's lingering output is removed.
- Mass and signed moment, whole/late windows, R and D denominators remain
  separate. New full-history dt/N refinements determine floors. Frozen scores
  are retained even if supplemental refinement changes uncertainty estimates.

## Candidate and exposure review

- Original E/G/F and all old artifacts are unchanged. Physical jumps supervise
  local J only. Physical impulse curves supervise the ERA geometry only.
  Response regression never adjusts the physical centers to cancel F error.
- All descendants group by original preparation. Training-only standardization,
  SVD temporal bases and coefficient fitting are repeated inside each fold.
  Four preparation groups are development, three untouched seeds are fresh.
  +.01 conditioning is withheld; negative conditioning never becomes a negative
  diagnostic probe for F. Fresh data do not select models or repair coefficients.
- The selected construction has nine scientific state scalars, not three: three
  physical centers, four event-deformation coordinates and two response-memory
  scalars. Zero auxiliary state means no new conditioning event relative to the
  baseline model; it does not assume the old write has fully equilibrated.
- Two repair families were explored: order4/6 physical transient realization
  and low-rank readout correction. Failed linear and interaction corrections
  remain in separate immutable stages. The final response memory uses a declared
  fixed20-unit exponential basis, not a uniquely identified physical time scale.
- Selection favors order4 for the response objective despite order6's smaller
  development geometry residual. Order6's spectral radius>1 is disclosed and
  its use is limited to the finite interval. No long-run stability claim follows.

## Findings requiring limits or fixes

1. The first development process failed importing the analysis module. Fixed the
   parenthesis before simulation; preserve the failed receipt and source commit.
2. Initial readout corrections failed resolved D even while all large R errors
   passed. Retained, followed by one directly driven response-memory form within
   the declared readout family. No threshold, amplitude or sensor change.
3. The selected four-mode geometry misses small induced C motion and retains
   early physical-coordinate errors. The report must not infer accurate geometry
   or minimal state from a response pass. No fresh-driven scientific repair.
4. A scalar mean error could conceal contrast/sign failures. Publication tests
   replay every saved score from its original four branches, separately per
   output/window; sign and peak timing are retained in each score record.
5. Restart in the same process is insufficient. The audit copies only four
   runtime modules, static coefficients and one checkpoint into an isolated
   directory, denies scientific imports and all .npz/.npy reads, and replays
   all24 selected fresh checkpoints in new processes.

No evidence supports arbitrary intervention closure, full absolute two-pulse
trace prediction, a practical switch, unique storage, or minimality. A bounded
response success with physical limitations is a publishable result. Fast tests,
portable integrity/retrieval, live base/head comparison and guarded merge remain
publication checks, not scientific-success criteria.

## Outcome and final claims review

Fresh:432/432 saved entries pass, but96 are repeated sham gates; report336
unique gates,48 R traces,36 D contrasts, three independent preparations. The
selected model's max R errors .322%/.404%, D4.314%/3.544%; withheld+.01 D4.309%.
All54 fresh absolute geometry gates pass, while induced-motion B/C relative
errors31.2%/100.0% do not. Small C effects are resolved by matched subtraction
under refinement. This boundary is shown explicitly in the geometry figure.
The supplemental larger numerical floor does not change any conclusion. No
fresh-driven fitting occurred and the larger fresh-better control is not
retroactively selected. All original source/artifact bytes are unchanged.

Final local validation: `make check` passes286 tests plus two subtests, with one
existing unavailable-manifest skip and one intentionally excluded historical
scientific CLI panel. The new three evidence tests replay all scores and hashes.
A fresh GitHub sparse clone independently retrieves all504 members/23,830,038
bytes and verifies every size/SHA256 at5d1d85e. No historical corpus is audited
or scientific panel rerun for this retrieval. Live CI/base/head state is checked
at merge time and merge confirmations are posted on the PRs.
