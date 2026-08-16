# Task 6 report — Golgi continuous similarity and morphology robustness

## TDD evidence

- **RED:** Added deterministic Golgi, perinuclear-ROI, and registered-DAPI fixtures
  plus synthetic tests before the module existed. The requested focused command
  failed at collection with `ModuleNotFoundError: No module named
  'cyclic_stability.golgi'`.
- **GREEN:** Added `cyclic_stability.golgi`, composing the existing consensus
  nucleus, physical threshold ensemble, resolution-normalized surface metrics,
  theoretical resolution, and atomic diagnostic publisher. Focused Golgi and
  morphology tests passed: `37 passed in 14.91s`.
- **Verification:** The complete synthetic suite collected 102 tests and completed
  successfully. `uvx ruff check` and `uvx ruff format --check` both passed on all
  Task 6 implementation and test files; `git diff --check` passed.

## Scientific boundary review

- Acquisition round 2 is the fixed reference irrespective of mapping order.
  Gradient correlation corrects each source background, smooths at every frozen
  physical sigma in ZYX voxel units, compares 3D gradient magnitudes only in one
  target-independent perinuclear ROI, and reports every scale plus its prespecified
  within-round median. Zero-variance gradients fail instead of returning a favorable
  correlation.
- The ROI is constructed once from the consensus DAPI surface: exterior to the
  nucleus, no farther than the frozen 18 um distance, and intersected with valid
  voxels in every registered Golgi round. Target masks never determine that ROI.
- Every configured sigma/SBR member is retained. Geometry retains every positive
  26-connected component, reports physical volume, count, finite-sample
  component-volume Gini, fragmentation density, exact compactness
  `36*pi*volume**2/surface_area**3`, surface-area-to-volume, and physical XYZ
  target-to-nuclear-surface distance quantiles.
- Jaccard, directed boundary distances, and resolution-tolerant surface Dice are
  always relative to acquisition round 2. Empty masks retain zero signal and
  volume, while derived geometry and agreement values are NaN so absence cannot earn
  favorable compactness or overlap.
- Absolute background-corrected target signal is retained as the primary signal
  endpoint. Round 1 remains labeled `round-1-exposure-sensitivity`; other rounds
  are labeled `matched-settings`, preserving the exposure-setting caveat without
  outcome selection.

## Scope and concern

The cell entry point alone reads TIFFs and writes atomic overlay/threshold-curve
diagnostics; the array kernel is numerical and only returns cell-level results. No
production data, Cy-ExM artifact/provenance rule, target-blind QC decision, or source
repository was modified. Production execution remains gated by the existing strict
provenance and QC boundary.

---

## Fix round 1/5 — common transform support and complete diagnostics

### TDD evidence

- **RED:** Added regressions for a finite target-padding slab excluded solely by an
  explicit transform-validity mask, strongly down-scaled and offset target signal,
  all-empty frozen members, 26-connected diagonal support, full endpoint/member
  coordinates, fixed-grid shape/validity validation, production cell delegation,
  composite diagnostic contents, and stale/failed atomic publication. The first run
  failed at collection because `ENSEMBLE_ENDPOINTS` was absent.
- **GREEN:** The focused Golgi plus morphology suite passed: `46 passed` (the
  20 numerical Golgi tests, 2 cell/atomic-diagnostics tests, and 24 morphology
  tests). The complete 111-test suite was then rerun in equivalent bounded
  runner batches: 51 non-Golgi/non-nuclear tests and 14 nuclear tests, alongside
  the 46 focused tests. Exact Ruff check/format gates passed after the fix.

### Fixes and boundary review

- The numerical kernel now accepts an explicit per-round binary fixed-grid
  `validity_masks` mapping and intersects all masks before building the DAPI-derived
  perinuclear ROI. The default all-true masks are documented as safe only for
  controlled same-grid array/synthetic callers; the production cell entry always
  requires explicit masks. It never inspects target values to decide support.
- `RoundArtifacts` identifies registered outputs and transforms but does not retain
  moving native source geometry, so this task cannot construct an exact transformed
  all-ones native mask without inventing geometry. The production entry therefore
  fails closed and requires caller-supplied masks generated from the validated
  nearest-neighbor affine/warp chain rather than silently marking padded finite
  output valid. Existing provenance and QC gates remain unchanged.
- Gradient degeneracy now means exactly constant sampled gradient values, avoiding
  an absolute `isclose` tolerance that rejected valid arbitrarily down-scaled signal.
  Each round emits the complete fixed ensemble-summary schema; undefined medians are
  explicitly NaN and mixed medians use only measured values.
- Diagnostic publishing now stores the consensus mask, fixed ROI, every frozen
  threshold member, and target arrays in an atomic NPZ; per-round composite MIP PNGs
  show round-2/current targets with ROI, nuclear, and union-of-frozen-member mask
  contours. Figures set PDF/PS Type 42 and SVG text-compatible font options. Stale
  diagnostic trees are replaced and a failed staged publication restores the prior
  complete tree.
- Nuclear distances are documented and characterized as physical XYZ voxel-center to
  nearest sampled mesh-vertex distances, an anisotropy-preserving approximation
  rather than an overclaimed exact point-to-triangle surface distance.

---

## Fix round 2/5 — autonomous provenance-bound validity and support-aware kernels

- The production cell entry is again exactly four positional arguments. For moving
  rounds it obtains the native target path from the validated registered-target
  completion request, verifies that immutable input identity, applies the validated
  persisted crop when present, and transforms a native all-ones image onto the
  registered round-2 grid via ANTs affine-before-warp nearest-neighbor application.
  Reference support is exactly its fixed grid. Transform-valid masks, crop/source/
  transform identities, and a deterministic validity digest are retained in result
  input identities; arbitrary caller masks are no longer accepted by production.
- `RoundArtifacts` now retains the audited native target and moving-crop evidence
  when a production channel-transform completion record supplies it. Synthetic
  records without that exact provenance remain `None`, causing production support
  reconstruction to fail rather than invent evidence.
- Array analysis now requires an explicit keyword validity mapping. Support-aware
  robust background, normalized Gaussian convolution, threshold masks, and gradient
  magnitudes exclude invalid regions before preprocessing. A regression changing
  only invalid support to `1e12` leaves correlations unchanged.
