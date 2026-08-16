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
