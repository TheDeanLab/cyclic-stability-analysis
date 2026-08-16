# Task 7 report — gated LAMP1 object stability analysis

## Scope

Implemented the per-cell LAMP1 gate, physical 3-D LoG/watershed detector, and
finite one-to-one resolution-aware matcher.  Cy-ExM and its source repository
were not modified.

## Scientific and operational behavior

- Only complete five-round, image-verified LAMP1 `CellSpec` values are eligible;
  no numeric identity is inferred.
- LoG scales are converted from physical micrometers to anisotropic ZYX voxel
  sigmas.  Robust background failure is a hard error.  Peaks use a 26-neighbour
  maximum criterion, physical-scale duplicate suppression, positive-response
  watershed support, and physical volume/surface geometry.
- Matching is Hungarian only over finite pairs within the supplied resolution-unit
  limit.  Lost and new objects are explicit and use reference/moving denominators
  respectively, including empty-side cases.
- Per-cell output retains all LoG/SBR ensemble coordinates, reports round-2 and
  adjacent-round comparisons, and atomically publishes diagnostics.
- The production LAMP1 validation gate exits successfully with a machine-readable
  `unavailable` status and `scheduled_cell_count: 0` while no mappings exist.

## TDD evidence

Before `lysosome.py` existed, the focused test command failed during collection as
expected with `ModuleNotFoundError: No module named 'cyclic_stability.lysosome'`.
After implementation, the focused detector/matcher/gate suite passed.

## Verification

```bash
export UV_CACHE_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/.cache
export UV_PYTHON_INSTALL_DIR=/project/bioinformatics/Danuser_lab/Dean/dean/uv/python
uv run --script scripts/run_tests.py tests/test_lysosome.py -q
uv run --script scripts/run_tests.py -q
uvx ruff check scripts/cyclic_stability/lysosome.py scripts/validate_inputs.py tests/synthetic.py tests/test_lysosome.py
uvx ruff format --check scripts/cyclic_stability/lysosome.py scripts/validate_inputs.py tests/synthetic.py tests/test_lysosome.py
uv run --script scripts/validate_inputs.py --analysis-config config/analysis.yaml --correspondence config/correspondence.csv --qc-decisions config/qc-decisions.csv --cohort lamp1 --run-dir results/lamp-gate
```

Results: focused tests `15 passed`; full tests completed successfully; Ruff checks
and formatting passed.  The production gate reported:

```text
unavailable: no image-verified LAMP1 cell mappings; scheduled 0 cells
```

The resulting `results/lamp-gate/validation/inputs.json` has `status` equal to
`unavailable`, the explicit no-mapping reason, and `scheduled_cell_count` equal to
zero.

## Review-fix amendment

The LAMP1 implementation now composes the established consensus nuclear surface
and producer-attested transform-valid support machinery.  It gates every moving
round on target-blind QC before opening a target volume, measures all object
distances to one accepted-DAPI consensus surface, and detects only in the common
valid support outside that nucleus.

The LoG response is now assembled from per-axis voxel-coordinate second
derivatives, each normalized by its squared voxel sigma.  This represents the
scale-normalized physical Laplacian in intensity units and has regression coverage
across three anisotropic samplings.  Peak duplicate suppression now uses the
theoretical lateral/lateral/axial resolution reference.

Long-form `objects.csv` and `matches.csv` are atomically published alongside the
NPZ diagnostic.  They retain stable object/match indices, coordinates, physical
measurements, comparison type and round identities.  The comparison helper emits
round-2-relative and adjacent results, including fractions and median lateral,
axial, and normalized displacement (NaN where no pair exists).

Additional focused coverage verifies missing/excluded QC behavior, producer-support
intersection and empty-ROI rejection, realistic distinct DAPI/LAMP fixtures,
atomic stale-output replacement, table schemas/content, and singleton NN handling.

Final review-fix verification: `22 passed` for `tests/test_lysosome.py`; full tests
completed successfully; exact Ruff check and format checks passed.

## Review-fix amendment 2

LAMP1's supplied common ROI is now preprocessing support, not merely a final mask.
Background estimation uses only positive support voxels; scale smoothing uses masked
normalized convolution; and an explicit support-safe three-point physical second
derivative is evaluated only after the actual derivative stencil is fully inside
support.  Invalid padding values therefore cannot influence detections, object
geometry, or output rows, while insufficient support is a hard error.

Adjacent comparisons now require consecutive acquisition numbers after QC filtering:
an accepted `2,4` pair is retained as round-2-relative but is not called adjacent.
Input identities now list only accepted-round DAPI/LAMP artifacts plus the exact
support provenance, so excluded inputs are not represented as contributors.

Regression coverage adds arbitrary high-valued padding invariance, thin-support
rejection, QC-gap adjacency behavior, and excluded-round provenance. Final focused
verification: `25 passed`; the full suite and exact Ruff/format checks completed
successfully.
