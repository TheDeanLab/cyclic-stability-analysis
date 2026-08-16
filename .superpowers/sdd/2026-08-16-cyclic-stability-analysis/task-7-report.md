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
