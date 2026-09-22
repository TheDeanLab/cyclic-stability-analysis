# Lamin watershed integration and rerun implementation plan

> Execute inline, using the existing committed analysis checkout and focused
> regression tests. The user authorized integration and the full Lamin rerun.

**Goal:** Remeasure all five Lamin fields with the validated physical watershed,
record the unresolved Cell4/N003 exclusion, and regenerate the figure.

**Architecture:** Reuse the existing consensus foreground, adjudication, surface
realization, task runner, and plotter. Enable watershed through an optional
scientific parameter; keep legacy configurations unchanged. Record a spatial,
voxel-count-guarded exclusion in the run configuration before surface realization.

**Tech Stack:** NumPy, SciPy, scikit-image, existing PEP 723 scientific environment.

**Design evidence:** `results/lamin-watershed-validation-20260921/README.md` and
`results/lamin-watershed-seed-pilot-20260921/README.md`.

## Constraints and reuse analysis

- XYZ spacing is in expanded micrometers; arrays use ZYX. Preserve foreground
  construction and 26-connectivity; h=2 µm controls peak prominence, not separation.
- Closest existing symbols: `_watershed_nuclear_labels` in the committed Golgi
  worktree; `adjudicate_nuclear_components`, `realize_nuclear_surfaces`,
  `analyze_lamin_cell`, `load_analysis_config_bytes`, `run_tasks`, and
  `plot_lamin_run.completed_summary` in the analysis checkout.
- Port the watershed helper only. Do not port unrelated normalization or Golgi
  changes, duplicate the runner, or alter existing round registration QC.
- Extend immutable configuration with optional seed prominence and explicit
  Lamin component exclusions. Exclusions select a component by a physical point,
  verify its exact expected voxel count, retain its QC/diagnostic row, and fail if
  the expected component no longer matches. Existing configs have neither change.
- Plot actual `N###` rows only; `aggregate` rows are summaries, not extra nuclei.
  Resolve terminal-selected attempts to support restart-safe runs.
- Preserve old outputs. Use a new frozen Lamin config and a new result root.
  Save old-to-new component correspondence and exclusion evidence beside the run.

## Tasks

- [x] Add focused regression tests: physical/anisotropic watershed, 26-connected
  foreground preservation, optional configuration and exclusion validation,
  immutable exclusion/QC retention, aggregate-free terminal-selected plotting.
- [x] Port the validated helper, extend configuration, and compose exclusion before
  existing surface realization. Document units, fail-closed exclusion behavior,
  and the run-specific settings. Fix the existing plot loader.
- [x] Run targeted and relevant broader tests; distinguish the two documented
  baseline failures from new failures. Format/lint changed files, review diff,
  and commit the implementation locally with this reuse analysis.
- [x] Run a bounded real-data Cell3 smoke test using the new settings and the frozen
  validated mask, inspect daughter surfaces and sampled round-2 Lamin profiles,
  and only then run the five-field workflow. Historical full fields require
  53-119 minutes each; the bounded smoke is not a production measurement.
- [x] Verify terminal records and all required artifacts, map original parent IDs
  to revised nuclei/exclusion, regenerate PNG/PDF/provenance with
  `scripts/plot_lamin_run.py`, inspect the plot, and commit the publication outputs
  and run report. Report artifact completion separately from scientific limits.

## Inspection findings incorporated in scope

- The old plot treated five aggregate rows as nuclei (19 displayed versus 14 actual).
- Surface realization computed filtered/remapped faces but returned original faces.
  A regression with one discarded curvature vertex demonstrates an out-of-bounds
  face index. Return the already computed faces; keep the existing 99% validity rule.
- Persist exact consensus labels and their nucleus-ID lookup as a diagnostic artifact
  to support the requested parent-to-daughter mapping and reproducible QC.

## Pre-launch validation

- Full suite: 445 passed, 2 pre-existing failures (373.41 seconds).
  The unchanged failures are the LAMP1 exposure-exception fixture replacement
  and the nuclear strict-failure monkeypatch missing `allow_degenerate`.
- Final inventory/configuration/plot integration subset: 21 passed.
- Focused watershed, spatial exclusion, face-remapping, and saved-label checks:
  14 passed. Modified-file Ruff/format and diff whitespace checks pass.
- Models/configuration static check with the scripts import root: zero errors
  and warnings (standard mode, existing Python 3.12 environment).
- Bounded real-data smoke passed: Cell3/N002 has the same two-region partition
  as the seed pilot (allowing centroid-based ID permutation). Both daughters
  produce supported round-2 Lamin profiles; Cell4/N003's spatial exclusion
  matches exactly 7,248,164 voxels. Smoke samples are not production results.
- The DAPI-only inventory now applies the same explicit exclusions as Lamin
  measurement; its regression and existing entry-point tests pass.

## Completed run

- Run `20260921T213601Z`: all five fields completed; plotting completed at
  2026-09-22 00:49:52 UTC and the full audit passed at 00:50:55 UTC.
- Accepted counts: 1, 4, 3, 3, 4 (15 total). Both expected pairs split; the
  original Cell4/N003 is excluded, and every original parent is accounted for.
- The audit verified 482 content identities and 240 plotted measurement rows.
  At final review all recorded sizes/mtimes remained unchanged, and all five
  terminal-selected records still matched their recorded SHA-256 digests.
- The final overview PNG was visually inspected. Figure, provenance, parent/daughter
  mapping, plotted measurements, comparison table, and report are saved under `plots/`.
