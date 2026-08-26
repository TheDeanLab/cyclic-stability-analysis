# Multinuclear Consensus Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single-largest-nucleus assumption with a target-blind consensus that accepts one through six geometrically plausible nuclei and performs Lamin analysis per nucleus.

**Architecture:** Construct and adjudicate connected DAPI components once, expose immutable per-nucleus surfaces plus a union mask, and carry an explicit `nucleus_id` through scientific outputs. Golgi, LAMP1, and raw-signal analysis consume only the accepted union mask; Lamin iterates individual nuclear surfaces and isolates failures to individual geometry endpoints.

**Tech Stack:** Python 3.12, NumPy, SciPy ndimage, scikit-image, Matplotlib, PyYAML, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-08-26-multinuclear-consensus-design.md`

## Global Constraints

- Accept exactly 1 through 6 geometrically valid DAPI components; fail closed outside that range.
- Configure nominal biological volume bounds as 100.0--5000.0 um3 and convert them by exactly `nominal_expansion_factor ** 3`.
- Require minimum sphericity 0.15 and maximum contact fraction 0.10 on every image face.
- DAPI alone determines nuclear inclusion; Lamin, Golgi, and LAMP1 must not influence it.
- Preserve the complete threshold mask for volume and overlap measurements.
- Use stable centroid-ordered identifiers `N001` through `N006`; use `aggregate` for field-level rows.
- Preserve the existing atomic diagnostics publication boundary and Matplotlib PDF/PS font type 42.
- Do not edit or rerun Cy-ExM registration artifacts.
- Preserve unrelated pre-existing changes in `config.py`, provenance code, CSV inputs, and tests; stage only task-scoped hunks.

---

### Task 1: Strict multinuclear configuration and output identity

**Files:**
- Modify: `scripts/cyclic_stability/models.py`
- Modify: `scripts/cyclic_stability/config.py`
- Modify: `config/analysis.yaml`
- Modify: `tests/test_config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `NuclearConfig.accepted_component_count: tuple[int, int]`
- Produces: `NuclearConfig.nominal_volume_um3: tuple[float, float]`
- Produces: `NuclearConfig.minimum_sphericity: float`
- Produces: `NuclearConfig.maximum_face_contact_fraction: float`
- Produces: `MetricRow.nucleus_id: str = "aggregate"`
- Produces: `EnsembleRow.nucleus_id: str = "aggregate"`

- [ ] **Step 1: Add failing parser tests for the exact accepted configuration**

```python
def test_analysis_config_parses_multinuclear_bounds(valid_analysis_files):
    config = load_analysis_config(*valid_analysis_files)
    assert config.nuclear.accepted_component_count == (1, 6)
    assert config.nuclear.nominal_volume_um3 == (100.0, 5000.0)
    assert config.nuclear.minimum_sphericity == 0.15
    assert config.nuclear.maximum_face_contact_fraction == 0.10
```

- [ ] **Step 2: Add parameterized failing tests for invalid bounds**

```python
@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("accepted_component_count", [0, 6]),
        ("accepted_component_count", [6, 1]),
        ("nominal_volume_um3", [5000.0, 100.0]),
        ("minimum_sphericity", 1.1),
        ("maximum_face_contact_fraction", 1.1),
    ],
)
def test_analysis_config_rejects_invalid_multinuclear_bounds(
    tmp_path, config_files, key, value
):
    analysis, correspondence, decisions = config_files
    data = yaml.safe_load(analysis.read_text(encoding="utf-8"))
    data["nuclear"][key] = value
    invalid = tmp_path / "invalid-analysis.yaml"
    invalid.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError):
        load_analysis_config(invalid, correspondence, decisions)
```

- [ ] **Step 3: Run the focused tests and verify RED**

Run: `uv run pytest tests/test_config.py -k "multinuclear or nuclear" -v`

Expected: failure because the four configuration fields are unknown or absent.

- [ ] **Step 4: Implement strict parsing and immutable model fields**

Extend the exact nuclear mapping keys and construct:

```python
NuclearConfig(
    consensus_smoothing_um=_number(nuclear_data["consensus_smoothing_um"], "nuclear.consensus_smoothing_um"),
    normal_half_window_um=_number(nuclear_data["normal_half_window_um"], "nuclear.normal_half_window_um"),
    normal_step_um=_number(nuclear_data["normal_step_um"], "nuclear.normal_step_um"),
    ridge_min_sbr=_number(nuclear_data["ridge_min_sbr"], "nuclear.ridge_min_sbr"),
    accepted_component_count=_integer_pair(nuclear_data["accepted_component_count"], "nuclear.accepted_component_count"),
    nominal_volume_um3=_ordered_positive_pair(nuclear_data["nominal_volume_um3"], "nuclear.nominal_volume_um3"),
    minimum_sphericity=_closed_fraction(nuclear_data["minimum_sphericity"], "nuclear.minimum_sphericity", lower_open=True),
    maximum_face_contact_fraction=_closed_fraction(nuclear_data["maximum_face_contact_fraction"], "nuclear.maximum_face_contact_fraction"),
)
```

Append `nucleus_id: str = "aggregate"` to both output row dataclasses so existing positional constructors remain valid while every serialized row gains an explicit analysis unit.

- [ ] **Step 5: Update the authoritative YAML**

```yaml
nuclear:
  consensus_smoothing_um: 0.558
  normal_half_window_um: 1.5
  normal_step_um: 0.05
  ridge_min_sbr: 3.0
  accepted_component_count: [1, 6]
  nominal_volume_um3: [100.0, 5000.0]
  minimum_sphericity: 0.15
  maximum_face_contact_fraction: 0.10
```

- [ ] **Step 6: Verify GREEN without overwriting unrelated changes**

Run: `uv run pytest tests/test_config.py -v`

Run: `git diff -- scripts/cyclic_stability/config.py tests/test_config.py config/analysis.yaml scripts/cyclic_stability/models.py`

Expected: tests pass and the diff retains all pre-existing user hunks.

- [ ] **Step 7: Commit only scoped hunks**

Stage `models.py` and `analysis.yaml` normally. Use an index-safe patch for overlapping `config.py` and `test_config.py`; confirm `git diff --cached --name-only` contains only Task 1 paths before committing `feat: configure multinuclear consensus bounds`.

---

### Task 2: Target-blind component adjudication and consensus model

**Files:**
- Modify: `scripts/cyclic_stability/nuclear.py`
- Modify: `tests/test_nuclear.py`
- Modify: `tests/synthetic.py`

**Interfaces:**
- Produces: `NuclearComponentQc(nucleus_id, accepted, rejection_reasons, expanded_volume_um3, nominal_volume_um3, sphericity, centroid_xyz_um, face_contact_fractions)`
- Produces: `NuclearSurface.nucleus_id: str`
- Produces: `NuclearConsensus(nuclei, mask_zyx, labels_zyx, component_qc, spacing_xyz_um, nominal_expansion_factor)`
- Changes: `consensus_nuclear_surface(dapi_rounds, spacing_xyz_um, config, nominal_expansion_factor) -> NuclearConsensus`

- [ ] **Step 1: Add synthetic component builders**

Add helpers that place nonoverlapping ellipsoids at requested centers and radii in a shared ZYX volume. Tests must use physical spacings and sizes that map unambiguously inside or outside configured nominal bounds.

- [ ] **Step 2: Write failing tests for one through six nuclei and deterministic IDs**

```python
@pytest.mark.parametrize("count", range(1, 7))
def test_consensus_accepts_one_through_six_nuclei(count):
    dapi = separated_nuclei(count, SPACING)
    result = consensus_nuclear_surface(
        {2: dapi}, SPACING, NUCLEAR_CONFIG, nominal_expansion_factor=4.5
    )
    assert [item.nucleus_id for item in result.nuclei] == [
        f"N{index:03d}" for index in range(1, count + 1)
    ]
    assert set(np.unique(result.labels_zyx)) == set(range(count + 1))
    assert np.array_equal(result.mask_zyx, result.labels_zyx > 0)
```

- [ ] **Step 3: Write failing tests for zero/seven and exact rejection reasons**

Cover `volume-below-minimum`, `volume-above-maximum`, `sphericity-below-minimum`, and `face-contact-above-maximum`. Assert seven otherwise valid objects raises an error reporting accepted count 7, while rejected debris remains present in `component_qc` when at least one valid nucleus remains.

- [ ] **Step 4: Verify RED**

Run: `uv run pytest tests/test_nuclear.py -k "component or nuclei or multinuclear" -v`

Expected: failures because the builder still selects only the largest component and lacks component QC.

- [ ] **Step 5: Implement adjudication without target data**

Replace both `_largest_component` calls in consensus construction with one fill/closing pass followed by 26-connected labeling. For each candidate, calculate physical mesh geometry and per-face fractions, convert volume using:

```python
nominal_volume_um3 = expanded_volume_um3 / nominal_expansion_factor**3
```

Collect all rejection reasons rather than stopping at the first. Sort accepted candidates by `(centroid_x, centroid_y, centroid_z)`, assign IDs, and build a read-only `uint16` accepted label image.

- [ ] **Step 6: Build each surface independently**

For every accepted component call `_surface_from_mask` and `_surface_normals_and_curvature` on its own mask. Freeze all arrays. Do not concatenate meshes or calculate a centroid spanning multiple nuclei.

- [ ] **Step 7: Verify focused GREEN and single-nucleus compatibility**

Run: `uv run pytest tests/test_nuclear.py -k "consensus or component or nuclei or face" -v`

Expected: all new tests pass; prior single-nucleus surface coordinates, outward normals, and curvature tests remain within their existing tolerances after accessing `result.nuclei[0]`.

- [ ] **Step 8: Commit**

Commit `nuclear.py`, `test_nuclear.py`, and `synthetic.py` as `feat: construct target-blind multinuclear consensus`.

---

### Task 3: Union-mask consumers and explicit analysis units

**Files:**
- Modify: `scripts/cyclic_stability/golgi.py`
- Modify: `scripts/cyclic_stability/lysosome.py`
- Modify: `scripts/cyclic_stability/signal.py`
- Modify: `scripts/cyclic_stability/task.py`
- Modify: `tests/test_golgi.py`
- Modify: `tests/test_lysosome.py`
- Modify: `tests/test_signal.py`
- Modify: `tests/test_task.py`

**Interfaces:**
- Consumes: `NuclearConsensus.mask_zyx`, `.nuclei`, `.component_qc`
- Produces: field-level rows with `nucleus_id="aggregate"`
- Produces: task table `nuclear-component-qc`

- [ ] **Step 1: Write failing union-mask tests**

Create a two-nucleus consensus and assert Golgi perinuclear ROI excludes both nuclei, LAMP1 extranuclear ROI excludes both, and Lamin/LAMP1 nearest-surface distances use the minimum distance to either accepted surface.

- [ ] **Step 2: Write failing serialization tests**

Assert all Golgi, LAMP1, and raw-signal dictionaries include `nucleus_id == "aggregate"`. Assert `default_marker_analyzer` publishes a `nuclear-component-qc` table with accepted and rejected component records but no target intensity or target-derived mask fields.

- [ ] **Step 3: Verify RED**

Run: `uv run pytest tests/test_golgi.py tests/test_lysosome.py tests/test_signal.py tests/test_task.py -k "nucleus or union or component_qc" -v`

Expected: failures at the changed consensus API and missing table.

- [ ] **Step 4: Adapt mask-only consumers**

Pass the accepted union mask wherever an organelle analysis needs nuclear inclusion/exclusion. Where a surface-distance calculation previously queried one mesh, construct one tree from the vertices of all accepted nuclear surfaces, without inventing a combined mesh or centroid.

- [ ] **Step 5: Publish target-blind component QC**

Extend `MarkerResult` with a frozen tuple of component-QC mappings or a typed row object and add:

```python
"nuclear-component-qc": tuple(asdict(row) for row in result.component_qc_rows)
```

to `TaskOutput.tables`. Ensure the component inventory is constructed from DAPI before any target-dependent metric and contains exact geometry/config status only.

- [ ] **Step 6: Verify GREEN**

Run: `uv run pytest tests/test_golgi.py tests/test_lysosome.py tests/test_signal.py tests/test_task.py -v`

- [ ] **Step 7: Commit carefully around pre-existing Lysosome changes**

Use index-safe patch staging for `tests/test_lysosome.py`; do not stage pre-existing provenance/config changes. Commit as `feat: propagate multinuclear union masks`.

---

### Task 4: Per-nucleus Lamin profiles and robust ensemble geometry

**Files:**
- Modify: `scripts/cyclic_stability/nuclear.py`
- Modify: `scripts/cyclic_stability/models.py`
- Modify: `tests/test_nuclear.py`

**Interfaces:**
- Consumes: `NuclearConsensus.nuclei`
- Produces: `LaminProfileRow.nucleus_id`
- Produces: per-nucleus `MetricRow` and `EnsembleRow`
- Produces: field-level robust median rows with `nucleus_id="aggregate"`
- Produces: geometry status and `valid_curvature_vertex_fraction`

- [ ] **Step 1: Write a failing ROI-isolation test**

Construct two nuclear surfaces with different known Lamin offsets. Assert every profile and metric for `N001` matches only its local ridge and every `N002` row matches only the second ridge.

- [ ] **Step 2: Write a failing fragmented-mask regression**

Build a nonempty mask with a valid main envelope and isolated components that produce zero interpolated normal lengths. Assert `_geometry` preserves exact volume and area/overlap inputs, returns `geometry_status="partial-curvature"` or `"invalid-curvature"`, and never raises solely because some curvature vertices are degenerate.

- [ ] **Step 3: Write a failing coverage-threshold test**

Assert curvature percentiles are finite when at least 99% of surface vertices have finite nonzero normals and are NaN when coverage is below 99%. Assert `valid_curvature_vertex_fraction` is always recorded for nonempty masks.

- [ ] **Step 4: Verify RED**

Run: `uv run pytest tests/test_nuclear.py -k "per_nucleus or fragmented or curvature_coverage" -v`

- [ ] **Step 5: Separate strict consensus normals from tolerant ensemble curvature**

Keep accepted DAPI nuclear surfaces strict. Add a geometry helper for threshold-ensemble masks that samples the signed-distance gradient, filters vertices by finite positive length, calculates curvature only for valid vertices, and applies the fixed 0.99 coverage rule. It must not modify the input mask.

- [ ] **Step 6: Iterate Lamin analysis by nucleus**

For each accepted nucleus, compute profiles and threshold masks inside that nucleus mask, attach `nucleus_id`, and create per-nucleus diagnostics. Aggregate field rows by taking the median of nucleus-level values; do not pool mesh vertices.

- [ ] **Step 7: Update diagnostics**

Publish per-nucleus profile/displacement plots and one target-blind component inventory using distinct colorblind-friendly colors, gray rejected candidates, `N001` labels, and the exact QC table. Retain:

```python
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
```

- [ ] **Step 8: Verify GREEN**

Run: `uv run pytest tests/test_nuclear.py -v`

Expected: the production-shaped Cell2 failure condition is represented by the fragmented synthetic regression and no longer aborts a cell.

- [ ] **Step 9: Commit**

Commit as `feat: measure Lamin stability per nucleus`.

---

### Task 5: End-to-end validation and production DAPI inventory

**Files:**
- Modify: `scripts/run_analysis.py` only if a validate-only inventory surface is absent
- Modify: `tests/test_run_analysis.py` only if the CLI changes
- Create at runtime: a new immutable directory below `results/`; never commit results

**Interfaces:**
- Consumes: all prior task interfaces
- Produces: pretty machine-readable component QC in normal task tables

- [ ] **Step 1: Run the complete synthetic suite**

Run: `uv run pytest -v`

Expected: all tests pass without warnings or leaked temporary directories.

- [ ] **Step 2: Run formatting and static gates**

Run: `uvx ruff check scripts tests`

Run: `uvx ruff format --check scripts tests`

Run: `git diff --check`

Expected: clean, apart from any explicitly identified pre-existing line-ending warning in an unrelated user-owned CSV.

- [ ] **Step 3: Run a target-blind production component inventory**

Use the existing analysis entry point in validate/inventory mode if available. Otherwise add a `--nuclear-inventory-only` mode that loads registered DAPI only, writes pretty JSON with four-space indentation and sorted keys, and exits before opening target channels.

The inventory must report, for every cohort/cell, candidate count, accepted count, accepted IDs, rejection reasons, nominal volumes, sphericities, and face-contact fractions.

- [ ] **Step 4: Inspect the one-to-six production contract**

Assert every field has one through six accepted nuclei. Do not launch target-specific work for any field outside the contract; report its blinded field identifier and DAPI-only rejection evidence for review.

- [ ] **Step 5: Run one Lamin cell as a production smoke test**

Run the prior failing Lamin Cell2 task in a new immutable run root. Confirm all nine threshold members per accepted round complete, fragmented masks retain volume/overlap rows, curvature statuses are explicit, diagnostics exist for every accepted nucleus, and task completion metadata validates.

- [ ] **Step 6: Final repository audit**

Run: `git status --short`

Run: `git log --oneline -6`

Confirm only known pre-existing user changes remain unstaged and every implementation commit is on `kdean/cyclic-stability-analysis`.

- [ ] **Step 7: Publish only when a remote exists**

Run: `git remote -v`. If no remote is configured, stop and report the exact commit IDs; do not invent a remote URL. If a verified remote exists, push the explicit branch and verify the remote ref matches local HEAD.

## Plan self-review

- Every design requirement maps to a task: strict configuration (Task 1), component model and QC (Task 2), union consumers (Task 3), per-nucleus Lamin and curvature tolerance (Task 4), diagnostics/provenance/production acceptance (Tasks 4--5).
- Interfaces use one consistent `nucleus_id` field and `aggregate` sentinel.
- The implementation never uses target channels to accept DAPI components.
- The plan contains no watershed, manual segmentation, or registration changes.
- Overlapping dirty files are explicitly staged by hunk to preserve user work.
