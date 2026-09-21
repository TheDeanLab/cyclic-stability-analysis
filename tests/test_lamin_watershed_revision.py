"""Regression evidence for the physical watershed and guarded Lamin exclusions."""

from dataclasses import replace

import numpy as np
import pytest
import yaml
from cyclic_stability import models, nuclear
from cyclic_stability.config import load_analysis_config
from synthetic import NUCLEAR_CONFIG, SPACING, separated_nuclei


def test_anisotropic_watershed_preserves_union_and_axis_permutation():
    shape = (35, 61, 137)
    z, y, x = np.indices(shape)
    spacing = (0.25, 0.5, 1.0)
    mask = (x * spacing[0] - 10) ** 2 + (y * spacing[1] - 15) ** 2 + (
        z - 17
    ) ** 2 < 7**2
    mask |= (x * spacing[0] - 23) ** 2 + (y * spacing[1] - 15) ** 2 + (
        z - 17
    ) ** 2 < 7**2
    a, count = nuclear._watershed_nuclear_labels(mask, spacing, 2.0)
    b, rotated_count = nuclear._watershed_nuclear_labels(
        mask.transpose(2, 1, 0), spacing[::-1], 2.0
    )
    assert count == rotated_count == 2
    assert np.array_equal(a > 0, mask)
    # Compare partitions independent of numerical label permutation.
    b = b.transpose(2, 1, 0)
    assert all(len(np.unique(b[a == i])) == 1 for i in (1, 2))


def test_diagonal_foreground_and_fallback_are_preserved():
    mask = np.zeros((9, 9, 9), bool)
    mask[1:4, 1:4, 1:4] = True
    mask[4:7, 4:7, 4:7] = True
    labels, count = nuclear._watershed_nuclear_labels(mask, (1.0, 1.0, 1.0), 100.0)
    assert count == 1
    assert np.array_equal(labels > 0, mask)


def test_optional_policy_is_parsed_and_changes_identity(lamin_config_files):
    config_files = lamin_config_files
    before = load_analysis_config(*config_files)
    assert before.nuclear.watershed_seed_prominence_um is None
    assert before.lamin_nuclear_exclusions == ()
    path = config_files[0]
    data = yaml.safe_load(path.read_text())
    data["nuclear"]["watershed_seed_prominence_um"] = 2.0
    data["lamin_nuclear_exclusions"] = [
        {
            "cell_id": "Cell1",
            "point_xyz_um": [10.0, 20.0, 30.0],
            "expected_voxel_count": 123,
            "reason": "merged-truncated-nuclei",
        }
    ]
    path.write_text(yaml.safe_dump(data))
    after = load_analysis_config(*config_files)
    assert after.nuclear.watershed_seed_prominence_um == 2.0
    assert after.lamin_nuclear_exclusions[0].expected_voxel_count == 123
    assert after.config_digest != before.config_digest


@pytest.mark.parametrize(
    "changes",
    [
        {"cell_id": "unknown"},
        {"point_xyz_um": [-1, 2, 3]},
        {"expected_voxel_count": True},
        {"reason": ""},
        {"point_xyz_um": [1, 2, float("nan")]},
    ],
)
def test_invalid_exclusion_is_rejected(lamin_config_files, changes):
    config_files = lamin_config_files
    path = config_files[0]
    data = yaml.safe_load(path.read_text())
    rule = {
        "cell_id": "Cell1",
        "point_xyz_um": [10.0, 20.0, 30.0],
        "expected_voxel_count": 123,
        "reason": "merged-truncated-nuclei",
    }
    rule.update(changes)
    data["lamin_nuclear_exclusions"] = [rule]
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        load_analysis_config(*config_files)


def test_exclusion_retains_qc_and_does_not_mutate_adjudication():
    dapi = separated_nuclei(2, SPACING)
    original = nuclear.adjudicate_nuclear_components(
        {2: dapi}, SPACING, NUCLEAR_CONFIG, 4.5
    )
    candidate = next(c for c in original._candidates if c.accepted)
    geometry = candidate.accepted_geometry
    local = np.argwhere(geometry.mask_zyx)[len(np.argwhere(geometry.mask_zyx)) // 2]
    global_zyx = local + np.array([s.start for s in geometry.region_zyx])
    rule = models.LaminNuclearExclusion(
        "Cell1",
        tuple(global_zyx[::-1] * SPACING),
        int(geometry.mask_zyx.sum()),
        "merged-truncated-nuclei",
    )
    updated = nuclear._apply_lamin_nuclear_exclusions(original, (rule,))
    assert sum(c.accepted for c in original.component_qc) == 2
    assert sum(c.accepted for c in updated.component_qc) == 1
    excluded = [
        c
        for c in updated.component_qc
        if "explicit-qc:merged-truncated-nuclei" in c.rejection_reasons
    ]
    assert len(excluded) == 1
    assert not next(
        c
        for c in updated.component_diagnostics
        if c.nucleus_id == excluded[0].nucleus_id
    ).accepted
    realized = nuclear.realize_nuclear_surfaces(updated, NUCLEAR_CONFIG)
    assert len(realized.nuclei) == 1
    with pytest.raises(ValueError, match="voxel count"):
        nuclear._apply_lamin_nuclear_exclusions(
            original, (replace(rule, expected_voxel_count=1),)
        )
    with pytest.raises(ValueError, match="match"):
        nuclear._apply_lamin_nuclear_exclusions(
            original, (replace(rule, point_xyz_um=(0.0, 0.0, 0.0)),)
        )


@pytest.fixture
def lamin_config_files(config_files):
    path = config_files[1]
    path.write_text(path.read_text().replace("golgi", "lamin"))
    return config_files


def test_filtered_surface_faces_reference_only_retained_vertices(monkeypatch):
    original = nuclear._surface_normals_and_curvature

    def one_missing_curvature(*args, **kwargs):
        normals, centroid, curvature = original(*args, **kwargs)
        curvature = curvature.copy()
        curvature[0] = np.nan
        return normals, centroid, curvature

    monkeypatch.setattr(
        nuclear, "_surface_normals_and_curvature", one_missing_curvature
    )
    result = nuclear.consensus_nuclear_surface(
        {2: separated_nuclei(1, SPACING)}, SPACING, NUCLEAR_CONFIG, 4.5
    )
    surface = result.nuclei[0]
    assert surface.faces.max() < len(surface.vertices_xyz_um)
    assert (
        len(surface.normals_xyz)
        == len(surface.mean_curvature_um_inverse)
        == len(surface.vertices_xyz_um)
    )
    assert np.all(np.isfinite(surface.mean_curvature_um_inverse))


def test_target_blind_inventory_honors_the_same_exclusion(monkeypatch):
    from types import SimpleNamespace

    import run_analysis

    dapi = separated_nuclei(2, SPACING)
    original = nuclear.adjudicate_nuclear_components(
        {2: dapi}, SPACING, NUCLEAR_CONFIG, 4.5
    )
    geometry = original._candidates[0].accepted_geometry
    assert geometry is not None
    voxel = np.argwhere(geometry.mask_zyx)[0]
    point = (voxel + [s.start for s in geometry.region_zyx])[::-1] * SPACING
    rule = models.LaminNuclearExclusion(
        "Cell1", tuple(point), int(geometry.mask_zyx.sum()), "reviewed-merge"
    )
    round_spec = SimpleNamespace(cohort="lamin", cell_id="Cell1", acquisition_round=2)
    cell = SimpleNamespace(cohort="lamin", cell_id="Cell1", rounds=(round_spec,))
    config = SimpleNamespace(
        nuclear=NUCLEAR_CONFIG,
        voxel_spacing_xyz_um=SPACING,
        nominal_expansion_factor=4.5,
        reference_acquisition_round=2,
        qc_decisions=(),
        lamin_nuclear_exclusions=(rule,),
        config_digest="test",
    )
    monkeypatch.setattr(run_analysis, "_validated_registered_dapi", lambda *args: dapi)
    result = run_analysis._nuclear_inventory_payload(config, (cell,))
    assert result["cells"][0]["accepted_count"] == 1
    assert any(
        "explicit-qc:reviewed-merge" in row["rejection_reasons"]
        for row in result["cells"][0]["components"]
    )
