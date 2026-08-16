"""Synthetic tests for Golgi continuous similarity and morphology robustness."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
import tifffile
from cyclic_stability.cyexm_artifacts import RoundArtifacts
from cyclic_stability.golgi import (
    ENSEMBLE_ENDPOINTS,
    _atomic_publish_directory,
    _component_volume_gini,
    _mask_geometry,
    _perinuclear_roi,
    analyze_golgi_arrays,
    analyze_golgi_cell,
    golgi_nucleus_distances,
    multiscale_gradient_similarity,
)
from cyclic_stability.models import CellSpec
from cyclic_stability.nuclear import NuclearSurface, consensus_nuclear_surface
from scipy.ndimage import distance_transform_edt
from synthetic import (
    NUCLEAR_CONFIG,
    synthetic_dapi_rounds,
    synthetic_golgi,
    synthetic_perinuclear_roi,
)


def _full_support(rounds: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """Declare intentional all-valid fixed-grid support for synthetic arrays only."""
    return {
        number: np.ones(image.shape, dtype=bool) for number, image in rounds.items()
    }


def test_gradient_similarity_is_robust_to_global_intensity_scaling(
    analysis_config,
) -> None:
    """Replacing background correction with raw intensity comparison breaks scaling."""
    reference = synthetic_golgi()
    rows = multiscale_gradient_similarity(
        reference,
        reference * 2.5 + 100.0,
        synthetic_perinuclear_roi(),
        analysis_config.voxel_spacing_xyz_um,
        analysis_config.ensembles.gaussian_sigma_um,
    )
    assert min(row.correlation for row in rows) > 0.99


def test_gradient_similarity_retains_strongly_downscaled_signal_with_offset(
    analysis_config,
) -> None:
    """An absolute near-zero check mistakes a real scaled gradient for zero variance."""
    reference = synthetic_golgi()
    rows = multiscale_gradient_similarity(
        reference,
        reference * 1e-6 + 100.0,
        synthetic_perinuclear_roi(),
        analysis_config.voxel_spacing_xyz_um,
        analysis_config.ensembles.gaussian_sigma_um,
    )
    assert min(row.correlation for row in rows) > 0.99


def test_support_aware_preprocessing_ignores_invalid_region_values(
    analysis_config,
) -> None:
    """A huge padded value must not change valid-region background, masks, or gradients."""
    reference = synthetic_golgi()
    moving = reference.copy()
    valid = np.ones(reference.shape, dtype=bool)
    valid[:, :, :24] = False
    roi = synthetic_perinuclear_roi() & valid
    baseline = multiscale_gradient_similarity(
        reference,
        moving,
        roi,
        analysis_config.voxel_spacing_xyz_um,
        (0.558,),
        validity_mask=valid,
    )
    moving[:, :, :24] = 1e12
    changed = multiscale_gradient_similarity(
        reference,
        moving,
        roi,
        analysis_config.voxel_spacing_xyz_um,
        (0.558,),
        validity_mask=valid,
    )
    assert changed == baseline
    rounds = {2: reference, 3: reference.copy()}
    validity_by_round = {2: valid, 3: valid}
    baseline_result = analyze_golgi_arrays(
        rounds,
        synthetic_dapi_rounds(),
        analysis_config,
        validity_masks=validity_by_round,
    )
    rounds[3][:, :, :24] = 1e12
    changed_result = analyze_golgi_arrays(
        rounds,
        synthetic_dapi_rounds(),
        analysis_config,
        validity_masks=validity_by_round,
    )
    assert [row.endpoint for row in changed_result.ensemble_rows] == [
        row.endpoint for row in baseline_result.ensemble_rows
    ]
    assert np.allclose(
        [row.value for row in changed_result.ensemble_rows],
        [row.value for row in baseline_result.ensemble_rows],
        equal_nan=True,
    )
    assert np.allclose(
        [row.value for row in changed_result.summary_rows],
        [row.value for row in baseline_result.summary_rows],
        equal_nan=True,
    )


def test_array_kernel_requires_explicit_validity_contract(analysis_config) -> None:
    """Three positional arrays do not prove that every fixed-grid voxel is valid."""
    rounds = {2: synthetic_golgi(), 3: synthetic_golgi()}
    with pytest.raises(TypeError, match="validity_masks"):
        analyze_golgi_arrays(rounds, synthetic_dapi_rounds(), analysis_config)


def test_fragmentation_change_is_reported_for_every_frozen_member(
    analysis_config,
) -> None:
    """Selecting one favorable scale or discarding small objects hides fragmentation."""
    result = analyze_golgi_arrays(
        {2: synthetic_golgi(components=1), 3: synthetic_golgi(components=4)},
        synthetic_dapi_rounds(),
        analysis_config,
        validity_masks=_full_support({2: synthetic_golgi(), 3: synthetic_golgi()}),
    )
    counts = [
        row.value
        for row in result.ensemble_rows
        if row.acquisition_round == 3 and row.endpoint == "component_count"
    ]
    assert len(counts) == len(analysis_config.ensembles.gaussian_sigma_um) * len(
        analysis_config.ensembles.threshold_sbr
    )
    assert np.median(counts) > 1


def test_perinuclear_roi_is_fixed_outside_consensus_and_limited_to_18_um(
    analysis_config,
) -> None:
    """A target-derived or round-specific ROI would make morphology non-comparable."""
    dapi = synthetic_dapi_rounds()
    surface = consensus_nuclear_surface(
        dapi, analysis_config.voxel_spacing_xyz_um, NUCLEAR_CONFIG
    )
    validity = {
        round_number: np.ones(surface.mask_zyx.shape, dtype=bool)
        for round_number in dapi
    }
    roi = _perinuclear_roi(surface.mask_zyx, validity, analysis_config)
    distances = distance_transform_edt(
        ~surface.mask_zyx,
        sampling=analysis_config.voxel_spacing_xyz_um[::-1],
    )
    assert not np.any(roi & surface.mask_zyx)
    assert float(distances[roi].max()) <= 18.0
    assert np.array_equal(
        roi, _perinuclear_roi(surface.mask_zyx, validity, analysis_config)
    )


def test_perinuclear_roi_intersects_explicit_transform_validity_not_target_values(
    analysis_config,
) -> None:
    """Target finiteness is not evidence that a moving transform sampled valid data."""
    dapi = synthetic_dapi_rounds()
    surface = consensus_nuclear_surface(
        dapi, analysis_config.voxel_spacing_xyz_um, NUCLEAR_CONFIG
    )
    validity = {
        round_number: np.ones(surface.mask_zyx.shape, dtype=bool)
        for round_number in dapi
    }
    validity[3][:, :, :20] = False
    roi = _perinuclear_roi(surface.mask_zyx, validity, analysis_config)
    assert not np.any(roi[:, :, :20])


def test_round_two_is_reference_regardless_of_mapping_order(analysis_config) -> None:
    """Using mapping insertion order silently changes all overlap comparisons."""
    reference = synthetic_golgi(components=1)
    moving = synthetic_golgi(components=4)
    result = analyze_golgi_arrays(
        {3: moving, 2: reference},
        {3: synthetic_dapi_rounds()[3], 2: synthetic_dapi_rounds()[2]},
        analysis_config,
        validity_masks=_full_support({2: reference, 3: moving}),
    )
    round_two = [
        row.value
        for row in result.ensemble_rows
        if row.acquisition_round == 2 and row.endpoint == "jaccard"
    ]
    assert round_two == [1.0] * len(round_two)


def test_gradient_similarity_rejects_zero_variance_comparison(analysis_config) -> None:
    """A constant gradient must not receive a perfect correlation score."""
    image = 1.0 + 0.05 * np.sin(np.indices((25, 25, 25))[0])
    roi = np.zeros_like(image, dtype=bool)
    roi[12, 12, 12] = True
    with pytest.raises(ValueError, match="zero variance"):
        multiscale_gradient_similarity(
            image,
            image,
            roi,
            analysis_config.voxel_spacing_xyz_um,
            (0.279,),
        )


def test_empty_masks_are_zero_volume_with_undefined_geometry() -> None:
    """Giving an empty mask compactness or overlap credit would favor signal loss."""
    geometry = _mask_geometry(np.zeros((9, 9, 9), dtype=bool), (0.1, 0.1, 0.3))
    assert geometry["volume_um3"] == 0.0
    assert geometry["component_count"] == 0.0
    assert math.isnan(geometry["compactness"])
    assert math.isnan(geometry["area_to_volume_um_inverse"])


def test_all_empty_members_keep_every_predeclared_summary_endpoint(
    analysis_config,
) -> None:
    """Dropping all-NaN endpoints creates a data-dependent outcome schema."""
    empty_config = replace(
        analysis_config,
        ensembles=replace(analysis_config.ensembles, threshold_sbr=(1e9,)),
    )
    result = analyze_golgi_arrays(
        {2: synthetic_golgi(), 3: synthetic_golgi()},
        synthetic_dapi_rounds(),
        empty_config,
        validity_masks=_full_support({2: synthetic_golgi(), 3: synthetic_golgi()}),
    )
    summaries = {
        row.endpoint: row.value
        for row in result.summary_rows
        if row.acquisition_round == 3
    }
    assert {f"golgi_{endpoint}_median" for endpoint in ENSEMBLE_ENDPOINTS} <= set(
        summaries
    )
    assert math.isnan(summaries["golgi_compactness_median"])
    assert summaries["golgi_volume_um3_median"] == 0.0


def test_component_gini_and_compactness_use_physical_definitions() -> None:
    """An uncorrected Gini or inverse compactness changes fragmentation conclusions."""
    assert _component_volume_gini(np.array((1.0, 2.0, 3.0))) == pytest.approx(1.0 / 3.0)
    cube = np.zeros((21, 21, 21), dtype=bool)
    cube[5:16, 5:16, 5:16] = True
    geometry = _mask_geometry(cube, (1.0, 1.0, 1.0))
    assert 0.0 < geometry["compactness"] < 1.0


def test_diagonal_positive_voxels_remain_one_26_connected_component() -> None:
    """Six-connectivity would falsely inflate fragmentation of diagonal Golgi support."""
    mask = np.zeros((9, 9, 9), dtype=bool)
    mask[3, 3, 3] = True
    mask[4, 4, 4] = True
    assert _mask_geometry(mask, (0.1, 0.1, 0.3))["component_count"] == 1.0


def test_ensemble_rows_preserve_all_coordinates_and_endpoints(analysis_config) -> None:
    """Collapsing sigma or threshold coordinates prevents sensitivity-curve review."""
    result = analyze_golgi_arrays(
        {2: synthetic_golgi(), 3: synthetic_golgi()},
        synthetic_dapi_rounds(),
        analysis_config,
        validity_masks=_full_support({2: synthetic_golgi(), 3: synthetic_golgi()}),
    )
    members = len(analysis_config.ensembles.gaussian_sigma_um) * len(
        analysis_config.ensembles.threshold_sbr
    )
    assert len(result.ensemble_rows) == 2 * members * (
        len(ENSEMBLE_ENDPOINTS)
        + len(analysis_config.ensembles.surface_tolerance_resolution_units)
    )
    assert {
        row.ensemble_name
        for row in result.ensemble_rows
        if row.endpoint == "volume_um3"
    } == {
        f"sigma={sigma:g};sbr={threshold:g}"
        for sigma in analysis_config.ensembles.gaussian_sigma_um
        for threshold in analysis_config.ensembles.threshold_sbr
    }


def test_array_kernel_rejects_missing_round_shape_and_validity_mismatch(
    analysis_config,
) -> None:
    """A mismatched or absent common-support mask invalidates fixed-grid comparison."""
    dapi = synthetic_dapi_rounds()
    with pytest.raises(ValueError, match="round 2"):
        analyze_golgi_arrays(
            {3: synthetic_golgi()},
            {3: dapi[3]},
            analysis_config,
            validity_masks=_full_support({3: synthetic_golgi()}),
        )
    with pytest.raises(ValueError, match="geometry"):
        analyze_golgi_arrays(
            {2: synthetic_golgi(), 3: np.ones((9, 9, 9))},
            dapi,
            analysis_config,
            validity_masks=_full_support({2: synthetic_golgi(), 3: np.ones((9, 9, 9))}),
        )
    with pytest.raises(ValueError, match="validity"):
        analyze_golgi_arrays(
            {2: synthetic_golgi(), 3: synthetic_golgi()},
            dapi,
            analysis_config,
            validity_masks={
                2: np.ones((9, 9, 9), dtype=bool),
                3: np.ones((9, 9, 9), dtype=bool),
            },
        )


def test_cell_entry_requires_explicit_common_support_and_publishes_overlays(
    analysis_config, tmp_path
) -> None:
    """Production API owns support reconstruction and retains its four-argument contract."""
    round_two = next(
        round_spec
        for round_spec in analysis_config.cells[0].rounds
        if round_spec.acquisition_round == 2
    )
    cell = CellSpec("golgi", "SyntheticCell", (round_two,))
    dapi_path = tmp_path / "dapi.tif"
    target_path = tmp_path / "target.tif"
    tifffile.imwrite(dapi_path, synthetic_dapi_rounds()[2])
    tifffile.imwrite(target_path, synthetic_golgi())
    artifact = RoundArtifacts(
        2,
        round_two.workflow_round,
        tmp_path,
        dapi_path,
        tmp_path / "tubulin.tif",
        target_path,
        None,
        None,
        None,
        None,
        None,
        None,
        (),
    )
    with pytest.raises(TypeError):
        analyze_golgi_cell(
            cell,
            (artifact,),
            analysis_config,
            tmp_path / "out",
            {2: np.ones(synthetic_golgi().shape, dtype=bool)},
        )


def test_atomic_golgi_diagnostic_publisher_restores_previous_tree_on_failure(
    tmp_path,
) -> None:
    """A failed overlay write must not leave a partial result tree as publication."""
    destination = tmp_path / "out"
    _atomic_publish_directory(
        destination,
        lambda staged: (staged / "complete.txt").write_text("old", encoding="utf-8"),
    )

    def failing_populate(staged) -> None:
        (staged / "partial.txt").write_text("new", encoding="utf-8")
        raise RuntimeError("synthetic diagnostic failure")

    with pytest.raises(RuntimeError, match="synthetic diagnostic failure"):
        _atomic_publish_directory(destination, failing_populate)
    assert (destination / "complete.txt").read_text(encoding="utf-8") == "old"
    assert not (destination / "partial.txt").exists()


def test_target_distances_use_full_physical_xyz_coordinates(analysis_config) -> None:
    """Index-space or XY-only distances mis-measure anisotropic Z displacement."""
    surface = consensus_nuclear_surface(
        synthetic_dapi_rounds(), analysis_config.voxel_spacing_xyz_um, NUCLEAR_CONFIG
    )
    mask = np.zeros(surface.mask_zyx.shape, dtype=bool)
    mask[40, 64, 64] = True
    distances = golgi_nucleus_distances(
        mask, surface, analysis_config.voxel_spacing_xyz_um
    )
    assert distances.shape == (1,)
    assert distances[0] > 3.0


def test_target_distance_characterizes_anisotropic_voxel_center_coordinates() -> None:
    """Treating ZYX indices as equal XYZ units would turn a 0.3 um Z step into one."""
    surface = NuclearSurface(
        np.zeros((3, 3, 3), dtype=bool),
        np.array(((0.0, 0.0, 0.0),)),
        np.empty((0, 3), dtype=int),
        np.array(((0.0, 0.0, 1.0),)),
        np.zeros(3),
        np.zeros(1),
        (0.1, 0.1, 0.3),
    )
    mask = np.zeros((3, 3, 3), dtype=bool)
    mask[1, 0, 0] = True
    assert golgi_nucleus_distances(mask, surface, (0.1, 0.1, 0.3)) == pytest.approx(
        (0.3,)
    )


@pytest.mark.parametrize("bad", (np.nan, np.inf, -1.0, 0.5, 2.0))
def test_gradient_similarity_rejects_nonbinary_roi_and_invalid_inputs(
    analysis_config, bad: float
) -> None:
    """Coercing malformed ROI values creates target-dependent false support."""
    roi = synthetic_perinuclear_roi().astype(float)
    roi[0, 0, 0] = bad
    with pytest.raises(ValueError, match="binary"):
        multiscale_gradient_similarity(
            synthetic_golgi(),
            synthetic_golgi(),
            roi,
            analysis_config.voxel_spacing_xyz_um,
            (0.279,),
        )
