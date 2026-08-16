"""Synthetic tests for Golgi continuous similarity and morphology robustness."""

from __future__ import annotations

import math

import numpy as np
import pytest
from cyclic_stability.golgi import (
    _component_volume_gini,
    _mask_geometry,
    _perinuclear_roi,
    analyze_golgi_arrays,
    golgi_nucleus_distances,
    multiscale_gradient_similarity,
)
from cyclic_stability.nuclear import consensus_nuclear_surface
from scipy.ndimage import distance_transform_edt
from synthetic import (
    NUCLEAR_CONFIG,
    synthetic_dapi_rounds,
    synthetic_golgi,
    synthetic_perinuclear_roi,
)


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


def test_fragmentation_change_is_reported_for_every_frozen_member(
    analysis_config,
) -> None:
    """Selecting one favorable scale or discarding small objects hides fragmentation."""
    result = analyze_golgi_arrays(
        {2: synthetic_golgi(components=1), 3: synthetic_golgi(components=4)},
        synthetic_dapi_rounds(),
        analysis_config,
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
    roi = _perinuclear_roi(surface.mask_zyx, dapi, analysis_config)
    distances = distance_transform_edt(
        ~surface.mask_zyx,
        sampling=analysis_config.voxel_spacing_xyz_um[::-1],
    )
    assert not np.any(roi & surface.mask_zyx)
    assert float(distances[roi].max()) <= 18.0
    assert np.array_equal(
        roi, _perinuclear_roi(surface.mask_zyx, dapi, analysis_config)
    )


def test_round_two_is_reference_regardless_of_mapping_order(analysis_config) -> None:
    """Using mapping insertion order silently changes all overlap comparisons."""
    reference = synthetic_golgi(components=1)
    moving = synthetic_golgi(components=4)
    result = analyze_golgi_arrays(
        {3: moving, 2: reference},
        {3: synthetic_dapi_rounds()[3], 2: synthetic_dapi_rounds()[2]},
        analysis_config,
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


def test_component_gini_and_compactness_use_physical_definitions() -> None:
    """An uncorrected Gini or inverse compactness changes fragmentation conclusions."""
    assert _component_volume_gini(np.array((1.0, 2.0, 3.0))) == pytest.approx(1.0 / 3.0)
    cube = np.zeros((21, 21, 21), dtype=bool)
    cube[5:16, 5:16, 5:16] = True
    geometry = _mask_geometry(cube, (1.0, 1.0, 1.0))
    assert 0.0 < geometry["compactness"] < 1.0


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
