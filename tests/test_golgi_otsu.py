"""Regression tests for target-specific Otsu and physical component filtering."""

from dataclasses import replace

import numpy as np
import pytest
from cyclic_stability import nuclear
from cyclic_stability.golgi_segmentation import otsu_component_member


def example():
    rng = np.random.default_rng(23)
    image = rng.normal(100, 2, (12, 32, 32))
    image[3:6, 4:7, 4:7] += 200
    image[7, 20, 20] += 200
    image[8, 21, 21] += 200
    image[9, 22, 22] += 200
    image[4, 25, 5] += 200
    return image, np.ones(image.shape, dtype=bool)


def test_otsu_retains_diagonal_components_at_physical_cutoff():
    image, roi = example()
    result = otsu_component_member(image, roi, (1.0, 1.0, 1.0), 3.0)
    assert result.mask_zyx[3:6, 4:7, 4:7].all()
    assert result.mask_zyx[7, 20, 20] and result.mask_zyx[9, 22, 22]
    assert not result.mask_zyx[4, 25, 5]
    assert result.rejected_mask_zyx[4, 25, 5]
    assert not np.any(result.mask_zyx & result.rejected_mask_zyx)
    assert result.threshold_sbr is None
    assert result.threshold_method == "plane-median-otsu"
    assert not result.mask_zyx.flags.writeable


def test_segmentation_is_invariant_to_additive_plane_offsets_and_excludes_roi():
    image, roi = example()
    roi[3:6, 4:7, 4:7] = False
    a = otsu_component_member(image, roi, (1.0, 1.0, 1.0), 3.0)
    offsets = np.arange(image.shape[0]) * 37.0
    b = otsu_component_member(image + offsets[:, None, None], roi, (1.0, 1.0, 1.0), 3.0)
    np.testing.assert_array_equal(a.mask_zyx, b.mask_zyx)
    np.testing.assert_allclose(b.plane_background - a.plane_background, offsets)
    assert not np.any(a.mask_zyx & ~roi)
    assert not np.any(a.rejected_mask_zyx & ~roi)


def test_constant_data_yields_explicit_empty_mask():
    image = np.full((5, 8, 8), 100.0)
    result = otsu_component_member(
        image, np.ones_like(image, dtype=bool), (1.0, 1.0, 1.0), 3.0
    )
    assert result.mask_status == "empty"
    assert not result.mask_zyx.any()
    assert result.threshold_value is None


@pytest.mark.parametrize("cutoff", [0, -1, float("nan")])
def test_invalid_size_cutoff_is_rejected(cutoff):
    image, roi = example()
    with pytest.raises(ValueError):
        otsu_component_member(image, roi, (1.0, 1.0, 1.0), cutoff)


def test_plane_background_normalization_is_opt_in_and_offset_invariant():
    image, _ = example()
    shifted = image + np.arange(image.shape[0])[:, None, None] * 20
    a = nuclear._normalized_dapi(image, plane_background_percentile=10.0)
    b = nuclear._normalized_dapi(shifted, plane_background_percentile=10.0)
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_registered_golgi_dispatch_uses_configured_otsu(analysis_config):
    from cyclic_stability.golgi import _support_threshold_ensemble

    config = replace(
        analysis_config,
        golgi=replace(
            analysis_config.golgi,
            segmentation_method="plane-median-otsu",
            minimum_component_volume_um3=0.1,
        ),
    )
    image, roi = example()
    result = _support_threshold_ensemble(image, roi, roi, config)
    assert len(result) == 1
    assert result[0].threshold_method == "plane-median-otsu"


def test_noise_only_volume_fails_signal_admission():
    image = np.random.default_rng(23).normal(100, 2, (32, 64, 64))
    with pytest.raises(ValueError, match="indistinguishable from background"):
        otsu_component_member(
            image, np.ones(image.shape, bool), (0.1083, 0.1083, 0.3), 0.1
        )


def test_physical_component_cutoff_rounds_up():
    image, roi = example()
    result = otsu_component_member(image, roi, (0.5, 0.5, 1), 0.76)
    assert not result.mask_zyx[7, 20, 20]  # Three voxels = .75, below .76.
    assert result.mask_zyx[3:6, 4:7, 4:7].all()


def test_native_signal_uses_recorded_plane_background(analysis_config):
    from cyclic_stability.signal import measure_signal

    image, roi = example()
    image += np.arange(image.shape[0])[:, None, None] * 100
    member = otsu_component_member(image, roi, (1.0, 1.0, 1.0), 3.0)
    spec = analysis_config.cell("golgi", "Cell1").rounds[1]
    (row,) = measure_signal(spec, image, roi, (member,))
    expected = (image - member.plane_background[:, None, None])[member.mask_zyx]
    assert row.integrated_per_ms == pytest.approx(
        expected.sum() / spec.target_exposure_ms
    )
    assert row.threshold_sbr is None
    assert row.threshold_method == "plane-median-otsu"
    assert row.threshold_value == member.threshold_value


def test_otsu_configuration_is_opt_in_and_validated(config_files):
    from cyclic_stability.config import load_analysis_config

    analysis, *_ = config_files
    assert load_analysis_config(*config_files).golgi.segmentation_method == "sbr"
    text = analysis.read_text().replace(
        "golgi:\n",
        "golgi:\n  segmentation_method: plane-median-otsu\n  minimum_component_volume_um3: 0.1\n",
    )
    analysis.write_text(text)
    assert (
        load_analysis_config(*config_files).golgi.segmentation_method
        == "plane-median-otsu"
    )
    analysis.write_text(text.replace("plane-median-otsu", "not-a-method"))
    with pytest.raises(ValueError):
        load_analysis_config(*config_files)


def test_registered_roi_excludes_rejected_dapi_foreground(analysis_config, monkeypatch):
    from cyclic_stability import golgi
    from cyclic_stability.nuclear import consensus_nuclear_surface
    from synthetic import synthetic_dapi_rounds, synthetic_golgi
    from test_golgi import _small_nucleus_config

    config = _small_nucleus_config(analysis_config)
    config = replace(
        config, golgi=replace(config.golgi, segmentation_method="plane-median-otsu")
    )
    dapi = synthetic_dapi_rounds()
    surface = consensus_nuclear_surface(
        dapi,
        config.voxel_spacing_xyz_um,
        config.nuclear,
        config.nominal_expansion_factor,
    )
    target = synthetic_golgi()
    support = {number: np.ones(target.shape, bool) for number in dapi}
    original_roi = golgi._perinuclear_roi(surface.mask_zyx, support, config)
    foreground = surface.mask_zyx.copy()
    # Simulate rejected DAPI foreground inside the otherwise eligible exterior ROI.
    rejected = np.zeros(target.shape, bool)
    points = np.argwhere(original_roi)
    selected = points[len(points) // 2 : len(points) // 2 + 100]
    rejected[tuple(selected.T)] = True
    foreground |= rejected
    monkeypatch.setattr(
        golgi,
        "consensus_nuclear_surface",
        lambda *args: replace(surface, foreground_mask_zyx=foreground),
    )
    result = golgi._analyze_golgi_arrays(
        {number: target for number in dapi}, dapi, config, validity_masks=support
    )
    assert np.any(rejected & original_roi)
    assert not np.any(result.roi & foreground)


def test_nonconstant_quantized_noise_with_zero_mad_fails_admission():
    image = (
        np.random.default_rng(23)
        .choice([100, 101], size=(32, 64, 64), p=[0.7, 0.3])
        .astype(np.uint16)
    )
    with pytest.raises(ValueError, match="indistinguishable from background"):
        otsu_component_member(
            image, np.ones(image.shape, bool), (0.1083, 0.1083, 0.3), 0.1
        )


def test_native_otsu_domain_is_dapi_only_and_perinuclear(analysis_config, monkeypatch):
    from cyclic_stability.signal import raw_structure_roi
    from scipy.ndimage import distance_transform_edt

    nucleus = np.zeros((9, 25, 25), bool)
    nucleus[3:6, 10:15, 10:15] = True
    monkeypatch.setattr(
        nuclear, "_segmented_dapi_foreground_mask", lambda *args: nucleus
    )
    config = replace(
        analysis_config,
        voxel_spacing_xyz_um=(0.5, 0.5, 0.5),
        golgi=replace(
            analysis_config.golgi,
            segmentation_method="plane-median-otsu",
            perinuclear_outer_distance_um=1.0,
        ),
    )
    roi = raw_structure_roi("golgi", np.zeros(nucleus.shape), config)
    expected = (~nucleus) & (
        distance_transform_edt(~nucleus, sampling=(0.5, 0.5, 0.5)) <= 1.0
    )
    np.testing.assert_array_equal(roi, expected)
    assert not roi[:, 0, :].any()
    legacy = raw_structure_roi("golgi", np.zeros(nucleus.shape), analysis_config)
    np.testing.assert_array_equal(legacy, ~nucleus)


def test_smoothing_cannot_leak_excluded_signal_and_preserves_plane_offsets():
    image, roi = example()
    roi[:, :3, :] = False
    kwargs = dict(smoothing_um=0.5)
    a = otsu_component_member(image, roi, (1.0, 1.0, 1.0), 3.0, **kwargs)
    changed = image.copy()
    changed[~roi] = 100_000
    offsets = np.arange(image.shape[0]) * 37
    b = otsu_component_member(
        changed + offsets[:, None, None], roi, (1.0, 1.0, 1.0), 3.0, **kwargs
    )
    np.testing.assert_array_equal(a.mask_zyx, b.mask_zyx)
    np.testing.assert_allclose(b.plane_background - a.plane_background, offsets)
    assert a.sigma_um == 0.5
    assert a.threshold_noise_scale > 0


def test_smoothing_noise_control_is_not_measured():
    image = np.random.default_rng(23).normal(100, 2, (32, 64, 64))
    with pytest.raises(ValueError, match="indistinguishable from background"):
        otsu_component_member(
            image,
            np.ones(image.shape, bool),
            (0.1083, 0.1083, 0.3),
            0.1,
            smoothing_um=0.1083,
        )


def test_roi_crop_restoration_preserves_original_geometry():
    image, roi = example()
    a = otsu_component_member(image, roi, (1.0, 1.0, 1.0), 3.0, smoothing_um=0.5)
    pad = ((2, 3), (4, 5), (6, 7))
    padded_image = np.pad(image, pad, constant_values=10000)
    padded_roi = np.pad(roi, pad, constant_values=False)
    b = otsu_component_member(
        padded_image, padded_roi, (1.0, 1.0, 1.0), 3.0, smoothing_um=0.5
    )
    np.testing.assert_array_equal(b.mask_zyx, np.pad(a.mask_zyx, pad))
    np.testing.assert_array_equal(b.rejected_mask_zyx, np.pad(a.rejected_mask_zyx, pad))
    np.testing.assert_allclose(b.plane_background[2:-3], a.plane_background)


def test_native_saved_roi_is_distinct_from_nuclear_exclusion(tmp_path):
    import json

    from cyclic_stability.golgi_segmentation import write_native_otsu_diagnostics

    image, roi = example()
    nucleus = np.zeros(image.shape, bool)
    nucleus[0, 0, 0] = True
    roi[0] = False  # Most of this excluded plane is outside the ROI, not nuclear.
    member = otsu_component_member(image, roi, (1.0, 1.0, 1.0), 3.0, smoothing_um=0.5)
    masks, metadata = write_native_otsu_diagnostics(
        tmp_path, 1, member, roi, nuclear_exclusion=nucleus, outer_distance_um=18.0
    )
    with np.load(masks) as saved:
        np.testing.assert_array_equal(saved["nuclear_exclusion_mask_zyx"], nucleus)
        np.testing.assert_array_equal(saved["structure_roi_zyx"], roi)
    info = json.loads(metadata.read_text())
    assert info["smoothing_sigma_um"] == 0.5
    assert info["robust_noise_scale"] == member.threshold_noise_scale
    assert info["raw_background_noise_scale"] == member.background.scale
