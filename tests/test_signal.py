"""Tests for unregistered repeated-label signal measurements."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from cyclic_stability.models import AnalysisConfig
from cyclic_stability.morphology import threshold_ensemble
from cyclic_stability.signal import measure_signal, raw_structure_roi
from scipy.ndimage import distance_transform_edt, gaussian_filter
from synthetic import ellipsoid_dapi, synthetic_roi, synthetic_target


def _ensemble(config: AnalysisConfig):
    return threshold_ensemble(
        synthetic_target(),
        synthetic_roi(),
        config.ensembles.gaussian_sigma_um,
        config.ensembles.threshold_sbr,
        config.voxel_spacing_xyz_um,
    )


def test_round_one_signal_is_kept_out_of_matched_setting_rows(
    analysis_config: AnalysisConfig,
) -> None:
    """Classifying round one as matched settings would confound its exposure change."""
    rows = measure_signal(
        analysis_config.cell("golgi", "Cell1").rounds[0],
        synthetic_target(),
        synthetic_roi(),
        _ensemble(analysis_config),
    )
    assert all(row.comparison_set == "round-1-exposure-sensitivity" for row in rows)
    assert all(row.exposure_normalized for row in rows)


def test_raw_signal_roi_uses_the_union_of_every_accepted_nucleus(
    analysis_config: AnalysisConfig,
) -> None:
    """Raw target ROIs must not select one favorable nucleus from the DAPI field."""
    spacing = (0.5, 0.5, 0.5)
    dapi = ellipsoid_dapi(
        ((6.0, 6.0, 6.0), (20.0, 6.0, 6.0)),
        ((4.0, 4.0, 4.0), (4.0, 4.0, 4.0)),
        spacing,
        (28, 28, 56),
    )
    config = replace(
        analysis_config,
        voxel_spacing_xyz_um=spacing,
        nominal_expansion_factor=1.0,
    )
    lamin_roi = raw_structure_roi("lamin", dapi, config)
    golgi_roi = raw_structure_roi("golgi", dapi, config)
    lamp_roi = raw_structure_roi("lamp1", dapi, config)
    assert np.any(lamin_roi[:, :, :28])
    assert np.any(lamin_roi[:, :, 28:])
    distance = distance_transform_edt(~lamin_roi, sampling=spacing[::-1])
    assert np.all(golgi_roi[lamin_roi])
    assert np.array_equal(golgi_roi, distance <= 18.0)
    assert np.array_equal(lamp_roi, ~lamin_roi)


def test_raw_signal_roi_does_not_require_surface_curvature(
    analysis_config: AnalysisConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Raw intensity support needs accepted masks, not strict surface geometry."""
    spacing = (0.5, 0.5, 0.5)
    dapi = ellipsoid_dapi(
        ((8.0, 8.0, 8.0),),
        ((5.0, 5.0, 5.0),),
        spacing,
        (36, 36, 36),
    )
    config = replace(
        analysis_config,
        voxel_spacing_xyz_um=spacing,
        nominal_expansion_factor=1.0,
    )

    def fail_surface(*_args, **_kwargs):
        raise ValueError("surface curvature is unavailable")

    monkeypatch.setattr(
        "cyclic_stability.nuclear.realize_nuclear_surfaces", fail_surface
    )

    roi = raw_structure_roi("golgi", dapi, config)

    assert np.any(roi)


def test_raw_signal_rows_serialize_explicit_aggregate_nucleus_id(
    analysis_config: AnalysisConfig,
) -> None:
    """Field-level raw-signal rows must state their aggregate analysis unit."""
    rows = measure_signal(
        analysis_config.cell("golgi", "Cell1").rounds[1],
        synthetic_target(),
        synthetic_roi(),
        _ensemble(analysis_config),
    )
    assert rows
    dictionaries = [
        {field: getattr(row, field) for field in row.__dataclass_fields__}
        for row in rows
    ]
    assert all(row["nucleus_id"] == "aggregate" for row in dictionaries)


@pytest.mark.parametrize("round_index", (1, 2, 3, 4))
def test_later_round_signal_is_labeled_as_matched_settings(
    analysis_config: AnalysisConfig, round_index: int
) -> None:
    """Grouping rounds two through five with exposure sensitivity would misstate the design."""
    rows = measure_signal(
        analysis_config.cell("golgi", "Cell1").rounds[round_index],
        synthetic_target(),
        synthetic_roi(),
        _ensemble(analysis_config),
    )
    assert all(row.comparison_set == "matched-settings" for row in rows)
    assert all(row.exposure_normalized for row in rows)


def test_signal_integrates_pristine_original_not_smoothed_intensities(
    analysis_config: AnalysisConfig,
) -> None:
    """Replacing raw signal with a smoothed image changes its corrected integrated value."""
    target = synthetic_target()
    pristine = target.copy()
    structure_roi = synthetic_roi()
    ensemble = _ensemble(analysis_config)
    member = next(item for item in ensemble if item.mask_status == "nonempty")
    rows = measure_signal(
        analysis_config.cell("golgi", "Cell1").rounds[1],
        target,
        structure_roi,
        ensemble,
    )
    row = next(
        item
        for item in rows
        if (item.sigma_um, item.threshold_sbr)
        == (member.sigma_um, member.threshold_sbr)
    )
    expected_mask = structure_roi & member.mask_zyx
    independently_smoothed = gaussian_filter(
        pristine,
        sigma=(
            member.sigma_um / analysis_config.voxel_spacing_xyz_um[2],
            member.sigma_um / analysis_config.voxel_spacing_xyz_um[1],
            member.sigma_um / analysis_config.voxel_spacing_xyz_um[0],
        ),
    )
    assert np.array_equal(
        member.mask_zyx,
        (
            independently_smoothed
            > member.background.median + member.threshold_sbr * member.background.scale
        )
        & structure_roi,
    )
    expected = np.clip(pristine - member.background.median, 0.0, None)[
        expected_mask
    ].sum()
    assert row.integrated_per_ms == expected / 200.0
    assert row.raw_percentiles[50] == np.percentile(pristine[expected_mask], 50)
    assert np.array_equal(target, pristine)


def test_signal_retains_empty_member_as_an_explicit_unmeasured_row(
    analysis_config: AnalysisConfig,
) -> None:
    """An empty high-threshold member must not abort or masquerade as zero signal."""
    member = _ensemble(analysis_config)[0]
    empty_mask = np.zeros_like(member.mask_zyx)
    empty_mask.setflags(write=False)
    empty_member = replace(member, mask_zyx=empty_mask, mask_status="empty")
    rows = measure_signal(
        analysis_config.cell("golgi", "Cell1").rounds[1],
        synthetic_target(),
        synthetic_roi(),
        (empty_member,),
    )
    empty_rows = [row for row in rows if row.measurement_status == "empty-mask"]
    assert empty_rows
    assert all(row.integrated_per_ms is None for row in empty_rows)
    assert all(row.signal_to_background_ratio is None for row in empty_rows)
    assert all(not row.raw_percentiles for row in empty_rows)
    assert [(row.sigma_um, row.threshold_sbr) for row in empty_rows] == [
        (empty_member.sigma_um, empty_member.threshold_sbr)
    ]


def test_signal_rejects_round_outside_the_five_round_design(
    analysis_config: AnalysisConfig,
) -> None:
    """A synthetic or direct caller cannot silently invent a matched-settings round."""
    invalid_round = replace(
        analysis_config.cell("golgi", "Cell1").rounds[0], acquisition_round=6
    )
    with pytest.raises(ValueError, match="1--5"):
        measure_signal(
            invalid_round,
            synthetic_target(),
            synthetic_roi(),
            _ensemble(analysis_config),
        )


@pytest.mark.parametrize("invalid_value", (np.nan, np.inf, -1.0, 0.5, 2.0))
def test_signal_rejects_nonbinary_structure_roi(
    analysis_config: AnalysisConfig, invalid_value: float
) -> None:
    """Invalid ROI values must fail rather than becoming target-selected foreground."""
    roi = synthetic_roi().astype(np.float64)
    roi[0, 0, 0] = invalid_value
    with pytest.raises(ValueError, match="binary"):
        measure_signal(
            analysis_config.cell("golgi", "Cell1").rounds[1],
            synthetic_target(),
            roi,
            _ensemble(analysis_config),
        )
