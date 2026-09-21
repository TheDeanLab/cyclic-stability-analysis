"""Unregistered target-channel repeated-label signal summaries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt

from cyclic_stability.models import AnalysisConfig, RoundSpec
from cyclic_stability.morphology import MaskMember, validated_binary_mask


@dataclass(frozen=True, slots=True)
class SignalRow:
    """One raw-target summary or explicit no-measurement ensemble-member status.

    ``empty-mask`` rows preserve the member's sigma and threshold but keep all
    signal values absent. They must not be interpreted as measured zero signal.
    """

    cohort: str
    cell_id: str
    acquisition_round: int
    comparison_set: str
    exposure_normalized: bool
    sigma_um: float
    threshold_sbr: float
    measurement_status: Literal["measured", "empty-mask"]
    integrated_per_ms: float | None
    signal_to_background_ratio: float | None
    positive_volume_um3: float | None
    raw_percentiles: Mapping[int, float]
    corrected_percentiles: Mapping[int, float]
    nucleus_id: str = "aggregate"


PERCENTILES = (10, 25, 50, 75, 90, 95, 99)


def raw_structure_roi(
    cohort: str, raw_dapi_zyx: NDArray[np.generic], config: AnalysisConfig
) -> NDArray[np.bool_]:
    """Return a target-independent, raw-DAPI structure ROI for one acquisition.

    This shared Task 4 helper deliberately never receives a target image.  A
    per-round consensus call avoids cross-round native-grid assumptions.
    """
    from cyclic_stability.nuclear import (
        _adjudicated_nuclear_mask,
        adjudicate_nuclear_components,
    )

    adjudication = adjudicate_nuclear_components(
        {config.reference_acquisition_round: raw_dapi_zyx},
        config.voxel_spacing_xyz_um,
        config.nuclear,
        config.nominal_expansion_factor,
    )
    nuclear_mask = _adjudicated_nuclear_mask(adjudication, config.nuclear)
    if cohort == "lamin":
        return nuclear_mask
    if cohort == "golgi":
        distance = distance_transform_edt(
            ~nuclear_mask, sampling=config.voxel_spacing_xyz_um[::-1]
        )
        return distance <= config.golgi.perinuclear_outer_distance_um
    if cohort == "lamp1":
        return ~nuclear_mask
    raise ValueError(f"unsupported cohort for raw signal ROI: {cohort}")


def _percentiles(values: NDArray[np.float64]) -> Mapping[int, float]:
    """Return the fixed audited intensity percentile set for non-empty values."""
    if values.size == 0:
        raise ValueError("signal mask is empty inside the independent structure ROI")
    return MappingProxyType(
        {
            percentile: float(np.percentile(values, percentile))
            for percentile in PERCENTILES
        }
    )


def measure_signal(
    round_spec: RoundSpec,
    target_zyx: NDArray[np.generic],
    structure_roi: NDArray[np.bool_],
    ensemble: tuple[MaskMember, ...],
) -> tuple[SignalRow, ...]:
    """Measure raw unregistered target signal within a target-independent ROI.

    Each member's smoothed threshold mask is intersected with the independently
    supplied structure ROI. Correction and integration always use the original
    target intensities, never the image used to construct a threshold mask. Empty
    intersections emit an explicit unmeasured row so no configured curve member is
    lost or reported as zero signal.
    """
    target = np.asarray(target_zyx)
    roi = validated_binary_mask(structure_roi, "structure ROI")
    if target.ndim != 3 or roi.ndim != 3 or target.shape != roi.shape:
        raise ValueError("target and structure ROI must be matching 3D ZYX volumes")
    if not np.issubdtype(target.dtype, np.number) or np.issubdtype(
        target.dtype, np.complexfloating
    ):
        raise ValueError("target must have a real numeric dtype")
    if not np.all(np.isfinite(target)):
        raise ValueError("target must contain only finite values")
    if not ensemble:
        raise ValueError("ensemble must contain at least one mask member")
    if round_spec.acquisition_round not in {1, 2, 3, 4, 5}:
        raise ValueError("acquisition_round must be in 1--5")
    if round_spec.target_exposure_ms <= 0.0:
        raise ValueError("target exposure must be positive")
    target_float = target.astype(np.float64, copy=False)
    exposure_sensitivity = round_spec.acquisition_round == 1
    comparison_set = (
        "round-1-exposure-sensitivity" if exposure_sensitivity else "matched-settings"
    )
    rows: list[SignalRow] = []
    for member in ensemble:
        if member.mask_zyx.shape != target.shape:
            raise ValueError("ensemble mask must match target geometry")
        mask = roi & validated_binary_mask(member.mask_zyx, "ensemble mask")
        raw = target_float[mask]
        if raw.size == 0:
            rows.append(
                SignalRow(
                    cohort=round_spec.cohort,
                    cell_id=round_spec.cell_id,
                    acquisition_round=round_spec.acquisition_round,
                    comparison_set=comparison_set,
                    exposure_normalized=True,
                    sigma_um=member.sigma_um,
                    threshold_sbr=member.threshold_sbr,
                    measurement_status="empty-mask",
                    integrated_per_ms=None,
                    signal_to_background_ratio=None,
                    positive_volume_um3=None,
                    raw_percentiles=MappingProxyType({}),
                    corrected_percentiles=MappingProxyType({}),
                )
            )
            continue
        corrected = np.clip(target_float - member.background.median, 0.0, None)
        corrected_values = corrected[mask]
        raw_percentiles = _percentiles(raw)
        corrected_percentiles = _percentiles(corrected_values)
        integrated_per_ms = float(
            corrected_values.sum() / round_spec.target_exposure_ms
        )
        sbr = float(
            (np.median(raw) - member.background.median) / member.background.scale
        )
        positive_volume_um3 = float(mask.sum() * np.prod(member.spacing_xyz_um))
        rows.append(
            SignalRow(
                cohort=round_spec.cohort,
                cell_id=round_spec.cell_id,
                acquisition_round=round_spec.acquisition_round,
                comparison_set=comparison_set,
                exposure_normalized=True,
                sigma_um=member.sigma_um,
                threshold_sbr=member.threshold_sbr,
                measurement_status="measured",
                integrated_per_ms=integrated_per_ms,
                signal_to_background_ratio=sbr,
                positive_volume_um3=positive_volume_um3,
                raw_percentiles=raw_percentiles,
                corrected_percentiles=corrected_percentiles,
            )
        )
    return tuple(rows)


__all__ = ["PERCENTILES", "SignalRow", "measure_signal", "raw_structure_roi"]
