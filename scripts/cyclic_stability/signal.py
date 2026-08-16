"""Unregistered target-channel repeated-label signal summaries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from cyclic_stability.models import RoundSpec
from cyclic_stability.morphology import MaskMember


@dataclass(frozen=True, slots=True)
class SignalRow:
    """One raw-target signal summary for a structure ROI and ensemble member."""

    cohort: str
    cell_id: str
    acquisition_round: int
    comparison_set: str
    exposure_normalized: bool
    sigma_um: float
    threshold_sbr: float
    integrated_per_ms: float
    signal_to_background_ratio: float
    positive_volume_um3: float
    raw_percentiles: Mapping[int, float]
    corrected_percentiles: Mapping[int, float]


PERCENTILES = (10, 25, 50, 75, 90, 95, 99)


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
    supplied structure ROI.  Correction and integration always use the original
    target intensities, never the image used to construct a threshold mask.
    """
    target = np.asarray(target_zyx)
    roi = np.asarray(structure_roi, dtype=bool)
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
        mask = roi & member.mask_zyx
        raw = target_float[mask]
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
                exposure_normalized=exposure_sensitivity,
                sigma_um=member.sigma_um,
                threshold_sbr=member.threshold_sbr,
                integrated_per_ms=integrated_per_ms,
                signal_to_background_ratio=sbr,
                positive_volume_um3=positive_volume_um3,
                raw_percentiles=raw_percentiles,
                corrected_percentiles=corrected_percentiles,
            )
        )
    return tuple(rows)


__all__ = ["PERCENTILES", "SignalRow", "measure_signal"]
