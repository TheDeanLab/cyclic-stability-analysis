"""Per-cell Golgi continuous-image and threshold-ensemble stability measures."""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt, gaussian_filter, label
from scipy.spatial import cKDTree
from skimage.measure import mesh_surface_area

from cyclic_stability.cyexm_artifacts import RoundArtifacts
from cyclic_stability.images import load_volume, robust_background
from cyclic_stability.models import (
    AnalysisConfig,
    CellSpec,
    EnsembleRow,
    MarkerResult,
    MetricRow,
)
from cyclic_stability.morphology import (
    gaussian_sigma_zyx,
    resolution_surface_metrics,
    threshold_ensemble,
    validated_binary_mask,
)
from cyclic_stability.nuclear import (
    REFERENCE_ROUND,
    NuclearSurface,
    _accepted_rounds,
    _atomic_publish_directory,
    _surface_from_mask,
    consensus_nuclear_surface,
)
from cyclic_stability.units import theoretical_resolution


@dataclass(frozen=True, slots=True)
class GradientSimilarityRow:
    """One physical smoothing-scale gradient correlation in the fixed perinuclear ROI."""

    sigma_um: float
    correlation: float


def _volume(image: NDArray[np.generic], name: str) -> NDArray[np.float64]:
    """Validate a finite real ZYX volume at this public numerical boundary."""
    value = np.asarray(image)
    if (
        value.ndim != 3
        or not np.issubdtype(value.dtype, np.number)
        or np.issubdtype(value.dtype, np.complexfloating)
    ):
        raise ValueError(f"{name} must be one real numeric 3D ZYX volume")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values")
    return value.astype(np.float64, copy=False)


def _spacing(config: AnalysisConfig) -> tuple[float, float, float]:
    """Validate public XYZ spacing through the shared physical-sigma primitive."""
    # ``gaussian_sigma_zyx`` owns the canonical public-XYZ validation.
    gaussian_sigma_zyx(
        config.ensembles.gaussian_sigma_um[0], config.voxel_spacing_xyz_um
    )
    return config.voxel_spacing_xyz_um


def _gradient_magnitude(
    image: NDArray[np.float64],
    spacing_xyz_um: tuple[float, float, float],
    sigma_um: float,
) -> NDArray[np.float64]:
    """Background-correct, physically smooth, and differentiate one target volume."""
    background = robust_background(image)
    corrected = np.clip(image - background.median, 0.0, None)
    smoothed = gaussian_filter(corrected, gaussian_sigma_zyx(sigma_um, spacing_xyz_um))
    sx, sy, sz = spacing_xyz_um
    gradients = np.gradient(smoothed, sz, sy, sx)
    return np.sqrt(sum(gradient * gradient for gradient in gradients))


def multiscale_gradient_similarity(
    reference: NDArray[np.generic],
    moving: NDArray[np.generic],
    roi: NDArray[np.bool_],
    spacing_xyz_um: tuple[float, float, float],
    sigma_values_um: tuple[float, ...],
) -> tuple[GradientSimilarityRow, ...]:
    """Correlate 3D gradient magnitudes at each frozen physical smoothing scale.

    The target-independent ``roi`` is fixed before any target thresholding.  Each
    image receives its own robust background correction before physical smoothing,
    so a global positive intensity scale and offset do not change this similarity.
    """
    fixed = _volume(reference, "reference")
    moved = _volume(moving, "moving")
    support = validated_binary_mask(roi, "ROI")
    if fixed.shape != moved.shape or fixed.shape != support.shape:
        raise ValueError("reference, moving, and ROI must be matching 3D ZYX volumes")
    if not np.any(support):
        raise ValueError("gradient similarity ROI is empty")
    if not sigma_values_um:
        raise ValueError("sigma_values_um must be non-empty")
    rows: list[GradientSimilarityRow] = []
    for raw_sigma in sigma_values_um:
        sigma_um = float(raw_sigma)
        fixed_gradient = _gradient_magnitude(fixed, spacing_xyz_um, sigma_um)[support]
        moving_gradient = _gradient_magnitude(moved, spacing_xyz_um, sigma_um)[support]
        if np.isclose(np.std(fixed_gradient), 0.0) or np.isclose(
            np.std(moving_gradient), 0.0
        ):
            raise ValueError("gradient similarity has zero variance in the fixed ROI")
        correlation = float(np.corrcoef(fixed_gradient, moving_gradient)[0, 1])
        if not math.isfinite(correlation):
            raise ValueError("gradient similarity correlation is non-finite")
        rows.append(GradientSimilarityRow(sigma_um, correlation))
    return tuple(rows)


def golgi_nucleus_distances(
    mask: NDArray[np.bool_],
    nuclear_surface: NuclearSurface,
    spacing_xyz_um: tuple[float, float, float],
) -> NDArray[np.float64]:
    """Return physical XYZ target-voxel distances to the consensus nuclear surface."""
    positive = validated_binary_mask(mask, "Golgi mask")
    if positive.ndim != 3 or positive.shape != nuclear_surface.mask_zyx.shape:
        raise ValueError("Golgi mask and nuclear surface must share 3D ZYX geometry")
    if tuple(spacing_xyz_um) != tuple(nuclear_surface.spacing_xyz_um):
        raise ValueError("Golgi and nuclear-surface spacing must match")
    indices_zyx = np.argwhere(positive)
    if indices_zyx.size == 0:
        return np.empty(0, dtype=np.float64)
    sx, sy, sz = spacing_xyz_um
    points_xyz = indices_zyx[:, ::-1] * np.array((sx, sy, sz))
    return cKDTree(nuclear_surface.vertices_xyz_um).query(points_xyz, k=1)[0]


def _component_volume_gini(volumes_um3: NDArray[np.float64]) -> float:
    """Return the finite-sample Gini coefficient of positive component volumes."""
    values = np.sort(np.asarray(volumes_um3, dtype=np.float64))
    if (
        values.ndim != 1
        or values.size == 0
        or np.any(~np.isfinite(values))
        or np.any(values <= 0.0)
    ):
        raise ValueError("component volumes must be a non-empty finite positive vector")
    if values.size == 1:
        return 0.0
    ranks = np.arange(1, values.size + 1, dtype=np.float64)
    population_gini = (
        2.0 * float(np.dot(ranks, values)) / (values.size * float(values.sum()))
        - (values.size + 1.0) / values.size
    )
    return float(population_gini * values.size / (values.size - 1.0))


def _mask_geometry(
    mask: NDArray[np.bool_], spacing_xyz_um: tuple[float, float, float]
) -> dict[str, float]:
    """Measure all positive components, preserving explicit empty-mask semantics."""
    positive = validated_binary_mask(mask, "Golgi mask")
    if positive.ndim != 3:
        raise ValueError("Golgi mask must be 3D ZYX")
    voxel_volume = float(np.prod(spacing_xyz_um))
    volume = float(positive.sum() * voxel_volume)
    if volume == 0.0:
        return {
            "volume_um3": 0.0,
            "component_count": 0.0,
            "component_volume_gini": math.nan,
            "fragmentation_per_um3": math.nan,
            "compactness": math.nan,
            "area_to_volume_um_inverse": math.nan,
        }
    labels, count = label(positive, structure=np.ones((3, 3, 3), dtype=bool))
    sizes = np.bincount(labels.ravel())[1:]
    component_volumes = sizes.astype(np.float64) * voxel_volume
    vertices, faces = _surface_from_mask(positive, spacing_xyz_um)
    area = float(mesh_surface_area(vertices, faces))
    return {
        "volume_um3": volume,
        "component_count": float(count),
        "component_volume_gini": _component_volume_gini(component_volumes),
        "fragmentation_per_um3": float(count / volume),
        "compactness": float(36.0 * math.pi * volume * volume / area**3),
        "area_to_volume_um_inverse": area / volume,
    }


def _perinuclear_roi(
    nuclear_mask: NDArray[np.bool_],
    rounds: Mapping[int, NDArray[np.generic]],
    config: AnalysisConfig,
) -> NDArray[np.bool_]:
    """Build one fixed exterior 18 um ROI intersected with all valid target voxels."""
    nucleus = validated_binary_mask(nuclear_mask, "consensus nuclear mask")
    if nucleus.ndim != 3:
        raise ValueError("consensus nuclear mask must be 3D ZYX")
    if not rounds:
        raise ValueError("Golgi ROI requires at least one registered round")
    valid = np.ones(nucleus.shape, dtype=bool)
    for round_number, image in rounds.items():
        volume = _volume(image, f"Golgi round {round_number}")
        if volume.shape != nucleus.shape:
            raise ValueError(
                "Golgi rounds and consensus nuclear mask must share geometry"
            )
        valid &= np.isfinite(volume)
    sx, sy, sz = _spacing(config)
    exterior_distance = distance_transform_edt(~nucleus, sampling=(sz, sy, sx))
    roi = (
        (~nucleus)
        & (exterior_distance <= config.golgi.perinuclear_outer_distance_um)
        & valid
    )
    if not np.any(roi):
        raise ValueError("fixed perinuclear ROI is empty")
    roi.setflags(write=False)
    return roi


def _ensemble_row(
    round_number: int,
    endpoint: str,
    sigma_um: float,
    threshold_sbr: float,
    value: float,
    units: str,
    config: AnalysisConfig,
) -> EnsembleRow:
    return EnsembleRow(
        "golgi",
        "array",
        round_number,
        endpoint,
        f"sigma={sigma_um:g};sbr={threshold_sbr:g}",
        threshold_sbr,
        value,
        units,
        config.config_digest,
    )


def _comparison_values(
    reference_mask: NDArray[np.bool_],
    moving_mask: NDArray[np.bool_],
    config: AnalysisConfig,
) -> dict[str, float]:
    """Return reference-round surface measures without favorable empty agreement."""
    if not np.any(reference_mask) or not np.any(moving_mask):
        return {
            "jaccard": math.nan,
            "fixed_to_moving_boundary_distance_p95": math.nan,
            "moving_to_fixed_boundary_distance_p95": math.nan,
            **{
                f"surface_dice_{tolerance:g}": math.nan
                for tolerance in config.ensembles.surface_tolerance_resolution_units
            },
        }
    resolution = theoretical_resolution(
        config.optics.target_wavelength_um,
        config.optics.numerical_aperture,
        config.optics.immersion_refractive_index,
        config.nominal_expansion_factor,
    )
    metrics = resolution_surface_metrics(
        reference_mask,
        moving_mask,
        config.voxel_spacing_xyz_um,
        resolution,
        config.ensembles.surface_tolerance_resolution_units,
    )
    return {
        "jaccard": metrics.jaccard,
        "fixed_to_moving_boundary_distance_p95": float(
            metrics.fixed_to_moving_normalized_distance_p95
        ),
        "moving_to_fixed_boundary_distance_p95": float(
            metrics.moving_to_fixed_normalized_distance_p95
        ),
        **{
            f"surface_dice_{tolerance:g}": value
            for tolerance, value in metrics.surface_dice.items()
        },
    }


def analyze_golgi_arrays(
    golgi_rounds: Mapping[int, NDArray],
    dapi_rounds: Mapping[int, NDArray],
    config: AnalysisConfig,
) -> MarkerResult:
    """Analyze one cell's registered Golgi arrays with round 2 fixed as reference.

    This numerical kernel makes no filesystem writes and has no cohort aggregation.
    Its ``array`` identity is rebound to the validated cell only by
    :func:`analyze_golgi_cell`, the I/O-owning entry point.
    """
    if REFERENCE_ROUND not in golgi_rounds or REFERENCE_ROUND not in dapi_rounds:
        raise ValueError("Golgi analysis requires acquisition round 2")
    if set(golgi_rounds) != set(dapi_rounds):
        raise ValueError(
            "registered Golgi and DAPI rounds must have identical inventories"
        )
    rounds = {
        int(number): _volume(image, f"Golgi round {number}")
        for number, image in golgi_rounds.items()
    }
    surface = consensus_nuclear_surface(
        dapi_rounds, config.voxel_spacing_xyz_um, config.nuclear
    )
    roi = _perinuclear_roi(surface.mask_zyx, rounds, config)
    reference = rounds[REFERENCE_ROUND]
    ensemble_by_round = {
        round_number: threshold_ensemble(
            image,
            roi,
            config.ensembles.gaussian_sigma_um,
            config.ensembles.threshold_sbr,
            config.voxel_spacing_xyz_um,
        )
        for round_number, image in rounds.items()
    }
    reference_members = ensemble_by_round[REFERENCE_ROUND]
    ensemble_rows: list[EnsembleRow] = []
    summary_rows: list[MetricRow] = []
    for round_number in sorted(rounds):
        similarities = multiscale_gradient_similarity(
            reference,
            rounds[round_number],
            roi,
            config.voxel_spacing_xyz_um,
            config.ensembles.gaussian_sigma_um,
        )
        comparison_set = (
            "round-1-exposure-sensitivity" if round_number == 1 else "matched-settings"
        )
        for row in similarities:
            summary_rows.append(
                MetricRow(
                    "golgi",
                    "array",
                    round_number,
                    f"golgi_gradient_similarity_sigma_{row.sigma_um:g}",
                    row.correlation,
                    "correlation",
                    comparison_set,
                    True,
                    config.config_digest,
                )
            )
        summary_rows.append(
            MetricRow(
                "golgi",
                "array",
                round_number,
                "golgi_gradient_similarity_median",
                float(np.median([row.correlation for row in similarities])),
                "correlation",
                comparison_set,
                True,
                config.config_digest,
            )
        )
        current_values: dict[str, list[float]] = {}
        for reference_member, member in zip(
            reference_members, ensemble_by_round[round_number], strict=True
        ):
            geometry = _mask_geometry(member.mask_zyx, config.voxel_spacing_xyz_um)
            distances = golgi_nucleus_distances(
                member.mask_zyx, surface, config.voxel_spacing_xyz_um
            )
            background = member.background
            corrected_signal = float(
                np.clip(rounds[round_number] - background.median, 0.0, None)[
                    member.mask_zyx
                ].sum()
            )
            values = {
                "absolute_background_corrected_signal": corrected_signal,
                **geometry,
                "nuclear_distance_p25_um": float(np.percentile(distances, 25.0))
                if distances.size
                else math.nan,
                "nuclear_distance_p50_um": float(np.percentile(distances, 50.0))
                if distances.size
                else math.nan,
                "nuclear_distance_p75_um": float(np.percentile(distances, 75.0))
                if distances.size
                else math.nan,
                **_comparison_values(
                    reference_member.mask_zyx, member.mask_zyx, config
                ),
            }
            units = {
                "absolute_background_corrected_signal": "a.u.",
                "volume_um3": "um3",
                "component_count": "count",
                "component_volume_gini": "fraction",
                "fragmentation_per_um3": "1/um3",
                "compactness": "fraction",
                "area_to_volume_um_inverse": "1/um",
                "nuclear_distance_p25_um": "um",
                "nuclear_distance_p50_um": "um",
                "nuclear_distance_p75_um": "um",
                "jaccard": "fraction",
                "fixed_to_moving_boundary_distance_p95": "resolution-units",
                "moving_to_fixed_boundary_distance_p95": "resolution-units",
                **{
                    f"surface_dice_{tolerance:g}": "fraction"
                    for tolerance in config.ensembles.surface_tolerance_resolution_units
                },
            }
            for endpoint, value in values.items():
                ensemble_rows.append(
                    _ensemble_row(
                        round_number,
                        endpoint,
                        member.sigma_um,
                        member.threshold_sbr,
                        value,
                        units[endpoint],
                        config,
                    )
                )
                if math.isfinite(value):
                    current_values.setdefault(endpoint, []).append(value)
        for endpoint, values in current_values.items():
            summary_rows.append(
                MetricRow(
                    "golgi",
                    "array",
                    round_number,
                    f"golgi_{endpoint}_median",
                    float(np.median(values)),
                    units[endpoint],
                    comparison_set,
                    True,
                    config.config_digest,
                )
            )
    return MarkerResult(
        "golgi", "array", tuple(summary_rows), tuple(ensemble_rows), (), ()
    )


def _rebind_cell(result: MarkerResult, cell: CellSpec) -> MarkerResult:
    """Replace array-kernel placeholder identity without changing numerical results."""
    return MarkerResult(
        cell.cohort,
        cell.cell_id,
        tuple(
            replace(row, cohort=cell.cohort, cell_id=cell.cell_id)
            for row in result.summary_rows
        ),
        tuple(
            replace(row, cohort=cell.cohort, cell_id=cell.cell_id)
            for row in result.ensemble_rows
        ),
        result.diagnostics,
        result.input_identities,
    )


def _write_diagnostics(
    output_dir: Path, rounds: Mapping[int, NDArray], result: MarkerResult
) -> tuple[Path, ...]:
    """Atomically publish registered overlays and the complete threshold-response table."""
    staged_paths: tuple[Path, ...] = ()
    staging_root: Path | None = None

    def populate(staged: Path) -> None:
        nonlocal staged_paths, staging_root
        staging_root = staged
        overlay = staged / "golgi-registered-overlays.npz"
        np.savez_compressed(
            overlay, **{f"round_{number}": image for number, image in rounds.items()}
        )
        curves = staged / "golgi-threshold-robustness.csv"
        with curves.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(EnsembleRow.__dataclass_fields__)
            writer.writerows(
                [getattr(row, name) for name in EnsembleRow.__dataclass_fields__]
                for row in result.ensemble_rows
            )
        staged_paths = (overlay, curves)

    _atomic_publish_directory(output_dir, populate)
    assert staging_root is not None
    return tuple(output_dir / path.relative_to(staging_root) for path in staged_paths)


def analyze_golgi_cell(
    cell: CellSpec,
    artifacts: tuple[RoundArtifacts, ...],
    config: AnalysisConfig,
    output_dir: Path,
) -> MarkerResult:
    """Load validated registered Golgi inputs and publish one cell's diagnostics."""
    if cell.cohort != "golgi":
        raise ValueError("Golgi analysis requires a golgi CellSpec")
    by_round = {artifact.acquisition_round: artifact for artifact in artifacts}
    if set(by_round) != {round_spec.acquisition_round for round_spec in cell.rounds}:
        raise ValueError(
            "cell and validated artifact acquisition-round inventories disagree"
        )
    accepted = _accepted_rounds(cell, config)
    dapi = {
        number: load_volume(by_round[number].registered_dapi) for number in accepted
    }
    golgi = {
        number: load_volume(by_round[number].registered_target) for number in accepted
    }
    result = _rebind_cell(analyze_golgi_arrays(golgi, dapi, config), cell)
    diagnostics = _write_diagnostics(output_dir, golgi, result)
    return replace(
        result,
        diagnostics=diagnostics,
        input_identities=tuple(
            str(path)
            for number in accepted
            for path in (
                by_round[number].registered_dapi,
                by_round[number].registered_target,
            )
        ),
    )


__all__ = [
    "GradientSimilarityRow",
    "analyze_golgi_arrays",
    "analyze_golgi_cell",
    "golgi_nucleus_distances",
    "multiscale_gradient_similarity",
]
