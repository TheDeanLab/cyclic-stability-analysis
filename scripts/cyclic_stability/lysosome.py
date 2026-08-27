"""Gated LAMP1 object detection and one-to-one resolution-aware matching."""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from itertools import pairwise
from pathlib import Path
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import (
    binary_erosion,
    gaussian_filter,
    maximum_filter,
)
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes, mesh_surface_area
from skimage.segmentation import watershed

from cyclic_stability.cyexm_artifacts import RoundArtifacts
from cyclic_stability.golgi import _artifact_validity_masks
from cyclic_stability.images import BackgroundSummary, load_volume
from cyclic_stability.models import (
    AnalysisConfig,
    CellSpec,
    EnsembleRow,
    MarkerResult,
    MetricRow,
)
from cyclic_stability.morphology import gaussian_sigma_zyx, validated_binary_mask
from cyclic_stability.nuclear import (
    REFERENCE_ROUND,
    NuclearSurface,
    _accepted_rounds,
    _atomic_publish_directory,
    consensus_nuclear_surface,
    write_nuclear_component_diagnostic,
)
from cyclic_stability.units import ResolutionReference, theoretical_resolution


@dataclass(frozen=True, slots=True)
class LysosomeObject:
    """One watershed-separated LAMP1 object in physical XYZ coordinates."""

    centroid_xyz_um: tuple[float, float, float]
    volume_um3: float
    equivalent_diameter_um: float
    sphericity: float
    nearest_neighbor_distance_um: float
    nucleus_relative_distance_um: float

    def __post_init__(self) -> None:
        if len(self.centroid_xyz_um) != 3 or not all(
            math.isfinite(value) for value in self.centroid_xyz_um
        ):
            raise ValueError(
                "lysosome object coordinates must be three finite XYZ values"
            )


@dataclass(frozen=True, slots=True)
class DetectionSet:
    """All detected LAMP1 objects for one physical LoG/SBR ensemble member."""

    sigma_um: float
    detection_sbr: float
    objects: tuple[LysosomeObject, ...]

    @property
    def object_count(self) -> int:
        """Return the number of watershed objects in this ensemble member."""
        return len(self.objects)

    @property
    def coordinates_xyz_um(self) -> tuple[tuple[float, float, float], ...]:
        """Return stable object centroids in physical XYZ micrometers."""
        return tuple(object_.centroid_xyz_um for object_ in self.objects)


@dataclass(frozen=True, slots=True)
class ObjectMatchResult:
    """One finite, one-to-one LAMP1 assignment with explicit lost/new outcomes."""

    matched_pairs: tuple[tuple[int, int], ...]
    normalized_displacements: tuple[float, ...]
    lateral_displacements_um: tuple[float, ...]
    axial_displacements_um: tuple[float, ...]
    matched_count: int
    lost_count: int
    new_count: int
    matched_fraction: float
    lost_fraction: float
    new_fraction: float


def _spacing(
    spacing_xyz_um: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Validate public XYZ spacing through the shared physical conversion helper."""
    gaussian_sigma_zyx(1.0, spacing_xyz_um)
    return tuple(float(value) for value in spacing_xyz_um)


def _volume(image: NDArray[np.generic], name: str) -> NDArray[np.float64]:
    """Return exactly one finite, real 3-D ZYX volume."""
    values = np.asarray(image)
    if (
        values.ndim != 3
        or not np.issubdtype(values.dtype, np.number)
        or np.issubdtype(values.dtype, np.complexfloating)
    ):
        raise ValueError(f"{name} must be one real numeric 3D ZYX volume")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    return values.astype(np.float64, copy=False)


def _positive_values(values: tuple[float, ...], name: str) -> tuple[float, ...]:
    """Validate one non-empty ordered physical parameter sequence."""
    result = tuple(float(value) for value in values)
    if not result or any(not math.isfinite(value) or value <= 0.0 for value in result):
        raise ValueError(f"{name} must contain finite positive values")
    return result


def _maxima(
    response: NDArray[np.float64],
    roi: NDArray[np.bool_],
    threshold: float,
    spacing_xyz_um: tuple[float, float, float],
    resolution: ResolutionReference,
) -> tuple[tuple[int, int, int], ...]:
    """Select 26-neighbour maxima with lateral/lateral/axial duplicate units."""
    maxima = (
        response == maximum_filter(response, size=(3, 3, 3), mode="nearest")
    ) & roi
    candidates = np.argwhere(maxima & (response > threshold))
    ordered = sorted(
        candidates.tolist(), key=lambda item: (-response[tuple(item)], *item)
    )
    sx, sy, sz = spacing_xyz_um
    resolution_xyz = np.array(
        (
            resolution.lateral_expanded_um,
            resolution.lateral_expanded_um,
            resolution.axial_expanded_um,
        ),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(resolution_xyz)) or np.any(resolution_xyz <= 0.0):
        raise ValueError(
            "resolution must contain finite positive expanded-space values"
        )
    retained: list[tuple[int, int, int]] = []
    for zyx in ordered:
        xyz = np.array((zyx[2] * sx, zyx[1] * sy, zyx[0] * sz), dtype=np.float64)
        if all(
            np.linalg.norm(
                (xyz - np.array((point[2] * sx, point[1] * sy, point[0] * sz)))
                / resolution_xyz
            )
            > 1.0
            for point in retained
        ):
            retained.append(tuple(zyx))
    return tuple(retained)


def _physical_log_response(
    image: NDArray[np.float64],
    sigma_um: float,
    spacing_xyz_um: tuple[float, float, float],
) -> NDArray[np.float64]:
    """Return scale-normalized negative physical Laplacian in source-intensity units.

    SciPy derivative filters are expressed per voxel coordinate.  Multiplying each
    second derivative by that axis's squared voxel sigma yields ``sigma_um**2``
    times the corresponding physical-coordinate derivative, preserving response
    units across anisotropic samplings.
    """
    sigma_zyx = gaussian_sigma_zyx(sigma_um, spacing_xyz_um)
    response = np.zeros_like(image, dtype=np.float64)
    for axis, sigma_vox in enumerate(sigma_zyx):
        order = [0, 0, 0]
        order[axis] = 2
        response -= sigma_vox**2 * gaussian_filter(
            image, sigma=sigma_zyx, order=tuple(order), mode="nearest"
        )
    return response


def _support_background(
    image: NDArray[np.float64], support: NDArray[np.bool_]
) -> BackgroundSummary:
    """Estimate robust background only from finite positive transform-valid samples."""
    positive = image[support & (image > 0.0)]
    if positive.size == 0:
        raise ValueError("support background has no positive voxels")
    upper = float(np.percentile(positive, 30.0))
    background = positive[positive <= upper]
    if background.size == 0:
        raise ValueError("support background selection is empty")
    median = float(np.median(background))
    scale = float(1.4826 * np.median(np.abs(background - median)))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("support background robust scale must be finite and positive")
    return BackgroundSummary(median, scale, int(positive.size), int(background.size))


def _support_safe_physical_log(
    image: NDArray[np.float64],
    support: NDArray[np.bool_],
    sigma_um: float,
    spacing_xyz_um: tuple[float, float, float],
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Return masked-normalized physical LoG only where its 3-point stencil is safe."""
    sigma_zyx = gaussian_sigma_zyx(sigma_um, spacing_xyz_um)
    weights = support.astype(np.float64, copy=False)
    denominator = gaussian_filter(weights, sigma_zyx, mode="nearest")
    smoothed = np.divide(
        gaussian_filter(image * weights, sigma_zyx, mode="nearest"),
        denominator,
        out=np.zeros_like(image),
        where=denominator > np.finfo(np.float64).eps,
    )
    safe = binary_erosion(support, structure=np.ones((3, 3, 3), dtype=bool))
    if not np.any(safe):
        raise ValueError(
            "LAMP1 support-boundary exclusion leaves no derivative stencil"
        )
    response = np.zeros_like(smoothed)
    for axis, sigma_vox in enumerate(sigma_zyx):
        ahead = [slice(None)] * 3
        center = [slice(None)] * 3
        behind = [slice(None)] * 3
        ahead[axis] = slice(2, None)
        center[axis] = slice(1, -1)
        behind[axis] = slice(None, -2)
        destination = [slice(None)] * 3
        destination[axis] = slice(1, -1)
        response[tuple(destination)] -= sigma_vox**2 * (
            smoothed[tuple(ahead)]
            - 2.0 * smoothed[tuple(center)]
            + smoothed[tuple(behind)]
        )
    return response, safe


def _surface_area(
    mask: NDArray[np.bool_], spacing_xyz_um: tuple[float, float, float]
) -> float:
    """Return a padded marching-cubes physical surface area for one non-empty object."""
    sx, sy, sz = spacing_xyz_um
    vertices, faces, _, _ = marching_cubes(
        np.pad(mask.astype(np.float32), 1), level=0.5, spacing=(sz, sy, sx)
    )
    return float(mesh_surface_area(vertices, faces))


def _objects(
    response: NDArray[np.float64],
    roi: NDArray[np.bool_],
    peaks: tuple[tuple[int, int, int], ...],
    spacing_xyz_um: tuple[float, float, float],
) -> tuple[LysosomeObject, ...]:
    """Watershed positive LoG support and summarize every retained physical object."""
    if not peaks:
        return ()
    markers = np.zeros(response.shape, dtype=np.int32)
    for index, peak in enumerate(peaks, start=1):
        markers[peak] = index
    labels = watershed(-response, markers, mask=(response > 0.0) & roi)
    sx, sy, sz = spacing_xyz_um
    objects: list[LysosomeObject] = []
    for index in range(1, len(peaks) + 1):
        mask = labels == index
        voxels = np.argwhere(mask)
        if not len(voxels):
            continue
        centroid_zyx = voxels.mean(axis=0)
        centroid = (
            float(centroid_zyx[2] * sx),
            float(centroid_zyx[1] * sy),
            float(centroid_zyx[0] * sz),
        )
        volume = float(mask.sum() * sx * sy * sz)
        area = _surface_area(mask, spacing_xyz_um)
        objects.append(
            LysosomeObject(
                centroid,
                volume,
                float((6.0 * volume / np.pi) ** (1.0 / 3.0)),
                float(np.pi ** (1.0 / 3.0) * (6.0 * volume) ** (2.0 / 3.0) / area),
                math.nan,
                math.nan,
            )
        )
    if len(objects) < 2:
        return tuple(objects)
    points = np.array([object_.centroid_xyz_um for object_ in objects])
    distances, _ = cKDTree(points).query(points, k=2)
    return tuple(
        replace(object_, nearest_neighbor_distance_um=float(distances[index, 1]))
        for index, object_ in enumerate(objects)
    )


def detect_lysosomes(
    image: NDArray[np.generic],
    roi: NDArray[np.generic],
    spacing_xyz_um: tuple[float, float, float],
    sigma_values_um: tuple[float, ...],
    detection_sbr: tuple[float, ...],
    *,
    resolution: ResolutionReference,
) -> tuple[DetectionSet, ...]:
    """Detect physical 3-D LoG LAMP1 objects for every declared ensemble member."""
    source = _volume(image, "LAMP1 image")
    structure_roi = validated_binary_mask(roi, "LAMP1 ROI")
    if source.shape != structure_roi.shape:
        raise ValueError("LAMP1 image and ROI must share a 3D ZYX geometry")
    spacing = _spacing(spacing_xyz_um)
    sigmas = _positive_values(sigma_values_um, "sigma_values_um")
    thresholds = _positive_values(detection_sbr, "detection_sbr")
    background = _support_background(source, structure_roi)
    corrected = source - background.median
    result: list[DetectionSet] = []
    for sigma_um in sigmas:
        response, derivative_support = _support_safe_physical_log(
            corrected, structure_roi, sigma_um, spacing
        )
        for threshold_sbr in thresholds:
            peaks = _maxima(
                response,
                derivative_support,
                threshold_sbr * background.scale,
                spacing,
                resolution,
            )
            result.append(
                DetectionSet(
                    sigma_um,
                    threshold_sbr,
                    _objects(response, derivative_support, peaks, spacing),
                )
            )
    return tuple(result)


def _coordinates(detections: DetectionSet) -> NDArray[np.float64]:
    """Return a finite N-by-3 XYZ coordinate matrix for a detection set."""
    result = np.asarray(detections.coordinates_xyz_um, dtype=np.float64)
    if result.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if result.shape != (detections.object_count, 3) or not np.all(np.isfinite(result)):
        raise ValueError("detection coordinates must be finite XYZ values")
    return result


def match_lysosomes(
    reference: DetectionSet,
    moving: DetectionSet,
    resolution: ResolutionReference,
    max_units: float,
) -> ObjectMatchResult:
    """Assign only finite one-to-one LAMP1 pairs within the stated resolution limit."""
    if not math.isfinite(max_units) or max_units <= 0.0:
        raise ValueError("max_units must be finite and positive")
    reference_xyz = _coordinates(reference)
    moving_xyz = _coordinates(moving)
    reference_count, moving_count = len(reference_xyz), len(moving_xyz)
    if not reference_count or not moving_count:
        return ObjectMatchResult(
            (),
            (),
            (),
            (),
            0,
            reference_count,
            moving_count,
            0.0,
            0.0 if not reference_count else 1.0,
            0.0 if not moving_count else 1.0,
        )
    resolution_xyz = np.array(
        (
            resolution.lateral_expanded_um,
            resolution.lateral_expanded_um,
            resolution.axial_expanded_um,
        ),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(resolution_xyz)) or np.any(resolution_xyz <= 0.0):
        raise ValueError(
            "resolution must contain finite positive expanded-space values"
        )
    delta = moving_xyz[:, None, :] - reference_xyz[None, :, :]
    cost = np.sqrt(np.sum((delta / resolution_xyz) ** 2, axis=-1))
    cost[cost > max_units] = np.inf
    possible_rows = np.any(np.isfinite(cost), axis=1)
    possible_cols = np.any(np.isfinite(cost), axis=0)
    pairs: list[tuple[int, int]] = []
    if np.any(possible_rows) and np.any(possible_cols):
        row_indices = np.flatnonzero(possible_rows)
        col_indices = np.flatnonzero(possible_cols)
        finite_cost = cost[np.ix_(row_indices, col_indices)]
        sentinel = (
            max_units
            + max(1.0, float(np.max(finite_cost[np.isfinite(finite_cost)])))
            * 1_000_000.0
        )
        assignment_rows, assignment_cols = linear_sum_assignment(
            np.where(np.isfinite(finite_cost), finite_cost, sentinel)
        )
        pairs = sorted(
            (
                (int(col_indices[col]), int(row_indices[row]))
                for row, col in zip(assignment_rows, assignment_cols, strict=True)
                if math.isfinite(float(finite_cost[row, col]))
            )
        )
    normalized = tuple(
        float(cost[moving_index, reference_index])
        for reference_index, moving_index in pairs
    )
    lateral = tuple(
        float(
            np.hypot(
                *(moving_xyz[moving_index, :2] - reference_xyz[reference_index, :2])
            )
        )
        for reference_index, moving_index in pairs
    )
    axial = tuple(
        float(abs(moving_xyz[moving_index, 2] - reference_xyz[reference_index, 2]))
        for reference_index, moving_index in pairs
    )
    matched = len(pairs)
    return ObjectMatchResult(
        tuple(pairs),
        normalized,
        lateral,
        axial,
        matched,
        reference_count - matched,
        moving_count - matched,
        matched / reference_count,
        (reference_count - matched) / reference_count,
        (moving_count - matched) / moving_count,
    )


def eligible_lamp_cells(config: AnalysisConfig) -> tuple[CellSpec, ...]:
    """Return only complete, image-verified LAMP1 cells in stable biological-ID order."""
    return tuple(
        cell
        for cell in config.cells
        if cell.cohort == "lamp1"
        and tuple(round_spec.acquisition_round for round_spec in cell.rounds)
        == (1, 2, 3, 4, 5)
        and all(
            round_spec.identity_method == "image-verified" for round_spec in cell.rounds
        )
    )


def _with_nuclear_distances(
    detections: DetectionSet,
    surfaces: tuple[NuclearSurface, ...],
) -> DetectionSet:
    """Annotate objects with minimum distance to any accepted nuclear surface."""
    if not surfaces:
        raise ValueError("LAMP1 nuclear distance requires accepted nuclei")
    spacing = surfaces[0].spacing_xyz_um
    if any(surface.spacing_xyz_um != spacing for surface in surfaces):
        raise ValueError("accepted nuclear surfaces must share physical spacing")
    vertices_xyz_um = np.concatenate(
        tuple(surface.vertices_xyz_um for surface in surfaces), axis=0
    )
    if not detections.objects:
        return detections
    coordinates_xyz_um = np.asarray(detections.coordinates_xyz_um, dtype=np.float64)
    distances = cKDTree(vertices_xyz_um).query(coordinates_xyz_um, k=1)[0]
    return replace(
        detections,
        objects=tuple(
            replace(object_, nucleus_relative_distance_um=float(distance))
            for object_, distance in zip(detections.objects, distances, strict=True)
        ),
    )


def _fixed_lamp_roi(
    nuclear_mask: NDArray[np.bool_],
    validity_masks: Mapping[int, NDArray[np.bool_]],
) -> NDArray[np.bool_]:
    """Return one target-independent exterior ROI on common producer-valid support."""
    if not validity_masks:
        raise ValueError("LAMP1 ROI requires producer-attested transform support")
    nucleus = validated_binary_mask(nuclear_mask, "consensus nuclear mask")
    if nucleus.ndim != 3:
        raise ValueError("consensus nuclear mask must be 3D ZYX")
    common = np.ones(nucleus.shape, dtype=bool)
    for number, raw_mask in validity_masks.items():
        mask = validated_binary_mask(raw_mask, f"validity mask for round {number}")
        if mask.ndim != 3 or mask.shape != common.shape:
            raise ValueError(
                "transform support and consensus nucleus must share geometry"
            )
        common &= mask
    roi = common & ~nucleus
    if not np.any(roi):
        raise ValueError("fixed LAMP1 ROI is empty")
    roi.setflags(write=False)
    return roi


def _median_or_nan(values: list[float]) -> float:
    """Return a median only when a finite measurement exists, without warnings."""
    finite = [value for value in values if math.isfinite(value)]
    return float(np.median(finite)) if finite else math.nan


def _comparison_specs(rounds: tuple[int, ...]) -> tuple[tuple[str, int, int], ...]:
    """Return explicit round-2-relative and adjacent comparison pairs in order."""
    ordered = tuple(sorted(rounds))
    return tuple(
        ("round2-relative", REFERENCE_ROUND, number) for number in ordered
    ) + tuple(
        ("adjacent", previous, current)
        for previous, current in pairwise(ordered)
        if current == previous + 1
    )


def _rows_for_detections(
    cell: CellSpec, round_number: int, detections: DetectionSet, config: AnalysisConfig
) -> tuple[EnsembleRow, ...]:
    """Emit all object summaries for one declared physical LoG/SBR coordinate."""
    name = f"log_sigma_um={detections.sigma_um:g};detection_sbr={detections.detection_sbr:g}"
    objects = detections.objects
    values = {
        "lysosome_object_count": (float(detections.object_count), "count"),
        "lysosome_volume_median": (
            float(np.median([item.volume_um3 for item in objects]))
            if objects
            else math.nan,
            "um3",
        ),
        "lysosome_equivalent_diameter_median": (
            float(np.median([item.equivalent_diameter_um for item in objects]))
            if objects
            else math.nan,
            "um",
        ),
        "lysosome_sphericity_median": (
            float(np.median([item.sphericity for item in objects]))
            if objects
            else math.nan,
            "fraction",
        ),
        "lysosome_nearest_neighbor_distance_median": (
            _median_or_nan([item.nearest_neighbor_distance_um for item in objects]),
            "um",
        ),
        "lysosome_nucleus_relative_distance_median": (
            _median_or_nan([item.nucleus_relative_distance_um for item in objects]),
            "um",
        ),
    }
    return tuple(
        EnsembleRow(
            cohort=cell.cohort,
            cell_id=cell.cell_id,
            acquisition_round=round_number,
            endpoint=endpoint,
            ensemble_name=name,
            ensemble_value=detections.detection_sbr,
            sigma_um=detections.sigma_um,
            threshold_sbr=detections.detection_sbr,
            surface_tolerance_resolution_units=None,
            value=value,
            units=units,
            source_task_identity=config.config_digest,
        )
        for endpoint, (value, units) in values.items()
    )


def analyze_lamp1_cell(
    cell: CellSpec,
    artifacts: tuple[RoundArtifacts, ...],
    config: AnalysisConfig,
    output_dir: Path,
) -> MarkerResult:
    """Analyze one eligible LAMP1 cell without cohort-level inference or implicit IDs."""
    if cell not in eligible_lamp_cells(config):
        raise ValueError("LAMP1 analysis requires a complete image-verified CellSpec")
    by_round: Mapping[int, RoundArtifacts] = {
        item.acquisition_round: item for item in artifacts
    }
    if set(by_round) != {item.acquisition_round for item in cell.rounds}:
        raise ValueError(
            "cell and validated artifact acquisition-round inventories disagree"
        )
    # The target-blind gate must run before any LAMP target is opened.
    try:
        accepted = _accepted_rounds(cell, config)
    except ValueError as error:
        raise ValueError(
            "target-blind QC decisions are required for every moving LAMP1 round"
        ) from error
    if REFERENCE_ROUND not in accepted:
        raise ValueError("LAMP1 analysis requires acquisition round 2")
    dapi = {
        number: load_volume(by_round[number].registered_dapi) for number in accepted
    }
    consensus = consensus_nuclear_surface(
        dapi,
        config.voxel_spacing_xyz_um,
        config.nuclear,
        config.nominal_expansion_factor,
    )
    validity_masks, validity_digest, validity_identities = _artifact_validity_masks(
        {number: by_round[number] for number in accepted}, config
    )
    roi = _fixed_lamp_roi(consensus.mask_zyx, validity_masks)
    lamp = {
        number: load_volume(by_round[number].registered_target) for number in accepted
    }
    if len({image.shape for image in (*lamp.values(), *dapi.values())}) != 1:
        raise ValueError(
            "registered LAMP1 and DAPI volumes must share fixed-grid geometry"
        )
    resolution = theoretical_resolution(
        config.optics.target_wavelength_um,
        config.optics.numerical_aperture,
        config.optics.immersion_refractive_index,
        config.nominal_expansion_factor,
    )
    detections = {
        number: tuple(
            _with_nuclear_distances(item, consensus.nuclei)
            for item in detect_lysosomes(
                lamp[number],
                roi,
                config.voxel_spacing_xyz_um,
                config.lysosome.log_sigma_um,
                config.lysosome.detection_sbr,
                resolution=resolution,
            )
        )
        for number in accepted
    }
    ensemble_rows = tuple(
        row
        for number in sorted(detections)
        for item in detections[number]
        for row in _rows_for_detections(cell, number, item, config)
    )
    summary_rows: list[MetricRow] = []
    matches: list[tuple[str, int, int, DetectionSet, ObjectMatchResult]] = []
    for comparison, reference_round, moving_round in _comparison_specs(accepted):
        reference_members = {
            (item.sigma_um, item.detection_sbr): item
            for item in detections[reference_round]
        }
        moving_members = {
            (item.sigma_um, item.detection_sbr): item
            for item in detections[moving_round]
        }
        for key, reference_member in reference_members.items():
            match = match_lysosomes(
                reference_member,
                moving_members[key],
                resolution,
                config.lysosome.match_resolution_units,
            )
            matches.append(
                (comparison, reference_round, moving_round, reference_member, match)
            )
            for endpoint, value, units in (
                ("matched_fraction", match.matched_fraction, "fraction"),
                ("lost_fraction", match.lost_fraction, "fraction"),
                ("new_fraction", match.new_fraction, "fraction"),
                (
                    "lateral_displacement_median",
                    _median_or_nan(list(match.lateral_displacements_um)),
                    "um",
                ),
                (
                    "axial_displacement_median",
                    _median_or_nan(list(match.axial_displacements_um)),
                    "um",
                ),
                (
                    "normalized_displacement_median",
                    _median_or_nan(list(match.normalized_displacements)),
                    "resolution-units",
                ),
            ):
                summary_rows.append(
                    MetricRow(
                        cell.cohort,
                        cell.cell_id,
                        moving_round,
                        f"lysosome_{endpoint}_sigma_{key[0]:g}_sbr_{key[1]:g}",
                        value,
                        units,
                        comparison,
                        True,
                        config.config_digest,
                    )
                )
    staged_paths: tuple[Path, ...] = ()
    staging_root: Path | None = None

    def populate(staged: Path) -> None:
        nonlocal staged_paths, staging_root
        staging_root = staged
        component_diagnostic = write_nuclear_component_diagnostic(
            staged, cell.cell_id, dapi, consensus
        )
        diagnostic = staged / "lamp1-detections.npz"
        np.savez_compressed(
            diagnostic,
            **{
                f"round_{number}_sigma_{item.sigma_um:g}_sbr_{item.detection_sbr:g}_xyz_um": np.asarray(
                    item.coordinates_xyz_um
                )
                for number, items in detections.items()
                for item in items
            },
        )
        table = staged / "lamp1-ensemble.csv"
        with table.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(EnsembleRow.__dataclass_fields__)
            writer.writerows(
                [getattr(row, field) for field in EnsembleRow.__dataclass_fields__]
                for row in ensemble_rows
            )
        objects_table = staged / "objects.csv"
        with objects_table.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                (
                    "acquisition_round",
                    "log_sigma_um",
                    "detection_sbr",
                    "object_index",
                    "x_um",
                    "y_um",
                    "z_um",
                    "volume_um3",
                    "equivalent_diameter_um",
                    "sphericity",
                    "nearest_neighbor_distance_um",
                    "nucleus_relative_distance_um",
                    "nucleus_id",
                )
            )
            for number, members in sorted(detections.items()):
                for member in members:
                    for index, object_ in enumerate(member.objects):
                        writer.writerow(
                            (
                                number,
                                member.sigma_um,
                                member.detection_sbr,
                                index,
                                *object_.centroid_xyz_um,
                                object_.volume_um3,
                                object_.equivalent_diameter_um,
                                object_.sphericity,
                                object_.nearest_neighbor_distance_um,
                                object_.nucleus_relative_distance_um,
                                "aggregate",
                            )
                        )
        matches_table = staged / "matches.csv"
        with matches_table.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                (
                    "comparison_type",
                    "reference_round",
                    "moving_round",
                    "log_sigma_um",
                    "detection_sbr",
                    "reference_object_index",
                    "moving_object_index",
                    "reference_x_um",
                    "reference_y_um",
                    "reference_z_um",
                    "moving_x_um",
                    "moving_y_um",
                    "moving_z_um",
                    "lateral_displacement_um",
                    "axial_displacement_um",
                    "normalized_displacement",
                    "nucleus_id",
                )
            )
            for (
                comparison,
                reference_round,
                moving_round,
                reference_member,
                match,
            ) in matches:
                moving_member = next(
                    item
                    for item in detections[moving_round]
                    if (item.sigma_um, item.detection_sbr)
                    == (reference_member.sigma_um, reference_member.detection_sbr)
                )
                for pair_index, (reference_index, moving_index) in enumerate(
                    match.matched_pairs
                ):
                    writer.writerow(
                        (
                            comparison,
                            reference_round,
                            moving_round,
                            reference_member.sigma_um,
                            reference_member.detection_sbr,
                            reference_index,
                            moving_index,
                            *reference_member.objects[reference_index].centroid_xyz_um,
                            *moving_member.objects[moving_index].centroid_xyz_um,
                            match.lateral_displacements_um[pair_index],
                            match.axial_displacements_um[pair_index],
                            match.normalized_displacements[pair_index],
                            "aggregate",
                        )
                    )
        staged_paths = (
            component_diagnostic,
            diagnostic,
            table,
            objects_table,
            matches_table,
        )

    _atomic_publish_directory(output_dir, populate)
    assert staging_root is not None
    diagnostics = tuple(
        output_dir / item.relative_to(staging_root) for item in staged_paths
    )
    identities = (
        tuple(
            str(path)
            for number in accepted
            for artifact in (by_round[number],)
            for path in (artifact.registered_dapi, artifact.registered_target)
        )
        + validity_identities
        + (f"validity-mask-sha256:{validity_digest}",)
    )
    return MarkerResult(
        cell.cohort,
        cell.cell_id,
        tuple(summary_rows),
        ensemble_rows,
        diagnostics,
        identities,
        tuple(MappingProxyType(asdict(row)) for row in consensus.component_qc),
    )


__all__ = [
    "DetectionSet",
    "LysosomeObject",
    "ObjectMatchResult",
    "analyze_lamp1_cell",
    "detect_lysosomes",
    "eligible_lamp_cells",
    "match_lysosomes",
]
