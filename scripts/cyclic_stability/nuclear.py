"""Target-independent consensus nuclei and Lamin envelope measurements."""

from __future__ import annotations

import csv
import math
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import (
    binary_closing,
    binary_fill_holes,
    distance_transform_edt,
    gaussian_filter,
    label,
    map_coordinates,
)
from skimage.filters import threshold_otsu
from skimage.measure import marching_cubes, mesh_surface_area

from cyclic_stability.cyexm_artifacts import RoundArtifacts
from cyclic_stability.images import BackgroundSummary, load_volume, robust_background
from cyclic_stability.models import (
    AnalysisConfig,
    CellSpec,
    EnsembleRow,
    MarkerResult,
    MetricRow,
    NuclearConfig,
)
from cyclic_stability.morphology import (
    gaussian_sigma_zyx,
    resolution_surface_metrics,
    threshold_ensemble,
)
from cyclic_stability.units import theoretical_resolution

REFERENCE_ROUND = 2
_DAPI_LATERAL_RESOLUTION_UM = 0.61 * 0.405 / 1.4


@dataclass(frozen=True, slots=True)
class NuclearComponentQc:
    """Target-blind physical adjudication of one consensus DAPI component.

    Face fractions are ordered ``z-min, z-max, y-min, y-max, x-min, x-max``.
    """

    nucleus_id: str
    accepted: bool
    rejection_reasons: tuple[str, ...]
    expanded_volume_um3: float
    nominal_volume_um3: float
    sphericity: float
    centroid_xyz_um: tuple[float, float, float]
    face_contact_fractions: tuple[float, float, float, float, float, float]


@dataclass(frozen=True, slots=True)
class NuclearSurface:
    """A target-independent common nuclear surface in physical coordinates."""

    nucleus_id: str
    mask_zyx: NDArray[np.bool_]
    vertices_xyz_um: NDArray[np.float64]
    faces: NDArray[np.int_]
    normals_xyz: NDArray[np.float64]
    centroid_xyz_um: NDArray[np.float64]
    mean_curvature_um_inverse: NDArray[np.float64]
    spacing_xyz_um: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class NuclearConsensus:
    """Immutable accepted nuclei, their union, and all DAPI component QC."""

    nuclei: tuple[NuclearSurface, ...]
    mask_zyx: NDArray[np.bool_]
    labels_zyx: NDArray[np.uint16]
    component_qc: tuple[NuclearComponentQc, ...]
    spacing_xyz_um: tuple[float, float, float]
    nominal_expansion_factor: float


@dataclass(frozen=True, slots=True)
class _NuclearCandidate:
    mask_zyx: NDArray[np.bool_]
    vertices_xyz_um: NDArray[np.float64]
    faces: NDArray[np.int_]
    accepted: bool
    rejection_reasons: tuple[str, ...]
    expanded_volume_um3: float
    nominal_volume_um3: float
    sphericity: float
    centroid_xyz_um: tuple[float, float, float]
    face_contact_fractions: tuple[float, float, float, float, float, float]


@dataclass(frozen=True, slots=True)
class LaminProfileRow:
    """One surface-vertex normal profile and its round-level support summaries."""

    acquisition_round: int
    vertex_index: int
    supported: bool
    peak_offset_um: float
    displacement_um: float
    absolute_displacement_um: float
    fwhm_um: float
    integrated_ridge_intensity: float
    surface_coverage: float
    continuity: float
    continuity_change_from_reference: float


def _spacing(
    spacing_xyz_um: tuple[float, float, float],
) -> tuple[float, float, float]:
    if len(spacing_xyz_um) != 3:
        raise ValueError("spacing_xyz_um must contain exactly three values")
    result = tuple(float(value) for value in spacing_xyz_um)
    if any(not math.isfinite(value) or value <= 0.0 for value in result):
        raise ValueError("spacing_xyz_um must be finite and positive")
    return result


def _validate_config(config: NuclearConfig) -> None:
    values = (
        config.consensus_smoothing_um,
        config.normal_half_window_um,
        config.normal_step_um,
        config.ridge_min_sbr,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("nuclear configuration values must be finite and positive")
    quotient = config.normal_half_window_um / config.normal_step_um
    if not math.isclose(quotient, round(quotient), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("normal_half_window_um must be divisible by normal_step_um")


def _volume(value: NDArray[np.generic], name: str) -> NDArray[np.float64]:
    image = np.asarray(value)
    if (
        image.ndim != 3
        or not np.issubdtype(image.dtype, np.number)
        or np.issubdtype(image.dtype, np.complexfloating)
    ):
        raise ValueError(f"{name} must be one real numeric 3D ZYX volume")
    if not np.all(np.isfinite(image)):
        raise ValueError(f"{name} must contain only finite values")
    return image.astype(np.float64, copy=False)


def _normalized_dapi(image: NDArray[np.float64]) -> NDArray[np.float64]:
    background = robust_background(image)
    corrected = np.clip(image - background.median, 0.0, None)
    positive = corrected[corrected > 0.0]
    if positive.size == 0:
        raise ValueError("DAPI has no positive background-corrected support")
    scale = float(np.percentile(positive, 99.0))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("DAPI positive 99th percentile must be finite and positive")
    return corrected / scale


def _physical_ball(
    radius_um: float, spacing_xyz_um: tuple[float, float, float]
) -> NDArray[np.bool_]:
    sx, sy, sz = spacing_xyz_um
    radii = tuple(math.ceil(radius_um / step) for step in (sz, sy, sx))
    z, y, x = np.ogrid[
        -radii[0] : radii[0] + 1,
        -radii[1] : radii[1] + 1,
        -radii[2] : radii[2] + 1,
    ]
    return (z * sz) ** 2 + (y * sy) ** 2 + (x * sx) ** 2 <= radius_um**2


def _face_contact_fractions(
    mask: NDArray[np.bool_],
) -> tuple[float, float, float, float, float, float]:
    return (
        float(mask[0].mean()),
        float(mask[-1].mean()),
        float(mask[:, 0, :].mean()),
        float(mask[:, -1, :].mean()),
        float(mask[:, :, 0].mean()),
        float(mask[:, :, -1].mean()),
    )


def _surface_from_mask(
    mask: NDArray[np.bool_], spacing_xyz_um: tuple[float, float, float]
) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    sx, sy, sz = spacing_xyz_um
    padded = np.pad(mask.astype(np.float32), 1)
    try:
        vertices_zyx, faces, _, _ = marching_cubes(
            padded, level=0.5, spacing=(sz, sy, sx)
        )
    except (RuntimeError, ValueError) as error:
        raise ValueError(
            "consensus nucleus cannot form a nondegenerate surface"
        ) from error
    vertices_zyx -= np.array((sz, sy, sx))
    return vertices_zyx[:, ::-1], faces.astype(int, copy=False)


def _surface_normals_and_curvature(
    mask: NDArray[np.bool_],
    vertices_xyz_um: NDArray[np.float64],
    spacing_xyz_um: tuple[float, float, float],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Sample outward physical normals and matching signed-distance curvature.

    The signed field is positive outside the foreground and negative inside:
    ``edt(~mask) - edt(mask)``. Its normalized physical gradient is outward and
    its divergence uses exactly the same orientation. One-voxel zero padding
    provides an outside field for an allowed face-contacting nucleus, with the
    interpolation coordinates shifted by exactly that padding.
    """
    sx, sy, sz = spacing_xyz_um
    padded = np.pad(mask, 1)
    signed = distance_transform_edt(
        ~padded, sampling=(sz, sy, sx)
    ) - distance_transform_edt(padded, sampling=(sz, sy, sx))
    gradients_zyx = np.gradient(signed, sz, sy, sx)
    gradient_xyz = np.stack(
        (gradients_zyx[2], gradients_zyx[1], gradients_zyx[0]), axis=-1
    )
    norm = np.linalg.norm(gradient_xyz, axis=-1, keepdims=True)
    if not np.all(np.isfinite(norm)):
        raise ValueError("signed-distance gradient is non-finite")
    outward = np.divide(
        gradient_xyz,
        norm,
        out=np.zeros_like(gradient_xyz),
        where=norm > 0.0,
    )
    divergence = (
        np.gradient(outward[..., 0], sx, axis=2)
        + np.gradient(outward[..., 1], sy, axis=1)
        + np.gradient(outward[..., 2], sz, axis=0)
    )
    sample_zyx = np.vstack(
        (
            vertices_xyz_um[:, 2] / sz + 1.0,
            vertices_xyz_um[:, 1] / sy + 1.0,
            vertices_xyz_um[:, 0] / sx + 1.0,
        )
    )
    sampled_outward = np.vstack(
        [map_coordinates(outward[..., axis], sample_zyx, order=1) for axis in range(3)]
    ).T
    lengths = np.linalg.norm(sampled_outward, axis=1)
    if not np.all(np.isfinite(lengths)) or np.any(lengths <= 0.0):
        raise ValueError("surface normal interpolation is degenerate")
    normals = sampled_outward / lengths[:, None]
    foreground_zyx = np.argwhere(mask)
    if foreground_zyx.size == 0:
        raise ValueError("surface mask has no foreground centroid")
    centroid = foreground_zyx.mean(axis=0)[::-1] * np.array((sx, sy, sz))
    curvature = map_coordinates(divergence, sample_zyx, order=1)
    if not np.all(np.isfinite(curvature)):
        raise ValueError("surface curvature interpolation is non-finite")
    return normals, centroid, curvature


def consensus_nuclear_surface(
    dapi_rounds: Mapping[int, NDArray[np.generic]],
    spacing_xyz_um: tuple[float, float, float],
    config: NuclearConfig,
    nominal_expansion_factor: float,
) -> NuclearConsensus:
    """Construct a target-independent, component-aware median DAPI consensus.

    ``dapi_rounds`` must already contain only the acquisition rounds admitted by
    recorded target-blind QC.  Acquisition round 2 is required even when no
    moving round was accepted, preventing a consensus without the biological
    reference image.
    """
    spacing = _spacing(spacing_xyz_um)
    _validate_config(config)
    expansion = float(nominal_expansion_factor)
    if not math.isfinite(expansion) or expansion <= 0.0:
        raise ValueError("nominal_expansion_factor must be finite and positive")
    if REFERENCE_ROUND not in dapi_rounds:
        raise ValueError("consensus nuclear surface requires acquisition round 2")
    if not dapi_rounds:
        raise ValueError("consensus nuclear surface requires at least one DAPI round")
    volumes = {
        int(round_number): _volume(image, "DAPI")
        for round_number, image in dapi_rounds.items()
    }
    shapes = {volume.shape for volume in volumes.values()}
    if len(shapes) != 1:
        raise ValueError("QC-accepted DAPI rounds must share one registered geometry")
    median = np.median(
        np.stack([_normalized_dapi(volume) for volume in volumes.values()]), axis=0
    )
    smoothed = gaussian_filter(
        median, sigma=gaussian_sigma_zyx(config.consensus_smoothing_um, spacing)
    )
    if not np.any(np.isfinite(smoothed)) or float(smoothed.max()) <= float(
        smoothed.min()
    ):
        raise ValueError("consensus DAPI intensity is degenerate")
    threshold_mask = smoothed > threshold_otsu(smoothed)
    filled = binary_fill_holes(threshold_mask)
    mask = (
        binary_closing(
            filled, structure=_physical_ball(_DAPI_LATERAL_RESOLUTION_UM, spacing)
        )
        | filled
    )
    candidate_labels, candidate_count = label(
        mask, structure=np.ones((3, 3, 3), dtype=bool)
    )
    sx, sy, sz = spacing
    voxel_volume_um3 = sx * sy * sz
    minimum_volume, maximum_volume = config.nominal_volume_um3
    candidates: list[_NuclearCandidate] = []
    for component_index in range(1, candidate_count + 1):
        component_mask = candidate_labels == component_index
        expanded_volume_um3 = float(component_mask.sum()) * voxel_volume_um3
        nominal_volume_um3 = expanded_volume_um3 / expansion**3
        vertices, faces = _surface_from_mask(component_mask, spacing)
        surface_area_um2 = float(mesh_surface_area(vertices, faces))
        sphericity = float(
            math.pi ** (1.0 / 3.0)
            * (6.0 * expanded_volume_um3) ** (2.0 / 3.0)
            / surface_area_um2
        )
        centroid_zyx = np.argwhere(component_mask).mean(axis=0)
        centroid_xyz_um = (
            float(centroid_zyx[2] * sx),
            float(centroid_zyx[1] * sy),
            float(centroid_zyx[0] * sz),
        )
        face_contact_fractions = _face_contact_fractions(component_mask)
        rejection_reasons: list[str] = []
        if nominal_volume_um3 < minimum_volume:
            rejection_reasons.append("volume-below-minimum")
        if nominal_volume_um3 > maximum_volume:
            rejection_reasons.append("volume-above-maximum")
        if sphericity < config.minimum_sphericity:
            rejection_reasons.append("sphericity-below-minimum")
        if any(
            fraction > config.maximum_face_contact_fraction
            for fraction in face_contact_fractions
        ):
            rejection_reasons.append("face-contact-above-maximum")
        candidates.append(
            _NuclearCandidate(
                component_mask,
                vertices,
                faces,
                not rejection_reasons,
                tuple(rejection_reasons),
                expanded_volume_um3,
                nominal_volume_um3,
                sphericity,
                centroid_xyz_um,
                face_contact_fractions,
            )
        )

    candidates.sort(key=lambda item: item.centroid_xyz_um)
    accepted_count = sum(item.accepted for item in candidates)
    minimum_count, maximum_count = config.accepted_component_count
    if not minimum_count <= accepted_count <= maximum_count:
        rejected_reasons = sorted(
            {
                reason
                for candidate in candidates
                for reason in candidate.rejection_reasons
            }
        )
        reason_suffix = (
            f"; rejected component reasons: {', '.join(rejected_reasons)}"
            if rejected_reasons
            else ""
        )
        raise ValueError(
            "accepted nuclear component count "
            f"{accepted_count} is outside configured range "
            f"[{minimum_count}, {maximum_count}]{reason_suffix}"
        )

    accepted_labels = np.zeros(mask.shape, dtype=np.uint16)
    nuclei: list[NuclearSurface] = []
    component_qc: list[NuclearComponentQc] = []
    accepted_index = 0
    for candidate_index, candidate in enumerate(candidates, start=1):
        if candidate.accepted:
            accepted_index += 1
            nucleus_id = f"N{accepted_index:03d}"
            normals, centroid, curvature = _surface_normals_and_curvature(
                candidate.mask_zyx, candidate.vertices_xyz_um, spacing
            )
            for array in (
                candidate.mask_zyx,
                candidate.vertices_xyz_um,
                candidate.faces,
                normals,
                centroid,
                curvature,
            ):
                array.setflags(write=False)
            accepted_labels[candidate.mask_zyx] = accepted_index
            nuclei.append(
                NuclearSurface(
                    nucleus_id,
                    candidate.mask_zyx,
                    candidate.vertices_xyz_um,
                    candidate.faces,
                    normals,
                    centroid,
                    curvature,
                    spacing,
                )
            )
        else:
            nucleus_id = f"C{candidate_index:03d}"
        component_qc.append(
            NuclearComponentQc(
                nucleus_id,
                candidate.accepted,
                candidate.rejection_reasons,
                candidate.expanded_volume_um3,
                candidate.nominal_volume_um3,
                candidate.sphericity,
                candidate.centroid_xyz_um,
                candidate.face_contact_fractions,
            )
        )
    union_mask = accepted_labels > 0
    union_mask.setflags(write=False)
    accepted_labels.setflags(write=False)
    return NuclearConsensus(
        tuple(nuclei),
        union_mask,
        accepted_labels,
        tuple(component_qc),
        spacing,
        expansion,
    )


def _profile_offsets(config: NuclearConfig) -> NDArray[np.float64]:
    count = round(config.normal_half_window_um / config.normal_step_um)
    return np.arange(-count, count + 1, dtype=np.float64) * config.normal_step_um


def _sample_profiles(
    surface: NuclearSurface,
    image: NDArray[np.float64],
    offsets_um: NDArray[np.float64],
) -> NDArray[np.float64]:
    sx, sy, sz = surface.spacing_xyz_um
    xyz = (
        surface.vertices_xyz_um[:, None, :]
        + surface.normals_xyz[:, None, :] * offsets_um[None, :, None]
    )
    coordinates = np.stack(
        (xyz[..., 2] / sz, xyz[..., 1] / sy, xyz[..., 0] / sx), axis=0
    ).reshape(3, -1)
    sampled = map_coordinates(image, coordinates, order=1, mode="constant", cval=np.nan)
    return sampled.reshape(xyz.shape[:2])


def _crossing(
    positions: NDArray[np.float64],
    values: NDArray[np.float64],
    level: float,
    start: int,
    direction: int,
) -> float | None:
    index = start
    while 0 <= index + direction < values.size:
        next_index = index + direction
        first, second = values[index], values[next_index]
        if first >= level >= second or first <= level <= second:
            if first == second:
                return float(positions[index])
            fraction = float((level - first) / (second - first))
            return float(
                positions[index] + fraction * (positions[next_index] - positions[index])
            )
        index = next_index
    return None


def _ridge(
    profile: NDArray[np.float64],
    offsets_um: NDArray[np.float64],
    background: BackgroundSummary,
    config: NuclearConfig,
) -> tuple[bool, float, float, float]:
    if not np.all(np.isfinite(profile)):
        return False, math.nan, math.nan, math.nan
    corrected = profile - background.median
    peak_index = int(np.argmax(corrected))
    peak = float(corrected[peak_index])
    if (
        peak_index in (0, corrected.size - 1)
        or peak <= config.ridge_min_sbr * background.scale
    ):
        return False, math.nan, math.nan, math.nan
    level = peak / 2.0
    left = _crossing(offsets_um, corrected, level, peak_index, -1)
    right = _crossing(offsets_um, corrected, level, peak_index, 1)
    if left is None or right is None or right <= left:
        return False, math.nan, math.nan, math.nan
    return (
        True,
        float(offsets_um[peak_index]),
        float(right - left),
        float(np.trapezoid(np.clip(corrected, 0.0, None), offsets_um)),
    )


def _continuity(
    faces: NDArray[np.int_], supported: NDArray[np.bool_]
) -> tuple[float, float]:
    coverage = float(supported.mean())
    if not np.any(supported):
        return coverage, 0.0
    parent = np.arange(supported.size)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def join(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for triangle in faces:
        for first, second in (
            (int(triangle[0]), int(triangle[1])),
            (int(triangle[1]), int(triangle[2])),
            (int(triangle[2]), int(triangle[0])),
        ):
            if supported[first] and supported[second]:
                join(first, second)
    roots = np.array([find(int(index)) for index in np.flatnonzero(supported)])
    _, counts = np.unique(roots, return_counts=True)
    return coverage, float(counts.max() / supported.size)


def lamin_normal_profiles(
    surface: NuclearSurface,
    lamin_rounds: Mapping[int, NDArray[np.generic]],
    config: NuclearConfig,
    background_by_round: Mapping[int, BackgroundSummary],
) -> tuple[LaminProfileRow, ...]:
    """Measure signed Lamin ridge locations relative to acquisition round 2.

    The normal direction is physical XYZ, whereas volumes are sampled in ZYX.
    A missing or endpoint ridge remains unsupported with NaN geometric values;
    it is never assigned a zero displacement.
    """
    _spacing(surface.spacing_xyz_um)
    _validate_config(config)
    if REFERENCE_ROUND not in lamin_rounds:
        raise ValueError("Lamin profiles require acquisition round 2")
    if set(lamin_rounds) != set(background_by_round):
        raise ValueError(
            "Lamin volumes and background summaries must have matching rounds"
        )
    offsets = _profile_offsets(config)
    per_round: dict[int, list[tuple[bool, float, float, float]]] = {}
    for round_number, raw in lamin_rounds.items():
        image = _volume(raw, "Lamin")
        if image.shape != surface.mask_zyx.shape:
            raise ValueError("Lamin volume and consensus surface geometry must match")
        background = background_by_round[round_number]
        if (
            not math.isfinite(background.median)
            or not math.isfinite(background.scale)
            or background.scale <= 0.0
        ):
            raise ValueError(
                "Lamin background must be finite with positive robust scale"
            )
        per_round[int(round_number)] = [
            _ridge(profile, offsets, background, config)
            for profile in _sample_profiles(surface, image, offsets)
        ]
    reference = per_round[REFERENCE_ROUND]
    summaries: dict[int, tuple[float, float]] = {}
    for round_number, profiles in per_round.items():
        summaries[round_number] = _continuity(
            surface.faces, np.array([profile[0] for profile in profiles], dtype=bool)
        )
    reference_continuity = summaries[REFERENCE_ROUND][1]
    rows: list[LaminProfileRow] = []
    for round_number in sorted(per_round):
        coverage, continuity = summaries[round_number]
        for index, (supported, peak, fwhm, intensity) in enumerate(
            per_round[round_number]
        ):
            reference_supported, reference_peak, _, _ = reference[index]
            displacement = (
                0.0
                if round_number == REFERENCE_ROUND and supported
                else peak - reference_peak
                if supported and reference_supported
                else math.nan
            )
            rows.append(
                LaminProfileRow(
                    round_number,
                    index,
                    supported,
                    peak,
                    displacement,
                    abs(displacement) if math.isfinite(displacement) else math.nan,
                    fwhm,
                    intensity,
                    coverage,
                    continuity,
                    continuity - reference_continuity,
                )
            )
    return tuple(rows)


def _accepted_rounds(cell: CellSpec, config: AnalysisConfig) -> tuple[int, ...]:
    decisions = {
        decision.acquisition_round: decision.decision
        for decision in config.qc_decisions
        if decision.cohort == cell.cohort and decision.cell_id == cell.cell_id
    }
    accepted = [REFERENCE_ROUND]
    for round_spec in cell.rounds:
        round_number = round_spec.acquisition_round
        if round_number == REFERENCE_ROUND:
            continue
        decision = decisions.get(round_number)
        if decision is None:
            raise ValueError(
                f"target-specific Lamin analysis requires QC decision for round {round_number}"
            )
        if decision == "accept":
            accepted.append(round_number)
    return tuple(sorted(accepted))


def _metric_rows(
    cell: CellSpec,
    profiles: tuple[LaminProfileRow, ...],
    config: AnalysisConfig,
) -> tuple[MetricRow, ...]:
    rows: list[MetricRow] = []
    for round_number in sorted({row.acquisition_round for row in profiles}):
        subset = [row for row in profiles if row.acquisition_round == round_number]
        current_supported = [row for row in subset if row.supported]
        paired_supported = [
            row
            for row in subset
            if row.supported and math.isfinite(row.displacement_um)
        ]
        values = {
            "lamin_surface_coverage": (subset[0].surface_coverage, "fraction"),
            "lamin_continuity": (subset[0].continuity, "fraction"),
            "lamin_continuity_change_from_round_2": (
                subset[0].continuity_change_from_reference,
                "fraction",
            ),
        }
        if paired_supported:
            values.update(
                {
                    "lamin_signed_displacement_median": (
                        float(
                            np.median([row.displacement_um for row in paired_supported])
                        ),
                        "um",
                    ),
                    "lamin_absolute_displacement_median": (
                        float(
                            np.median(
                                [
                                    row.absolute_displacement_um
                                    for row in paired_supported
                                ]
                            )
                        ),
                        "um",
                    ),
                }
            )
        if current_supported:
            values.update(
                {
                    "lamin_fwhm_median": (
                        float(np.median([row.fwhm_um for row in current_supported])),
                        "um",
                    ),
                    "lamin_integrated_ridge_intensity_median": (
                        float(
                            np.median(
                                [
                                    row.integrated_ridge_intensity
                                    for row in current_supported
                                ]
                            )
                        ),
                        "a.u. um",
                    ),
                }
            )
        for endpoint, (value, units) in values.items():
            rows.append(
                MetricRow(
                    cell.cohort,
                    cell.cell_id,
                    round_number,
                    endpoint,
                    value,
                    units,
                    "round-1-exposure-sensitivity"
                    if round_number == 1
                    else "matched-settings",
                    True,
                    config.config_digest,
                )
            )
    return tuple(rows)


def _write_profile_diagnostics(
    output_dir: Path, surface: NuclearSurface, profiles: tuple[LaminProfileRow, ...]
) -> tuple[Path, ...]:
    """Write portable surface coordinates plus XY/XZ/YZ profile projections."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    surface_path = output_dir / "consensus-nuclear-surface.npz"
    np.savez_compressed(
        surface_path,
        vertices_xyz_um=surface.vertices_xyz_um,
        normals_xyz=surface.normals_xyz,
        faces=surface.faces,
    )
    profile_path = output_dir / "lamin-normal-profiles.csv"
    with profile_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(LaminProfileRow.__dataclass_fields__)
        writer.writerows(
            [getattr(row, name) for name in LaminProfileRow.__dataclass_fields__]
            for row in profiles
        )
    latest_round = max(row.acquisition_round for row in profiles)
    displacements = np.full(surface.vertices_xyz_um.shape[0], np.nan)
    for row in profiles:
        if row.acquisition_round == latest_round:
            displacements[row.vertex_index] = row.displacement_um
    projection_paths: list[Path] = []
    for plane, axes in (("xy", (0, 1)), ("xz", (0, 2)), ("yz", (1, 2))):
        figure, axis = plt.subplots(figsize=(5, 5), layout="constrained")
        plotted = np.isfinite(displacements)
        axis.scatter(
            surface.vertices_xyz_um[~plotted, axes[0]],
            surface.vertices_xyz_um[~plotted, axes[1]],
            color="lightgray",
            s=1.0,
            rasterized=True,
        )
        scatter = axis.scatter(
            surface.vertices_xyz_um[plotted, axes[0]],
            surface.vertices_xyz_um[plotted, axes[1]],
            c=displacements[plotted],
            cmap="coolwarm",
            s=1.0,
            rasterized=True,
        )
        figure.colorbar(scatter, ax=axis, label="Lamin displacement (um)")
        axis.set(
            xlabel=f"{plane[0].upper()} (um)",
            ylabel=f"{plane[1].upper()} (um)",
            aspect="equal",
        )
        path = output_dir / f"lamin-surface-profile-{plane}.png"
        figure.savefig(path, dpi=200)
        plt.close(figure)
        projection_paths.append(path)
    return (surface_path, profile_path, *projection_paths)


def _atomic_publish_directory(
    destination: Path, populate: Callable[[Path], None]
) -> None:
    """Atomically replace a complete Lamin output tree on its destination filesystem."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent)
    )
    backup = destination.with_name(f".{destination.name}.previous")
    if backup.exists():
        shutil.rmtree(backup)
    try:
        populate(staged)
        if destination.exists():
            os.replace(destination, backup)
        os.replace(staged, destination)
    except BaseException:
        shutil.rmtree(staged, ignore_errors=True)
        if backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    else:
        shutil.rmtree(backup, ignore_errors=True)


def _geometry(
    mask: NDArray[np.bool_],
    spacing_xyz_um: tuple[float, float, float],
) -> dict[str, float]:
    """Return physical mask geometry without treating an empty mask as zero shape."""
    volume = float(mask.sum() * np.prod(spacing_xyz_um))
    if volume == 0.0:
        return {
            "volume_um3": 0.0,
            "area_to_volume_um_inverse": math.nan,
            "sphericity": math.nan,
            "mean_curvature_p25_um_inverse": math.nan,
            "mean_curvature_p50_um_inverse": math.nan,
            "mean_curvature_p75_um_inverse": math.nan,
        }
    vertices, faces = _surface_from_mask(mask, spacing_xyz_um)
    _, _, curvature = _surface_normals_and_curvature(mask, vertices, spacing_xyz_um)
    area = float(mesh_surface_area(vertices, faces))
    return {
        "volume_um3": volume,
        "area_to_volume_um_inverse": area / volume,
        "sphericity": float(
            np.pi ** (1.0 / 3.0) * (6.0 * volume) ** (2.0 / 3.0) / area
        ),
        "mean_curvature_p25_um_inverse": float(np.percentile(curvature, 25.0)),
        "mean_curvature_p50_um_inverse": float(np.percentile(curvature, 50.0)),
        "mean_curvature_p75_um_inverse": float(np.percentile(curvature, 75.0)),
    }


def _ensemble_row(
    cell: CellSpec,
    round_number: int,
    endpoint: str,
    name: str,
    ensemble_value: float,
    value: float | None,
    units: str,
    config: AnalysisConfig,
) -> EnsembleRow:
    """Keep one declared ensemble coordinate beside every derived endpoint."""
    return EnsembleRow(
        cell.cohort,
        cell.cell_id,
        round_number,
        endpoint,
        name,
        ensemble_value,
        math.nan if value is None else value,
        units,
        config.config_digest,
    )


def analyze_lamin_cell(
    cell: CellSpec,
    artifacts: tuple[RoundArtifacts, ...],
    config: AnalysisConfig,
    output_dir: Path,
) -> MarkerResult:
    """Analyze one QC-eligible Lamin cell without any group-level inference."""
    if cell.cohort != "lamin":
        raise ValueError("Lamin analysis requires a lamin CellSpec")
    by_round = {artifact.acquisition_round: artifact for artifact in artifacts}
    if set(by_round) != {round_spec.acquisition_round for round_spec in cell.rounds}:
        raise ValueError(
            "cell and validated artifact acquisition-round inventories disagree"
        )
    accepted = _accepted_rounds(cell, config)
    dapi = {
        round_number: load_volume(by_round[round_number].registered_dapi)
        for round_number in accepted
    }
    lamin = {
        round_number: load_volume(by_round[round_number].registered_target)
        for round_number in accepted
    }
    surface = consensus_nuclear_surface(
        dapi, config.voxel_spacing_xyz_um, config.nuclear
    )
    backgrounds = {
        round_number: robust_background(image) for round_number, image in lamin.items()
    }
    profiles = lamin_normal_profiles(surface, lamin, config.nuclear, backgrounds)
    resolution = theoretical_resolution(
        config.optics.target_wavelength_um,
        config.optics.numerical_aperture,
        config.optics.immersion_refractive_index,
        config.nominal_expansion_factor,
    )
    reference_members = threshold_ensemble(
        lamin[REFERENCE_ROUND],
        surface.mask_zyx,
        config.ensembles.gaussian_sigma_um,
        config.ensembles.threshold_sbr,
        config.voxel_spacing_xyz_um,
    )
    ensemble_rows: list[EnsembleRow] = []
    for round_number in accepted:
        members = threshold_ensemble(
            lamin[round_number],
            surface.mask_zyx,
            config.ensembles.gaussian_sigma_um,
            config.ensembles.threshold_sbr,
            config.voxel_spacing_xyz_um,
        )
        for reference_member, member in zip(reference_members, members, strict=True):
            geometry = _geometry(member.mask_zyx, config.voxel_spacing_xyz_um)
            surface_metrics = resolution_surface_metrics(
                reference_member.mask_zyx,
                member.mask_zyx,
                config.voxel_spacing_xyz_um,
                resolution,
                config.ensembles.surface_tolerance_resolution_units,
            )
            name = f"sigma={member.sigma_um:g};sbr={member.threshold_sbr:g}"
            ensemble_value = member.threshold_sbr
            ensemble_rows.extend(
                _ensemble_row(
                    cell,
                    round_number,
                    f"lamin_mask_{endpoint}",
                    name,
                    ensemble_value,
                    value,
                    units,
                    config,
                )
                for endpoint, value, units in (
                    ("volume", geometry["volume_um3"], "um3"),
                    ("area_to_volume", geometry["area_to_volume_um_inverse"], "1/um"),
                    ("sphericity", geometry["sphericity"], "fraction"),
                    (
                        "mean_curvature_p25",
                        geometry["mean_curvature_p25_um_inverse"],
                        "1/um",
                    ),
                    (
                        "mean_curvature_p50",
                        geometry["mean_curvature_p50_um_inverse"],
                        "1/um",
                    ),
                    (
                        "mean_curvature_p75",
                        geometry["mean_curvature_p75_um_inverse"],
                        "1/um",
                    ),
                )
            )
            ensemble_rows.extend(
                (
                    _ensemble_row(
                        cell,
                        round_number,
                        "lamin_jaccard",
                        name,
                        ensemble_value,
                        surface_metrics.jaccard,
                        "fraction",
                        config,
                    ),
                    _ensemble_row(
                        cell,
                        round_number,
                        "lamin_fixed_to_moving_boundary_distance_p95",
                        name,
                        ensemble_value,
                        surface_metrics.fixed_to_moving_normalized_distance_p95,
                        "resolution-units",
                        config,
                    ),
                    _ensemble_row(
                        cell,
                        round_number,
                        "lamin_moving_to_fixed_boundary_distance_p95",
                        name,
                        ensemble_value,
                        surface_metrics.moving_to_fixed_normalized_distance_p95,
                        "resolution-units",
                        config,
                    ),
                )
            )
            for tolerance, dice in surface_metrics.surface_dice.items():
                ensemble_rows.append(
                    _ensemble_row(
                        cell,
                        round_number,
                        "lamin_surface_dice",
                        name,
                        tolerance,
                        dice,
                        "fraction",
                        config,
                    )
                )
    staged_diagnostics: tuple[Path, ...] = ()
    staging_root: Path | None = None

    def populate_diagnostics(staged: Path) -> None:
        nonlocal staged_diagnostics, staging_root
        staging_root = staged
        staged_diagnostics = _write_profile_diagnostics(staged, surface, profiles)

    _atomic_publish_directory(output_dir, populate_diagnostics)
    assert staging_root is not None
    diagnostics = tuple(
        output_dir / path.relative_to(staging_root) for path in staged_diagnostics
    )
    identities = tuple(
        str(path)
        for artifact in artifacts
        for path in (artifact.registered_dapi, artifact.registered_target)
    )
    nuclear_geometry = _geometry(surface.mask_zyx, config.voxel_spacing_xyz_um)
    return MarkerResult(
        cell.cohort,
        cell.cell_id,
        _metric_rows(cell, profiles, config)
        + tuple(
            MetricRow(
                cell.cohort,
                cell.cell_id,
                REFERENCE_ROUND,
                f"consensus_nuclear_{endpoint}",
                value,
                units,
                "QC-accepted-DAPI-rounds",
                True,
                config.config_digest,
            )
            for endpoint, value, units in (
                ("volume", nuclear_geometry["volume_um3"], "um3"),
                (
                    "area_to_volume",
                    nuclear_geometry["area_to_volume_um_inverse"],
                    "1/um",
                ),
                ("sphericity", nuclear_geometry["sphericity"], "fraction"),
                (
                    "mean_curvature_p25",
                    nuclear_geometry["mean_curvature_p25_um_inverse"],
                    "1/um",
                ),
                (
                    "mean_curvature_p50",
                    nuclear_geometry["mean_curvature_p50_um_inverse"],
                    "1/um",
                ),
                (
                    "mean_curvature_p75",
                    nuclear_geometry["mean_curvature_p75_um_inverse"],
                    "1/um",
                ),
            )
        ),
        tuple(ensemble_rows),
        diagnostics,
        identities,
    )


__all__ = [
    "LaminProfileRow",
    "NuclearComponentQc",
    "NuclearConsensus",
    "NuclearSurface",
    "analyze_lamin_cell",
    "consensus_nuclear_surface",
    "lamin_normal_profiles",
]
