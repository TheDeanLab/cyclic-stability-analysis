"""Per-cell Golgi continuous-image and threshold-ensemble stability measures."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from cyexm.registration._request_policy import channel_transform_parameters
from cyexm.registration.artifacts import FileIdentity
from cyexm.registration.backend import AntsRegistrationBackend
from cyexm.runtime import NativeThreadConfig
from numpy.typing import NDArray
from scipy.ndimage import binary_erosion, distance_transform_edt, gaussian_filter, label
from scipy.spatial import cKDTree
from skimage.measure import mesh_surface_area

from cyclic_stability.cyexm_artifacts import RoundArtifacts
from cyclic_stability.images import BackgroundSummary, load_volume, robust_background
from cyclic_stability.models import (
    AnalysisConfig,
    CellSpec,
    EnsembleRow,
    MarkerResult,
    MetricRow,
)
from cyclic_stability.morphology import (
    MaskMember,
    gaussian_sigma_zyx,
    resolution_surface_metrics,
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


ENSEMBLE_ENDPOINTS = (
    "absolute_background_corrected_signal",
    "volume_um3",
    "component_count",
    "component_volume_gini",
    "fragmentation_per_um3",
    "compactness",
    "area_to_volume_um_inverse",
    "nuclear_distance_p25_um",
    "nuclear_distance_p50_um",
    "nuclear_distance_p75_um",
    "jaccard",
    "fixed_to_moving_boundary_distance_p95",
    "moving_to_fixed_boundary_distance_p95",
)


def _ensemble_units(config: AnalysisConfig) -> dict[str, str]:
    """Return the fixed units schema for every non-image ensemble endpoint."""
    return {
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
    validity_mask: NDArray[np.bool_],
    spacing_xyz_um: tuple[float, float, float],
    sigma_um: float,
) -> NDArray[np.float64]:
    """Background-correct, physically smooth, and differentiate one target volume."""
    background = _support_background(image, validity_mask)
    # Gradients are invariant to the robust offset; retaining negative corrected
    # values avoids fabricating a zero-gradient plateau at the support boundary.
    corrected = image - background.median
    smoothed = _masked_gaussian(
        corrected, validity_mask, gaussian_sigma_zyx(sigma_um, spacing_xyz_um)
    )
    sx, sy, sz = spacing_xyz_um
    gradients = np.gradient(smoothed, sz, sy, sx)
    return np.sqrt(sum(gradient * gradient for gradient in gradients))


def _support_background(
    image: NDArray[np.float64], validity_mask: NDArray[np.bool_]
) -> BackgroundSummary:
    """Estimate robust background solely from the common transform-valid support."""
    return robust_background(np.where(validity_mask, image, 0.0))


def _masked_gaussian(
    image: NDArray[np.float64],
    validity_mask: NDArray[np.bool_],
    sigma_zyx: tuple[float, float, float],
) -> NDArray[np.float64]:
    """Smooth valid samples with normalized convolution, never padding intensities."""
    support = validity_mask.astype(np.float64, copy=False)
    denominator = gaussian_filter(support, sigma_zyx)
    numerator = gaussian_filter(image * support, sigma_zyx)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > np.finfo(np.float64).eps,
    )


def _effective_gradient_roi(
    roi: NDArray[np.bool_], validity_mask: NDArray[np.bool_]
) -> NDArray[np.bool_]:
    """Exclude all samples whose three-dimensional gradient stencil touches padding.

    A one-voxel 26-neighbour erosion makes the central-difference policy used by
    :func:`numpy.gradient` explicit.  Thin support fails rather than acquiring a
    favourable derivative from a padded neighbour.
    """
    effective = roi & binary_erosion(
        validity_mask, structure=np.ones((3, 3, 3), dtype=bool), border_value=0
    )
    if not np.any(effective):
        raise ValueError(
            "gradient similarity ROI is empty after support-boundary exclusion"
        )
    return effective


def multiscale_gradient_similarity(
    reference: NDArray[np.generic],
    moving: NDArray[np.generic],
    roi: NDArray[np.bool_],
    spacing_xyz_um: tuple[float, float, float],
    sigma_values_um: tuple[float, ...],
    *,
    validity_mask: NDArray[np.bool_] | None = None,
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
    valid = (
        np.ones(fixed.shape, dtype=bool)
        if validity_mask is None
        else validated_binary_mask(validity_mask, "validity mask")
    )
    if valid.shape != fixed.shape:
        raise ValueError("validity mask must match reference geometry")
    if np.any(support & ~valid):
        raise ValueError("ROI must be contained in the validity mask")
    effective_roi = _effective_gradient_roi(support, valid)
    rows: list[GradientSimilarityRow] = []
    for raw_sigma in sigma_values_um:
        sigma_um = float(raw_sigma)
        fixed_gradient = _gradient_magnitude(fixed, valid, spacing_xyz_um, sigma_um)[
            effective_roi
        ]
        moving_gradient = _gradient_magnitude(moved, valid, spacing_xyz_um, sigma_um)[
            effective_roi
        ]
        if np.ptp(fixed_gradient) == 0.0 or np.ptp(moving_gradient) == 0.0:
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
    """Return physical XYZ distances to the nearest sampled nuclear mesh vertex.

    Target voxels are represented by their voxel centers.  This is a reproducible
    nearest-vertex approximation to a triangulated surface distance, not an exact
    point-to-triangle distance; spacing anisotropy is nevertheless preserved.
    """
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
    validity_masks: Mapping[int, NDArray[np.bool_]],
    config: AnalysisConfig,
) -> NDArray[np.bool_]:
    """Build one fixed exterior ROI from explicit transform-valid support masks.

    A registered target can contain finite padding outside its transformed source
    support.  Therefore validity is an input derived from the transform chain, not
    from target intensities.
    """
    nucleus = validated_binary_mask(nuclear_mask, "consensus nuclear mask")
    if nucleus.ndim != 3:
        raise ValueError("consensus nuclear mask must be 3D ZYX")
    if not validity_masks:
        raise ValueError("Golgi ROI requires at least one registered round")
    valid = np.ones(nucleus.shape, dtype=bool)
    for round_number, mask in validity_masks.items():
        transform_valid = validated_binary_mask(
            mask, f"validity mask for round {round_number}"
        )
        if transform_valid.ndim != 3 or transform_valid.shape != nucleus.shape:
            raise ValueError(
                "transform validity masks and consensus nuclear mask must share geometry"
            )
        valid &= transform_valid
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


def _common_validity_masks(
    rounds: Mapping[int, NDArray[np.float64]],
    validity_masks: Mapping[int, NDArray[np.bool_]],
) -> dict[int, NDArray[np.bool_]]:
    """Validate explicit fixed-grid transform support for every registered round."""
    if set(validity_masks) != set(rounds):
        raise ValueError("transform validity masks must match registered Golgi rounds")
    result: dict[int, NDArray[np.bool_]] = {}
    for round_number, raw_mask in validity_masks.items():
        mask = validated_binary_mask(
            raw_mask, f"validity mask for round {round_number}"
        )
        if mask.ndim != 3 or mask.shape != rounds[round_number].shape:
            raise ValueError(
                "transform validity mask geometry must match its Golgi round"
            )
        result[round_number] = mask
    return result


def _support_threshold_ensemble(
    image: NDArray[np.float64],
    roi: NDArray[np.bool_],
    validity_mask: NDArray[np.bool_],
    config: AnalysisConfig,
) -> tuple[MaskMember, ...]:
    """Build the frozen Golgi mask grid using only common-valid source support."""
    background = _support_background(image, validity_mask)
    members: list[MaskMember] = []
    for sigma_um in config.ensembles.gaussian_sigma_um:
        smoothed = _masked_gaussian(
            image,
            validity_mask,
            gaussian_sigma_zyx(sigma_um, config.voxel_spacing_xyz_um),
        )
        for threshold_sbr in config.ensembles.threshold_sbr:
            mask = (
                smoothed > background.median + threshold_sbr * background.scale
            ) & roi
            mask.setflags(write=False)
            members.append(
                MaskMember(
                    mask,
                    sigma_um,
                    threshold_sbr,
                    background,
                    config.voxel_spacing_xyz_um,
                    "nonempty" if np.any(mask) else "empty",
                )
            )
    return tuple(members)


def _artifact_validity_masks(
    artifacts: Mapping[int, RoundArtifacts],
    config: AnalysisConfig,
    *,
    _backend: AntsRegistrationBackend | None = None,
) -> tuple[dict[int, NDArray[np.bool_]], str, tuple[str, ...]]:
    """Recreate support from the registered-target producer record alone.

    The target-transform completion record is authoritative for native source,
    fixed grid, recorded XYZ pixel size, and affine-before-warp chain.  Analysis
    spacing is deliberately not substituted for the producer geometry.
    """
    reference_artifact = artifacts[REFERENCE_ROUND]
    backend = _backend or AntsRegistrationBackend(
        NativeThreadConfig(itk_threads=1, blas_threads=1)
    )
    fixed_shape = tuple(load_volume(reference_artifact.registered_target).shape)
    reference_mask = np.ones(fixed_shape, dtype=bool)
    masks: dict[int, NDArray[np.bool_]] = {REFERENCE_ROUND: reference_mask}
    provenance: dict[str, object] = {
        "reference": {
            "registered_target": FileIdentity.from_path(
                reference_artifact.registered_target
            ).to_dict(),
            "semantics": "untransformed-fixed-reference-grid-all-valid-v1",
            "mask": _validity_mask_provenance(reference_mask),
        },
        "moving": [],
    }
    for round_number, artifact in artifacts.items():
        if round_number == REFERENCE_ROUND:
            continue
        if (
            artifact.native_target is None
            or artifact.affine is None
            or artifact.warp is None
        ):
            raise ValueError(
                "validated moving artifact lacks exact native target or transform chain"
            )
        target_record = next(
            (
                record
                for record in artifact.completion_records
                if record.artifact.path == str(artifact.registered_target.resolve())
            ),
            None,
        )
        if target_record is None or target_record.artifact != FileIdentity.from_path(
            artifact.registered_target
        ):
            raise ValueError("registered target lacks matching completion provenance")
        request = target_record.request
        if (
            request.stage != "channel-transform-v1"
            or request.method != "warp-affine"
            or request.accuracy != "not-applicable"
            or len(request.inputs) != 4
            or request.parameters.get("interpolation") != "nearestNeighbor"
        ):
            raise ValueError(
                "registered target is not an exact nearest-neighbor channel transform"
            )
        try:
            spacing = tuple(
                float(value) for value in request.parameters["pixel_size_xyz_um"]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "registered target lacks recorded producer pixel size"
            ) from error
        if len(spacing) != 3 or any(
            not math.isfinite(value) or value <= 0.0 for value in spacing
        ):
            raise ValueError(
                "registered target has invalid recorded producer pixel size"
            )
        if request.parameters != channel_transform_parameters(
            pixel_size_xyz=spacing, interpolation="nearestNeighbor"
        ):
            raise ValueError(
                "registered target transform parameters do not match producer policy"
            )
        fixed_identity, source_identity, affine_identity, warp_identity = request.inputs
        for identity in request.inputs:
            try:
                current = FileIdentity.from_path(Path(identity.path))
            except (OSError, RuntimeError, ValueError) as error:
                raise ValueError(
                    "registered target input identity is unavailable"
                ) from error
            if current != identity:
                raise ValueError(
                    "registered target input identity changed after completion"
                )
        source = artifact.native_target
        if (
            source_identity != FileIdentity.from_path(source)
            or fixed_identity
            != FileIdentity.from_path(reference_artifact.registered_dapi)
            or affine_identity != FileIdentity.from_path(artifact.affine)
            or warp_identity != FileIdentity.from_path(artifact.warp)
            or request.upstream != warp_identity
        ):
            raise ValueError(
                "registered target inputs do not match the selected artifact chain"
            )
        fixed = backend.load_image(Path(fixed_identity.path), spacing)
        moving = backend.load_image(source, spacing)
        if backend.geometry(fixed) != request.geometry:
            raise ValueError(
                "registered target fixed geometry differs from its completion request"
            )
        ones = backend.from_array(
            np.ones(
                tuple(np.asarray(backend.to_array(moving)).shape), dtype=np.float32
            ),
            reference=moving,
        )
        transformed = backend.apply_transform_chain(
            fixed=fixed,
            moving=ones,
            affine_path=artifact.affine,
            warp_path=artifact.warp,
            interpolation="nearestNeighbor",
        )
        mask = np.asarray(backend.to_array(transformed)) >= 0.5
        if mask.shape != fixed_shape:
            raise ValueError(
                "transformed validity mask does not match reference geometry"
            )
        masks[round_number] = mask
        moving_provenance = provenance["moving"]
        assert isinstance(moving_provenance, list)
        moving_provenance.append(
            {
                "acquisition_round": round_number,
                "registered_target": target_record.artifact.to_dict(),
                "request": request.to_dict(),
                "fixed": fixed_identity.to_dict(),
                "source": source_identity.to_dict(),
                "affine": affine_identity.to_dict(),
                "warp": warp_identity.to_dict(),
                "semantics": "native-all-ones-affine-before-warp-nearest-neighbor-v1",
                "mask": {
                    **_validity_mask_provenance(mask),
                },
            }
        )
    serialized = json.dumps(provenance, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(serialized.encode()).hexdigest()
    return (
        masks,
        digest,
        (
            f"validity-provenance:{serialized}",
            f"validity-provenance-sha256:{digest}",
        ),
    )


def _validity_mask_provenance(mask: NDArray[np.bool_]) -> dict[str, object]:
    """Return canonical bytes-and-layout evidence for one returned validity mask."""
    contiguous = np.ascontiguousarray(mask)
    return {
        "shape_zyx": list(contiguous.shape),
        "dtype": contiguous.dtype.str,
        "order": "C",
        "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
    }


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
    *,
    validity_masks: Mapping[int, NDArray[np.bool_]],
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
    if len({image.shape for image in rounds.values()}) != 1:
        raise ValueError("registered Golgi rounds must share fixed-grid geometry")
    common_validity = _common_validity_masks(rounds, validity_masks)
    common_support = np.logical_and.reduce(tuple(common_validity.values()))
    surface = consensus_nuclear_surface(
        dapi_rounds, config.voxel_spacing_xyz_um, config.nuclear
    )
    roi = _perinuclear_roi(surface.mask_zyx, common_validity, config)
    reference = rounds[REFERENCE_ROUND]
    ensemble_by_round = {
        round_number: _support_threshold_ensemble(
            image,
            roi,
            common_support,
            config,
        )
        for round_number, image in rounds.items()
    }
    reference_members = ensemble_by_round[REFERENCE_ROUND]
    ensemble_rows: list[EnsembleRow] = []
    summary_rows: list[MetricRow] = []
    units = _ensemble_units(config)
    endpoints = (
        *ENSEMBLE_ENDPOINTS,
        *(
            f"surface_dice_{tolerance:g}"
            for tolerance in config.ensembles.surface_tolerance_resolution_units
        ),
    )
    for round_number in sorted(rounds):
        similarities = multiscale_gradient_similarity(
            reference,
            rounds[round_number],
            roi,
            config.voxel_spacing_xyz_um,
            config.ensembles.gaussian_sigma_um,
            validity_mask=common_support,
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
        for endpoint in endpoints:
            values = current_values.get(endpoint, [])
            summary_rows.append(
                MetricRow(
                    "golgi",
                    "array",
                    round_number,
                    f"golgi_{endpoint}_median",
                    float(np.median(values)) if values else math.nan,
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
    output_dir: Path,
    rounds: Mapping[int, NDArray[np.float64]],
    result: MarkerResult,
    surface: NuclearSurface,
    roi: NDArray[np.bool_],
    threshold_masks: Mapping[int, tuple[MaskMember, ...]],
) -> tuple[Path, ...]:
    """Atomically publish fixed-grid composite overlays and threshold-response curves."""
    staged_paths: tuple[Path, ...] = ()
    staging_root: Path | None = None

    def populate(staged: Path) -> None:
        nonlocal staged_paths, staging_root
        import matplotlib.pyplot as plt

        staging_root = staged
        overlay = staged / "golgi-registered-overlays.npz"
        np.savez_compressed(
            overlay,
            consensus_nuclear_mask_zyx=surface.mask_zyx,
            fixed_perinuclear_roi_zyx=roi,
            **{f"round_{number}_target_zyx": image for number, image in rounds.items()},
            **{
                f"round_{number}_sigma_{member.sigma_um:g}_sbr_{member.threshold_sbr:g}_threshold_mask_zyx": member.mask_zyx
                for number, members in threshold_masks.items()
                for member in members
            },
        )
        curves = staged / "golgi-threshold-robustness.csv"
        with curves.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(EnsembleRow.__dataclass_fields__)
            writer.writerows(
                [getattr(row, name) for name in EnsembleRow.__dataclass_fields__]
                for row in result.ensemble_rows
            )
        plt.rcParams.update(
            {"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"}
        )
        reference = rounds[REFERENCE_ROUND].max(axis=0)
        roi_mip = roi.max(axis=0)
        nucleus_mip = surface.mask_zyx.max(axis=0)
        figure_paths: list[Path] = []
        union_masks = {
            number: np.logical_or.reduce([member.mask_zyx for member in members])
            for number, members in threshold_masks.items()
        }
        for round_number, image in sorted(rounds.items()):
            figure, axes = plt.subplots(1, 2, figsize=(8, 4), layout="constrained")
            for axis, title, mip, mask in (
                (
                    axes[0],
                    "round 2 reference",
                    reference,
                    union_masks[REFERENCE_ROUND],
                ),
                (
                    axes[1],
                    f"round {round_number}",
                    image.max(axis=0),
                    union_masks[round_number],
                ),
            ):
                finite = mip[np.isfinite(mip)]
                low, high = np.percentile(finite, (1.0, 99.5))
                axis.imshow(mip, cmap="magma", vmin=low, vmax=high)
                axis.contour(roi_mip, levels=(0.5,), colors="cyan", linewidths=0.7)
                axis.contour(nucleus_mip, levels=(0.5,), colors="white", linewidths=0.7)
                axis.contour(
                    mask.max(axis=0), levels=(0.5,), colors="lime", linewidths=0.7
                )
                axis.set_title(title)
                axis.set_axis_off()
            path = staged / f"golgi-overlay-round-{round_number}.png"
            figure.savefig(path, dpi=200)
            plt.close(figure)
            figure_paths.append(path)
        staged_paths = (overlay, curves, *figure_paths)

    _atomic_publish_directory(output_dir, populate)
    assert staging_root is not None
    return tuple(output_dir / path.relative_to(staging_root) for path in staged_paths)


def analyze_golgi_cell(
    cell: CellSpec,
    artifacts: tuple[RoundArtifacts, ...],
    config: AnalysisConfig,
    output_dir: Path,
) -> MarkerResult:
    """Load validated registered Golgi inputs and publish one cell's diagnostics.

    The entry point reconstructs target-independent support from each audited native
    target input and its persisted affine-before-warp chain using nearest-neighbor
    interpolation. It never accepts caller-supplied target-support masks.
    """
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
    common_validity, validity_digest, validity_identities = _artifact_validity_masks(
        {number: by_round[number] for number in accepted}, config
    )
    result = _rebind_cell(
        analyze_golgi_arrays(golgi, dapi, config, validity_masks=common_validity), cell
    )
    surface = consensus_nuclear_surface(
        dapi, config.voxel_spacing_xyz_um, config.nuclear
    )
    roi = _perinuclear_roi(surface.mask_zyx, common_validity, config)
    threshold_masks = {
        round_number: _support_threshold_ensemble(
            image,
            roi,
            np.logical_and.reduce(tuple(common_validity.values())),
            config,
        )
        for round_number, image in golgi.items()
    }
    diagnostics = _write_diagnostics(
        output_dir, golgi, result, surface, roi, threshold_masks
    )
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
        )
        + validity_identities
        + (f"validity-mask-sha256:{validity_digest}",),
    )


__all__ = [
    "ENSEMBLE_ENDPOINTS",
    "GradientSimilarityRow",
    "analyze_golgi_arrays",
    "analyze_golgi_cell",
    "golgi_nucleus_distances",
    "multiscale_gradient_similarity",
]
