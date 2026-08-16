"""Strict parsers that convert frozen YAML and CSV inputs into value objects."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, TypeVar, cast

import yaml

from cyclic_stability.models import (
    AnalysisConfig,
    CellSpec,
    EnsembleConfig,
    ExecutionProfile,
    GolgiConfig,
    LysosomeConfig,
    NuclearConfig,
    OpticsConfig,
    QcDecision,
    RoundSpec,
)

COHORTS = frozenset(("golgi", "lamin", "lamp1"))
ROUND_NUMBERS = frozenset((1, 2, 3, 4, 5))
CORRESPONDENCE_COLUMNS = (
    "cohort",
    "cell_id",
    "acquisition_round",
    "workflow_round",
    "nd2_relative",
    "tiff_directory_relative",
    "z_start_offset",
    "target_channel",
    "target_exposure_ms",
    "target_laser_percent",
    "tubulin_exposure_ms",
    "tubulin_laser_percent",
    "dapi_exposure_ms",
    "dapi_laser_percent",
    "dataset_manifest",
    "execution_profile",
    "workflow_dataset_id",
    "identity_method",
)
QC_COLUMNS = (
    "cohort",
    "cell_id",
    "acquisition_round",
    "decision",
    "reasons",
    "reviewer",
    "reviewed_at",
)
T = TypeVar("T")


def _mapping(value: object, name: str, keys: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")  # noqa: TRY004
    actual = set(value)
    unknown = actual - keys
    missing = keys - actual
    if unknown:
        raise ValueError(f"{name} has unknown keys: {sorted(unknown)}")
    if missing:
        raise ValueError(f"{name} is missing keys: {sorted(missing)}")
    return cast(Mapping[str, Any], value)


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")  # noqa: TRY004
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: str, name: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _csv_number(value: str, name: str) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a number") from error
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _csv_round(value: str, name: str) -> int:
    result = _nonnegative_int(value, name)
    if result not in ROUND_NUMBERS:
        raise ValueError(f"{name} must be in 1--5")
    return result


def _sequence_numbers(
    value: object, name: str, expected_length: int | None = None
) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a numeric sequence")  # noqa: TRY004
    result = tuple(
        _number(item, f"{name}[{index}]") for index, item in enumerate(value)
    )
    if not result or (expected_length is not None and len(result) != expected_length):
        length = expected_length if expected_length is not None else "non-zero"
        raise ValueError(f"{name} must have length {length}")
    return result


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"cannot read YAML {path}: {error}") from error
    return _mapping(
        value,
        "analysis YAML",
        {
            "schema_version",
            "reference_acquisition_round",
            "voxel_spacing_xyz_um",
            "nominal_expansion_factor",
            "optics",
            "ensembles",
            "nuclear",
            "golgi",
            "lysosome",
            "paths",
        },
    )


def _parse_scientific_config(
    data: Mapping[str, Any],
) -> tuple[
    int,
    tuple[float, float, float],
    float,
    OpticsConfig,
    EnsembleConfig,
    NuclearConfig,
    GolgiConfig,
    LysosomeConfig,
    Mapping[str, Any],
]:
    if data["schema_version"] != 1 or isinstance(data["schema_version"], bool):
        raise ValueError("schema_version must be integer 1")
    reference = _positive_int(
        data["reference_acquisition_round"], "reference_acquisition_round"
    )
    if reference not in ROUND_NUMBERS:
        raise ValueError("reference_acquisition_round must be in 1--5")
    spacing = _sequence_numbers(
        data["voxel_spacing_xyz_um"], "voxel_spacing_xyz_um", expected_length=3
    )
    optics_data = _mapping(
        data["optics"],
        "optics",
        {"numerical_aperture", "immersion_refractive_index", "wavelength_um"},
    )
    wavelengths = _mapping(
        optics_data["wavelength_um"],
        "optics.wavelength_um",
        {"dapi", "tubulin", "target"},
    )
    ensembles_data = _mapping(
        data["ensembles"],
        "ensembles",
        {"gaussian_sigma_um", "threshold_sbr", "surface_tolerance_resolution_units"},
    )
    nuclear_data = _mapping(
        data["nuclear"],
        "nuclear",
        {
            "consensus_smoothing_um",
            "normal_half_window_um",
            "normal_step_um",
            "ridge_min_sbr",
        },
    )
    golgi_data = _mapping(data["golgi"], "golgi", {"perinuclear_outer_distance_um"})
    lysosome_data = _mapping(
        data["lysosome"],
        "lysosome",
        {"log_sigma_um", "detection_sbr", "match_resolution_units"},
    )
    paths = _mapping(
        data["paths"],
        "paths",
        {"nd2_root", "resubmission_root", "correspondence_markdown"},
    )
    if any(not isinstance(paths[key], str) or not paths[key] for key in paths):
        raise ValueError("paths values must be non-empty strings")
    return (
        reference,
        cast(tuple[float, float, float], spacing),
        _number(data["nominal_expansion_factor"], "nominal_expansion_factor"),
        OpticsConfig(
            _number(optics_data["numerical_aperture"], "optics.numerical_aperture"),
            _number(
                optics_data["immersion_refractive_index"],
                "optics.immersion_refractive_index",
            ),
            _number(wavelengths["dapi"], "optics.wavelength_um.dapi"),
            _number(wavelengths["tubulin"], "optics.wavelength_um.tubulin"),
            _number(wavelengths["target"], "optics.wavelength_um.target"),
        ),
        EnsembleConfig(
            _sequence_numbers(
                ensembles_data["gaussian_sigma_um"], "ensembles.gaussian_sigma_um"
            ),
            _sequence_numbers(
                ensembles_data["threshold_sbr"], "ensembles.threshold_sbr"
            ),
            _sequence_numbers(
                ensembles_data["surface_tolerance_resolution_units"],
                "ensembles.surface_tolerance_resolution_units",
            ),
        ),
        NuclearConfig(
            *(
                _number(nuclear_data[key], f"nuclear.{key}")
                for key in (
                    "consensus_smoothing_um",
                    "normal_half_window_um",
                    "normal_step_um",
                    "ridge_min_sbr",
                )
            )
        ),
        GolgiConfig(
            _number(
                golgi_data["perinuclear_outer_distance_um"],
                "golgi.perinuclear_outer_distance_um",
            )
        ),
        LysosomeConfig(
            _sequence_numbers(lysosome_data["log_sigma_um"], "lysosome.log_sigma_um"),
            _sequence_numbers(lysosome_data["detection_sbr"], "lysosome.detection_sbr"),
            _number(
                lysosome_data["match_resolution_units"],
                "lysosome.match_resolution_units",
            ),
        ),
        paths,
    )


def _read_csv(path: Path, columns: tuple[str, ...], label: str) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != columns:
                raise ValueError(
                    f"{label} columns must exactly match the frozen schema"
                )
            rows = list(reader)
    except OSError as error:
        raise ValueError(f"cannot read {label} {path}: {error}") from error
    for line_number, row in enumerate(rows, start=2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(
                f"{label} row {line_number} has an incorrect number of columns"
            )
        if any(not value.strip() for value in row.values()):
            raise ValueError(f"{label} row {line_number} contains an empty value")
    return rows


def _resolve_path(root: Path, relative: str, name: str) -> Path:
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        raise ValueError(f"{name} must be a safe relative path")
    return root / value


def _parse_rounds(
    rows: list[dict[str, str]], paths: Mapping[str, Any]
) -> tuple[CellSpec, ...]:
    cells: dict[tuple[str, str], list[RoundSpec]] = {}
    auxiliary_lasers: dict[tuple[str, str, int], tuple[float, float]] = {}
    seen: set[tuple[str, str, int]] = set()
    nd2_root = Path(cast(str, paths["nd2_root"]))
    resubmission_root = Path(cast(str, paths["resubmission_root"]))
    expected_target = {"golgi": "647_golgi", "lamin": "647_lamin", "lamp1": "647_lamp1"}
    for line_number, row in enumerate(rows, start=2):
        cohort = row["cohort"]
        if cohort not in COHORTS:
            raise ValueError(
                f"correspondence row {line_number} has unknown cohort {cohort!r}"
            )
        cell_id = row["cell_id"]
        if not cell_id:
            raise ValueError(f"correspondence row {line_number} has an empty cell_id")
        acquisition_round = _csv_round(row["acquisition_round"], "acquisition_round")
        workflow_round = _csv_round(row["workflow_round"], "workflow_round")
        key = (cohort, cell_id, acquisition_round)
        if key in seen:
            raise ValueError(
                f"duplicate cell-round: {cohort}/{cell_id}/{acquisition_round}"
            )
        seen.add(key)
        if row["target_channel"] != expected_target[cohort]:
            raise ValueError(f"target channel does not match cohort {cohort}")
        identity_method = row["identity_method"]
        if cohort == "lamp1" and identity_method != "image-verified":
            raise ValueError("LAMP1 identity must be image-verified")
        if identity_method not in {"exact-pixel-audit", "image-verified"}:
            raise ValueError(f"invalid identity_method {identity_method!r}")
        if not row["workflow_dataset_id"]:
            raise ValueError("workflow_dataset_id must be non-empty")
        auxiliary_lasers[key] = (
            _csv_number(row["tubulin_laser_percent"], "tubulin_laser_percent"),
            _csv_number(row["dapi_laser_percent"], "dapi_laser_percent"),
        )
        round_spec = RoundSpec(
            cohort=cast(Literal["golgi", "lamin", "lamp1"], cohort),
            cell_id=cell_id,
            acquisition_round=acquisition_round,
            workflow_round=workflow_round,
            nd2_path=_resolve_path(nd2_root, row["nd2_relative"], "nd2_relative"),
            tiff_directory=_resolve_path(
                nd2_root, row["tiff_directory_relative"], "tiff_directory_relative"
            ),
            z_start_offset=_nonnegative_int(row["z_start_offset"], "z_start_offset"),
            target_exposure_ms=_csv_number(
                row["target_exposure_ms"], "target_exposure_ms"
            ),
            target_laser_percent=_csv_number(
                row["target_laser_percent"], "target_laser_percent"
            ),
            tubulin_exposure_ms=_csv_number(
                row["tubulin_exposure_ms"], "tubulin_exposure_ms"
            ),
            dapi_exposure_ms=_csv_number(row["dapi_exposure_ms"], "dapi_exposure_ms"),
            dataset_manifest=_resolve_path(
                resubmission_root, row["dataset_manifest"], "dataset_manifest"
            ),
            execution_profile=_resolve_path(
                resubmission_root, row["execution_profile"], "execution_profile"
            ),
            workflow_dataset_id=row["workflow_dataset_id"],
            identity_method=cast(
                Literal["exact-pixel-audit", "image-verified"], identity_method
            ),
        )
        cells.setdefault((cohort, cell_id), []).append(round_spec)
    for cohort in ("golgi", "lamin"):
        cohort_cells = [item for item in cells if item[0] == cohort]
        if cohort_cells and (
            len(cohort_cells) != 5
            or sum(len(cells[item]) for item in cohort_cells) != 25
        ):
            raise ValueError(
                f"{cohort} cohort must contain exactly five cells by five rounds"
            )
    for (cohort, cell_id), cell_rounds in cells.items():
        rounds = {round_spec.acquisition_round for round_spec in cell_rounds}
        if rounds != ROUND_NUMBERS:
            raise ValueError(
                f"{cohort}/{cell_id} must contain acquisition rounds 1--5 exactly once"
            )
    _validate_cohort_settings(cells, auxiliary_lasers)
    return tuple(
        CellSpec(
            cast(Literal["golgi", "lamin", "lamp1"], cohort),
            cell_id,
            tuple(sorted(rounds, key=lambda item: item.acquisition_round)),
        )
        for (cohort, cell_id), rounds in sorted(cells.items())
    )


def _validate_cohort_settings(
    cells: Mapping[tuple[str, str], list[RoundSpec]],
    auxiliary_lasers: Mapping[tuple[str, str, int], tuple[float, float]],
) -> None:
    for cohort in COHORTS:
        by_round: dict[int, tuple[float, float, float, float, float, float]] = {}
        for (row_cohort, _), rounds in cells.items():
            if row_cohort != cohort:
                continue
            for round_spec in rounds:
                settings = (
                    round_spec.target_exposure_ms,
                    round_spec.target_laser_percent,
                    round_spec.tubulin_exposure_ms,
                    round_spec.dapi_exposure_ms,
                    *auxiliary_lasers[
                        (
                            round_spec.cohort,
                            round_spec.cell_id,
                            round_spec.acquisition_round,
                        )
                    ],
                )
                prior = by_round.setdefault(round_spec.acquisition_round, settings)
                if prior != settings:
                    raise ValueError(
                        f"inconsistent settings within {cohort} acquisition round {round_spec.acquisition_round}"
                    )


def _parse_qc_decisions(path: Path) -> tuple[QcDecision, ...]:
    rows = _read_csv(path, QC_COLUMNS, "QC decisions")
    decisions: list[QcDecision] = []
    seen: set[tuple[str, str, int]] = set()
    for line_number, row in enumerate(rows, start=2):
        cohort = row["cohort"]
        if cohort not in COHORTS:
            raise ValueError(
                f"QC decision row {line_number} has unknown cohort {cohort!r}"
            )
        acquisition_round = _csv_round(row["acquisition_round"], "QC acquisition_round")
        if acquisition_round == 2:
            raise ValueError("QC decisions apply only to moving (non-reference) rounds")
        if row["decision"] not in {"accept", "exclude"}:
            raise ValueError("QC decision must be accept or exclude")
        key = (cohort, row["cell_id"], acquisition_round)
        if key in seen:
            raise ValueError(
                f"duplicate QC decision: {cohort}/{row['cell_id']}/{acquisition_round}"
            )
        seen.add(key)
        reasons = tuple(
            reason.strip() for reason in row["reasons"].split(";") if reason.strip()
        )
        if not reasons:
            raise ValueError("QC decision reasons must contain at least one token")
        decisions.append(
            QcDecision(
                cast(Literal["golgi", "lamin", "lamp1"], cohort),
                row["cell_id"],
                acquisition_round,
                cast(Literal["accept", "exclude"], row["decision"]),
                reasons,
                row["reviewer"],
                row["reviewed_at"],
            )
        )
    return tuple(decisions)


def _digest(
    analysis: Mapping[str, Any],
    correspondence: list[dict[str, str]],
    decisions: tuple[QcDecision, ...],
) -> str:
    payload = {
        "analysis": analysis,
        "correspondence": correspondence,
        "qc_decisions": [
            {
                "cohort": decision.cohort,
                "cell_id": decision.cell_id,
                "acquisition_round": decision.acquisition_round,
                "decision": decision.decision,
                "reasons": decision.reasons,
                "reviewer": decision.reviewer,
                "reviewed_at": decision.reviewed_at,
            }
            for decision in decisions
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_analysis_config(
    path: Path, correspondence_path: Path, qc_path: Path
) -> AnalysisConfig:
    """Load all frozen scientific inputs and reject malformed or incomplete inventories."""
    data = _load_yaml(path)
    (
        reference,
        spacing,
        expansion,
        optics,
        ensembles,
        nuclear,
        golgi,
        lysosome,
        paths,
    ) = _parse_scientific_config(data)
    correspondence = _read_csv(
        correspondence_path, CORRESPONDENCE_COLUMNS, "correspondence"
    )
    cells = _parse_rounds(correspondence, paths)
    decisions = _parse_qc_decisions(qc_path)
    return AnalysisConfig(
        reference,
        spacing,
        expansion,
        optics,
        ensembles,
        nuclear,
        golgi,
        lysosome,
        cells,
        decisions,
        _digest(data, correspondence, decisions),
    )


def load_execution_profile(path: Path) -> ExecutionProfile:
    """Load a strict machine-specific profile without changing scientific inputs."""
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"cannot read execution profile {path}: {error}") from error
    data = _mapping(
        value,
        "execution profile",
        {"mode", "max_workers", "native_threads", "scratch_root", "slurm"}
        if isinstance(value, Mapping) and "slurm" in value
        else {"mode", "max_workers", "native_threads", "scratch_root"},
    )
    mode = data["mode"]
    if mode not in {"sequential", "process", "slurm"}:
        raise ValueError("execution mode must be sequential, process, or slurm")
    profile = ExecutionProfile(
        cast(Literal["sequential", "process", "slurm"], mode),
        _positive_int(data["max_workers"], "max_workers"),
        _positive_int(data["native_threads"], "native_threads"),
        Path(cast(str, data["scratch_root"])),
    )
    if mode != "slurm":
        return profile
    slurm = _mapping(
        data["slurm"],
        "slurm",
        {
            "processes_per_array_element",
            "array_concurrency",
            "cpus_per_task",
            "memory_gib",
            "walltime_hours",
        },
    )
    return ExecutionProfile(
        profile.mode,
        profile.max_workers,
        profile.native_threads,
        profile.scratch_root,
        _positive_int(
            slurm["processes_per_array_element"], "slurm.processes_per_array_element"
        ),
        _positive_int(slurm["array_concurrency"], "slurm.array_concurrency"),
        _positive_int(slurm["cpus_per_task"], "slurm.cpus_per_task"),
        _number(slurm["memory_gib"], "slurm.memory_gib"),
        _number(slurm["walltime_hours"], "slurm.walltime_hours"),
    )
