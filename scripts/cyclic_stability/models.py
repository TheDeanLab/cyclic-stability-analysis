"""Immutable value objects at the scientific configuration boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True, slots=True)
class OpticsConfig:
    """Objective and wavelength values in micrometers."""

    numerical_aperture: float
    immersion_refractive_index: float
    dapi_wavelength_um: float
    tubulin_wavelength_um: float
    target_wavelength_um: float


@dataclass(frozen=True, slots=True)
class EnsembleConfig:
    """Frozen multiscale and threshold analysis settings."""

    gaussian_sigma_um: tuple[float, ...]
    threshold_sbr: tuple[float, ...]
    surface_tolerance_resolution_units: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class NuclearConfig:
    """DAPI consensus and normal-profile settings."""

    consensus_smoothing_um: float
    normal_half_window_um: float
    normal_step_um: float
    ridge_min_sbr: float
    accepted_component_count: tuple[int, int]
    nominal_volume_um3: tuple[float, float]
    minimum_sphericity: float
    maximum_face_contact_fraction: float


@dataclass(frozen=True, slots=True)
class GolgiConfig:
    """Golgi morphology settings."""

    perinuclear_outer_distance_um: float


@dataclass(frozen=True, slots=True)
class LysosomeConfig:
    """LAMP1 detection and matching settings."""

    log_sigma_um: tuple[float, ...]
    detection_sbr: tuple[float, ...]
    match_resolution_units: float


@dataclass(frozen=True, slots=True)
class RoundSpec:
    """One validated cell/acquisition round and its immutable inputs."""

    cohort: Literal["golgi", "lamin", "lamp1"]
    cell_id: str
    acquisition_round: int
    workflow_round: int
    nd2_path: Path
    tiff_directory: Path
    z_start_offset: int
    target_exposure_ms: float
    target_laser_percent: float
    tubulin_exposure_ms: float
    dapi_exposure_ms: float
    dataset_manifest: Path
    execution_profile: Path
    workflow_dataset_id: str
    identity_method: Literal["exact-pixel-audit", "image-verified"]


@dataclass(frozen=True, slots=True)
class CellSpec:
    """The five validated acquisition rounds for one biological-cell label."""

    cohort: Literal["golgi", "lamin", "lamp1"]
    cell_id: str
    rounds: tuple[RoundSpec, ...]


@dataclass(frozen=True, slots=True)
class ExecutionProfile:
    """Machine-specific execution settings, separate from scientific inputs."""

    mode: Literal["sequential", "process", "slurm"]
    max_workers: int
    native_threads: int
    scratch_root: Path
    processes_per_array_element: int | None = None
    array_concurrency: int | None = None
    cpus_per_task: int | None = None
    memory_gib: float | None = None
    walltime_hours: float | None = None


@dataclass(frozen=True, slots=True)
class QcDecision:
    """A target-blind human decision for one moving round."""

    cohort: Literal["golgi", "lamin", "lamp1"]
    cell_id: str
    acquisition_round: int
    decision: Literal["accept", "exclude"]
    reasons: tuple[str, ...]
    reviewer: str
    reviewed_at: str


@dataclass(frozen=True, slots=True)
class MetricRow:
    """One compact within-cell summary metric."""

    cohort: str
    cell_id: str
    acquisition_round: int
    endpoint: str
    value: float
    units: str
    comparison_set: str
    qc_included: bool
    source_task_identity: str
    nucleus_id: str = "aggregate"


@dataclass(frozen=True, slots=True)
class EnsembleRow:
    """One compact within-cell sensitivity-ensemble metric."""

    cohort: str
    cell_id: str
    acquisition_round: int
    endpoint: str
    ensemble_name: str
    ensemble_value: float
    sigma_um: float
    threshold_sbr: float
    surface_tolerance_resolution_units: float | None
    value: float | str
    units: str
    source_task_identity: str
    nucleus_id: str = "aggregate"


@dataclass(frozen=True, slots=True)
class MarkerResult:
    """One cell's derived summary tables, diagnostics, and source identities."""

    cohort: str
    cell_id: str
    summary_rows: tuple[MetricRow, ...]
    ensemble_rows: tuple[EnsembleRow, ...]
    diagnostics: tuple[Path, ...]
    input_identities: tuple[str, ...]
    component_qc_rows: tuple[Mapping[str, object], ...] = ()


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    """Fully validated scientific configuration and audited cell inventory."""

    reference_acquisition_round: int
    voxel_spacing_xyz_um: tuple[float, float, float]
    nominal_expansion_factor: float
    optics: OpticsConfig
    ensembles: EnsembleConfig
    nuclear: NuclearConfig
    golgi: GolgiConfig
    lysosome: LysosomeConfig
    cells: tuple[CellSpec, ...]
    qc_decisions: tuple[QcDecision, ...]
    config_digest: str

    def cell(self, cohort: str, cell_id: str) -> CellSpec:
        """Return the unique validated cell matching ``cohort`` and ``cell_id``."""
        matches = [
            cell
            for cell in self.cells
            if cell.cohort == cohort and cell.cell_id == cell_id
        ]
        if len(matches) != 1:
            raise KeyError(
                f"expected one cell for {cohort}/{cell_id}; got {len(matches)}"
            )
        return matches[0]
