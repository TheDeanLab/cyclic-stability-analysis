"""Tests for the strict, immutable scientific configuration boundary."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import yaml
from cyclic_stability.config import load_analysis_config, load_execution_profile
from cyclic_stability.models import EnsembleRow, MetricRow


def _lamp_rows(cell_id: str) -> str:
    return "".join(
        f"lamp1,{cell_id},{acquisition_round},{acquisition_round},"
        f"round{acquisition_round}/DAPI_tubulin_lamp1_001.nd2,"
        f"Lysosomes/{cell_id},0,647_lamp1,"
        f"{150 if acquisition_round == 1 else 200},50,200,50,200,75,"
        f"lamp.yaml,exec.yaml,lamp-{cell_id},image-verified\n"
        for acquisition_round in range(1, 6)
    )


def test_config_preserves_acquisition_and_workflow_rounds(
    config_files: tuple[Path, ...],
) -> None:
    """Changing the explicit Golgi permutation must not silently relabel rounds."""
    analysis, correspondence, decisions = config_files
    config = load_analysis_config(analysis, correspondence, decisions)
    golgi = config.cell("golgi", "Cell1")
    assert [item.acquisition_round for item in golgi.rounds] == [1, 2, 3, 4, 5]
    assert [item.workflow_round for item in golgi.rounds] == [2, 1, 3, 4, 5]
    assert config.reference_acquisition_round == 2
    assert config.nominal_expansion_factor == 4.5


def test_analysis_config_parses_multinuclear_bounds(
    config_files: tuple[Path, ...],
) -> None:
    """The frozen analysis manifest declares the accepted nuclear population."""
    config = load_analysis_config(*config_files)
    assert config.nuclear.accepted_component_count == (1, 6)
    assert config.nuclear.nominal_volume_um3 == (100.0, 5000.0)
    assert config.nuclear.minimum_sphericity == 0.15
    assert config.nuclear.maximum_face_contact_fraction == 0.10


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("accepted_component_count", [0, 6]),
        ("accepted_component_count", [6, 1]),
        ("accepted_component_count", [1, 7]),
        ("nominal_volume_um3", [5000.0, 100.0]),
        ("minimum_sphericity", 1.1),
        ("maximum_face_contact_fraction", 1.1),
    ],
)
def test_analysis_config_rejects_invalid_multinuclear_bounds(
    tmp_path: Path,
    config_files: tuple[Path, ...],
    key: str,
    value: object,
) -> None:
    """Population-selection bounds must remain physically meaningful."""
    analysis, correspondence, decisions = config_files
    data = yaml.safe_load(analysis.read_text(encoding="utf-8"))
    data["nuclear"][key] = value
    invalid = tmp_path / "invalid-analysis.yaml"
    invalid.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError):
        load_analysis_config(invalid, correspondence, decisions)


def test_output_rows_default_to_an_explicit_aggregate_nucleus_id() -> None:
    """Legacy row constructors serialize their unambiguous aggregate analysis unit."""
    metric = MetricRow("golgi", "Cell1", 2, "volume", 1.0, "um3", "all", True, "id")
    ensemble = EnsembleRow(
        "golgi", "Cell1", 2, "volume", "sigma=1", 1.0, 1.0, "um3", "id"
    )
    assert asdict(metric)["nucleus_id"] == "aggregate"
    assert asdict(ensemble)["nucleus_id"] == "aggregate"


def test_config_rejects_numeric_lamp_identity(
    tmp_path: Path, config_files: tuple[Path, ...]
) -> None:
    """Numeric filenames must not admit unresolved LAMP1 identity mappings."""
    analysis, correspondence, decisions = config_files
    text = correspondence.read_text(encoding="utf-8")
    text += (
        "lamp1,Cell1,1,1,DAPI_tubulin_lamp_003.nd2,LAMP1/Cell1,0,"
        "647_lamp1,150,50,200,50,200,75,lamp.yaml,exec.yaml,lamp-cell1,"
        "numeric-suffix\n"
    )
    invalid = tmp_path / "correspondence.csv"
    invalid.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="LAMP1 identity must be image-verified"):
        load_analysis_config(analysis, invalid, decisions)


def test_config_requires_biological_round_two_reference(
    config_files: tuple[Path, ...],
) -> None:
    """Changing the fixed round would invalidate every reference comparison."""
    analysis, correspondence, decisions = config_files
    invalid = analysis.with_name("wrong-reference.yaml")
    invalid.write_text(
        analysis.read_text(encoding="utf-8").replace(
            "reference_acquisition_round: 2", "reference_acquisition_round: 1"
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError, match="reference_acquisition_round must be exactly 2"
    ):
        load_analysis_config(invalid, correspondence, decisions)


def test_config_requires_five_image_verified_lamp_cells(
    config_files: tuple[Path, ...],
) -> None:
    """One mapped LAMP1 cell is insufficient for the predeclared production gate."""
    analysis, correspondence, decisions = config_files
    incomplete = correspondence.with_name("one-lamp-cell.csv")
    incomplete.write_text(
        correspondence.read_text(encoding="utf-8") + _lamp_rows("verified_alpha"),
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError, match="lamp1 cohort must contain exactly five cells by five rounds"
    ):
        load_analysis_config(analysis, incomplete, decisions)


def test_config_rejects_suffix_derived_lamp_cell_identity(
    config_files: tuple[Path, ...],
) -> None:
    """Image verification cannot turn a numeric source suffix into a cell identity."""
    analysis, correspondence, decisions = config_files
    invalid = correspondence.with_name("numeric-lamp-cell.csv")
    lamp_cells = (
        "Cell1",
        "verified_alpha",
        "verified_beta",
        "verified_gamma",
        "verified_delta",
    )
    invalid.write_text(
        correspondence.read_text(encoding="utf-8")
        + "".join(_lamp_rows(cell_id) for cell_id in lamp_cells),
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError, match="LAMP1 cell_id must not be numeric or suffix-derived"
    ):
        load_analysis_config(analysis, invalid, decisions)


def test_config_preserves_image_verified_biological_lamp_identifiers(
    config_files: tuple[Path, ...],
) -> None:
    """A reviewed biological ID may contain the LAMP1 marker name but no numeric suffix."""
    analysis, correspondence, decisions = config_files
    valid = correspondence.with_name("image-verified-lamp.csv")
    lamp_cells = (
        "LAMP1-verified-alpha",
        "LAMP1-verified-beta",
        "LAMP1-verified-gamma",
        "LAMP1-verified-delta",
        "LAMP1-verified-epsilon",
    )
    valid.write_text(
        correspondence.read_text(encoding="utf-8")
        + "".join(_lamp_rows(cell_id) for cell_id in lamp_cells),
        encoding="utf-8",
    )
    config = load_analysis_config(analysis, valid, decisions)
    assert {cell.cell_id for cell in config.cells if cell.cohort == "lamp1"} == set(
        lamp_cells
    )


def test_config_rejects_boolean_physical_values(config_files: tuple[Path, ...]) -> None:
    """A YAML boolean must never be accepted as a positive numeric setting."""
    analysis, correspondence, decisions = config_files
    invalid = analysis.with_name("invalid-analysis.yaml")
    invalid.write_text(
        analysis.read_text(encoding="utf-8").replace(
            "nominal_expansion_factor: 4.5", "nominal_expansion_factor: true"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="nominal_expansion_factor.*number"):
        load_analysis_config(invalid, correspondence, decisions)


def test_config_rejects_non_numeric_auxiliary_laser_setting(
    config_files: tuple[Path, ...],
) -> None:
    """A non-numeric tubulin laser setting must not pass the strict CSV boundary."""
    analysis, correspondence, decisions = config_files
    invalid = correspondence.with_name("invalid-laser.csv")
    invalid.write_text(
        correspondence.read_text(encoding="utf-8").replace(
            ",200,50,200,75,", ",200,not-a-number,200,75,", 1
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="tubulin_laser_percent.*number"):
        load_analysis_config(analysis, invalid, decisions)


def test_config_rejects_incomplete_golgi_inventory(
    config_files: tuple[Path, ...],
) -> None:
    """Dropping one audited Golgi round must reject a scientifically incomplete cohort."""
    analysis, correspondence, decisions = config_files
    rows = correspondence.read_text(encoding="utf-8").splitlines()
    incomplete = correspondence.with_name("incomplete.csv")
    incomplete.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(
        ValueError, match="golgi cohort must contain exactly five cells by five rounds"
    ):
        load_analysis_config(analysis, incomplete, decisions)


def test_config_rejects_qc_decision_for_unknown_cell(
    config_files: tuple[Path, ...],
) -> None:
    """A target-blind decision must refer to a loaded non-reference cell-round."""
    analysis, correspondence, decisions = config_files
    invalid = decisions.with_name("unknown-qc.csv")
    invalid.write_text(
        decisions.read_text(encoding="utf-8")
        + "golgi,UnknownCell,1,accept,overlay-ok,reviewer,2026-08-16T00:00:00Z\n",
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError, match="QC decision does not match a loaded cell-round"
    ):
        load_analysis_config(analysis, correspondence, invalid)


@pytest.mark.parametrize(
    ("text", "message"),
    (
        (
            "mode: slurm\nmax_workers: 2\nnative_threads: 1\nscratch_root: /tmp/scratch\n",
            "SLURM execution profile requires a slurm mapping",
        ),
        (
            "mode: sequential\nmax_workers: 2\nnative_threads: 1\nscratch_root: /tmp/scratch\nslurm: {}\n",
            "only valid when mode is slurm",
        ),
    ),
)
def test_execution_profile_rejects_mode_schema_mismatch(
    tmp_path: Path, text: str, message: str
) -> None:
    """SLURM settings must be present exactly when the profile selects SLURM."""
    profile = tmp_path / "invalid-execution.yaml"
    profile.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_execution_profile(profile)
