"""Tests for the strict, immutable scientific configuration boundary."""

from __future__ import annotations

from pathlib import Path

import pytest
from cyclic_stability.config import load_analysis_config


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
