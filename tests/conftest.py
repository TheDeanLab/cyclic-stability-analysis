"""Shared fixtures for cyclic-stability tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from synthetic import ANALYSIS_YAML, write_synthetic_correspondence


@pytest.fixture
def config_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Write frozen scientific, correspondence, and empty-QC inputs."""
    analysis = tmp_path / "analysis.yaml"
    correspondence = tmp_path / "correspondence.csv"
    decisions = tmp_path / "qc-decisions.csv"
    analysis.write_text(ANALYSIS_YAML, encoding="utf-8")
    write_synthetic_correspondence(correspondence)
    decisions.write_text(
        "cohort,cell_id,acquisition_round,decision,reasons,reviewer,reviewed_at\n",
        encoding="utf-8",
    )
    return analysis, correspondence, decisions
