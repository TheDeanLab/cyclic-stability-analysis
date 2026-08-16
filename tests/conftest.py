"""Shared fixtures for cyclic-stability tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from cyclic_stability.config import load_analysis_config
from cyclic_stability.models import AnalysisConfig
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


@pytest.fixture
def analysis_config(config_files: tuple[Path, Path, Path]) -> AnalysisConfig:
    """Return the validated frozen configuration used by Task 4 sensitivity tests."""
    return load_analysis_config(*config_files)
