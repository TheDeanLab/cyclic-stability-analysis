"""Behavioral tests for exact ND2-to-TIFF source correspondence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import tifffile

from cyclic_stability.models import RoundSpec
from cyclic_stability.provenance import PlaneReader, compare_nd2_export


@dataclass(frozen=True)
class _FakePlaneReader:
    planes: dict[tuple[str, int], np.ndarray]

    def nd2_plane(self, _path: Path, channel: str, plane_index: int) -> np.ndarray:
        return self.planes[channel, plane_index]

    def tiff_plane(self, path: Path, plane_index: int) -> np.ndarray:
        channel = (
            "target"
            if "647_golgi" in path.name
            else "tubulin"
            if "488_tubulin" in path.name
            else "dapi"
        )
        return self.planes[channel, plane_index + 1]


@pytest.fixture
def fake_round(tmp_path: Path) -> RoundSpec:
    """A validated round with five-plane channel TIFF exports."""
    for channel, name in (
        ("target", "647_golgi"),
        ("tubulin", "488_tubulin"),
        ("dapi", "405_DAPI"),
    ):
        tifffile.imwrite(
            tmp_path / f"round_2_{name}.tif",
            np.arange(5 * 2 * 2, dtype=np.uint16).reshape(5, 2, 2),
        )
    return RoundSpec(
        cohort="golgi",
        cell_id="Cell1",
        acquisition_round=2,
        workflow_round=1,
        nd2_path=tmp_path / "source.nd2",
        tiff_directory=tmp_path,
        z_start_offset=1,
        target_exposure_ms=200.0,
        target_laser_percent=50.0,
        tubulin_exposure_ms=200.0,
        dapi_exposure_ms=200.0,
        dataset_manifest=tmp_path / "dataset.yaml",
        execution_profile=tmp_path / "execution.yaml",
        workflow_dataset_id="synthetic",
        identity_method="exact-pixel-audit",
    )


@pytest.fixture
def fake_plane_reader(fake_round: RoundSpec) -> PlaneReader:
    """Exact in-memory ND2 planes, deliberately separate from TIFF storage."""
    exported = {
        "target": tifffile.imread(fake_round.tiff_directory / "round_2_647_golgi.tif"),
        "tubulin": tifffile.imread(
            fake_round.tiff_directory / "round_2_488_tubulin.tif"
        ),
        "dapi": tifffile.imread(fake_round.tiff_directory / "round_2_405_DAPI.tif"),
    }
    return _FakePlaneReader(
        {
            (channel, plane + fake_round.z_start_offset): values[plane]
            for channel, values in exported.items()
            for plane in (0, 2, 4)
        }
    )


def test_exact_export_match_checks_three_channels_and_three_planes(
    fake_round: RoundSpec, fake_plane_reader: PlaneReader
) -> None:
    """A channel swap or Z-offset error must make an exact source audit fail."""
    matches = compare_nd2_export(fake_round, reader=fake_plane_reader)
    assert [(match.channel, match.plane_indices) for match in matches] == [
        ("target", (0, 2, 4)),
        ("tubulin", (0, 2, 4)),
        ("dapi", (0, 2, 4)),
    ]
    assert all(match.exact for match in matches)
