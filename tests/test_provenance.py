"""Behavioral tests for exact ND2-to-TIFF source correspondence."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest
import tifffile
from cyclic_stability.models import RoundSpec
from cyclic_stability.provenance import (
    PlaneReader,
    attested_pretransform_channels,
    channel_tiff_paths,
    compare_nd2_export,
)


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


@dataclass
class _CountingMismatchReader:
    calls: list[tuple[str, str, int]]

    def nd2_plane(self, _path: Path, channel: str, plane_index: int) -> np.ndarray:
        self.calls.append(("nd2", channel, plane_index))
        return (
            np.ones((1, 1), dtype=np.uint16)
            if channel == "target" and plane_index == 1
            else np.zeros((1, 1), dtype=np.uint16)
        )

    def tiff_plane(self, path: Path, plane_index: int) -> np.ndarray:
        channel = (
            "target"
            if "647_golgi" in path.name
            else "tubulin"
            if "488_tubulin" in path.name
            else "dapi"
        )
        self.calls.append(("tiff", channel, plane_index))
        return np.zeros((1, 1), dtype=np.uint16)


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
        producer_source_z_start_offset=0,
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


def test_mismatch_still_reads_every_declared_channel_plane(
    fake_round: RoundSpec,
) -> None:
    """A first-plane mismatch must not overstate unexamined planes as audited."""
    reader = _CountingMismatchReader([])
    matches = compare_nd2_export(fake_round, reader=reader)
    assert [match.exact for match in matches] == [False, True, True]
    assert len([call for call in reader.calls if call[0] == "nd2"]) == 9
    assert len([call for call in reader.calls if call[0] == "tiff"]) == 9


def test_three_plane_stacks_are_audited_at_first_middle_and_last(
    fake_round: RoundSpec,
) -> None:
    """The stated first/middle/last policy needs no undocumented fifth plane."""
    for path in fake_round.tiff_directory.glob("*.tif"):
        tifffile.imwrite(
            path, np.zeros((3, 2, 2), dtype=np.uint16), photometric="minisblack"
        )
    reader = _CountingMismatchReader([])
    matches = compare_nd2_export(fake_round, reader=reader)
    assert [match.plane_indices for match in matches] == [(0, 1, 2)] * 3


def test_lamp1_uses_the_exported_uppercase_target_basename(
    fake_round: RoundSpec,
) -> None:
    """Production LAMP1 exports retain the marker's uppercase basename."""
    lamp_round = RoundSpec(
        cohort="lamp1",
        cell_id="LAMP1-verified-alpha",
        acquisition_round=fake_round.acquisition_round,
        workflow_round=fake_round.workflow_round,
        nd2_path=fake_round.nd2_path,
        tiff_directory=fake_round.tiff_directory,
        z_start_offset=fake_round.z_start_offset,
        producer_source_z_start_offset=fake_round.producer_source_z_start_offset,
        target_exposure_ms=fake_round.target_exposure_ms,
        target_laser_percent=fake_round.target_laser_percent,
        tubulin_exposure_ms=fake_round.tubulin_exposure_ms,
        dapi_exposure_ms=fake_round.dapi_exposure_ms,
        dataset_manifest=fake_round.dataset_manifest,
        execution_profile=fake_round.execution_profile,
        workflow_dataset_id=fake_round.workflow_dataset_id,
        identity_method="image-verified",
    )
    paths = dict(channel_tiff_paths(lamp_round))
    assert paths["target"].name == "round_2_647_LAMP1.tif"


def test_rotation_seed_attests_only_every_declared_moving_channel(
    fake_round: RoundSpec,
) -> None:
    """A declared rotation seed is the sole exception to exact export identity."""
    rotated = fake_round.tiff_directory / "rotated"
    rotated.mkdir()
    seeded = replace(
        fake_round,
        acquisition_round=4,
        workflow_round=4,
        tiff_directory=rotated,
    )
    for _channel, path in channel_tiff_paths(seeded):
        tifffile.imwrite(path, np.zeros((5, 2, 2), dtype=np.uint16))
    record = {
        "schema_version": 1,
        "purpose": "one-off target-blind Cell 1 round-4 tubulin XY rotation seed",
        "rotation_xy_deg": -6.3,
        "rotation_about": "image center",
        "translation": "intentionally omitted; estimated by the regular Cy-ExM workflow",
        "interpolation": "linear",
        "fill_value": 0,
        "source_directory": str(fake_round.tiff_directory),
        "rotated_channels": [
            path.name for _channel, path in channel_tiff_paths(seeded)
        ],
    }
    (rotated.parent / "rotation-seed.json").write_text(json.dumps(record))

    assert attested_pretransform_channels(seeded) == frozenset(
        {"target", "tubulin", "dapi"}
    )

    record["rotated_channels"] = record["rotated_channels"][:-1]
    (rotated.parent / "rotation-seed.json").write_text(json.dumps(record))
    assert attested_pretransform_channels(seeded) == frozenset()
