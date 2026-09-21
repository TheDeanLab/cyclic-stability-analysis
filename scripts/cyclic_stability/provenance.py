"""Exact, read-only source correspondence checks for cyclic imaging exports."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Protocol

import nd2
import numpy as np
import tifffile
from cyexm.errors import InputValidationError

from cyclic_stability.models import RoundSpec


@dataclass(frozen=True, slots=True)
class ChannelMatch:
    """Exact ND2/TIFF equality result for one source channel."""

    channel: str
    plane_indices: tuple[int, ...]
    exact: bool


class PlaneReader(Protocol):
    """Read only individual ND2 and TIFF planes needed for an audit."""

    def nd2_plane(self, path: Path, channel: str, plane_index: int) -> np.ndarray:
        """Return one ND2 source plane for the named channel."""

    def tiff_plane(self, path: Path, plane_index: int) -> np.ndarray:
        """Return one TIFF export plane."""


class _ProductionPlaneReader:
    _CHANNEL_INDICES: ClassVar[dict[str, int]] = {
        "target": 0,
        "tubulin": 1,
        "dapi": 2,
    }

    def nd2_plane(self, path: Path, channel: str, plane_index: int) -> np.ndarray:
        """Read one physical ND2 Z plane without materializing the volume."""
        try:
            channel_index = self._CHANNEL_INDICES[channel]
            with nd2.ND2File(path) as source:
                frame = np.array(source.read_frame(plane_index), copy=True)
        except (KeyError, OSError, RuntimeError, ValueError) as error:
            raise InputValidationError(
                f"cannot read ND2 plane {channel}/{plane_index}: {path}"
            ) from error
        if frame.ndim != 3 or channel_index >= frame.shape[0]:
            raise InputValidationError(
                f"ND2 does not contain expected three-channel plane: {path}"
            )
        return np.array(frame[channel_index], copy=True)

    def tiff_plane(self, path: Path, plane_index: int) -> np.ndarray:
        """Read one physical TIFF Z plane without materializing the stack."""
        try:
            with tifffile.TiffFile(path) as source:
                return np.array(source.pages[plane_index].asarray(), copy=True)
        except (IndexError, OSError, RuntimeError, ValueError) as error:
            raise InputValidationError(
                f"cannot read TIFF plane {plane_index}: {path}"
            ) from error


def _target_basename(round_spec: RoundSpec) -> str:
    target = {"golgi": "647_golgi", "lamin": "647_lamin", "lamp1": "647_LAMP1"}
    return target[round_spec.cohort]


def channel_tiff_paths(round_spec: RoundSpec) -> tuple[tuple[str, Path], ...]:
    """Return audited channels in fixed target, tubulin, DAPI order."""
    prefix = f"round_{round_spec.acquisition_round}_"
    return (
        (
            "target",
            round_spec.tiff_directory / f"{prefix}{_target_basename(round_spec)}.tif",
        ),
        ("tubulin", round_spec.tiff_directory / f"{prefix}488_tubulin.tif"),
        ("dapi", round_spec.tiff_directory / f"{prefix}405_DAPI.tif"),
    )


def attested_pretransform_channels(round_spec: RoundSpec) -> frozenset[str]:
    """Return moving channels covered by a complete explicit rotation attestation.

    A one-off, target-blind pre-rotation intentionally changes TIFF pixels before
    normal Cy-ExM registration. It is admissible only when the adjacent
    ``rotation-seed.json`` declares every moving channel and the finite rotation
    operation. Reference rounds and incomplete or malformed records retain the
    ordinary exact ND2/TIFF requirement.
    """
    if round_spec.acquisition_round == 2:
        return frozenset()
    record = round_spec.tiff_directory.parent / "rotation-seed.json"
    try:
        payload = json.loads(record.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return frozenset()
    expected = dict(channel_tiff_paths(round_spec))
    declared = payload.get("rotated_channels") if isinstance(payload, dict) else None
    angle = payload.get("rotation_xy_deg") if isinstance(payload, dict) else None
    source_directory = (
        payload.get("source_directory") if isinstance(payload, dict) else None
    )
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("rotation_about") != "image center"
        or payload.get("interpolation") != "linear"
        or payload.get("fill_value") != 0
        or not isinstance(angle, (int, float))
        or isinstance(angle, bool)
        or not math.isfinite(angle)
        or angle == 0
        or not isinstance(source_directory, str)
        or not Path(source_directory).is_dir()
        or not isinstance(declared, list)
        or {item for item in declared if isinstance(item, str)}
        != {path.name for path in expected.values()}
        or len(declared) != len(expected)
    ):
        return frozenset()
    return frozenset(expected)


def _audit_planes(path: Path) -> tuple[int, ...]:
    """Select first, middle, and final TIFF planes as deterministic audit anchors."""
    try:
        with tifffile.TiffFile(path) as source:
            count = len(source.pages)
    except (OSError, RuntimeError, ValueError) as error:
        raise InputValidationError(f"cannot inspect TIFF stack: {path}") from error
    if count < 3:
        raise InputValidationError(f"TIFF stack requires at least three planes: {path}")
    return 0, count // 2, count - 1


def compare_nd2_export(
    round_spec: RoundSpec, reader: PlaneReader | None = None
) -> tuple[ChannelMatch, ...]:
    """Compare three ND2 source channels to their TIFF exports exactly."""
    source = _ProductionPlaneReader() if reader is None else reader
    matches: list[ChannelMatch] = []
    for channel, tiff_path in channel_tiff_paths(round_spec):
        indices = _audit_planes(tiff_path)
        comparisons = tuple(
            np.array_equal(
                source.nd2_plane(
                    round_spec.nd2_path,
                    channel,
                    index + round_spec.z_start_offset,
                ),
                source.tiff_plane(tiff_path, index),
            )
            for index in indices
        )
        matches.append(ChannelMatch(channel, indices, all(comparisons)))
    return tuple(matches)


__all__ = [
    "ChannelMatch",
    "PlaneReader",
    "attested_pretransform_channels",
    "channel_tiff_paths",
    "compare_nd2_export",
]
