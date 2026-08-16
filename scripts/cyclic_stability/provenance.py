"""Exact, read-only source correspondence checks for cyclic imaging exports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import nd2
import numpy as np
import tifffile

from cyclic_stability.models import RoundSpec
from cyexm.errors import InputValidationError


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
    _CHANNEL_INDICES = {"target": 0, "tubulin": 1, "dapi": 2}

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
    target = {"golgi": "647_golgi", "lamin": "647_lamin", "lamp1": "647_lamp1"}
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


def _audit_planes(path: Path) -> tuple[int, ...]:
    """Select first, middle, and final TIFF planes as deterministic audit anchors."""
    try:
        with tifffile.TiffFile(path) as source:
            count = len(source.pages)
    except (OSError, RuntimeError, ValueError) as error:
        raise InputValidationError(f"cannot inspect TIFF stack: {path}") from error
    if count < 5:
        raise InputValidationError(f"TIFF stack requires at least five planes: {path}")
    return 0, count // 2, count - 1


def compare_nd2_export(
    round_spec: RoundSpec, reader: PlaneReader | None = None
) -> tuple[ChannelMatch, ...]:
    """Compare three ND2 source channels to their TIFF exports exactly."""
    source = _ProductionPlaneReader() if reader is None else reader
    matches: list[ChannelMatch] = []
    for channel, tiff_path in channel_tiff_paths(round_spec):
        indices = _audit_planes(tiff_path)
        exact = all(
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
        matches.append(ChannelMatch(channel, indices, exact))
    return tuple(matches)


__all__ = ["ChannelMatch", "PlaneReader", "channel_tiff_paths", "compare_nd2_export"]
