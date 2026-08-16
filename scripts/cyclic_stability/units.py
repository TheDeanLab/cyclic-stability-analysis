"""Explicit theoretical optical references in expanded and nominal native units."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ResolutionReference:
    """Theoretical diffraction reference, labeled by the physical space used."""

    lateral_expanded_um: float
    axial_expanded_um: float
    lateral_nominal_native_um: float
    axial_nominal_native_um: float
    kind: str


def theoretical_resolution(
    wavelength_um: float,
    na: float,
    refractive_index: float,
    expansion: float,
) -> ResolutionReference:
    """Return the theoretical Rayleigh-style resolution reference in micrometers.

    The lateral ``0.61 lambda / NA`` and axial ``2 n lambda / NA²`` values
    describe the expanded specimen.  Nominal native-space values divide those
    theoretical references by the supplied expansion factor; they are not
    experimentally measured resolutions.
    """
    values = (wavelength_um, na, refractive_index, expansion)
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("optical values and expansion must be finite and positive")
    lateral = 0.61 * wavelength_um / na
    axial = 2.0 * refractive_index * wavelength_um / (na * na)
    return ResolutionReference(
        lateral_expanded_um=lateral,
        axial_expanded_um=axial,
        lateral_nominal_native_um=lateral / expansion,
        axial_nominal_native_um=axial / expansion,
        kind="theoretical",
    )


__all__ = ["ResolutionReference", "theoretical_resolution"]
