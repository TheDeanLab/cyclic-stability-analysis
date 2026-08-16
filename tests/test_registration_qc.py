"""Target-blind registration-QC behavior tests."""

from __future__ import annotations

from pathlib import Path

from cyclic_stability.registration_qc import seam_adjacency_mask


def test_internal_core_boundary_marks_only_its_16_pixel_neighborhood(
    tmp_path: Path,
) -> None:
    """Seam summaries must distinguish stitched boundaries from image interiors."""
    manifest = tmp_path / "two-core-manifest.json"
    manifest.write_text(
        '{"chunks": [{"z_start": 0, "z_end": 8, "y_start": 0, "y_end": 8, '
        '"x_start": 0, "x_end": 4}, {"z_start": 0, "z_end": 8, '
        '"y_start": 0, "y_end": 8, "x_start": 4, "x_end": 40}]}',
        encoding="utf-8",
    )
    mask = seam_adjacency_mask(manifest, (8, 8, 40))
    assert mask[:, :, 4].all()
    assert mask[:, :, 19].all()
    assert not mask[:, :, 21].any()
