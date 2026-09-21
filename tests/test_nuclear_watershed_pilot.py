from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from scripts.nuclear_watershed_pilot import (
    WatershedSpec,
    dapi_review_projections,
    label_review_projections,
    physical_image_geometry,
    run_nuclear_pilot,
    split_touching_nuclei,
)


def test_physical_watershed_splits_two_touching_spheres_and_preserves_union() -> None:
    zz, yy, xx = np.indices((32, 56, 56), dtype=np.float32)
    first = (zz - 16.0) ** 2 + (yy - 28.0) ** 2 + (xx - 18.0) ** 2 <= 10.0**2
    second = (zz - 16.0) ** 2 + (yy - 28.0) ** 2 + (xx - 34.0) ** 2 <= 10.0**2
    merged = first | second

    result = split_touching_nuclei(
        merged,
        (1.0, 1.0, 1.0),
        WatershedSpec("split", seed_prominence_um=2.0),
    )

    assert result.seed_count == 2
    assert set(np.unique(result.labels)) == {0, 1, 2}
    assert np.array_equal(result.labels > 0, merged)
    assert result.labels[16, 28, 18] != result.labels[16, 28, 34]


def test_watershed_spec_rejects_nonpositive_prominence() -> None:
    try:
        WatershedSpec("invalid", seed_prominence_um=0.0)
    except ValueError as error:
        assert "seed_prominence_um" in str(error)
    else:
        raise AssertionError("zero prominence was accepted")


def test_review_projections_use_selected_plane_and_remove_plane_background() -> None:
    volume = np.zeros((4, 3, 3), dtype=np.float32)
    volume[0] = 100.0
    volume[2, 1, 1] = 20.0

    xy, xz, yz = dapi_review_projections(
        volume,
        xy_plane_index=2,
        plane_background_percentile=10.0,
        orthogonal_projection_percentile=100.0,
    )

    assert np.array_equal(xy, volume[2])
    assert np.all(xz[0] == 0.0)
    assert np.all(yz[0] == 0.0)
    assert xz[2, 1] == 20.0
    assert yz[2, 1] == 20.0


def test_label_review_xy_uses_same_plane_instead_of_maximum_projection() -> None:
    labels = np.zeros((4, 3, 3), dtype=np.uint16)
    labels[2, 1, 1] = 1
    labels[0, 0, 0] = 2

    xy, xz, yz = label_review_projections(labels, xy_plane_index=2)

    assert xy[0][1, 1]
    assert not np.any(xy[1])
    assert xz[1][0, 0]
    assert yz[1][0, 0]


def test_physical_image_geometry_maps_top_row_to_small_vertical_coordinate() -> None:
    x_centers, y_centers, extent = physical_image_geometry(
        (2, 3), horizontal_spacing_um=1.0, vertical_spacing_um=2.0
    )

    assert np.array_equal(x_centers, np.array((0.5, 1.5, 2.5)))
    assert np.array_equal(y_centers, np.array((1.0, 3.0)))
    assert extent == (0.0, 3.0, 4.0, 0.0)


def _write_pilot_inputs(root: Path) -> tuple[Path, Path, Path, np.ndarray]:
    zz, yy, xx = np.indices((24, 48, 48), dtype=np.float32)
    first = (zz - 12.0) ** 2 + (yy - 24.0) ** 2 + (xx - 15.0) ** 2 <= 9.0**2
    second = (zz - 12.0) ** 2 + (yy - 24.0) ** 2 + (xx - 33.0) ** 2 <= 9.0**2
    foreground = first | second
    source = root / "source.npz"
    np.savez_compressed(
        source,
        consensus_nuclear_mask_zyx=foreground,
        crop_start_zyx=np.array((2, 5, 7), dtype=np.int64),
        crop_stop_zyx=np.array((26, 53, 55), dtype=np.int64),
        full_shape_zyx=np.array((30, 60, 70), dtype=np.int64),
    )
    full_dapi = np.zeros((30, 60, 70), dtype=np.uint16)
    crop = (100 + 20 * foreground + 5 * first).astype(np.uint16)
    full_dapi[2:26, 5:53, 7:55] = crop
    dapi_path = root / "reference-dapi.tif"
    tifffile.imwrite(dapi_path, full_dapi)
    golgi = np.zeros((48, 48), dtype=np.uint8)
    golgi[20:28, 20:28] = 1
    golgi_path = root / "round-5-conservative-mask.tif"
    tifffile.imwrite(golgi_path, golgi)
    return source, dapi_path, golgi_path, foreground


def test_run_nuclear_pilot_writes_labels_metrics_and_auditable_context(
    tmp_path: Path,
) -> None:
    source, dapi_path, golgi_path, foreground = _write_pilot_inputs(tmp_path)
    output = tmp_path / "output"

    run_nuclear_pilot(
        source,
        dapi_path,
        golgi_path,
        output,
        (WatershedSpec("split", 2.0),),
        (1.0, 1.0, 1.0),
        nominal_expansion_factor=1.0,
        xy_plane_1based=12,
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["pilot_scope"] == "physical-3d-distance-watershed"
    assert manifest["foreground_policy"] == "preserve-saved-consensus-union-exactly"
    assert manifest["source_consensus_npz"]["path"] == str(source.resolve())
    assert manifest["display_reference_dapi"]["path"] == str(dapi_path.resolve())
    assert manifest["round_5_conservative_golgi_mask"]["path"] == str(
        golgi_path.resolve()
    )
    assert manifest["geometry"]["cropped_shape_zyx"] == [24, 48, 48]
    original = tifffile.imread(output / "inputs" / "original-consensus-mask.ome.tif")
    labels = tifffile.imread(output / "candidates" / "split" / "labels.ome.tif")
    assert original.shape == labels.shape == foreground.shape
    assert np.array_equal(original > 0, foreground)
    assert np.array_equal(labels > 0, foreground)
    assert set(np.unique(labels)) == {0, 1, 2}
    with (output / "component-metrics.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert {row["candidate"] for row in rows} == {"split"}
    assert (output / "reports" / "nuclear-watershed-comparison.png").is_file()


def test_run_nuclear_pilot_refuses_existing_output(tmp_path: Path) -> None:
    source, dapi_path, golgi_path, _foreground = _write_pilot_inputs(tmp_path)
    output = tmp_path / "output"
    output.mkdir()

    with pytest.raises(FileExistsError, match="output directory already exists"):
        run_nuclear_pilot(
            source,
            dapi_path,
            golgi_path,
            output,
            (WatershedSpec("split", 2.0),),
            (1.0, 1.0, 1.0),
            nominal_expansion_factor=1.0,
            xy_plane_1based=12,
        )
