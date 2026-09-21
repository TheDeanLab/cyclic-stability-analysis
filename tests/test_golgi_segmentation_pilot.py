from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
from scipy.ndimage import binary_dilation

from scripts.golgi_segmentation_pilot import (
    CandidateSpec,
    dog_hysteresis_mask,
    mask_statistics,
    run_pilot,
)


def _synthetic_ribbon() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260830)
    yy, xx = np.indices((128, 160), dtype=np.float32)
    image = 100.0 + 0.08 * xx + 0.04 * yy
    image += rng.normal(0.0, 1.5, image.shape).astype(np.float32)
    ribbon = (
        (xx >= 35)
        & (xx <= 125)
        & (np.abs(yy - (58.0 + 7.0 * np.sin(xx / 14.0))) <= 2.0)
    )
    image[ribbon] += 35.0
    support = np.ones(image.shape, dtype=bool)
    support[:6] = False
    support[-6:] = False
    support[:, :6] = False
    support[:, -6:] = False
    return image, support, ribbon


def test_dog_hysteresis_retains_ribbon_and_rejects_background() -> None:
    image, support, ribbon = _synthetic_ribbon()
    spec = CandidateSpec(
        name="balanced",
        feature_sigma_um=0.20,
        background_sigma_um=2.0,
        low_sbr=2.0,
        high_sbr=6.0,
        minimum_size_voxels=16,
    )

    result = dog_hysteresis_mask(image, support, (0.10, 0.10), spec)

    assert not np.any(result.mask & ~support)
    assert np.count_nonzero(result.mask & ribbon) / np.count_nonzero(ribbon) >= 0.80
    far_background = support & ~binary_dilation(ribbon, iterations=6)
    assert (
        np.count_nonzero(result.mask & far_background)
        / np.count_nonzero(far_background)
        < 0.002
    )
    assert result.noise_scale > 0.0
    assert result.high_threshold > result.low_threshold


def test_mask_statistics_reports_nuclear_overlap_without_excluding_it() -> None:
    image = np.arange(36, dtype=np.float32).reshape(6, 6)
    support = np.ones((6, 6), dtype=bool)
    mask = np.zeros((6, 6), dtype=bool)
    mask[2:4, 2:5] = True
    nuclear = np.zeros((6, 6), dtype=bool)
    nuclear[2:4, 2:3] = True

    row = mask_statistics(image, mask, support, nuclear)

    assert row["mask_pixels"] == 6
    assert row["support_fraction"] == pytest.approx(1.0 / 6.0)
    assert row["nuclear_overlap_fraction"] == pytest.approx(1.0 / 3.0)
    assert row["masked_intensity_median"] == pytest.approx(18.0)
    assert row["background_intensity_median"] == pytest.approx(17.5)


@pytest.mark.parametrize(
    "changes",
    (
        {"feature_sigma_um": 0.0},
        {"background_sigma_um": 0.2},
        {"low_sbr": 6.0},
        {"high_sbr": 2.0},
        {"minimum_size_voxels": 0},
    ),
)
def test_candidate_spec_rejects_invalid_parameters(changes: dict[str, float]) -> None:
    values: dict[str, object] = {
        "name": "candidate",
        "feature_sigma_um": 0.2,
        "background_sigma_um": 2.0,
        "low_sbr": 2.0,
        "high_sbr": 6.0,
        "minimum_size_voxels": 16,
    }
    values.update(changes)

    with pytest.raises(ValueError):
        CandidateSpec(**values)  # type: ignore[arg-type]


def _write_source_npz(path: Path) -> None:
    _z, y, x = np.indices((7, 32, 40), dtype=np.float32)
    nucleus = ((x - 18.0) ** 2 + (y - 16.0) ** 2) <= 7.0**2
    roi = ((x - 18.0) ** 2 + (y - 16.0) ** 2) <= 15.0**2
    roi &= ~nucleus
    arrays: dict[str, np.ndarray] = {
        "consensus_nuclear_mask_zyx": nucleus,
        "fixed_perinuclear_roi_zyx": roi,
        "crop_start_zyx": np.array((2, 10, 20), dtype=np.int64),
        "crop_stop_zyx": np.array((9, 42, 60), dtype=np.int64),
        "full_shape_zyx": np.array((12, 80, 100), dtype=np.int64),
    }
    for round_number in range(1, 6):
        image = 100.0 + 0.02 * x + 0.03 * y
        ribbon = (np.abs(y - (16.0 + np.sin(x / 5.0))) < 1.5) & (x > 8) & (x < 32)
        image = image + ribbon * (20.0 + round_number)
        arrays[f"round_{round_number}_target_zyx"] = image.astype(np.float32)
    np.savez_compressed(path, **arrays)


def test_run_pilot_writes_auditable_individual_candidate_outputs(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.npz"
    _write_source_npz(source)
    output = tmp_path / "pilot"
    candidate = CandidateSpec("test", 0.2, 1.0, 2.0, 5.0, 4)

    run_pilot(source, output, (candidate,), (0.1, 0.1, 0.3))

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["pilot_scope"] == "xy-maximum-projection"
    assert manifest["source"]["path"] == str(source.resolve())
    assert manifest["source"]["sha256"]
    assert manifest["geometry"] == {
        "crop_start_zyx": [2, 10, 20],
        "crop_stop_zyx": [9, 42, 60],
        "cropped_shape_zyx": [7, 32, 40],
        "full_shape_zyx": [12, 80, 100],
        "voxel_spacing_xyz_um": [0.1, 0.1, 0.3],
    }
    assert manifest["nuclear_mask_policy"] == "context-only-not-hard-exclusion"
    assert (output / "reports" / "test-all-rounds.png").is_file()
    for round_number in range(1, 6):
        volume_path = (
            output / "inputs" / f"round-{round_number}-golgi-registered.ome.tif"
        )
        assert tifffile.imread(volume_path).shape == (7, 32, 40)
        assert (
            output / "candidates" / "test" / f"round-{round_number}-mip-mask.tif"
        ).is_file()
        assert (
            output / "candidates" / "test" / f"round-{round_number}-feature.tif"
        ).is_file()
    with (output / "candidate-metrics.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5
    assert {row["candidate"] for row in rows} == {"test"}
    assert {int(row["acquisition_round"]) for row in rows} == set(range(1, 6))


def test_run_pilot_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    source = tmp_path / "source.npz"
    _write_source_npz(source)
    output = tmp_path / "pilot"
    output.mkdir()

    with pytest.raises(FileExistsError, match="output directory already exists"):
        run_pilot(
            source,
            output,
            (CandidateSpec("test", 0.2, 1.0, 2.0, 5.0, 4),),
            (0.1, 0.1, 0.3),
        )
