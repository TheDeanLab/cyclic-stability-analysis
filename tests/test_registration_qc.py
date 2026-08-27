"""Target-blind registration-QC behavior tests."""

from __future__ import annotations

import runpy
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from cyclic_stability import registration_qc
from cyclic_stability.cyexm_artifacts import RoundArtifacts
from cyclic_stability.models import CellSpec, RoundSpec
from cyclic_stability.registration_qc import (
    _physical_affine,
    analyze_registration_qc,
    decompose_affine,
    publish_qc_outputs,
    seam_adjacency_mask,
)
from scipy.linalg import polar
from scipy.spatial.transform import Rotation


def test_qc_process_queue_is_bounded_and_preserves_input_order() -> None:
    """The cell queue must reject unsafe bounds and return deterministic ordering."""
    script = Path(__file__).parents[1] / "scripts" / "run_registration_qc.py"
    namespace = runpy.run_path(str(script), run_name="qc_cli_test")

    assert "_ordered_process_map" in namespace
    ordered_map = namespace["_ordered_process_map"]
    assert ordered_map(abs, (-3, -1, -2), max_workers=2) == (3, 1, 2)
    with pytest.raises(ValueError, match="positive integer"):
        ordered_map(abs, (-1,), max_workers=0)


def test_qc_cli_configures_native_threads_before_scientific_imports(
    tmp_path: Path,
) -> None:
    """The standalone CLI must establish its runtime policy before NumPy/ANTs load."""
    script = Path(__file__).parents[1] / "scripts" / "run_registration_qc.py"
    probe = textwrap.dedent(
        f"""
        import builtins
        import os
        import runpy
        import sys

        native_variables = (
            "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        )
        for name in native_variables:
            os.environ.pop(name, None)

        original_import = builtins.__import__
        saw_analyzer = False

        def checked_import(name, globals=None, locals=None, fromlist=(), level=0):
            global saw_analyzer
            if name == "cyclic_stability.registration_qc":
                saw_analyzer = True
                missing = [item for item in native_variables if item not in os.environ]
                if missing:
                    raise RuntimeError(f"scientific analyzer imported before {{missing}}")
            return original_import(name, globals, locals, fromlist, level)

        builtins.__import__ = checked_import
        sys.argv = [
            str({str(script)!r}),
            "--analysis-config", str({str(tmp_path / "missing-analysis.yaml")!r}),
            "--correspondence", str({str(tmp_path / "missing-correspondence.csv")!r}),
            "--run-dir", str({str(tmp_path / "unused-output")!r}),
        ]
        try:
            runpy.run_path({str(script)!r}, run_name="__main__")
        except (FileNotFoundError, ValueError):
            pass
        if not saw_analyzer:
            raise RuntimeError("scientific analyzer was never imported")
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", probe],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


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


def test_nested_cyexm_manifest_marks_the_actual_internal_core_boundary(
    tmp_path: Path,
) -> None:
    """The persisted schema-v2 nested chunk record must locate the seam."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"schema_version": 2, "chunks": ['
        '{"chunk": {"z_start": 0, "z_end": 8, "y_start": 0, '
        '"y_end": 8, "x_start": 0, "x_end": 20}}, '
        '{"chunk": {"z_start": 0, "z_end": 8, "y_start": 0, '
        '"y_end": 8, "x_start": 20, "x_end": 40}}]}',
        encoding="utf-8",
    )
    assert seam_adjacency_mask(manifest, (8, 8, 40))[:, :, 20].all()


def test_affine_decomposition_reconstructs_rotation_scale_and_shear() -> None:
    """Discarding the RQ orthogonal factor must fail this reconstruction contract."""
    scale = np.asarray((1.2, 0.8, 1.4))
    shear = np.asarray((0.15, -0.2, 0.1))
    upper = np.asarray(
        (
            (scale[0], np.tan(shear[0]) * scale[1], np.tan(shear[1]) * scale[2]),
            (0.0, scale[1], np.tan(shear[2]) * scale[2]),
            (0.0, 0.0, scale[2]),
        )
    )
    rotation = Rotation.from_euler("xyz", (10.0, -20.0, 35.0), degrees=True)
    matrix = np.eye(4)
    matrix[:3, :3] = upper @ rotation.as_matrix()
    matrix[:3, 3] = (3.0, -4.0, 5.0)

    summary = decompose_affine(matrix)

    reconstructed_upper = np.asarray(
        (
            (
                summary.scale_xyz[0],
                np.tan(np.radians(summary.shear_xy_deg)) * summary.scale_xyz[1],
                np.tan(np.radians(summary.shear_xz_deg)) * summary.scale_xyz[2],
            ),
            (
                0.0,
                summary.scale_xyz[1],
                np.tan(np.radians(summary.shear_yz_deg)) * summary.scale_xyz[2],
            ),
            (0.0, 0.0, summary.scale_xyz[2]),
        )
    )
    reconstructed = (
        reconstructed_upper
        @ Rotation.from_euler("xyz", summary.rotation_xyz_deg, degrees=True).as_matrix()
    )
    assert reconstructed == pytest.approx(matrix[:3, :3], abs=1e-10)
    assert summary.translation_xyz_um == pytest.approx((3.0, -4.0, 5.0))
    assert summary.polar_rotation_xyz_deg == pytest.approx(
        Rotation.from_matrix(polar(matrix[:3, :3])[0]).as_euler("xyz", degrees=True)
    )


def test_centered_ants_affine_is_converted_to_physical_xyz_offset() -> None:
    """ANTs center-of-rotation parameters must contribute to the 4-by-4 offset."""

    class Transform:
        parameters = np.asarray(
            (2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 7.0, 8.0, 9.0)
        )
        fixed_parameters = np.asarray((10.0, 20.0, 30.0))

    assert _physical_affine(Transform())[:3, 3] == pytest.approx((-3.0, 8.0, 9.0))


def test_improper_affine_is_hard_invalid_geometry() -> None:
    """A reflection cannot be emitted as a valid row with NaN pose values."""
    matrix = np.eye(4)
    matrix[0, 0] = -1.0
    with pytest.raises(ValueError, match="improper"):
        decompose_affine(matrix)


def test_publish_qc_outputs_replaces_a_complete_stale_tree_even_when_empty(
    tmp_path: Path,
) -> None:
    """A retry with no rows must atomically remove stale prior QC evidence."""
    destination = tmp_path / "output" / "qc"
    destination.mkdir(parents=True)
    (destination / "stale.txt").write_text("stale", encoding="utf-8")

    publish_qc_outputs(destination, (), (), ())

    assert not (destination / "stale.txt").exists()
    assert (
        (destination / "registration-qc.csv").read_text(encoding="utf-8").endswith("\n")
    )
    assert (destination / "deformation.csv").is_file()
    assert (destination / "decision-requests.csv").is_file()


def test_failed_overlay_publication_preserves_the_prior_complete_tree(
    tmp_path: Path,
) -> None:
    """An operational output failure must not replace a prior complete QC tree."""
    destination = tmp_path / "qc"
    destination.mkdir()
    (destination / "prior.txt").write_text("complete", encoding="utf-8")

    def fail_after_staging(overlays: Path) -> tuple[Path, ...]:
        overlays.mkdir(parents=True)
        (overlays / "partial.tif").write_bytes(b"partial")
        raise OSError("simulated full filesystem")

    with pytest.raises(OSError, match="simulated full filesystem"):
        publish_qc_outputs(destination, (), (), (), fail_after_staging)
    assert (destination / "prior.txt").read_text(encoding="utf-8") == "complete"
    assert not (destination / "registration-qc.csv").exists()


def _target_blind_inputs(tmp_path: Path) -> tuple[CellSpec, tuple[RoundArtifacts, ...]]:
    rounds = tuple(
        RoundSpec(
            cohort="golgi",
            cell_id="Cell1",
            acquisition_round=acquisition,
            workflow_round=acquisition,
            nd2_path=tmp_path / f"round-{acquisition}.nd2",
            tiff_directory=tmp_path / "raw",
            z_start_offset=0,
            producer_source_z_start_offset=0,
            target_exposure_ms=1.0,
            target_laser_percent=1.0,
            tubulin_exposure_ms=1.0,
            dapi_exposure_ms=1.0,
            dataset_manifest=tmp_path / "dataset.yaml",
            execution_profile=tmp_path / "execution.yaml",
            workflow_dataset_id="synthetic",
            identity_method="exact-pixel-audit",
        )
        for acquisition in (1, 2)
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"chunks": [{"chunk": {"z_start": 0, "z_end": 5, '
        '"y_start": 0, "y_end": 5, "x_start": 0, "x_end": 2}}, '
        '{"chunk": {"z_start": 0, "z_end": 5, "y_start": 0, '
        '"y_end": 5, "x_start": 2, "x_end": 5}}]}',
        encoding="utf-8",
    )

    def artifact(acquisition: int) -> RoundArtifacts:
        root = tmp_path / f"round-{acquisition}"
        return RoundArtifacts(
            acquisition_round=acquisition,
            workflow_round=acquisition,
            workflow_output_root=root,
            registered_dapi=root / "dapi.tif",
            registered_tubulin=root / "tubulin.tif",
            registered_target=root / "FORBIDDEN-target.tif",
            affine=None if acquisition == 2 else root / "affine.mat",
            warp=None if acquisition == 2 else root / "warp.nii.gz",
            pre_metrics=None if acquisition == 2 else root / "pre.txt",
            linear_metrics=None if acquisition == 2 else root / "linear.txt",
            nonlinear_metrics=None if acquisition == 2 else root / "nonlinear.txt",
            chunk_manifest=None if acquisition == 2 else manifest,
            completion_records=(),
        )

    return CellSpec("golgi", "Cell1", rounds), (artifact(1), artifact(2))


def test_analyzer_reads_only_blinded_channels_and_writes_metrics_and_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DAPI/tubulin-only evaluation must produce all three stage overlays and metrics."""
    cell, artifacts = _target_blind_inputs(tmp_path)
    read_paths: list[Path] = []
    backend_read_paths: list[Path] = []
    backend_spacings: list[tuple[float, float, float]] = []

    class Backend:
        def __init__(self, native_threads: object) -> None:
            pass

        def read_affine(self, path: Path) -> object:
            return SimpleNamespace(
                parameters=np.r_[np.eye(3).reshape(-1), (0.0, 0.0, 0.0)],
                fixed_parameters=np.zeros(3),
            )

        def read_warp(self, path: Path) -> str:
            return "warp"

        def to_array(self, value: object) -> np.ndarray:
            if value == "warp":
                return np.zeros((5, 5, 5, 3), dtype=np.float64)
            return np.zeros((5, 5, 5), dtype=np.float64)

        def load_image(self, path: Path, spacing: tuple[float, float, float]) -> Path:
            backend_read_paths.append(path)
            backend_spacings.append(spacing)
            return path

        def apply_affine(self, **kwargs: object) -> object:
            return kwargs["fixed"]

    def read_volume(path: Path) -> np.ndarray:
        read_paths.append(path)
        return np.ones((5, 5, 5), dtype=np.float64)

    metric_paths: list[Path] = []
    gradient_calls = 0
    gradient_spacings: list[tuple[float, float, float]] = []
    real_deformation_gradient = registration_qc.deformation_gradient

    def counted_deformation_gradient(
        warp: np.ndarray, spacing: tuple[float, float, float]
    ) -> np.ndarray:
        nonlocal gradient_calls
        gradient_calls += 1
        gradient_spacings.append(spacing)
        return real_deformation_gradient(warp, spacing)

    monkeypatch.setattr(registration_qc, "AntsRegistrationBackend", Backend)
    monkeypatch.setattr(registration_qc, "_load_overlay_volume", read_volume)
    monkeypatch.setattr(
        registration_qc, "deformation_gradient", counted_deformation_gradient
    )
    monkeypatch.setattr(
        registration_qc,
        "read_metrics",
        lambda path: metric_paths.append(path) or {"Correlation": 0.5},
    )

    result = analyze_registration_qc(cell, artifacts, tmp_path / "out")

    assert len(result.overlay_paths) == 18
    assert result.decision_requests[0]["decision"] == ""
    assert result.registration_rows[0]["pre_Correlation"] == 0.5
    assert len(metric_paths) == 3
    assert gradient_calls == 1
    assert all("FORBIDDEN" not in str(path) for path in read_paths)
    assert all("FORBIDDEN" not in str(path) for path in backend_read_paths)
    assert set(backend_spacings) == {(0.1083, 0.1083, 0.3)}
    assert gradient_spacings == [(0.1083, 0.1083, 0.3)]
    assert {path.name.split("-")[2] for path in result.overlay_paths} == {
        "pre",
        "affine",
        "nonlinear",
    }

    def improper_affine(self: object, path: Path) -> object:
        return SimpleNamespace(
            parameters=np.r_[np.diag((-1.0, 1.0, 1.0)).reshape(-1), (0.0, 0.0, 0.0)],
            fixed_parameters=np.zeros(3),
        )

    monkeypatch.setattr(Backend, "read_affine", improper_affine)
    invalid = analyze_registration_qc(cell, artifacts, tmp_path / "invalid")
    assert invalid.decision_requests[0]["automated_valid"] is False
    assert "improper" in str(invalid.decision_requests[0]["automated_reason"])

    def output_error(*args: object, **kwargs: object) -> tuple[Path, ...]:
        raise OSError("simulated output failure")

    monkeypatch.setattr(registration_qc, "publish_qc_outputs", output_error)
    with pytest.raises(OSError, match="simulated output failure"):
        analyze_registration_qc(cell, artifacts, tmp_path / "failed-output")
