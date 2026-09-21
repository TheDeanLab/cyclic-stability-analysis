"""Target-blind production inventory behavior for the analysis entry point."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
from cyclic_stability.models import NuclearConfig
from cyclic_stability.nuclear import NuclearComponentQc
from cyclic_stability.task import RetainedContent
from cyexm.spatial import ImageGeometry

from scripts import run_analysis


def test_profile_threads_are_exported_before_scientific_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-default profile must configure every native runtime from the shell."""
    monkeypatch.delenv("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", raising=False)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    monkeypatch.delenv("NUMEXPR_NUM_THREADS", raising=False)
    monkeypatch.delenv("VECLIB_MAXIMUM_THREADS", raising=False)

    run_analysis._export_native_threads(4)

    assert {name: os.environ[name] for name in run_analysis._THREAD_ENVIRONMENT} == {
        name: "4" for name in run_analysis._THREAD_ENVIRONMENT
    }


def test_validated_registered_dapi_uses_dapi_only_snapshot(
    tmp_path: Path, monkeypatch
) -> None:
    """The entry point must not fall back to whole-workflow target validation."""
    expected = tmp_path / "registered-405-DAPI.tif"
    expected.write_bytes(b"descriptor-backed-DAPI")
    round_spec = SimpleNamespace(acquisition_round=2)
    config = SimpleNamespace(voxel_spacing_xyz_um=(0.1, 0.1, 0.3))
    observed: list[object] = []
    retained_inputs: list[RetainedContent] = []

    def snapshot(value, actual_config):
        retained = RetainedContent.from_path(expected)
        retained_inputs.append(retained)
        observed.append((value, actual_config))
        return SimpleNamespace(
            content=retained,
            close=retained.close,
        )

    monkeypatch.setattr(
        "cyclic_stability.cyexm_artifacts.snapshot_registered_dapi", snapshot
    )

    monkeypatch.setattr(
        "cyclic_stability.images.load_volume",
        lambda source: np.frombuffer(source.read(), dtype=np.uint8),
    )

    resolved = run_analysis._validated_registered_dapi(round_spec, config)

    assert observed == [(round_spec, config)]
    assert bytes(resolved) == b"descriptor-backed-DAPI"
    assert all(retained.closed for retained in retained_inputs)


def test_nuclear_inventory_only_exits_before_target_tasks_and_writes_exact_qc(
    tmp_path: Path, monkeypatch
) -> None:
    """Removing the early inventory exit must expose the forbidden task builder."""
    inputs = []
    for name in (
        "analysis.yaml",
        "correspondence.csv",
        "qc-decisions.csv",
        "execution.yaml",
    ):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        inputs.append(path)
    rounds = tuple(
        SimpleNamespace(
            cohort="lamin",
            cell_id="Field-A",
            acquisition_round=number,
        )
        for number in range(1, 6)
    )
    cell = SimpleNamespace(cohort="lamin", cell_id="Field-A", rounds=rounds)
    decisions = tuple(
        SimpleNamespace(
            cohort="lamin",
            cell_id="Field-A",
            acquisition_round=number,
            decision="accept" if number in (1, 4) else "exclude",
        )
        for number in (1, 3, 4, 5)
    )
    config = SimpleNamespace(
        cells=(cell,),
        qc_decisions=decisions,
        reference_acquisition_round=2,
        voxel_spacing_xyz_um=(0.1, 0.1, 0.3),
        nominal_expansion_factor=4.5,
        nuclear=NuclearConfig(
            consensus_smoothing_um=0.5,
            normal_half_window_um=1.5,
            normal_step_um=0.05,
            ridge_min_sbr=3.0,
            accepted_component_count=(1, 6),
            nominal_volume_um3=(100.0, 5000.0),
            minimum_sphericity=0.15,
            maximum_face_contact_fraction=0.1,
        ),
        config_digest="inventory-digest",
    )
    profile = SimpleNamespace(mode="sequential", native_threads=1, scratch_root=None)
    monkeypatch.setattr(
        "cyclic_stability.config.load_analysis_config_bytes",
        lambda *_args, **_kwargs: config,
    )
    monkeypatch.setattr(
        "cyclic_stability.config.load_execution_profile_bytes",
        lambda *_args, **_kwargs: profile,
    )
    monkeypatch.setattr(
        "cyclic_stability.task.build_tasks",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("target task inventory must not be built")
        ),
    )
    dapi_paths = {}
    for number in (1, 2, 4):
        path = tmp_path / f"round-{number}-405-DAPI.tif"
        path.write_bytes(f"dapi-{number}-v1".encode())
        dapi_paths[number] = path
    retained_inputs: list[RetainedContent] = []

    def snapshot_registered_dapi(
        round_spec, actual_config, *, expected_fixed_geometry=None
    ):
        assert actual_config is config
        retained = RetainedContent.from_path(dapi_paths[round_spec.acquisition_round])
        retained_inputs.append(retained)
        geometry = ImageGeometry(
            shape=(3, 3, 3),
            origin=(0.0, 0.0, 0.0),
            spacing=config.voxel_spacing_xyz_um,
            direction=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        )
        if expected_fixed_geometry is not None:
            assert geometry == expected_fixed_geometry
        return SimpleNamespace(
            path=Path(retained.identity.path),
            consumer_path=retained.consumer_path,
            content=retained,
            geometry=geometry,
            identities=(retained.identity,),
        )

    monkeypatch.setattr(
        "cyclic_stability.cyexm_artifacts.snapshot_registered_dapi",
        snapshot_registered_dapi,
    )
    monkeypatch.setattr(run_analysis, "_software_identity", lambda: "repository-v1")
    opened: list[str] = []

    def load_volume(source) -> np.ndarray:
        path = Path(source.name)
        assert path.parent == Path("/proc/self/fd")
        canonical = Path(os.readlink(path))
        opened.append(canonical.name)
        if "DAPI" not in canonical.name or "target" in canonical.name.lower():
            raise AssertionError(f"target-blind inventory opened {path}")
        return np.ones((3, 3, 3), dtype=np.float64)

    monkeypatch.setattr("cyclic_stability.images.load_volume", load_volume)
    component_qc = (
        NuclearComponentQc(
            "N001",
            True,
            (),
            72900.0,
            800.0,
            0.91,
            (1.0, 2.0, 3.0),
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ),
        NuclearComponentQc(
            "C002",
            False,
            ("volume-below-minimum", "face-contact-above-maximum"),
            729.0,
            8.0,
            0.42,
            (4.0, 5.0, 6.0),
            (0.2, 0.0, 0.0, 0.0, 0.0, 0.0),
        ),
    )
    monkeypatch.setattr(
        "cyclic_stability.nuclear.consensus_nuclear_surface",
        lambda *_args: SimpleNamespace(component_qc=component_qc),
    )
    monkeypatch.setattr(
        "cyclic_stability.nuclear.adjudicate_nuclear_components",
        lambda *_args: SimpleNamespace(component_qc=component_qc),
    )
    monkeypatch.setattr(
        "cyclic_stability.nuclear.realize_nuclear_surfaces",
        lambda *_args: SimpleNamespace(component_qc=component_qc),
    )
    run_dir = tmp_path / "inventory-run"
    result = run_analysis.main(
        [
            "--analysis-config",
            str(inputs[0]),
            "--correspondence",
            str(inputs[1]),
            "--qc-decisions",
            str(inputs[2]),
            "--execution-profile",
            str(inputs[3]),
            "--run-dir",
            str(run_dir),
            "--nuclear-inventory-only",
        ]
    )

    assert result == 0
    assert opened == [
        "round-1-405-DAPI.tif",
        "round-2-405-DAPI.tif",
        "round-4-405-DAPI.tif",
    ]
    task_dir = run_dir / "tasks" / "nuclear-inventory"
    terminal = json.loads((task_dir / "terminal.json").read_text(encoding="utf-8"))
    assert terminal["status"] == "complete"
    first_attempt = terminal["attempt_id"]
    attempt_dir = task_dir / "attempts" / first_attempt
    inventory = attempt_dir / "nuclear-inventory.json"
    payload = json.loads(inventory.read_text(encoding="utf-8"))
    assert payload["accepted_component_count_range"] == [1, 6]
    assert payload["all_fields_ready"] is True
    assert payload["all_fields_surface_ready"] is True
    assert payload["all_fields_within_count_contract"] is True
    assert payload["cells"] == [
        {
            "accepted_count": 1,
            "accepted_ids": ["N001"],
            "candidate_count": 2,
            "cell_id": "Field-A",
            "cohort": "lamin",
            "components": [
                {
                    "accepted": True,
                    "centroid_xyz_um": [1.0, 2.0, 3.0],
                    "expanded_volume_um3": 72900.0,
                    "face_contact_fractions": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "nominal_volume_um3": 800.0,
                    "nucleus_id": "N001",
                    "rejection_reasons": [],
                    "sphericity": 0.91,
                },
                {
                    "accepted": False,
                    "centroid_xyz_um": [4.0, 5.0, 6.0],
                    "expanded_volume_um3": 729.0,
                    "face_contact_fractions": [0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "nominal_volume_um3": 8.0,
                    "nucleus_id": "C002",
                    "rejection_reasons": [
                        "volume-below-minimum",
                        "face-contact-above-maximum",
                    ],
                    "sphericity": 0.42,
                },
            ],
            "field_id": "lamin/Field-A",
            "ready": True,
            "surface_realization": {"status": "complete"},
            "surface_ready": True,
            "within_accepted_component_contract": True,
        }
    ]
    assert inventory.read_text(encoding="utf-8") == (
        json.dumps(payload, indent=4, sort_keys=True) + "\n"
    )
    assert not list(attempt_dir.glob(".nuclear-inventory.json.*.tmp"))
    record = json.loads((attempt_dir / "record.json").read_text(encoding="utf-8"))
    assert record["software_identity"] == "repository-v1"
    assert {Path(item["path"]) for item in record["input_identities"]}.issuperset(
        set(dapi_paths.values())
    )

    assert (
        run_analysis.main(
            [
                "--analysis-config",
                str(inputs[0]),
                "--correspondence",
                str(inputs[1]),
                "--qc-decisions",
                str(inputs[2]),
                "--execution-profile",
                str(inputs[3]),
                "--run-dir",
                str(run_dir),
                "--nuclear-inventory-only",
            ]
        )
        == 0
    )
    assert json.loads((task_dir / "terminal.json").read_text())["attempt_id"] == (
        first_attempt
    )

    dapi_paths[1].write_bytes(b"dapi-1-v2")
    assert (
        run_analysis.main(
            [
                "--analysis-config",
                str(inputs[0]),
                "--correspondence",
                str(inputs[1]),
                "--qc-decisions",
                str(inputs[2]),
                "--execution-profile",
                str(inputs[3]),
                "--run-dir",
                str(run_dir),
                "--nuclear-inventory-only",
            ]
        )
        == 0
    )
    second_attempt = json.loads((task_dir / "terminal.json").read_text())["attempt_id"]
    assert second_attempt == first_attempt
    assert inventory.is_file()

    monkeypatch.setattr(run_analysis, "_software_identity", lambda: "repository-v2")
    assert (
        run_analysis.main(
            [
                "--analysis-config",
                str(inputs[0]),
                "--correspondence",
                str(inputs[1]),
                "--qc-decisions",
                str(inputs[2]),
                "--execution-profile",
                str(inputs[3]),
                "--run-dir",
                str(run_dir),
                "--nuclear-inventory-only",
            ]
        )
        == 0
    )
    third_attempt = json.loads((task_dir / "terminal.json").read_text())["attempt_id"]
    assert third_attempt == first_attempt
    assert retained_inputs
    assert all(retained.closed for retained in retained_inputs)


def test_nuclear_inventory_decode_cannot_publish_transient_replacement_bytes(
    tmp_path: Path, monkeypatch
) -> None:
    """Replacing and restoring caller DAPI during decode yields attested A or failure."""
    inputs = []
    for name in (
        "analysis.yaml",
        "correspondence.csv",
        "qc-decisions.csv",
        "execution.yaml",
    ):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        inputs.append(path)
    round_spec = SimpleNamespace(cohort="lamin", cell_id="Field-A", acquisition_round=2)
    cell = SimpleNamespace(cohort="lamin", cell_id="Field-A", rounds=(round_spec,))
    config = SimpleNamespace(
        cells=(cell,),
        qc_decisions=(),
        reference_acquisition_round=2,
        voxel_spacing_xyz_um=(0.1, 0.1, 0.3),
        nominal_expansion_factor=4.5,
        nuclear=NuclearConfig(
            consensus_smoothing_um=0.5,
            normal_half_window_um=1.5,
            normal_step_um=0.05,
            ridge_min_sbr=3.0,
            accepted_component_count=(1, 6),
            nominal_volume_um3=(1.0, 5000.0),
            minimum_sphericity=0.0,
            maximum_face_contact_fraction=1.0,
        ),
        config_digest="replacement-regression",
    )
    profile = SimpleNamespace(mode="sequential", native_threads=1, scratch_root=None)
    monkeypatch.setattr(
        "cyclic_stability.config.load_analysis_config_bytes",
        lambda *_args, **_kwargs: config,
    )
    monkeypatch.setattr(
        "cyclic_stability.config.load_execution_profile_bytes",
        lambda *_args, **_kwargs: profile,
    )
    source = tmp_path / "round-2-405-DAPI.tif"
    tifffile.imwrite(source, np.full((3, 3, 3), 7, dtype=np.uint16))
    original_payload = source.read_bytes()
    original_mtime_ns = source.stat().st_mtime_ns
    replacement = tmp_path / "replacement-DAPI.tif"
    tifffile.imwrite(replacement, np.full((3, 3, 3), 99, dtype=np.uint16))
    retained_inputs: list[RetainedContent] = []

    def snapshot_registered_dapi(
        _round_spec, actual_config, *, expected_fixed_geometry=None
    ):
        assert actual_config is config
        retained = RetainedContent.from_path(source)
        retained_inputs.append(retained)
        geometry = ImageGeometry(
            shape=(3, 3, 3),
            origin=(0.0, 0.0, 0.0),
            spacing=config.voxel_spacing_xyz_um,
            direction=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        )
        assert expected_fixed_geometry is None or geometry == expected_fixed_geometry
        return SimpleNamespace(
            path=Path(retained.identity.path),
            consumer_path=retained.consumer_path,
            content=retained,
            geometry=geometry,
            identities=(retained.identity,),
        )

    monkeypatch.setattr(
        "cyclic_stability.cyexm_artifacts.snapshot_registered_dapi",
        snapshot_registered_dapi,
    )
    from cyclic_stability import images

    real_load_volume = images.load_volume
    decoded: list[int] = []

    def replace_during_decode(source_handle) -> np.ndarray:
        path = Path(source_handle.name)
        assert path.parent == Path("/proc/self/fd")
        os.replace(replacement, source)
        volume = np.array(real_load_volume(source_handle), copy=True)
        decoded.append(int(volume[0, 0, 0]))
        restored = tmp_path / "restored-DAPI.tif"
        restored.write_bytes(original_payload)
        os.utime(restored, ns=(restored.stat().st_atime_ns, original_mtime_ns))
        os.replace(restored, source)
        return volume

    monkeypatch.setattr(images, "load_volume", replace_during_decode)

    def adjudicate(dapi, *_args):
        observed = float(next(iter(dapi.values()))[0, 0, 0])
        return SimpleNamespace(
            component_qc=(
                NuclearComponentQc(
                    "N001",
                    True,
                    (),
                    observed,
                    observed,
                    1.0,
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                ),
            )
        )

    monkeypatch.setattr(
        "cyclic_stability.nuclear.adjudicate_nuclear_components", adjudicate
    )
    monkeypatch.setattr(
        "cyclic_stability.nuclear.realize_nuclear_surfaces",
        lambda *_args: SimpleNamespace(),
    )
    monkeypatch.setattr(run_analysis, "_software_identity", lambda: "repository-v1")
    run_dir = tmp_path / "inventory-run"

    result = run_analysis.main(
        [
            "--analysis-config",
            str(inputs[0]),
            "--correspondence",
            str(inputs[1]),
            "--qc-decisions",
            str(inputs[2]),
            "--execution-profile",
            str(inputs[3]),
            "--run-dir",
            str(run_dir),
            "--nuclear-inventory-only",
        ]
    )

    task_dir = run_dir / "tasks" / "nuclear-inventory"
    terminal = json.loads((task_dir / "terminal.json").read_text())
    attempt = task_dir / "attempts" / terminal["attempt_id"]
    payload = json.loads((attempt / "nuclear-inventory.json").read_text())
    assert decoded == [7]
    assert payload["cells"][0]["components"][0]["expanded_volume_um3"] == 7.0
    assert terminal["status"] == "complete"
    assert result == 0
    assert all(retained.closed for retained in retained_inputs)


def test_nuclear_inventory_reports_out_of_contract_before_target_work(
    tmp_path: Path, monkeypatch
) -> None:
    """Dropping the contract result must turn a zero-nucleus field into success."""
    cell = SimpleNamespace(
        cohort="lamin",
        cell_id="Field-Z",
        rounds=(
            SimpleNamespace(cohort="lamin", cell_id="Field-Z", acquisition_round=2),
        ),
    )
    config = SimpleNamespace(
        cells=(cell,),
        qc_decisions=(),
        reference_acquisition_round=2,
        voxel_spacing_xyz_um=(0.1, 0.1, 0.3),
        nominal_expansion_factor=4.5,
        nuclear=NuclearConfig(
            consensus_smoothing_um=0.5,
            normal_half_window_um=1.5,
            normal_step_um=0.05,
            ridge_min_sbr=3.0,
            accepted_component_count=(1, 6),
            nominal_volume_um3=(100.0, 5000.0),
            minimum_sphericity=0.15,
            maximum_face_contact_fraction=0.1,
        ),
    )
    monkeypatch.setattr(
        run_analysis,
        "_validated_registered_dapi",
        lambda _round, _config: np.ones((3, 3, 3), dtype=np.float64),
        raising=False,
    )
    monkeypatch.setattr(
        "cyclic_stability.images.load_volume",
        lambda _path: np.ones((3, 3, 3), dtype=np.float64),
    )
    monkeypatch.setattr(
        "cyclic_stability.nuclear.consensus_nuclear_surface",
        lambda *_args: SimpleNamespace(component_qc=()),
    )
    monkeypatch.setattr(
        "cyclic_stability.nuclear.adjudicate_nuclear_components",
        lambda *_args: SimpleNamespace(component_qc=()),
    )

    def fail_count(*_args):
        raise ValueError("accepted nuclear component count 0 is outside range")

    monkeypatch.setattr("cyclic_stability.nuclear.realize_nuclear_surfaces", fail_count)

    payload = run_analysis._nuclear_inventory_payload(config, (cell,))

    assert payload == {
        "accepted_component_count_range": [1, 6],
        "all_fields_ready": False,
        "all_fields_surface_ready": False,
        "all_fields_within_count_contract": False,
        "cells": [
            {
                "accepted_count": 0,
                "accepted_ids": [],
                "candidate_count": 0,
                "cell_id": "Field-Z",
                "cohort": "lamin",
                "components": [],
                "field_id": "lamin/Field-Z",
                "ready": False,
                "surface_realization": {
                    "error_type": "ValueError",
                    "message": "accepted nuclear component count 0 is outside range",
                    "status": "failed",
                },
                "surface_ready": False,
                "within_accepted_component_contract": False,
            }
        ],
    }


def test_nuclear_inventory_publishes_component_qc_when_strict_normals_fail(
    tmp_path: Path, monkeypatch
) -> None:
    """Letting strict realization abort would discard target-blind component rows."""
    cell = SimpleNamespace(
        cohort="golgi",
        cell_id="Field-N",
        rounds=(
            SimpleNamespace(cohort="golgi", cell_id="Field-N", acquisition_round=2),
        ),
    )
    config = SimpleNamespace(
        cells=(cell,),
        qc_decisions=(),
        reference_acquisition_round=2,
        voxel_spacing_xyz_um=(0.1, 0.1, 0.3),
        nominal_expansion_factor=4.5,
        nuclear=NuclearConfig(
            consensus_smoothing_um=0.5,
            normal_half_window_um=1.5,
            normal_step_um=0.05,
            ridge_min_sbr=3.0,
            accepted_component_count=(1, 6),
            nominal_volume_um3=(100.0, 5000.0),
            minimum_sphericity=0.15,
            maximum_face_contact_fraction=0.1,
        ),
    )
    component = NuclearComponentQc(
        "N001",
        True,
        (),
        72900.0,
        800.0,
        0.91,
        (1.0, 2.0, 3.0),
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    monkeypatch.setattr(
        run_analysis,
        "_validated_registered_dapi",
        lambda _round, _config: np.ones((3, 3, 3), dtype=np.float64),
    )
    monkeypatch.setattr(
        "cyclic_stability.images.load_volume",
        lambda _path: np.ones((3, 3, 3), dtype=np.float64),
    )
    monkeypatch.setattr(
        "cyclic_stability.nuclear.adjudicate_nuclear_components",
        lambda *_args: SimpleNamespace(component_qc=(component,)),
    )
    monkeypatch.setattr(
        "cyclic_stability.nuclear.consensus_nuclear_surface",
        lambda *_args: SimpleNamespace(component_qc=(component,)),
    )

    def fail_surface(*_args):
        raise ValueError("surface normal interpolation is degenerate")

    monkeypatch.setattr(
        "cyclic_stability.nuclear.realize_nuclear_surfaces", fail_surface
    )

    payload = run_analysis._nuclear_inventory_payload(config, (cell,))

    assert payload["all_fields_ready"] is False
    assert payload["all_fields_surface_ready"] is False
    assert payload["all_fields_within_count_contract"] is True
    row = payload["cells"][0]
    assert row["candidate_count"] == 1
    assert row["accepted_count"] == 1
    assert row["components"][0]["nucleus_id"] == "N001"
    assert row["within_accepted_component_contract"] is True
    assert row["surface_ready"] is False
    assert row["ready"] is False
    assert row["surface_realization"] == {
        "error_type": "ValueError",
        "message": "surface normal interpolation is degenerate",
        "status": "failed",
    }


def test_nuclear_inventory_rejects_empty_selection(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """Treating an empty cohort filter as all-valid would be vacuous success."""
    inputs = []
    for name in (
        "analysis.yaml",
        "correspondence.csv",
        "qc-decisions.csv",
        "execution.yaml",
    ):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        inputs.append(path)
    config = SimpleNamespace(
        cells=(),
        qc_decisions=(),
        reference_acquisition_round=2,
        config_digest="empty-inventory",
    )
    profile = SimpleNamespace(mode="sequential", native_threads=1, scratch_root=None)
    monkeypatch.setattr(
        "cyclic_stability.config.load_analysis_config_bytes",
        lambda *_args, **_kwargs: config,
    )
    monkeypatch.setattr(
        "cyclic_stability.config.load_execution_profile_bytes",
        lambda *_args, **_kwargs: profile,
    )
    monkeypatch.setattr(
        "cyclic_stability.task.build_tasks",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("target task inventory must not be built")
        ),
    )

    result = run_analysis.main(
        [
            "--analysis-config",
            str(inputs[0]),
            "--correspondence",
            str(inputs[1]),
            "--qc-decisions",
            str(inputs[2]),
            "--execution-profile",
            str(inputs[3]),
            "--run-dir",
            str(tmp_path / "run"),
            "--cohort",
            "lamp1",
            "--nuclear-inventory-only",
        ]
    )

    assert result == 1
    assert "failed target-blind nuclear inventory" in capsys.readouterr().out
    terminal = json.loads(
        (tmp_path / "run/tasks/nuclear-inventory/terminal.json").read_text()
    )
    failure = json.loads(
        (
            tmp_path
            / "run/tasks/nuclear-inventory/attempts"
            / terminal["attempt_id"]
            / "failure.json"
        ).read_text()
    )
    assert failure["message"] == "no cells selected for nuclear inventory"
    assert not (tmp_path / "run" / "nuclear-inventory.json").exists()
