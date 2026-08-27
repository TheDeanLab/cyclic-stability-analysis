"""Target-blind production inventory behavior for the analysis entry point."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from cyclic_stability.models import NuclearConfig
from cyclic_stability.nuclear import NuclearComponentQc
from scripts import run_analysis


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
    profile = SimpleNamespace(mode="sequential", native_threads=1)
    monkeypatch.setattr(
        "cyclic_stability.config.load_analysis_config", lambda *_args: config
    )
    monkeypatch.setattr(
        "cyclic_stability.config.load_execution_profile", lambda _path: profile
    )
    monkeypatch.setattr(
        "cyclic_stability.task.build_tasks",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("target task inventory must not be built")
        ),
    )
    monkeypatch.setattr(
        run_analysis,
        "_validated_registered_dapi",
        lambda round_spec: (
            tmp_path / f"round-{round_spec.acquisition_round}-405-DAPI.tif"
        ),
        raising=False,
    )
    opened: list[str] = []

    def load_volume(path: Path) -> np.ndarray:
        opened.append(path.name)
        if "DAPI" not in path.name or "target" in path.name.lower():
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
    inventory = run_dir / "nuclear-inventory.json"
    payload = json.loads(inventory.read_text(encoding="utf-8"))
    assert payload["accepted_component_count_range"] == [1, 6]
    assert payload["all_fields_within_contract"] is True
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
            "within_accepted_component_contract": True,
        }
    ]
    assert inventory.read_text(encoding="utf-8") == (
        json.dumps(payload, indent=4, sort_keys=True) + "\n"
    )
    assert not list(run_dir.glob(".nuclear-inventory.json.*.tmp"))


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
        lambda _round: tmp_path / "round-2-405-DAPI.tif",
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

    payload = run_analysis._nuclear_inventory_payload(config, (cell,))

    assert payload == {
        "accepted_component_count_range": [1, 6],
        "all_fields_within_contract": False,
        "cells": [
            {
                "accepted_count": 0,
                "accepted_ids": [],
                "candidate_count": 0,
                "cell_id": "Field-Z",
                "cohort": "lamin",
                "components": [],
                "field_id": "lamin/Field-Z",
                "within_accepted_component_contract": False,
            }
        ],
    }
