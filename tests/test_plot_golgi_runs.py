"""Regression tests for the Golgi cohort summary plot."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).parents[1] / "scripts" / "plot_golgi_runs.py"
SPEC = importlib.util.spec_from_file_location("plot_golgi_runs", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
plot_golgi_runs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot_golgi_runs)


def _write_completed_golgi_run(root: Path, cell_id: str) -> None:
    """Write one terminal-selected five-round Golgi result for CLI-layout tests."""
    task_dir = root / "tasks" / f"golgi-{cell_id}"
    attempt_dir = task_dir / "attempts" / "attempt-001"
    attempt_dir.mkdir(parents=True)
    (task_dir / "terminal.json").write_text(
        json.dumps({"status": "complete", "attempt_id": "attempt-001"}),
        encoding="utf-8",
    )
    summary_rows = []
    raw_rows = []
    for round_ in range(1, 6):
        for endpoint, *_rest in plot_golgi_runs.PANELS:
            if endpoint == "raw_integrated_per_ms":
                continue
            summary_rows.append(
                {
                    "cohort": "golgi",
                    "qc_included": True,
                    "cell_id": cell_id,
                    "acquisition_round": round_,
                    "endpoint": endpoint,
                    "value": float(round_),
                }
            )
        raw_rows.append(
            {
                "measurement_status": "measured",
                "nucleus_id": "aggregate",
                "cell_id": cell_id,
                "acquisition_round": round_,
                "integrated_per_ms": float(round_),
            }
        )
    pd.DataFrame(summary_rows).to_csv(attempt_dir / "summary.csv", index=False)
    pd.DataFrame(raw_rows).to_csv(attempt_dir / "raw-signal.csv", index=False)


def test_normalize_to_reference_round_is_paired_by_cell() -> None:
    """A cohort mean must not replace each cell's own round-two baseline."""
    table = pd.DataFrame(
        {
            "cell_id": ["Cell1", "Cell1", "Cell2", "Cell2"],
            "acquisition_round": [2, 3, 2, 3],
            "value": [10.0, 15.0, 100.0, 200.0],
        }
    )

    normalized = plot_golgi_runs.normalize_to_reference_round(table)

    assert normalized["value"].tolist() == [1.0, 1.5, 1.0, 2.0]


def test_bootstrap_mean_ci_collapses_for_identical_cells() -> None:
    """Identical biological replicates have a zero-width bootstrap interval."""
    table = pd.DataFrame(
        {
            "acquisition_round": [2, 2, 2, 3, 3, 3],
            "value": [1.0, 1.0, 1.0, 1.5, 1.5, 1.5],
        }
    )

    aggregate = plot_golgi_runs.bootstrap_mean_ci(table)

    assert aggregate["mean"].tolist() == [1.0, 1.5]
    assert aggregate["ci_low"].tolist() == [1.0, 1.5]
    assert aggregate["ci_high"].tolist() == [1.0, 1.5]
    assert aggregate["n"].tolist() == [3, 3]


def test_absolute_golgi_panels_do_not_force_round_two_to_one() -> None:
    """Absolute morphology and signal panels must retain their physical scale."""
    normalization = {
        endpoint: normalize for endpoint, *_rest, normalize in plot_golgi_runs.PANELS
    }

    assert normalization["golgi_volume_um3_median"] is False
    assert normalization["golgi_compactness_median"] is False
    assert normalization["raw_integrated_per_ms"] is False


def test_consolidated_root_loads_five_completed_golgi_tasks(tmp_path: Path) -> None:
    """One new cohort root must be sufficient input for the final plotter."""
    root = tmp_path / "consolidated"
    for index in range(5):
        _write_completed_golgi_run(root, f"Cell{index + 1}")

    data = plot_golgi_runs.read_completed_cells([root])

    assert set(data["cell_id"]) == {f"Cell{index + 1}" for index in range(5)}
