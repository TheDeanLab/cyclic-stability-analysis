# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "matplotlib>=3.8,<4",
#   "numpy>=1.26,<3",
#   "pandas>=2.2,<3",
# ]
# ///
"""Regression checks for the Lamin summary-plot annotations."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).parents[1] / "scripts" / "plot_lamin_run.py"
SPEC = importlib.util.spec_from_file_location("plot_lamin_run", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
plot_lamin_run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot_lamin_run)


def test_completed_run_annotations_do_not_report_historical_failures() -> None:
    tasks = pd.DataFrame(
        {
            "cell_id": ["Cell1", "Cell2", "Cell3", "Cell4", "Cell5"],
            "status": ["complete"] * 5,
        }
    )

    assert (
        plot_lamin_run.task_outcome_note(tasks)
        == "Complete: Cell1, Cell2, Cell3, Cell4, Cell5\nFailed: none"
    )
    assert (
        plot_lamin_run.cohort_scope_note(tasks)
        == "This is a complete 5-cell cohort view."
    )


def test_bootstrap_mean_ci_collapses_for_identical_nuclei() -> None:
    data = pd.DataFrame(
        {
            "acquisition_round": [2, 2, 2, 3, 3, 3],
            "value": [1.5, 1.5, 1.5, 2.0, 2.0, 2.0],
        }
    )

    aggregate = plot_lamin_run.bootstrap_mean_ci(data)

    assert aggregate["mean"].tolist() == [1.5, 2.0]
    assert aggregate["ci_low"].tolist() == [1.5, 2.0]
    assert aggregate["ci_high"].tolist() == [1.5, 2.0]
    assert aggregate["n"].tolist() == [3, 3]


if __name__ == "__main__":
    test_completed_run_annotations_do_not_report_historical_failures()
    test_bootstrap_mean_ci_collapses_for_identical_nuclei()


def test_plot_uses_terminal_attempt_and_never_counts_aggregate(tmp_path):
    import json

    task = tmp_path / "tasks/lamin-Cell1"
    for name, value in (("obsolete", 99.0), ("selected", 1.0)):
        attempt = task / "attempts" / name
        attempt.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "cell_id": "Cell1",
                    "nucleus_id": nucleus_id,
                    "acquisition_round": 2,
                    "endpoint": "lamin_fwhm_median",
                    "value": value,
                    "qc_included": True,
                    "comparison_set": "matched-settings",
                }
                for nucleus_id in ("N001", "aggregate")
            ]
        ).to_csv(attempt / "summary.csv", index=False)
    (task / "terminal.json").write_text(
        json.dumps({"status": "complete", "attempt_id": "selected"})
    )
    tasks = pd.DataFrame([{"key": "lamin-Cell1", "status": "complete"}])
    result = plot_lamin_run.completed_summary(tmp_path, tasks)
    assert result.nucleus_id.tolist() == ["N001"]
    assert result.value.tolist() == [1.0]


def test_exposure_normalization_preserves_constant_signal_rate():
    data = pd.DataFrame(
        {
            "cell_id": ["Cell2"] * 3,
            "nucleus_id": ["N001"] * 3,
            "acquisition_round": [1, 2, 3],
            "target_exposure_ms": [150.0, 200.0, 200.0],
            "value": [1500.0, 2000.0, 4000.0],
        }
    )
    assert plot_lamin_run.normalize_intensity(data).value.tolist() == [1.0, 1.0, 2.0]


def test_round_one_included_without_overriding_qc(tmp_path):
    import json

    task = tmp_path / "tasks/lamin-Cell2"
    attempt = task / "attempts/selected"
    attempt.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "cell_id": "Cell2",
                "nucleus_id": nucleus,
                "acquisition_round": 1,
                "endpoint": "lamin_fwhm_median",
                "value": 1.2,
                "qc_included": qc,
                "comparison_set": "round-1-exposure-sensitivity",
            }
            for nucleus, qc in [("N001", True), ("N002", False), ("aggregate", True)]
        ]
    ).to_csv(attempt / "summary.csv", index=False)
    (task / "terminal.json").write_text(
        json.dumps({"status": "complete", "attempt_id": "selected"})
    )
    tasks = pd.DataFrame([{"key": "lamin-Cell2", "status": "complete"}])
    result = plot_lamin_run.completed_summary(tmp_path, tasks)
    assert result.nucleus_id.tolist() == ["N001"]
    assert result.acquisition_round.tolist() == [1]


def test_missing_exposure_rejected(tmp_path):
    import pytest

    (tmp_path / "config").mkdir()
    pd.DataFrame(
        [
            {
                "cohort": "lamin",
                "cell_id": "Cell2",
                "acquisition_round": 2,
                "target_exposure_ms": 200,
                "target_laser_percent": 40,
            }
        ]
    ).to_csv(tmp_path / "config/correspondence.csv", index=False)
    summary = pd.DataFrame([{"cell_id": "Cell2", "acquisition_round": 1}])
    with pytest.raises(ValueError, match="missing or invalid target exposure"):
        plot_lamin_run.acquisition_settings(tmp_path, summary)
