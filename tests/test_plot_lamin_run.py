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
