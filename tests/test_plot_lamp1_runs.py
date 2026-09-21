"""Regression tests for the LAMP1 cohort summary plot."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).parents[1] / "scripts" / "plot_lamp1_runs.py"
SPEC = importlib.util.spec_from_file_location("plot_lamp1_runs", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
plot_lamp1_runs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot_lamp1_runs)


def _write_completed_lamp1_run(root: Path, cell_id: str, offset: float) -> None:
    """Write one complete, two-setting LAMP1 task with known per-cell medians."""
    key = f"lamp1-{cell_id}"
    attempt_id = "attempt-001"
    task_dir = root / "tasks" / key
    attempt_dir = task_dir / "attempts" / attempt_id
    attempt_dir.mkdir(parents=True)
    (task_dir / "terminal.json").write_text(
        json.dumps({"status": "complete", "attempt_id": attempt_id}), encoding="utf-8"
    )

    comparison_rows: list[dict[str, object]] = []
    for round_ in range(1, 6):
        for sigma, sbr, value in ((0.35, 3.0, 0.4), (0.70, 4.0, 0.8)):
            for endpoint, scale in (
                ("lysosome_matched_fraction", 1.0),
                ("lysosome_lost_fraction", 0.5),
                ("lysosome_new_fraction", 0.25),
                ("lysosome_normalized_displacement_median", 2.0),
            ):
                comparison_rows.append(
                    {
                        "acquisition_round": round_,
                        "cell_id": cell_id,
                        "cohort": "lamp1",
                        "comparison_set": "round2-relative",
                        "endpoint": f"{endpoint}_sigma_{sigma:g}_sbr_{sbr:g}",
                        "nucleus_id": "aggregate",
                        "qc_included": True,
                        "units": "fraction",
                        "value": (value + offset) * scale,
                    }
                )
    pd.DataFrame(comparison_rows).to_csv(attempt_dir / "summary.csv", index=False)

    morphology_rows: list[dict[str, object]] = []
    for round_ in range(1, 6):
        for sigma, sbr, value in ((0.35, 3.0, 8.0), (0.70, 4.0, 12.0)):
            for endpoint, scale, units in (
                ("lysosome_object_count", 1.0, "count"),
                ("lysosome_volume_median", 2.0, "um3"),
                ("lysosome_equivalent_diameter_median", 0.5, "um"),
            ):
                morphology_rows.append(
                    {
                        "cohort": "lamp1",
                        "cell_id": cell_id,
                        "acquisition_round": round_,
                        "endpoint": endpoint,
                        "ensemble_name": f"sigma={sigma:g};sbr={sbr:g}",
                        "ensemble_value": sbr,
                        "sigma_um": sigma,
                        "threshold_sbr": sbr,
                        "value": (value + offset) * scale,
                        "units": units,
                        "nucleus_id": "aggregate",
                    }
                )
    diagnostics = attempt_dir / "diagnostics"
    diagnostics.mkdir()
    pd.DataFrame(morphology_rows).to_csv(
        diagnostics / "lamp1-ensemble.csv", index=False
    )


def test_render_collapses_detector_settings_within_cell_before_bootstrap(
    tmp_path: Path,
) -> None:
    """Pooling detector settings as biological replicates must change the aggregate."""
    run_dirs = []
    for index in range(5):
        root = tmp_path / f"cell-{index + 1}"
        _write_completed_lamp1_run(root, f"Cell{index + 1}", float(index))
        run_dirs.append(root)

    output_stem = tmp_path / "plots" / "lamp1-overview"
    plot_lamp1_runs.render(run_dirs, output_stem)

    aggregate = pd.read_csv(output_stem.with_suffix(".aggregate.csv"))
    matched = aggregate.loc[
        aggregate["endpoint"].eq("lysosome_matched_fraction")
        & aggregate["acquisition_round"].eq(1)
    ].iloc[0]
    # Each cell median is 0.6 + its offset, hence the five-cell mean is 2.6.
    assert matched["mean"] == 2.6
    assert matched["n"] == 5
    assert output_stem.with_suffix(".png").is_file()
    assert output_stem.with_suffix(".pdf").is_file()
    provenance = json.loads(output_stem.with_suffix(".provenance.json").read_text())
    assert provenance["within_cell_ensemble_summary"] == "median"
    assert provenance["cell_count"] == 5


def test_render_accepts_one_consolidated_five_cell_cohort_root(tmp_path: Path) -> None:
    """A cohort root with five completed terminal-selected tasks is valid input."""
    cohort_root = tmp_path / "cohort"
    for index in range(5):
        _write_completed_lamp1_run(cohort_root, f"Cell{index + 1}", float(index))

    output_stem = tmp_path / "plots" / "consolidated-overview"
    plot_lamp1_runs.render([cohort_root], output_stem)

    aggregate = pd.read_csv(output_stem.with_suffix(".aggregate.csv"))
    assert set(aggregate["n"]) == {5}
    provenance = json.loads(output_stem.with_suffix(".provenance.json").read_text())
    assert provenance["cell_count"] == 5
    assert provenance["run_dirs"] == [str(cohort_root.resolve())]
