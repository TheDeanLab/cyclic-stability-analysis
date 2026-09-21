# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "matplotlib>=3.8,<4",
#   "numpy>=1.26,<3",
#   "pandas>=2.2,<3",
# ]
# ///
"""Render a publication-style, five-cell LAMP1 cyclic-stability summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COMPARISON_ENDPOINTS = (
    "lysosome_matched_fraction",
    "lysosome_lost_fraction",
    "lysosome_new_fraction",
    "lysosome_normalized_displacement_median",
)
MORPHOLOGY_ENDPOINTS = (
    "lysosome_object_count",
    "lysosome_volume_median",
    "lysosome_equivalent_diameter_median",
)
ALL_ROUNDS = (1, 2, 3, 4, 5)
COMPARISON_ROUNDS = ALL_ROUNDS
ROUND_COLORS = {
    1: "#CC79A7",
    2: "#0072B2",
    3: "#009E73",
    4: "#E69F00",
    5: "#D55E00",
}
SUMMARY_COLOR = "#332288"
LOST_COLOR = "#D55E00"
NEW_COLOR = "#009E73"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_830


def parse_args() -> argparse.Namespace:
    """Parse completed per-cell LAMP1 run roots and the requested output stem."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        required=True,
        help="Either one consolidated five-cell cohort root or five per-cell run roots.",
    )
    parser.add_argument("--output-stem", type=Path, required=True)
    return parser.parse_args()


def _completed_task(run_dir: Path) -> tuple[str, Path]:
    """Return the one completed LAMP1 task and its terminal-selected attempt."""
    candidates: list[tuple[str, Path]] = []
    for terminal_path in sorted((run_dir / "tasks").glob("lamp1-*/terminal.json")):
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        if terminal.get("status") != "complete":
            continue
        attempt_id = terminal.get("attempt_id")
        if not isinstance(attempt_id, str) or not attempt_id:
            raise ValueError(f"completed terminal lacks an attempt ID: {terminal_path}")
        task_dir = terminal_path.parent
        candidates.append(
            (task_dir.name.removeprefix("lamp1-"), task_dir / "attempts" / attempt_id)
        )
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one completed LAMP1 task under {run_dir}; found {len(candidates)}"
        )
    cell_id, attempt_dir = candidates[0]
    if not attempt_dir.is_dir():
        raise FileNotFoundError(attempt_dir)
    return cell_id, attempt_dir


def _completed_tasks(run_dir: Path) -> list[tuple[str, Path]]:
    """Return every terminal-selected completed LAMP1 task below one run root."""
    candidates: list[tuple[str, Path]] = []
    for terminal_path in sorted((run_dir / "tasks").glob("lamp1-*/terminal.json")):
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        if terminal.get("status") != "complete":
            continue
        attempt_id = terminal.get("attempt_id")
        if not isinstance(attempt_id, str) or not attempt_id:
            raise ValueError(f"completed terminal lacks an attempt ID: {terminal_path}")
        task_dir = terminal_path.parent
        attempt_dir = task_dir / "attempts" / attempt_id
        if not attempt_dir.is_dir():
            raise FileNotFoundError(attempt_dir)
        candidates.append((task_dir.name.removeprefix("lamp1-"), attempt_dir))
    return candidates


def _base_comparison_endpoint(endpoint: str) -> str | None:
    """Return the ensemble-independent comparison endpoint name, if supported."""
    for expected in COMPARISON_ENDPOINTS:
        if endpoint == expected or endpoint.startswith(f"{expected}_sigma_"):
            return expected
    return None


def _median_by_cell_round_endpoint(table: pd.DataFrame) -> pd.DataFrame:
    """Collapse detector settings before cohort-level aggregation."""
    grouped = (
        table.groupby(["cell_id", "acquisition_round", "endpoint"], as_index=False)
        .agg(value=("value", "median"), units=("units", "first"))
        .sort_values(["endpoint", "cell_id", "acquisition_round"])
    )
    if not np.all(np.isfinite(grouped["value"])):
        raise ValueError("LAMP1 ensemble medians must be finite")
    return grouped


def _required_rows(data: pd.DataFrame, cell_ids: set[str]) -> None:
    """Require exactly one cell-level value for every expected endpoint and round."""
    expected = {
        (cell_id, round_, endpoint)
        for cell_id in cell_ids
        for endpoint in COMPARISON_ENDPOINTS
        for round_ in COMPARISON_ROUNDS
    } | {
        (cell_id, round_, endpoint)
        for cell_id in cell_ids
        for endpoint in MORPHOLOGY_ENDPOINTS
        for round_ in ALL_ROUNDS
    }
    observed = {
        (str(row.cell_id), int(row.acquisition_round), str(row.endpoint))
        for row in data.itertuples(index=False)
    }
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if missing or unexpected or len(data) != len(expected):
        raise ValueError(
            "LAMP1 completed outputs have incomplete or duplicate endpoint/round rows: "
            f"missing={missing}; unexpected={unexpected}"
        )


def read_completed_cells(run_dirs: list[Path]) -> pd.DataFrame:
    """Load and median-collapse all valid endpoints from five completed cell runs."""
    if len(run_dirs) == 1:
        completed = _completed_tasks(run_dirs[0])
    elif len(run_dirs) == 5:
        completed = [_completed_task(run_dir) for run_dir in run_dirs]
    else:
        raise ValueError(
            "expected one consolidated cohort root or five per-cell LAMP1 run roots; "
            f"received {len(run_dirs)}"
        )
    tables: list[pd.DataFrame] = []
    seen_cells: set[str] = set()
    if len(completed) != 5:
        raise ValueError(f"expected five completed LAMP1 tasks; found {len(completed)}")
    for cell_id, attempt_dir in completed:
        if cell_id in seen_cells:
            raise ValueError(f"cell appears in more than one run directory: {cell_id}")
        seen_cells.add(cell_id)

        summary = pd.read_csv(attempt_dir / "summary.csv")
        summary = summary.loc[
            summary["cohort"].eq("lamp1")
            & summary["qc_included"].eq(True)
            & summary["comparison_set"].eq("round2-relative")
            & summary["nucleus_id"].eq("aggregate"),
            ["cell_id", "acquisition_round", "endpoint", "value", "units"],
        ].copy()
        summary["endpoint"] = summary["endpoint"].map(_base_comparison_endpoint)
        summary = summary.loc[summary["endpoint"].notna()].copy()

        ensemble = pd.read_csv(attempt_dir / "diagnostics" / "lamp1-ensemble.csv")
        ensemble = ensemble.loc[
            ensemble["cohort"].eq("lamp1")
            & ensemble["cell_id"].eq(cell_id)
            & ensemble["nucleus_id"].eq("aggregate")
            & ensemble["endpoint"].isin(MORPHOLOGY_ENDPOINTS),
            ["cell_id", "acquisition_round", "endpoint", "value", "units"],
        ].copy()
        tables.extend((summary, ensemble))
    collapsed = _median_by_cell_round_endpoint(pd.concat(tables, ignore_index=True))
    _required_rows(collapsed, seen_cells)
    return collapsed


def normalize_to_reference_round(table: pd.DataFrame) -> pd.DataFrame:
    """Normalize every cell to its own round-two morphology measurement."""
    normalized = table.copy()
    baseline = normalized.loc[normalized["acquisition_round"].eq(2)].set_index(
        "cell_id"
    )["value"]
    if baseline.index.has_duplicates or set(baseline.index) != set(
        normalized["cell_id"]
    ):
        raise ValueError(
            "every cell must have exactly one round-two morphology baseline"
        )
    normalized["value"] = [
        value / baseline[cell_id]
        for cell_id, value in zip(
            normalized["cell_id"], normalized["value"], strict=True
        )
    ]
    if not np.all(np.isfinite(normalized["value"])):
        raise ValueError("round-two-normalized LAMP1 values must be finite")
    return normalized


def bootstrap_mean_ci(table: pd.DataFrame) -> pd.DataFrame:
    """Return a percentile-bootstrap 95% CI across independent cells per round."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    records: list[dict[str, float | int]] = []
    for acquisition_round, group in table.groupby("acquisition_round", sort=True):
        values = group["value"].to_numpy(dtype=float)
        draws = rng.choice(
            values, size=(BOOTSTRAP_REPLICATES, len(values)), replace=True
        ).mean(axis=1)
        ci_low, ci_high = np.quantile(draws, [0.025, 0.975])
        records.append(
            {
                "acquisition_round": int(acquisition_round),
                "mean": float(values.mean()),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "n": len(values),
            }
        )
    return pd.DataFrame.from_records(records)


def _aggregate_endpoint(
    data: pd.DataFrame, endpoint: str, *, normalize: bool
) -> pd.DataFrame:
    """Aggregate one endpoint after optional within-cell round-two normalization."""
    endpoint_data = data.loc[data["endpoint"].eq(endpoint)].copy()
    if normalize:
        endpoint_data = normalize_to_reference_round(endpoint_data)
    aggregate = bootstrap_mean_ci(endpoint_data)
    aggregate.insert(0, "endpoint", endpoint)
    aggregate.insert(1, "round2_normalized", normalize)
    return aggregate


def _draw_single_panel(
    axis: plt.Axes,
    data: pd.DataFrame,
    aggregate: pd.DataFrame,
    title: str,
    units: str,
    *,
    normalize: bool = False,
) -> None:
    """Draw one endpoint's paired cell trajectories and cell-bootstrap mean CI."""
    if normalize:
        data = normalize_to_reference_round(data)
        axis.axhline(1.0, color="#a1a1aa", linewidth=1, linestyle="--", zorder=0)
    for _cell_id, group in data.groupby("cell_id", sort=True):
        group = group.sort_values("acquisition_round")
        axis.plot(
            group["acquisition_round"],
            group["value"],
            color="#d4d4d8",
            linewidth=1.1,
            zorder=1,
        )
        axis.scatter(
            group["acquisition_round"],
            group["value"],
            s=30,
            c=[ROUND_COLORS[int(value)] for value in group["acquisition_round"]],
            edgecolors="white",
            linewidths=0.7,
            zorder=2,
        )
    axis.fill_between(
        aggregate["acquisition_round"],
        aggregate["ci_low"],
        aggregate["ci_high"],
        color=SUMMARY_COLOR,
        alpha=0.20,
        linewidth=0,
        zorder=2.5,
    )
    axis.plot(
        aggregate["acquisition_round"],
        aggregate["mean"],
        color=SUMMARY_COLOR,
        linewidth=2.6,
        zorder=3,
    )
    axis.set_title(title)
    axis.set_xlabel("Acquisition round")
    axis.set_ylabel(units)
    axis.set_xticks(ALL_ROUNDS)
    axis.grid(axis="y", color="#e4e4e7", linewidth=0.8)


def _draw_lost_new_panel(
    axis: plt.Axes, data: pd.DataFrame, aggregate: pd.DataFrame
) -> None:
    """Draw retention-loss and novel-detection fractions together."""
    for endpoint, color, label in (
        ("lysosome_lost_fraction", LOST_COLOR, "lost"),
        ("lysosome_new_fraction", NEW_COLOR, "new"),
    ):
        endpoint_data = data.loc[data["endpoint"].eq(endpoint)]
        for _cell_id, group in endpoint_data.groupby("cell_id", sort=True):
            group = group.sort_values("acquisition_round")
            axis.plot(
                group["acquisition_round"],
                group["value"],
                color=color,
                alpha=0.22,
                linewidth=1.0,
                zorder=1,
            )
        summary = aggregate.loc[aggregate["endpoint"].eq(endpoint)]
        axis.fill_between(
            summary["acquisition_round"],
            summary["ci_low"],
            summary["ci_high"],
            color=color,
            alpha=0.18,
            linewidth=0,
            zorder=2,
        )
        axis.plot(
            summary["acquisition_round"],
            summary["mean"],
            color=color,
            linewidth=2.5,
            label=label,
            zorder=3,
        )
    axis.set_title("Lost and new fractions")
    axis.set_xlabel("Acquisition round")
    axis.set_ylabel("fraction")
    axis.set_xticks(ALL_ROUNDS)
    axis.grid(axis="y", color="#e4e4e7", linewidth=0.8)
    axis.legend(frameon=False, loc="best")


def render(run_dirs: list[Path], output_stem: Path) -> None:
    """Render audited LAMP1 cohort figures and aggregate data from five cell runs."""
    data = read_completed_cells(run_dirs)
    aggregate = pd.concat(
        (
            _aggregate_endpoint(data, "lysosome_matched_fraction", normalize=False),
            _aggregate_endpoint(data, "lysosome_lost_fraction", normalize=False),
            _aggregate_endpoint(data, "lysosome_new_fraction", normalize=False),
            _aggregate_endpoint(
                data, "lysosome_normalized_displacement_median", normalize=False
            ),
            _aggregate_endpoint(data, "lysosome_object_count", normalize=True),
            _aggregate_endpoint(data, "lysosome_volume_median", normalize=True),
            _aggregate_endpoint(
                data, "lysosome_equivalent_diameter_median", normalize=True
            ),
        ),
        ignore_index=True,
    )
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
        }
    )
    figure, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    figure.subplots_adjust(
        left=0.08, right=0.98, bottom=0.10, top=0.84, hspace=0.55, wspace=0.34
    )
    figure.suptitle(
        "Cyclic LAMP1 stability: five-cell cohort",
        x=0.08,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    figure.text(
        0.08,
        0.935,
        "Pale lines: cells after median aggregation across nine detector settings. "
        "Indigo line and band: mean ± bootstrap 95% CI across five cells.",
        ha="left",
        color="#3f3f46",
    )
    panel_specs = (
        ("lysosome_matched_fraction", "Matched fraction", "fraction", False),
        (
            "lysosome_normalized_displacement_median",
            "Median displacement",
            "resolution units",
            False,
        ),
        ("lysosome_object_count", "Object count", "relative to round 2", True),
        ("lysosome_volume_median", "Median volume", "relative to round 2", True),
        (
            "lysosome_equivalent_diameter_median",
            "Median equivalent diameter",
            "relative to round 2",
            True,
        ),
    )
    _draw_single_panel(
        axes[0, 0],
        data.loc[data["endpoint"].eq("lysosome_matched_fraction")],
        aggregate.loc[aggregate["endpoint"].eq("lysosome_matched_fraction")],
        "Matched fraction",
        "fraction",
    )
    _draw_lost_new_panel(axes[0, 1], data, aggregate)
    for axis, (endpoint, title, units, normalize) in zip(
        axes.flat[2:], panel_specs[1:], strict=True
    ):
        _draw_single_panel(
            axis,
            data.loc[data["endpoint"].eq(endpoint)],
            aggregate.loc[aggregate["endpoint"].eq(endpoint)],
            title,
            units,
            normalize=normalize,
        )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    aggregate.to_csv(output_stem.with_suffix(".aggregate.csv"), index=False)
    for suffix in (".png", ".pdf"):
        figure.savefig(output_stem.with_suffix(suffix), dpi=300, bbox_inches="tight")
    plt.close(figure)
    provenance = {
        "run_dirs": [str(path.resolve()) for path in run_dirs],
        "cell_ids": sorted(data["cell_id"].unique().tolist()),
        "cell_count": int(data["cell_id"].nunique()),
        "within_cell_ensemble_summary": "median",
        "ensemble_coordinates": {
            "smoothing_sigma_um": [0.35, 0.7, 1.4],
            "detection_sbr": [3.0, 4.0, 5.0],
        },
        "summary_statistic": "mean across five cells",
        "confidence_interval": {
            "method": "percentile bootstrap over biological cells per round",
            "level": 0.95,
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
        },
        "comparison_rounds": list(COMPARISON_ROUNDS),
        "morphology_rounds": list(ALL_ROUNDS),
        "round_2_role": "fixed reference and morphology-normalization baseline",
        "panels": [
            "matched fraction",
            "lost and new fractions",
            "median normalized displacement",
            "object count relative to round 2",
            "median volume relative to round 2",
            "median equivalent diameter relative to round 2",
        ],
    }
    output_stem.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    arguments = parse_args()
    render(
        [path.resolve() for path in arguments.run_dir], arguments.output_stem.resolve()
    )
