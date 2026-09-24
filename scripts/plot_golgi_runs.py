# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "matplotlib>=3.8,<4",
#   "numpy>=1.26,<3",
#   "pandas>=2.2,<3",
# ]
# ///
"""Render a publication-style, five-cell Golgi cyclic-stability summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PANELS = (
    ("golgi_gradient_similarity_median", "Gradient similarity", "correlation", False),
    ("golgi_jaccard_median", "Jaccard overlap", "fraction", False),
    ("golgi_volume_um3_median", "Golgi volume", "µm³", False),
    ("golgi_compactness_median", "Golgi compactness", "fraction", False),
    (
        "golgi_nuclear_distance_p50_um_median",
        "Median perinuclear distance",
        "µm",
        False,
    ),
    (
        "raw_integrated_per_ms",
        "Raw integrated Golgi signal",
        "background-corrected counts / ms",
        False,
    ),
)
ROUND_COLORS = {
    1: "#CC79A7",  # exposure-sensitivity round: Okabe-Ito magenta
    2: "#0072B2",
    3: "#009E73",
    4: "#E69F00",
    5: "#D55E00",
}
SUMMARY_COLOR = "#332288"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_829


def parse_args() -> argparse.Namespace:
    """Parse Golgi run roots and the requested output stem."""
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
    """Return the one completed Golgi task and its immutable attempt directory."""
    candidates: list[tuple[str, Path]] = []
    for terminal_path in sorted((run_dir / "tasks").glob("golgi-*/terminal.json")):
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        if terminal.get("status") != "complete":
            continue
        attempt_id = terminal.get("attempt_id")
        if not isinstance(attempt_id, str) or not attempt_id:
            raise ValueError(f"completed terminal lacks an attempt ID: {terminal_path}")
        task_dir = terminal_path.parent
        candidates.append(
            (task_dir.name.removeprefix("golgi-"), task_dir / "attempts" / attempt_id)
        )
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one completed Golgi task under {run_dir}; found {len(candidates)}"
        )
    cell_id, attempt_dir = candidates[0]
    if not attempt_dir.is_dir():
        raise FileNotFoundError(attempt_dir)
    return cell_id, attempt_dir


def _completed_tasks(run_dir: Path) -> list[tuple[str, Path]]:
    """Return every terminal-selected completed Golgi task below one run root."""
    candidates: list[tuple[str, Path]] = []
    for terminal_path in sorted((run_dir / "tasks").glob("golgi-*/terminal.json")):
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
        candidates.append((task_dir.name.removeprefix("golgi-"), attempt_dir))
    return candidates


def _raw_signal_summary(raw_signal: pd.DataFrame) -> pd.DataFrame:
    """Collapse the nine segmentation settings to one raw-signal value per cell/round."""
    measured = raw_signal.loc[
        raw_signal["measurement_status"].eq("measured")
        & raw_signal["nucleus_id"].eq("aggregate")
    ].copy()
    if measured.empty:
        raise ValueError("raw-signal table has no measured aggregate rows")
    collapsed = (
        measured.groupby(["cell_id", "acquisition_round"], as_index=False)[
            "integrated_per_ms"
        ]
        .median()
        .rename(columns={"integrated_per_ms": "value"})
    )
    collapsed["endpoint"] = "raw_integrated_per_ms"
    return collapsed


def read_completed_cells(run_dirs: list[Path]) -> pd.DataFrame:
    """Load selected summary and raw-signal endpoints from unique completed cells."""
    requested_endpoints = {
        panel[0] for panel in PANELS if panel[0].startswith("golgi_")
    }
    tables: list[pd.DataFrame] = []
    seen_cells: set[str] = set()
    if len(run_dirs) == 1:
        completed = _completed_tasks(run_dirs[0])
    elif len(run_dirs) == 5:
        completed = [_completed_task(run_dir) for run_dir in run_dirs]
    else:
        raise ValueError(
            "expected one consolidated cohort root or five per-cell Golgi run roots; "
            f"received {len(run_dirs)}"
        )
    if len(completed) != 5:
        raise ValueError(f"expected five completed Golgi tasks; found {len(completed)}")
    for cell_id, attempt_dir in completed:
        if cell_id in seen_cells:
            raise ValueError(f"cell appears in more than one run directory: {cell_id}")
        seen_cells.add(cell_id)
        summary = pd.read_csv(attempt_dir / "summary.csv")
        selected = summary.loc[
            summary["cohort"].eq("golgi")
            & summary["qc_included"].eq(True)
            & summary["endpoint"].isin(requested_endpoints),
            ["cell_id", "acquisition_round", "endpoint", "value"],
        ].copy()
        raw_signal = _raw_signal_summary(pd.read_csv(attempt_dir / "raw-signal.csv"))
        tables.extend((selected, raw_signal))
    if len(seen_cells) != 5:
        raise ValueError(
            f"expected five unique completed cells; found {len(seen_cells)}"
        )
    combined = pd.concat(tables, ignore_index=True)
    expected = {(panel[0], round_) for panel in PANELS for round_ in range(1, 6)}
    observed = {
        (str(row.endpoint), int(row.acquisition_round))
        for row in combined.itertuples(index=False)
    }
    missing = sorted(expected - observed)
    if missing:
        raise ValueError(
            f"completed Golgi outputs are missing endpoint/round rows: {missing}"
        )
    return combined


def normalize_to_reference_round(table: pd.DataFrame) -> pd.DataFrame:
    """Normalize each cell to its own round-two value before cohort aggregation."""
    normalized = table.copy()
    baseline = normalized.loc[normalized["acquisition_round"].eq(2)].set_index(
        "cell_id"
    )["value"]
    if baseline.index.has_duplicates or set(baseline.index) != set(
        normalized["cell_id"]
    ):
        raise ValueError("every cell must have exactly one round-two baseline")
    normalized["value"] = [
        value / baseline[cell_id]
        for cell_id, value in zip(
            normalized["cell_id"], normalized["value"], strict=True
        )
    ]
    if not np.all(np.isfinite(normalized["value"])):
        raise ValueError("reference-normalized values must be finite")
    return normalized


def bootstrap_mean_ci(table: pd.DataFrame) -> pd.DataFrame:
    """Return a percentile-bootstrap 95% CI across biological cells per round."""
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


def _aggregate_panels(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate each explicitly selected endpoint for plotting and CSV export."""
    records: list[pd.DataFrame] = []
    for endpoint, title, unit, normalize in PANELS:
        endpoint_data = data.loc[data["endpoint"].eq(endpoint)].copy()
        if normalize:
            endpoint_data = normalize_to_reference_round(endpoint_data)
        aggregate = bootstrap_mean_ci(endpoint_data)
        aggregate.insert(0, "endpoint", endpoint)
        aggregate.insert(1, "title", title)
        aggregate.insert(2, "units", unit)
        aggregate.insert(3, "reference_normalized", normalize)
        records.append(aggregate)
    return pd.concat(records, ignore_index=True)


def render(run_dirs: list[Path], output_stem: Path) -> None:
    """Render plots and audited aggregate artifacts for five completed cell runs."""
    data = read_completed_cells(run_dirs)
    aggregate = _aggregate_panels(data)
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
        "Cyclic Golgi stability: five-cell cohort",
        x=0.08,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    figure.text(
        0.08,
        0.935,
        "Points and pale lines: individual cells. Indigo line and band: mean ± bootstrap 95% CI. "
        "Round 1 is an exposure-sensitivity condition.",
        ha="left",
        color="#3f3f46",
    )
    for axis, (endpoint, title, unit, normalize) in zip(axes.flat, PANELS, strict=True):
        endpoint_data = data.loc[data["endpoint"].eq(endpoint)].copy()
        if normalize:
            endpoint_data = normalize_to_reference_round(endpoint_data)
            axis.axhline(1.0, color="#a1a1aa", linewidth=1, linestyle="--", zorder=0)
        for _cell_id, group in endpoint_data.groupby("cell_id", sort=True):
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
        summary = aggregate.loc[aggregate["endpoint"].eq(endpoint)]
        axis.fill_between(
            summary["acquisition_round"],
            summary["ci_low"],
            summary["ci_high"],
            color=SUMMARY_COLOR,
            alpha=0.20,
            linewidth=0,
            zorder=2.5,
        )
        axis.plot(
            summary["acquisition_round"],
            summary["mean"],
            color=SUMMARY_COLOR,
            linewidth=2.6,
            zorder=3,
        )
        axis.set_title(title)
        axis.set_xlabel("Acquisition round")
        axis.set_ylabel(unit)
        axis.set_xticks([1, 2, 3, 4, 5])
        axis.grid(axis="y", color="#e4e4e7", linewidth=0.8)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    aggregate.to_csv(output_stem.with_suffix(".aggregate.csv"), index=False)
    for suffix in (".png", ".pdf"):
        figure.savefig(output_stem.with_suffix(suffix), dpi=300, bbox_inches="tight")
    plt.close(figure)
    provenance = {
        "run_dirs": [str(path.resolve()) for path in run_dirs],
        "cell_ids": sorted(data["cell_id"].unique().tolist()),
        "summary_statistic": "mean across five cells",
        "confidence_interval": {
            "method": "percentile bootstrap over biological cells per round",
            "level": 0.95,
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
        },
        "within_cell_normalization": None,
        "round_1_interpretation": "exposure-sensitivity condition",
        "panels": [
            {
                "endpoint": endpoint,
                "title": title,
                "units": unit,
                "reference_normalized": normalize,
            }
            for endpoint, title, unit, normalize in PANELS
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
