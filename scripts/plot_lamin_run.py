# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "matplotlib>=3.8,<4",
#   "numpy>=1.26,<3",
#   "pandas>=2.2,<3",
# ]
# ///
"""Render a transparent, publication-style overview of one Lamin analysis run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ENDPOINTS = (
    ("lamin_surface_coverage", "Surface coverage", "fraction", False),
    ("lamin_continuity", "Continuity", "fraction", False),
    ("lamin_fwhm_median", "Ridge FWHM", "µm", False),
    (
        "lamin_integrated_ridge_intensity_median",
        "Integrated ridge intensity",
        "exposure-normalized, relative to round 2",
        True,
    ),
)
STATUS_COLORS = {"complete": "#1b9e77", "failed": "#d95f02"}
ROUND_COLORS = {1: "#438078", 2: "#355c7d", 3: "#6c5b7b", 4: "#c06c84", 5: "#f67280"}
SUMMARY_COLOR = "#2563eb"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_829


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    return parser.parse_args()


def task_table(run_dir: Path) -> pd.DataFrame:
    tasks = pd.read_csv(run_dir / "compact-tasks.csv")
    tasks["cell_id"] = tasks["key"].str.removeprefix("lamin-")
    return tasks


def task_outcome_note(tasks: pd.DataFrame) -> str:
    """Summarize the current task terminal states for the status panel."""
    completed = ", ".join(tasks.loc[tasks.status.eq("complete"), "cell_id"])
    failed = ", ".join(tasks.loc[tasks.status.eq("failed"), "cell_id"]) or "none"
    return f"Complete: {completed}\nFailed: {failed}"


def cohort_scope_note(tasks: pd.DataFrame) -> str:
    """Describe whether the plotted task table covers the complete cohort."""
    completed = int(tasks.status.eq("complete").sum())
    total = len(tasks)
    if completed == total:
        return f"This is a complete {total}-cell cohort view."
    return "This is a partial cohort view, not a cohort estimate."


def completed_summary(run_dir: Path, tasks: pd.DataFrame) -> pd.DataFrame:
    tables: list[pd.DataFrame] = []
    for row in tasks.loc[tasks["status"].eq("complete")].itertuples():
        task_dir = run_dir / "tasks" / row.key
        terminal = json.loads((task_dir / "terminal.json").read_text())
        attempt_id = terminal.get("attempt_id")
        if (
            terminal.get("status") != "complete"
            or not isinstance(attempt_id, str)
            or not attempt_id
            or Path(attempt_id).name != attempt_id
            or attempt_id in {".", ".."}
        ):
            raise ValueError(f"invalid completed terminal record for {row.key}")
        summary_path = task_dir / "attempts" / attempt_id / "summary.csv"
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        tables.append(pd.read_csv(summary_path))
    if not tables:
        raise ValueError("the requested run has no completed Lamin tasks")
    summary = pd.concat(tables, ignore_index=True)
    return summary.loc[
        summary["qc_included"].eq(True)
        & summary["comparison_set"].isin(
            ["matched-settings", "round-1-exposure-sensitivity"]
        )
        & summary["endpoint"].isin([value[0] for value in ENDPOINTS])
        & summary["nucleus_id"].str.fullmatch(r"N[0-9]+", na=False)
    ].copy()


def acquisition_settings(run_dir: Path, summary: pd.DataFrame) -> pd.DataFrame:
    """Read exposures from the immutable run configuration, validating every key."""
    settings = pd.read_csv(run_dir / "config" / "correspondence.csv")
    settings = settings.loc[
        settings["cohort"].eq("lamin"),
        ["cell_id", "acquisition_round", "target_exposure_ms", "target_laser_percent"],
    ]
    keys = summary[["cell_id", "acquisition_round"]].drop_duplicates()
    settings = keys.merge(
        settings, on=["cell_id", "acquisition_round"], how="left", validate="one_to_one"
    )
    exposure = settings["target_exposure_ms"]
    if not (np.isfinite(exposure) & exposure.gt(0)).all():
        raise ValueError("missing or invalid target exposure in run configuration")
    if (
        settings["target_laser_percent"].isna().any()
        or settings["target_laser_percent"].nunique() != 1
    ):
        raise ValueError("exposure normalization requires matched target laser power")
    return settings


def normalize_intensity(table: pd.DataFrame) -> pd.DataFrame:
    """Normalize intensity per millisecond to the same nucleus in round 2."""
    normalized = table.copy()
    normalized["value"] = normalized["value"] / normalized["target_exposure_ms"]
    baseline = normalized.loc[normalized["acquisition_round"].eq(2)].set_index(
        ["cell_id", "nucleus_id"]
    )["value"]
    if (
        baseline.index.has_duplicates
        or not (np.isfinite(baseline) & baseline.gt(0)).all()
    ):
        raise ValueError(
            "round-2 intensity baselines must be unique, finite and positive"
        )
    keys = list(zip(normalized["cell_id"], normalized["nucleus_id"], strict=True))
    normalized["value"] = [
        value / baseline[key]
        for key, value in zip(keys, normalized["value"], strict=True)
    ]
    return normalized


def bootstrap_mean_ci(table: pd.DataFrame) -> pd.DataFrame:
    """Return the mean and percentile bootstrap 95% CI for every round."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    records: list[dict[str, float | int]] = []
    for acquisition_round, group in table.groupby("acquisition_round", sort=True):
        values = group["value"].to_numpy(dtype=float)
        resampled_means = rng.choice(
            values, size=(BOOTSTRAP_REPLICATES, len(values)), replace=True
        ).mean(axis=1)
        ci_low, ci_high = np.quantile(resampled_means, [0.025, 0.975])
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


def render(run_dir: Path, output_stem: Path) -> None:
    tasks = task_table(run_dir)
    summary = completed_summary(run_dir, tasks)
    settings = acquisition_settings(run_dir, summary)
    summary = summary.merge(
        settings, on=["cell_id", "acquisition_round"], validate="many_to_one"
    )
    counts = (
        summary.drop_duplicates(["cell_id", "nucleus_id", "acquisition_round"])
        .groupby("acquisition_round")
        .size()
    )
    exposure_by_round = settings.groupby("acquisition_round")[
        "target_exposure_ms"
    ].unique()
    exposure_note = "; ".join(
        f"R{r}: {'/'.join(f'{v:g}' for v in values)} ms"
        for r, values in exposure_by_round.items()
    )
    endpoint_counts = []

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
        }
    )
    figure, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    figure.subplots_adjust(
        left=0.07, right=0.98, bottom=0.1, top=0.84, hspace=0.55, wspace=0.35
    )
    figure.suptitle(
        "Cyclic Lamin stability: completed-run overview",
        x=0.01,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    figure.text(
        0.01,
        0.935,
        "Acquisition rounds 1–5; points are QC-included nuclei and lines link "
        "the same nucleus across rounds.",
        ha="left",
        color="#3f3f46",
    )

    status_axis = axes[0, 0]
    status_axis.barh(
        tasks["cell_id"],
        np.ones(len(tasks)),
        color=[STATUS_COLORS[value] for value in tasks["status"]],
        edgecolor="white",
        height=0.72,
    )
    status_axis.set_xlim(0, 1)
    status_axis.set_xticks([])
    status_axis.set_title("Task outcome")
    for index, row in enumerate(tasks.itertuples()):
        status_axis.text(
            0.5,
            index,
            row.status,
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
        )
    status_axis.invert_yaxis()
    status_axis.text(
        0,
        -0.22,
        task_outcome_note(tasks),
        transform=status_axis.transAxes,
        va="top",
        fontsize=9,
        color="#3f3f46",
    )

    for axis, (endpoint, title, unit, normalize) in zip(
        axes.flat[1:5], ENDPOINTS, strict=True
    ):
        data = summary.loc[summary.endpoint.eq(endpoint)].copy()
        if normalize:
            data = normalize_intensity(data)
            axis.axhline(1, color="#a1a1aa", linewidth=1, linestyle="--", zorder=0)
        for (_cell, _nucleus), group in data.groupby(
            ["cell_id", "nucleus_id"], sort=True
        ):
            group = group.sort_values("acquisition_round")
            axis.plot(
                group["acquisition_round"],
                group["value"],
                color="#d1d5db",
                linewidth=1.1,
                zorder=1,
            )
            axis.scatter(
                group["acquisition_round"],
                group["value"],
                s=28,
                c=[ROUND_COLORS[round_] for round_ in group["acquisition_round"]],
                edgecolors="white",
                linewidths=0.7,
                zorder=2,
            )
        aggregate = bootstrap_mean_ci(data)
        endpoint_counts.extend(
            {
                "endpoint": endpoint,
                "acquisition_round": int(row.acquisition_round),
                "n": int(row.n),
            }
            for row in aggregate.itertuples()
        )
        axis.fill_between(
            aggregate["acquisition_round"],
            aggregate["ci_low"],
            aggregate["ci_high"],
            color=SUMMARY_COLOR,
            alpha=0.22,
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
        axis.set_ylabel(unit)
        axis.set_xticks([1, 2, 3, 4, 5])
        axis.grid(axis="y", color="#e4e4e7", linewidth=0.8)
        axis.text(
            0.02,
            0.96,
            "grey: nucleus; blue: mean with 95% CI\n"
            + "n = "
            + ", ".join(str(n) for n in aggregate["n"]),
            transform=axis.transAxes,
            va="top",
            fontsize=8,
            color="#52525b",
        )

    axes[1, 2].axis("off")
    included = summary[["cell_id", "nucleus_id"]].drop_duplicates()
    axes[1, 2].text(
        0,
        0.95,
        "Interpretation boundary",
        transform=axes[1, 2].transAxes,
        va="top",
        fontweight="bold",
        fontsize=12,
    )
    axes[1, 2].text(
        0,
        0.78,
        f"{cohort_scope_note(tasks)}\n\n"
        f"Quantitative panels use {len(included)} QC-included nuclei from "
        f"{len(tasks.loc[tasks.status.eq('complete')])}/5 completed cells.\n\n"
        "Nuclei by round: "
        + ", ".join(f"R{r}={n}" for r, n in counts.items())
        + "\n\n"
        + exposure_note
        + "\n"
        "Intensity is normalized by exposure, then to round 2.\n"
        "Assumes linear detector response; signal detectability may differ.\n"
        "Existing registration QC exclusions are retained.\n\n"
        + f"Run: {run_dir.name}",
        transform=axes[1, 2].transAxes,
        va="top",
        fontsize=10,
        color="#3f3f46",
        linespacing=1.5,
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        figure.savefig(output_stem.with_suffix(suffix), dpi=300, bbox_inches="tight")
    plt.close(figure)

    provenance = {
        "run_dir": str(run_dir.resolve()),
        "source_task_table": str((run_dir / "compact-tasks.csv").resolve()),
        "completed_cells": tasks.loc[tasks.status.eq("complete"), "cell_id"].tolist(),
        "failed_cells": tasks.loc[tasks.status.eq("failed"), "cell_id"].tolist(),
        "quantitative_comparison_set": "all-acquisition-rounds",
        "source_comparison_sets": sorted(summary["comparison_set"].unique().tolist()),
        "acquisition_rounds": sorted(int(r) for r in counts.index),
        "sample_sizes": endpoint_counts,
        "acquisition_settings_source": str(
            (run_dir / "config/correspondence.csv").resolve()
        ),
        "acquisition_settings": settings.to_dict(orient="records"),
        "intensity_normalization": {
            "formula": "(I_round / exposure_ms_round) / (I_round2 / exposure_ms_round2)",
            "reference_acquisition_round": 2,
            "assumption": "linear detector response; matched target laser power",
        },
        "qc_decisions_source": str((run_dir / "config/qc-decisions.csv").resolve()),
        "qc_exclusions_preserved": True,
        "quantitative_endpoints": [item[0] for item in ENDPOINTS],
        "summary_statistic": "mean",
        "aggregate_rows_excluded": True,
        "included_nuclei": included.to_dict(orient="records"),
        "confidence_interval": {
            "method": "percentile bootstrap over QC-included nuclei per round",
            "level": 0.95,
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
        },
    }
    output_stem.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    args = parse_args()
    render(args.run_dir.resolve(), args.output_stem.resolve())
