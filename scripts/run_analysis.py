# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "cy-exm[workflow] @ git+https://github.com/TheDeanLab/Cy-ExM.git@f6e0de36cd7f6b5e85f093884c2cf224225f4ec2",
#   "nd2==0.11.3",
#   "PyYAML>=6,<7",
#   "psutil>=5.9,<8",
#   "numpy>=1.26,<3",
#   "scipy>=1.12,<2",
#   "scikit-image>=0.22,<1",
#   "tifffile>=2025.1.10",
#   "pandas>=2.2,<3",
#   "matplotlib>=3.8,<4",
#   "seaborn>=0.13,<1",
#   "Jinja2>=3.1,<4",
# ]
# ///
"""Run restart-safe, target-blind-gated cyclic marker-cell analyses."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cyclic_stability.task import RetainedContent

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

_THREAD_ENVIRONMENT = (
    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _export_native_threads(native_threads: int) -> None:
    """Set native thread variables before importing scientific runtime modules."""
    if isinstance(native_threads, bool) or native_threads < 1:
        raise ValueError("native_threads must be a positive integer")
    value = str(native_threads)
    for name in _THREAD_ENVIRONMENT:
        os.environ[name] = value


@dataclass(frozen=True, slots=True)
class NuclearInventoryRequest:
    """Raw CLI-owned paths and selectors permitted before artifact admission."""

    analysis_config: Path
    correspondence: Path
    qc_decisions: Path
    execution_profile: Path
    publish_latest: bool
    cohorts: tuple[str, ...]
    cell_id: str | None


@dataclass(frozen=True, slots=True)
class NuclearInventoryDapi:
    """One retained byte owner in an admitted producer snapshot."""

    cohort: str
    cell_id: str
    acquisition_round: int
    content: RetainedContent


@dataclass(frozen=True, slots=True)
class NuclearInventoryContext:
    """Immutable scientific context returned by under-lock inventory snapshotting."""

    config: object
    cells: tuple[object, ...]
    dapi: tuple[NuclearInventoryDapi, ...]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-config", type=Path, required=True)
    parser.add_argument("--correspondence", type=Path, required=True)
    parser.add_argument("--qc-decisions", type=Path, required=True)
    parser.add_argument("--execution-profile", type=Path, required=True)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--run-dir", type=Path)
    destination.add_argument("--results-root", type=Path)
    parser.add_argument("--cohort", action="append", default=[])
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--cell")
    selection.add_argument("--pilot-first-accepted", action="store_true")
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--force-task", action="store_true")
    parser.add_argument("--nuclear-inventory-only", action="store_true")
    parser.add_argument(
        "--golgi-crop-preflight-only",
        action="store_true",
        help="write target-specific coarse crop evidence without running Golgi metrics",
    )
    parser.add_argument(
        "--progress-interval-s",
        type=float,
        default=30.0,
        help="seconds between running-task heartbeat messages (default: 30)",
    )
    return parser


def _pilot_cell(config, cohorts: tuple[str, ...]):
    if len(cohorts) != 1:
        raise ValueError("--pilot-first-accepted requires exactly one --cohort")
    decisions = {
        (value.cohort, value.cell_id, value.acquisition_round): value.decision
        for value in config.qc_decisions
    }
    eligible = []
    for cell in config.cells:
        if cohorts and cell.cohort not in cohorts:
            continue
        moving = [item for item in cell.rounds if item.acquisition_round != 2]
        if moving and all(
            decisions.get((item.cohort, item.cell_id, item.acquisition_round))
            == "accept"
            for item in moving
        ):
            eligible.append(cell)
    if not eligible:
        raise ValueError("no cell has accept decisions for every moving round")
    return min(eligible, key=lambda value: (value.cohort, value.cell_id))


def _write_selection(run_dir: Path, payload: dict[str, object]) -> None:
    """Durably publish the optional selection record without partial JSON."""
    target = run_dir / "selection.json"
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    directory = os.open(run_dir, os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _write_pretty_json(path: Path, payload: dict[str, object]) -> None:
    """Durably publish human-readable inventory JSON without a partial target."""
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=4, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _validated_registered_dapi(round_spec, config):
    """Decode one completed DAPI from its owned snapshot without resolving targets."""
    from cyclic_stability.cyexm_artifacts import snapshot_registered_dapi
    from cyclic_stability.images import load_volume

    snapshot = snapshot_registered_dapi(round_spec, config)
    try:
        with snapshot.content.open_binary() as source:
            return load_volume(source)
    finally:
        snapshot.close()


def _snapshot_nuclear_inventory(
    request: NuclearInventoryRequest,
    run_root: Path,
    task_root: Path,
):
    """Discover and parse one coherent DAPI-only inventory after task admission."""
    from cyclic_stability.config import (
        load_analysis_config_bytes,
        load_execution_profile_bytes,
    )
    from cyclic_stability.cyexm_artifacts import snapshot_registered_dapi
    from cyclic_stability.runner import initialize_run_from_bytes
    from cyclic_stability.task import (
        ArtifactSnapshot,
        ContentIdentity,
        configure_native_threads,
    )

    expected_task_root = run_root / "tasks" / "nuclear-inventory"
    if task_root != expected_task_root:
        raise ValueError(
            "inventory snapshot received an inconsistent canonical task root"
        )

    source_paths = (
        request.analysis_config,
        request.correspondence,
        request.qc_decisions,
        request.execution_profile,
    )
    source_snapshots = ContentIdentity.read_bytes_many(source_paths)
    source_identities = tuple(identity for identity, _payload in source_snapshots)
    source_payloads = tuple(payload for _identity, payload in source_snapshots)
    canonical_paths = tuple(Path(identity.path) for identity in source_identities)
    config = load_analysis_config_bytes(
        source_payloads[0],
        source_payloads[1],
        source_payloads[2],
        analysis_path=canonical_paths[0],
        correspondence_path=canonical_paths[1],
        qc_path=canonical_paths[2],
    )
    profile = load_execution_profile_bytes(
        source_payloads[3], source_path=canonical_paths[3]
    )
    configure_native_threads(profile.native_threads)
    cells = tuple(
        cell
        for cell in config.cells
        if (not request.cohorts or cell.cohort in request.cohorts)
        and (request.cell_id is None or cell.cell_id == request.cell_id)
    )
    if request.cell_id is not None and not cells:
        raise KeyError(
            "--cell does not match a configured cell in the requested cohort"
        )
    if not cells:
        raise ValueError("no cells selected for nuclear inventory")
    dapi_inputs: list[NuclearInventoryDapi] = []
    provenance: list[ContentIdentity] = list(source_identities)
    owned_inputs: list[RetainedContent] = []
    try:
        for cell in cells:
            inventory_rounds = _inventory_rounds(config, cell)
            if not inventory_rounds:
                raise ValueError(f"{cell.cohort}/{cell.cell_id} has no DAPI inventory")
            references = tuple(
                round_spec
                for round_spec in inventory_rounds
                if round_spec.acquisition_round == config.reference_acquisition_round
            )
            if len(references) != 1:
                raise ValueError(
                    f"{cell.cohort}/{cell.cell_id} lacks one configured DAPI reference"
                )
            reference = snapshot_registered_dapi(references[0], config)
            owned_inputs.append(reference.content)
            by_round = {references[0].acquisition_round: reference}
            for round_spec in inventory_rounds:
                if round_spec.acquisition_round == config.reference_acquisition_round:
                    continue
                snapshot = snapshot_registered_dapi(
                    round_spec,
                    config,
                    expected_fixed_geometry=reference.geometry,
                )
                owned_inputs.append(snapshot.content)
                by_round[round_spec.acquisition_round] = snapshot
            if any(
                snapshot.geometry != reference.geometry
                for snapshot in by_round.values()
            ):
                raise ValueError(
                    f"{cell.cohort}/{cell.cell_id} DAPI artifacts do not share "
                    "the exact fixed-grid geometry"
                )
            for round_spec in inventory_rounds:
                snapshot = by_round[round_spec.acquisition_round]
                dapi_inputs.append(
                    NuclearInventoryDapi(
                        cell.cohort,
                        cell.cell_id,
                        round_spec.acquisition_round,
                        snapshot.content,
                    )
                )
                provenance.extend(snapshot.identities)
        identities_by_path: dict[Path, ContentIdentity] = {}
        identities: list[ContentIdentity] = []
        for identity in provenance:
            path = Path(identity.path)
            prior = identities_by_path.get(path)
            if prior is not None and prior != identity:
                raise ValueError(f"conflicting DAPI inventory snapshots for {path}")
            if prior is None:
                identities_by_path[path] = identity
                identities.append(identity)
        initialize_run_from_bytes(
            run_root,
            snapshots={
                "analysis.yaml": source_payloads[0],
                "correspondence.csv": source_payloads[1],
                "qc-decisions.csv": source_payloads[2],
                "execution-profile.yaml": source_payloads[3],
            },
            config_digest=config.config_digest,
            results_root=run_root.parent if request.publish_latest else None,
        )
        return ArtifactSnapshot(
            context=NuclearInventoryContext(config, cells, tuple(dapi_inputs)),
            config_digest=config.config_digest,
            input_identities=tuple(identities),
            owned_inputs=tuple(owned_inputs),
            native_threads=profile.native_threads,
            scratch_root=profile.scratch_root,
        )
    except BaseException:
        for item in owned_inputs:
            item.close()
        raise


def _publish_nuclear_inventory(
    attempt_dir: Path, raw_context: object
) -> tuple[Path, ...]:
    """Publish inventory strictly from the admitted immutable producer context."""
    if not isinstance(raw_context, NuclearInventoryContext):
        raise TypeError("nuclear inventory producer requires its admitted context")
    dapi_sources = {
        (item.cohort, item.cell_id, item.acquisition_round): item.content
        for item in raw_context.dapi
    }
    payload = _nuclear_inventory_payload(
        raw_context.config, raw_context.cells, dapi_sources
    )
    path = attempt_dir / "nuclear-inventory.json"
    _write_pretty_json(path, payload)
    return (path,)


def _inventory_rounds(config, cell) -> tuple[object, ...]:
    """Return reference and target-blind-QC-accepted rounds for one field."""
    decisions = {
        (item.cohort, item.cell_id, item.acquisition_round): item.decision
        for item in config.qc_decisions
    }
    missing = [
        item.acquisition_round
        for item in cell.rounds
        if item.acquisition_round != config.reference_acquisition_round
        and (item.cohort, item.cell_id, item.acquisition_round) not in decisions
    ]
    if missing:
        raise ValueError(
            f"target-blind QC decisions are missing for moving rounds: {missing}"
        )
    return tuple(
        item
        for item in sorted(cell.rounds, key=lambda value: value.acquisition_round)
        if item.acquisition_round == config.reference_acquisition_round
        or decisions[(item.cohort, item.cell_id, item.acquisition_round)] == "accept"
    )


def _nuclear_inventory_payload(
    config,
    cells,
    dapi_sources: dict[tuple[str, str, int], object] | None = None,
) -> dict[str, object]:
    """Build complete DAPI component evidence without opening a target channel."""
    from cyclic_stability.images import load_volume
    from cyclic_stability.nuclear import (
        _apply_lamin_nuclear_exclusions,
        adjudicate_nuclear_components,
        realize_nuclear_surfaces,
    )

    minimum_count, maximum_count = config.nuclear.accepted_component_count
    rows: list[dict[str, object]] = []

    def load_dapi(item):
        if dapi_sources is None:
            return _validated_registered_dapi(item, config)
        source = dapi_sources[(item.cohort, item.cell_id, item.acquisition_round)]
        if not hasattr(source, "open_binary"):
            return load_volume(source)
        with source.open_binary() as handle:
            return load_volume(handle)

    for cell in sorted(cells, key=lambda value: (value.cohort, value.cell_id)):
        accepted_rounds = _inventory_rounds(config, cell)
        dapi = {item.acquisition_round: load_dapi(item) for item in accepted_rounds}
        adjudication = adjudicate_nuclear_components(
            dapi,
            config.voxel_spacing_xyz_um,
            config.nuclear,
            config.nominal_expansion_factor,
        )
        exclusions = tuple(
            rule
            for rule in getattr(config, "lamin_nuclear_exclusions", ())
            if cell.cohort == "lamin" and rule.cell_id == cell.cell_id
        )
        if exclusions:
            adjudication = _apply_lamin_nuclear_exclusions(adjudication, exclusions)
        components = tuple(asdict(item) for item in adjudication.component_qc)
        accepted_ids = [
            str(item["nucleus_id"]) for item in components if item["accepted"] is True
        ]
        accepted_count = len(accepted_ids)
        count_within_contract = minimum_count <= accepted_count <= maximum_count
        try:
            realize_nuclear_surfaces(adjudication, config.nuclear)
        except Exception as error:  # noqa: BLE001 - inventory records strict failure
            surface_realization = {
                "error_type": type(error).__name__,
                "message": str(error),
                "status": "failed",
            }
        else:
            surface_realization = {"status": "complete"}
        surface_ready = surface_realization == {"status": "complete"}
        rows.append(
            {
                "accepted_count": accepted_count,
                "accepted_ids": accepted_ids,
                "candidate_count": len(components),
                "cell_id": cell.cell_id,
                "cohort": cell.cohort,
                "components": list(components),
                "field_id": f"{cell.cohort}/{cell.cell_id}",
                "ready": count_within_contract and surface_ready,
                "surface_realization": surface_realization,
                "surface_ready": surface_ready,
                "within_accepted_component_contract": count_within_contract,
            }
        )
    return {
        "accepted_component_count_range": [minimum_count, maximum_count],
        "all_fields_ready": all(bool(item["ready"]) for item in rows),
        "all_fields_surface_ready": all(bool(item["surface_ready"]) for item in rows),
        "all_fields_within_count_contract": all(
            bool(item["within_accepted_component_contract"]) for item in rows
        ),
        "cells": rows,
    }


def _software_identity() -> str:
    """Return pinned interpreter, analysis-tree, Cy-ExM, and runtime versions."""
    from importlib.metadata import version

    dependencies = (
        "nd2",
        "numpy",
        "scipy",
        "scikit-image",
        "tifffile",
        "PyYAML",
        "psutil",
    )
    versions = {name: version(name) for name in dependencies}
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    tree = hashlib.sha256()
    for path in sorted((ROOT / "scripts").rglob("*.py")):
        relative = path.relative_to(ROOT).as_posix().encode()
        tree.update(len(relative).to_bytes(8, "big"))
        tree.update(relative)
        payload = path.read_bytes()
        tree.update(len(payload).to_bytes(8, "big"))
        tree.update(payload)
    return json.dumps(
        {
            "analysis_commit": commit,
            "analysis_tree_sha256": tree.hexdigest(),
            "cyexm_commit": "f6e0de36cd7f6b5e85f093884c2cf224225f4ec2",
            "python": platform.python_version(),
            "dependencies": versions,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def main(argv: list[str] | None = None) -> int:
    """Initialize one immutable run and execute its selected deterministic task set."""
    args = _parser().parse_args(argv)
    # Parse the profile through the lightweight configuration module, then set the
    # environment before runner/task imports can transitively import NumPy/SciPy.
    from cyclic_stability.config import load_execution_profile

    profile = load_execution_profile(args.execution_profile)
    _export_native_threads(profile.native_threads)

    from cyclic_stability.config import load_analysis_config
    from cyclic_stability.runner import initialize_run, run_tasks
    from cyclic_stability.task import (
        ArtifactTask,
        ContentIdentity,
        build_tasks,
        configure_native_threads,
        run_artifact_task,
        run_cell_task,
    )

    cohorts = tuple(args.cohort)
    known_cohorts = {"golgi", "lamin", "lamp1"}
    if any(cohort not in known_cohorts for cohort in cohorts):
        raise ValueError("--cohort must name a configured cohort")
    if args.nuclear_inventory_only:
        if args.pilot_first_accepted or args.task_index is not None or args.force_task:
            raise ValueError(
                "--nuclear-inventory-only cannot be combined with pilot, task-index, "
                "or force selection"
            )
        if args.run_dir is not None:
            run_dir = args.run_dir
            results_root = None
        else:
            assert args.results_root is not None
            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            run_dir = (
                args.results_root
                / f"{timestamp}-nuclear-inventory-{uuid.uuid4().hex[:8]}"
            )
            results_root = args.results_root
        request = NuclearInventoryRequest(
            args.analysis_config,
            args.correspondence,
            args.qc_decisions,
            args.execution_profile,
            results_root is not None,
            cohorts,
            args.cell,
        )
        print(
            f"progress: starting target-blind nuclear inventory ({len(cohorts) or 'all'} cohorts)",
            file=sys.stderr,
            flush=True,
        )
        record = run_artifact_task(
            ArtifactTask(
                key="nuclear-inventory",
                software_identity=_software_identity(),
                snapshot=lambda canonical_run, canonical_task: (
                    _snapshot_nuclear_inventory(
                        request,
                        canonical_run,
                        canonical_task,
                    )
                ),
                producer=_publish_nuclear_inventory,
            ),
            run_dir,
        )
        if record.status != "complete":
            print(f"failed target-blind nuclear inventory: {record.attempt_dir}")
            return 1
        inventory_outputs = [
            Path(identity.path)
            for identity in record.output_identities
            if Path(identity.path).name == "nuclear-inventory.json"
        ]
        if len(inventory_outputs) != 1:
            raise ValueError("complete nuclear inventory lacks one exact JSON output")
        path = inventory_outputs[0]
        payload = json.loads(path.read_text(encoding="utf-8"))
        print(
            f"{'reused' if record.reused else 'wrote'} target-blind nuclear inventory "
            f"for {len(payload['cells'])} cells: {path}"
        )
        return 0 if payload["all_fields_ready"] is True else 1
    config = load_analysis_config(
        args.analysis_config, args.correspondence, args.qc_decisions
    )
    configure_native_threads(profile.native_threads)
    if profile.mode == "slurm" and args.task_index is None:
        raise ValueError("SLURM execution requires --task-index")
    selected = None
    if args.cell is not None:
        selected = next(
            (
                cell
                for cell in config.cells
                if cell.cell_id == args.cell and (not cohorts or cell.cohort in cohorts)
            ),
            None,
        )
    if args.cell is not None and selected is None:
        raise KeyError(
            "--cell does not match a configured cell in the requested cohort"
        )
    if args.golgi_crop_preflight_only:
        if (
            args.nuclear_inventory_only
            or args.pilot_first_accepted
            or args.task_index is not None
        ):
            raise ValueError(
                "--golgi-crop-preflight-only cannot be combined with inventory, pilot, or task-index selection"
            )
        if cohorts != ("golgi",) or selected is None:
            raise ValueError(
                "--golgi-crop-preflight-only requires --cohort golgi and exactly one --cell"
            )
        from cyclic_stability.cyexm_artifacts import resolve_round_artifacts
        from cyclic_stability.golgi import (
            _golgi_crop_plan_from_artifacts,
            _require_compact_crop,
            _write_crop_preflight,
        )
        from cyclic_stability.nuclear import _accepted_rounds

        artifacts = tuple(
            resolve_round_artifacts(round_spec) for round_spec in selected.rounds
        )
        accepted = _accepted_rounds(selected, config)
        plan = _golgi_crop_plan_from_artifacts(
            {artifact.acquisition_round: artifact for artifact in artifacts},
            accepted,
            config,
        )
        preflight_dir = (args.run_dir or args.results_root) / "golgi-crop-preflight"
        _write_crop_preflight(preflight_dir, selected, accepted, plan)
        print(
            "wrote Golgi crop preflight "
            f"({plan.crop.retained_fraction:.1%} retained): {preflight_dir}"
        )
        _require_compact_crop(plan, preflight_dir)
        return 0
    if args.pilot_first_accepted:
        selected = _pilot_cell(config, cohorts)
    tasks = build_tasks(
        config,
        cohorts,
        cells=(selected,) if selected is not None else None,
    )
    run_input_identities = tuple(
        ContentIdentity.from_path(path)
        for path in (
            args.analysis_config,
            args.correspondence,
            args.qc_decisions,
            args.execution_profile,
        )
    )
    software_identity = _software_identity()
    tasks = tuple(
        replace(
            task,
            input_identities=(*task.input_identities, *run_input_identities),
            software_identity=software_identity,
        )
        for task in tasks
    )
    if args.task_index is not None:
        if profile.mode != "slurm":
            raise ValueError("--task-index requires a slurm execution profile")
        if args.cell is not None or args.pilot_first_accepted:
            raise ValueError("--task-index cannot be combined with cell selection")
        if args.task_index < 0 or args.task_index >= len(tasks):
            raise IndexError("--task-index is outside deterministic task inventory")
        tasks = (tasks[args.task_index],)
    if args.force_task:
        tasks = tuple(replace(task, force=True) for task in tasks)
    if args.run_dir is not None:
        run_dir = args.run_dir
        results_root = None
    else:
        assert args.results_root is not None
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_dir = args.results_root / f"{timestamp}-{config.config_digest[:12]}"
        results_root = args.results_root
    initialize_run(
        run_dir,
        analysis_config=args.analysis_config,
        correspondence=args.correspondence,
        qc_decisions=args.qc_decisions,
        execution_profile=args.execution_profile,
        config_digest=config.config_digest,
        results_root=results_root,
    )
    if selected is not None:
        _write_selection(
            run_dir,
            {
                "mode": "pilot-first-accepted",
                "cohort": selected.cohort,
                "cell_id": selected.cell_id,
            },
        )
    if not tasks:
        _write_selection(
            run_dir, {"status": "unavailable", "reason": "no schedulable tasks"}
        )
        print("unavailable: no schedulable tasks")
        return 0
    if profile.mode == "slurm":
        record = run_cell_task(
            replace(
                tasks[0],
                native_threads=profile.native_threads,
                scratch_root=profile.scratch_root,
            ),
            run_dir,
        )
        print(f"{record.status}: {record.key}")
        return 0 if record.status == "complete" else 1
    record = run_tasks(
        tasks, profile, run_dir, progress_interval_s=args.progress_interval_s
    )
    failures = sum(item.status != "complete" for item in record.tasks)
    print(
        f"ran {len(record.tasks)} tasks; failed={failures}; compact={record.compact_table}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
