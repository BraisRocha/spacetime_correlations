"""
Finalize one grid_p50 submission: merge the scratch files into output/.

Runs on the submit host once every job of a submission has finished. DAGMan
calls it through the ``SCRIPT POST`` line of the submission's .dag, but it
only ever looks at the filesystem, never at the queue, so it can equally be
run by hand:

    bash condor/grid_p50/finalize_grid_p50.sh <submission_id>

Everything a running submission produces lives in scratch/grid_p50/<id>/.
This script turns it into the four files that are worth keeping, writes them
to output/montecarlo/grid_p50/<id>/, and then deletes the scratch directory:

    pvalues_lambda_merged.pkl    (durations, intensities, pvalues)
    pvalues_poisson_merged.pkl   same layout
    metadata.json                the fields shared by every job, plus one
                                 row per job for those that differ
    run.log                      a short report of the submission

The grid axes come from params.txt, the copy of the parameter file recorded
at submission time, not from the per-job files that happen to be present: a
whole duration that failed has to show up as a row of NaN, not as a missing
row that would silently shift the axis.

The scratch directory is removed only when every cell arrived. A submission
that is still sitting in scratch/ is one that needs looking at, and it holds
everything needed to do so: the tracebacks, the per-job logs, and the
pickles of the cells that did work, so that rerunning the failed ones and
running this again rebuilds the full grid. ``--keep-scratch`` keeps it in
any case.

Exit status is 0 for a complete grid and 1 when anything is missing, so that
DAGMan reports a submission needing attention as a failed node.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np


# Per-job files written by run_grid_p50.py, keyed by the job id in the name.
JOB_FILE_PATTERNS = {
    "lambda": re.compile(r"^pvalues_lambda_job(\d+)\.pkl$"),
    "poisson": re.compile(r"^pvalues_poisson_job(\d+)\.pkl$"),
    "metadata": re.compile(r"^metadata_job(\d+)\.json$"),
}

STATS = ("lambda", "poisson")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge one grid_p50 submission from scratch/ into output/.",
    )
    parser.add_argument(
        "--scratch-dir",
        type=Path,
        required=True,
        help="scratch/grid_p50/<submission_id>",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="output/montecarlo/grid_p50/<submission_id>",
    )
    parser.add_argument(
        "--keep-scratch",
        action="store_true",
        help="Do not delete the scratch directory, even on a complete grid.",
    )
    return parser.parse_args()


# ----------------------------------------------------------------------
# The grid that was submitted
# ----------------------------------------------------------------------
def read_grid(
    params_file: Path,
) -> tuple[np.ndarray, np.ndarray, dict[int, tuple[int, int]], list[str]]:
    """
    Read params.txt and return the full grid axes plus the cell of each job.

    Job ids are line numbers: HTCondor gives Process 0 to the first line of
    the queue file. Both the axes and the mapping therefore describe the grid
    as submitted, whether or not every job produced anything.
    """
    lines = [line for line in params_file.read_text().splitlines() if line.strip()]
    points = [
        (float(parts[0]), float(parts[1]))
        for parts in (line.split() for line in lines)
    ]

    durations = np.array(sorted({d for d, _ in points}), dtype=float)
    intensities = np.array(sorted({i for _, i in points}), dtype=float)

    d_index = {value: k for k, value in enumerate(durations)}
    i_index = {value: k for k, value in enumerate(intensities)}
    job_cells = {
        job_id: (d_index[d], i_index[i]) for job_id, (d, i) in enumerate(points)
    }
    return durations, intensities, job_cells, lines


def scan_job_files(data_dir: Path) -> dict[str, dict[int, Path]]:
    """Return {kind: {job_id: path}} for the per-job files in ``data_dir``."""
    found: dict[str, dict[int, Path]] = {kind: {} for kind in JOB_FILE_PATTERNS}
    for path in data_dir.iterdir():
        for kind, pattern in JOB_FILE_PATTERNS.items():
            match = pattern.match(path.name)
            if match is not None:
                found[kind][int(match.group(1))] = path
                break
    return found


# ----------------------------------------------------------------------
# Merging onto the full grid
# ----------------------------------------------------------------------
def build_grid(
    durations: np.ndarray,
    intensities: np.ndarray,
    job_cells: dict[int, tuple[int, int]],
    job_files: dict[int, Path],
) -> np.ndarray | None:
    """Place the per-job p-values on the full grid, leaving missing cells NaN."""
    grid = None

    for job_id, path in sorted(job_files.items()):
        with path.open("rb") as fh:
            _, _, pvalues = pickle.load(fh)
        cell = np.asarray(pvalues, dtype=float).reshape(-1)

        if grid is None:
            grid = np.full((len(durations), len(intensities), cell.size), np.nan)
        elif cell.size != grid.shape[2]:
            raise ValueError(
                f"{path.name} holds {cell.size} simulations but the grid has "
                f"{grid.shape[2]}: the submission mixes different n_simulations."
            )

        i, j = job_cells[job_id]
        grid[i, j, :] = cell

    return grid


# ----------------------------------------------------------------------
# Condensing the per-job metadata into a single file
# ----------------------------------------------------------------------
def flatten(mapping: dict, prefix: str = "") -> dict:
    """Flatten nested dicts into {"flare.duration_days": value} form."""
    flat = {}
    for key, value in mapping.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, f"{name}."))
        else:
            flat[name] = value
    return flat


def unflatten(flat: dict) -> dict:
    """Inverse of ``flatten``, to keep the common fields readable."""
    nested: dict = {}
    for name, value in flat.items():
        parts = name.split(".")
        node = nested
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return nested


def condense_metadata(metadata: dict[int, dict]) -> tuple[dict, list[dict]]:
    """
    Fold the per-job metadata into shared fields and one row per job.

    A field is shared when every job reports the same value for it. The rest
    (the grid point itself, the runtime, the per-cell percentiles) differ and
    are kept per job. The shared fields are returned nested as they were
    written, so metadata.json reads like a per-job metadata file with the
    varying fields taken out.
    """
    if not metadata:
        return {}, []

    flats = {job_id: flatten(meta) for job_id, meta in metadata.items()}
    job_ids = sorted(flats)
    first = flats[job_ids[0]]

    shared_keys = set.intersection(*(set(flats[job_id]) for job_id in job_ids))
    common_flat = {
        key: first[key]
        for key in sorted(shared_keys)
        if key != "job_id"
        and all(flats[job_id][key] == first[key] for job_id in job_ids)
    }

    per_job = [
        {
            "job_id": job_id,
            **{
                key: value
                for key, value in flats[job_id].items()
                if key != "job_id" and key not in common_flat
            },
        }
        for job_id in job_ids
    ]
    return unflatten(common_flat), per_job


def summarise(per_job: list[dict]) -> dict:
    """Submission-wide numbers worth keeping once the per-job files are gone."""

    def column(name: str) -> list:
        return [row[name] for row in per_job if row.get(name) is not None]

    summary: dict = {"n_jobs": len(per_job)}

    runtimes = column("runtime_seconds")
    if runtimes:
        summary["runtime_seconds"] = {
            "min": float(np.min(runtimes)),
            "median": float(np.median(runtimes)),
            "max": float(np.max(runtimes)),
            "total": float(np.sum(runtimes)),
        }

    completed = column("n_simulations_completed")
    if completed:
        summary["n_simulations_total"] = int(np.sum(completed))

    zero_flare = column("n_zero_flare")
    if zero_flare:
        summary["n_zero_flare_total"] = int(np.sum(zero_flare))

    return summary


# ----------------------------------------------------------------------
# Which jobs did not finish cleanly
# ----------------------------------------------------------------------
def find_suspect_jobs(
    grids: dict[str, np.ndarray | None],
    job_cells: dict[int, tuple[int, int]],
    metadata: dict[int, dict],
    scratch_dir: Path,
) -> dict[int, list[str]]:
    """Return {job_id: [reasons]} for every job that needs a human."""
    suspects: dict[int, list[str]] = {}

    def flag(job_id: int, reason: str) -> None:
        suspects.setdefault(job_id, []).append(reason)

    for job_id, (i, j) in job_cells.items():
        for stat_name, grid in grids.items():
            if grid is None or np.isnan(grid[i, j, :]).any():
                flag(job_id, f"no {stat_name} p-values")

        meta = metadata.get(job_id)
        if meta is None:
            flag(job_id, "no metadata")
        else:
            requested = meta.get("n_simulations_requested")
            completed = meta.get("n_simulations_completed")
            if requested is not None and completed != requested:
                flag(job_id, f"only {completed}/{requested} simulations")

        err_file = scratch_dir / f"grid_p50_{job_id}.err"
        if err_file.is_file() and err_file.stat().st_size > 0:
            flag(job_id, f"non-empty {err_file.name}")

    return suspects


# ----------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    scratch_dir: Path = args.scratch_dir
    output_dir: Path = args.output_dir
    data_dir = scratch_dir / "data"
    params_file = scratch_dir / "params.txt"

    if not scratch_dir.is_dir():
        print(f"ERROR: no scratch directory at {scratch_dir}")
        print("A finalized submission has none: its results are in output/.")
        return 1
    if not params_file.is_file():
        print(f"ERROR: no parameter grid at {params_file}")
        return 1
    if not data_dir.is_dir():
        print(f"ERROR: no job output at {data_dir}: no job wrote anything")
        return 1

    durations, intensities, job_cells, params = read_grid(params_file)
    found = scan_job_files(data_dir)

    metadata = {
        job_id: json.loads(path.read_text())
        for job_id, path in sorted(found["metadata"].items())
    }

    grids = {
        stat_name: build_grid(
            durations=durations,
            intensities=intensities,
            job_cells=job_cells,
            job_files=found[stat_name],
        )
        for stat_name in STATS
    }

    if all(grid is None for grid in grids.values()):
        print(f"ERROR: no p-value files in {data_dir}")
        return 1

    suspects = find_suspect_jobs(grids, job_cells, metadata, scratch_dir)
    complete = not suspects

    # ------------------------------------------------------------------
    # Write what is worth keeping
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)

    for stat_name, grid in grids.items():
        if grid is None:
            continue
        with (output_dir / f"pvalues_{stat_name}_merged.pkl").open("wb") as fh:
            pickle.dump((durations, intensities, grid), fh)

    common, per_job = condense_metadata(metadata)
    summary = summarise(per_job)

    with (output_dir / "metadata.json").open("w") as fh:
        json.dump(
            {
                **common,
                "submission_id": output_dir.name,
                "finalized_at": datetime.now().isoformat(timespec="seconds"),
                "n_jobs_submitted": len(job_cells),
                "summary": summary,
                "per_job": per_job,
            },
            fh,
            indent=2,
        )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    report: list[str] = []

    def say(line: str = "") -> None:
        """Print for the DAGMan POST log, and keep for run.log."""
        print(line)
        report.append(line)

    say("=== grid_p50 finalize report ===")
    say(f"Submission ID   : {output_dir.name}")
    say(f"Finalized at    : {datetime.now().isoformat(timespec='seconds')}")
    say(f"Results         : {output_dir}")
    say()
    say(
        f"Grid            : {len(durations)} durations x {len(intensities)} "
        f"intensities = {len(job_cells)} cells"
    )
    say(f"Cells delivered : {len(job_cells) - len(suspects)} / {len(job_cells)}")

    for stat_name, grid in grids.items():
        if grid is not None:
            say(
                f"Merged {stat_name:<8} : pvalues_{stat_name}_merged.pkl  "
                f"shape {grid.shape}"
            )

    expected_n = common.get("expected_n")
    if expected_n is not None:
        say(f"expected_n      : {expected_n:.2f} events in window")

    runtime = summary.get("runtime_seconds")
    if runtime is not None:
        say(
            f"Runtime per job : min {runtime['min']:.1f} s | "
            f"median {runtime['median']:.1f} s | max {runtime['max']:.1f} s | "
            f"total {runtime['total'] / 3600:.2f} h"
        )

    n_sims = summary.get("n_simulations_total")
    n_zero = summary.get("n_zero_flare_total")
    if n_sims:
        say(f"Simulations     : {n_sims} total")
        if n_zero is not None:
            say(
                f"Zero-flare      : {n_zero} / {n_sims} realizations "
                f"({100.0 * n_zero / n_sims:.2f}%)"
            )

    say(f"Metadata        : metadata.json ({len(per_job)} per-job rows)")
    say()

    # ------------------------------------------------------------------
    # The scratch directory
    # ------------------------------------------------------------------
    if not complete:
        missing_path = output_dir / "MISSING_JOBS.txt"
        with missing_path.open("w") as fh:
            fh.write(f"# grid_p50 submission {output_dir.name}\n")
            fh.write("# job_id\treasons\t|\tduration_days intensity seed\n")
            for job_id in sorted(suspects):
                point = params[job_id] if 0 <= job_id < len(params) else "-"
                fh.write(f"{job_id}\t{'; '.join(suspects[job_id])}\t|\t{point}\n")

        say(f"Suspect jobs    : {len(suspects)}, listed in {missing_path.name}")
        for job_id in sorted(suspects)[:10]:
            say(f"    job {job_id}: {'; '.join(suspects[job_id])}")
        if len(suspects) > 10:
            say(f"    ... and {len(suspects) - 10} more")
        say()
        say(f"Scratch kept    : {scratch_dir}")
        say("The merged grid holds NaN for those cells, which the plotting")
        say("scripts handle. Rerun them into the same scratch directory and")
        say("run this script again to rebuild the grid without the holes.")
        say("Status          : INCOMPLETE")
    elif args.keep_scratch:
        say(f"Scratch kept    : {scratch_dir} (--keep-scratch)")
        say("Status          : OK")
    else:
        say(f"Scratch removed : {scratch_dir}")
        say("Status          : OK")

    (output_dir / "run.log").write_text("\n".join(report) + "\n")

    if complete and not args.keep_scratch:
        shutil.rmtree(scratch_dir)

    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
