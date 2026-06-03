"""
Merge per-job ``results_job{N}.npz`` files from one submission directory
into a single ``results_merged.npz`` file.

Each per-job file is expected to contain the same set of keys. The merged
file holds, for every key, a 1D array of length ``n_jobs`` where index ``i``
corresponds to ``job_id == i`` (jobs are sorted numerically by their id).

Usage (from the repo root, with the package importable in the active env)::

    python -m spacetimecorr.io.merge <submission_dir>
    python -m spacetimecorr.io.merge <submission_dir> --output-name custom.npz

``<submission_dir>`` may contain the per-job files directly or inside a
``data/`` subdirectory; both layouts are auto-detected. Job ids must be
contiguous from 0 (``results_job0.npz``, ``results_job1.npz``, ...).
"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import numpy as np


_RESULTS_RE = re.compile(r"^results_job(\d+)\.npz$")


def _find_job_files(submission_dir: Path) -> list[tuple[int, Path]]:
    jobs: list[tuple[int, Path]] = []
    for path in submission_dir.iterdir():
        match = _RESULTS_RE.match(path.name)
        if match is not None:
            jobs.append((int(match.group(1)), path))
    jobs.sort(key=lambda x: x[0])
    return jobs


def merge_results(
    submission_dir: Path,
    output_name: str = "results_merged.npz",
) -> Path:
    """
    Merge ``results_job{N}.npz`` files into one.

    Looks for per-job files inside ``submission_dir/data/`` when that
    subdirectory exists, otherwise falls back to ``submission_dir`` itself.
    The merged file is written to the same directory as the per-job files.

    Parameters
    ----------
    submission_dir : Path
        Run directory (may contain a ``data/`` subdirectory).
    output_name : str, optional
        Filename for the merged output.

    Returns
    -------
    out_path : Path
        Path to the merged file.
    """
    submission_dir = Path(submission_dir)
    data_dir = submission_dir / "data"
    submission_dir = data_dir if data_dir.is_dir() else submission_dir

    jobs = _find_job_files(submission_dir)
    if not jobs:
        raise FileNotFoundError(
            f"No results_job*.npz files found in {submission_dir}"
        )

    job_ids = [j for j, _ in jobs]
    if job_ids != list(range(len(job_ids))):
        missing = sorted(set(range(max(job_ids) + 1)) - set(job_ids))
        if missing:
            raise ValueError(
                f"Job ids in {submission_dir} are not contiguous from 0: "
                f"missing {missing}"
            )

    first = np.load(jobs[0][1])
    keys = list(first.files)
    first.close()

    stacked: dict[str, np.ndarray] = {key: [] for key in keys}
    for _, path in jobs:
        with np.load(path) as data:
            if list(data.files) != keys:
                raise ValueError(
                    f"Key mismatch between {jobs[0][1].name} and {path.name}: "
                    f"{keys} vs {list(data.files)}"
                )
            for key in keys:
                stacked[key].append(data[key])

    merged = {key: np.stack(values) for key, values in stacked.items()}
    merged["job_id"] = np.array(job_ids, dtype=np.int64)

    out_path = submission_dir / output_name
    np.savez_compressed(out_path, **merged)
    return out_path


def merge_grid_pvalues(
    submission_dir: Path,
    stat_name: str,
    output_name: str | None = None,
) -> Path:
    """
    Merge per-job ``pvalues_{stat}_job{N}.pkl`` files from a 2D grid scan
    into a single pickle indexed by (duration, intensity, simulation).

    Each per-job file holds a tuple ``(durations, intensities, pvalues)``
    where ``durations`` and ``intensities`` are length-1 arrays naming
    the grid cell and ``pvalues`` has shape ``(1, 1, n_simulations)``.

    The merged file holds a tuple ``(durations, intensities, pvalues)``
    where ``durations`` and ``intensities`` are the sorted unique axis
    values and ``pvalues`` has shape
    ``(n_durations, n_intensities, n_simulations)``.

    Parameters
    ----------
    submission_dir : Path
        Run directory (may contain a ``data/`` subdirectory).
    stat_name : str
        Either ``"lambda"`` or ``"poisson"``.
    output_name : str, optional
        Filename for the merged output. Defaults to
        ``"pvalues_{stat_name}_merged.pkl"``.

    Returns
    -------
    out_path : Path
        Path to the merged file.
    """
    if stat_name not in ("lambda", "poisson"):
        raise ValueError(
            f"stat_name must be 'lambda' or 'poisson', got {stat_name!r}"
        )

    submission_dir = Path(submission_dir)
    data_dir = submission_dir / "data"
    submission_dir = data_dir if data_dir.is_dir() else submission_dir

    pattern = re.compile(rf"^pvalues_{stat_name}_job(\d+)\.pkl$")
    job_files: list[tuple[int, Path]] = []
    for path in submission_dir.iterdir():
        match = pattern.match(path.name)
        if match is not None:
            job_files.append((int(match.group(1)), path))
    job_files.sort(key=lambda x: x[0])

    if not job_files:
        raise FileNotFoundError(
            f"No pvalues_{stat_name}_job*.pkl files found in {submission_dir}"
        )

    per_job: list[tuple[float, float, np.ndarray]] = []
    for _, path in job_files:
        with path.open("rb") as fh:
            durs, ints, pvals = pickle.load(fh)
        pvals = np.asarray(pvals)
        if pvals.ndim != 3 or pvals.shape[0] != 1 or pvals.shape[1] != 1:
            raise ValueError(
                f"{path.name}: expected pvalues shape (1, 1, n_sims), "
                f"got {pvals.shape}"
            )
        per_job.append((float(durs[0]), float(ints[0]), pvals[0, 0, :]))

    n_sims_per_cell = {p.shape[0] for _, _, p in per_job}
    if len(n_sims_per_cell) != 1:
        raise ValueError(
            f"Per-job pvalue arrays have varying n_simulations: "
            f"{n_sims_per_cell}"
        )
    n_sims = n_sims_per_cell.pop()

    durations = np.array(sorted({d for d, _, _ in per_job}), dtype=float)
    intensities = np.array(sorted({f for _, f, _ in per_job}), dtype=float)

    pvalues_3d = np.full(
        (len(durations), len(intensities), n_sims), np.nan, dtype=float,
    )
    seen = np.zeros((len(durations), len(intensities)), dtype=bool)

    for d, f, p in per_job:
        i = int(np.searchsorted(durations, d))
        j = int(np.searchsorted(intensities, f))
        if seen[i, j]:
            raise ValueError(
                f"Duplicate grid cell (duration={d}, intensity={f}) "
                f"encountered while merging {stat_name} p-values."
            )
        pvalues_3d[i, j, :] = p
        seen[i, j] = True

    if output_name is None:
        output_name = f"pvalues_{stat_name}_merged.pkl"

    out_path = submission_dir / output_name
    with out_path.open("wb") as fh:
        pickle.dump((durations, intensities, pvalues_3d), fh)
    return out_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge per-job result files from a submission directory. "
            "By default, merges results_job{N}.npz into results_merged.npz. "
            "With --mode grid-pvalues, merges pvalues_{lambda,poisson}_job{N}.pkl "
            "into 3D pickles indexed by (duration, intensity, simulation)."
        )
    )
    parser.add_argument(
        "submission_dir",
        type=Path,
        help="Submission directory containing per-job files.",
    )
    parser.add_argument(
        "--mode",
        choices=("stack", "grid-pvalues"),
        default="stack",
        help=(
            "stack: legacy mode, stacks results_job{N}.npz files. "
            "grid-pvalues: merges per-job pickle p-value files into 3D arrays."
        ),
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help=(
            "Filename for the merged output. Defaults to "
            "results_merged.npz (stack mode) or "
            "pvalues_{stat}_merged.pkl (grid-pvalues mode)."
        ),
    )
    parser.add_argument(
        "--stat",
        choices=("lambda", "poisson", "both"),
        default="both",
        help=(
            "For grid-pvalues mode: which statistic(s) to merge. "
            "Defaults to merging both."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.mode == "stack":
        out_path = merge_results(
            submission_dir=args.submission_dir,
            output_name=args.output_name or "results_merged.npz",
        )
        with np.load(out_path) as data:
            n_jobs = len(data["job_id"])
            keys = [k for k in data.files if k != "job_id"]
        print(f"Merged {n_jobs} jobs into {out_path}")
        print(f"  keys: {keys}")
        return

    stats = ("lambda", "poisson") if args.stat == "both" else (args.stat,)
    if args.output_name is not None and len(stats) > 1:
        raise ValueError(
            "--output-name cannot be combined with --stat both; "
            "run once per statistic if you need custom names."
        )
    for stat_name in stats:
        out_path = merge_grid_pvalues(
            submission_dir=args.submission_dir,
            stat_name=stat_name,
            output_name=args.output_name,
        )
        with out_path.open("rb") as fh:
            durations, intensities, pvalues = pickle.load(fh)
        print(f"Merged {stat_name} p-values into {out_path}")
        print(
            f"  durations: {len(durations)}, intensities: {len(intensities)}, "
            f"pvalues shape: {pvalues.shape}"
        )


if __name__ == "__main__":
    main()
