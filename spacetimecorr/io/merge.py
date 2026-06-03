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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge per-job results_job{N}.npz files from a submission "
            "directory into a single results_merged.npz."
        )
    )
    parser.add_argument(
        "submission_dir",
        type=Path,
        help="Submission directory containing results_job*.npz files.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="results_merged.npz",
        help="Filename for the merged output (default: results_merged.npz).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out_path = merge_results(
        submission_dir=args.submission_dir,
        output_name=args.output_name,
    )
    with np.load(out_path) as data:
        n_jobs = len(data["job_id"])
        keys = [k for k in data.files if k != "job_id"]
    print(f"Merged {n_jobs} jobs into {out_path}")
    print(f"  keys: {keys}")


if __name__ == "__main__":
    main()
