from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json


def make_run_dir(
    base_dir: Path,
    run_code: str,
    seed: int,
    job_id: str | None = None,
    submission_id: str | None = None,
) -> tuple[Path, str]:
    """
    Create and return a unique output directory for one run.

    Parameters
    ----------
    base_dir : Path
        Base output directory, e.g. project_root / "output" / "scripts".
    run_code : str
        Short code identifying the script or experiment, e.g. "fi".
    seed : int
        Random seed for the run.
    job_id : str | None, optional
        Optional job identifier for parallel/batch runs.
    submission_id : str | None, optional
        If provided, use this as the directory name (shared across all jobs
        in the same submission). The directory is created with exist_ok=True
        so concurrent jobs can safely call this simultaneously.
        When None, a unique name is generated from timestamp and seed.

    Returns
    -------
    outdir : Path
        Path to the created run directory.
    run_name : str
        Name used to build ``outdir``.
    """
    if submission_id is not None:
        run_name = submission_id
        outdir = base_dir / run_code / run_name
        outdir.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{timestamp}_seed{seed}"
        if job_id is not None:
            run_name += f"_job{job_id}"
        outdir = base_dir / run_code / run_name
        outdir.mkdir(parents=True, exist_ok=False)

    return outdir, run_name


def write_metadata(
    outdir: Path,
    metadata: dict,
    filename: str = "metadata.json",
) -> None:
    """
    Write run metadata to ``outdir / filename``.

    Parameters
    ----------
    outdir : Path
        Output directory of the run (must already exist).
    metadata : dict
        JSON-serialisable mapping of metadata fields. Numpy arrays and other
        non-serialisable objects must be converted by the caller.
    filename : str, optional
        Name of the output file. Defaults to ``"metadata.json"``.
        For multi-job submissions use e.g. ``"metadata_job0.json"``.
    """
    with open(outdir / filename, "w") as f:
        json.dump(metadata, f, indent=2)