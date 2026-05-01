from pathlib import Path
from datetime import datetime
import json


def make_run_dir(
    base_dir: Path,
    run_code: str,
    seed: int,
    job_id: str | None = None,
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

    Returns
    -------
    outdir : Path
        Path to the created run directory.
    run_name : str
        Name component (timestamp_seed[_job]) used to build ``outdir``.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_name = f"{timestamp}_seed{seed}"
    if job_id is not None:
        run_name += f"_job{job_id}"

    outdir = base_dir / run_code / run_name
    outdir.mkdir(parents=True, exist_ok=False)

    return outdir, run_name


def write_metadata(outdir: Path, metadata: dict) -> None:
    """
    Write run metadata to ``outdir / "metadata.json"``.

    Parameters
    ----------
    outdir : Path
        Output directory of the run (must already exist).
    metadata : dict
        JSON-serialisable mapping of metadata fields. Numpy arrays and other
        non-serialisable objects must be converted by the caller.
    """
    with open(outdir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)