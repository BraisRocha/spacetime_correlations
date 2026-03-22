from pathlib import Path
from datetime import datetime
import json


def make_run_dir(
    base_dir: Path,
    run_code: str,
    seed: int,
    job_id: str | None = None,
) -> Path:
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
    Path
        Path to the created run directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_name = f"{timestamp}_seed{seed}"
    if job_id is not None:
        run_name += f"_job{job_id}"

    outdir = base_dir / run_code / run_name
    outdir.mkdir(parents=True, exist_ok=False)

    return outdir


def write_metadata(outdir: Path, metadata: dict) -> None:
    """
    Write run metadata to a JSON file inside the output directory.
    """
    with open(outdir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)