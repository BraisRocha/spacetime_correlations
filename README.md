# Spacetime Correlations (`spacetimecorr`)

`spacetimecorr` is a Python package for simulating and analysing **spatiotemporal
correlations** in ultra-high-energy cosmic-ray (UHECR) arrival directions.

The package is built around a small number of composable primitives:
an `Observatory` and an `ExposureModel` define the detector geometry and
directional exposure; a `SkyWindow` selects a region of the sky; an
`EventSample` is drawn directly inside that window with the right
exposure weighting; and a `Flare` can be optionally overlaid on top of
the background to study localised signals. The Lambda anisotropy
estimator and its conditional / marginal distributions live in
`spacetimecorr.statistics` and are evaluated on an `EventSample`.

Reproducibility is enforced through `RNGManager`, which provides
deterministic, named, independent random streams derived from a single
master seed.

## Installation

### Requirements
- Python `>=3.10`
- Core dependencies (declared in `pyproject.toml`):
  - `numpy`
  - `astropy`
  - `scipy`
  - `matplotlib`
- Optional extras:
  - `skymap` — installs `healpy`, required only for HEALPix sky-map generation/plotting APIs.
  - `scripts` — installs `tqdm`, used by the helper scripts under `scripts/`.

### Install (editable)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

### Install with sky-map support

```bash
pip install -e ".[skymap]"
```

### Install with the script helpers

```bash
pip install -e ".[scripts]"
```

You can combine extras: `pip install -e ".[skymap,scripts]"`.

### Keeping dependencies updated

```bash
pip list --outdated
```

## Repository layout

```text
spacetime_correlations/
├── pyproject.toml
├── README.md
├── TODO.md
├── scripts/                            # Python scripts (backend-agnostic)
│   ├── diagnostics/                    # Standalone sanity-check scripts
│   ├── montecarlo/                     # Monte-Carlo runners (one process = one run)
│   │   ├── run_null.py                 # null hypothesis (pure isotropy)
│   │   ├── run_compare_bg_signal.py    # fixed injection vs background diagnostic
│   │   ├── run_scan_intensity.py       # 1-D scan over flare S/N ratio
│   │   ├── run_scan_correlation.py     # 1-D scan over correlation type
│   │   └── run_grid_p50.py             # 2-D (duration, intensity) grid; one Condor job per point
│   └── plots/                          # Plotting helpers for Monte-Carlo outputs
├── condor/                             # HTCondor submission layer
│   └── grid_p50/                       # one folder per submitted script
│       ├── grid_p50.sub                # condor_submit file (absolute cluster paths)
│       ├── run_grid_p50.sh             # wrapper HTCondor runs, once per grid point
│       ├── submit_grid_p50.sh          # builds the grid and submits it
│       └── finalize_grid_p50.sh        # merges the results once every job is done
├── scratch/                            # One folder per running submission (gitignored)
├── output/                             # Scientific results (gitignored)
│   ├── montecarlo/                     # one folder per run, mirrors scripts/montecarlo/
│   └── diagnostics/                    # mirrors scripts/diagnostics/
└── spacetimecorr/                      # Python package
    ├── __init__.py
    ├── observatory.py                  # Observatory location (lat/lon/alt)
    ├── exposure.py                     # Directional exposure model
    ├── skywindow.py                    # Circular sky windows (spherical caps)
    ├── event_sample.py                 # Event container + full-sky / in-window factories
    ├── flare.py                        # Synthetic flare component (signal injection)
    ├── statistics.py                   # Lambda estimator and its distributions
    ├── rng.py                          # Reproducible named RNG streams
    └── io/                             # Logging and run-output helpers
        ├── __init__.py
        ├── logs.py
        └── output.py
```

## Quick start

A minimal end-to-end example: generate background events inside a sky
window with the correct directional exposure, overlay a synthetic
flare, and evaluate the Lambda anisotropy estimator on both the
background-only and flare-injected samples.

```python
import numpy as np
import astropy.units as u
from astropy.time import Time

from spacetimecorr import (
    RNGManager,
    EventSample,
    SkyWindow,
    Observatory,
    ExposureModel,
    Flare,
    lambda_estimator,
)

# Observation interval
n_total = int(1e5)                                  # equivalent full-sky population
t0 = Time("2026-01-01T00:00:00", scale="utc")
tf = t0 + 1 * u.week

# Reproducible RNG streams
rngm = RNGManager(seed=42)
rng_events = rngm.get("events")
rng_exposure = rngm.get("exposure")
rng_flare = rngm.get("flare")

# Detector geometry and directional exposure
obs = Observatory(latitude=-35.15, longitude=-69.2, altitude=1425)
exposure_model = ExposureModel(observatory=obs, t0=t0, tf=tf, rng=rng_exposure)

# Sky window and per-window background sample
window = SkyWindow(centre=np.array([30.0, 0.0]), radius=2.0)
sample = EventSample.in_window(
    window=window,
    n_total=n_total,
    exposure_model=exposure_model,
    t0=t0,
    tf=tf,
    rng=rng_events,
)
sample.assign_directional_exposure(window=window, exposure_model=exposure_model)

lam_bkg = lambda_estimator(sample=sample)

# Synthetic flare overlaid on the sample (overdensity injection)
flare = Flare(
    n_flare=20,
    duration=1 * u.day,
    t0=t0,
    tf=tf,
    centre=window.centre,
    exposure_model=exposure_model,
    rng=rng_flare,
)
flare.generate_in_window(window=window, sigma=1.0)  # sigma in degrees

sample.inject_flare(flare=flare, mode="overdensity")
sample.assign_directional_exposure(window=window, exposure_model=exposure_model)

lam_flare = lambda_estimator(sample=sample)

print(f"Events in window: {sample.n_sample}")
print(f"Expected events:  {sample.expected_n:.2f}")
print(f"Lambda (bkg):     {lam_bkg:.3f}")
print(f"Lambda (+flare):  {lam_flare:.3f}")
```

## Running the 2-D (duration, intensity) grid on HTCondor

`run_grid_p50.py` evaluates the sensitivity of the Lambda and Poisson tests
over a 2-D grid of flare **duration** x flare **intensity**, with **one
Condor job per grid cell**.

The submission layer lives in `condor/grid_p50/` and has three parts:

| File | Role |
|------|------|
| `submit_grid_p50.sh` | what you run: builds the parameter grid and submits it |
| `run_grid_p50.sh` | what HTCondor runs on a worker node, once per grid cell |
| `finalize_grid_p50.sh` | what runs at the end, once every job has finished |

### Submit

From the submit host, with the `stc_env` conda environment active:

```bash
bash condor/grid_p50/submit_grid_p50.sh
```

It prints a **submission ID** (a timestamp) that names the run from there on.
The grid ranges are set inside the script; edit them there if needed.

### What happens next

While the jobs run, everything they produce goes to
`scratch/grid_p50/<ID>/`: the parameter grid, the per-job `.out` / `.err`,
and one set of results per cell. Nothing is written to `output/` yet.

When the last job finishes, HTCondor itself launches the final step, so you
do not have to wait around for it. It checks that every cell arrived, merges
the per-cell results, writes them to `output/montecarlo/grid_p50/<ID>/`, and
deletes the scratch directory.

**A scratch directory that is still there is the sign that a submission
needs looking at.** It is kept whenever a cell is missing, with the failed
jobs' tracebacks and logs, and with the results of the cells that did work
so the grid can be rebuilt once they have been rerun. Use `--keep-scratch`
at submission time to keep it in any case.

### Results

`output/montecarlo/grid_p50/<ID>/` ends up holding four files:

| File | Contents |
|------|----------|
| `pvalues_lambda_merged.pkl` | `(durations, intensities, pvalues)`, `pvalues` of shape `(n_durations, n_intensities, n_simulations)` |
| `pvalues_poisson_merged.pkl` | same layout |
| `metadata.json` | the settings shared by every job, plus one row per grid cell |
| `run.log` | short report: cells delivered, `expected_n`, runtimes |

A cell whose job failed is `NaN` in the merged arrays, and the plotting
script renders it blank, so an incomplete grid can still be plotted.

### Plot

Open `scripts/plots/plot_grid_p50.py` and set your submission ID in the
bottom `__main__` block:

```python
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    run_dir = project_root / "output" / "montecarlo" / "grid_p50" / "<ID>"  # <- your submission ID
    output_dir = run_dir / "figures"                                        # <- where the PNGs go
    main(run_dir=run_dir, output_dir=output_dir)
```

```bash
python scripts/plots/plot_grid_p50.py
```

## Script naming convention

Monte-Carlo scripts follow a two-part scheme: `<mode>_<what_varies>.py`.

| Prefix | Meaning |
|--------|---------|
| `run_null` | No injection; establishes the null Lambda distribution |
| `run_compare_*` | Pair of distributions compared side by side |
| `run_scan_*` | 1-D parameter sweep (one parameter varies) |
| `run_grid_*` | 2-D parameter sweep; designed for Condor array jobs |

Plot scripts mirror the same root name (`plot_null.py`, `plot_scan_intensity.py`, …).
Condor files in `condor/` are grouped in a folder with the same root name
(`condor/grid_p50/`, holding `grid_p50.sub` and the `submit_`, `run_` and
`finalize_` scripts).

The scripts under `scripts/` are provided as worked examples of the
analysis workflows the package supports; new studies are expected to
add their own scripts following the same conventions. Outputs are
written under `output/` (created automatically by the helper utilities).

## Notes

- APIs are still evolving and may change between versions.
- `spacetimecorr` can be imported without `healpy`; `healpy` is loaded only when calling HEALPix map/plot methods.
- See `TODO.md` for known issues and follow-up work that has been deferred.

## Python 3.9 compatibility

The package targets **Python >= 3.9** (see `requires-python` in `pyproject.toml`).
The natural minimum would be 3.10, because the code makes heavy use of two
Python 3.10+ features. To support the 3.9 interpreters found on some clusters,
the following accommodations were made:

1. **PEP 604 union annotations (`X | Y`).** Used throughout the type hints.
   Natively this requires 3.10. It is made 3.9-safe by adding
   `from __future__ import annotations` at the top of every module that uses
   it, which defers annotation evaluation (the hints become strings and are
   never evaluated at runtime).
2. **`@dataclass(slots=True)`.** The `slots=` parameter is a runtime feature of
   3.10+ and cannot be deferred. It was removed from the two affected
   dataclasses (`Observatory` in `spacetimecorr/observatory.py` and `SkyWindow`
   in `spacetimecorr/skywindow.py`), losing only a minor memory optimisation;
   behaviour is otherwise identical. These lines are marked with a
   `# NOTE: no slots=True (requires Python >= 3.10)` comment.

### Reverting to Python >= 3.10 only

If 3.9 support is no longer needed, undo the above:

- Set `requires-python = ">=3.10"` in `pyproject.toml`.
- Restore `slots=True` on the two dataclasses (search for the
  `# NOTE: no slots=True` comments and change `@dataclass(frozen=True)` back to
  `@dataclass(frozen=True, slots=True)`).
- Optionally remove the `from __future__ import annotations` lines (harmless to
  keep, but unnecessary on 3.10+).

Compatibility was verified with [`vermin`](https://github.com/netromdk/vermin):
`vermin --eval-annotations spacetimecorr scripts tests` reports a minimum
required version of 3.9.
