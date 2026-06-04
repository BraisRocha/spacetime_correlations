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
├── jobs/                               # Submission layer (local and HTCondor)
│   ├── condor/                         # HTCondor submit files and wrappers
│   │   └── grid_p50/
│   │       ├── grid_p50.sub            # condor_submit file
│   │       ├── grid_p50_params.txt     # parameter grid (shared with local)
│   │       ├── run_grid_p50.sh         # bash wrapper executed by Condor
│   │       └── submit_grid_p50.sh      # convenience submit script
│   └── local/                          # Local launchers (iterate over the same grids)
│       └── run_grid_p50.sh
├── logs/
│   └── condor/                         # HTCondor stdout/stderr (gitignored)
├── output/                             # Scientific results (gitignored)
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

`run_grid_p50.py` evaluates the sensitivity of the Lambda and Poisson
tests over a 2-D grid of flare **duration** × flare **intensity**, with
**one Condor job per grid cell**. Each job injects a flare on top of
`n_simulations` background realizations and stores the full
per-simulation p-value distribution for both tests. The end-to-end
workflow has three steps: **submit → merge → plot**.

### 1. Submit the grid

```bash
bash jobs/condor/grid_p50/submit_grid_p50.sh
```

This script:
1. Regenerates the parameter grid `jobs/condor/grid_p50/grid_p50_params.txt`
   (durations × intensities × seed — edit the ranges at the top of the
   script if needed).
2. Builds a **submission ID** from the current timestamp
   (e.g. `20260525_153127`) and submits all cells to HTCondor.

The submission ID is printed to the terminal and is also the name of the
output directory. **Write it down — you need it for the plot step.**

Each job writes into `output/scripts/grid_p50/<ID>/data/` and produces
four files (`N` = the job/process number):

| File | Contents |
|------|----------|
| `run_job{N}.log` | Per-job run log |
| `metadata_job{N}.json` | Per-job metadata (`expected_n`, `T_obs`, flare params, …) |
| `pvalues_lambda_job{N}.pkl` | Lambda p-values for that cell, as `(durations, intensities, pvalues)` with `pvalues` of shape `(1, 1, n_simulations)` |
| `pvalues_poisson_job{N}.pkl` | Poisson p-values for that cell, same layout |

### 2. Merge the per-job pickles (from the terminal)

Once all jobs have finished, collapse the per-cell pickles into one
pickle per statistic:

```bash
python -m spacetimecorr.io.merge output/scripts/grid_p50/<ID> --mode grid-pvalues
```

Replace `<ID>` with your submission ID. The command auto-detects the
`data/` subdirectory and writes, next to the per-job files:

- `pvalues_lambda_merged.pkl`
- `pvalues_poisson_merged.pkl`

Each merged file holds a tuple `(durations, intensities, pvalues)` where
`durations` and `intensities` are the sorted grid axes and `pvalues` has
shape `(n_durations, n_intensities, n_simulations)`.

> By default both statistics are merged (`--stat both`). Use
> `--stat lambda` or `--stat poisson` to merge only one.

### 3. Plot the results

The percentile (median by default) is now computed **inside the plot
script** from the merged p-value distributions, so you only need to point
it at the run. Open `scripts/plots/plot_grid_p50.py` and edit the bottom
`__main__` block to set **your submission ID** and the desired output
path:

```python
if __name__ == "__main__":
    run_dir = Path("output/scripts/grid_p50/<ID>")   # <- your submission ID
    output_dir = run_dir / "figures"                 # <- where the PNGs go
    main(run_dir=run_dir, output_dir=output_dir)
```

Then run it:

```bash
python scripts/plots/plot_grid_p50.py
```

**Finding the ID:** it is the name of the run directory, and it is also
printed in every per-job log. Open any `output/scripts/grid_p50/<ID>/data/run_job*.log`
and read the line:

```text
Simulation ID: 20260525_153127
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
Job files in `jobs/` follow the same root without the `run_`/`plot_` prefix
(`grid_p50.sub`, `run_grid_p50.sh`).

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
