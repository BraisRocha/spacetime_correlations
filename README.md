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
