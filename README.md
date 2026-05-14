# Spacetime Correlations (`spacetimecorr`)

`spacetimecorr` is a Python package for simulating and analyzing **spatiotemporal correlations** in ultra-high-energy cosmic ray (UHECR) arrival directions.

Current functionality includes:
- isotropic event simulation in equatorial coordinates,
- circular sky-window event selection,
- observatory-based directional exposure modeling,
- Monte Carlo scripts for isotropy, flare-injection, and sensitivity studies,
- diagnostic plotting helpers.

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
├── scripts/                          # Python scripts (backend-agnostic)
│   ├── diagnostics/
│   │   ├── exposure_diagnostic.py
│   │   ├── flare_diagnostic.py
│   │   └── sampling_diagnostic.py
│   ├── montecarlo/
│   │   ├── run_null.py               # null hypothesis (pure isotropy)
│   │   ├── run_compare_bg_signal.py  # fixed injection vs background diagnostic
│   │   ├── run_scan_intensity.py     # 1-D scan over flare S/N ratio
│   │   ├── run_scan_correlation.py   # 1-D scan over correlation type
│   │   └── run_grid_p50.py          # 2-D (duration, intensity) grid; one Condor job per point
│   └── plots/
│       ├── plot_null.py
│       ├── plot_compare_bg_signal.py
│       ├── plot_scan_intensity.py
│       └── plot_scan_correlation.py
├── jobs/                             # Submission layer (local and HTCondor)
│   ├── condor/                       # HTCondor submit files and wrappers
│   │   ├── grid_p50.sub              # condor_submit file
│   │   ├── grid_p50_params.txt       # parameter grid (shared with local)
│   │   └── run_grid_p50.sh           # bash wrapper executed by Condor
│   └── local/                        # Local launchers (iterate over same grids)
│       └── run_grid_p50.sh
├── logs/
│   └── condor/                       # HTCondor stdout/stderr (gitignored)
├── output/                           # Scientific results (gitignored)
└── spacetimecorr/                    # Python package
    ├── __init__.py
    ├── event_sample.py
    ├── exposure.py
    ├── flare.py
    ├── observatory.py
    ├── rng.py
    ├── skywindow.py
    ├── statistics.py
    └── io/
        ├── __init__.py
        ├── logs.py
        └── output.py
```

## Quick start

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
)

# Observation interval
n_events = int(1e5)
t0 = Time("2026-01-01T00:00:00", scale="utc")
tf = t0 + 1 * u.week

# Reproducible RNG streams
rngm = RNGManager(seed=42)
rng_events = rngm.get("events")
rng_exposure = rngm.get("exposure")

# Simulated full sample
sample = EventSample(n_events=n_events, t0=t0, tf=tf, rng=rng_events)

# Sky-window selection
window = SkyWindow(centre=np.array([30.0, 0.0]), radius=2.0)
subsample = sample.select_subsample(window)

# Directional exposure model
obs = Observatory(latitude=-35.15, longitude=-69.2, altitude=1425)
exposure = ExposureModel(observatory=obs, t0=t0, tf=tf, rng=rng_exposure)

subsample.assign_directional_exposure(
    window=window,
    exposure_model=exposure,
)

print("Selected events:", subsample.n_events)
print("Has exposure:", subsample.has_exposure)
```

## Script naming convention

Monte Carlo scripts follow a two-part scheme: `<mode>_<what_varies>.py`.

| Prefix | Meaning |
|--------|---------|
| `run_null` | No injection; establishes the null Lambda distribution |
| `run_compare_*` | Pair of distributions compared side by side |
| `run_scan_*` | 1-D parameter sweep (one parameter varies) |
| `run_grid_*` | 2-D parameter sweep; designed for Condor array jobs |

Plot scripts mirror the same root name (`plot_null.py`, `plot_scan_intensity.py`, …).
Job files in `jobs/` follow the same root without the `run_`/`plot_` prefix
(`grid_p50.sub`, `run_grid_p50.sh`).

## Diagnostics and Monte Carlo scripts

```bash
python scripts/diagnostics/sampling_diagnostic.py
python scripts/diagnostics/exposure_diagnostic.py
python scripts/diagnostics/flare_diagnostic.py
python scripts/montecarlo/run_null.py
python scripts/montecarlo/run_compare_bg_signal.py
python scripts/montecarlo/run_scan_intensity.py
python scripts/montecarlo/run_scan_correlation.py
```

Plotting helpers for the Monte Carlo outputs live in `scripts/plots/`.

Outputs are written under `output/` (created by helper utilities/scripts).

## Notes

- APIs are still evolving and may change between versions.
- `spacetimecorr` can be imported without `healpy`; `healpy` is loaded only when calling HEALPix map/plot methods.
- See `TODO.md` for known issues and follow-up work that has been deferred.
