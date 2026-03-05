# Spacetime Correlations (`spacetimecorr`)

`spacetimecorr` is a Python package for simulating and analyzing **spatiotemporal correlations** in ultra-high-energy cosmic ray (UHECR) arrival directions.

It currently focuses on:
- isotropic event simulation in equatorial coordinates,
- circular sky-window event selection,
- observatory-dependent directional exposure modeling,
- basic statistical tooling and diagnostic plotting.

## Installation

### Requirements
- Python `>=3.10`
- Dependencies declared in `pyproject.toml`:
  - `numpy`
  - `astropy`
  - `scipy`
  - `matplotlib`
  - `cartopy`

### Install (editable)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Current repository layout

```text
spacetime_correlations/
├── main.py
├── pyproject.toml
├── README.md
├── scripts/
│   └── diagnostics/
│       ├── sampling_diagnostic.py
│       └── exposure_diagnostic.py
└── spacetimecorr/
    ├── __init__.py
    ├── analysis.py
    ├── event_sample.py
    ├── exposure.py
    ├── flare.py
    ├── observatory.py
    ├── rng.py
    ├── skywindow.py
    ├── statistics.py
    └── plotting/
        ├── __init__.py
        ├── events_plots.py
        └── exposure_plots.py
```

## Main components

### `RNGManager`
Creates deterministic named random-number generators. Different modules can request independent streams by name while keeping reproducibility from one master seed.

### `EventSample`
Represents a collection of events over an observation interval `[t0, tf]`.

Key behavior:
- isotropic RA/Dec sampling (`sample_equatorial_coordinates`),
- derived properties such as observation duration (`T_obs`) and expected event rate,
- windowed subsample creation,
- directional exposure attachment for selected samples.

### `SkyWindow`
Defines a spherical-cap sky selection with:
- `centre = [RA_deg, Dec_deg]`
- `radius` in degrees.

Provides containment masks, selected subsamples, and uniform full-sky expected-count estimates.

### `Observatory`
Dataclass wrapper around observatory coordinates (`latitude`, `longitude`, `altitude`) with validation and cached `astropy.coordinates.EarthLocation`.

### `ExposureModel`
Maps times and directions to cumulative directional exposure and samples exposure-space values for selected events.

### Plotting utilities (`spacetimecorr.plotting`)
Includes helpers to generate:
- plain RA/Dec scatter plots,
- hammer projections and density maps,
- exposure diagnostic plots.

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

# Sky window selection
window = SkyWindow(centre=np.array([30.0, 0.0]), radius=2.0)
subsample = window.select(sample)

# Directional exposure model
obs = Observatory(latitude=-35.15, longitude=-69.2, altitude=1425)
exposure = ExposureModel(observatory=obs, t0=t0, tf=tf, rng=rng_exposure)

subsample.add_directional_exposure_for_window(
    parent_sample=sample,
    window=window,
    exposure_model=exposure,
)

print("Selected events:", subsample.n_events)
print("Has exposure:", subsample.has_exposure)
```

## Diagnostics

Run the current sampling diagnostic script:

```bash
python scripts/diagnostics/sampling_diagnostic.py
```

Outputs are written to `output/diagnostics/events/`.

> Note: `scripts/diagnostics/exposure_diagnostic.py` currently exists as a placeholder file.

## Development notes

- `main.py` runs repeated simulation loops for a baseline pipeline.
- `analysis.py`, `statistics.py`, and `flare.py` are present but still evolving.
- APIs are still under active development and may change.
