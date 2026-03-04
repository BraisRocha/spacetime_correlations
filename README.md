# Spacetime Correlations (`spacetimecorr`)

`spacetimecorr` is a Python package for simulating and analyzing **spatiotemporal correlations** in ultra-high-energy cosmic ray (UHECR) arrival directions.

## spacetimecorr/
At a high level, the project helps you:
- simulate isotropic event samples on the sphere,
- define circular sky windows (spherical caps),
- model directional exposure from a ground observatory over an observation interval,
- sample event exposure values and produce diagnostic plots.

---

## Installation

### Requirements
- Python `>=3.10`.
- Core dependencies (declared in `pyproject.toml`):
  - `numpy`
  - `astropy`
  - `scipy`
  - `matplotlib`
  - `cartopy`

### Install in editable mode

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

> If you do not install the package, you can still run scripts by setting `PYTHONPATH=.`.

---

## Repository layout

```text
spacetime_correlations/
├── spacetimecorr/
│   ├── __init__.py
│   ├── rng.py
│   ├── event_sample.py
│   ├── skywindow.py
│   ├── observatory.py
│   ├── exposure.py
│   ├── analysis.py
│   ├── flare.py
│   └── plotting/
|       ├── __init__.py
│       ├── events_plots.py
│       └── exposure_plots.py
├── scripts/
│   └── diagnostics/
|       ├── exposure_diagnostic.py
│       └── sampling_diagnostic.py
├── pyproject.toml
└── README.md
```

---

## Core concepts

### `RNGManager`
Creates deterministic named random-number streams so different parts of the pipeline can remain reproducible and independent.

### `EventSample`
Represents a set of events observed between `t0` and `tf`.
- `sample_equatorial_coordinates()` draws isotropic RA/Dec.
- `subset(mask)` creates a selected subsample.
- `add_directional_exposure_for_window(...)` attaches sampled cumulative directional exposure to a selected sample.

### `SkyWindow`
Defines a spherical-cap selection via:
- `centre = [RA_deg, Dec_deg]`
- `radius` (degrees)

Main methods:
- `contains(sample)` → boolean mask,
- `select(sample)` → selected `EventSample`,
- `expected_counts_in_window(sample)` under uniform full-sky assumption.

### `Observatory`
Stores observatory latitude/longitude/altitude and an `astropy` `EarthLocation`.

### `ExposureModel`
Maps times to cumulative directional exposure and samples exposure-space event values.
- `to_directional_exposure(t, centre)`
- `max_directional_exposure(centre)`
- `sample_directional_exposure(...)`

---

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
sample.sample_equatorial_coordinates()

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

---

## Running diagnostics

The repository includes a diagnostic script:

```bash
mkdir -p output/diagnostics/events
PYTHONPATH=. python scripts/diagnosis/events_diagnosis.py
```

---

## Development status

This project is actively evolving. APIs may change as analysis workflows and statistical modules are refined.
