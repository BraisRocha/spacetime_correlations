from pathlib import Path
import numpy as np

import astropy.units as u
from astropy.time import Time

from spacetimecorr import EventSample, RNGManager, SkyWindow, Observatory, ExposureModel
import spacetimecorr.plotting as stcp


if __name__ == "__main__":

    # ------------------
    # Configuration
    # ------------------
    n_events = int(1e5)

    t0 = Time("2026-01-01T00:00:00", scale="utc")
    tf = t0 + 1 * u.week

    rng = RNGManager(seed=42)
    rng_events = rng.get("events")
    rng_exposure = rng.get("exposure")

    # ------------------
    # Create sample
    # ------------------
    sample = EventSample(
        n_events=n_events,
        t0=t0,
        tf=tf,
        rng=rng_events,
    )

    sample.sample_equatorial_coordinates()

    if not sample.is_populated:
        raise RuntimeError("An issue related to the sampling of spatial coordiantes arose.")

    # ------------------
    # Output directory (already exists)
    # ------------------
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    outdir = PROJECT_ROOT / "output" / "diagnostics" / "events"

    stcp.plot_plain(sample, outdir/ "plain_projection.png")
    stcp.plot_hammer(sample, outdir / "hammer_projection.png")
    stcp.plot_hammer_heatmap(sample, outdir / "hammer_heatmap.png")

    #We make now a window selection

    #Define a window
    centre=np.array([30, 0])
    window = SkyWindow(centre=centre, radius=2)

    window_subsample  = window.select(sample=sample)
    expected_counts = window.expected_counts_in_window(sample=sample)

    stcp.plot_plain(window_subsample, outdir / "window_sample_hammer_projection.png")

    #Add directional exposure to the window
    observatory = Observatory(latitude=-35.15, longitude=-69.2, altitude=1425)
    exposure = ExposureModel(observatory=observatory, t0=t0, tf=tf, rng=rng_exposure)

    window_subsample.add_directional_exposure_for_window(
        parent_sample=sample, window=window, exposure_model=exposure
    )

    if not window_subsample.has_exposure:
        raise RuntimeError("An issue related to the sampling of directional exposure arose.")

    print(f"Saved diagnostic plots to: {outdir}")