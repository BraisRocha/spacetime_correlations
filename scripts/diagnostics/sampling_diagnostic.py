from pathlib import Path
import numpy as np

import astropy.units as u
from astropy.time import Time

from spacetimecorr import (
    EventSample,
    RNGManager,
    SkyWindow,
    Observatory,
    ExposureModel,
)

import spacetimecorr.plotting as stcp


def main() -> None:
    """Generate a diagnostic event sample and produce validation plots."""

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    n_events = int(1e5)

    t0 = Time("2026-01-01T00:00:00", scale="utc")
    tf = t0 + 1 * u.week

    rng_manager = RNGManager(seed=42)
    rng_events = rng_manager.get("events")
    rng_exposure = rng_manager.get("exposure")

    # ------------------------------------------------------------------
    # Create event sample
    # ------------------------------------------------------------------
    sample = EventSample(
        n_events=n_events,
        t0=t0,
        tf=tf,
        rng=rng_events,
    )

    sample.sample_equatorial_coordinates()

    if not sample.is_populated:
        raise RuntimeError(
            "Event sample was not populated correctly. "
            "Issue encountered during coordinate sampling."
        )

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    project_root = Path(__file__).resolve().parents[2]
    outdir = project_root / "output" / "diagnostics" / "events"
    outdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Basic sky plots
    # ------------------------------------------------------------------
    stcp.plot_plain(sample, outdir / "plain_projection.png")
    stcp.plot_hammer(sample, outdir / "hammer_projection.png")
    stcp.plot_hammer_heatmap(sample, outdir / "hammer_heatmap.png")

    # ------------------------------------------------------------------
    # Sky window selection
    # ------------------------------------------------------------------
    centre = np.array([30.0, 0.0])
    window = SkyWindow(centre=centre, radius=2)

    window_subsample = window.select(sample=sample)
    expected_counts = window.expected_counts_in_window(sample=sample)

    print(
    "Window selection:\n"
    f"Expected events: {expected_counts}\n"
    f"Observed events: {window_subsample.n_events}"
    )

    stcp.plot_plain(
        window_subsample,
        outdir / "window_sample_plain_projection.png",
    )

    # ------------------------------------------------------------------
    # Directional exposure
    # ------------------------------------------------------------------
    observatory = Observatory(
        latitude=-35.15,
        longitude=-69.2,
        altitude=1425,
    )

    exposure_model = ExposureModel(
        observatory=observatory,
        t0=t0,
        tf=tf,
        rng=rng_exposure,
    )

    window_subsample.add_directional_exposure_for_window(
        parent_sample=sample,
        window=window,
        exposure_model=exposure_model,
    )

    if not window_subsample.has_exposure:
        raise RuntimeError(
            "Directional exposure computation failed."
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    stcp.plot_events_vs_exposure(
        window_subsample,
        outdir / "events_vs_exposure_plot.png",
    )

    print(f"Saved diagnostic plots to: {outdir}")


if __name__ == "__main__":
    main()