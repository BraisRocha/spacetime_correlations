from pathlib import Path

import astropy.units as u
from astropy.time import Time

from spacetimecorr import EventSample, RNGManager
import spacetimecorr.plotting as stcp


if __name__ == "__main__":

    # ------------------
    # Configuration
    # ------------------
    n_events = int(1e4)

    t0 = Time("2026-01-01T00:00:00", scale="utc")
    tf = t0 + 1 * u.week

    rng = RNGManager(seed=42)
    rng_events = rng.get("events")

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

    if not sample.is_populated():
        raise RuntimeError(
            "Some error ocurred throughout the coordinates sampling process")

    # ------------------
    # Output directory (already exists)
    # ------------------
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    outdir = PROJECT_ROOT / "output" / "diagnostics" / "events"

    stcp.plot_plain(sample, outdir/ "plain_projection.png")
    stcp.plot_hammer(sample, outdir / "hammer_projection.png")
    stcp.plot_hammer_heatmap(sample, outdir / "hammer_heatmap.png")

    print(f"Saved diagnostic plots to: {outdir}")