#Code to scan the expected number of events with th declination

from pathlib import Path

import astropy.units as u
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from astropy.time import Time

import spacetimecorr as stc

# Reuse the paper-quality matplotlib style shipped with the plotting scripts.
_RC_FILE = (
    Path(__file__).resolve().parents[1] / "plots" / "matplotlibrc_test"
)
if _RC_FILE.exists():
    mpl.rc_file(_RC_FILE, use_default_template=False)


def main(seed:int) -> None:

    n_total = int(3e5)

    rng_manager = stc.RNGManager(seed=seed)
    rng_events = rng_manager.get("events")
    rng_exposure = rng_manager.get("exposure")

    # Pierre Auger Observatory coordinates
    latitude_pa = -35.15
    longitude_pa = -69.15
    altitude_pa = 1425
    observatory_resolution = 1. # degree

    observatory = stc.Observatory(
        latitude=latitude_pa,
        longitude=longitude_pa,
        altitude=altitude_pa,
    )

    # Observation interval
    T_obs = 12 * u.year
    t0 = Time("2010-01-01T00:00:00", scale="utc")
    tf = t0 + T_obs

    exposure_model = stc.ExposureModel(
        observatory=observatory,
        t0=t0,
        tf=tf,
        rng=rng_exposure,
        theta_max_deg=80
    )

    dec = np.linspace(-90, 45, 1000)
    centres = np.column_stack((np.zeros_like(dec), dec))
    radii = 1.05 * observatory_resolution

    sky_grid = stc.SkyGrid(
        centres=centres,
        radii=radii,
    )

    mu = sky_grid.expected_n_in_window(
        n_events=n_total,
        exposure_model=exposure_model,
    )

    # mu is (roughly) flat in solid angle where the exposure is uniform, so
    # plotting against sin(dec) is the natural x-axis for a directional scan.
    sin_dec = np.sin(np.deg2rad(dec))

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.plot(sin_dec, mu)
    ax.set_xlabel(r"$\sin(\delta)$")
    ax.set_ylabel(r"$\mu$")
    ax.set_title(
        rf"$\psi = {radii:.2f}^\circ$, "
        rf"$N_\mathrm{{tot}} = {n_total:.0e}$"
    )
    ax.set_xlim(-1.0, 1.0)
    ax.margins(x=3, y=0.05)


    out_path = Path(__file__).resolve().parent / "expected_n_vs_sindec.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {out_path}")


if __name__ == '__main__':
    main(seed=42)