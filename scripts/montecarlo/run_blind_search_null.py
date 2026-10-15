"""
Blind all-sky search on a background-only sample, over a HEALPix grid.

Unlike the targeted search (one window per source), the blind search tiles
the field of view with the pixels of a :class:`~spacetimecorr.HealpixGrid`
and scans every one. No flares are injected: this is the null (background
only) case. The pipeline is:

    1. Generate one full-sky sample of ``n_total`` events distributed as the
       observatory's exposure (``EventSample.full_sky``).
    2. Build the grid: every HEALPix pixel at resolution ``nside`` whose
       centre lies inside the field of view.
    3. Assign every event to its pixel with a single ``ang2pix`` call
       (``HealpixGrid.pixel_of``).
    4. For every pixel, select its events with that assignment, attach
       directional exposure and compute the Lambda statistic and a Poisson
       counting statistic with their p-values. The expected count of each
       pixel is weighted by the exposure at its centre.

Pixels that straddle the FoV edge (centre inside, part of the pixel outside)
are skipped. A pixel with fewer than 2 events cannot be analysed with Lambda
and is dropped. Both are recorded as NaN, and the number of pixels actually
tested is saved in the metadata.

The outputs are length-``n_windows`` arrays of the two statistics and their
p-values, aligned with the grid's pixel indices and centres.
"""

import time
from pathlib import Path

import astropy.units as u
import healpy as hp
import numpy as np

from astropy.time import Time
from tqdm import tqdm

import spacetimecorr as stc
from spacetimecorr.io import setup_logger, make_run_dir, write_metadata


def main(seed: int) -> None:
    """Run a blind all-sky search on a background-only sample.

    Generates one full-sky sample distributed as the exposure, tiles the FoV
    with a :class:`~spacetimecorr.HealpixGrid`, and for every pixel selects
    its events and computes the Lambda and Poisson statistics with their
    p-values.
    """

    start_time = time.time()

    # ------------------------------------------------------------------
    # Simulation parameters
    # ------------------------------------------------------------------
    n_total = int(3e5)
    nside = 32                     # HEALPix resolution of the search grid

    # Observational interval
    T_obs = 10 * u.year
    t0 = Time("2013-01-01T00:00:00", scale="utc")
    tf = t0 + T_obs

    # Pierre Auger Observatory
    latitude_pa = -35.15
    longitude_pa = -69.15
    altitude_pa = 1425
    theta_max_deg = 80.0           # zenith-angle cut defining the field of view

    # ------------------------------------------------------------------
    # RNG streams
    # ------------------------------------------------------------------
    rng_manager = stc.RNGManager(seed=seed)
    rng_sample = rng_manager.get("sample")        # full-sky isotropic candidates
    rng_exposure = rng_manager.get("exposure")    # direction thinning + exposure sampling

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    project_root = Path(__file__).resolve().parents[2]
    base_dir = project_root / "output" / "montecarlo"

    outdir, sim_ID = make_run_dir(
        base_dir=base_dir,
        run_code="blind_search_null",
        seed=seed,
    )

    # ------------------------------------------------------------------
    # Logger
    # ------------------------------------------------------------------
    logger = setup_logger(log_path=outdir / "run.log", name="blind_search_null")

    logger.info("Starting blind search (background only) run")
    logger.info("Simulation ID: %s", sim_ID)
    logger.info("Output directory: %s", outdir)
    logger.info("Seed: %d", seed)

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    observatory = stc.Observatory(
        latitude=latitude_pa, longitude=longitude_pa, altitude=altitude_pa,
        area=3000.0,
    )
    exposure_model = stc.ExposureModel(
        observatory=observatory, t0=t0, tf=tf, rng=rng_exposure,
        theta_max_deg=theta_max_deg,
    )

    # ------------------------------------------------------------------
    # 1) Full-sky sample distributed as the exposure
    # ------------------------------------------------------------------
    sample = stc.EventSample.full_sky(
        n_total=n_total, exposure_model=exposure_model,
        t0=t0, tf=tf, rng=rng_sample,
    )
    logger.info("Generated full-sky sample of %d events.", sample.n_sample)

    # ------------------------------------------------------------------
    # 2) HEALPix grid covering the FoV
    # ------------------------------------------------------------------
    grid = stc.HealpixGrid(exposure_model, nside=nside)
    n_windows = len(grid)
    logger.info(
        "Built HEALPix grid: nside=%d, %d pixels with centre inside the FoV "
        "(%.3f deg^2 each).",
        nside, n_windows, grid.solid_angle * np.degrees(1.0) ** 2,
    )

    # Pixels lying entirely inside the FoV. The FoV is a declination band and
    # a pixel's extreme declinations sit at its corners, so a pixel is inside
    # when all four corners have exposure. The others straddle the FoV edge
    # and are skipped in the scan.
    corners = hp.boundaries(nside, grid.pixels, step=1)   # (W, 3, 4) unit vectors
    corners_dec = np.degrees(np.arcsin(corners[:, 2, :]))
    inside_fov = np.all(exposure_model.spatial_exposure(corners_dec) > 0.0, axis=1)
    n_edge = int(np.count_nonzero(~inside_fov))
    logger.info("%d pixels straddle the FoV edge and are skipped.", n_edge)

    # Expected background count per pixel, weighted by the exposure at its
    # centre.
    grid_expected_n = grid.expected_n_in_window(n_total, exposure_model)

    # ------------------------------------------------------------------
    # 3) Assign every event to its pixel at once
    # ------------------------------------------------------------------
    event_pixel = grid.pixel_of(sample.ra, sample.dec)

    # ------------------------------------------------------------------
    # Storage (one entry per pixel, aligned with `grid`)
    # ------------------------------------------------------------------
    lambda_array = np.full(n_windows, np.nan)
    n_sample_array = np.zeros(n_windows, dtype=int)
    pvalues_lambda = np.full(n_windows, np.nan)
    pvalues_poisson = np.full(n_windows, np.nan)

    n_dropped = 0   # pixels with fewer than 2 events (routine)
    n_failed = 0    # pixels with >= 2 events that still raised (a bug)

    # ------------------------------------------------------------------
    # 4) Scan: select each pixel's events and apply the Lambda method
    # ------------------------------------------------------------------
    # Both "too few events" and genuine errors raise ValueError; the event
    # count tells them apart. A pixel with n < 2 cannot be analysed and is
    # dropped. One with n >= 2 should never raise, so it is logged with its
    # traceback and counted separately.
    for i, window in enumerate(tqdm(grid, total=n_windows, desc="Pixels scanned")):
        mask = event_pixel == window.ipix
        n_sample_array[i] = np.count_nonzero(mask)

        if not inside_fov[i]:
            continue

        try:
            subsample = sample.select_subsample(window, mask=mask)
            subsample.assign_cumul_exposure(
                window=window, exposure_model=exposure_model,
            )

            lambda_array[i] = stc.lambda_estimator(sample=subsample)

            pvalues_lambda[i] = stc.lambda_marginal_sf(
                lambda_array[i], grid_expected_n[i]
            )
            pvalues_poisson[i] = stc.poisson_mid_p_value(
                n_sample_array[i], grid_expected_n[i]
            )

        except ValueError:
            lambda_array[i] = np.nan
            if n_sample_array[i] < 2:
                n_dropped += 1
            else:
                n_failed += 1
                logger.exception(
                    "Pixel %d (centre=[%.2f, %.2f], n_sample=%d) failed; "
                    "recorded as NaN.",
                    window.ipix, window.centre[0], window.centre[1],
                    n_sample_array[i],
                )

    n_tested = n_windows - n_edge - n_dropped - n_failed
    logger.info(
        "Scan finished: %d pixels | %d tested | %d on the FoV edge | "
        "%d dropped (n < 2) | %d failed.",
        n_windows, n_tested, n_edge, n_dropped, n_failed,
    )

    # ------------------------------------------------------------------
    # Save outputs and metadata
    # ------------------------------------------------------------------
    np.savez_compressed(
        outdir / "results.npz",
        pixels=grid.pixels,               # (W,)   HEALPix RING index of each pixel
        centres=grid.centres,             # (W, 2) pixel centres [RA, Dec] (deg)
        inside_fov=inside_fov,            # (W,)   False for pixels on the FoV edge (skipped)
        expected_n=grid_expected_n,       # (W,)   exposure-weighted background count
        lambda_array=lambda_array,        # (W,)   Lambda statistic per pixel
        n_sample_array=n_sample_array,    # (W,)   event count per pixel
        pvalues_lambda=pvalues_lambda,    # (W,)   marginal Lambda p-value
        pvalues_poisson=pvalues_poisson,  # (W,)   Poisson counting mid-p value
    )

    elapsed = time.time() - start_time

    write_metadata(
        outdir=outdir,
        metadata={
            "script": Path(__file__).name,
            "run_code": "blind_search_null",
            "seed": seed,
            "runtime_seconds": elapsed,
            "n_total": n_total,
            "nside": nside,
            "pixel_solid_angle_sr": grid.solid_angle,
            "n_windows": n_windows,
            "n_windows_tested": n_tested,
            "n_windows_edge": n_edge,
            "n_windows_dropped": n_dropped,
            "n_windows_failed": n_failed,
            "t0": t0.isot,
            "tf": tf.isot,
            "T_obs_days": T_obs.to_value(u.day),
            "theta_max_deg": theta_max_deg,
            "latitude_pa_deg": latitude_pa,
            "longitude_pa_deg": longitude_pa,
            "altitude_pa_m": altitude_pa,
        },
    )

    logger.info("Saved results to %s", outdir / "results.npz")
    logger.info("Run finished in %.2f seconds", elapsed)


if __name__ == "__main__":
    seed = 42
    main(seed)
