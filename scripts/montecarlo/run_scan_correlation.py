"""
Correlation-type scan: compare Lambda sensitivity to spatial, temporal,
and spatio-temporal signals.

Three flare-injection cases are run on top of the same background
realisation per trial.  They map onto the two ``EventSample.inject_flare``
modes as follows:

- **Spatio-temporal (ST)** — short ``flare_duration``,
  ``mode="overdensity"``.  Both an excess of events in the window and a
  temporal cluster are introduced, exercising Lambda as intended.

- **Temporal-only (T)** — short ``flare_duration``,
  ``mode="no_overdensity"``.  The flare *replaces* ``n_flare`` events
  in the in-window sample, preserving ``n_sample``.  No spatial
  overdensity at the window-count level; only the temporal pattern is
  anomalous.

- **Spatial-only (S)** — ``flare_duration = T_obs``,
  ``mode="overdensity"``.  Events are added inside the window (spatial
  excess) but the flare's uniform temporal spread leaves the temporal
  pattern indistinguishable from the background.

Output is a ``results.npz`` with the Lambda + p-value + n_sample
arrays for each of the four columns (``bkg`` / ``ST`` / ``T`` / ``S``)
plus a ``metadata.json`` describing the run.
"""

from pathlib import Path

import astropy.units as u
import numpy as np
import scipy.stats as scp
import time
import copy

from astropy.time import Time
from tqdm import tqdm

import spacetimecorr as stc
from spacetimecorr.io import setup_logger, make_run_dir, write_metadata


def main(seed: int) -> None:
    start_time = time.time()

    # ------------------------------------------------------------------
    # Simulation parameters
    # ------------------------------------------------------------------
    n_total = int(1e6)
    n_simulations = int(1e4)
    max_attempts = int(3 * n_simulations)

    # Observation interval
    T_obs = 10 * u.year
    t0 = Time("2013-01-01T00:00:00", scale="utc")
    tf = t0 + T_obs

    # Sky window
    centre = np.array([30.0, 0.0])
    radius = 1.5

    # Pierre Auger Observatory
    latitude_pa = -35.15
    longitude_pa = -69.15
    altitude_pa = 1425

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    project_root = Path(__file__).resolve().parents[2]
    base_dir = project_root / "output" / "scripts"

    outdir, sim_ID = make_run_dir(
        base_dir=base_dir,
        run_code="scan_correlation",
        seed=seed,
    )

    # ------------------------------------------------------------------
    # Logger and metadata
    # ------------------------------------------------------------------
    logger = setup_logger(log_path=outdir / "run.log", name="scan_correlation")

    logger.info("Starting correlation scan run")
    logger.info("Simulation ID: %s", sim_ID)
    logger.info("Output directory: %s", outdir)
    logger.info("Seed: %d", seed)

    # ------------------------------------------------------------------
    # RNGs and models
    # ------------------------------------------------------------------
    rng_manager = stc.RNGManager(seed=seed)
    rng_events = rng_manager.get("events")
    rng_exposure = rng_manager.get("exposure")
    rng_flare = rng_manager.get("flare")

    window = stc.SkyWindow(centre=centre, radius=radius)
    observatory = stc.Observatory(
        latitude=latitude_pa, longitude=longitude_pa, altitude=altitude_pa,
    )
    exposure_model = stc.ExposureModel(
        observatory=observatory, t0=t0, tf=tf, rng=rng_exposure,
    )

    expected_n = window.expected_n_in_window(n_total, exposure_model)

    # Flare design (constant across realisations)
    flare_duration_ST_T = 30 * u.day        # short flare → temporal cluster
    flare_duration_S = T_obs                # long flare → temporally uniform
    flare_sigma = 1.0                       # deg
    mu_flare = 0.2 * expected_n             # mean flare multiplicity per trial

    logger.info("expected_n=%g, mu_flare=%g", expected_n, mu_flare)

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------
    lambda_bkg = []
    lambda_ST = []
    lambda_T = []
    lambda_S = []

    n_sample_bkg = []
    n_sample_ST = []
    n_sample_T = []
    n_sample_S = []

    n_success = 0
    n_failures = 0
    attempt = 0
    n_zero_flare = 0

    pbar = tqdm(total=n_simulations, desc="Successful simulations")

    # ------------------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------------------
    while n_success < n_simulations and attempt < max_attempts:
        attempt += 1

        try:
            # --- Background-only ---
            bkg_sample = stc.EventSample.in_window(
                window=window,
                n_total=n_total,
                exposure_model=exposure_model,
                t0=t0, tf=tf,
                rng=rng_events,
            )
            bkg_sample.assign_directional_exposure(
                window=window, exposure_model=exposure_model,
            )
            lambda_stat_bkg = stc.lambda_estimator(sample=bkg_sample)

            # Draw flare multiplicity once per trial; the same n_flare is
            # used for ST, T, and S.
            n_flare = int(scp.poisson.rvs(mu_flare, random_state=rng_flare))

            if n_flare == 0:
                # The three injection cases collapse to the background-only
                # subsample; record the background result for each so the
                # MC statistics stay unbiased.
                logger.info(
                    "Simulation attempt %d: drawn flare multiplicity is zero "
                    "(mu=%.3f). No flare injected.",
                    attempt, mu_flare,
                )
                n_zero_flare += 1

                lambda_bkg.append(lambda_stat_bkg)
                lambda_ST.append(lambda_stat_bkg)
                lambda_T.append(lambda_stat_bkg)
                lambda_S.append(lambda_stat_bkg)
                n_sample_bkg.append(bkg_sample.n_sample)
                n_sample_ST.append(bkg_sample.n_sample)
                n_sample_T.append(bkg_sample.n_sample)
                n_sample_S.append(bkg_sample.n_sample)
                n_success += 1
                pbar.update(1)
                continue

            # --- ST: spatio-temporal (overdensity, short flare) ---
            flare_ST = stc.Flare(
                n_flare=n_flare, duration=flare_duration_ST_T,
                t0=t0, tf=tf, centre=window.centre,
                exposure_model=exposure_model, rng=rng_flare,
            )
            flare_ST.generate_in_window(window=window, sigma=flare_sigma)

            sample_ST = copy.deepcopy(bkg_sample)
            sample_ST.inject_flare(flare=flare_ST, mode="overdensity")
            sample_ST.assign_directional_exposure(
                window=window, exposure_model=exposure_model,
            )
            lambda_stat_ST = stc.lambda_estimator(sample=sample_ST)

            # --- T: temporal-only (no_overdensity, short flare, same flare as ST) ---
            sample_T = copy.deepcopy(bkg_sample)
            sample_T.inject_flare(flare=flare_ST, mode="no_overdensity")
            sample_T.assign_directional_exposure(
                window=window, exposure_model=exposure_model,
            )
            lambda_stat_T = stc.lambda_estimator(sample=sample_T)

            # --- S: spatial-only (overdensity, long flare with no temporal cluster) ---
            flare_S = stc.Flare(
                n_flare=n_flare, duration=flare_duration_S,
                t0=t0, tf=tf, centre=window.centre,
                exposure_model=exposure_model, rng=rng_flare,
            )
            flare_S.generate_in_window(window=window, sigma=flare_sigma)

            sample_S = copy.deepcopy(bkg_sample)
            sample_S.inject_flare(flare=flare_S, mode="overdensity")
            sample_S.assign_directional_exposure(
                window=window, exposure_model=exposure_model,
            )
            lambda_stat_S = stc.lambda_estimator(sample=sample_S)

            # Record only after the full chain succeeds.
            lambda_bkg.append(lambda_stat_bkg)
            lambda_ST.append(lambda_stat_ST)
            lambda_T.append(lambda_stat_T)
            lambda_S.append(lambda_stat_S)
            n_sample_bkg.append(bkg_sample.n_sample)
            n_sample_ST.append(sample_ST.n_sample)
            n_sample_T.append(sample_T.n_sample)
            n_sample_S.append(sample_S.n_sample)

            n_success += 1
            pbar.update(1)

        except RuntimeError:
            n_failures += 1
            logger.exception(
                "Simulation attempt %d failed "
                "(successes=%d, failures=%d)",
                attempt, n_success, n_failures,
            )
            continue

    pbar.close()

    logger.info(
        "Run finished: attempts=%d, successes=%d, failures=%d",
        attempt, n_success, n_failures,
    )
    if attempt > 0:
        logger.info(
            "Zero-flare realizations: %d / %d (%.2f%%)",
            n_zero_flare, attempt, 100.0 * n_zero_flare / attempt,
        )

    # ------------------------------------------------------------------
    # Final checks
    # ------------------------------------------------------------------
    if n_success == 0:
        raise RuntimeError(
            f"All simulation attempts failed. See log file: {outdir / 'run.log'}"
        )

    if n_success < n_simulations:
        warning_msg = (
            f"Requested {n_simulations} successful simulations, "
            f"but only obtained {n_success} before reaching "
            f"max_attempts={max_attempts}."
        )
        logger.warning(warning_msg)
        print(f"Warning: {warning_msg}")

    # ------------------------------------------------------------------
    # Convert to arrays and compute p-values
    # ------------------------------------------------------------------
    lambda_bkg = np.array(lambda_bkg)
    lambda_ST = np.array(lambda_ST)
    lambda_T = np.array(lambda_T)
    lambda_S = np.array(lambda_S)

    n_sample_bkg = np.array(n_sample_bkg)
    n_sample_ST = np.array(n_sample_ST)
    n_sample_T = np.array(n_sample_T)
    n_sample_S = np.array(n_sample_S)

    p_values_bkg = stc.lambda_marginal_sf(lambda_bkg, expected_n)
    p_values_ST = stc.lambda_marginal_sf(lambda_ST, expected_n)
    p_values_T = stc.lambda_marginal_sf(lambda_T, expected_n)
    p_values_S = stc.lambda_marginal_sf(lambda_S, expected_n)

    # ------------------------------------------------------------------
    # Save outputs and metadata
    # ------------------------------------------------------------------
    np.savez_compressed(
        outdir / "results.npz",
        lambda_bkg=lambda_bkg,
        lambda_ST=lambda_ST,
        lambda_T=lambda_T,
        lambda_S=lambda_S,
        n_sample_bkg=n_sample_bkg,
        n_sample_ST=n_sample_ST,
        n_sample_T=n_sample_T,
        n_sample_S=n_sample_S,
        p_values_bkg=p_values_bkg,
        p_values_ST=p_values_ST,
        p_values_T=p_values_T,
        p_values_S=p_values_S,
    )

    elapsed = time.time() - start_time

    write_metadata(
        outdir=outdir,
        metadata={
            "script": Path(__file__).name,
            "run_code": "scan_correlation",
            "seed": seed,
            "runtime_seconds": elapsed,
            "n_total": n_total,
            "expected_n": expected_n,
            "n_simulations_requested": n_simulations,
            "n_simulations_successful": n_success,
            "max_attempts": max_attempts,
            "n_zero_flare": int(n_zero_flare),
            "time": {
                "t0": t0.isot,
                "tf": tf.isot,
                "T_obs_days": T_obs.to_value(u.day),
            },
            "window": {
                "centre_deg": centre.tolist(),
                "radius_deg": radius,
            },
            "observatory": {
                "latitude_deg": latitude_pa,
                "longitude_deg": longitude_pa,
                "altitude_m": altitude_pa,
            },
            "flare": {
                "duration_ST_T_days": flare_duration_ST_T.to_value(u.day),
                "duration_S_days": flare_duration_S.to_value(u.day),
                "sigma_deg": flare_sigma,
                "mu_flare": mu_flare,
            },
            "modes": {
                "ST": "overdensity",
                "T":  "no_overdensity",
                "S":  "overdensity",
            },
        },
    )

    logger.info("Saved results to %s", outdir / "results.npz")
    logger.info(f"Simulation finished in {elapsed:.2f} seconds")


if __name__ == "__main__":
    seed = 42
    main(seed)
