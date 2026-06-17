"""
Targeted multi-flare search over a SkyGrid.

Defines a set of flares at the positions of real catalogue sources that
fall inside the Pierre Auger field of view, each with its own duration
(drawn between 1 hour and 1 month) and builds one circular search window
per flare with a
:class:`~spacetimecorr.SkyGrid`. All flares share the same intensity
(signal-to-noise ratio): it is derived so that, if every window yielded
the same p-value, their Fisher combination would reach a chosen target
significance (see :func:`spacetimecorr.fisher_equal_sigma`). For every
window the script:

    1. simulates in-window isotropy via ``EventSample.in_window`` and
       attaches directional exposure,
    2. injects the flare drawn for that window,
    3. computes the Lambda estimator and a Poisson counting statistic,
       together with their p-values.

The collected outputs are length-``number_of_flares`` arrays of the two
statistics and their p-values, plus the window centres, radii and the
per-flare durations.
"""

import time
from pathlib import Path

import astropy.units as u
import numpy as np
import scipy.stats as scp

from astropy.time import Time
from tqdm import tqdm

import spacetimecorr as stc
from spacetimecorr.io import setup_logger, make_run_dir, write_metadata


def load_fov_sources(
    path: Path,
    latitude: float,
    theta_max_deg: float,
    dec_margin_deg: float = 0.0,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Load source positions from a catalogue and keep those inside the FoV.

    The catalogue is a whitespace-separated text file whose first two columns
    are ``RA`` and ``Dec`` in degrees (further columns are ignored). A source
    at declination ``dec`` culminates at zenith angle ``|dec - latitude|``, so
    it ever enters a field of view of half-aperture ``theta_max_deg`` only when
    ``|dec - latitude| <= theta_max_deg``. Sources outside this band are
    rejected; ``dec_margin_deg`` shrinks the accepted band symmetrically so
    sources sitting right at the FoV edges (where exposure is marginal) are
    dropped too.

    Returns
    -------
    centres : np.ndarray
        ``(M, 2)`` array of surviving ``[RA, Dec]`` positions in degrees.
    dec_bounds : tuple[float, float]
        The ``(dec_min, dec_max)`` band actually applied, in degrees.
    """
    catalog = np.loadtxt(path, usecols=(0, 1))
    catalog = np.atleast_2d(catalog)
    ra, dec = catalog[:, 0], catalog[:, 1]

    dec_min = latitude - theta_max_deg + dec_margin_deg
    dec_max = latitude + theta_max_deg - dec_margin_deg
    mask = (dec >= dec_min) & (dec <= dec_max)

    return np.column_stack((ra[mask], dec[mask])), (dec_min, dec_max)


def main(seed: int) -> None:
    """Run a multi-target flare search.

    Lays down one flare at each catalogue source that falls inside Auger's
    field of view — each with its own duration (1 hour to 1 month) and a
    shared intensity (signal-to-noise ratio) fixed from a target combined
    Fisher significance — and places one circular search window around
    each via a :class:`~spacetimecorr.SkyGrid`. For every window the run
    simulates in-window isotropy, injects the corresponding flare, and
    computes both a Poisson counting statistic and the Lambda estimator.
    It collects, per window, the two statistics with their p-values along
    with the window centres, radii and durations.
    """

    start_time = time.time()

    # ------------------------------------------------------------------
    # Simulation parameters
    # ------------------------------------------------------------------
    n_total = int(5e5)
    max_attempts = 20

    # Observational interval
    T_obs = 10 * u.year
    t0 = Time("2013-01-01T00:00:00", scale="utc")
    tf = t0 + T_obs

    # Pierre Auger Observatory
    latitude_pa = -35.15
    longitude_pa = -69.15
    altitude_pa = 1425
    theta_max_deg = 60.0           # zenith-angle cut defining the field of view
    observatory_resolution = 1. # degree

    # ------------------------------------------------------------------
    # RNG streams (created here so the flare layout is reproducible)
    # ------------------------------------------------------------------
    rng_manager = stc.RNGManager(seed=seed)
    rng_layout = rng_manager.get("flare_layout")   # flare positions / intensities
    rng_sample = rng_manager.get("sample")         # in-window event draw
    rng_exposure = rng_manager.get("exposure")     # directional-exposure sampling
    rng_flare = rng_manager.get("flare")           # flare time/space generation

    # ------------------------------------------------------------------
    # Flare layout and search windows
    # ------------------------------------------------------------------
    flare_duration_range = (1 * u.hour, 30 * u.day)   # per-flare duration, drawn uniformly
    flare_sigma = 1.0              # angular spread of each flare [deg]
    search_radius = 1.05 * observatory_resolution           # search-window radius that maximises SNR

    # Flare positions are taken from a real source catalogue instead of being
    # drawn at random. Sources outside the field of view are rejected, with an
    # extra margin below the upper Dec edge (see `load_fov_sources`).
    project_root = Path(__file__).resolve().parents[2]
    catalog_path = project_root / "inputs" / "catalogs" / "6-UNID_RA_Dec_Flux_Dist.cat"
    dec_margin_deg = 3.0           # reject sources within this margin of the FoV Dec edge

    flare_centres, dec_bounds = load_fov_sources(
        catalog_path,
        latitude=latitude_pa,
        theta_max_deg=theta_max_deg,
        dec_margin_deg=dec_margin_deg,
    )
    dec_min, dec_max = dec_bounds
    number_of_flares = len(flare_centres)

    durations_lo = flare_duration_range[0].to_value(u.s)
    durations_hi = flare_duration_range[1].to_value(u.s)
    flare_durations = rng_layout.uniform(
        durations_lo, durations_hi, size=number_of_flares,
    ) * u.s

    # All flares share the same intensity (S/N = n_flare / sqrt(expected_n)).
    # It is set so that, if every one of the `number_of_flares` windows
    # returned the same p-value, their Fisher combination would reach this
    # target significance. The per-flare Gaussian significance is identified
    # with the S/N ratio of the Poisson counting excess.
    target_fisher_sigma = 2.0
    #flare_intensity = stc.fisher_equal_sigma(target_fisher_sigma, number_of_flares)
    flare_intensity = 0.5 # There is a problem with the derived method that need to be solved
    print(flare_intensity)

    # One circular search window per flare.
    grid = stc.SkyGrid(centres=flare_centres, radii=search_radius)

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    base_dir = project_root / "output" / "scripts"

    outdir, sim_ID = make_run_dir(
        base_dir=base_dir,
        run_code="targeted_search",
        seed=seed,
    )

    # ------------------------------------------------------------------
    # Logger
    # ------------------------------------------------------------------
    logger = setup_logger(log_path=outdir / "run.log", name="targeted_search")

    logger.info("Starting targeted search run")
    logger.info("Simulation ID: %s", sim_ID)
    logger.info("Output directory: %s", outdir)
    logger.info("Seed: %d", seed)
    logger.info(
        "Loaded %d catalogue sources inside the FoV from %s "
        "(theta_max=%.1f deg, Dec margin=%.1f deg -> Dec=[%.2f, %.2f] deg)",
        number_of_flares, catalog_path,
        theta_max_deg, dec_margin_deg, dec_min, dec_max,
    )
    logger.info(
        "Defined %d flares | radius=%.2f deg, duration in [%s, %s], sigma=%.2f deg",
        len(grid), search_radius,
        flare_duration_range[0], flare_duration_range[1],
        flare_sigma,
    )
    logger.info(
        "Target combined Fisher significance=%.2f sigma over %d flares "
        "-> shared S/N=%.3f",
        target_fisher_sigma, number_of_flares, flare_intensity,
    )

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    observatory = stc.Observatory(
        latitude=latitude_pa, longitude=longitude_pa, altitude=altitude_pa,
    )
    exposure_model = stc.ExposureModel(
        observatory=observatory, t0=t0, tf=tf, rng=rng_exposure,
        theta_max_deg=theta_max_deg,
    )

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------
    # One entry per window/flare, aligned with the order of `grid`. NaN marks
    # a window that never completed (see the failure handling below).
    lambda_array = np.full(number_of_flares, np.nan)
    n_sample_array = np.full(number_of_flares, np.nan)

    pvalues_lambda = np.full(number_of_flares, np.nan)
    pvalues_poisson = np.full(number_of_flares, np.nan)

    # Exposure-weighted expected count per window, shape (N,). Fixed across
    # attempts, so it is computed once for the whole grid.
    grid_expected_n = grid.expected_n_in_window(n_total, exposure_model)

    n_zero_flare = 0   # windows whose flare multiplicity drew zero
    n_failed = 0       # windows that exhausted max_attempts without success

    pbar = tqdm(total=number_of_flares, desc="Windows analysed")

    # Each window is analysed independently. An attempt can fail
    # stochastically — most commonly when `Flare.generate_in_window` cannot
    # land enough events inside the cap before hitting its draw budget
    # (raised as RuntimeError). Such attempts are retried up to
    # `max_attempts`; configuration/contract errors (ValueError) are left to
    # propagate so they surface loudly instead of being silently retried.
    for i, window in enumerate(grid):
        success = False
        attempt = 0

        while not success and attempt < max_attempts:
            attempt += 1

            try:
                # --- Background: isotropic events drawn inside the window ---
                sample = stc.EventSample.in_window(
                    window=window,
                    n_total=n_total,
                    exposure_model=exposure_model,
                    t0=t0,
                    tf=tf,
                    rng=rng_sample,
                )

                # --- Flare multiplicity from the per-window S/N ratio ---
                # Intensity is defined as S/N, so the flare count scales with
                # the Poisson noise sqrt(expected_n) of the background.
                mu_flare = flare_intensity * np.sqrt(grid_expected_n[i])
                n_flare = int(scp.poisson.rvs(mu_flare, random_state=rng_flare))

                if n_flare == 0:
                    logger.info(
                        "Window %d (centre=[%.2f, %.2f], radius=%.2f deg): "
                        "flare multiplicity drew zero (mu_flare=%.3f); "
                        "no flare injected.",
                        i, window.centre[0], window.centre[1], window.radius,
                        mu_flare,
                    )
                    n_zero_flare += 1

                else:
                    flare = stc.Flare(
                        n_flare=n_flare,
                        duration=flare_durations[i],
                        t0=t0, tf=tf,
                        centre=window.centre,
                        exposure_model=exposure_model,
                        rng=rng_flare,
                    )
                    flare.generate_in_window(window=window, sigma=flare_sigma)
                    sample.inject_flare(flare=flare, mode="overdensity")

                # --- Attach exposure and compute the statistic ---
                sample.assign_directional_exposure(
                    window=window, exposure_model=exposure_model,
                )

                lambda_array[i] = stc.lambda_estimator(sample=sample)
                n_sample_array[i] = sample.n_sample

                pvalues_lambda[i] = stc.lambda_marginal_sf(
                    lambda_array[i], grid_expected_n[i]
                )
                pvalues_poisson[i] = stc.poisson_mid_p_value(
                    n_sample_array[i], grid_expected_n[i]
                )

                success = True

            except RuntimeError:
                logger.exception(
                    "Window %d attempt %d/%d failed (centre=[%.2f, %.2f], "
                    "radius=%.2f deg); retrying.",
                    i, attempt, max_attempts,
                    window.centre[0], window.centre[1], window.radius,
                )
                continue

        if not success:
            n_failed += 1
            logger.error(
                "Window %d (centre=[%.2f, %.2f], radius=%.2f deg) failed after "
                "%d attempts; recorded as NaN.",
                i, window.centre[0], window.centre[1], window.radius,
                max_attempts,
            )

        pbar.update(1)

    pbar.close()

    logger.info(
        "Loop finished: %d windows | %d failed | %d zero-flare draws.",
        number_of_flares, n_failed, n_zero_flare,
    )

    # ------------------------------------------------------------------
    # Save outputs and metadata
    # ------------------------------------------------------------------
    # Per-window arrays are aligned with the order of `grid`; failed windows
    # appear as NaN throughout. Common (scalar) flare parameters go into the
    # metadata, the per-window ones into the arrays.
    np.savez_compressed(
        outdir / "results.npz",
        centres=grid.centres,             # (N, 2) window centres [RA, Dec] (deg)
        radii=grid.radii,                 # (N,)   window radii (deg)
        flare_durations_days=flare_durations.to_value(u.day),  # (N,) per-flare duration (days)
        flare_intensity=flare_intensity,  # scalar shared S/N ratio
        expected_n=grid_expected_n,       # (N,)   exposure-weighted background count
        lambda_array=lambda_array,        # (N,)   Lambda statistic per window
        n_sample_array=n_sample_array,    # (N,)   in-window event count per window
        pvalues_lambda=pvalues_lambda,    # (N,)   marginal Lambda p-value
        pvalues_poisson=pvalues_poisson,  # (N,)   Poisson counting mid-p value
    )

    elapsed = time.time() - start_time

    write_metadata(
        outdir=outdir,
        metadata={
            "script": Path(__file__).name,
            "run_code": "targeted_search",
            "seed": seed,
            "runtime_seconds": elapsed,
            "n_total": n_total,
            "number_of_flares": number_of_flares,
            "n_failed": n_failed,
            "n_zero_flare": n_zero_flare,
            "max_attempts": max_attempts,
            "t0": t0.isot,
            "tf": tf.isot,
            "T_obs_days": T_obs.to_value(u.day),
            "search_radius_deg": search_radius,
            "flare_duration_range_hours": [
                flare_duration_range[0].to_value(u.hour),
                flare_duration_range[1].to_value(u.hour),
            ],
            "flare_sigma_deg": flare_sigma,
            "target_fisher_sigma": target_fisher_sigma,
            "flare_intensity": float(flare_intensity),
            "theta_max_deg": theta_max_deg,
            "catalog_path": str(catalog_path),
            "dec_margin_deg": dec_margin_deg,
            "dec_bounds_deg": [dec_min, dec_max],
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







