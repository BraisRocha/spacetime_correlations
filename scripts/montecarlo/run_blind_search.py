"""
Blind all-sky flare search over a HEALPix SkyGrid.

Unlike the targeted search (one window per source), the blind search tiles
the whole field of view with a :class:`~spacetimecorr.SkyGrid` built from
HEALPix pixel centres and scans every window. The pipeline is:

    1. Generate one isotropic full-sky sample of ``n_total`` events
       (``EventSample.full_sky``) — no windows involved yet.
    2. Inject flares at catalogue source positions. Each flare is generated
       window-free with ``Flare.generate`` (a Gaussian cluster around the
       source, thinned by detection only) and injected with
       ``mode="no_overdensity"`` so ``n_total`` is preserved — only the
       spatial/temporal signature of the flares is imprinted. ``inject_flare``
       is called once per flare and only ever displaces background events, so
       several flares accumulate cleanly in the same sample.
    3. Build the blind HEALPix grid of fixed-radius windows covering the FoV.
    4. For every window, select the events inside it with ``SkyWindow.contains``
       (via ``EventSample.select_subsample``), attach directional exposure and
       compute the Lambda statistic and a Poisson counting statistic with their
       p-values.

The collected outputs are length-``n_windows`` arrays of the two statistics
and their p-values, plus the window centres and radii (and, for reference,
the injected flare positions, durations and multiplicities).

Known simplification (deliberate, to be addressed later)
--------------------------------------------------------
The full-sky selection uses ``window.expected_n_in_window(n_total)`` *without*
an exposure model, so the expected background count is the bare sky-fraction
value and is identical for every equal-radius window. The exposure still
enters the per-window directional-exposure *sampling*, but not the expected
count, so absolute p-values are not yet exposure-correct.
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
    """Run a blind all-sky flare search.

    Generates one isotropic full-sky sample, injects window-free flares at
    catalogue source positions (no-overdensity, so ``n_total`` is preserved),
    tiles the FoV with a HEALPix :class:`~spacetimecorr.SkyGrid`, and for every
    window selects the in-window events and computes the Lambda and Poisson
    statistics with their p-values.
    """

    start_time = time.time()

    # ------------------------------------------------------------------
    # Simulation parameters
    # ------------------------------------------------------------------
    n_total = int(5e5)
    max_attempts = 20              # retries for stochastic flare generation

    # Observational interval
    T_obs = 10 * u.year
    t0 = Time("2013-01-01T00:00:00", scale="utc")
    tf = t0 + T_obs

    # Pierre Auger Observatory
    latitude_pa = -35.15
    longitude_pa = -69.15
    altitude_pa = 1425
    theta_max_deg = 60.0           # zenith-angle cut defining the field of view
    observatory_resolution = 1.0   # angular resolution [deg]

    # ------------------------------------------------------------------
    # RNG streams
    # ------------------------------------------------------------------
    rng_manager = stc.RNGManager(seed=seed)
    rng_sample = rng_manager.get("sample")        # full-sky isotropic draw + removals
    rng_layout = rng_manager.get("flare_layout")  # per-flare durations
    rng_flare = rng_manager.get("flare")          # flare generation + multiplicity
    rng_exposure = rng_manager.get("exposure")    # directional-exposure sampling

    # ------------------------------------------------------------------
    # Flare layout (positions, durations, intensity)
    # ------------------------------------------------------------------
    flare_duration_range = (1 * u.hour, 30 * u.day)   # per-flare duration, drawn uniformly
    flare_sigma = 1.0              # angular spread of each flare [deg]
    search_radius = 1.05 * observatory_resolution     # SNR-optimal search radius [deg]

    # Flare positions are taken from a real source catalogue. Sources outside
    # the FoV are rejected, with an extra margin below the Dec edges.
    project_root = Path(__file__).resolve().parents[2]
    catalog_path = project_root / "inputs" / "catalogs" / "6-UNID_RA_Dec_Flux_Dist.cat"
    dec_margin_deg = 3.0

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

    # Expected background count inside one search-radius cap (sky-fraction
    # only, matching the full-sky selection). Shared by all flares, so the
    # flare multiplicity mu = (S/N) * sqrt(expected_n_region) is common.
    search_sky_fraction = 0.5 * (1.0 - np.cos(np.deg2rad(search_radius)))
    expected_n_region = n_total * search_sky_fraction

    flare_intensity = 0.5          # shared S/N = n_flare / sqrt(expected_n_region)
    mu_flare = flare_intensity * np.sqrt(expected_n_region)

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    base_dir = project_root / "output" / "scripts"

    outdir, sim_ID = make_run_dir(
        base_dir=base_dir,
        run_code="blind_search",
        seed=seed,
    )

    # ------------------------------------------------------------------
    # Logger
    # ------------------------------------------------------------------
    logger = setup_logger(log_path=outdir / "run.log", name="blind_search")

    logger.info("Starting blind search run")
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
        "Flare layout | radius=%.2f deg, duration in [%s, %s], sigma=%.2f deg, "
        "S/N=%.3f -> mu_flare=%.3f (expected_n_region=%.2f)",
        search_radius, flare_duration_range[0], flare_duration_range[1],
        flare_sigma, flare_intensity, mu_flare, expected_n_region,
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
    # 1) Full-sky isotropic sample
    # ------------------------------------------------------------------
    sample = stc.EventSample.full_sky(
        n_total=n_total, t0=t0, tf=tf, rng=rng_sample,
    )
    logger.info("Generated full-sky sample of %d events.", sample.n_sample)

    # ------------------------------------------------------------------
    # 2) Generate window-free flares and inject them (no-overdensity)
    # ------------------------------------------------------------------
    # Each flare is a Gaussian cluster around its source, thinned by detection
    # only (Flare.generate). Generation can fail stochastically (RuntimeError
    # when the cluster cannot be filled within the draw budget); such flares
    # are retried and, if still failing, skipped. Injection removes only
    # background events, so flares accumulate without displacing one another
    # and the total count stays n_total.
    flare_n_injected = np.zeros(number_of_flares, dtype=int)
    n_zero_flare = 0
    n_failed_flares = 0

    for i in range(number_of_flares):
        n_flare = int(scp.poisson.rvs(mu_flare, random_state=rng_flare))
        if n_flare == 0:
            n_zero_flare += 1
            continue

        for attempt in range(1, max_attempts + 1):
            try:
                flare = stc.Flare(
                    n_flare=n_flare,
                    duration=flare_durations[i],
                    t0=t0, tf=tf,
                    centre=flare_centres[i],
                    exposure_model=exposure_model,
                    rng=rng_flare,
                )
                flare.generate(sigma=flare_sigma)
                sample.inject_flare(flare=flare, mode="no_overdensity")
                flare_n_injected[i] = n_flare
                break
            except RuntimeError:
                logger.exception(
                    "Flare %d generation attempt %d/%d failed "
                    "(centre=[%.2f, %.2f]); retrying.",
                    i, attempt, max_attempts,
                    flare_centres[i, 0], flare_centres[i, 1],
                )
        else:
            n_failed_flares += 1
            logger.error(
                "Flare %d (centre=[%.2f, %.2f]) failed after %d attempts; "
                "not injected.",
                i, flare_centres[i, 0], flare_centres[i, 1], max_attempts,
            )

    n_injected = int(np.count_nonzero(flare_n_injected))
    n_signal = int(flare_n_injected.sum())
    logger.info(
        "Injected %d flares (%d signal events) | %d zero-multiplicity, "
        "%d generation failures | n_sample=%d (expected %d).",
        n_injected, n_signal, n_zero_flare, n_failed_flares,
        sample.n_sample, n_total,
    )

    # ------------------------------------------------------------------
    # 3) Blind HEALPix grid covering the FoV
    # ------------------------------------------------------------------
    grid = stc.SkyGrid.from_healpix(
        radius=search_radius,
        observatory=observatory,
        theta_max_deg=theta_max_deg,
    )
    n_windows = len(grid)
    logger.info("Built blind HEALPix grid of %d windows.", n_windows)

    # ------------------------------------------------------------------
    # Storage (one entry per window, aligned with `grid`)
    # ------------------------------------------------------------------
    lambda_array = np.full(n_windows, np.nan)
    n_sample_array = np.full(n_windows, np.nan)
    pvalues_lambda = np.full(n_windows, np.nan)
    pvalues_poisson = np.full(n_windows, np.nan)

    # Expected background count per window (sky-fraction only; see module
    # docstring). Equal for all equal-radius windows.
    grid_expected_n = grid.expected_n_in_window(n_total)

    n_failed = 0   # windows whose analysis could not complete

    # ------------------------------------------------------------------
    # 4) Scan: select in-window events and apply the Lambda method
    # ------------------------------------------------------------------
    # The sample is fixed, so a window's only stochastic ingredient is the
    # directional-exposure draw. Failures here (no/<2 events, degenerate
    # exposure) are rare and not helped by retrying the same fixed selection,
    # so a failed window is logged and left as NaN.
    for i, window in enumerate(tqdm(grid, total=n_windows, desc="Windows scanned")):
        try:
            subsample = sample.select_subsample(window)
            subsample.assign_directional_exposure(
                window=window, exposure_model=exposure_model,
            )

            lambda_array[i] = stc.lambda_estimator(sample=subsample)
            n_sample_array[i] = subsample.n_sample

            pvalues_lambda[i] = stc.lambda_marginal_sf(
                lambda_array[i], grid_expected_n[i]
            )
            pvalues_poisson[i] = stc.poisson_mid_p_value(
                n_sample_array[i], grid_expected_n[i]
            )

        except (ValueError, RuntimeError):
            n_failed += 1
            logger.exception(
                "Window %d (centre=[%.2f, %.2f], radius=%.2f deg) failed; "
                "recorded as NaN.",
                i, window.centre[0], window.centre[1], window.radius,
            )

    logger.info("Scan finished: %d windows | %d failed.", n_windows, n_failed)

    # ------------------------------------------------------------------
    # Save outputs and metadata
    # ------------------------------------------------------------------
    np.savez_compressed(
        outdir / "results.npz",
        centres=grid.centres,             # (W, 2) window centres [RA, Dec] (deg)
        radii=grid.radii,                 # (W,)   window radii (deg)
        expected_n=grid_expected_n,       # (W,)   sky-fraction background count
        lambda_array=lambda_array,        # (W,)   Lambda statistic per window
        n_sample_array=n_sample_array,    # (W,)   in-window event count per window
        pvalues_lambda=pvalues_lambda,    # (W,)   marginal Lambda p-value
        pvalues_poisson=pvalues_poisson,  # (W,)   Poisson counting mid-p value
        flare_centres=flare_centres,      # (M, 2) injected flare positions (deg)
        flare_durations_days=flare_durations.to_value(u.day),  # (M,) per-flare duration
        flare_n_injected=flare_n_injected,  # (M,) per-flare event multiplicity
    )

    elapsed = time.time() - start_time

    write_metadata(
        outdir=outdir,
        metadata={
            "script": Path(__file__).name,
            "run_code": "blind_search",
            "seed": seed,
            "runtime_seconds": elapsed,
            "n_total": n_total,
            "n_windows": n_windows,
            "n_failed_windows": n_failed,
            "number_of_flares": number_of_flares,
            "n_injected_flares": n_injected,
            "n_signal_events": n_signal,
            "n_zero_flare": n_zero_flare,
            "n_failed_flares": n_failed_flares,
            "max_attempts": max_attempts,
            "t0": t0.isot,
            "tf": tf.isot,
            "T_obs_days": T_obs.to_value(u.day),
            "search_radius_deg": search_radius,
            "expected_n_region": expected_n_region,
            "flare_duration_range_hours": [
                flare_duration_range[0].to_value(u.hour),
                flare_duration_range[1].to_value(u.hour),
            ],
            "flare_sigma_deg": flare_sigma,
            "flare_intensity": float(flare_intensity),
            "mu_flare": float(mu_flare),
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
