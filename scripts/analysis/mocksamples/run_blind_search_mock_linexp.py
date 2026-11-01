"""
Blind all-sky search on a mock data sample over the HEALPix pixels at
resolution ``nside``, with Lambda built from the CDF of the gaps for a
fixed observation period instead of the exponential CDF.

Same as ``run_blind_search_mock.py`` except for the windows and Lambda:

    1. Read the sample (``inputs/mocksamples/dataset_stcwg_vert/``).
    2. Windows: the HEALPix pixels at ``nside`` whose centre lies in the
       FoV. Those within one pixel radius of the FoV edge are skipped.
    3. For every window, the normalised cumulative exposure of its events
       at the pixel centre (``ExposureModel.norm_cumul_exposure``), sorted,
       and the gaps between consecutive values, scaled to
       ``x = gap * mu / E_max``. Two statistics:

           Lambda     = -sum log F(x_i),   F(x) = 1 - exp(-x) + x exp(-x) / mu
                                                 (stc.lambda_estimator_suppressed_cdf)
           Lambda_exp = -sum log(1 - exp(-x_i))  (stc.lambda_estimator)

       ``F`` is the CDF of the density ``(mu + 1 - x) / mu * exp(-x)``.
       Both get their p-value from ``lambda_marginal_sf``, with the Poisson
       p-value and the significances as in ``run_blind_search_mock.py``.

``results.npz`` and ``metadata.json`` have the keys of
``run_blind_search_mock.py``, with Lambda the new statistic, so
``plot_blind_search_mock.py`` reads them as they are. The original
statistic is saved as ``lambda_array_exp``, ``pvalues_lambda_exp`` and
``sigma_lambda_exp``.
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import healpy as hp
import numpy as np

from tqdm import tqdm

import spacetimecorr as stc
from spacetimecorr.io import (
    setup_logger, make_run_dir, write_metadata, load_active_area, load_mock_sample,
)


def main(dataset: str, nside: int = 32) -> Path:
    """Run the blind search on one mock sample, e.g. ``dataset="dataset_0"``.

    The run is written to
    ``output/analysis/mocksamples/blind_search_mock_linexp/<dataset>_<timestamp>/``,
    whose path is returned.
    """

    start_time = time.time()

    # ------------------------------------------------------------------
    # Analysis parameters
    # ------------------------------------------------------------------
    sigma_threshold = 5.0          # windows above this significance are reported

    # Pierre Auger Observatory
    latitude_pa = -35.15
    longitude_pa = -69.15
    altitude_pa = 1425
    theta_max_deg = 60.0           # zenith-angle cut defining the field of view

    project_root = Path(__file__).resolve().parents[3]
    input_path = (
        project_root / "inputs" / "mocksamples" / "dataset_stcwg_vert" / f"{dataset}.dat"
    )
    active_area_path = (
        project_root / "inputs" / "active_area" / "ActiveArea_SD1500_UB_6T5.npz"
    )

    # No random draws happen in this analysis; the generator is only
    # required by the ExposureModel and EventSample constructors.
    rng = np.random.default_rng(0)

    # ------------------------------------------------------------------
    # Output directory and logger
    # ------------------------------------------------------------------
    outdir, _ = make_run_dir(
        base_dir=project_root / "output" / "analysis" / "mocksamples",
        run_code="blind_search_mock_linexp",
        seed=0,
        submission_id=f"{dataset}_{datetime.now():%Y%m%d_%H%M%S}",
    )
    logger = setup_logger(log_path=outdir / "run.log", name="blind_search_mock_linexp")

    logger.info("Starting blind search on mock sample %s", dataset)
    logger.info("Input: %s", input_path)
    logger.info("Output directory: %s", outdir)

    # ------------------------------------------------------------------
    # 1) Read the sample
    # ------------------------------------------------------------------
    times, ra, dec = load_mock_sample(input_path)
    n_total = ra.size

    # The observation interval is the span of the sample itself.
    t0 = times.min()
    tf = times.max()
    T_obs_days = float((tf - t0).to_value("day"))

    logger.info(
        "Read %d events | t0 = %s | tf = %s | T_obs = %.2f d",
        n_total, t0.isot, tf.isot, T_obs_days,
    )

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    observatory = stc.Observatory(
        latitude=latitude_pa, longitude=longitude_pa, altitude=altitude_pa,
        area=load_active_area(active_area_path),
    )

    exposure_model = stc.ExposureModel(
        observatory=observatory, t0=t0, tf=tf, rng=rng,
        theta_max_deg=theta_max_deg,
    )

    sample = stc.EventSample._from_arrays(
        ra=ra, dec=dec, n_total=n_total, t0=t0, tf=tf, rng=rng,
        spatial_type="data", expected_n=float(n_total),
        exposure_model=exposure_model,
    )

    # ------------------------------------------------------------------
    # 2) Windows: the HEALPix pixels inside the FoV
    # ------------------------------------------------------------------
    grid = stc.HealpixGrid(exposure_model, nside=nside)
    n_windows = len(grid)

    # Windows lying entirely inside the FoV. The others straddle the FoV
    # edge and are skipped in the scan.
    window_radius_deg = float(np.degrees(hp.max_pixrad(nside)))
    inside_fov = (
        np.abs(grid.centres[:, 1] - latitude_pa) + window_radius_deg < theta_max_deg
    )
    n_edge = int(np.count_nonzero(~inside_fov))
    logger.info(
        "Built grid: %d windows, the nside=%d HEALPix pixels inside the FoV "
        "(%.2f deg^2 each). %d straddle the FoV edge and are skipped.",
        n_windows, nside, grid.solid_angle * np.degrees(1.0) ** 2, n_edge,
    )

    pixel_of_event = grid.pixel_of(ra, dec)

    # ------------------------------------------------------------------
    # Storage (one entry per window, aligned with `grid`)
    # ------------------------------------------------------------------
    lambda_array = np.full(n_windows, np.nan)
    lambda_array_exp = np.full(n_windows, np.nan)
    n_sample_array = np.zeros(n_windows, dtype=int)
    expected_n_array = np.full(n_windows, np.nan)   # NaN for windows not tested
    pvalues_lambda = np.full(n_windows, np.nan)
    pvalues_lambda_exp = np.full(n_windows, np.nan)
    pvalues_poisson = np.full(n_windows, np.nan)

    n_dropped = 0   # windows with fewer than 2 events (routine)
    n_failed = 0    # windows with >= 2 events that still raised

    # ------------------------------------------------------------------
    # 3) Scan
    # ------------------------------------------------------------------
    # disable=None turns the bar off when not on a terminal.
    for i, window in enumerate(
        tqdm(grid, total=n_windows, desc="Windows scanned", disable=None)
    ):
        mask = pixel_of_event == window.ipix
        n_sample_array[i] = np.count_nonzero(mask)

        if not inside_fov[i]:
            continue

        if n_sample_array[i] < 2:
            n_dropped += 1
            continue

        try:
            # The exposure of the events comes first: its pass over the A(t)
            # steps also stores the exposure over the whole run in the
            # model's cache, which the expected count (set by
            # select_subsample) and E_max then read from there.
            exposure = exposure_model.norm_cumul_exposure(times[mask], window.centre)

            subsample = sample.select_subsample(window, mask=mask)
            subsample.exposure = exposure
            mu = subsample.expected_n
            subsample.expected_exposure_rate = (
                mu / exposure_model.max_norm_cumul_exposure(window.centre)
            )
            expected_n_array[i] = mu

            lambda_array[i] = stc.lambda_estimator_suppressed_cdf(
                exposure, subsample.expected_exposure_rate, mu,
            )
            lambda_array_exp[i] = stc.lambda_estimator(sample=subsample)

            # Clipped at 1, as in run_blind_search_mock.py.
            pvalues_lambda[i] = min(1.0, stc.lambda_marginal_sf(lambda_array[i], mu))
            pvalues_lambda_exp[i] = min(
                1.0, stc.lambda_marginal_sf(lambda_array_exp[i], mu),
            )
            pvalues_poisson[i] = stc.poisson_mid_p_value(n_sample_array[i], mu)

        except ValueError:
            lambda_array[i] = np.nan
            lambda_array_exp[i] = np.nan
            pvalues_lambda[i] = np.nan
            pvalues_lambda_exp[i] = np.nan
            n_failed += 1
            logger.exception(
                "Window %d (pixel %d, centre=[%.2f, %.2f], n_sample=%d) failed; "
                "recorded as NaN.",
                i, window.ipix, window.centre[0], window.centre[1],
                n_sample_array[i],
            )

    n_tested = n_windows - n_edge - n_dropped - n_failed
    logger.info(
        "Scan finished: %d windows | %d tested | %d on the FoV edge | "
        "%d dropped (n < 2) | %d failed.",
        n_windows, n_tested, n_edge, n_dropped, n_failed,
    )

    # ------------------------------------------------------------------
    # Significances and pixels above the threshold
    # ------------------------------------------------------------------
    tested = np.isfinite(pvalues_lambda)
    sigma_lambda = np.full(n_windows, np.nan)
    sigma_lambda_exp = np.full(n_windows, np.nan)
    sigma_poisson = np.full(n_windows, np.nan)
    sigma_lambda[tested] = stc.pvalue_to_sigma(pvalues_lambda[tested])
    sigma_lambda_exp[tested] = stc.pvalue_to_sigma(pvalues_lambda_exp[tested])
    sigma_poisson[tested] = stc.pvalue_to_sigma(pvalues_poisson[tested])

    # Windows above the threshold, most significant first, for the metadata.
    above_threshold = {}
    for name, sigma in (
        ("lambda", sigma_lambda), ("lambda_exp", sigma_lambda_exp),
        ("poisson", sigma_poisson),
    ):
        above = np.flatnonzero(sigma > sigma_threshold)     # NaN compares False
        above = above[np.argsort(-sigma[above])]
        above_threshold[name] = [
            {
                "pixel": int(grid.pixels[j]),
                "ra_deg": float(grid.centres[j, 0]),
                "dec_deg": float(grid.centres[j, 1]),
                "n_sample": int(n_sample_array[j]),
                "expected_n": float(expected_n_array[j]),
                "sigma": float(sigma[j]),
            }
            for j in above
        ]

    # Probability of at least one window above the threshold by chance, for
    # n_tested independent windows with uniform p-values. The windows do not
    # overlap; the Poisson mid-p value is still discrete.
    p_threshold = float(stc.sigma_to_pvalue(sigma_threshold))
    prob_any_above = 1.0 - (1.0 - p_threshold) ** n_tested

    # Number of windows above 3, 4 and 5 sigma in each test, and the number
    # expected by chance, n_tested * p(sigma).
    sigma_table = {}
    for threshold in (3.0, 4.0, 5.0):
        sigma_table[f"{threshold:g}sigma"] = {
            "n_lambda": int(np.count_nonzero(sigma_lambda > threshold)),
            "n_lambda_exp": int(np.count_nonzero(sigma_lambda_exp > threshold)),
            "n_poisson": int(np.count_nonzero(sigma_poisson > threshold)),
            "n_expected": n_tested * float(stc.sigma_to_pvalue(threshold)),
        }

    # ------------------------------------------------------------------
    # Save outputs and metadata
    # ------------------------------------------------------------------
    np.savez_compressed(
        outdir / "results.npz",
        pixels=grid.pixels,                       # (W,)   HEALPix RING index of each window
        centres=grid.centres,                     # (W, 2) window centres [RA, Dec] (deg)
        inside_fov=inside_fov,                    # (W,)   False for windows on the FoV edge (skipped)
        expected_n=expected_n_array,              # (W,)   exposure-weighted expected count (NaN if not tested)
        lambda_array=lambda_array,                # (W,)   Lambda with F(x) = 1 - e^-x + x e^-x / mu
        lambda_array_exp=lambda_array_exp,        # (W,)   Lambda with the exponential CDF (stc.lambda_estimator)
        n_sample_array=n_sample_array,            # (W,)   event count per window
        pvalues_lambda=pvalues_lambda,            # (W,)   marginal p-value of Lambda
        pvalues_lambda_exp=pvalues_lambda_exp,    # (W,)   marginal p-value of Lambda_exp
        pvalues_poisson=pvalues_poisson,          # (W,)   Poisson counting mid-p value
        sigma_lambda=sigma_lambda,                # (W,)   one-sided significance, Lambda
        sigma_lambda_exp=sigma_lambda_exp,        # (W,)   one-sided significance, Lambda_exp
        sigma_poisson=sigma_poisson,              # (W,)   one-sided significance, Poisson
    )

    elapsed = time.time() - start_time

    write_metadata(
        outdir=outdir,
        metadata={
            "script": Path(__file__).name,
            "run_code": "blind_search_mock_linexp",
            "dataset": dataset,
            "input_path": str(input_path),
            "active_area_path": str(active_area_path),
            "runtime_seconds": elapsed,
            "n_total": n_total,
            "nside": nside,
            "window_radius_deg": window_radius_deg,
            "window_solid_angle_sr": grid.solid_angle,
            "n_windows": n_windows,
            "n_windows_tested": n_tested,
            "n_windows_edge": n_edge,
            "n_windows_dropped": n_dropped,
            "n_windows_failed": n_failed,
            "sigma_threshold": sigma_threshold,
            "p_threshold": p_threshold,
            "prob_any_above_threshold": prob_any_above,
            "pixels_above_threshold": above_threshold,
            "sigma_table": sigma_table,
            "t0": t0.isot,
            "tf": tf.isot,
            "T_obs_days": T_obs_days,
            "theta_max_deg": theta_max_deg,
            "latitude_pa_deg": latitude_pa,
            "longitude_pa_deg": longitude_pa,
            "altitude_pa_m": altitude_pa,
            "max_area_km2": observatory.max_area,
        },
    )

    logger.info("Saved results to %s", outdir / "results.npz")
    logger.info("Run finished in %.2f seconds", elapsed)
    return outdir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Blind search on one mock sample over the HEALPix pixels, with "
            "Lambda built from the CDF of the gaps for a fixed observation period."
        )
    )
    parser.add_argument(
        "--dataset", type=str, default="dataset_0",
        help="Mock sample to analyse, the file name without .dat.",
    )
    parser.add_argument(
        "--nside", type=int, default=32,
        help="HEALPix resolution of the windows.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(dataset=args.dataset, nside=args.nside)
