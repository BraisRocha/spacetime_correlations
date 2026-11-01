"""
Blind all-sky search on a mock data sample, over the HEALPix pixels at
resolution ``nside_window``, with the exposure of the events computed in
their sub-pixels at the finer resolution ``nside_exposure``.

Same as ``run_blind_search_mock.py`` except for the windows and the
exposure. The pipeline is:

    1. Read the sample (``inputs/mocksamples/dataset_stcwg_vert/``).
    2. Windows: the HEALPix pixels at ``nside_window`` whose centre lies in
       the FoV. Those within one pixel radius of the FoV edge are skipped.
       Each window is the union of ``(nside_exposure / nside_window)**2``
       sub-pixels at ``nside_exposure``.
    3. For every sub-pixel of the tested windows, compute the normalised
       cumulative exposure of its events at the sub-pixel centre
       (``ExposureModel.norm_cumul_exposure``) and divide it by its value
       over the whole run (``max_norm_cumul_exposure``)::

           u = eps(t) / eps(tf),    0 <= u <= 1

       Under isotropy the events of every sub-pixel are uniform in ``u``.
       ``eps(tf)`` changes with declination across a window, so the values
       ``eps`` of different sub-pixels are not on one scale, but the values
       ``u`` are. The expected count of every sub-pixel, ``mu_sub``, is
       computed as in ``run_blind_search_mock.py``.
    4. For every window, merge the ``u`` of its sub-pixels. Independent
       Poisson processes uniform in ``u`` add up to one, with expected count
       ``mu = sum(mu_sub)``. Lambda is computed on ``u`` with expected rate
       ``mu`` (the window spans ``u`` from 0 to 1), with its p-value, the
       Poisson p-value, and their significances.

Windows above ``sigma_threshold`` in either test are saved, with their
direction, in ``metadata.json``. ``results.npz`` and ``metadata.json`` have
the keys of ``run_blind_search_mock.py``, with ``nside`` the resolution of
the windows, so ``plot_blind_search_mock.py`` reads them as they are.
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


def sub_pixels(ipix: int, nside: int, nside_sub: int) -> np.ndarray:
    """RING indices at ``nside_sub`` of the sub-pixels of pixel ``ipix``
    (RING) at ``nside``.

    In NEST ordering the sub-pixels of pixel ``p`` are
    ``p * k ... (p + 1) * k - 1``, with ``k = (nside_sub / nside)**2``.
    """
    k = (nside_sub // nside) ** 2
    nest = hp.ring2nest(nside, ipix)
    return hp.nest2ring(nside_sub, nest * k + np.arange(k))


def main(dataset: str, submission_id: str | None = None) -> None:
    """Run the blind search on one mock sample, e.g. ``dataset="dataset_0"``.

    With ``submission_id`` (an HTCondor batch), the run is written to
    ``output/analysis/mocksamples/blind_search_mock_nside8/<submission_id>/<dataset>/``;
    without it, to
    ``output/analysis/mocksamples/blind_search_mock_nside8/<dataset>_<timestamp>/``.
    """

    start_time = time.time()

    # ------------------------------------------------------------------
    # Analysis parameters
    # ------------------------------------------------------------------
    nside_window = 8               # HEALPix resolution of the windows
    nside_exposure = 32            # HEALPix resolution of the exposure
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
    if submission_id is None:
        outdir, _ = make_run_dir(
            base_dir=project_root / "output" / "analysis" / "mocksamples",
            run_code="blind_search_mock_nside8",
            seed=0,
            submission_id=f"{dataset}_{datetime.now():%Y%m%d_%H%M%S}",
        )
    else:
        outdir, _ = make_run_dir(
            base_dir=project_root / "output" / "analysis" / "mocksamples" / "blind_search_mock_nside8",
            run_code=submission_id,
            seed=0,
            submission_id=dataset,
        )
    logger = setup_logger(log_path=outdir / "run.log", name="blind_search_mock_nside8")

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

    # ------------------------------------------------------------------
    # 2) Windows: the pixels at nside_window inside the FoV
    # ------------------------------------------------------------------
    grid = stc.HealpixGrid(exposure_model, nside=nside_window)
    n_windows = len(grid)

    # Windows lying entirely inside the FoV. The others straddle the FoV
    # edge and are skipped in the scan.
    window_radius_deg = float(np.degrees(hp.max_pixrad(nside_window)))
    inside_fov = (
        np.abs(grid.centres[:, 1] - latitude_pa) + window_radius_deg < theta_max_deg
    )
    n_edge = int(np.count_nonzero(~inside_fov))
    logger.info(
        "Built grid: %d windows, the nside=%d HEALPix pixels inside the FoV "
        "(%.2f deg^2 each, each made of %d sub-pixels at nside=%d). "
        "%d straddle the FoV edge and are skipped.",
        n_windows, nside_window, grid.solid_angle * np.degrees(1.0) ** 2,
        (nside_exposure // nside_window) ** 2, nside_exposure, n_edge,
    )

    window_of_event = hp.ang2pix(nside_window, ra, dec, lonlat=True)
    sub_pixel_of_event = hp.ang2pix(nside_exposure, ra, dec, lonlat=True)

    # ------------------------------------------------------------------
    # 3) u = eps(t) / eps(tf) of every event, in its sub-pixel, and the
    #    expected count of every sub-pixel
    # ------------------------------------------------------------------
    u_events = np.full(n_total, np.nan)            # NaN for events not in a tested window
    mu_sub = np.full(hp.nside2npix(nside_exposure), np.nan)

    subs_tested = np.concatenate([
        sub_pixels(ipix, nside_window, nside_exposure)
        for ipix in grid.pixels[inside_fov]
    ])

    # disable=None turns the bar off when not on a terminal (HTCondor jobs).
    for sub in tqdm(subs_tested, desc="Sub-pixels", disable=None):
        centre = np.array(hp.pix2ang(nside_exposure, sub, lonlat=True))
        mask = sub_pixel_of_event == sub

        # The exposure of the events comes first: its pass over the A(t)
        # steps also stores the exposure over the whole run in the model's
        # cache, which eps(tf) and the expected count then read from there.
        if np.any(mask):
            eps = exposure_model.norm_cumul_exposure(times[mask], centre)
            u_events[mask] = eps / exposure_model.max_norm_cumul_exposure(centre)

        mu_sub[sub] = stc.HealpixWindow(
            nside=nside_exposure, ipix=int(sub),
        ).expected_n_in_window(n_total, exposure_model)

    # ------------------------------------------------------------------
    # Storage (one entry per window, aligned with `grid`)
    # ------------------------------------------------------------------
    lambda_array = np.full(n_windows, np.nan)
    n_sample_array = np.zeros(n_windows, dtype=int)
    expected_n_array = np.full(n_windows, np.nan)   # NaN for windows not tested
    pvalues_lambda = np.full(n_windows, np.nan)
    pvalues_poisson = np.full(n_windows, np.nan)

    n_dropped = 0   # windows with fewer than 2 events (routine)
    n_failed = 0    # windows with >= 2 events that still raised

    # ------------------------------------------------------------------
    # 4) Scan: merge the sub-pixels of every window, then the Lambda method
    # ------------------------------------------------------------------
    for i, ipix in enumerate(grid.pixels):
        mask = window_of_event == ipix
        n_sample_array[i] = np.count_nonzero(mask)

        if not inside_fov[i]:
            continue

        if n_sample_array[i] < 2:
            n_dropped += 1
            continue

        try:
            mu = float(np.sum(mu_sub[sub_pixels(ipix, nside_window, nside_exposure)]))
            expected_n_array[i] = mu

            # u runs from 0 to 1 in every sub-pixel, so the expected rate
            # per unit u of the window is mu.
            subsample = stc.EventSample._from_arrays(
                ra=ra[mask], dec=dec[mask], n_total=n_total, t0=t0, tf=tf, rng=rng,
                spatial_type="data", expected_n=mu,
                window=grid[i], exposure_model=exposure_model,
                exposure=u_events[mask], expected_exposure_rate=mu,
            )

            lambda_array[i] = stc.lambda_estimator(sample=subsample)

            # Clipped at 1, as in run_blind_search_mock.py.
            pvalues_lambda[i] = min(
                1.0,
                stc.lambda_marginal_sf(lambda_array[i], expected_n_array[i]),
            )
            pvalues_poisson[i] = stc.poisson_mid_p_value(
                n_sample_array[i], expected_n_array[i]
            )

        except ValueError:
            lambda_array[i] = np.nan
            n_failed += 1
            logger.exception(
                "Window %d (pixel %d, centre=[%.2f, %.2f], n_sample=%d) failed; "
                "recorded as NaN.",
                i, ipix, grid.centres[i, 0], grid.centres[i, 1],
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
    sigma_poisson = np.full(n_windows, np.nan)
    sigma_lambda[tested] = stc.pvalue_to_sigma(pvalues_lambda[tested])
    sigma_poisson[tested] = stc.pvalue_to_sigma(pvalues_poisson[tested])

    # Windows above the threshold, most significant first, for the metadata.
    above_threshold = {}
    for name, sigma in (("lambda", sigma_lambda), ("poisson", sigma_poisson)):
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
            "n_poisson": int(np.count_nonzero(sigma_poisson > threshold)),
            "n_expected": n_tested * float(stc.sigma_to_pvalue(threshold)),
        }

    # ------------------------------------------------------------------
    # Save outputs and metadata
    # ------------------------------------------------------------------
    np.savez_compressed(
        outdir / "results.npz",
        pixels=grid.pixels,               # (W,)   HEALPix RING index (nside_window) of each window
        centres=grid.centres,             # (W, 2) window centres [RA, Dec] (deg)
        inside_fov=inside_fov,            # (W,)   False for windows on the FoV edge (skipped)
        expected_n=expected_n_array,      # (W,)   sum of the expected counts of the sub-pixels (NaN if not tested)
        lambda_array=lambda_array,        # (W,)   Lambda statistic per window
        n_sample_array=n_sample_array,    # (W,)   event count per window
        pvalues_lambda=pvalues_lambda,    # (W,)   marginal Lambda p-value
        pvalues_poisson=pvalues_poisson,  # (W,)   Poisson counting mid-p value
        sigma_lambda=sigma_lambda,        # (W,)   one-sided significance, Lambda
        sigma_poisson=sigma_poisson,      # (W,)   one-sided significance, Poisson
    )

    elapsed = time.time() - start_time

    write_metadata(
        outdir=outdir,
        metadata={
            "script": Path(__file__).name,
            "run_code": "blind_search_mock_nside8",
            "dataset": dataset,
            "input_path": str(input_path),
            "active_area_path": str(active_area_path),
            "runtime_seconds": elapsed,
            "n_total": n_total,
            "nside": nside_window,
            "nside_exposure": nside_exposure,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Blind search on one mock sample over the nside=8 HEALPix pixels, "
            "with the exposure computed in their nside=32 sub-pixels."
        )
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset_50",
        help="Mock sample to analyse, the file name without .dat.",
    )
    parser.add_argument(
        "--submission-id",
        type=str,
        default=None,
        help=(
            "Identifier shared by all jobs of one HTCondor batch. When given, "
            "the run goes to blind_search_mock_nside8/<submission_id>/<dataset>/."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(dataset=args.dataset, submission_id=args.submission_id)
