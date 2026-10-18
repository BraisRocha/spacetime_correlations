"""
Blind all-sky search on a mock data sample, over circular windows centred
on a HEALPix grid.

The sample is read from ``inputs/mocksamples/dataset_stcwg_vert/``. The
pipeline is:

    1. Read the sample: arrival time (Unix UTC seconds plus nanoseconds),
       RA and Dec. The last column (angular resolution) is not used. The
       events are sorted by Dec, so that each window finds its events by
       binary search (``SkyWindow.contains`` with ``dec_sorted=True``).
    2. Build the grid: a cap of radius ``window_radius_deg`` around the
       centre of every HEALPix pixel at resolution ``nside`` whose centre
       lies inside the field of view. The pixel spacing is much smaller
       than the cap radius, so neighbouring windows overlap heavily and
       any flare lies close to the centre of some window. Windows that
       straddle the FoV edge are skipped.
    3. For every window, select its events, convert their arrival times
       into normalised cumulative exposure at the window centre
       (``ExposureModel.norm_cumul_exposure``), and compute the Lambda and
       Poisson statistics, their p-values and their significances. The
       exposure, and with it the expected count of each window, includes
       the active area of the array A(t), minute by minute, read from the
       file in ``inputs/active_area/`` built by
       ``scripts/preprocessing/process_active_area.py``.

Windows above ``sigma_threshold`` in either test are saved, with their
direction, in ``metadata.json``.

Time precision
--------------
astropy's ``Time`` stores every time as two float64 numbers, which keeps
it to well below a nanosecond: the nanoseconds of the input are not lost
on reading (a single float64 of the Unix time would round them to
~0.2 us).

The exposure of an event is computed from the time elapsed since ``t0``
as one float64 in seconds, and is itself one float64. Over the ~19-year
run of ``dataset_stcwg_vert`` both resolve ~120 ns, so the exposure values
carry the arrival times to ~120 ns, not 1 ns. Lambda only uses the
differences between the exposures of events in the same window; in
``dataset_0`` the closest two events anywhere on the sky are ~69 us apart,
about 580 times more than that.
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from astropy.time import Time
from tqdm import tqdm

import spacetimecorr as stc
from spacetimecorr.io import (
    setup_logger, make_run_dir, write_metadata, load_active_area,
)


def load_mock_sample(path: Path) -> tuple[Time, np.ndarray, np.ndarray]:
    """Read a mock sample: arrival times, RA and Dec (deg).

    Columns: Unix UTC seconds, nanoseconds, RA, Dec, angular resolution
    (unused). Both time columns are integers below 2**53, so float64 holds
    them exactly; they are then passed to ``Time`` as two numbers.
    Nanosecond values outside ``[0, 1e9)`` are taken literally, as an offset
    from the second.
    """
    seconds, nanoseconds, ra, dec = np.loadtxt(path, usecols=(0, 1, 2, 3), unpack=True)
    times = Time(seconds, nanoseconds * 1e-9, format="unix", scale="utc")
    return times, ra, dec


def main(dataset: str, submission_id: str | None = None) -> None:
    """Run the blind search on one mock sample, e.g. ``dataset="dataset_0"``.

    With ``submission_id`` (an HTCondor batch), the run is written to
    ``output/analysis/blind_search_mock/<submission_id>/<dataset>/``;
    without it, to ``output/analysis/blind_search_mock/<dataset>_<timestamp>/``.
    """

    start_time = time.time()

    # ------------------------------------------------------------------
    # Analysis parameters
    # ------------------------------------------------------------------
    nside = 128                    # HEALPix resolution of the window centres
    window_radius_deg = 1.05       # angular radius of every window
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
            base_dir=project_root / "output" / "analysis",
            run_code="blind_search_mock",
            seed=0,
            submission_id=f"{dataset}_{datetime.now():%Y%m%d_%H%M%S}",
        )
    else:
        outdir, _ = make_run_dir(
            base_dir=project_root / "output" / "analysis" / "blind_search_mock",
            run_code=submission_id,
            seed=0,
            submission_id=dataset,
        )
    logger = setup_logger(log_path=outdir / "run.log", name="blind_search_mock")

    logger.info("Starting blind search on mock sample %s", dataset)
    logger.info("Input: %s", input_path)
    logger.info("Output directory: %s", outdir)

    # ------------------------------------------------------------------
    # 1) Read the sample
    # ------------------------------------------------------------------
    times, ra, dec = load_mock_sample(input_path)
    n_total = ra.size

    # Sorted by Dec, all three arrays with the same permutation, so that
    # SkyWindow.contains can find each window's Dec band by binary search.
    order = np.argsort(dec)
    times, ra, dec = times[order], ra[order], dec[order]

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
    # The area of the observatory is the active area A(t) of the SD-1500
    # array (6T5 cells), as steps, zero during the bad periods.
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
    # 2) Windows centred on the HEALPix pixels inside the FoV
    # ------------------------------------------------------------------
    healpix = stc.HealpixGrid(exposure_model, nside=nside)
    grid = stc.SkyGrid(healpix.centres, window_radius_deg)
    n_windows = len(grid)
    window_solid_angle = float(grid[0].solid_angle)
    logger.info(
        "Built grid: %d windows of radius %.2f deg (%.3f deg^2 each), "
        "centred on the nside=%d HEALPix pixels inside the FoV.",
        n_windows, window_radius_deg,
        window_solid_angle * np.degrees(1.0) ** 2, nside,
    )

    # Windows lying entirely inside the FoV, the declination band
    # |dec - latitude| < theta_max. The others straddle the FoV edge and are
    # skipped in the scan.
    inside_fov = (
        np.abs(grid.centres[:, 1] - latitude_pa) + window_radius_deg < theta_max_deg
    )
    n_edge = int(np.count_nonzero(~inside_fov))
    logger.info("%d windows straddle the FoV edge and are skipped.", n_edge)

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
    # 3) Scan: exposure from the arrival times, then the Lambda method
    # ------------------------------------------------------------------
    # disable=None turns the bar off when not on a terminal (HTCondor jobs).
    for i, window in enumerate(
        tqdm(grid, total=n_windows, desc="Windows scanned", disable=None)
    ):
        mask = window.contains(ra, dec, dec_sorted=True)
        n_sample_array[i] = np.count_nonzero(mask)

        if not inside_fov[i]:
            continue

        # Lambda needs at least 2 events. Skipped before computing anything:
        # every exposure computation costs a full pass over the A(t) steps.
        if n_sample_array[i] < 2:
            n_dropped += 1
            continue

        try:
            # The order of these steps matters for the run time. The exposure
            # of the events comes first: its pass over the A(t) steps also
            # stores the exposure over the whole run in the model's cache.
            # The expected count (set by select_subsample) and the expected
            # rate below then read it from there, with no second pass.
            #
            # Exposure of every event from its arrival time, evaluated at the
            # window centre.
            exposure = exposure_model.norm_cumul_exposure(times[mask], window.centre)

            subsample = sample.select_subsample(window, mask=mask)
            subsample.exposure = exposure
            # Expected rate per unit exposure, as in
            # EventSample.generate_cumul_exposure.
            subsample.expected_exposure_rate = (
                subsample.expected_n
                / exposure_model.max_norm_cumul_exposure(window.centre)
            )
            expected_n_array[i] = subsample.expected_n

            lambda_array[i] = stc.lambda_estimator(sample=subsample)

            # Clipped at 1: for windows with far fewer events than expected
            # (the exposure model does not yet match the mock), the true
            # p-value is within ~1e-14 of 1 and rounding can push it above.
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
                "Window %d (centre=[%.2f, %.2f], n_sample=%d) failed; "
                "recorded as NaN.",
                i, window.centre[0], window.centre[1],
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
                "pixel": int(healpix.pixels[j]),
                "ra_deg": float(grid.centres[j, 0]),
                "dec_deg": float(grid.centres[j, 1]),
                "n_sample": int(n_sample_array[j]),
                "expected_n": float(expected_n_array[j]),
                "sigma": float(sigma[j]),
            }
            for j in above
        ]

    # Probability of at least one window above the threshold by chance, for
    # n_tested windows with uniform p-values. Windows are treated as
    # independent, which they are not: neighbouring windows overlap heavily
    # and share most of their events, and the Poisson mid-p value is discrete.
    p_threshold = float(stc.sigma_to_pvalue(sigma_threshold))
    prob_any_above = 1.0 - (1.0 - p_threshold) ** n_tested

    # Number of windows above 3, 4 and 5 sigma in each test, and the number
    # expected by chance, n_tested * p(sigma), under the same assumptions.
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
        pixels=healpix.pixels,            # (W,)   HEALPix RING index of each window centre
        centres=grid.centres,             # (W, 2) window centres [RA, Dec] (deg)
        inside_fov=inside_fov,            # (W,)   False for windows on the FoV edge (skipped)
        expected_n=expected_n_array,      # (W,)   exposure-weighted expected count (NaN if not tested)
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
            "run_code": "blind_search_mock",
            "dataset": dataset,
            "input_path": str(input_path),
            "active_area_path": str(active_area_path),
            "runtime_seconds": elapsed,
            "n_total": n_total,
            "nside": nside,
            "window_radius_deg": window_radius_deg,
            "window_solid_angle_sr": window_solid_angle,
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
            "Blind search on one mock sample over circular windows centred "
            "on a HEALPix grid."
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
            "the run goes to blind_search_mock/<submission_id>/<dataset>/."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(dataset=args.dataset, submission_id=args.submission_id)
