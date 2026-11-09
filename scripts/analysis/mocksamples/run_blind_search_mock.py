"""
Blind all-sky search on a mock data sample over the HEALPix pixels, with
the windows, the exposure model and the suppressed Lambda of the H0 Monte
Carlo (``h0_mc_distribution/h0_pipeline.py``), so that every pixel can be
compared with its H0 distribution.

The pipeline is:

    1. Read the sample (``inputs/mocksamples/dataset_stcwg_vert/``) and build
       the exposure model, both with ``h0_pipeline.load_sample_and_model``:
       the active area A(t) of the SD-1500 array, minute by minute, and
       ``t0`` and ``tf`` the first and last events.
    2. Windows: the pixels at ``h0_pipeline.NSIDE`` lying entirely inside
       the field of view (``h0_pipeline.fov_pixels``). Every event is
       assigned to its pixel.
    3. For every pixel, the expected count ``mu = N E(pixel) / E_sky``
       and, with at least 2 events, the normalised cumulative exposure of
       its events at the pixel centre (``ExposureModel.norm_cumul_exposure``)
       and the suppressed Lambda (``stc.lambda_estimator_suppressed_cdf``)
       with the rate ``mu / eps(tf)``.

No p-values are computed here: they come from the H0 distributions of
Lambda and their tail fits (``h0_mc_distribution/mc_pvalues_blind_search.py``).

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
import sys
import time
from datetime import datetime
from pathlib import Path

import healpy as hp
import numpy as np

from tqdm import tqdm

import spacetimecorr as stc
from spacetimecorr.io import setup_logger, make_run_dir, write_metadata

# The windows, the model and the constants are those of the H0 Monte Carlo.
sys.path.insert(0, str(Path(__file__).resolve().parent / "h0_mc_distribution"))
from h0_pipeline import (  # noqa: E402
    ACTIVE_AREA_PATH, ALTITUDE_PA, LATITUDE_PA, LONGITUDE_PA, NSIDE, PROJECT_ROOT,
    THETA_MAX_DEG, fov_pixels, load_sample_and_model,
)


def main(dataset: str, submission_id: str | None = None) -> None:
    """Run the blind search on one mock sample, e.g. ``dataset="dataset_0"``.

    With ``submission_id`` (an HTCondor batch), the run is written to
    ``output/analysis/mocksamples/blind_search_mock/<submission_id>/<dataset>/``;
    without it, to
    ``output/analysis/mocksamples/blind_search_mock/<dataset>_<timestamp>/``.
    """

    start_time = time.time()

    input_path = (
        PROJECT_ROOT / "inputs" / "mocksamples" / "dataset_stcwg_vert" / f"{dataset}.dat"
    )

    # ------------------------------------------------------------------
    # Output directory and logger
    # ------------------------------------------------------------------
    if submission_id is None:
        outdir, _ = make_run_dir(
            base_dir=PROJECT_ROOT / "output" / "analysis" / "mocksamples",
            run_code="blind_search_mock",
            seed=0,
            submission_id=f"{dataset}_{datetime.now():%Y%m%d_%H%M%S}",
        )
    else:
        outdir, _ = make_run_dir(
            base_dir=PROJECT_ROOT / "output" / "analysis" / "mocksamples" / "blind_search_mock",
            run_code=submission_id,
            seed=0,
            submission_id=dataset,
        )
    logger = setup_logger(log_path=outdir / "run.log", name="blind_search_mock")

    logger.info("Starting blind search on mock sample %s", dataset)
    logger.info("Input: %s", input_path)
    logger.info("Output directory: %s", outdir)

    # ------------------------------------------------------------------
    # 1) Read the sample and build the exposure model
    # ------------------------------------------------------------------
    times, ra, dec, observatory, exposure_model = load_sample_and_model(dataset)
    n_total = ra.size
    t0, tf = exposure_model.t0, exposure_model.tf
    T_obs_days = float((tf - t0).to_value("day"))

    logger.info(
        "Read %d events | t0 = %s | tf = %s | T_obs = %.2f d",
        n_total, t0.isot, tf.isot, T_obs_days,
    )

    # ------------------------------------------------------------------
    # 2) Windows: the pixels inside the FoV, and the events of each
    # ------------------------------------------------------------------
    pixels, rings = fov_pixels()
    windows = [stc.HealpixWindow(nside=NSIDE, ipix=int(p)) for p in pixels]
    centres = np.array([w.centre for w in windows])
    n_windows = pixels.size

    # Column of every pixel of the sphere in the arrays below, -1 for the
    # pixels not in the field of view. The events are sorted by column, so
    # that the events of every pixel are one slice of `order`.
    column = np.full(hp.nside2npix(NSIDE), -1)
    column[pixels] = np.arange(n_windows)
    event_col = column[hp.ang2pix(NSIDE, ra, dec, lonlat=True)]
    kept = np.flatnonzero(event_col >= 0)
    order = kept[np.argsort(event_col[kept], kind="stable")]
    col_edges = np.searchsorted(event_col[order], np.arange(n_windows + 1))

    logger.info(
        "%d pixels at nside=%d inside the FoV (%.3f deg^2 each), with %d of the %d events.",
        n_windows, NSIDE, hp.nside2pixarea(NSIDE, degrees=True), kept.size, n_total,
    )

    # ------------------------------------------------------------------
    # Storage (one entry per pixel, aligned with `pixels`)
    # ------------------------------------------------------------------
    n_sample_array = np.diff(col_edges)
    expected_n = np.full(n_windows, np.nan)
    eps_max = np.full(n_windows, np.nan)
    lambda_array = np.full(n_windows, np.nan)

    n_failed = 0    # pixels with >= 2 events that still raised

    # ------------------------------------------------------------------
    # 3) Scan: exposure from the arrival times, then Lambda
    # ------------------------------------------------------------------
    # disable=None turns the bar off when not on a terminal (HTCondor jobs).
    for j in tqdm(range(n_windows), desc="Pixels scanned", disable=None):
        events = order[col_edges[j]:col_edges[j + 1]]
        centre = windows[j].centre

        # The exposure of the events comes first: its pass over the A(t)
        # steps also stores the exposure over the whole run in the model's
        # cache, which mu and eps(tf) then read with no second pass (one
        # pass for a pixel with fewer than 2 events).
        if events.size >= 2:
            exposure = exposure_model.norm_cumul_exposure(times[events], centre)
        expected_n[j] = windows[j].expected_n_in_window(n_total, exposure_model)
        eps_max[j] = exposure_model.max_norm_cumul_exposure(centre)

        # Lambda needs at least 2 events.
        if events.size < 2:
            continue
        try:
            lambda_array[j] = stc.lambda_estimator_suppressed_cdf(
                exposure, expected_n[j] / eps_max[j], expected_n[j],
            )
        except ValueError:
            n_failed += 1
            logger.exception(
                "Pixel %d (centre=[%.2f, %.2f], n_sample=%d) failed; recorded as NaN.",
                pixels[j], centre[0], centre[1], events.size,
            )

    n_dropped = int(np.count_nonzero(n_sample_array < 2))
    n_tested = int(np.count_nonzero(np.isfinite(lambda_array)))
    logger.info(
        "Scan finished: %d pixels | %d tested | %d dropped (n < 2) | %d failed.",
        n_windows, n_tested, n_dropped, n_failed,
    )

    # ------------------------------------------------------------------
    # Save outputs and metadata
    # ------------------------------------------------------------------
    np.savez_compressed(
        outdir / "results.npz",
        pixels=pixels,                    # (W,)   HEALPix RING index of every pixel
        rings=rings,                      # (W,)   ring of every pixel, 1 at the north pole
        centres=centres,                  # (W, 2) pixel centres [RA, Dec] (deg)
        n_sample_array=n_sample_array,    # (W,)   events per pixel
        expected_n=expected_n,            # (W,)   mu = N E(pixel) / E_sky
        eps_max=eps_max,                  # (W,)   eps(tf) at the pixel centre
        lambda_array=lambda_array,        # (W,)   suppressed Lambda; NaN with n < 2 or failed
    )

    elapsed = time.time() - start_time

    write_metadata(
        outdir=outdir,
        metadata={
            "script": Path(__file__).name,
            "run_code": "blind_search_mock",
            "dataset": dataset,
            "input_path": str(input_path),
            "active_area_path": str(ACTIVE_AREA_PATH),
            "runtime_seconds": elapsed,
            "n_total": n_total,
            "n_events_in_windows": int(kept.size),
            "nside": NSIDE,
            "window_solid_angle_sr": float(hp.nside2pixarea(NSIDE)),
            "n_windows": n_windows,
            "n_windows_tested": n_tested,
            "n_windows_dropped": n_dropped,
            "n_windows_failed": n_failed,
            "t0": t0.isot,
            "tf": tf.isot,
            "T_obs_days": T_obs_days,
            "theta_max_deg": THETA_MAX_DEG,
            "latitude_pa_deg": LATITUDE_PA,
            "longitude_pa_deg": LONGITUDE_PA,
            "altitude_pa_m": ALTITUDE_PA,
            "max_area_km2": observatory.max_area,
        },
    )

    logger.info("Saved results to %s", outdir / "results.npz")
    logger.info("Run finished in %.2f seconds", elapsed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Blind search on one mock sample over the HEALPix pixels of the "
            "H0 Monte Carlo, with the suppressed Lambda."
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
