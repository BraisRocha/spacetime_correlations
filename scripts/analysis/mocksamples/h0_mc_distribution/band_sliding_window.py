"""
Monte Carlo distribution of Lambda under isotropy in one declination band,
with the observation window of every realisation drawn from the sample.

In ``h0_pipeline.py`` every sky reuses all the arrival times of the sample,
so ``t0``, ``tf`` and ``T_obs`` are the same in every sky. Here every
realisation ``k`` of the band, from the generator of
``SeedSequence(seed, spawn_key=(k,))``, is built as follows:

1. A sky is scrambled from the sample (``isotropic_scrambling.scramble``)
   and its ``N_b`` events in the strip ``|dec - dec_ring| <= R + margin``
   are kept.
2. A start ``j`` is drawn uniform in ``[0, N_total - N_b]``, and the strip
   events take the ``N_b`` consecutive arrival times ``t_j ... t_{j+N_b-1}``
   of the sorted sample: every event keeps its local direction and is
   converted back to ICRS at its new time::

       v_new = R(t_new) R(t_old)^T v_old

3. The exposure model runs over ``[t_j, t_{j+N_b-1}]``, and Lambda is
   computed in every cap of the band as in ``h0_pipeline.py band``.

The declination of an event depends only on its zenith angle and azimuth,
not on its arrival time (up to precession, which the margin of the strip
covers). The rate of the strip is then that of the whole sky times its
share ``p`` of the exposure, the same in every window::

    p = int_strip omega(dec) cos(dec) d(dec) / int_sky omega(dec) cos(dec) d(dec)

with ``omega = ExposureModel.spatial_exposure``. The window of ``N_b``
strip events is that of ``N_b / p`` events over the whole sky, so the
expected count of a cap is ``expected_n_in_window(N_b / p, model)``.

Output, in ``<output_dir>/bands/band_ringXXX.npz``, with the keys of
``h0_pipeline.py merge``, so that ``fit_h0_bands.py`` and
``fit_h0_tails.py`` run on ``<output_dir>`` as on a pipeline run. ``mu``
and ``eps_max`` are per (realisation, cap), and ``t0`` and ``tf`` (Unix s)
per realisation.
"""

import argparse
from pathlib import Path

import healpy as hp
import numpy as np

from tqdm import tqdm

import spacetimecorr as stc

from h0_pipeline import (
    NSIDE, WINDOW_RADIUS_DEG, THETA_MAX_DEG, fov_rings, load_sample_and_model,
)
from isotropic_scrambling import zenith_angles, local_to_icrs_matrices, scramble

STRIP_MARGIN_DEG = 0.2      # widening of the strip, for the shift of Dec by precession


def strip_share(exposure_model: stc.ExposureModel, dec_ring: float, half_width: float) -> float:
    """Share ``p`` of the exposure of the whole sky in the strip
    ``|dec - dec_ring| <= half_width`` (deg)."""
    def weighted_integral(low, high):
        dec = np.linspace(low, high, 20001)
        return np.trapezoid(
            exposure_model.spatial_exposure(dec) * np.cos(np.radians(dec)), np.radians(dec),
        )
    return float(
        weighted_integral(dec_ring - half_width, dec_ring + half_width)
        / weighted_integral(-90.0, 90.0)
    )


def unit_vectors(ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    """ICRS unit vectors of ``(ra, dec)`` (deg), shape ``(N, 3)``."""
    ra, dec = np.radians(ra), np.radians(dec)
    return np.column_stack((np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)))


def main(dataset: str, dec_target: float, n_skies: int, seed: int, output_dir: Path) -> None:
    times, ra, dec, observatory, exposure_model = load_sample_and_model(dataset)
    # Sorted, so that consecutive indices are consecutive arrival times.
    by_time = times.argsort()
    times, ra, dec = times[by_time], ra[by_time], dec[by_time]
    n_total = times.size

    print("Computing the zenith angles and the local-to-ICRS matrices")
    theta = zenith_angles(times, ra, dec, observatory.location)
    matrices = local_to_icrs_matrices(times, observatory.location)

    # The ring of the FoV closest to dec_target.
    rings = fov_rings()
    dec_of_ring = {r: hp.pix2ang(NSIDE, int(px[0]), lonlat=True)[1] for r, px in rings.items()}
    ring = min(dec_of_ring, key=lambda r: abs(dec_of_ring[r] - dec_target))
    pixels = rings[ring]
    centres = np.column_stack(hp.pix2ang(NSIDE, pixels, lonlat=True))
    windows = [stc.SkyWindow(c, WINDOW_RADIUS_DEG) for c in centres]
    n_caps = pixels.size
    dec_ring = centres[0, 1]

    centre_vecs = unit_vectors(*centres.T)
    cos_radius = np.cos(np.radians(WINDOW_RADIUS_DEG))
    strip_half_width = WINDOW_RADIUS_DEG + STRIP_MARGIN_DEG
    p = strip_share(exposure_model, dec_ring, strip_half_width)
    print(
        f"Ring {ring} (Dec {dec_ring:.2f}): {n_caps} caps, strip share p = {p:.5f}, "
        f"about {p * n_total:.0f} events per realisation"
    )

    # The EventSample constructor requires a generator but draws nothing
    # from it here; the realisations draw from their own streams.
    rng = np.random.default_rng(0)

    n_band = np.zeros(n_skies, dtype=int)
    window_start = np.zeros(n_skies, dtype=int)
    t0_unix = np.zeros(n_skies)
    tf_unix = np.zeros(n_skies)
    mu = np.full((n_skies, n_caps), np.nan)
    eps_max = np.full((n_skies, n_caps), np.nan)
    n_sample = np.zeros((n_skies, n_caps), dtype=int)
    lambdas = np.full((n_skies, n_caps), np.nan)
    lambdas_exp = np.full((n_skies, n_caps), np.nan)
    n_failed = 0      # (realisation, cap) with >= 2 events that still raised

    for k in tqdm(range(n_skies), desc="Realisations", disable=None):
        sky_rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(k,)))
        order, sky_ra, sky_dec = scramble(theta, matrices, sky_rng)

        # --------------------------------------------------------------
        # The strip events, moved to N_b consecutive arrival times from a
        # random start
        # --------------------------------------------------------------
        strip = np.flatnonzero(np.abs(sky_dec - dec_ring) <= strip_half_width)
        n_b = strip.size
        j = int(sky_rng.integers(0, n_total - n_b + 1))
        new_index = np.arange(j, j + n_b)

        v_old = unit_vectors(sky_ra[strip], sky_dec[strip])
        v_local = np.einsum("nji,nj->ni", matrices[order[strip]], v_old)
        x, y, z = np.einsum("nij,nj->in", matrices[new_index], v_local)
        # As in scramble: R(t) is unit only to ~1e-4, so the norm is divided out.
        new_ra = np.degrees(np.arctan2(y, x)) % 360.0
        new_dec = np.degrees(np.arcsin(z / np.sqrt(x**2 + y**2 + z**2)))

        cap_of_pair, event_of_pair = np.nonzero(
            centre_vecs @ unit_vectors(new_ra, new_dec).T >= cos_radius
        )

        # --------------------------------------------------------------
        # Exposure over the window, then both Lambda in every cap
        # --------------------------------------------------------------
        window_times = times[new_index]
        model = stc.ExposureModel(
            observatory=observatory, t0=window_times[0], tf=window_times[-1],
            rng=rng, theta_max_deg=THETA_MAX_DEG,
        )
        n_equivalent = n_b / p
        n_band[k], window_start[k] = n_b, j
        t0_unix[k], tf_unix[k] = model.t0.unix, model.tf.unix

        for c in range(n_caps):
            cap_times = window_times[event_of_pair[cap_of_pair == c]]
            n = cap_times.size
            n_sample[k, c] = n
            # The exposure pass stores eps(tf) in the model's cache, which
            # the expected count and eps_max then read with no second pass.
            if n > 0:
                eps = model.norm_cumul_exposure(cap_times, centres[c])
            mu[k, c] = windows[c].expected_n_in_window(n_equivalent, model)
            eps_max[k, c] = model.max_norm_cumul_exposure(centres[c])
            if n < 2:
                continue
            rate = mu[k, c] / eps_max[k, c]

            # EventSample stores the coordinates but Lambda does not use
            # them; the cap centre stands in for them.
            subsample = stc.EventSample._from_arrays(
                ra=np.full(n, centres[c, 0]), dec=np.full(n, centres[c, 1]),
                n_total=n_equivalent, t0=model.t0, tf=model.tf, rng=rng,
                spatial_type="data", expected_n=mu[k, c],
                window=windows[c], exposure_model=model,
                exposure=eps, expected_exposure_rate=rate,
            )
            # Both raise on the same tied exposures, so a (realisation, cap)
            # gets either both values or neither.
            try:
                value = stc.lambda_estimator_suppressed_cdf(eps, rate, mu[k, c])
                value_exp = stc.lambda_estimator(sample=subsample)
            except ValueError:
                n_failed += 1
                continue
            lambdas[k, c], lambdas_exp[k, c] = value, value_exp

    path = output_dir / "bands" / f"band_ring{ring:03d}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        ring=ring, dec=dec_ring, radius_deg=WINDOW_RADIUS_DEG,
        skies=np.arange(n_skies),           # (S,)    realisation index k of every row
        pixels=pixels,                      # (C,)    HEALPix RING index of the pixel at the centre of every cap
        centres=centres,                    # (C, 2)  cap centres [RA, Dec] (deg)
        p_strip=p,                          #         share of the exposure of the sky in the strip
        n_band=n_band,                      # (S,)    events in the strip
        window_start=window_start,          # (S,)    index j of the first arrival time of the window
        t0=t0_unix, tf=tf_unix,             # (S,)    start and end of the window (Unix s)
        mu=mu,                              # (S, C)  expected count
        eps_max=eps_max,                    # (S, C)  eps(tf)
        n_sample=n_sample,                  # (S, C)  events per (realisation, cap)
        lambdas=lambdas,                    # (S, C)  Lambda with the suppressed CDF; NaN with n < 2 or failed
        lambdas_exp=lambdas_exp,            # (S, C)  Lambda with the exponential CDF; NaN likewise
        n_failed=n_failed,
    )
    t_obs_days = (tf_unix - t0_unix) / 86400.0
    print(
        f"{np.count_nonzero(np.isfinite(lambdas))} Lambda values, {n_failed} failed; "
        f"T_obs from {t_obs_days.min():.0f} to {t_obs_days.max():.0f} days. Saved {path}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lambda under isotropy in one declination band, with the window "
                    "of every realisation drawn from the sample."
    )
    parser.add_argument("--dataset", default="dataset_0")
    parser.add_argument("--dec", type=float, default=-30.0,
                        help="The band is the ring of the FoV closest to this Dec (deg).")
    parser.add_argument("--n-skies", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.dataset, args.dec, args.n_skies, args.seed, args.output_dir)
