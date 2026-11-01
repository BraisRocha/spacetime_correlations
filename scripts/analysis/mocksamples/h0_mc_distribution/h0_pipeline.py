"""
Monte Carlo distribution of Lambda under isotropy, per declination band.

The windows are circular caps of radius ``WINDOW_RADIUS_DEG`` centred on
the HEALPix pixels (RING ordering, ``NSIDE``), as in
``run_blind_search_mock.py``. A band is the caps centred on one ring of
pixels, numbered as in healpy from 1 at the north pole, for the rings whose
caps lie entirely inside the field of view. Every band is filled with the
Lambda values of its caps in isotropic skies scrambled from a mock sample
(``isotropic_scrambling.scramble``), sky ``k`` from the generator of
``SeedSequence(seed, spawn_key=(k,))``. Sky ``k`` is the same in every job,
so the bands are those of one set of full skies.

Two Lambda are computed from the same gaps: with the suppressed CDF of the
gaps (``stc.lambda_estimator_suppressed_cdf``), the one histogrammed, and
with the exponential CDF (``stc.lambda_estimator``).

One subcommand per node of the DAG in ``condor/h0_mc_distribution/``:

    plan     Skies per band for at least ``n_min`` Lambda values::

                 S_b = ceil(n_min / (n_caps_b * P(n >= 2 | mu_b)))

             split into jobs of at most ``max_skies_per_job`` skies, one
             line per job: ``dataset ring k_start k_stop seed``.
    prepare  Zenith angle of every event of the sample and local-to-ICRS
             matrix of every arrival time, computed once for all the jobs,
             so that every job builds the same skies.
    band     Skies ``k_start ... k_stop - 1`` of one band, in blocks. For
             every sky of a block, keep every (event, cap) pair of the band
             within the radius; for every cap, compute the normalised
             cumulative exposure of its events in all the skies of the block
             in one pass at the cap centre (``ExposureModel.norm_cumul_exposure``),
             then both Lambda in every (sky, cap) with at least 2 events,
             with the rate ``mu / eps(tf)`` of the cap.
    merge    Per band, the jobs in sky order, histograms of both Lambda
             against ``lambda_marginal_pdf(x, mu)`` for ``n >= 2`` with KS
             tests, and a summary. The scratch directory is removed once every
             job is in.

The model, ``t0`` and ``tf`` are built as in ``run_blind_search_mock.py``.
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import healpy as hp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as scp

from astropy.utils import iers
from tqdm import tqdm

import spacetimecorr as stc
from spacetimecorr.io import load_active_area, load_mock_sample

from isotropic_scrambling import zenith_angles, local_to_icrs_matrices, scramble

# The worker nodes have no network: astropy uses the IERS tables it ships
# with instead of trying to download newer ones.
iers.conf.auto_download = False

# ----------------------------------------------------------------------
# Global style: load the matplotlibrc of the plots/ folder
# ----------------------------------------------------------------------
RC_FILE = Path(__file__).resolve().parents[3] / "plots" / "matplotlibrc_test"
if RC_FILE.exists():
    mpl.rc_file(RC_FILE, use_default_template=False)

# ----------------------------------------------------------------------
# Analysis parameters
# ----------------------------------------------------------------------
NSIDE = 32                     # HEALPix resolution of the cap centres
WINDOW_RADIUS_DEG = 1.05       # angular radius of every cap

# Pierre Auger Observatory
LATITUDE_PA = -35.15
LONGITUDE_PA = -69.15
ALTITUDE_PA = 1425
THETA_MAX_DEG = 60.0           # zenith-angle cut defining the field of view

PROJECT_ROOT = Path(__file__).resolve().parents[4]
ACTIVE_AREA_PATH = PROJECT_ROOT / "inputs" / "active_area" / "ActiveArea_SD1500_UB_6T5.npz"


def load_sample_and_model(dataset: str):
    """Arrival times, RA, Dec, observatory and exposure model of ``dataset``."""
    times, ra, dec = load_mock_sample(
        PROJECT_ROOT / "inputs" / "mocksamples" / "dataset_stcwg_vert" / f"{dataset}.dat"
    )
    observatory = stc.Observatory(
        latitude=LATITUDE_PA, longitude=LONGITUDE_PA, altitude=ALTITUDE_PA,
        area=load_active_area(ACTIVE_AREA_PATH),
    )
    # The ExposureModel constructor requires a generator but draws nothing
    # from it here.
    exposure_model = stc.ExposureModel(
        observatory=observatory, t0=times.min(), tf=times.max(),
        rng=np.random.default_rng(0), theta_max_deg=THETA_MAX_DEG,
    )
    return times, ra, dec, observatory, exposure_model


def fov_rings() -> dict[int, np.ndarray]:
    """Pixels of every ring whose caps lie entirely inside the FoV, by ring
    number (1 at the north pole). As in ``run_blind_search_mock.py``, a cap
    is inside when ``|dec - latitude| + radius < theta_max``.
    """
    _, dec_centres = hp.pix2ang(NSIDE, np.arange(hp.nside2npix(NSIDE)), lonlat=True)
    dec_rings = np.unique(dec_centres)[::-1]      # north to south, as the ring numbers
    inside_fov = np.abs(dec_rings - LATITUDE_PA) + WINDOW_RADIUS_DEG < THETA_MAX_DEG
    return {
        i + 1: np.flatnonzero(dec_centres == d)
        for i, d in enumerate(dec_rings) if inside_fov[i]
    }


def band_file(scratch_dir: Path, ring: int, k_start: int) -> Path:
    """Results of the job of ``ring`` starting at sky ``k_start``."""
    return scratch_dir / f"band_ring{ring:03d}_k{k_start:07d}.npz"


# ----------------------------------------------------------------------
# plan
# ----------------------------------------------------------------------

def plan(dataset: str, n_min: int, max_skies_per_job: int, seed: int, output: Path) -> None:
    """Write the job list, one line ``dataset ring k_start k_stop seed``
    per job, to ``output``."""
    times, _, _, _, exposure_model = load_sample_and_model(dataset)
    n_total = times.size

    lines = []
    for ring, pixels in fov_rings().items():
        # All the caps of a ring share their declination; the first one
        # gives the expected count of the ring.
        centre = np.array(hp.pix2ang(NSIDE, int(pixels[0]), lonlat=True))
        mu = stc.SkyWindow(centre, WINDOW_RADIUS_DEG).expected_n_in_window(
            n_total, exposure_model,
        )
        p_two = 1.0 - np.exp(-mu) * (1.0 + mu)
        n_skies = int(np.ceil(n_min / (pixels.size * p_two)))
        for k_start in range(0, n_skies, max_skies_per_job):
            k_stop = min(k_start + max_skies_per_job, n_skies)
            lines.append(f"{dataset} {ring} {k_start} {k_stop} {seed}")

    output.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(lines)} jobs for {len(fov_rings())} bands to {output}")


# ----------------------------------------------------------------------
# prepare
# ----------------------------------------------------------------------

def prepare(dataset: str, scratch_dir: Path) -> None:
    """Save the zenith angles and the local-to-ICRS matrices of ``dataset``
    to ``scratch_dir/prepare.npz``."""
    times, ra, dec, observatory, _ = load_sample_and_model(dataset)
    theta = zenith_angles(times, ra, dec, observatory.location)
    matrices = local_to_icrs_matrices(times, observatory.location)
    np.savez(scratch_dir / "prepare.npz", theta=theta, matrices=matrices)
    print(
        f"Saved {scratch_dir / 'prepare.npz'}: {theta.size} events, "
        f"largest zenith angle {theta.max():.2f} deg"
    )


# ----------------------------------------------------------------------
# band
# ----------------------------------------------------------------------

def band(
    dataset: str, ring: int, k_start: int, k_stop: int, seed: int,
    scratch_dir: Path, block_size: int,
) -> None:
    """Lambda in every cap of ``ring`` in the skies ``k_start ... k_stop - 1``,
    saved to :func:`band_file`."""
    times, _, _, _, exposure_model = load_sample_and_model(dataset)
    n_total = times.size
    t0, tf = exposure_model.t0, exposure_model.tf
    with np.load(scratch_dir / "prepare.npz") as f:
        theta, matrices = f["theta"], f["matrices"]

    # The EventSample constructor requires a generator but draws nothing
    # from it here; the skies draw from their own streams.
    rng = np.random.default_rng(0)

    # The caps of the band, centred on the pixels of the ring.
    pixels = fov_rings()[ring]
    centres = np.column_stack(hp.pix2ang(NSIDE, pixels, lonlat=True))
    windows = [stc.SkyWindow(c, WINDOW_RADIUS_DEG) for c in centres]
    n_caps = pixels.size

    # An event is in a cap when the dot product of their unit vectors is at
    # least cos(radius), as in SkyWindow.contains.
    ra_c, dec_c = np.radians(centres).T
    centre_vecs = np.column_stack((
        np.cos(dec_c) * np.cos(ra_c), np.cos(dec_c) * np.sin(ra_c), np.sin(dec_c),
    ))
    cos_radius = np.cos(np.radians(WINDOW_RADIUS_DEG))
    dec_ring = centres[0, 1]
    # Widened as in SkyWindow.contains, so that rounding never drops an
    # event the exact test would keep.
    dec_half_width = WINDOW_RADIUS_DEG + 1e-9

    # Expected count and exposure over the whole run of every cap, filled
    # after its first exposure pass (see below).
    mu = np.full(n_caps, np.nan)
    eps_max = np.full(n_caps, np.nan)

    n_skies = k_stop - k_start
    n_sample = np.zeros((n_skies, n_caps), dtype=int)
    lambdas = np.full((n_skies, n_caps), np.nan)
    lambdas_exp = np.full((n_skies, n_caps), np.nan)
    n_failed = 0      # (sky, cap) with >= 2 events that still raised

    # disable=None turns the bar off when not on a terminal (HTCondor jobs).
    for block_start in tqdm(range(k_start, k_stop, block_size), desc="Blocks", disable=None):
        block_stop = min(block_start + block_size, k_stop)

        # --------------------------------------------------------------
        # The (event, cap) pairs of the band in every sky of the block:
        # index of the arrival time, cap, sky (counted from k_start). Caps
        # can overlap, so an event can be in more than one.
        # --------------------------------------------------------------
        time_index, cap, sky = [], [], []
        for k in range(block_start, block_stop):
            # The spawn key k gives every sky its own stream. RNGManager is
            # not used: it hashes the stream names to 32 bits, and among
            # ~1e5 names some would repeat, giving identical skies.
            sky_rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(k,)))
            order, sky_ra, sky_dec = scramble(theta, matrices, sky_rng)

            near = np.flatnonzero(np.abs(sky_dec - dec_ring) <= dec_half_width)
            ra_n, dec_n = np.radians(sky_ra[near]), np.radians(sky_dec[near])
            event_vecs = np.column_stack((
                np.cos(dec_n) * np.cos(ra_n), np.cos(dec_n) * np.sin(ra_n), np.sin(dec_n),
            ))
            cap_of_pair, event_of_pair = np.nonzero(centre_vecs @ event_vecs.T >= cos_radius)

            time_index.append(order[near[event_of_pair]])
            cap.append(cap_of_pair)
            sky.append(np.full(cap_of_pair.size, k - k_start))
        time_index, cap, sky = (np.concatenate(a) for a in (time_index, cap, sky))

        # Sorted by cap, then by sky, so that every (sky, cap) is one slice.
        sort = np.lexsort((sky, cap))
        time_index, cap, sky = time_index[sort], cap[sort], sky[sort]
        cap_edges = np.searchsorted(cap, np.arange(n_caps + 1))
        skies_in_block = np.arange(block_start - k_start, block_stop - k_start + 1)

        # --------------------------------------------------------------
        # Exposure of the events of all the skies of the block in every
        # cap, at its centre, then both Lambda in every sky
        # --------------------------------------------------------------
        for j in range(n_caps):
            lo, hi = cap_edges[j], cap_edges[j + 1]
            if hi == lo:
                continue
            eps = exposure_model.norm_cumul_exposure(times[time_index[lo:hi]], centres[j])

            # The exposure pass also stores the exposure over the whole run
            # in the model's cache, which the expected count and eps(tf)
            # then read with no second pass.
            if np.isnan(mu[j]):
                mu[j] = windows[j].expected_n_in_window(n_total, exposure_model)
                eps_max[j] = exposure_model.max_norm_cumul_exposure(centres[j])
            rate = mu[j] / eps_max[j]

            sky_edges = np.searchsorted(sky[lo:hi], skies_in_block)

            for s, a, b in zip(skies_in_block[:-1], sky_edges[:-1], sky_edges[1:]):
                n_sample[s, j] = b - a
                if b - a < 2:
                    continue
                # EventSample stores the coordinates but Lambda does not
                # use them; the cap centre stands in for them.
                subsample = stc.EventSample._from_arrays(
                    ra=np.full(b - a, centres[j, 0]), dec=np.full(b - a, centres[j, 1]),
                    n_total=n_total, t0=t0, tf=tf, rng=rng,
                    spatial_type="data", expected_n=mu[j],
                    window=windows[j], exposure_model=exposure_model,
                    exposure=eps[a:b], expected_exposure_rate=rate,
                )
                # Both raise on the same tied exposures, so a (sky, cap)
                # gets either both values or neither.
                try:
                    value = stc.lambda_estimator_suppressed_cdf(eps[a:b], rate, mu[j])
                    value_exp = stc.lambda_estimator(sample=subsample)
                except ValueError:
                    n_failed += 1
                    continue
                lambdas[s, j], lambdas_exp[s, j] = value, value_exp

    # Caps with no event in any sky of the job.
    for j in np.flatnonzero(np.isnan(mu)):
        mu[j] = windows[j].expected_n_in_window(n_total, exposure_model)
        eps_max[j] = exposure_model.max_norm_cumul_exposure(centres[j])

    path = band_file(scratch_dir, ring, k_start)
    np.savez_compressed(
        path,
        ring=ring,
        skies=np.arange(k_start, k_stop),   # (S,)    sky index k of every row
        pixels=pixels,                      # (C,)    HEALPix RING index of the pixel at the centre of every cap
        centres=centres,                    # (C, 2)  cap centres [RA, Dec] (deg)
        mu=mu,                              # (C,)    expected count
        eps_max=eps_max,                    # (C,)    eps(tf)
        n_sample=n_sample,                  # (S, C)  events per (sky, cap)
        lambdas=lambdas,                    # (S, C)  Lambda with the suppressed CDF; NaN with n < 2 or failed
        lambdas_exp=lambdas_exp,            # (S, C)  Lambda with the exponential CDF; NaN likewise
        n_failed=n_failed,
    )
    print(
        f"Ring {ring}, skies {k_start}-{k_stop - 1}: {n_caps} caps, "
        f"{np.count_nonzero(np.isfinite(lambdas))} Lambda values, {n_failed} failed. "
        f"Saved {path}"
    )


# ----------------------------------------------------------------------
# merge
# ----------------------------------------------------------------------

def plot_band(
    lambdas: np.ndarray, lambdas_exp: np.ndarray, mu: np.ndarray, *,
    title: str, output_path: Path,
):
    """Histograms of the Lambda values with the suppressed CDF and with the
    exponential CDF, with Poisson errors, against the marginal density of
    Lambda for ``n >= 2``, averaged over the values with the ``mu`` of each
    and scaled to counts per bin. The two arrays come from the same
    (sky, cap), so they share ``mu``. Returns a KS test of each against the
    marginal.

    The averaged density and CDF are evaluated on a grid of 4000 points
    and interpolated, so that their cost does not grow with the number of
    values.
    """
    # Caps with n < 2 have no Lambda, so every cap follows the marginal
    # density divided by its P(n >= 2). The average runs over the distinct
    # values of mu, each weighted by the number of Lambda values with it.
    mu, n_mu = np.unique(mu, return_counts=True)
    weight = n_mu / n_mu.sum()
    p_two = 1.0 - np.exp(-mu) * (1.0 + mu)

    low = min(lambdas.min(), lambdas_exp.min())
    high = max(lambdas.max(), lambdas_exp.max())
    grid = np.linspace(max(low, 1e-6), high, 4000)
    pdf_grid = np.zeros_like(grid)
    sf_grid = np.zeros_like(grid)
    for m, p, w in zip(mu, p_two, weight):
        pdf_grid += w * stc.lambda_marginal_pdf(grid, m) / p
        sf_grid += w * stc.lambda_marginal_sf(grid, m) / p

    def cdf(x):
        return np.interp(x, grid, 1.0 - sf_grid)

    ks = scp.kstest(lambdas, cdf)
    ks_exp = scp.kstest(lambdas_exp, cdf)

    bin_width = 4.0
    bins = np.arange(bin_width * np.floor(low / bin_width), high + bin_width, bin_width)
    bin_centres = 0.5 * (bins[1:] + bins[:-1])

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for values, colour, label in (
        (lambdas, "C0", rf"suppressed CDF, KS $p = {ks.pvalue:.2g}$"),
        (lambdas_exp, "C1", rf"exponential CDF, KS $p = {ks_exp.pvalue:.2g}$"),
    ):
        counts, _ = np.histogram(values, bins=bins)
        filled = counts > 0
        ax.hist(values, bins=bins, histtype="step", linewidth=1.2, color=colour, label=label)
        ax.errorbar(bin_centres[filled], counts[filled], yerr=np.sqrt(counts[filled]),
                    fmt="none", ecolor=colour, elinewidth=0.8, capsize=1.5)
    ax.plot(grid, lambdas.size * bin_width * pdf_grid,
            color="black", linestyle="--", linewidth=0.8,
            label=rf"marginal, $\bar\mu = {np.sum(weight * mu):.1f}$")

    # Headroom above the data for the legend.
    peak = max(
        lambdas.size * bin_width * pdf_grid.max(),
        np.histogram(lambdas, bins=bins)[0].max(),
        np.histogram(lambdas_exp, bins=bins)[0].max(),
    )
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(0.0, 1.4 * peak)
    ax.set_xlabel(r"$\Lambda$")
    ax.set_ylabel("caps")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=6)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return ks, ks_exp


def merge(scratch_dir: Path, output_dir: Path, keep_scratch: bool) -> int:
    """Merge the jobs of ``scratch_dir`` per band into ``output_dir``.

    The jobs expected are those of ``scratch_dir/params.txt``. The scratch
    directory is removed only when all of them arrived, unless
    ``keep_scratch``. Returns 0 then, and 1 when any is missing.
    """
    jobs = [line.split() for line in (scratch_dir / "params.txt").read_text().splitlines() if line]

    paths_by_ring: dict[int, list[Path]] = {}
    missing = []
    for _, ring, k_start, _, _ in jobs:
        path = band_file(scratch_dir, int(ring), int(k_start))
        if path.exists():
            paths_by_ring.setdefault(int(ring), []).append(path)
        else:
            missing.append(path.name)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for ring, paths in tqdm(sorted(paths_by_ring.items()), desc="Bands", disable=None):
        # The file names carry k_start zero-padded, so they sort in sky order.
        parts = []
        for path in sorted(paths):
            with np.load(path) as f:
                parts.append({key: f[key] for key in f.files})
        first = parts[0]

        skies = np.concatenate([p["skies"] for p in parts])
        n_sample = np.concatenate([p["n_sample"] for p in parts])
        lambdas = np.concatenate([p["lambdas"] for p in parts])
        lambdas_exp = np.concatenate([p["lambdas_exp"] for p in parts])
        n_failed = int(sum(p["n_failed"] for p in parts))
        dec_ring = float(first["centres"][0, 1])

        np.savez_compressed(
            output_dir / f"band_ring{ring:03d}.npz",
            ring=ring, dec=dec_ring, radius_deg=WINDOW_RADIUS_DEG, skies=skies,
            pixels=first["pixels"], centres=first["centres"],
            mu=first["mu"], eps_max=first["eps_max"], n_sample=n_sample,
            lambdas=lambdas, lambdas_exp=lambdas_exp, n_failed=n_failed,
        )

        # (sky, cap) with >= 2 events whose Lambda raised are left out of
        # the histograms; they are not a random subset, so they are reported.
        if n_failed > 0:
            print(f"Ring {ring}: {n_failed} (sky, cap) with n >= 2 failed and are left out")

        # Both Lambda are finite on the same (sky, cap).
        tested = np.isfinite(lambdas)
        values, values_exp = lambdas[tested], lambdas_exp[tested]
        stats = {}
        if values.size > 0:
            ks, ks_exp = plot_band(
                values, values_exp, np.broadcast_to(first["mu"], lambdas.shape)[tested],
                title=(
                    rf"ring {ring}, Dec $= {dec_ring:.1f}^\circ$, "
                    rf"$N_{{\rm side}} = {NSIDE}$, {values.size} values"
                ),
                output_path=output_dir / f"band_ring{ring:03d}.png",
            )
            for suffix, v, k in (("", values, ks), ("_exp", values_exp, ks_exp)):
                stats.update({
                    f"lambda{suffix}_mean": float(v.mean()),
                    f"lambda{suffix}_std": float(v.std()),
                    f"ks{suffix}_statistic": float(k.statistic),
                    f"ks{suffix}_pvalue": float(k.pvalue),
                })
        else:
            print(f"Ring {ring}: no Lambda values, no histogram")

        summary.append({
            "ring": ring,
            "dec_deg": dec_ring,
            "n_caps": int(first["pixels"].size),
            "n_skies": int(skies.size),
            "mu_mean": float(first["mu"].mean()),
            "n_lambda": int(values.size),
            "n_failed": n_failed,
            **stats,
        })

    complete = not missing
    with open(output_dir / "summary.json", "w") as f:
        json.dump({
            "merged": datetime.now().isoformat(timespec="seconds"),
            "jobs_expected": len(jobs),
            "jobs_missing": missing,
            "bands": summary,
        }, f, indent=2)
    shutil.copy(scratch_dir / "params.txt", output_dir / "params.txt")

    print(
        f"Merged {len(jobs) - len(missing)} of {len(jobs)} jobs into "
        f"{len(summary)} bands in {output_dir}"
    )
    if complete and not keep_scratch:
        shutil.rmtree(scratch_dir)
        print(f"Removed {scratch_dir}")
    elif not complete:
        print(f"{len(missing)} jobs missing; kept {scratch_dir}")
    return 0 if complete else 1


# ----------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monte Carlo distribution of Lambda under isotropy, per declination band."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("plan", help="Write the job list.")
    p.add_argument("--dataset", default="dataset_0")
    p.add_argument("--n-min", type=int, required=True, help="Minimum Lambda values per band.")
    p.add_argument("--max-skies-per-job", type=int, default=30000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)

    p = sub.add_parser("prepare", help="Zenith angles and local-to-ICRS matrices.")
    p.add_argument("--dataset", default="dataset_0")
    p.add_argument("--scratch-dir", type=Path, required=True)

    p = sub.add_parser("band", help="Lambda in one band for a range of skies.")
    p.add_argument("--dataset", default="dataset_0")
    p.add_argument("--ring", type=int, required=True)
    p.add_argument("--k-start", type=int, required=True)
    p.add_argument("--k-stop", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scratch-dir", type=Path, required=True)
    p.add_argument("--block-size", type=int, default=1000, help="Skies per block.")

    p = sub.add_parser("merge", help="Merge the jobs per band into output/.")
    p.add_argument("--scratch-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--keep-scratch", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.command == "plan":
        plan(args.dataset, args.n_min, args.max_skies_per_job, args.seed, args.output)
    elif args.command == "prepare":
        prepare(args.dataset, args.scratch_dir)
    elif args.command == "band":
        band(args.dataset, args.ring, args.k_start, args.k_stop, args.seed,
             args.scratch_dir, args.block_size)
    elif args.command == "merge":
        raise SystemExit(merge(args.scratch_dir, args.output_dir, args.keep_scratch))
