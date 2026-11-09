"""
Monte Carlo distribution of Lambda under isotropy, per ring of HEALPix
pixels.

The windows are the HEALPix pixels (RING ordering, ``NSIDE``) lying
entirely inside the field of view (:func:`fov_pixels`). A band is the
pixels of one ring, numbered as in healpy from 1 at the north pole.

The skies are drawn by ``isotropic_scrambling.isotropic_sky``: the whole sky
as a Poisson process of ``lam = N - 1`` events per unit of whole-sky
cumulative exposure, ``N`` the events of the sample, with the zenith angles
of the sample. Sky ``k`` comes from the generator of
``SeedSequence(seed, spawn_key=(k,))``, and every sky is drawn once, in one
job. In every (sky, pixel) with at least 2 events, the suppressed Lambda
(``stc.lambda_estimator_suppressed_cdf``) is computed with the expected
count of that sky, ``mu = N_sky E(pixel) / E_sky``, and the rate
``mu / eps(tf)`` of the pixel.

One subcommand per node of the DAG in ``condor/h0_mc_distribution/``:

    plan     Skies ``0 ... n_skies - 1`` split into jobs of at most
             ``skies_per_job`` skies, one line per job:
             ``dataset k_start k_stop seed``.
    prepare  Zenith angle of every event of the sample, ``lam`` and the
             pixels with their rings, computed once for all the jobs.
    skies    Skies ``k_start ... k_stop - 1``. Every sky is drawn and its
             events assigned to the pixels; for every pixel, the normalised
             cumulative exposure of its events in all the skies of the job
             is computed in one pass at the pixel centre
             (``ExposureModel.norm_cumul_exposure``), then Lambda in every
             sky.
    merge    The jobs in sky order: the table of every (sky, pixel), and per
             ring the band npz, the histogram of Lambda and a summary. The
             scratch directory is removed once every job is in.

The model, ``t0`` and ``tf`` are built as in ``run_blind_search_mock.py``.
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import astropy.units as u
import healpy as hp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from astropy.utils import iers
from tqdm import tqdm

import spacetimecorr as stc
from spacetimecorr.io import load_active_area, load_mock_sample

from isotropic_scrambling import exposure_time_table, isotropic_sky, zenith_angles

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
NSIDE = 32                     # HEALPix resolution of the pixels
WINDOW_RADIUS_DEG = 1.05       # radius of the caps of fov_rings (band_sliding_window.py)
N_BINS = 60                    # histogram bins over the range of Lambda of a band

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
    is inside when ``|dec - latitude| + radius < theta_max``. Used by
    ``band_sliding_window.py``; the pipeline uses :func:`fov_pixels`.
    """
    _, dec_centres = hp.pix2ang(NSIDE, np.arange(hp.nside2npix(NSIDE)), lonlat=True)
    dec_rings = np.unique(dec_centres)[::-1]      # north to south, as the ring numbers
    inside_fov = np.abs(dec_rings - LATITUDE_PA) + WINDOW_RADIUS_DEG < THETA_MAX_DEG
    return {
        i + 1: np.flatnonzero(dec_centres == d)
        for i, d in enumerate(dec_rings) if inside_fov[i]
    }


def fov_pixels() -> tuple[np.ndarray, np.ndarray]:
    """RING indices of the pixels lying entirely inside the field of view,
    with all four corners at ``|dec - latitude| < theta_max``, and the ring
    of each, numbered from 1 at the north pole as in healpy."""
    ipix = np.arange(hp.nside2npix(NSIDE))
    _, dec_centres = hp.pix2ang(NSIDE, ipix, lonlat=True)
    corners = hp.boundaries(NSIDE, ipix, step=1)           # (pixel, xyz, corner)
    dec_corners = np.degrees(np.arcsin(corners[:, 2, :]))
    inside = np.all(np.abs(dec_corners - LATITUDE_PA) < THETA_MAX_DEG, axis=1)

    # The centres of a ring share their declination; counted from the north.
    dec_rings = np.unique(dec_centres)
    rings = dec_rings.size - np.searchsorted(dec_rings, dec_centres)
    return ipix[inside], rings[inside]


# ----------------------------------------------------------------------
# plan
# ----------------------------------------------------------------------

def plan(dataset: str, n_skies: int, skies_per_job: int, seed: int, output: Path) -> None:
    """Write the job list for the skies ``0 ... n_skies - 1``, one line
    ``dataset k_start k_stop seed`` per job of at most ``skies_per_job``
    skies, to ``output``."""
    lines = [
        f"{dataset} {k_start} {min(k_start + skies_per_job, n_skies)} {seed}"
        for k_start in range(0, n_skies, skies_per_job)
    ]
    output.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(lines)} jobs for {n_skies} skies to {output}")


# ----------------------------------------------------------------------
# prepare
# ----------------------------------------------------------------------

def prepare(dataset: str, scratch_dir: Path) -> None:
    """Save to ``scratch_dir/prepare.npz`` what every job shares: the zenith
    angle of every event of ``dataset``, the whole-sky rate ``lam`` of the
    skies, and the pixels of :func:`fov_pixels` with their rings.

    The run starts and ends with an event, so its ``N`` events span ``N - 1``
    gaps, and ``lam = N - 1`` events per unit of the whole-sky cumulative
    exposure (``isotropic_scrambling.isotropic_sky``).
    """
    times, ra, dec, observatory, _ = load_sample_and_model(dataset)
    theta = zenith_angles(times, ra, dec, observatory.location)
    lam = times.size - 1.0
    pixels, rings = fov_pixels()
    np.savez(scratch_dir / "prepare.npz", theta=theta, lam=lam, pixels=pixels, rings=rings)
    print(
        f"Saved {scratch_dir / 'prepare.npz'}: {theta.size} events, "
        f"largest zenith angle {theta.max():.2f} deg, lambda = {lam:.0f}, "
        f"{pixels.size} pixels in {np.unique(rings).size} rings"
    )


# ----------------------------------------------------------------------
# skies
# ----------------------------------------------------------------------

def job_file(scratch_dir: Path, k_start: int) -> Path:
    """Results of the job starting at sky ``k_start``."""
    return scratch_dir / f"skies_k{k_start:07d}.npz"


def skies(dataset: str, k_start: int, k_stop: int, seed: int, scratch_dir: Path) -> None:
    """Lambda in every pixel of :func:`fov_pixels` in the skies
    ``k_start ... k_stop - 1``, saved to :func:`job_file`."""
    _, _, _, observatory, exposure_model = load_sample_and_model(dataset)
    t0 = exposure_model.t0
    with np.load(scratch_dir / "prepare.npz") as f:
        theta, lam, pixels, rings = f["theta"], float(f["lam"]), f["pixels"], f["rings"]
    edges, cumul = exposure_time_table(exposure_model)

    windows = [stc.HealpixWindow(nside=NSIDE, ipix=int(p)) for p in pixels]
    n_pixels = pixels.size
    # Column of every pixel of the sphere in the arrays below, -1 for the
    # pixels not in the field of view.
    column = np.full(hp.nside2npix(NSIDE), -1)
    column[pixels] = np.arange(n_pixels)

    # --------------------------------------------------------------
    # The events of every sky in the pixels: arrival time, column, sky
    # (counted from k_start)
    # --------------------------------------------------------------
    n_skies = k_stop - k_start
    n_total = np.zeros(n_skies, dtype=int)
    seconds, col, sky = [], [], []
    # disable=None turns the bar off when not on a terminal (HTCondor jobs).
    for s, k in enumerate(tqdm(range(k_start, k_stop), desc="Skies", disable=None)):
        # The spawn key k gives every sky its own stream. RNGManager is not
        # used: it hashes the stream names to 32 bits, and among ~1e5 names
        # some would repeat, giving identical skies.
        sky_rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(k,)))
        sky_seconds, sky_ra, sky_dec = isotropic_sky(
            lam, theta, edges, cumul, t0, observatory.location, sky_rng,
        )
        n_total[s] = sky_seconds.size
        sky_col = column[hp.ang2pix(NSIDE, sky_ra, sky_dec, lonlat=True)]
        kept = sky_col >= 0
        seconds.append(sky_seconds[kept])
        col.append(sky_col[kept])
        sky.append(np.full(np.count_nonzero(kept), s))
    seconds, col, sky = (np.concatenate(a) for a in (seconds, col, sky))

    # Sorted by column, then by sky, so that every (sky, pixel) is one slice.
    sort = np.lexsort((sky, col))
    seconds, col, sky = seconds[sort], col[sort], sky[sort]
    col_edges = np.searchsorted(col, np.arange(n_pixels + 1))

    # --------------------------------------------------------------
    # Exposure of the events of all the skies in every pixel, at its
    # centre, then Lambda in every sky with mu from its own count
    # --------------------------------------------------------------
    mu_per_event = np.full(n_pixels, np.nan)     # mu = N * mu_per_event
    eps_max = np.full(n_pixels, np.nan)
    n_sample = np.zeros((n_skies, n_pixels), dtype=int)
    lambdas = np.full((n_skies, n_pixels), np.nan)
    n_failed = 0      # (sky, pixel) with >= 2 events that still raised

    for j in tqdm(range(n_pixels), desc="Pixels", disable=None):
        lo, hi = col_edges[j], col_edges[j + 1]
        centre = windows[j].centre
        if hi > lo:
            eps = exposure_model.norm_cumul_exposure(t0 + seconds[lo:hi] * u.s, centre)

        # The exposure pass also stores the exposure over the whole run in
        # the model's cache, which mu and eps(tf) then read with no second
        # pass (one pass for a pixel with no event).
        mu_per_event[j] = windows[j].expected_n_in_window(1.0, exposure_model)
        eps_max[j] = exposure_model.max_norm_cumul_exposure(centre)
        if hi == lo:
            continue

        sky_edges = np.searchsorted(sky[lo:hi], np.arange(n_skies + 1))
        n_sample[:, j] = np.diff(sky_edges)
        for s in np.flatnonzero(n_sample[:, j] >= 2):
            mu = n_total[s] * mu_per_event[j]
            try:
                lambdas[s, j] = stc.lambda_estimator_suppressed_cdf(
                    eps[sky_edges[s]:sky_edges[s + 1]], mu / eps_max[j], mu,
                )
            except ValueError:
                n_failed += 1

    path = job_file(scratch_dir, k_start)
    np.savez_compressed(
        path,
        skies=np.arange(k_start, k_stop),   # (S,)    sky index k of every row
        n_total=n_total,                    # (S,)    events of every sky
        pixels=pixels,                      # (P,)    HEALPix RING index of every column
        rings=rings,                        # (P,)    ring of every pixel
        mu_per_event=mu_per_event,          # (P,)    E(pixel) / E_sky: mu = n_total * mu_per_event
        eps_max=eps_max,                    # (P,)    eps(tf)
        n_sample=n_sample,                  # (S, P)  events per (sky, pixel)
        lambdas=lambdas,                    # (S, P)  suppressed Lambda; NaN with n < 2 or failed
        n_failed=n_failed,
    )
    print(
        f"Skies {k_start}-{k_stop - 1}: {n_pixels} pixels, "
        f"{np.count_nonzero(np.isfinite(lambdas))} Lambda values, {n_failed} failed. "
        f"Saved {path}"
    )


# ----------------------------------------------------------------------
# merge
# ----------------------------------------------------------------------

def plot_band(lambdas: np.ndarray, *, label: str, title: str, output_path: Path) -> None:
    """Histogram of the suppressed Lambda values of a band, with Poisson
    errors."""
    bins = np.linspace(lambdas.min(), lambdas.max(), N_BINS + 1)
    bin_centres = 0.5 * (bins[1:] + bins[:-1])
    counts, _ = np.histogram(lambdas, bins=bins)
    filled = counts > 0

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.hist(lambdas, bins=bins, histtype="step", linewidth=1.2, color="C0", label=label)
    ax.errorbar(bin_centres[filled], counts[filled], yerr=np.sqrt(counts[filled]),
                fmt="none", ecolor="C0", elinewidth=0.8, capsize=1.5)

    # Headroom above the data for the legend.
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(0.0, 1.4 * counts.max())
    ax.set_xlabel(r"$\Lambda$")
    ax.set_ylabel("counts")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=6)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def merge(scratch_dir: Path, output_dir: Path, keep_scratch: bool) -> int:
    """Merge the jobs of ``scratch_dir`` into ``output_dir``: the table of
    every (sky, pixel) in ``skies.npz``, the npz and the plot of every ring
    in ``bands/``, the summary and the parameters.

    The jobs expected are those of ``scratch_dir/params.txt``. The scratch
    directory is removed only when all of them arrived, unless
    ``keep_scratch``. Returns 0 then, and 1 when any is missing.
    """
    jobs = [line.split() for line in (scratch_dir / "params.txt").read_text().splitlines() if line]
    present, missing = [], []
    for _, k_start, k_stop, _ in jobs:
        path = job_file(scratch_dir, int(k_start))
        if path.exists():
            present.append((int(k_start), int(k_stop), path))
        else:
            missing.append(path.name)
    present.sort()

    # --------------------------------------------------------------
    # The table of every (sky, pixel), in sky order. Lambda in float32
    # (7 significant digits) and n in int16 keep it at ~1.3 GB for 25000
    # skies.
    # --------------------------------------------------------------
    with np.load(present[0][2]) as f:
        pixels, rings = f["pixels"], f["rings"]
        mu_per_event, eps_max = f["mu_per_event"], f["eps_max"]
    n_rows = sum(k_stop - k_start for k_start, k_stop, _ in present)
    skies = np.empty(n_rows, dtype=int)
    n_total = np.empty(n_rows, dtype=int)
    n_sample = np.empty((n_rows, pixels.size), dtype=np.int16)
    lambdas = np.empty((n_rows, pixels.size), dtype=np.float32)
    row = 0
    for k_start, k_stop, path in tqdm(present, desc="Jobs", disable=None):
        rows = slice(row, row + k_stop - k_start)
        with np.load(path) as f:
            skies[rows], n_total[rows] = f["skies"], f["n_total"]
            n_sample[rows], lambdas[rows] = f["n_sample"], f["lambdas"]
        row += k_stop - k_start

    np.savez(
        output_dir / "skies.npz",
        skies=skies,                # (S,)    sky index k of every row
        n_total=n_total,            # (S,)    events of every sky
        pixels=pixels,              # (P,)    HEALPix RING index of every column
        rings=rings,                # (P,)    ring of every pixel
        mu_per_event=mu_per_event,  # (P,)    mu = n_total * mu_per_event
        eps_max=eps_max,            # (P,)    eps(tf)
        n_sample=n_sample,          # (S, P)  events per (sky, pixel)
        lambdas=lambdas,            # (S, P)  suppressed Lambda; NaN with n < 2 or failed
    )

    # --------------------------------------------------------------
    # Every ring: its columns of the table and the histogram
    # --------------------------------------------------------------
    bands_dir = output_dir / "bands"
    bands_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for ring in tqdm(np.unique(rings), desc="Bands", disable=None):
        cols = np.flatnonzero(rings == ring)
        dec_ring = float(hp.pix2ang(NSIDE, int(pixels[cols[0]]), lonlat=True)[1])
        mu = n_total[:, None] * mu_per_event[None, cols]
        band_lambdas, band_n = lambdas[:, cols], n_sample[:, cols]

        np.savez_compressed(
            bands_dir / f"band_ring{ring:03d}.npz",
            ring=ring, dec=dec_ring, skies=skies, n_total=n_total,
            pixels=pixels[cols],                 # (P,)    HEALPix RING index of every column
            mu_per_event=mu_per_event[cols],     # (P,)
            eps_max=eps_max[cols],               # (P,)
            mu=mu,                               # (S, P)  expected count of every (sky, pixel)
            n_sample=band_n,                     # (S, P)
            lambdas=band_lambdas,                # (S, P)  suppressed Lambda; NaN with n < 2 or failed
        )

        # (sky, pixel) with >= 2 events whose Lambda raised are left out of
        # the histogram; they are not a random subset, so they are reported.
        tested = np.isfinite(band_lambdas)
        n_failed = int(np.count_nonzero((band_n >= 2) & ~tested))
        if n_failed > 0:
            print(f"Ring {ring}: {n_failed} (sky, pixel) with n >= 2 failed and are left out")

        values = band_lambdas[tested]
        stats = {}
        if values.size > 0:
            plot_band(
                values,
                label=rf"suppressed $\Lambda$, $\bar\mu = {mu[tested].mean():.1f}$",
                title=rf"ring {ring}, Dec $= {dec_ring:.1f}^\circ$",
                output_path=bands_dir / f"band_ring{ring:03d}.png",
            )
            stats = {"lambda_mean": float(values.mean()), "lambda_std": float(values.std())}
        else:
            print(f"Ring {ring}: no Lambda values, no histogram")

        summary.append({
            "ring": int(ring),
            "dec_deg": dec_ring,
            "n_pixels": int(cols.size),
            "n_skies": int(skies.size),
            "mu_mean": float(mu.mean()),
            "mu_min": float(mu.min()),
            "mu_max": float(mu.max()),
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
        description="Monte Carlo distribution of Lambda under isotropy, per ring of HEALPix pixels."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("plan", help="Write the job list.")
    p.add_argument("--dataset", default="dataset_0")
    p.add_argument("--n-skies", type=int, required=True)
    p.add_argument("--skies-per-job", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)

    p = sub.add_parser("prepare", help="Zenith angles, whole-sky rate and pixels.")
    p.add_argument("--dataset", default="dataset_0")
    p.add_argument("--scratch-dir", type=Path, required=True)

    p = sub.add_parser("skies", help="Lambda in every pixel for a range of skies.")
    p.add_argument("--dataset", default="dataset_0")
    p.add_argument("--k-start", type=int, required=True)
    p.add_argument("--k-stop", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scratch-dir", type=Path, required=True)

    p = sub.add_parser("merge", help="Merge the jobs per band into output/.")
    p.add_argument("--scratch-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--keep-scratch", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.command == "plan":
        plan(args.dataset, args.n_skies, args.skies_per_job, args.seed, args.output)
    elif args.command == "prepare":
        prepare(args.dataset, args.scratch_dir)
    elif args.command == "skies":
        skies(args.dataset, args.k_start, args.k_stop, args.seed, args.scratch_dir)
    elif args.command == "merge":
        raise SystemExit(merge(args.scratch_dir, args.output_dir, args.keep_scratch))
