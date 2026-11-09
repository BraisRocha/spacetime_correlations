"""
One isotropic sky scrambled from a mock sample, binned in HEALPix pixels:
Lambda of every pixel and its p-value from the conditional pdf of Lambda.

The sky is scrambled as in ``h0_pipeline.py`` (``isotropic_scrambling``),
from the generator of ``SeedSequence(seed, spawn_key=(sky,))``. The default
sky is above the 25000 of the h0_mc_distribution run, so it is not one of
its skies.

The pixels are scanned as in the nside=32 blind search: the pixels with
centre in the field of view, without those with a corner outside it, and
the exposure of the events of a pixel evaluated at its centre.

Lambda is computed with two CDF of the gaps between events:

- exponential (``stc.lambda_estimator``);
- suppressed (``stc.lambda_estimator_suppressed_cdf``).

The p-value of each is ``P(Lambda >= Lambda_obs | n)`` with the observed
number of events ``n`` (``stc.lambda_conditional_pvalue_and_sigma``).

Output, in ``output/analysis/mocksamples/isotropic_sky/<dataset>_sky<sky>/``:
``results.npz`` (the values of every pixel, NaN where not tested),
``pvalues.png`` and ``significances.png``.
"""

import os

# One thread per worker: numpy's linear algebra would otherwise spread over
# all the cores of the machine in every worker.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import argparse
import multiprocessing
from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as scp
from tqdm import tqdm

import spacetimecorr as stc

# The import also loads the matplotlibrc of the plots/ folder.
from h0_pipeline import LATITUDE_PA, NSIDE, PROJECT_ROOT, THETA_MAX_DEG, load_sample_and_model
from isotropic_scrambling import local_to_icrs_matrices, scramble, zenith_angles

# Lambda -> (legend label, colour).
ESTIMATORS = {
    "exponential": ("exponential CDF", "C1"),
    "suppressed": ("suppressed CDF", "C0"),
}

# Set in main before the workers are forked, which inherit it.
SCAN = {}


# ----------------------------------------------------------------------
# Scan
# ----------------------------------------------------------------------

def scan_pixel(i: int) -> tuple:
    """``(i, mu, Lambda of every estimator)`` of the pixel ``i`` of the grid."""
    window = SCAN["grid"][i]
    model = SCAN["model"]
    events = SCAN["events"][SCAN["starts"][i]:SCAN["ends"][i]]

    # The exposure pass also stores the exposure over the whole run in the
    # model's cache, which the expected count and eps(tf) then read with no
    # second pass.
    exposure = model.norm_cumul_exposure(SCAN["times"][SCAN["order"][events]], window.centre)
    mu = window.expected_n_in_window(SCAN["n_total"], model)
    rate = mu / model.max_norm_cumul_exposure(window.centre)

    # EventSample stores the coordinates but Lambda does not use them; the
    # pixel centre stands in for them.
    subsample = stc.EventSample._from_arrays(
        ra=np.full(events.size, window.centre[0]), dec=np.full(events.size, window.centre[1]),
        n_total=SCAN["n_total"], t0=SCAN["t0"], tf=SCAN["tf"], rng=np.random.default_rng(0),
        spatial_type="data", expected_n=mu, window=window, exposure_model=model,
        exposure=exposure, expected_exposure_rate=rate,
    )
    # Both raise on the same tied exposures.
    try:
        values = (
            stc.lambda_estimator(sample=subsample),
            stc.lambda_estimator_suppressed_cdf(exposure, rate, mu),
        )
    except ValueError:
        values = (np.nan, np.nan)
    return (i, mu, *values)


# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------

def plot_pvalues(pvalues: dict, *, title: str, output_path: Path) -> None:
    """Histogram of the p-values of every estimator against a uniform
    distribution."""
    bins = np.linspace(0.0, 1.0, 41)
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    top = 0
    for name, p in pvalues.items():
        label, colour = ESTIMATORS[name]
        ax.hist(p, bins=bins, histtype="step", linewidth=1.2, color=colour, label=label)
        top = max(top, np.histogram(p, bins=bins)[0].max())
    ax.axhline(p.size / (bins.size - 1), color="black", linestyle="--", linewidth=0.8, label="uniform")
    ax.set_ylim(0.0, 1.45 * top)
    ax.set_xlabel(r"$p$-value")
    ax.set_ylabel("counts")
    ax.set_title(title)
    ax.legend(loc="upper center", fontsize=6, ncol=2)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_significances(sigmas: dict, *, title: str, output_path: Path) -> None:
    """Histogram of the significances of every estimator, in log scale,
    against a standard normal distribution."""
    bins = np.arange(-5.0, 5.25, 0.25)
    grid = np.linspace(-5.0, 5.0, 500)
    size = next(iter(sigmas.values())).size
    curve = size * (bins[1] - bins[0]) * scp.norm.pdf(grid)

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for name, s in sigmas.items():
        label, colour = ESTIMATORS[name]
        ax.hist(s, bins=bins, histtype="step", linewidth=1.2, color=colour, label=label)
    ax.plot(grid, curve, color="black", linestyle="--", linewidth=0.8, label=r"$\mathcal{N}(0, 1)$")
    ax.set_yscale("log")
    ax.set_xlim(-5.0, 5.0)
    ax.set_ylim(0.5, 1.5 * curve.max())
    ax.set_xlabel(r"Significance [$\sigma$]")
    ax.set_ylabel("counts")
    ax.set_title(title)
    ax.legend(loc="lower center", fontsize=6)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main(dataset: str, sky: int, seed: int, n_jobs: int) -> None:
    times, ra, dec, observatory, model = load_sample_and_model(dataset)
    n_total = times.size

    theta = zenith_angles(times, ra, dec, observatory.location)
    matrices = local_to_icrs_matrices(times, observatory.location)
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(sky,)))
    order, sky_ra, sky_dec = scramble(theta, matrices, rng)
    del theta, matrices

    grid = stc.HealpixGrid(model, nside=NSIDE)

    # A pixel is inside the field of view when its four corners are.
    corners = hp.boundaries(NSIDE, grid.pixels, step=1)
    corners_dec = np.degrees(np.arcsin(corners[:, 2, :]))
    inside_fov = np.all(np.abs(corners_dec - LATITUDE_PA) < THETA_MAX_DEG, axis=1)

    # The events of pixel i of the grid are events[starts[i]:ends[i]].
    event_pixel = grid.pixel_of(sky_ra, sky_dec)
    events = np.argsort(event_pixel, kind="stable")
    starts = np.searchsorted(event_pixel[events], grid.pixels, side="left")
    ends = np.searchsorted(event_pixel[events], grid.pixels, side="right")
    n_sample = ends - starts

    SCAN.update(
        grid=grid, model=model, times=times, order=order, events=events,
        starts=starts, ends=ends, n_total=n_total, t0=times.min(), tf=times.max(),
    )

    # Lambda needs at least 2 events.
    to_scan = np.flatnonzero(inside_fov & (n_sample >= 2))
    print(f"{len(grid)} pixels, {np.count_nonzero(~inside_fov)} on the FoV edge, "
          f"{to_scan.size} to scan with {n_jobs} workers")

    expected_n = np.full(len(grid), np.nan)
    lambdas = {name: np.full(len(grid), np.nan) for name in ESTIMATORS}
    # fork, so that the workers inherit SCAN; Python 3.14 no longer uses it
    # by default.
    with multiprocessing.get_context("fork").Pool(n_jobs) as pool:
        results = pool.imap_unordered(scan_pixel, to_scan, chunksize=8)
        for i, mu, *values in tqdm(results, total=to_scan.size, desc="Pixels scanned", disable=None):
            expected_n[i] = mu
            for name, value in zip(ESTIMATORS, values):
                lambdas[name][i] = value

    tested = np.isfinite(lambdas["exponential"])
    n_tested = n_sample[tested]
    pvalues, sigmas = {}, {}
    arrays = {}
    pull = (n_tested - expected_n[tested]) / np.sqrt(expected_n[tested])
    for name in ESTIMATORS:
        p, s = stc.lambda_conditional_pvalue_and_sigma(lambdas[name][tested], n_tested)
        pvalues[name], sigmas[name] = p, s
        arrays[f"pvalues_{name}"] = np.full(len(grid), np.nan)
        arrays[f"pvalues_{name}"][tested] = p
        arrays[f"sigma_{name}"] = np.full(len(grid), np.nan)
        arrays[f"sigma_{name}"][tested] = s
        arrays[f"lambda_{name}"] = lambdas[name]

        ks = scp.kstest(p, "uniform")
        print(f"{name}: KS against uniform stat {ks.statistic:.4f}, p = {ks.pvalue:.3g}; "
              f"sigma mean {s.mean():.3f}, std {s.std():.3f}, range [{s.min():.2f}, {s.max():.2f}]; "
              f"correlation with (n - mu)/sqrt(mu) {np.corrcoef(s, pull)[0, 1]:.3f}")
        print("   " + ", ".join(
            f"p < {t:g}: {np.count_nonzero(p < t)} (expected {t * p.size:.1f})"
            for t in (0.5, 0.1, 0.01, 1e-3)
        ) + f", p > 0.9: {np.count_nonzero(p > 0.9)}")

    output_dir = PROJECT_ROOT / "output" / "analysis" / "mocksamples" / "isotropic_sky" / f"{dataset}_sky{sky}"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "results.npz",
        pixels=grid.pixels, centres=grid.centres, inside_fov=inside_fov,
        n_sample=n_sample, expected_n=expected_n, sky=sky, seed=seed, **arrays,
    )
    title = rf"isotropic sky, $N_{{\rm side}} = {NSIDE}$"
    plot_pvalues(pvalues, title=title, output_path=output_dir / "pvalues.png")
    plot_significances(sigmas, title=title, output_path=output_dir / "significances.png")
    print(f"{np.count_nonzero(tested)} pixels tested. Saved to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Conditional p-values of the pixels of one isotropic scrambled sky."
    )
    parser.add_argument("--dataset", default="dataset_0", help="Mock sample to scramble.")
    parser.add_argument("--sky", type=int, default=100_000, help="Spawn key of the sky.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed.")
    parser.add_argument("--n-jobs", type=int, default=8, help="Worker processes.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.dataset, args.sky, args.seed, args.n_jobs)
