"""
Diagnosis of a mock sample: gaps between the exposures of the events in
every HEALPix pixel of the field of view.

The pipeline is:

    1. Read the sample and take every HEALPix pixel (RING ordering,
       resolution ``nside``) inside the FoV. The pixels do not overlap, so
       every event is in at most one of them. Pixels within one pixel
       radius of the FoV edge are skipped, as the windows that straddle the
       edge in ``run_blind_search_mock.py``.
    2. For every pixel, compute the normalised cumulative exposure of its
       events at the pixel centre (``ExposureModel.norm_cumul_exposure``),
       sort them and take the gaps between consecutive values.
    3. Scale the gaps of each pixel by its expected rate per unit exposure,
       ``mu / E_max``, with ``mu`` the expected count of the pixel, as for
       Lambda in ``run_blind_search_mock.py``. The gaps of all the pixels
       then go in one histogram, compared with the exponential of mean 1,
       with a KS test, and with the density ``(mu + 1 - x) / mu * exp(-x)``.
    4. The same with the arrival times instead of the exposures: the gaps
       between consecutive times of each pixel, scaled by ``n / T_obs``.

The model, ``t0`` and ``tf`` are built as in ``run_blind_search_mock.py``.

The exposures take ~40 min per dataset, so the gaps are saved to
``output/analysis/mocksamples/diagnosis/gaps_<dataset>_allsky_nside<nside>.npz`` on
the first run. Later runs read them from there and only redraw the plots;
``recompute=True`` computes them again.
"""

from pathlib import Path

import healpy as hp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as scp

from tqdm import tqdm

import spacetimecorr as stc
from spacetimecorr.io import load_active_area, load_mock_sample

# ----------------------------------------------------------------------
# Global style: load the matplotlibrc of the plots/ folder
# ----------------------------------------------------------------------
RC_FILE = Path(__file__).resolve().parents[2] / "plots" / "matplotlibrc_test"
if RC_FILE.exists():
    mpl.rc_file(RC_FILE, use_default_template=False)


def linear_exp_pdf(x: np.ndarray, mu_per_pixel: np.ndarray) -> np.ndarray:
    """Density of the scaled gaps of all the pixels for the model::

        f(gap) = (mu + 1 - rate * gap) / E_max * exp(-rate * gap),
        rate = mu / E_max

    In the scaled gap ``x = rate * gap``, with ``d gap = dx / rate``::

        f(x) = (mu + 1 - x) / mu * exp(-x)

    set to 0 for ``x > mu + 1``. The pixels are averaged, each weighted by
    its expected number of gaps, ``mu - 1``.
    """
    x = np.asarray(x, dtype=float)[..., None]
    mu = mu_per_pixel.astype(float)
    weight = np.clip(mu - 1.0, 0.0, None)
    pdf = np.clip(mu + 1.0 - x, 0.0, None) / mu * np.exp(-x)
    return np.sum(weight * pdf, axis=-1) / np.sum(weight)


def linear_exp_cdf(x: np.ndarray, mu_per_pixel: np.ndarray) -> np.ndarray:
    """CDF of :func:`linear_exp_pdf`::

        F(x) = int_0^x (mu + 1 - t) / mu * exp(-t) dt
             = 1 - exp(-x) + x * exp(-x) / mu

    up to ``x = mu + 1``, where the density reaches 0, and 1 beyond. The
    pixels are averaged with the weights of :func:`linear_exp_pdf`.
    """
    x = np.ravel(np.asarray(x, dtype=float))
    mu = mu_per_pixel.astype(float)
    weight = np.clip(mu - 1.0, 0.0, None)
    weight = weight / weight.sum()

    # In chunks of x, so that the (x, pixel) array stays small.
    out = np.empty(x.size)
    for start in range(0, x.size, 1000):
        xs = x[start:start + 1000, None]
        cdf = np.minimum(1.0 - np.exp(-xs) + xs * np.exp(-xs) / mu, 1.0)
        out[start:start + 1000] = cdf @ weight
    return out


def plot_gaps(
    scaled_gaps: np.ndarray,
    *,
    dataset: str,
    n_pixels: int,
    nside: int,
    xlabel: str,
    output_path: Path,
    mu_per_pixel: np.ndarray | None = None,
):
    """Histogram of the scaled gaps with Poisson errors, against the
    exponential of mean 1 scaled to counts per bin, with a KS test.
    With ``mu_per_pixel``, also against :func:`linear_exp_pdf`, with its
    own KS test.

    Returns the KS test against the exponential, and the one against
    :func:`linear_exp_pdf` (``None`` without ``mu_per_pixel``).
    """
    # The gaps of one pixel share its window and are not strictly
    # independent, so the p-values are approximate.
    ks_exp = scp.kstest(scaled_gaps, scp.expon.cdf)
    ks_linear = None
    if mu_per_pixel is not None:
        ks_linear = scp.kstest(scaled_gaps, lambda s: linear_exp_cdf(s, mu_per_pixel))

    bin_width = 0.25
    bins = np.arange(0.0, np.ceil(scaled_gaps.max()) + bin_width, bin_width)
    x = np.linspace(0.0, bins[-1], 400)

    counts, _ = np.histogram(scaled_gaps, bins=bins)
    bin_centres = 0.5 * (bins[1:] + bins[:-1])
    filled = counts > 0

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.hist(scaled_gaps, bins=bins, histtype="step", linewidth=1.2,
            label=dataset.replace("_", r"\_"))
    ax.errorbar(bin_centres[filled], counts[filled], yerr=np.sqrt(counts[filled]),
                fmt="none", ecolor="C0", elinewidth=0.8, capsize=1.5)
    ax.plot(x, scaled_gaps.size * bin_width * scp.expon.pdf(x),
            color="black", linestyle="--", linewidth=0.8, label="exp")
    if mu_per_pixel is not None:
        ax.plot(x, scaled_gaps.size * bin_width * linear_exp_pdf(x, mu_per_pixel),
                color="C3", linestyle="-.", linewidth=0.8,
                label=r"$\frac{\mu+1-x}{\mu}\,e^{-x}$")
    ks_text = rf"KS exp: $p = {ks_exp.pvalue:.2g}$"
    if ks_linear is not None:
        ks_text += "\n" rf"KS $\frac{{\mu+1-x}}{{\mu}}\,e^{{-x}}$: $p = {ks_linear.pvalue:.2g}$"
    ax.text(
        0.04, 0.05, ks_text,
        transform=ax.transAxes, ha="left", va="bottom", fontsize=7,
    )

    # A bin holds at least 1 gap, so the axis starts just below 1, with a
    # label at every decade.
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.5)
    ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, numticks=20))
    ax.set_xlim(0.0, bins[-1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("gaps")
    ax.set_title(rf"{n_pixels} pixels, $N_{{\rm side}} = {nside}$, {scaled_gaps.size} gaps")
    ax.legend(loc="upper right")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")

    return ks_exp, ks_linear


def compute_gaps(dataset: str, nside: int, project_root: Path) -> dict:
    """Scaled exposure gaps and scaled time gaps of all the pixels in the FoV.

    Returns a dict with ``scaled_gaps`` (scaled by ``mu / E_max``),
    ``scaled_time_gaps`` (scaled by ``n / T_obs``), ``n_per_pixel`` (events
    of every pixel with at least 2), ``mu_per_pixel`` (expected count of
    those pixels), ``n_pixels`` (pixels tested) and ``n_edge`` (pixels on
    the FoV edge, skipped).
    """

    # Pierre Auger Observatory
    latitude_pa = -35.15
    longitude_pa = -69.15
    altitude_pa = 1425
    theta_max_deg = 60.0

    input_path = (
        project_root / "inputs" / "mocksamples" / "dataset_stcwg_vert" / f"{dataset}.dat"
    )
    active_area_path = (
        project_root / "inputs" / "active_area" / "ActiveArea_SD1500_UB_6T5.npz"
    )

    # ------------------------------------------------------------------
    # Sample and model
    # ------------------------------------------------------------------
    times, ra, dec = load_mock_sample(input_path)
    n_total = ra.size
    t0, tf = times.min(), times.max()

    # Arrival times as seconds since t0, for the time gaps
    seconds = (times - t0).to_value("s")
    T_obs_s = float((tf - t0).to_value("s"))

    observatory = stc.Observatory(
        latitude=latitude_pa, longitude=longitude_pa, altitude=altitude_pa,
        area=load_active_area(active_area_path),
    )
    exposure_model = stc.ExposureModel(
        observatory=observatory, t0=t0, tf=tf, rng=np.random.default_rng(0),
        theta_max_deg=theta_max_deg,
    )

    # ------------------------------------------------------------------
    # Pixels inside the FoV, away from its edge
    # ------------------------------------------------------------------
    grid = stc.HealpixGrid(exposure_model, nside=nside)
    pixel_radius_deg = np.degrees(hp.max_pixrad(nside))
    inside_fov = (
        np.abs(grid.centres[:, 1] - latitude_pa) + pixel_radius_deg < theta_max_deg
    )
    pixels = grid.pixels[inside_fov]
    centres = grid.centres[inside_fov]

    pixel_of_event = grid.pixel_of(ra, dec)

    # ------------------------------------------------------------------
    # Gaps of every pixel: in exposure, scaled by mu / E_max, and in time,
    # scaled by n / T_obs
    # ------------------------------------------------------------------
    scaled_gaps = []
    scaled_time_gaps = []
    n_per_pixel = []
    mu_per_pixel = []
    for ipix, centre in tqdm(zip(pixels, centres), total=pixels.size, desc="Pixels"):
        mask = pixel_of_event == ipix
        n = np.count_nonzero(mask)
        if n < 2:
            continue
        n_per_pixel.append(n)

        # The exposure of the events comes first: it stores the exposure
        # over the whole run in the model's cache, which the expected count
        # and E_max then read with no second pass.
        exposure = np.sort(exposure_model.norm_cumul_exposure(times[mask], centre))
        window = stc.HealpixWindow(nside=nside, ipix=int(ipix))
        mu = window.expected_n_in_window(n_total, exposure_model)
        mu_per_pixel.append(mu)

        rate = mu / exposure_model.max_norm_cumul_exposure(centre)
        scaled_gaps.append(np.diff(exposure) * rate)

        time_rate = n / T_obs_s
        scaled_time_gaps.append(np.diff(np.sort(seconds[mask])) * time_rate)

    return {
        "scaled_gaps": np.concatenate(scaled_gaps),
        "scaled_time_gaps": np.concatenate(scaled_time_gaps),
        "n_per_pixel": np.array(n_per_pixel),
        "mu_per_pixel": np.array(mu_per_pixel),
        "n_pixels": pixels.size,
        "n_edge": int(np.count_nonzero(~inside_fov)),
    }


def main(dataset: str = "dataset_0", nside: int = 32, recompute: bool = False) -> None:
    """Histograms of the scaled exposure gaps and of the scaled time gaps
    of all the pixels in the FoV.

    The gaps are read from the saved ``.npz`` if it exists, unless
    ``recompute`` is True; otherwise they are computed and saved.
    """
    project_root = Path(__file__).resolve().parents[3]
    output_dir = project_root / "output" / "analysis" / "mocksamples" / "diagnosis"
    gaps_path = output_dir / f"gaps_{dataset}_allsky_nside{nside}.npz"

    if gaps_path.exists() and not recompute:
        with np.load(gaps_path) as f:
            result = {key: f[key] for key in f.files}
        print(f"Read the gaps from {gaps_path}")
    else:
        result = compute_gaps(dataset, nside, project_root)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(gaps_path, **result)
        print(f"Saved the gaps to {gaps_path}")

    scaled_gaps = result["scaled_gaps"]
    n_per_pixel = result["n_per_pixel"]
    n_pixels = int(result["n_pixels"])
    print(
        f"{dataset}: {n_pixels} pixels ({int(result['n_edge'])} on the FoV edge "
        f"skipped), {n_per_pixel.sum()} events, {scaled_gaps.size} gaps"
    )

    # ------------------------------------------------------------------
    # One figure per kind of gap
    # ------------------------------------------------------------------
    for name, gaps, xlabel, mu_line in (
        ("exposure", scaled_gaps, r"scaled gap, $\Delta\varepsilon \, \mu / \varepsilon_{\max}$",
         result["mu_per_pixel"]),
        ("time", result["scaled_time_gaps"], r"scaled gap, $\Delta t \, n / T_{\rm obs}$", None),
    ):
        ks_exp, ks_linear = plot_gaps(
            gaps, dataset=dataset, n_pixels=n_pixels, nside=nside, xlabel=xlabel,
            output_path=output_dir / f"gaps_{dataset}_allsky_nside{nside}_{name}.png",
            mu_per_pixel=mu_line,
        )
        line = (
            f"{name} gaps: mean scaled gap = {gaps.mean():.4f} | "
            f"KS exp: D = {ks_exp.statistic:.4f}, p = {ks_exp.pvalue:.3g}"
        )
        if ks_linear is not None:
            line += (
                f" | KS (mu+1-x)/mu exp(-x): D = {ks_linear.statistic:.4f}, "
                f"p = {ks_linear.pvalue:.3g}"
            )
        print(line)


if __name__ == "__main__":
    main(dataset="dataset_0", nside=32)
