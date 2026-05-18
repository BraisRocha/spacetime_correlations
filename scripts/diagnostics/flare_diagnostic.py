"""
Flare visual / inspection diagnostic.

Generates one ``Flare`` realisation inside a sky window, prints a text
summary, saves the underlying arrays to ``.npz`` and produces a small
collection of quick-look plots (sky heatmap, time histogram, exposure
distribution).

This script does NOT assert correctness — it is intended for human
inspection.  Pass/fail checks of the Flare contract live in
``tests/test_flare.py``.
"""

from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.time import Time
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from spacetimecorr import (
    ExposureModel,
    Flare,
    Observatory,
    RNGManager,
    SkyWindow,
)


def build_output_dir() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    outdir = project_root / "output" / "diagnostics" / "flare"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


# -------------------------------------------------------------------------
# Text summary
# -------------------------------------------------------------------------


def flare_summary_text(flare: Flare, window: SkyWindow, max_rows: int = 10) -> str:
    """Return a human-readable diagnostic summary."""
    lines = []
    lines.append("=" * 70)
    lines.append("FLARE DIAGNOSTIC SUMMARY")
    lines.append("=" * 70)

    lines.append(f"n_flare                  : {flare.n_flare}")
    lines.append(f"spatial_profile          : {flare.spatial_profile}")
    lines.append(f"time_profile             : {flare.time_profile}")
    lines.append(f"flare_type               : {flare.flare_type}")
    lines.append(f"centre [RA, Dec] deg     : {flare.centre}")
    lines.append(f"duration [s]             : {flare.duration}")
    lines.append(f"window centre [RA, Dec]  : {window.centre}")
    lines.append(f"window radius [deg]      : {window.radius}")

    lengths = {
        "ra":       None if flare.ra       is None else len(flare.ra),
        "dec":      None if flare.dec      is None else len(flare.dec),
        "time":     None if flare.time     is None else len(flare.time),
        "exposure": None if flare.exposure is None else len(flare.exposure),
    }

    lines.append("")
    lines.append("Stored array lengths:")
    for k, v in lengths.items():
        lines.append(f"  {k:12s}: {v}")

    if flare.ra is None or flare.dec is None or flare.time is None:
        lines.append("")
        lines.append("Flare is not fully populated.")
        return "\n".join(lines)

    inside = window.contains(flare.ra, flare.dec)
    lines.append("")
    lines.append(f"All events inside window?: {bool(np.all(inside))}")
    lines.append(f"Events inside window     : {int(np.count_nonzero(inside))} / {len(inside)}")

    tmin = flare.time.min()
    tmax = flare.time.max()
    dt_sec = (tmax - tmin).to_value(u.s)
    in_obs = bool(np.all((flare.time >= flare.t0) & (flare.time <= flare.tf)))

    lines.append("")
    lines.append("Time diagnostics:")
    lines.append(f"  earliest event         : {tmin.isot}")
    lines.append(f"  latest event           : {tmax.isot}")
    lines.append(f"  span [s]               : {dt_sec:.3f}")
    lines.append(f"  flare duration [s]     : {flare.duration:.3f}")
    lines.append(f"  inside [t0, tf]?       : {in_obs}")

    if flare.exposure is not None and len(flare.exposure) > 0:
        lines.append("")
        lines.append("Directional exposure diagnostics:")
        lines.append(f"  min                    : {np.min(flare.exposure):.6g}")
        lines.append(f"  max                    : {np.max(flare.exposure):.6g}")
        lines.append(f"  mean                   : {np.mean(flare.exposure):.6g}")

    nshow = min(max_rows, len(flare.ra))
    lines.append("")
    lines.append(f"First {nshow} events:")
    lines.append(" idx |      RA [deg] |     Dec [deg] | time | exposure")
    lines.append("-" * 90)

    for i in range(nshow):
        exp_i = None if flare.exposure is None else float(flare.exposure[i])
        exp_txt = "None" if exp_i is None else f"{exp_i:.6g}"
        lines.append(
            f"{i:4d} | "
            f"{flare.ra[i]:13.6f} | "
            f"{flare.dec[i]:13.6f} | "
            f"{flare.time[i].isot} | "
            f"{exp_txt}"
        )

    return "\n".join(lines)


# -------------------------------------------------------------------------
# Save arrays
# -------------------------------------------------------------------------


def save_flare_arrays(flare: Flare, outdir: Path, stem: str = "flare") -> Path:
    path = outdir / f"{stem}_arrays.npz"

    np.savez_compressed(
        path,
        ra=np.array([]) if flare.ra is None else flare.ra,
        dec=np.array([]) if flare.dec is None else flare.dec,
        time_isot=np.array([]) if flare.time is None else np.array(flare.time.isot),
        time_jd=np.array([]) if flare.time is None else flare.time.jd,
        exposure=np.array([]) if flare.exposure is None else flare.exposure,
        centre=flare.centre,
        n_flare=flare.n_flare,
        duration_sec=flare.duration,
        spatial_profile="" if flare.spatial_profile is None else flare.spatial_profile,
        time_profile="" if flare.time_profile is None else flare.time_profile,
    )
    return path


# -------------------------------------------------------------------------
# Plots
# -------------------------------------------------------------------------


def save_flare_plots(flare: Flare, window: SkyWindow, outdir: Path) -> list[Path]:
    if flare.ra is None or flare.dec is None or flare.time is None:
        raise ValueError("Flare must be generated before plotting.")

    saved = []

    # Sky positions (KDE heatmap)
    ra = np.asarray(flare.ra)
    dec = np.asarray(flare.dec)
    fig, ax = plt.subplots(figsize=(6, 5))
    xmin, xmax = ra.min(), ra.max()
    ymin, ymax = dec.min(), dec.max()
    dx = xmax - xmin
    dy = ymax - ymin
    xmin -= 0.1 * dx if dx > 0 else 0.1
    xmax += 0.1 * dx if dx > 0 else 0.1
    ymin -= 0.1 * dy if dy > 0 else 0.1
    ymax += 0.1 * dy if dy > 0 else 0.1
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 150), np.linspace(ymin, ymax, 150))
    values = np.vstack([ra, dec])
    kde = gaussian_kde(values)
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    im = ax.pcolormesh(xx, yy, zz, shading="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Estimated density")
    ax.contour(xx, yy, zz, levels=6, linewidths=1.0)
    ax.scatter([flare.centre[0]], [flare.centre[1]], marker="x", s=100, linewidths=2, label="flare centre")
    ax.scatter([window.centre[0]], [window.centre[1]], marker="+", s=120, linewidths=2, label="window centre")
    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("Dec [deg]")
    ax.set_title("Accepted flare events on sky")
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    fig.tight_layout()
    p = outdir / "sky_heatmap.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    # Time histogram
    plt.figure(figsize=(6, 4))
    t_ref = flare.time.min()
    offsets = (flare.time - t_ref).to_value(u.h)
    plt.hist(offsets, bins="fd", alpha=0.8, edgecolor="black", linewidth=0.8)
    plt.xlabel("Time offset from first accepted event [h]")
    plt.ylabel("Counts")
    plt.title("Accepted event times")
    plt.tight_layout()
    p = outdir / "time_hist.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    saved.append(p)

    # Exposure histogram
    if flare.exposure is not None and len(flare.exposure) > 0:
        plt.figure(figsize=(6, 4))
        plt.hist(flare.exposure, bins="fd", alpha=0.8, edgecolor="black", linewidth=0.8)
        plt.xlabel("Directional exposure")
        plt.ylabel("Counts")
        plt.title("Accepted event exposure")
        plt.tight_layout()
        p = outdir / "exposure_hist.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        saved.append(p)

        # Exposure vs time (point density)
        plt.figure(figsize=(6, 4))
        x = np.asarray(offsets)
        y = np.asarray(flare.exposure)
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)
        idx = np.argsort(z)
        x, y, z = x[idx], y[idx], z[idx]
        sc = plt.scatter(x, y, c=z, s=20, alpha=0.8)
        plt.colorbar(sc, label="Point density")
        plt.xlabel("Time offset from first accepted event [h]")
        plt.ylabel("Directional exposure")
        plt.title("Accepted events exposure vs time")
        plt.tight_layout()
        p = outdir / "exposure_vs_time.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        saved.append(p)

    return saved


# -------------------------------------------------------------------------
# Runner
# -------------------------------------------------------------------------


def run_flare_diagnostic(
    flare: Flare,
    window: SkyWindow,
    sigma: float,
    efficiency=None,
    max_rows: int = 10,
    stem: str = "flare",
) -> None:
    outdir = build_output_dir()

    flare.generate_in_window(window=window, sigma=sigma, efficiency=efficiency)

    summary = flare_summary_text(flare, window=window, max_rows=max_rows)
    print(summary)

    summary_path = outdir / f"{stem}_summary.txt"
    summary_path.write_text(summary, encoding="utf-8")

    arrays_path = save_flare_arrays(flare, outdir=outdir, stem=stem)
    plot_paths = save_flare_plots(flare, window=window, outdir=outdir)

    print("\nSaved diagnostic files:")
    print(f"  Summary : {summary_path}")
    print(f"  Arrays  : {arrays_path}")
    for p in plot_paths:
        print(f"  Plot    : {p}")


# -------------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------------


if __name__ == "__main__":

    rng_manager = RNGManager(seed=42)
    rng_flare = rng_manager.get("flare")
    rng_exposure = rng_manager.get("exposure")

    t0 = Time("2025-01-01T00:00:00")
    tf = Time("2025-01-14T00:00:00")

    duration = 3 * u.day
    n_flare = 10000
    sigma = 1.5  # deg

    centre = np.array([30.0, 0.0])
    radius = 2.0

    # Pierre Auger Observatory
    observatory = Observatory(latitude=-35.15, longitude=-69.15, altitude=1425.0)
    window = SkyWindow(centre=centre, radius=radius)
    exposure_model = ExposureModel(
        observatory=observatory, t0=t0, tf=tf, rng=rng_exposure,
    )

    flare = Flare(
        n_flare=n_flare,
        duration=duration,
        t0=t0,
        tf=tf,
        centre=centre,
        exposure_model=exposure_model,
        rng=rng_flare,
    )

    run_flare_diagnostic(
        flare=flare,
        window=window,
        sigma=sigma,
        efficiency=None,
        max_rows=12,
    )
