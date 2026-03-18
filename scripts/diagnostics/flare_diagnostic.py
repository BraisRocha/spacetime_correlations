from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.time import Time
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from spacetimecorr import Flare
from spacetimecorr import SkyWindow
from spacetimecorr import ExposureModel
from spacetimecorr import RNGManager
from spacetimecorr import Observatory

def build_output_dir() -> Path:
    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    project_root = Path(__file__).resolve().parents[2]
    outdir = project_root / "output" / "diagnostics" / "flare"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def flare_summary_text(flare: Flare, window: SkyWindow, max_rows: int = 10) -> str:
    """Return a human-readable diagnostic summary."""
    lines = []
    lines.append("=" * 70)
    lines.append("FLARE DIAGNOSTIC SUMMARY")
    lines.append("=" * 70)

    lines.append(f"n_events stored          : {flare.n_events}")
    lines.append(f"spatial_profile          : {flare.spatial_profile}")
    lines.append(f"time_profile             : {flare.time_profile}")
    lines.append(f"centre [RA, Dec] deg     : {flare.centre}")
    lines.append(f"duration [s]             : {flare.duration}")
    lines.append(f"window centre [RA, Dec]  : {window.centre}")

    lengths = {
        "RA": None if flare.RA is None else len(flare.RA),
        "Dec": None if flare.Dec is None else len(flare.Dec),
        "time": None if flare.time is None else len(flare.time),
        "dir_exposure": None if flare.dir_exposure is None else len(flare.dir_exposure),
    }

    lines.append("")
    lines.append("Stored array lengths:")
    for k, v in lengths.items():
        lines.append(f"  {k:12s}: {v}")

    if flare.RA is None or flare.Dec is None or flare.time is None:
        lines.append("")
        lines.append("Flare is not fully populated.")
        return "\n".join(lines)

    inside = window.contains(flare.RA, flare.Dec)
    lines.append("")
    lines.append(f"All events inside window?: {bool(np.all(inside))}")
    lines.append(f"Events inside window     : {np.count_nonzero(inside)} / {len(inside)}")

    tmin = flare.time.min()
    tmax = flare.time.max()
    dt_sec = (tmax - tmin).to_value(u.s) if len(flare.time) > 0 else 0.0
    in_obs = bool(np.all((flare.time >= flare.t0) & (flare.time <= flare.tf)))

    lines.append("")
    lines.append("Time diagnostics:")
    lines.append(f"  earliest event         : {tmin.isot}")
    lines.append(f"  latest event           : {tmax.isot}")
    lines.append(f"  span [s]               : {dt_sec:.3f}")
    lines.append(f"  flare duration [s]     : {flare.duration:.3f}")
    lines.append(f"  inside [t0, tf]?       : {in_obs}")

    if flare.dir_exposure is not None and len(flare.dir_exposure) > 0:
        lines.append("")
        lines.append("Directional exposure diagnostics:")
        lines.append(f"  min                    : {np.min(flare.dir_exposure):.6g}")
        lines.append(f"  max                    : {np.max(flare.dir_exposure):.6g}")
        lines.append(f"  mean                   : {np.mean(flare.dir_exposure):.6g}")

    nshow = min(max_rows, len(flare.RA))
    lines.append("")
    lines.append(f"First {nshow} events:")
    lines.append(" idx |      RA [deg] |     Dec [deg] | time | dir_exposure")
    lines.append("-" * 90)

    for i in range(nshow):
        exp_i = None if flare.dir_exposure is None else flare.dir_exposure[i]
        exp_txt = "None" if exp_i is None else f"{exp_i:.6g}"
        lines.append(
            f"{i:4d} | "
            f"{flare.RA[i]:13.6f} | "
            f"{flare.Dec[i]:13.6f} | "
            f"{flare.time[i].isot} | "
            f"{exp_txt}"
        )

    return "\n".join(lines)


def save_flare_arrays(flare: Flare, outdir: Path, stem: str = "flare") -> Path:
    """Save flare arrays to a compressed NumPy file."""
    path = outdir / f"{stem}_arrays.npz"

    np.savez_compressed(
        path,
        RA=np.array([]) if flare.RA is None else flare.RA,
        Dec=np.array([]) if flare.Dec is None else flare.Dec,
        time_isot=np.array([]) if flare.time is None else np.array(flare.time.isot),
        time_jd=np.array([]) if flare.time is None else flare.time.jd,
        dir_exposure=np.array([]) if flare.dir_exposure is None else flare.dir_exposure,
        centre=flare.centre,
        n_events=flare.n_events,
        duration_sec=flare.duration,
        spatial_profile="" if flare.spatial_profile is None else flare.spatial_profile,
        time_profile="" if flare.time_profile is None else flare.time_profile,
    )
    return path


def save_flare_plots(flare: Flare, window: SkyWindow, outdir: Path, stem: str = "flare") -> list[Path]:
    """Save quick-look diagnostic plots."""
    if flare.RA is None or flare.Dec is None or flare.time is None:
        raise ValueError("Flare must be generated before plotting.")

    saved = []

    # Sky positions
    ra = np.asarray(flare.RA)
    dec = np.asarray(flare.Dec)
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
    p = outdir / f"sky_heatmap.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    # Time histogram
    plt.figure(figsize=(6, 4))
    t0 = flare.time.min()
    offsets = (flare.time - t0).to_value(u.h)
    plt.hist(offsets, bins="fd", alpha=0.8, edgecolor="black", linewidth=0.8)
    plt.xlabel("Time offset from first accepted event [h]")
    plt.ylabel("Counts")
    plt.title("Accepted event times")
    plt.tight_layout()
    p = outdir / f"time_hist.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    saved.append(p)

    # Exposure histogram
    if flare.dir_exposure is not None and len(flare.dir_exposure) > 0:
        plt.figure(figsize=(6, 4))
        plt.hist(flare.dir_exposure, bins="fd", alpha=0.8, edgecolor="black", linewidth=0.8)
        plt.xlabel("Directional exposure")
        plt.ylabel("Counts")
        plt.title("Accepted event exposure")
        plt.tight_layout()
        p = outdir / f"dir_exposure_hist.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        saved.append(p)

        # Exposure vs time
        plt.figure(figsize=(6, 4))
        x = np.asarray(offsets)
        y = np.asarray(flare.dir_exposure)
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)
        idx = np.argsort(z)
        x, y, z = x[idx], y[idx], z[idx]
        sc = plt.scatter(x, y, c=z, s=20, alpha=0.8)
        plt.colorbar(sc, label="Point density")
        plt.xlabel("Time offset from first accepted event [s]")
        plt.ylabel("Directional exposure")
        plt.title("Accepted events exposure vs time")
        plt.tight_layout()
        p = outdir / f"exposure_vs_time.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        saved.append(p)

    return saved


def run_flare_diagnostic(
    flare: Flare,
    window: SkyWindow,
    sigma: float,
    efficiency=None,
    max_rows: int = 10,
    stem: str = "flare",
) -> None:
    """
    Generate one flare realization, print diagnostics, and save outputs.
    """
    outdir = build_output_dir()

    flare.generate_in_window(window=window, sigma=sigma, efficiency=efficiency)

    summary = flare_summary_text(flare, window=window, max_rows=max_rows)
    print(summary)

    summary_path = outdir / f"{stem}_summary.txt"
    summary_path.write_text(summary, encoding="utf-8")

    arrays_path = save_flare_arrays(flare, outdir=outdir, stem=stem)
    plot_paths = save_flare_plots(flare, window=window, outdir=outdir, stem=stem)

    print("\nSaved diagnostic files:")
    print(f"  Summary : {summary_path}")
    print(f"  Arrays  : {arrays_path}")
    for p in plot_paths:
        print(f"  Plot    : {p}")


# ------------------------------------------------------------------
# EXAMPLE USAGE
# ------------------------------------------------------------------

if __name__ == "__main__":

    rng_manager = RNGManager(seed=42)
    rng_flare = rng_manager.get("flare")
    rng_exposure = rng_manager.get("exposure")

    # Observation interval
    t0 = Time("2025-01-01T00:00:00")
    tf = Time("2025-01-14T00:00:00")

    # Flare parameters
    duration = 3 * u.day               # 1 day
    n_events = 100000
    sigma = 1.5                        # deg

    # Sky window parameters (RA [deg], Dec [deg], radius [deg])
    centre = np.array([30.0, 0.0])
    radius = 2.0

    # Pierre Auger Observatory coordinates
    latitude_pa = -35.15
    longitude_pa = -69.15
    altitude_pa = 1425

    window = SkyWindow(
        centre=centre,
        radius=radius,
    )

    observatory = Observatory(
        latitude=latitude_pa,
        longitude=longitude_pa,
        altitude=altitude_pa,
    )
    exposure_model = ExposureModel(
        observatory=observatory,
        t0=t0,
        tf=tf,
        rng=rng_exposure,
    )

    flare = Flare(
        n_events=n_events,
        duration=duration,
        t0=t0,
        tf=tf,
        centre=centre,
        exposure=exposure_model,
        rng=rng_flare,
    )

    run_flare_diagnostic(
        flare=flare,
        window=window,
        sigma=sigma,
        efficiency=None,
        max_rows=12,
    )