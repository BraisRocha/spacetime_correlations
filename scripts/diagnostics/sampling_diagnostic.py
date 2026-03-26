from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from spacetimecorr import EventSample
from spacetimecorr import RNGManager
from spacetimecorr import SkyWindow
from spacetimecorr import ExposureModel
from spacetimecorr import Observatory


# ------------------------------------------------------------------
# Output helpers
# ------------------------------------------------------------------

def build_output_dir() -> Path:
    """Create the base output directory for EventSample diagnostics."""
    project_root = Path(__file__).resolve().parents[2]
    outdir = project_root / "output" / "diagnostics" / "eventsample"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def build_case_output_dir(case_name: str) -> Path:
    """Create a subdirectory for one diagnostic case."""
    outdir = build_output_dir() / case_name
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


# ------------------------------------------------------------------
# Text summary
# ------------------------------------------------------------------

def event_sample_summary_text(
    sample: EventSample,
    *,
    label: str = "EVENT SAMPLE",
    max_rows: int = 10,
) -> str:
    """Return a human-readable diagnostic summary."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"{label} DIAGNOSTIC SUMMARY")
    lines.append("=" * 70)

    lines.append(f"n_events stored          : {sample.n_events}")
    lines.append(f"spatial_type             : {sample.spatial_type}")
    lines.append(f"t0                       : {sample.t0.isot}")
    lines.append(f"tf                       : {sample.tf.isot}")
    lines.append(f"T_obs [s]                : {sample.T_obs.to_value(u.s):.6f}")
    lines.append(f"exp_rate_time [1/s]      : {sample.exp_rate_time:.6g}")
    lines.append(f"is_populated             : {sample.is_populated}")
    lines.append(f"has_exposure             : {sample.has_exposure}")
    lines.append(f"has_flare                : {sample.has_flare}")

    lengths = {
        "RA": None if sample.RA is None else len(sample.RA),
        "Dec": None if sample.Dec is None else len(sample.Dec),
        "dir_exposure": None if sample.dir_exposure is None else len(sample.dir_exposure),
    }

    lines.append("")
    lines.append("Stored array lengths:")
    for k, v in lengths.items():
        lines.append(f"  {k:12s}: {v}")

    if sample.RA is None or sample.Dec is None:
        lines.append("")
        lines.append("Sample is not populated.")
        return "\n".join(lines)

    ra = np.asarray(sample.RA)
    dec = np.asarray(sample.Dec)

    lines.append("")
    lines.append("Coordinate diagnostics:")
    lines.append(f"  finite RA?             : {bool(np.all(np.isfinite(ra)))}")
    lines.append(f"  finite Dec?            : {bool(np.all(np.isfinite(dec)))}")
    lines.append(f"  RA min [deg]           : {np.min(ra):.6f}")
    lines.append(f"  RA max [deg]           : {np.max(ra):.6f}")
    lines.append(f"  Dec min [deg]          : {np.min(dec):.6f}")
    lines.append(f"  Dec max [deg]          : {np.max(dec):.6f}")
    lines.append(f"  RA in [0, 360)?        : {bool(np.all((ra >= 0.0) & (ra < 360.0)))}")
    lines.append(f"  Dec in [-90, 90]?      : {bool(np.all((dec >= -90.0) & (dec <= 90.0)))}")

    sin_dec = np.sin(np.deg2rad(dec))
    lines.append("")
    lines.append("Isotropy quick-check diagnostics:")
    lines.append(f"  mean(sin Dec)          : {np.mean(sin_dec):.6g}")
    lines.append(f"  std(sin Dec)           : {np.std(sin_dec):.6g}")
    lines.append(f"  mean(RA) [deg]         : {np.mean(ra):.6g}")

    if sample.expected_counts is not None:
        lines.append("")
        lines.append("Window / counts diagnostics:")
        lines.append(f"  expected_counts        : {sample.expected_counts:.6g}")

    if sample.dir_exposure is not None and len(sample.dir_exposure) > 0:
        eps = np.asarray(sample.dir_exposure)
        lines.append("")
        lines.append("Directional exposure diagnostics:")
        lines.append(f"  method                 : {sample.dir_exposure_method}")
        lines.append(f"  finite?                : {bool(np.all(np.isfinite(eps)))}")
        lines.append(f"  min                    : {np.min(eps):.6g}")
        lines.append(f"  max                    : {np.max(eps):.6g}")
        lines.append(f"  mean                   : {np.mean(eps):.6g}")

        if sample.exp_rate_exposure is not None:
            lines.append(f"  exp_rate_exposure      : {sample.exp_rate_exposure:.6g}")

    if sample.has_flare:
        lines.append("")
        lines.append("Flare diagnostics:")
        lines.append(f"  flare_type             : {getattr(sample, 'flare_type', None)}")
        flare_indices = getattr(sample, "flare_indices", None)
        lines.append(
            f"  number injected        : "
            f"{None if flare_indices is None else len(flare_indices)}"
        )

    nshow = min(max_rows, len(ra))
    lines.append("")
    lines.append(f"First {nshow} events:")
    lines.append(" idx |      RA [deg] |     Dec [deg] | dir_exposure")
    lines.append("-" * 70)

    for i in range(nshow):
        exp_i = None if sample.dir_exposure is None else sample.dir_exposure[i]
        exp_txt = "None" if exp_i is None else f"{exp_i:.6g}"
        lines.append(
            f"{i:4d} | "
            f"{sample.RA[i]:13.6f} | "
            f"{sample.Dec[i]:13.6f} | "
            f"{exp_txt}"
        )

    return "\n".join(lines)


# ------------------------------------------------------------------
# Save arrays
# ------------------------------------------------------------------

def save_event_sample_arrays(
    sample: EventSample,
    outdir: Path,
    stem: str,
) -> Path:
    """Save sample arrays to a compressed NumPy file."""
    path = outdir / f"{stem}_arrays.npz"

    np.savez_compressed(
        path,
        RA=np.array([]) if sample.RA is None else sample.RA,
        Dec=np.array([]) if sample.Dec is None else sample.Dec,
        dir_exposure=np.array([]) if sample.dir_exposure is None else sample.dir_exposure,
        n_events=sample.n_events,
        t0_isot=sample.t0.isot,
        tf_isot=sample.tf.isot,
        T_obs_s=sample.T_obs.to_value(u.s),
        exp_rate_time=sample.exp_rate_time,
        spatial_type="" if sample.spatial_type is None else sample.spatial_type,
        expected_counts=np.nan if sample.expected_counts is None else sample.expected_counts,
        exp_rate_exposure=np.nan if sample.exp_rate_exposure is None else sample.exp_rate_exposure,
        dir_exposure_method=""
        if sample.dir_exposure_method is None
        else sample.dir_exposure_method,
    )
    return path


# ------------------------------------------------------------------
# Plot helpers
# ------------------------------------------------------------------

def save_coordinate_plots(
    sample: EventSample,
    outdir: Path,
) -> list[Path]:
    """
    Save coordinate diagnostic plots.
    Intended especially for the full sample, but works for any populated sample.
    """
    if sample.RA is None or sample.Dec is None:
        raise ValueError("Sample must be populated before plotting.")

    saved = []
    ra = np.asarray(sample.RA)
    dec = np.asarray(sample.Dec)

    plt.figure(figsize=(6, 4))
    plt.hist(ra, bins="fd", alpha=0.8, edgecolor="black", linewidth=0.8)
    plt.xlabel("RA [deg]")
    plt.ylabel("Counts")
    plt.title("Right ascension distribution")
    plt.tight_layout()
    p = outdir / f"ra_hist.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    saved.append(p)

    plt.figure(figsize=(6, 4))
    plt.hist(dec, bins="fd", alpha=0.8, edgecolor="black", linewidth=0.8)
    plt.xlabel("Dec [deg]")
    plt.ylabel("Counts")
    plt.title("Declination distribution")
    plt.tight_layout()
    p = outdir / f"dec_hist.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    saved.append(p)

    sin_dec = np.sin(np.deg2rad(dec))
    plt.figure(figsize=(6, 4))
    plt.hist(sin_dec, bins="fd", alpha=0.8, edgecolor="black", linewidth=0.8)
    plt.xlabel(r"$\sin(\mathrm{Dec})$")
    plt.ylabel("Counts")
    plt.title("sin(Dec) distribution")
    plt.tight_layout()
    p = outdir / f"sin_dec_hist.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    saved.append(p)

    x = np.asarray(ra)
    y = np.asarray(dec)
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    idx = np.argsort(z)
    x, y, z = x[idx], y[idx], z[idx]
    plt.figure(figsize=(6.5, 5))
    sc = plt.scatter(x, y, c=z, s=8, alpha=0.8)
    plt.colorbar(sc, label="Point density")
    plt.xlabel("RA [deg]")
    plt.ylabel("Dec [deg]")
    plt.title("Sky scatter")
    plt.tight_layout()
    p = outdir / f"sky_scatter.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    saved.append(p)

    return saved


def save_exposure_plots(
    sample: EventSample,
    outdir: Path,
) -> list[Path]:
    """
    Save exposure-related plots.
    Intended for subsamples where directional exposure has been attached.
    """
    saved = []

    if sample.dir_exposure is None or len(sample.dir_exposure) == 0:
        return saved

    eps = np.asarray(sample.dir_exposure)
    ra = None if sample.RA is None else np.asarray(sample.RA)

    plt.figure(figsize=(6, 4))
    plt.hist(eps, bins="fd", alpha=0.8, edgecolor="black", linewidth=0.8)
    plt.xlabel("Directional exposure")
    plt.ylabel("Counts")
    plt.title("Directional exposure distribution")
    plt.tight_layout()
    p = outdir / f"dir_exposure_hist.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    saved.append(p)

    return saved


def save_skymap_plot(
    sample: EventSample,
    outdir: Path,
    *,
    nside: int = 32,
    mask_fov: bool = False,
    location=None,
    zenith_max=None,
    title: str = "Event sample skymap",
) -> Path:
    """Save the HEALPix skymap using the class plotting method."""
    fig, ax = sample.plot_skymap(
        nside=nside,
        mask_fov=mask_fov,
        location=location,
        zenith_max=zenith_max,
        title=title,
        output_file=None,
        show=False,
    )
    path = outdir / f"skymap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ------------------------------------------------------------------
# Main diagnostic runner
# ------------------------------------------------------------------

def run_event_sample_diagnostic(
    sample: EventSample,
    *,
    case_name: str,
    label: str,
    stem: str,
    max_rows: int = 10,
    save_coordinates: bool = True,
    save_exposure: bool = True,
    save_skymap: bool = True,
    nside: int = 32,
    mask_fov: bool = False,
    location=None,
    zenith_max=None,
) -> None:
    """
    Print diagnostics for one EventSample and save summary, arrays, and plots.

    Parameters
    ----------
    sample : EventSample
        Sample to diagnose.
    case_name : str
        Name of the output subdirectory.
    label : str
        Label used in the printed/saved summary title.
    stem : str
        Prefix for saved files.
    """
    outdir = build_case_output_dir(case_name)

    summary = event_sample_summary_text(
        sample,
        label=label,
        max_rows=max_rows,
    )
    print(summary)

    summary_path = outdir / f"{stem}_summary.txt"
    summary_path.write_text(summary, encoding="utf-8")

    arrays_path = save_event_sample_arrays(
        sample,
        outdir=outdir,
        stem=stem,
    )

    plot_paths = []

    if save_coordinates:
        plot_paths.extend(
            save_coordinate_plots(
                sample,
                outdir=outdir,
            )
        )

    if save_exposure:
        plot_paths.extend(
            save_exposure_plots(
                sample,
                outdir=outdir,
            )
        )

    if save_skymap:
        p = save_skymap_plot(
            sample,
            outdir=outdir,
            nside=nside,
            mask_fov=mask_fov,
            location=location,
            zenith_max=zenith_max,
            title=f"{label.title()} skymap",
        )
        plot_paths.append(p)

    print("\nSaved diagnostic files:")
    print(f"  Summary : {summary_path}")
    print(f"  Arrays  : {arrays_path}")
    for p in plot_paths:
        print(f"  Plot    : {p}")


# ------------------------------------------------------------------
# Example usage
# ------------------------------------------------------------------

if __name__ == "__main__":

    rng_manager = RNGManager(seed=42)
    rng_sample = rng_manager.get("sample")
    rng_exposure = rng_manager.get("exposure")

    # Observation interval
    t0 = Time("2025-01-01T00:00:00")
    tf = Time("2025-01-14T00:00:00")

    # --------------------------------------------------------------
    # Full sample
    # --------------------------------------------------------------
    n_events = 100000
    sample = EventSample(
        n_events=n_events,
        t0=t0,
        tf=tf,
        rng=rng_sample,
    )

    run_event_sample_diagnostic(
        sample,
        case_name="full_sample",
        label="FULL SAMPLE",
        stem="full_sample",
        max_rows=12,
        save_coordinates=True,
        save_exposure=False,
        save_skymap=True,
        nside=32,
        mask_fov=False,
    )

    # --------------------------------------------------------------
    # Subsample + exposure
    # --------------------------------------------------------------
    centre = np.array([30.0, 0.0])   # deg
    radius = 20.0                    # deg

    window = SkyWindow(
        centre=centre,
        radius=radius,
    )

    subsample = sample.select_subsample(window)

    latitude_pa = -35.15
    longitude_pa = -69.15
    altitude_pa = 1425

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

    subsample.add_directional_exposure(
        window=window,
        exposure_model=exposure_model,
    )

    # Use this if Observatory exposes an EarthLocation:
    try:
        location = observatory.location
    except AttributeError:
        location = EarthLocation(
            lat=latitude_pa * u.deg,
            lon=longitude_pa * u.deg,
            height=altitude_pa * u.m,
        )

    zenith_max = 60 * u.deg

    run_event_sample_diagnostic(
        subsample,
        case_name="subsample",
        label="SUBSAMPLE",
        stem="subsample",
        max_rows=12,
        save_coordinates=True,
        save_exposure=True,
        save_skymap=True,
        nside=32,
        mask_fov=False,
        location=location,
        zenith_max=zenith_max,
    )