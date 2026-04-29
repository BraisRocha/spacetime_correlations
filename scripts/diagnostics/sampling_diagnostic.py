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
from spacetimecorr import Flare


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
# Text summaries
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

    lines.append(f"n_events stored              : {sample.n_events}")
    lines.append(f"expected_n                   : {sample.expected_n}")
    lines.append(f"spatial_type                 : {sample.spatial_type}")
    lines.append(f"exposure_type                : {sample.exposure_type}")
    lines.append(f"flare_type                   : {sample.flare_type}")
    lines.append(f"t0                           : {sample.t0.isot}")
    lines.append(f"tf                           : {sample.tf.isot}")
    lines.append(f"T_obs [s]                    : {sample.T_obs.to_value(u.s):.6f}")
    lines.append(f"expected_temporal_rate [1/s] : {sample.expected_temporal_rate:.6g}")
    lines.append(f"has_coordinates              : {sample.has_coordinates}")
    lines.append(f"has_exposure                 : {sample.has_exposure}")
    lines.append(f"has_flare                    : {sample.has_flare}")

    lengths = {
        "RA": None if sample.RA is None else len(sample.RA),
        "Dec": None if sample.Dec is None else len(sample.Dec),
        "exposure": None if sample.exposure is None else len(sample.exposure),
        "flare_mask": None if sample.flare_mask is None else len(sample.flare_mask),
    }

    lines.append("")
    lines.append("Stored array lengths:")
    for k, v in lengths.items():
        lines.append(f"  {k:12s}: {v}")

    if not sample.has_coordinates:
        lines.append("")
        lines.append("Sample is not populated.")
        return "\n".join(lines)

    ra = np.asarray(sample.RA, dtype=float)
    dec = np.asarray(sample.Dec, dtype=float)

    lines.append("")
    lines.append("Coordinate diagnostics:")
    lines.append(f"  finite RA?                 : {bool(np.all(np.isfinite(ra)))}")
    lines.append(f"  finite Dec?                : {bool(np.all(np.isfinite(dec)))}")
    lines.append(f"  RA min [deg]               : {np.min(ra):.6f}")
    lines.append(f"  RA max [deg]               : {np.max(ra):.6f}")
    lines.append(f"  Dec min [deg]              : {np.min(dec):.6f}")
    lines.append(f"  Dec max [deg]              : {np.max(dec):.6f}")
    lines.append(f"  RA in [0, 360)?            : {bool(np.all((ra >= 0.0) & (ra < 360.0)))}")
    lines.append(f"  Dec in [-90, 90]?          : {bool(np.all((dec >= -90.0) & (dec <= 90.0)))}")

    sin_dec = np.sin(np.deg2rad(dec))
    lines.append("")
    lines.append("Isotropy quick-check diagnostics:")
    lines.append(f"  mean(sin Dec)              : {np.mean(sin_dec):.6g}")
    lines.append(f"  std(sin Dec)               : {np.std(sin_dec):.6g}")
    lines.append(f"  mean(RA) [deg]             : {np.mean(ra):.6g}")

    lines.append("")
    lines.append("Count / expectation diagnostics:")
    lines.append(f"  expected_n                 : {sample.expected_n:.6g}")

    if sample.exposure is not None and len(sample.exposure) > 0:
        eps = np.asarray(sample.exposure, dtype=float)
        finite_eps = np.isfinite(eps)

        lines.append("")
        lines.append("Exposure diagnostics:")
        lines.append(f"  exposure_type              : {sample.exposure_type}")
        lines.append(f"  finite entries             : {int(np.count_nonzero(finite_eps))}")
        lines.append(f"  non-finite entries         : {int(np.count_nonzero(~finite_eps))}")

        if np.any(finite_eps):
            eps_fin = eps[finite_eps]
            lines.append(f"  min                        : {np.min(eps_fin):.6g}")
            lines.append(f"  max                        : {np.max(eps_fin):.6g}")
            lines.append(f"  mean                       : {np.mean(eps_fin):.6g}")

        lines.append(f"  expected_exposure_rate     : {sample.expected_exposure_rate}")

    if sample.flare_mask is not None:
        flare_mask = np.asarray(sample.flare_mask, dtype=bool)
        lines.append("")
        lines.append("Flare diagnostics:")
        lines.append(f"  flare_type                 : {sample.flare_type}")
        lines.append(f"  flare events flagged       : {int(np.count_nonzero(flare_mask))}")

    nshow = min(max_rows, len(ra))
    lines.append("")
    lines.append(f"First {nshow} events:")
    lines.append(" idx |      RA [deg] |     Dec [deg] |     exposure | flare")
    lines.append("-" * 78)

    for i in range(nshow):
        exp_i = None if sample.exposure is None else sample.exposure[i]
        flare_i = None if sample.flare_mask is None else bool(sample.flare_mask[i])

        if exp_i is None or not np.isfinite(exp_i):
            exp_txt = "None"
        else:
            exp_txt = f"{exp_i:.6g}"

        flare_txt = "None" if flare_i is None else str(flare_i)

        lines.append(
            f"{i:4d} | "
            f"{sample.RA[i]:13.6f} | "
            f"{sample.Dec[i]:13.6f} | "
            f"{exp_txt:12s} | "
            f"{flare_txt}"
        )

    return "\n".join(lines)


def flare_injection_summary_text(
    sample_before: EventSample,
    sample_after: EventSample,
    *,
    label: str = "FLARE INJECTION",
    max_rows: int = 10,
) -> str:
    """Return a human-readable summary for a flare injection test."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"{label} SUMMARY")
    lines.append("=" * 70)

    lines.append(f"n_events before                : {sample_before.n_events}")
    lines.append(f"n_events after                 : {sample_after.n_events}")
    lines.append(f"same sample size               : {sample_before.n_events == sample_after.n_events}")

    lines.append("")
    lines.append("Flare bookkeeping:")
    lines.append(f"  has_flare after injection    : {sample_after.has_flare}")
    lines.append(f"  flare_type                   : {sample_after.flare_type}")

    if sample_after.flare_mask is not None:
        n_flare = int(np.count_nonzero(sample_after.flare_mask))
        lines.append(f"  number flagged as flare      : {n_flare}")
    else:
        lines.append("  number flagged as flare      : None")

    if sample_before.RA is not None and sample_after.RA is not None:
        changed_ra = np.count_nonzero(sample_before.RA != sample_after.RA)
        changed_dec = np.count_nonzero(sample_before.Dec != sample_after.Dec)
        lines.append("")
        lines.append("Coordinate replacement check:")
        lines.append(f"  changed RA entries           : {changed_ra}")
        lines.append(f"  changed Dec entries          : {changed_dec}")

    if sample_before.exposure is not None and sample_after.exposure is not None:
        before = np.asarray(sample_before.exposure, dtype=float)
        after = np.asarray(sample_after.exposure, dtype=float)

        changed_exp = np.count_nonzero(~np.isclose(before, after, equal_nan=True))

        lines.append("")
        lines.append("Exposure replacement check:")
        lines.append(f"  changed exposure entries     : {changed_exp}")

    if sample_after.flare_mask is not None:
        flare_idx = np.flatnonzero(sample_after.flare_mask)
        nshow = min(max_rows, len(flare_idx))

        lines.append("")
        lines.append(f"First {nshow} flare-tagged events:")
        lines.append(" idx |      RA [deg] |     Dec [deg] |     exposure")
        lines.append("-" * 70)

        for idx in flare_idx[:nshow]:
            exp_i = None if sample_after.exposure is None else sample_after.exposure[idx]
            if exp_i is None or not np.isfinite(exp_i):
                exp_txt = "None"
            else:
                exp_txt = f"{exp_i:.6g}"

            lines.append(
                f"{idx:4d} | "
                f"{sample_after.RA[idx]:13.6f} | "
                f"{sample_after.Dec[idx]:13.6f} | "
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
        RA=np.array([]) if sample.RA is None else np.asarray(sample.RA),
        Dec=np.array([]) if sample.Dec is None else np.asarray(sample.Dec),
        exposure=np.array([]) if sample.exposure is None else np.asarray(sample.exposure),
        flare_mask=np.array([]) if sample.flare_mask is None else np.asarray(sample.flare_mask),
        n_events=sample.n_events,
        expected_n=sample.expected_n,
        t0_isot=sample.t0.isot,
        tf_isot=sample.tf.isot,
        T_obs_s=sample.T_obs.to_value(u.s),
        expected_temporal_rate=sample.expected_temporal_rate,
        spatial_type="" if sample.spatial_type is None else sample.spatial_type,
        exposure_type="" if sample.exposure_type is None else sample.exposure_type,
        expected_exposure_rate=np.nan if sample.expected_exposure_rate is None else sample.expected_exposure_rate,
        flare_type="" if sample.flare_type is None else sample.flare_type,
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
    ra = np.asarray(sample.RA, dtype=float)
    dec = np.asarray(sample.Dec, dtype=float)

    plt.figure(figsize=(6, 4))
    plt.hist(ra, bins="fd", alpha=0.8, edgecolor="black", linewidth=0.8)
    plt.xlabel("RA [deg]")
    plt.ylabel("Counts")
    plt.title("Right ascension distribution")
    plt.tight_layout()
    p = outdir / "ra_hist.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    saved.append(p)

    plt.figure(figsize=(6, 4))
    plt.hist(dec, bins="fd", alpha=0.8, edgecolor="black", linewidth=0.8)
    plt.xlabel("Dec [deg]")
    plt.ylabel("Counts")
    plt.title("Declination distribution")
    plt.tight_layout()
    p = outdir / "dec_hist.png"
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
    p = outdir / "sin_dec_hist.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    saved.append(p)

    if len(ra) >= 2:
        x = np.asarray(ra, dtype=float)
        y = np.asarray(dec, dtype=float)
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
        p = outdir / "sky_scatter.png"
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
    Intended for samples where exposure has been attached.
    """
    saved = []

    if sample.exposure is None or len(sample.exposure) == 0:
        return saved

    eps = np.asarray(sample.exposure, dtype=float)
    finite = np.isfinite(eps)

    if not np.any(finite):
        return saved

    eps = eps[finite]

    plt.figure(figsize=(6, 4))
    plt.hist(eps, bins="sqrt", alpha=0.8, edgecolor="black", linewidth=0.8)
    plt.xlabel("Exposure")
    plt.ylabel("Counts")
    plt.title("Exposure distribution")
    plt.tight_layout()
    p = outdir / "exposure_hist.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    saved.append(p)

    return saved


def save_skymap_plot(
    sample: EventSample,
    outdir: Path,
    *,
    filename: str = "skymap.png",
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
    path = outdir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ------------------------------------------------------------------
# Core diagnostic runners
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
        plot_paths.extend(save_coordinate_plots(sample, outdir=outdir))

    if save_exposure:
        plot_paths.extend(save_exposure_plots(sample, outdir=outdir))

    if save_skymap:
        p = save_skymap_plot(
            sample,
            outdir=outdir,
            filename="skymap.png",
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


def clone_event_sample(sample: EventSample) -> EventSample:
    """
    Clone an EventSample without resampling coordinates.

    This uses the internal constructor from arrays, so your EventSample._from_arrays
    method must already be fixed and consistent with flare_mask.
    """
    return EventSample._from_arrays(
        RA=np.array(sample.RA, copy=True),
        Dec=np.array(sample.Dec, copy=True),
        t0=sample.t0,
        tf=sample.tf,
        rng=sample.rng,
        spatial_type=sample.spatial_type,
        expected_n=sample.expected_n,
        exposure=None if sample.exposure is None else np.array(sample.exposure, copy=True),
        exposure_type=sample.exposure_type,
        expected_exposure_rate=sample.expected_exposure_rate,
        flare_mask=None if sample.flare_mask is None else np.array(sample.flare_mask, copy=True),
        flare_type=sample.flare_type,
    )


def run_flare_injection_diagnostic(
    parent_sample: EventSample,
    window: SkyWindow,
    exposure_model: ExposureModel,
    flare: Flare,
    *,
    case_name: str = "flare_injection",
    max_rows: int = 10,
    nside: int = 32,
    mask_fov: bool = False,
    location=None,
    zenith_max=None,
) -> None:
    """
    Run a diagnostic for flare injection.

    Saves diagnostics for:
    1) the full parent sample before injection,
    2) the full parent sample after flare injection,
    3) the cut sample after flare injection.
    """
    outdir = build_case_output_dir(case_name)

    # --------------------------------------------------------------
    # Full sample before injection
    # --------------------------------------------------------------
    full_before_dir = outdir / "full_sample_before_injection"
    full_before_dir.mkdir(parents=True, exist_ok=True)

    full_before_summary = event_sample_summary_text(
        parent_sample,
        label="FULL SAMPLE BEFORE FLARE INJECTION",
        max_rows=max_rows,
    )
    (full_before_dir / "full_sample_before_injection_summary.txt").write_text(
        full_before_summary,
        encoding="utf-8",
    )
    save_event_sample_arrays(parent_sample, full_before_dir, "full_sample_before_injection")
    save_coordinate_plots(parent_sample, full_before_dir)
    save_skymap_plot(
        parent_sample,
        full_before_dir,
        filename="skymap.png",
        nside=nside,
        mask_fov=mask_fov,
        location=location,
        zenith_max=zenith_max,
        title="Full sample before flare injection",
    )

    # --------------------------------------------------------------
    # Full sample after injection
    # --------------------------------------------------------------
    full_after = clone_event_sample(parent_sample)
    full_after.inject_flare(flare)

    full_after_dir = outdir / "full_sample_after_injection"
    full_after_dir.mkdir(parents=True, exist_ok=True)

    full_after_summary = event_sample_summary_text(
        full_after,
        label="FULL SAMPLE AFTER FLARE INJECTION",
        max_rows=max_rows,
    )
    full_injection_summary = flare_injection_summary_text(
        parent_sample,
        full_after,
        label="FULL-SAMPLE FLARE INJECTION CHECK",
        max_rows=max_rows,
    )

    (full_after_dir / "full_sample_after_injection_summary.txt").write_text(
        full_after_summary,
        encoding="utf-8",
    )
    (full_after_dir / "flare_injection_check.txt").write_text(
        full_injection_summary,
        encoding="utf-8",
    )

    save_event_sample_arrays(full_after, full_after_dir, "full_sample_after_injection")
    save_coordinate_plots(full_after, full_after_dir)
    save_exposure_plots(full_after, full_after_dir)
    save_skymap_plot(
        full_after,
        full_after_dir,
        filename="skymap.png",
        nside=nside,
        mask_fov=mask_fov,
        location=location,
        zenith_max=zenith_max,
        title="Full sample after flare injection",
    )

    # --------------------------------------------------------------
    # Cut sample after injection
    # --------------------------------------------------------------
    cut_after = full_after.select_subsample(window)

    cut_after_dir = outdir / "cut_after_injection"
    cut_after_dir.mkdir(parents=True, exist_ok=True)

    cut_after_summary = event_sample_summary_text(
        cut_after,
        label="CUT SAMPLE AFTER FLARE INJECTION",
        max_rows=max_rows,
    )
    (cut_after_dir / "cut_after_injection_summary.txt").write_text(
        cut_after_summary,
        encoding="utf-8",
    )

    save_event_sample_arrays(cut_after, cut_after_dir, "cut_after_injection")
    save_coordinate_plots(cut_after, cut_after_dir)
    save_exposure_plots(cut_after, cut_after_dir)
    save_skymap_plot(
        cut_after,
        cut_after_dir,
        filename="skymap.png",
        nside=nside,
        mask_fov=mask_fov,
        location=location,
        zenith_max=zenith_max,
        title="Cut sample after flare injection",
    )

    print("\n" + "=" * 70)
    print("FLARE INJECTION DIAGNOSTIC")
    print("=" * 70)
    print(f"Saved full sample before injection : {full_before_dir}")
    print(f"Saved full sample after injection  : {full_after_dir}")
    print(f"Saved cut sample after injection   : {cut_after_dir}")

def build_example_flare(
    *,
    rng_manager: RNGManager,
    window: SkyWindow,
    exposure_model: ExposureModel,
    t0: Time,
    tf: Time,
    n_flare: int = 200,
    flare_duration: u.Quantity = 1.0 * u.day,
    flare_sigma: float = 3.0,
) -> Flare:
    """
    Build and fully generate a flare object for the injection diagnostic.
    """
    rng_flare = rng_manager.get("flare")

    flare = Flare(
        n_events=n_flare,
        duration=flare_duration,
        t0=t0,
        tf=tf,
        centre=window.centre,
        exposure_model=exposure_model,
        rng=rng_flare,
    )
    flare.generate_in_window(
        window=window,
        sigma=flare_sigma,
    )

    return flare


# ------------------------------------------------------------------
# Example usage
# ------------------------------------------------------------------

if __name__ == "__main__":

    rng_manager = RNGManager(seed=42)
    rng_sample = rng_manager.get("sample")
    rng_exposure = rng_manager.get("exposure")

    # Observation interval
    t0 = Time("2025-01-01T00:00:00")
    tf = Time("2025-02-01T00:00:00")

    # --------------------------------------------------------------
    # Full sample
    # --------------------------------------------------------------
    n_events = 10000
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
    altitude_pa = 1425.0

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

    subsample.assign_directional_exposure(
        window=window,
        exposure_model=exposure_model,
    )

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

    # --------------------------------------------------------------
    # Flare injection diagnostic
    # --------------------------------------------------------------
    flare = build_example_flare(
        rng_manager=rng_manager,
        window=window,
        exposure_model=exposure_model,
        t0=t0,
        tf=tf,
        n_flare=200,
        flare_duration=1.0 * u.day,
        flare_sigma=3.0,
    )

    run_flare_injection_diagnostic(
        parent_sample=sample,
        window=window,
        exposure_model=exposure_model,
        flare=flare,
        case_name="flare_injection",
        max_rows=12,
        nside=32,
        mask_fov=False,
        location=location,
        zenith_max=zenith_max,
    )