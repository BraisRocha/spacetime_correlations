"""
EventSample visual / inspection diagnostic.

Two pipelines, side by side:

1. **Full-sky pipeline** — generate an isotropic full-sky sample via
   ``EventSample.full_sky``, then carve out a window via
   ``select_subsample`` and attach exposure via
   ``assign_directional_exposure``.

2. **Per-window pipeline** — generate events directly inside a window via
   ``EventSample.in_window``, then attach exposure.

For each pipeline this script prints a text summary, dumps the underlying
arrays as ``.npz``, and saves coordinate / exposure histograms plus a
HEALPix Hammer-projection skymap.  A flare-injection case is run on the
per-window output.

No assertions — pass/fail checks for these contracts live in
``tests/test_event_sample.py``.
"""

from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.time import Time
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from spacetimecorr import (
    EventSample,
    ExposureModel,
    Flare,
    Observatory,
    RNGManager,
    SkyWindow,
)


# -------------------------------------------------------------------------
# Output helpers
# -------------------------------------------------------------------------


def build_output_dir() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    outdir = project_root / "output" / "diagnostics" / "eventsample"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def build_case_output_dir(case_name: str) -> Path:
    outdir = build_output_dir() / case_name
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


# -------------------------------------------------------------------------
# Text summaries
# -------------------------------------------------------------------------


def event_sample_summary_text(
    sample: EventSample,
    *,
    label: str = "EVENT SAMPLE",
    max_rows: int = 10,
) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append(f"{label} DIAGNOSTIC SUMMARY")
    lines.append("=" * 70)

    lines.append(f"n_sample                     : {sample.n_sample}")
    lines.append(f"n_total                      : {sample.n_total}")
    lines.append(f"expected_n                   : {sample.expected_n}")
    lines.append(f"spatial_type                 : {sample.spatial_type}")
    lines.append(f"exposure_type                : {sample.exposure_type}")
    lines.append(f"flare_type                   : {sample.flare_type}")
    lines.append(f"window                       : "
                 f"{None if sample.window is None else f'centre={list(sample.window.centre)}, radius={sample.window.radius} deg'}")
    lines.append(f"exposure_model attached?     : {sample.exposure_model is not None}")
    lines.append(f"t0                           : {sample.t0.isot}")
    lines.append(f"tf                           : {sample.tf.isot}")
    lines.append(f"T_obs [s]                    : {sample.T_obs.to_value(u.s):.6f}")
    if sample.expected_n is not None:
        lines.append(f"expected_temporal_rate [1/s] : {sample.expected_temporal_rate:.6g}")
    lines.append(f"has_coordinates              : {sample.has_coordinates}")
    lines.append(f"has_exposure                 : {sample.has_exposure}")
    lines.append(f"has_flare                    : {sample.has_flare}")

    lengths = {
        "ra":         None if sample.ra         is None else len(sample.ra),
        "dec":        None if sample.dec        is None else len(sample.dec),
        "exposure":   None if sample.exposure   is None else len(sample.exposure),
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

    ra = np.asarray(sample.ra, dtype=float)
    dec = np.asarray(sample.dec, dtype=float)

    lines.append("")
    lines.append("Coordinate diagnostics:")
    lines.append(f"  finite ra?                 : {bool(np.all(np.isfinite(ra)))}")
    lines.append(f"  finite dec?                : {bool(np.all(np.isfinite(dec)))}")
    lines.append(f"  ra min [deg]               : {np.min(ra):.6f}")
    lines.append(f"  ra max [deg]               : {np.max(ra):.6f}")
    lines.append(f"  dec min [deg]              : {np.min(dec):.6f}")
    lines.append(f"  dec max [deg]              : {np.max(dec):.6f}")
    lines.append(f"  ra in [0, 360)?            : {bool(np.all((ra >= 0.0) & (ra < 360.0)))}")
    lines.append(f"  dec in [-90, 90]?          : {bool(np.all((dec >= -90.0) & (dec <= 90.0)))}")

    sin_dec = np.sin(np.deg2rad(dec))
    lines.append("")
    lines.append("Isotropy quick-check diagnostics:")
    lines.append(f"  mean(sin dec)              : {np.mean(sin_dec):.6g}")
    lines.append(f"  std(sin dec)               : {np.std(sin_dec):.6g}")
    lines.append(f"  mean(ra) [deg]             : {np.mean(ra):.6g}")

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
    lines.append(" idx |      ra [deg] |     dec [deg] |     exposure | flare")
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
            f"{sample.ra[i]:13.6f} | "
            f"{sample.dec[i]:13.6f} | "
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
    lines = []
    lines.append("=" * 70)
    lines.append(f"{label} SUMMARY")
    lines.append("=" * 70)

    lines.append(f"n_sample before                : {sample_before.n_sample}")
    lines.append(f"n_sample after                 : {sample_after.n_sample}")
    lines.append(f"same sample size               : {sample_before.n_sample == sample_after.n_sample}")

    lines.append("")
    lines.append("Flare bookkeeping:")
    lines.append(f"  has_flare after injection    : {sample_after.has_flare}")
    lines.append(f"  flare_type                   : {sample_after.flare_type}")

    if sample_after.flare_mask is not None:
        n_flare = int(np.count_nonzero(sample_after.flare_mask))
        lines.append(f"  number flagged as flare      : {n_flare}")
    else:
        lines.append("  number flagged as flare      : None")

    if sample_before.ra is not None and sample_after.ra is not None:
        changed_ra = int(np.count_nonzero(sample_before.ra != sample_after.ra))
        changed_dec = int(np.count_nonzero(sample_before.dec != sample_after.dec))
        lines.append("")
        lines.append("Coordinate replacement check:")
        lines.append(f"  changed ra entries           : {changed_ra}")
        lines.append(f"  changed dec entries          : {changed_dec}")

    if sample_before.exposure is not None and sample_after.exposure is not None:
        before = np.asarray(sample_before.exposure, dtype=float)
        after = np.asarray(sample_after.exposure, dtype=float)
        changed_exp = int(np.count_nonzero(~np.isclose(before, after, equal_nan=True)))
        lines.append("")
        lines.append("Exposure replacement check:")
        lines.append(f"  changed exposure entries     : {changed_exp}")

    if sample_after.flare_mask is not None:
        flare_idx = np.flatnonzero(sample_after.flare_mask)
        nshow = min(max_rows, len(flare_idx))
        lines.append("")
        lines.append(f"First {nshow} flare-tagged events:")
        lines.append(" idx |      ra [deg] |     dec [deg] |     exposure")
        lines.append("-" * 70)
        for idx in flare_idx[:nshow]:
            exp_i = None if sample_after.exposure is None else sample_after.exposure[idx]
            if exp_i is None or not np.isfinite(exp_i):
                exp_txt = "None"
            else:
                exp_txt = f"{exp_i:.6g}"
            lines.append(
                f"{idx:4d} | "
                f"{sample_after.ra[idx]:13.6f} | "
                f"{sample_after.dec[idx]:13.6f} | "
                f"{exp_txt}"
            )

    return "\n".join(lines)


# -------------------------------------------------------------------------
# Save arrays
# -------------------------------------------------------------------------


def save_event_sample_arrays(
    sample: EventSample,
    outdir: Path,
    stem: str,
) -> Path:
    path = outdir / f"{stem}_arrays.npz"
    np.savez_compressed(
        path,
        ra=np.array([]) if sample.ra is None else np.asarray(sample.ra),
        dec=np.array([]) if sample.dec is None else np.asarray(sample.dec),
        exposure=np.array([]) if sample.exposure is None else np.asarray(sample.exposure),
        flare_mask=np.array([]) if sample.flare_mask is None else np.asarray(sample.flare_mask),
        n_sample=sample.n_sample,
        n_total=sample.n_total,
        expected_n=np.nan if sample.expected_n is None else sample.expected_n,
        t0_isot=sample.t0.isot,
        tf_isot=sample.tf.isot,
        T_obs_s=sample.T_obs.to_value(u.s),
        spatial_type="" if sample.spatial_type is None else sample.spatial_type,
        exposure_type="" if sample.exposure_type is None else sample.exposure_type,
        expected_exposure_rate=np.nan if sample.expected_exposure_rate is None
                               else sample.expected_exposure_rate,
        flare_type="" if sample.flare_type is None else sample.flare_type,
    )
    return path


# -------------------------------------------------------------------------
# Plots
# -------------------------------------------------------------------------


def save_coordinate_plots(sample: EventSample, outdir: Path) -> list[Path]:
    if sample.ra is None or sample.dec is None:
        raise ValueError("Sample must be populated before plotting.")

    saved = []
    ra = np.asarray(sample.ra, dtype=float)
    dec = np.asarray(sample.dec, dtype=float)

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


def save_exposure_plots(sample: EventSample, outdir: Path) -> list[Path]:
    saved = []
    if sample.exposure is None or len(sample.exposure) == 0:
        return saved

    eps = np.asarray(sample.exposure, dtype=float)
    finite = np.isfinite(eps)
    if not np.any(finite):
        return saved

    plt.figure(figsize=(6, 4))
    plt.hist(eps[finite], bins="sqrt", alpha=0.8, edgecolor="black", linewidth=0.8)
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
) -> Path | None:
    """Save the HEALPix Hammer skymap; silently skip if ``healpy`` is missing."""
    try:
        fig, _ax = sample.plot_skymap(
            nside=nside,
            mask_fov=mask_fov,
            location=location,
            zenith_max=zenith_max,
            title=title,
            output_file=None,
            show=False,
        )
    except ModuleNotFoundError as exc:
        print(f"[sampling_diagnostic] skipping skymap ({filename}): {exc}")
        return None

    path = outdir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# -------------------------------------------------------------------------
# Core runners
# -------------------------------------------------------------------------


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
    """Print diagnostics for one EventSample and save summary, arrays, and plots."""
    outdir = build_case_output_dir(case_name)

    summary = event_sample_summary_text(sample, label=label, max_rows=max_rows)
    print(summary)
    (outdir / f"{stem}_summary.txt").write_text(summary, encoding="utf-8")

    arrays_path = save_event_sample_arrays(sample, outdir=outdir, stem=stem)

    plot_paths = []
    if save_coordinates:
        plot_paths.extend(save_coordinate_plots(sample, outdir=outdir))
    if save_exposure:
        plot_paths.extend(save_exposure_plots(sample, outdir=outdir))
    if save_skymap:
        skymap_path = save_skymap_plot(
            sample, outdir=outdir,
            filename="skymap.png", nside=nside,
            mask_fov=mask_fov, location=location, zenith_max=zenith_max,
            title=f"{label.title()} skymap",
        )
        if skymap_path is not None:
            plot_paths.append(skymap_path)

    print("\nSaved diagnostic files:")
    print(f"  Summary : {outdir / f'{stem}_summary.txt'}")
    print(f"  Arrays  : {arrays_path}")
    for p in plot_paths:
        print(f"  Plot    : {p}")


def clone_event_sample(sample: EventSample) -> EventSample:
    """Clone an EventSample without redrawing coordinates."""
    return EventSample._from_arrays(
        ra=np.array(sample.ra, copy=True),
        dec=np.array(sample.dec, copy=True),
        n_total=sample.n_total,
        t0=sample.t0,
        tf=sample.tf,
        rng=sample.rng,
        spatial_type=sample.spatial_type,
        expected_n=sample.expected_n,
        window=sample.window,
        exposure_model=sample.exposure_model,
        exposure=None if sample.exposure is None else np.array(sample.exposure, copy=True),
        exposure_type=sample.exposure_type,
        expected_exposure_rate=sample.expected_exposure_rate,
        flare_mask=None if sample.flare_mask is None else np.array(sample.flare_mask, copy=True),
        flare_type=sample.flare_type,
    )


def run_flare_injection_diagnostic(
    parent_sample: EventSample,
    window: SkyWindow,
    flare: Flare,
    *,
    case_name: str = "flare_injection",
    max_rows: int = 10,
    nside: int = 32,
    mask_fov: bool = False,
    location=None,
    zenith_max=None,
) -> None:
    """Run a flare-injection diagnostic on ``parent_sample`` (already populated)."""
    outdir = build_case_output_dir(case_name)

    # Before injection
    before_dir = outdir / "sample_before_injection"
    before_dir.mkdir(parents=True, exist_ok=True)
    before_summary = event_sample_summary_text(
        parent_sample, label="SAMPLE BEFORE FLARE INJECTION", max_rows=max_rows,
    )
    (before_dir / "summary.txt").write_text(before_summary, encoding="utf-8")
    save_event_sample_arrays(parent_sample, before_dir, "sample_before_injection")
    save_coordinate_plots(parent_sample, before_dir)
    save_exposure_plots(parent_sample, before_dir)
    save_skymap_plot(
        parent_sample, before_dir, filename="skymap.png", nside=nside,
        mask_fov=mask_fov, location=location, zenith_max=zenith_max,
        title="Sample before flare injection",
    )

    # After injection
    after = clone_event_sample(parent_sample)
    after.inject_flare(flare)

    after_dir = outdir / "sample_after_injection"
    after_dir.mkdir(parents=True, exist_ok=True)
    after_summary = event_sample_summary_text(
        after, label="SAMPLE AFTER FLARE INJECTION", max_rows=max_rows,
    )
    injection_summary = flare_injection_summary_text(
        parent_sample, after, label="FLARE INJECTION CHECK", max_rows=max_rows,
    )
    (after_dir / "summary.txt").write_text(after_summary, encoding="utf-8")
    (after_dir / "injection_check.txt").write_text(injection_summary, encoding="utf-8")
    save_event_sample_arrays(after, after_dir, "sample_after_injection")
    save_coordinate_plots(after, after_dir)
    save_exposure_plots(after, after_dir)
    save_skymap_plot(
        after, after_dir, filename="skymap.png", nside=nside,
        mask_fov=mask_fov, location=location, zenith_max=zenith_max,
        title="Sample after flare injection",
    )

    print("\n" + "=" * 70)
    print(f"FLARE INJECTION DIAGNOSTIC ({case_name})")
    print("=" * 70)
    print(f"Saved before injection : {before_dir}")
    print(f"Saved after  injection : {after_dir}")


# -------------------------------------------------------------------------
# Helpers for the example
# -------------------------------------------------------------------------


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
    rng_flare = rng_manager.get("flare")
    flare = Flare(
        n_flare=n_flare,
        duration=flare_duration,
        t0=t0, tf=tf,
        centre=window.centre,
        exposure_model=exposure_model,
        rng=rng_flare,
    )
    flare.generate_in_window(window=window, sigma=flare_sigma)
    return flare


# -------------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------------


if __name__ == "__main__":

    rng_manager = RNGManager(seed=42)
    rng_full_sky = rng_manager.get("full_sky_sample")
    rng_in_window = rng_manager.get("in_window_sample")
    rng_exposure = rng_manager.get("exposure")

    t0 = Time("2025-01-01T00:00:00")
    tf = Time("2025-02-01T00:00:00")

    centre = np.array([30.0, -30.0])
    radius = 20.0
    window = SkyWindow(centre=centre, radius=radius)

    observatory = Observatory(latitude=-35.15, longitude=-69.15, altitude=1425.0)
    exposure_model = ExposureModel(
        observatory=observatory, t0=t0, tf=tf, rng=rng_exposure,
    )
    location = observatory.location
    zenith_max = 60 * u.deg

    n_total = 10_000

    # ------------------------------------------------------------------
    # Pipeline 1: full-sky -> subsample -> exposure
    # ------------------------------------------------------------------
    full_sample = EventSample.full_sky(
        n_total=n_total, t0=t0, tf=tf, rng=rng_full_sky,
    )
    run_event_sample_diagnostic(
        full_sample,
        case_name="full_sky/full_sample",
        label="FULL-SKY SAMPLE",
        stem="full_sample",
        max_rows=12,
        save_coordinates=True, save_exposure=False, save_skymap=True,
    )

    subsample = full_sample.select_subsample(window)
    subsample.assign_directional_exposure(window=window, exposure_model=exposure_model)
    run_event_sample_diagnostic(
        subsample,
        case_name="full_sky/subsample",
        label="WINDOW SUBSAMPLE (from full-sky)",
        stem="subsample",
        max_rows=12,
        save_coordinates=True, save_exposure=True, save_skymap=True,
        location=location, zenith_max=zenith_max,
    )

    # ------------------------------------------------------------------
    # Pipeline 2: direct per-window sample -> exposure
    # ------------------------------------------------------------------
    in_window_sample = EventSample.in_window(
        window=window,
        n_total=n_total,
        exposure_model=exposure_model,
        t0=t0, tf=tf,
        rng=rng_in_window,
    )
    in_window_sample.assign_directional_exposure(window=window, exposure_model=exposure_model)
    run_event_sample_diagnostic(
        in_window_sample,
        case_name="in_window/sample",
        label="PER-WINDOW SAMPLE",
        stem="in_window_sample",
        max_rows=12,
        save_coordinates=True, save_exposure=True, save_skymap=True,
        location=location, zenith_max=zenith_max,
    )

    # ------------------------------------------------------------------
    # Flare injection diagnostic on the per-window sample
    # ------------------------------------------------------------------
    flare = build_example_flare(
        rng_manager=rng_manager,
        window=window,
        exposure_model=exposure_model,
        t0=t0, tf=tf,
        n_flare=50,
        flare_duration=1.0 * u.day,
        flare_sigma=3.0,
    )
    run_flare_injection_diagnostic(
        parent_sample=in_window_sample,
        window=window,
        flare=flare,
        case_name="in_window/flare_injection",
        max_rows=12,
        location=location, zenith_max=zenith_max,
    )
