from pathlib import Path

import numpy as np
from astropy.time import Time

from .output import write_metadata


class ActiveAreaProcessor:
    """
    Active area A(t) of the surface detector, minute by minute, in km^2.

    Built once from the hexagon files of one array, and saved with
    :meth:`save` to the file read by :func:`load_active_area`.

    Each active cell covers the area of one cell of a triangular grid of
    spacing ``d``, and the cells counted are those of ``config``::

        A(t) = N_cells(t) * (sqrt(3) / 2) * d^2

    with ``d`` = 1500, 750 or 433 m for the SD-1500, SD-750 and SD-433
    arrays. During bad periods A is considered to be 0.

    Parameters
    ----------
    data_dir : Path
        Folder holding the hexagon files ``SD{array_type}_HexagonsMn_{ub_type}_*.out``
        and the bad-period file. Each hexagon file has one row per minute:
        GPS time of the start of the minute, then the numbers of 6T5, 5T5
        and 4T5 cells, averaged over the per-second counts of the minute.
        They are rounded to whole cells before computing A.
    array_type : {"1500", "750", "433"}
        Array, i.e. grid spacing in m.
    ub_type : {"UB", "UUB"}
        Electronics.
    config : {"6T5", "6T5_5T5", "6T5_5T5_4T5"}
        Cells counted in N_cells, with weights on (N_6T5, N_5T5, N_4T5) of
        (1, 0, 0) for "6T5" (default), (1, 2/3, 0) for "6T5_5T5" and
        (1, 1, 2/3) for "6T5_5T5_4T5", taken from the Auger active-area code
        this processor was based on.
    bad_periods : bool
        If True (default), A = 0 during the bad periods.
    """

    _GRID_SPACING_KM = {"1500": 1.5, "750": 0.75, "433": 0.433}
    _CONFIG_WEIGHTS = {
        "6T5": np.array([1.0, 0.0, 0.0]),
        "6T5_5T5": np.array([1.0, 2.0 / 3.0, 0.0]),
        "6T5_5T5_4T5": np.array([1.0, 1.0, 2.0 / 3.0]),
    }
    # Bad-period file of each electronics. None is available for UUB yet.
    _BAD_PERIODS_FILE = {"UB": "BadPeriods_SD{array_type}_PhaseI.dat"}

    def __init__(
        self,
        data_dir: Path,
        array_type: str = "1500",
        ub_type: str = "UB",
        config: str = "6T5",
        bad_periods: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.array_type = array_type
        self.ub_type = ub_type
        self.config = config
        self.bad_periods = bad_periods

        self._check_attributes()

        spacing_km = self._GRID_SPACING_KM[self.array_type]
        self.cell_area = (np.sqrt(3.0) / 2.0) * spacing_km**2   # km^2

        self._read_hexagon_files()
        self._read_bad_periods()
        self._compute_area()
        self._build_steps()

    def _check_attributes(self) -> None:
        if self.array_type not in self._GRID_SPACING_KM:
            raise KeyError(f"Unknown array type: {self.array_type!r}")
        if self.ub_type not in ("UB", "UUB"):
            raise KeyError(f"Unknown electronics type: {self.ub_type!r}")
        if self.config not in self._CONFIG_WEIGHTS:
            raise KeyError(f"Unknown cell configuration: {self.config!r}")
        if self.bad_periods and self.ub_type not in self._BAD_PERIODS_FILE:
            raise KeyError(f"No bad-period file known for {self.ub_type!r}")

    def _read_hexagon_files(self) -> None:
        """Merge the hexagon files into one table, in time order."""
        # Exact pattern, so that "UB" does not also pick up the "UUB" files.
        pattern = f"SD{self.array_type}_HexagonsMn_{self.ub_type}_*.out"
        self.hexagon_paths = sorted(self.data_dir.glob(pattern))
        if not self.hexagon_paths:
            raise FileNotFoundError(f"No hexagon files {pattern} in {self.data_dir}")

        data = np.concatenate([np.loadtxt(path) for path in self.hexagon_paths])
        data = data[np.argsort(data[:, 0])]

        # One row per minute, with none missing: bad periods are handled by
        # setting A = 0, never by leaving rows out. Anything else (a gap, or a
        # repeated minute) is not expected and has to be looked into.
        spacing = np.diff(data[:, 0])
        irregular = np.flatnonzero(spacing != 60.0)
        if irregular.size:
            first = irregular[:5]
            listing = ", ".join(
                f"{Time(data[i, 0], format='gps').isot} (+{spacing[i]:.0f} s)" for i in first
            )
            raise ValueError(
                f"Hexagon rows are not one minute apart at {irregular.size} place(s), "
                f"e.g. after {listing}."
            )

        self.gps_time = data[:, 0]
        self.n_cells = data[:, 1:4]          # columns N_6T5, N_5T5, N_4T5

    def _read_bad_periods(self) -> None:
        """
        Read the bad periods, and flag the minutes that start inside one
        (used for the per-minute A of the plots).
        """
        if not self.bad_periods:
            self.bad_period_data = np.empty((0, 2))
            self.in_bad_period = np.zeros(self.gps_time.size, dtype=bool)
            return

        name = self._BAD_PERIODS_FILE[self.ub_type].format(array_type=self.array_type)
        # Columns 1 and 9: GPS times of the start and end of each period.
        self.bad_period_data = np.loadtxt(self.data_dir / name, usecols=(1, 9))

        # Inside a bad period when more periods have started than ended.
        start, end = self.bad_period_data.T
        n_started = np.searchsorted(start, self.gps_time, side="right")
        n_ended = np.searchsorted(end, self.gps_time, side="right")
        self.in_bad_period = n_started > n_ended

    def _compute_area(self) -> None:
        # The counts are rounded to whole cells, so that A changes less often
        # and more minutes merge into one step. This biases A slightly: for
        # the SD-1500 UB data, 2004-2026, it raises the integral of A over
        # time by 2e-5 in relative terms.
        n_cells = np.round(self.n_cells)
        self.area_no_bad = (n_cells @ self._CONFIG_WEIGHTS[self.config]) * self.cell_area

        # A per minute, 0 for the minutes that start inside a bad period. Used
        # for the plots; the saved steps (`_build_steps`) cut the minutes at
        # the exact limits of the bad periods instead.
        self.area = self.area_no_bad.copy()
        self.area[self.in_bad_period] = 0.0

    def _build_steps(self) -> None:
        """
        Steps of A(t) to save: the minutes, cut at the exact start and end of
        every bad period, with consecutive equal values merged.

        Each value of A holds until the next start time. A step starts at
        every minute and at every bad-period limit inside the data; it has
        A = 0 if it starts inside a bad period, and the area of its minute
        otherwise. A = 0 then covers the bad periods to the second, not
        whole minutes. Dropping a step whose A equals that of the step
        before leaves A(t) unchanged, with fewer steps.
        """
        starts = self.gps_time
        if self.bad_periods:
            # Bad-period limits inside the data, from the first minute to the
            # end of the last one.
            limits = self.bad_period_data.ravel()
            limits = limits[(limits > starts[0]) & (limits < starts[-1] + 60.0)]
            starts = np.union1d(starts, limits)

        minute = np.searchsorted(self.gps_time, starts, side="right") - 1
        area = self.area_no_bad[minute]
        if self.bad_periods:
            # Inside a bad period when more periods have started than ended.
            bad_start, bad_end = self.bad_period_data.T
            inside = (np.searchsorted(bad_start, starts, side="right")
                      > np.searchsorted(bad_end, starts, side="right"))
            area[inside] = 0.0

        # Keep the first step and those where A changes. The last one is
        # always kept too, even with the same A as the one before, so that
        # the last start time is the end of the data.
        keep = np.concatenate(([True], np.diff(area) != 0.0))
        keep[-1] = True
        self.step_gps_time = starts[keep]
        self.step_area = area[keep]

    def _file_stem(self) -> str:
        name = f"ActiveArea_SD{self.array_type}_{self.ub_type}_{self.config}"
        return name if self.bad_periods else name + "_noBadPeriods"

    def save(self, out_dir: Path) -> Path:
        """
        Save the merged steps to ``out_dir``, and return the path of the file.

        The ``.npz`` file holds ``gps_time``, the start of each step, and
        ``area``, A over it in km^2. A ``_metadata.json`` file next to it
        records the number of values before (one per minute) and after
        merging.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = self._file_stem()
        path = out_dir / f"{stem}.npz"
        np.savez_compressed(path, gps_time=self.step_gps_time, area=self.step_area)

        n_minutes = self.area.size
        n_steps = self.step_area.size
        write_metadata(
            out_dir,
            {
                "n_values_per_minute": n_minutes,
                "n_values_after_merge": n_steps,
                "reduction_percentage": round(100.0 * (1 - n_steps / n_minutes ), 2),
            },
            filename=f"{stem}_metadata.json",
        )
        return path

    def plot(self, out_dir: Path) -> list[Path]:
        """
        Scatter plots of A(t), one dot per minute, saved to ``out_dir``:
        one over the whole span, and one per calendar year (i.e. per
        hexagon file) in ``out_dir/yearly/`` with the bad periods marked.
        Returns the paths of the figures.

        In the yearly plots each bad period is a red band of its true
        width, plus a red tick at the top of the axes. Most bad periods
        last less than one pixel (~4.6 h), so the bands show the time lost
        and the ticks show where every period is.
        """
        import matplotlib.pyplot as plt

        out_dir = Path(out_dir)
        (out_dir / "yearly").mkdir(parents=True, exist_ok=True)
        stem = self._file_stem()

        year = Time(self.gps_time, format="gps").decimalyear
        bad = (
            Time(self.bad_period_data, format="gps").decimalyear
            if self.bad_period_data.size else np.empty((0, 2))
        )
        # Same y range in every plot, with room above the data for the ticks.
        y_max = 1.1 * self.area.max()

        def _plot(x_range, path, yearly):
            fig, ax = plt.subplots(figsize=(7.0, 2.5))
            if yearly:
                ax.broken_barh(
                    np.column_stack([bad[:, 0], bad[:, 1] - bad[:, 0]]), (0.0, 1.0),
                    transform=ax.get_xaxis_transform(),
                    facecolor="red", linewidth=0.0, alpha=0.3, zorder=0,
                )
                ax.vlines(bad[:, 0], 0.95, 1.0, transform=ax.get_xaxis_transform(),
                          color="red", linewidth=0.3)
            in_range = (year >= x_range[0]) & (year <= x_range[1])
            ax.plot(year[in_range], self.area[in_range], linestyle="none",
                    marker=".", markersize=0.3, markeredgewidth=0.0, color="C0",
                    rasterized=True)
            if yearly:
                # A tick at the start of every month, the year as the label.
                y = int(x_range[0])
                months = Time([f"{y}-{m:02d}-01" for m in range(1, 13)]).decimalyear
                ax.set_xticks(months, ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
                ax.set_xlabel(str(y))
            else:
                ax.set_xlabel("Year")
            ax.set_ylabel(r"$A$ [km$^2$]")
            ax.set_xlim(*x_range)
            ax.set_ylim(0.0, y_max)
            fig.tight_layout()
            fig.savefig(path, dpi=300)
            plt.close(fig)
            return path

        paths = [_plot((np.floor(year.min()), np.ceil(year.max())),
                       out_dir / f"{stem}.png", yearly=False)]
        for y in np.unique(np.floor(year)).astype(int):
            paths.append(_plot((y, y + 1), out_dir / "yearly" / f"{stem}_{y}.png",
                               yearly=True))
        return paths


def load_active_area(path: Path) -> tuple[Time, np.ndarray]:
    """
    Read a file written by :meth:`ActiveAreaProcessor.save`.

    Returns
    -------
    start_times : astropy.time.Time
        Start of every step, in time order. Each step lasts until the next
        start time.
    area : numpy.ndarray
        A(t) in km^2 over each step.

    The pair is what ``ExposureModel`` takes as ``active_area``.
    """
    with np.load(path) as data:
        return Time(data["gps_time"], format="gps"), data["area"]
