"""
Build the active-area file from the hexagon files. Run once, after
downloading the hexagon data.

Reads ``inputs/hexagon_data/`` and writes to ``inputs/active_area/``:

- ``ActiveArea_SD1500_UB_6T5.npz``, which the analyses read with
  ``spacetimecorr.io.load_active_area``;
- scatter plots of A(t) as a check: ``ActiveArea_SD1500_UB_6T5.png`` over
  the whole span, and one per year in ``yearly/`` with the bad periods
  marked.
"""

from pathlib import Path

import matplotlib as mpl

from spacetimecorr.io import ActiveAreaProcessor

# Global style: the matplotlibrc of the plotting scripts
RC_FILE = Path(__file__).resolve().parents[1] / "plots" / "matplotlibrc_test"
if RC_FILE.exists():
    mpl.rc_file(RC_FILE, use_default_template=False)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    out_dir = project_root / "inputs" / "active_area"

    processor = ActiveAreaProcessor(
        data_dir=project_root / "inputs" / "hexagon_data",
        array_type="1500",
        ub_type="UB",
        config="6T5",
        bad_periods=True,
    )
    print(f"Saved {processor.save(out_dir)}")
    figures = processor.plot(out_dir)
    print(f"Saved {len(figures)} figures to {out_dir}")


if __name__ == "__main__":
    main()
