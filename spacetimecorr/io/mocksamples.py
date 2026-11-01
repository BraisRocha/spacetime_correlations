from pathlib import Path

import numpy as np
from astropy.time import Time


def load_mock_sample(path: Path) -> tuple[Time, np.ndarray, np.ndarray]:
    """Read a mock sample: arrival times, RA and Dec (deg).

    Columns: Unix UTC seconds, nanoseconds, RA, Dec, angular resolution
    (unused). Both time columns are integers below 2**53, so float64 holds
    them exactly; they are then passed to ``Time`` as two numbers.
    Nanosecond values outside ``[0, 1e9)`` are taken literally, as an offset
    from the second.
    """
    seconds, nanoseconds, ra, dec = np.loadtxt(path, usecols=(0, 1, 2, 3), unpack=True)
    times = Time(seconds, nanoseconds * 1e-9, format="unix", scale="utc")
    return times, ra, dec
