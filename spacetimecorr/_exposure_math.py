"""
Analytic geometry for directional exposure.

Private helpers backing :class:`~spacetimecorr.exposure.ExposureModel`.
Everything here is a pure function of the quantities below: no astropy,
no observatory, no random state.

Notation, used throughout this module:

    lat     observatory latitude            dec   source declination
    h       hour angle, continuous (rad)    c     cos(theta_max), the zenith cut
    A       sin(lat) * sin(dec)             B     cos(lat) * cos(dec)

so that the zenith angle of the source obeys

    cos theta(h) = A + B cos h

and the source lies inside the acceptance cone when ``cos theta(h) >= c``.
"""

from __future__ import annotations

import math

import numpy as np

TWO_PI = 2.0 * np.pi


def geometric_coefficients(lat_rad: float, dec_rad: float) -> tuple[float, float]:
    """Return the coefficients ``(A, B)`` for a latitude and declination."""
    return (
        math.sin(lat_rad) * math.sin(dec_rad),
        math.cos(lat_rad) * math.cos(dec_rad),
    )


def cut_hour_angle(A: float, B: float, c: float) -> float:
    """
    Hour angle ``h*`` at which the source crosses the zenith cut.

    The source is inside the cone for ``|h| < h*``, so ``h*`` spans the
    three visibility regimes continuously::

        h* = 0                      never enters the cone  (A + B <= c)
        h* = pi                     never leaves it        (A - B >= c)
        h* = acos((c - A) / B)      partial visibility
    """
    if A + B <= c:
        return 0.0
    if A - B >= c:
        return math.pi
    return math.acos((c - A) / B)


def half_cycle_integral(A: float, B: float, h_star: float) -> float:
    """
    Integral of ``cos theta`` over the visible half-cycle, in h-space::

        int_0^{h*} (A + B cos h) dh = A h* + B sin h*

    This is the per-cycle plateau of :func:`cumulative_integral`.
    """
    return A * h_star + B * math.sin(h_star)


def cos_zenith(h: np.ndarray, A: float, B: float) -> np.ndarray:
    """``cos theta(h) = A + B cos h``, clipped to ``[-1, 1]``."""
    return np.clip(A + B * np.cos(h), -1.0, 1.0)


def acceptance(h: np.ndarray, A: float, B: float, c: float) -> np.ndarray:
    """Geometric acceptance: ``cos theta(h)`` inside the cut, ``0`` outside."""
    ct = cos_zenith(h, A, B)
    return np.where(ct >= c, ct, 0.0)


def cumulative_integral(
    h: np.ndarray, h0: float, A: float, B: float, c: float
) -> np.ndarray:
    """
    Integral of :func:`acceptance` from ``h0`` to ``h``, in h-space.

    Divide by the sidereal rate ``2 pi / T_sid`` to obtain seconds. Uses
    the periodic primitive, so the cost is independent of how many
    sidereal cycles separate ``h0`` from ``h``.
    """
    h_star = cut_hour_angle(A, B, c)

    # `cut_hour_angle` returns the literal 0.0 and math.pi in the two
    # degenerate regimes, so these equality tests are exact.
    if h_star == 0.0:
        return np.zeros_like(np.asarray(h, dtype=float))
    if h_star == math.pi:
        return A * (h - h0) + B * (np.sin(h) - np.sin(h0))

    plateau = half_cycle_integral(A, B, h_star)
    per_cycle = 2.0 * plateau

    def primitive(x: np.ndarray) -> np.ndarray:
        n = np.floor(x / TWO_PI)
        eta = x - TWO_PI * n
        out = (n * per_cycle).astype(float)

        # The three masks tile [0, 2 pi) without overlap. The integrand is
        # continuous at both boundaries, so the edge assignment is free.
        rising = eta < h_star
        flat = (eta >= h_star) & (eta < TWO_PI - h_star)
        setting = eta >= TWO_PI - h_star

        out[rising] += A * eta[rising] + B * np.sin(eta[rising])
        out[flat] += plateau
        out[setting] += (
            per_cycle + A * (eta[setting] - TWO_PI) + B * np.sin(eta[setting])
        )
        return out

    return primitive(np.atleast_1d(h)) - primitive(np.atleast_1d(h0))[0]
