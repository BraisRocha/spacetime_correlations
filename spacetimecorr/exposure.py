"""
Exposure of a ground array, by direction and over the whole sky.

The exposure of a ground array is::

    E = ∫_sky dOmega ∫_array ∫_{t0}^{tf}
            P(E, theta, phi, t, ...) cos(theta) dA dt

in km² sr yr, where ``cos(theta) dA`` is the effective area the array
presents to a shower arriving at zenith angle ``theta``.

Restricting the solid-angle integral to one region of sky, and the time
integral to ``[t0, t]``, gives the *cumulative directional* exposure.
That is what :class:`ExposureModel` evaluates, for a single sky direction
(RA, Dec) over an observation interval ``[t0, tf]``, from the observatory
latitude and the source declination. It provides:

- the detection probability, the integrand of the expression above at
  one time and direction, normalised to its maximum,
- the cumulative directional exposure, either normalised to ``[0, 1]``
  (:meth:`ExposureModel.norm_cumul_exposure`) or in km² sr yr
  (:meth:`ExposureModel.cumul_exposure`),
- the whole-sky exposure (:meth:`ExposureModel.total_exposure`) and the
  share of it falling on one window
  (:meth:`ExposureModel.relative_window_exposure`), which sets the
  expected counts per sky window,
- helpers for sampling event times under the directional-exposure
  distribution (used by :class:`~spacetimecorr.flare.Flare` and by
  :class:`~spacetimecorr.event_sample.EventSample` when assigning
  exposures).

Events whose zenith angle exceeds ``theta_max_deg`` are rejected; the
default 60° matches the Auger SD standard analysis cut.

The efficiency ``P`` and the instrumented area ``A(t)`` may be supplied
per event to :meth:`ExposureModel.detection_probability`, but the
closed-form cumulative integral assumes ``P = 1`` and ``A(t) = A_max``
throughout — see ``TODO.md``.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from astropy.time import Time

from . import _exposure_math as _m
from .observatory import Observatory


class ExposureModel:
    """
    Directional exposure model for a fixed source direction.

    This class provides:
      - detection probability in [0, 1]
      - cumulative directional exposure, normalised or in km² sr yr
      - Bernoulli thinning of candidate event times

    Notes
    -----
    The exposure density normalised to its largest value lies in
    [0, 1], so it doubles as a per-event detection probability: keeping
    each candidate event with that probability reproduces the directional
    exposure by Monte Carlo. See :meth:`detection_probability` and
    :meth:`detect_times`.
    """

    SIDEREAL_DAY_SEC = 86164.0905
    SECONDS_PER_YEAR = 365.25 * 86400.0   # Julian year

    def __init__(
        self,
        observatory: "Observatory",
        t0: Time,
        tf: Time,
        rng: np.random.Generator,
        *,
        theta_max_deg: float = 60.0,
    ):
        """
        Parameters
        ----------
        observatory : Observatory
            Observatory supplying the latitude, which fixes the local
            geometry, and the area, taken as ``A_max``.
        t0, tf : astropy.time.Time
            Start and end of the observation interval. Must satisfy ``tf > t0``.
        rng : numpy.random.Generator
            Random stream used by the Bernoulli thinning and exposure-space
            samplers. Typically obtained from :class:`RNGManager`.
        theta_max_deg : float, optional
            Maximum zenith angle in degrees for the acceptance cut.  Events
            with zenith angle ``theta > theta_max_deg`` are rejected.
            Must be in ``(0, 90]``.  Defaults to 60°, matching the Auger SD
            standard analysis cut.
        """

        if not isinstance(observatory, Observatory):
            raise TypeError("observatory must be an instance of Observatory.")
        if not isinstance(t0, Time) or not isinstance(tf, Time):
            raise TypeError("t0 and tf must be astropy.time.Time objects.")
        if tf <= t0:
            raise ValueError("tf must be strictly later than t0.")
        if not isinstance(rng, np.random.Generator):
            raise TypeError(
                "rng must be a numpy.random.Generator. "
                "Obtain one from RNGManager.get(name) and pass it here."
            )
        if not isinstance(theta_max_deg, (int, float)) or isinstance(theta_max_deg, bool):
            raise TypeError("theta_max_deg must be a numeric value in degrees.")
        if not (0.0 < theta_max_deg <= 90.0):
            raise ValueError("theta_max_deg must be in (0, 90].")

        self.observatory = observatory
        self.t0 = t0
        self.tf = tf
        self.rng = rng
        self.theta_max_deg = float(theta_max_deg)
        self._cos_theta_max = math.cos(math.radians(self.theta_max_deg))
        self._lat_rad = math.radians(self.observatory.latitude)
        self._sidereal_rate = 2.0 * math.pi / self.SIDEREAL_DAY_SEC

        # Observation span, held in both units. The cumulative exposure
        # integral comes out in seconds; dividing by `_t_obs_sec` is what
        # makes it dimensionless, and multiplying that back by `_t_obs_yr`
        # (with an area in km^2 and a solid angle in sr) is what rebuilds a
        # physical exposure in km^2 sr yr. Using the same span on both sides
        # is what makes the two cancel exactly.
        self._t_obs_sec = float((self.tf - self.t0).to_value("sec"))
        self._t_obs_yr = self._t_obs_sec / self.SECONDS_PER_YEAR

        # Integral of cos(theta) over the field-of-view cap, in sr. Set by
        # theta_max alone, and the same at every instant, which is what
        # makes the whole-sky exposure a closed form (see `total_exposure`).
        self._cap_cos_integral = (
            math.pi * math.sin(math.radians(self.theta_max_deg)) ** 2
        )

        # Cached sidereal time at t0 (function of observatory + t0 only).
        # Used by `_continuous_hour_angle`; precomputing it here avoids
        # rebuilding an astropy `Time` object and re-evaluating
        # `sidereal_time("mean")` on every call.
        self._sidereal_t0_rad = float(
            Time(self.t0, location=self.observatory.location)
            .sidereal_time("mean")
            .rad
        )

        # Cache for `max_norm_cumul_exposure`, keyed by (RA_deg, Dec_deg).
        # The value depends only on (observatory, t0, tf, centre), all of
        # which are immutable after construction.
        self._max_exposure_cache: dict[tuple[float, float], float] = {}

    # -------------------------------------------------------------------------
    # Private input / geometry helpers
    # -------------------------------------------------------------------------


    def _as_time_array(self, t: Time) -> tuple[Time, bool]:
        """
        Coerce a possibly-scalar ``Time`` into a 1-element ``Time`` array.

        Returns the array form together with a flag indicating whether the
        original input was scalar, so callers can re-wrap the result.
        """
        if not isinstance(t, Time):
            raise TypeError("Input must be an astropy.time.Time object.")
        scalar_input = bool(getattr(t, "isscalar", np.isscalar(t)))
        t_arr = t if not scalar_input else Time([t])
        return t_arr, scalar_input

    def _check_in_interval(self, t: Time) -> None:
        """Raise if any time falls outside the observation interval."""
        if np.any(t < self.t0) or np.any(t > self.tf):
            raise ValueError("All times must satisfy t0 <= t <= tf.")

    def _validate_centre(self, centre: np.ndarray) -> tuple[float, float]:
        """
        Coerce ``centre`` to ``(RA_deg, Dec_deg)`` floats and validate
        both shape and value ranges (RA in ``[0, 360)``, Dec in
        ``[-90, 90]``). Mirrors the validation done by :class:`SkyWindow`
        and :class:`Flare` so that nonphysical centres do not silently
        propagate through trigonometric expressions.
        """
        c = np.asarray(centre, dtype=float)
        if c.size != 2:
            raise TypeError("centre must be array-like with 2 elements: [RA_deg, Dec_deg].")
        ra_deg, dec_deg = c.reshape(2,)
        ra_deg = float(ra_deg)
        dec_deg = float(dec_deg)
        if not (0.0 <= ra_deg < 360.0):
            raise ValueError(f"RA must be in [0, 360); got {ra_deg}.")
        if not (-90.0 <= dec_deg <= 90.0):
            raise ValueError(f"Dec must be in [-90, 90]; got {dec_deg}.")
        return ra_deg, dec_deg
    
    def _continuous_hour_angle(self, t: Time, ra_deg: float) -> np.ndarray:
        """
        Continuous hour angle in radians, referenced to t0.
        """
        ra_rad = np.deg2rad(ra_deg)

        h0 = self._sidereal_t0_rad - ra_rad

        dt_sec = (t - self.t0).to_value("sec")
        return h0 + 2.0 * np.pi * dt_sec / self.SIDEREAL_DAY_SEC
    
    # -------------------------------------------------------------------------
    # Detection probability and thinning
    # -------------------------------------------------------------------------

    def detection_probability(
        self,
        t: Time,
        centre: np.ndarray,
        efficiency=None,
        area=None,
    ) -> np.ndarray | float:
        """
        Detection probability ``p_det(t)`` in a given direction at a given
        time.

        The exposure per unit solid angle and unit time is::

            dE/(dOmega dt) = P(E, theta, phi, t, ...)
                             * H(theta_max - theta(t))
                             * cos(theta(t)) * A(t)          [km²]

        Every factor is bounded, so::

            max{dE/(dOmega dt)} = A_max      (P = H = cos theta = 1,
                                              A = A_max)

        and this method returns the ratio of the two::

            p_det(t) = P(t) * H(theta_max - theta(t))
                            * cos(theta(t)) * A(t)/A_max

        Lying in ``[0, 1]``, ``p_det(t)`` acts as a detection probability:
        keeping each candidate event with probability ``p_det(t)`` is the
        Bernoulli thinning performed by :meth:`detect_times`.

        The two sky-geometry factors are always applied.  ``H`` is a hard
        gate — outside the field of view the event is rejected outright —
        while ``cos theta(t)`` is the projection of the array area onto
        the arrival direction.  The two detector-state factors are opt-in:
        with neither ``efficiency`` nor ``area`` given, both are 1 and the
        result is the purely geometric ``cos theta(t)`` inside the cut,
        ranging from ``cos(theta_max)`` at the boundary to 1 at the
        zenith, and 0 outside.

        Parameters
        ----------
        t : astropy.time.Time
            Scalar or array of candidate times in ``[t0, tf]``.
        centre : array-like of shape (2,)
            Sky direction ``[RA_deg, Dec_deg]``.
        efficiency : callable or None
            Hardware and reconstruction efficiency ``P``.  Must accept
            ``t`` and return one value in ``[0, 1]`` per input time.
        area : callable or None
            Instrumented area ``A(t)``, in the units of
            ``observatory.area``, which is taken as ``A_max``.  Must accept
            ``t`` and return one value per input time, none exceeding
            ``A_max``.

        Returns
        -------
        float or numpy.ndarray
            Detection probability in ``[0, 1]``.  A float is returned when
            ``t`` is scalar, otherwise an array shaped like ``t``.
        """
        t_arr, scalar_input = self._as_time_array(t)
        self._check_in_interval(t_arr)
        ra_deg, dec_deg = self._validate_centre(centre)

        A, B = _m.geometric_coefficients(self._lat_rad, math.radians(dec_deg))
        h = self._continuous_hour_angle(t_arr, ra_deg)
        p = _m.acceptance(h, A, B, self._cos_theta_max)

        if efficiency is not None:
            eff = np.asarray(efficiency(t_arr), dtype=float)
            if eff.shape != p.shape:
                raise ValueError("efficiency(t) must return an array with the same shape as t.")
            if np.any(eff < 0.0) or np.any(eff > 1.0):
                raise ValueError("efficiency(t) must lie in [0, 1].")
            p = p * eff

        if area is not None:
            a_t = np.asarray(area(t_arr), dtype=float)
            if a_t.shape != p.shape:
                raise ValueError("area(t) must return an array with the same shape as t.")
            if np.any(a_t < 0.0) or np.any(a_t > self.observatory.area):
                raise ValueError("area(t) must lie in [0, observatory.area].")
            p = p * (a_t / self.observatory.area)

        return float(p[0]) if scalar_input else p
    
    def detect_times(
        self,
        t: Time,
        centre: np.ndarray,
        efficiency=None,
        area=None,
        return_mask: bool = False,
        return_prob: bool = False,
        return_exposure: bool = False,
    ):
        """
        Apply detector thinning to candidate times.

        Each candidate is kept with probability ``p_det(t)`` from
        :meth:`detection_probability`: a uniform ``u`` is drawn from
        ``self.rng`` and the event survives when ``u < p_det(t)``.

        Parameters
        ----------
        t : astropy.time.Time
            Candidate event times (scalar or array).
        centre : array-like of shape (2,)
            ``[RA_deg, Dec_deg]``.
        efficiency, area : callable or None
            Detector-state factors forwarded to
            :meth:`detection_probability`.
        return_mask, return_prob, return_exposure : bool
            Toggle extra outputs.

        Returns
        -------
        Time or tuple
            * Without flags: the accepted times. For *array* input this is
              a (possibly empty) ``Time`` array. For *scalar* input this
              is the scalar ``Time`` if it was accepted, or ``None``.
            * With flags: a tuple of length ``1 + n_flags``. For array
              input the extras are arrays with the same shape as ``t``
              (mask, probability) or as the accepted subset (exposure).
              For scalar input the extras are scalars: ``mask`` is a
              ``bool``, ``prob`` is a ``float``, and ``exposure`` is a
              ``float`` if accepted or ``None`` otherwise.

        Notes
        -----
        Scalar/array return shapes are kept consistent across all flag
        combinations, so callers can use the same unpacking code for either
        form.
        """

        t_arr, scalar_input = self._as_time_array(t)

        # Computed once and reused for the Bernoulli draw and the optional
        # return value, so `return_prob` costs no extra evaluation.
        p = np.asarray(
            self.detection_probability(
                t_arr, centre, efficiency=efficiency, area=area
            ),
            dtype=float,
        )
        mask = self.rng.random(size=p.shape) < p
        t_acc = t_arr[mask]

        any_flag = return_mask or return_prob or return_exposure

        if scalar_input:
            accepted = bool(mask[0])
            t_out = t_arr[0] if accepted else None

            if not any_flag:
                return t_out

            extras: list = []
            if return_mask:
                extras.append(accepted)
            if return_prob:
                extras.append(float(p[0]))
            if return_exposure:
                if accepted:
                    eps = self.norm_cumul_exposure(
                        t_arr[mask], centre=centre
                    )
                    extras.append(float(np.asarray(eps).reshape(-1)[0]))
                else:
                    extras.append(None)
            return (t_out, *extras)

        # Array input
        if not any_flag:
            return t_acc

        outputs: list = [t_acc]
        if return_mask:
            outputs.append(mask)
        if return_prob:
            outputs.append(p)
        if return_exposure:
            exp_acc = (
                np.array([], dtype=float)
                if len(t_acc) == 0
                else self.norm_cumul_exposure(t_acc, centre=centre)
            )
            outputs.append(exp_acc)
        return tuple(outputs)
    
    # -------------------------------------------------------------------------
    # Cumulative directional exposure
    # -------------------------------------------------------------------------
    
    def norm_cumul_exposure(
        self,
        t: Time,
        centre: np.ndarray,
    ) -> np.ndarray | float:
        """
        Normalised cumulative directional exposure accumulated by ``t``.

        The cumulative directional exposure towards a region of sky is::

            E(t) = ∫_window dOmega ∫_{t0}^{t} ∫_array
                       P(E, theta, phi, t, ...) cos(theta) dA dt

        in km² sr yr, evaluated by :meth:`cumul_exposure`.  This method
        drops the two outer integrals — the direction enters as the single
        point ``centre`` — and divides by the span ``T_obs = tf - t0``::

            eps_norm(t) = (1 / T_obs) ∫_{t0}^{t} w(u) du

        where ``w(u)`` is the geometric integrand, i.e.
        :meth:`detection_probability` with no detector-state factors.  The result is dimensionless and
        lies in ``[0, 1]``: what was accumulated relative to the ideal
        case, a source held at the zenith for the whole interval.
        ``T_obs`` does not depend on the direction, so values at different
        directions stay comparable and a direction the array never sees
        gives ``0.0``, not an undefined ``0/0``.

        The integral is solved analytically in the hour angle — see
        :func:`~spacetimecorr._exposure_math.cumulative_integral`.

        Parameters
        ----------
        t : astropy.time.Time
            Scalar or array of evaluation times in ``[t0, tf]``.
        centre : array-like of shape (2,)
            Sky direction ``[RA_deg, Dec_deg]``.

        Returns
        -------
        float or numpy.ndarray
            Dimensionless value(s) in ``[0, 1]``.  A float is returned when
            ``t`` is scalar, otherwise an array shaped like ``t``.
        """
        t_arr, scalar_input = self._as_time_array(t)
        self._check_in_interval(t_arr)
        ra_deg, dec_deg = self._validate_centre(centre)

        A, B = _m.geometric_coefficients(self._lat_rad, math.radians(dec_deg))
        h = np.asarray(self._continuous_hour_angle(t_arr, ra_deg), dtype=float)
        h0 = float(self._continuous_hour_angle(Time([self.t0]), ra_deg)[0])

        # `cumulative_integral` works in hour-angle; dividing by the sidereal
        # rate converts it to seconds, and by the span normalises it to [0, 1].
        out = _m.cumulative_integral(h, h0, A, B, self._cos_theta_max) / (
            self._sidereal_rate * self._t_obs_sec
        )

        return float(out[0]) if scalar_input else out

    def max_norm_cumul_exposure(self, centre: np.ndarray) -> float:
        """
        Normalised cumulative directional exposure over the whole interval,
        i.e. :meth:`norm_cumul_exposure` evaluated at ``tf``.

        Being the largest value the normalised exposure reaches for this
        direction, it is what converts an expected number of events into a
        rate per unit exposure. Results are cached per ``(RA, Dec)``.

        Parameters
        ----------
        centre : array-like of shape (2,)
            Sky direction ``[RA_deg, Dec_deg]``.

        Returns
        -------
        float
            Dimensionless, in ``[0, 1]``. Zero for a direction that never
            enters the acceptance cone.
        """
        ra_deg, dec_deg = self._validate_centre(centre)
        key = (ra_deg, dec_deg)

        cached = self._max_exposure_cache.get(key)
        if cached is not None:
            return cached

        value = float(self.norm_cumul_exposure(self.tf, centre))
        self._max_exposure_cache[key] = value
        return value

    def cumul_exposure(self, t: Time, window) -> np.ndarray | float:
        """
        Cumulative directional exposure towards ``window``, in km² sr yr.

        Evaluates the full expression::

            E(t) = ∫_window dOmega ∫_{t0}^{t} ∫_array
                       P(E, theta, phi, t, ...) cos(theta) dA dt

        by taking the normalised integral at the window centre and
        restoring the three factors :meth:`norm_cumul_exposure` leaves
        out::

            E(t) = A_max * Omega_window * T_obs * eps_norm(t, centre)

        with ``A_max = observatory.area`` in km², ``Omega_window =
        window.solid_angle`` in sr and ``T_obs = tf - t0`` in years.

        The angular extent of the window enters only through its solid
        angle: the integrand is held at its value at ``window.centre``,
        which holds while the zenith angle varies little across the window.

        Parameters
        ----------
        t : astropy.time.Time
            Scalar or array of evaluation times in ``[t0, tf]``.
        window : SkyWindow
            Region of sky; supplies ``centre`` and ``solid_angle``.

        Returns
        -------
        float or numpy.ndarray
            Cumulative directional exposure in km² sr yr.
        """
        return (
            self.observatory.area
            * window.solid_angle
            * self._t_obs_yr
            * self.norm_cumul_exposure(t, window.centre)
        )

    def total_exposure(self) -> float:
        """
        Exposure over the whole sky and the whole interval, in km² sr yr.

        Integrating the array area out and swapping the two remaining
        integrals::

            E_sky = ∫_t A(t) [ ∫_sky H(theta_max - theta) cos(theta) dOmega ] dt

        At fixed time the map from equatorial to local coordinates is a
        rigid rotation, which preserves ``dOmega``, so the inner integral
        equals the same integral over the field-of-view cap in local
        coordinates — and that is the same at every instant::

            ∫_cap cos(theta) dOmega = 2 pi ∫_0^{theta_max} cos sin dtheta
                                    = pi sin²(theta_max)

        With ``A(t) = A_max`` the time integral is then trivial::

            E_sky = A_max * T_obs * pi sin²(theta_max)

        A time-dependent ``A(t)`` would leave this structure intact: it
        carries no direction dependence, so it factors out of the solid
        angle integral and the last two factors become ``∫ A(t) dt``.

        Returns
        -------
        float
            Total exposure in km² sr yr.
        """
        return self.observatory.area * self._t_obs_yr * self._cap_cos_integral

    def relative_window_exposure(self, window) -> float:
        """
        Exposure towards ``window`` relative to the whole sky, over the
        full interval ``[t0, tf]``::

            relative = E(window) / E_sky

        ``A_max`` and ``T_obs`` are common to both and cancel, leaving::

            relative = Omega_win * eps_norm(tf, centre) / (pi sin²(theta_max))

        built from :attr:`SkyWindow.solid_angle` and
        :meth:`max_norm_cumul_exposure` at the window centre.  Cancelling
        the common factors rather than dividing :meth:`cumul_exposure` by
        :meth:`total_exposure` also keeps the result defined when
        ``observatory.area`` is zero.

        Over a full-sky tiling these values sum to 1, and exactly so:
        ``∫_sky eps_norm dOmega = pi sin²(theta_max)`` is analytic, with no
        quadrature standing between the parts and the whole.

        Parameters
        ----------
        window : SkyWindow
            Region of sky; supplies ``centre`` and ``solid_angle``.

        Returns
        -------
        float
            Dimensionless, in ``[0, 1]``.
        """
        return (
            window.solid_angle
            * self.max_norm_cumul_exposure(window.centre)
            / self._cap_cos_integral
        )

    # -------------------------------------------------------------------------
    # Exposure-space sampling
    # -------------------------------------------------------------------------

    def sample_iso_cumul_exposure(
        self,
        n_events: int,
        expected_exposure_rate: float,
    ) -> Tuple[np.ndarray, str]:
        """
        Generate sampled cumulative directional exposure values.

        This method assumes that events follow a homogeneous Poisson
        process in *exposure space* with constant rate
        ``expected_exposure_rate``. Under this assumption, the gap between
        consecutive events

            delta_exposure[i] = exposure[i+1] - exposure[i]

        follows an exponential distribution

            f(delta_exposure) = rate * exp(-rate * delta_exposure).

        The implementation draws ``n_events - 1`` such gaps and cumulative-
        sums them, anchoring ``exposure[0] = 0``. This is the direct way to
        generate Poisson-process arrival points, and it applies no upper
        cutoff: with::

            expected_exposure_rate = n_events / max_norm_cumul_exposure

        the cumulative sum has expectation close to
        ``max_norm_cumul_exposure``, but individual draws are not bounded
        by it, so some values exceed it.

        That same expression fixes the units of the result: the values come
        out normalised, because they carry the scale of
        ``1 / expected_exposure_rate`` and that rate is built from
        :meth:`max_norm_cumul_exposure`, placing them on the same scale as
        :meth:`norm_cumul_exposure`.

        Parameters
        ----------
        n_events : int
            Number of exposure values to return (i.e., number of events
            in the target sample).
        expected_exposure_rate : float
            Event rate per unit cumulative exposure. Typically defined as
            ``parent_sample.n_events / max_norm_cumul_exposure(centre)``.

        Returns
        -------
        sample : np.ndarray of shape (n_events,)
            Sorted cumulative exposure values for each event, with
            ``sample[0] == 0``.
        method_name : str
            Identifier string describing the sampling strategy.
        """

        if not isinstance(n_events, int) or isinstance(n_events, bool):
            raise TypeError("n_events must be an integer.")
        if n_events <= 0:
            raise ValueError("n_events must be > 0.")
        if not isinstance(expected_exposure_rate, (int, float)) or isinstance(expected_exposure_rate, bool):
            raise TypeError("expected_exposure_rate must be numeric.")
        if expected_exposure_rate <= 0:
            # A zero or negative *rate* is a setup error from the caller
            # (the rate is `expected_n / max_exposure`, which is meaningful
            # only when both are positive). We reject it loudly instead of
            # silently returning an empty sample.
            raise ValueError("expected_exposure_rate must be > 0.")

        sample = np.zeros(shape=n_events)
        delta_exp = self.rng.exponential(scale=1.0 / expected_exposure_rate, size=n_events - 1)
        sample[1:] = np.cumsum(delta_exp)

        return sample, "exponential_delta_exposure_method"
