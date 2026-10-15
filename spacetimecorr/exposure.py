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
- the exposure per solid angle as a function of declination
  (:meth:`ExposureModel.spatial_exposure`) and the thinning of candidate
  directions by it (:meth:`ExposureModel.detect_directions`), used to
  draw whole-sky samples,
- helpers for sampling event times under the directional-exposure
  distribution (used by :class:`~spacetimecorr.flare.Flare` and by
  :class:`~spacetimecorr.event_sample.EventSample` when assigning
  exposures).

Events whose zenith angle exceeds ``theta_max_deg`` are rejected; the
default 60° matches the Auger SD standard analysis cut.

The area ``A(t)`` is taken from the observatory: constant, or piecewise
constant when ``observatory.area`` is given as steps. The efficiency ``P``
may be supplied per event to :meth:`ExposureModel.detection_probability`;
the cumulative exposure assumes ``P = 1``.
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
            geometry, and the area ``A(t)``, constant or as steps, whose
            largest value ``observatory.max_area`` is taken as ``A_max``.
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

        # Active area A(t) as steps over [t0, tf]: the step edges in seconds
        # since t0 (from 0 to T_obs) and A on each step, in km^2. A constant
        # area is a single step over the run: given as starting at t0 and,
        # for the end-of-data check in `_steps_in_run`, again at tf.
        area = self.observatory.area
        if not isinstance(area, tuple):
            area = (Time([self.t0, self.tf]), np.array([area, area]))
        self._area_edges_sec, self._area_km2 = self._steps_in_run(*area)

        # Time average <A> of the active area over the run, in km^2. The
        # whole-sky exposure is pi sin^2(theta_max) <A> T_obs.
        self._mean_area_km2 = float(
            np.sum(self._area_km2 * np.diff(self._area_edges_sec)) / self._t_obs_sec
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

        # Exposure per solid angle over the whole run, dE/dOmega(tf), keyed
        # by (RA_deg, Dec_deg). Internal, in km^2 yr: the public methods
        # either normalise it or multiply it by the solid angle of a window.
        # Filled by `_directional_exposure`, which reaches tf on every call.
        self._max_exposure_cache: dict[tuple[float, float], float] = {}

        # Largest `spatial_exposure` over the sky, the normalisation of the
        # direction thinning. Found on a 0.01 deg grid that includes both
        # poles, since the maximum can sit at one of them (at the Auger site
        # it is the south pole, which never leaves the field of view).
        self._max_spatial_exposure = float(
            np.max(self.spatial_exposure(np.linspace(-90.0, 90.0, 18001)))
        )

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

    def _steps_in_run(
        self, start_times: Time, area: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Steps of ``A(t)`` over ``[t0, tf]``: the step edges in seconds since
        ``t0`` (from 0 to ``T_obs``) and ``A`` on each step, in km^2.

        The steps come from ``observatory.area``, already validated there.
        """
        starts = (start_times - self.t0).to_value("sec")
        if starts[0] > 0.0:
            raise ValueError("The active area must start at or before t0.")
        # The last step has no recorded end: past its start there is no data.
        if starts[-1] < self._t_obs_sec:
            raise ValueError("tf is after the last start time of the active area.")

        # Steps overlapping [t0, tf]: from the one holding t0 to the last one
        # starting before tf, with the outer edges clipped to t0 and tf.
        first = np.searchsorted(starts, 0.0, side="right") - 1
        last = np.searchsorted(starts, self._t_obs_sec, side="left")
        edges_sec = np.concatenate(([0.0], starts[first + 1:last], [self._t_obs_sec]))
        return edges_sec, area[first:last]

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

        ``H`` is a hard gate — outside the field of view the event is
        rejected outright — while ``cos theta(t)`` is the projection of the
        array area onto the arrival direction.  ``A(t)`` is the area of the
        observatory, with ``A_max = observatory.max_area``: the factor is 1
        for a constant area, and 0 while the array is off (e.g. in a bad
        period).  These three factors are always applied.  ``P`` is applied
        only when given as ``efficiency``; otherwise it is 1.

        Parameters
        ----------
        t : astropy.time.Time
            Scalar or array of candidate times in ``[t0, tf]``.
        centre : array-like of shape (2,)
            Sky direction ``[RA_deg, Dec_deg]``.
        efficiency : callable or None
            Hardware and reconstruction efficiency ``P``.  Must accept
            ``t`` and return one value in ``[0, 1]`` per input time.

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

        # A(t)/A_max from the step of the active area holding each time.
        dt_sec = (t_arr - self.t0).to_value("sec")
        k = np.searchsorted(self._area_edges_sec, dt_sec, side="right") - 1
        k = np.clip(k, 0, self._area_km2.size - 1)
        p = p * (self._area_km2[k] / self.observatory.max_area)

        return float(p[0]) if scalar_input else p
    
    def detect_times(
        self,
        t: Time,
        centre: np.ndarray,
        efficiency=None,
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
        efficiency : callable or None
            Hardware and reconstruction efficiency ``P``, forwarded to
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
            self.detection_probability(t_arr, centre, efficiency=efficiency),
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

        The exposure per solid angle towards ``centre`` accumulated from
        ``t0`` to ``t`` is::

            dE/dOmega(t) = ∫_{t0}^{t} P(E, theta, phi, t', ...)
                                      * H(theta_max - theta(t'))
                                      * cos(theta(t')) * A(t') dt'   [km² yr]

        evaluated with ``P = 1`` by :meth:`_directional_exposure`.  Every
        factor is bounded, so::

            max{dE/dOmega} = A_max * T_obs      (P = H = cos theta = 1,
                                                 A = A_max over the whole run)

        with ``A_max = observatory.max_area`` and ``T_obs = tf - t0``, and
        this method returns the ratio of the two::

            eps_norm(t) = dE/dOmega(t) / (A_max * T_obs)

        The result is dimensionless and lies in ``[0, 1]``.  The
        normalisation does not depend on the direction, so values at
        different directions stay comparable, and a direction the array
        never sees gives ``0.0``.

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

        out = self._directional_exposure(t_arr, ra_deg, dec_deg) / (
            self.observatory.max_area * self._t_obs_yr
        )
        return float(out[0]) if scalar_input else out

    def _directional_exposure(
        self, t: Time, ra_deg: float, dec_deg: float,
    ) -> np.ndarray:
        """
        Exposure per solid angle towards ``(ra_deg, dec_deg)`` accumulated
        from ``t0`` to each ``t``, in km² yr::

            dE/dOmega(t) = ∫_{t0}^{t} P(E, theta, phi, t', ...)
                                      * H(theta_max - theta(t'))
                                      * cos(theta(t')) * A(t') dt'

        with ``P = 1``.  ``A(t')`` is constant on each step of the active
        area, so the integral is the sum over the steps of ``A`` times the
        analytic integral of ``H cos(theta)`` — see
        :func:`~spacetimecorr._exposure_math.area_weighted_integral`.

        The integral is carried to ``tf`` in the same pass, and that value,
        ``dE/dOmega(tf)``, is stored in ``_max_exposure_cache``.

        ``t`` must be a ``Time`` array within ``[t0, tf]``.
        """
        A, B = _m.geometric_coefficients(self._lat_rad, math.radians(dec_deg))
        h = np.asarray(self._continuous_hour_angle(t, ra_deg), dtype=float)
        h0 = self._sidereal_t0_rad - math.radians(ra_deg)
        h_edges = h0 + self._sidereal_rate * self._area_edges_sec

        integral, total = _m.area_weighted_integral(
            h, h0, A, B, self._cos_theta_max, h_edges, self._area_km2,
        )

        # The integral is in km² rad of hour angle; dividing by the sidereal
        # rate converts it to km² s, and then to km² yr.
        to_km2_yr = 1.0 / (self._sidereal_rate * self.SECONDS_PER_YEAR)
        self._max_exposure_cache[(ra_deg, dec_deg)] = total * to_km2_yr
        return integral * to_km2_yr

    def _directional_exposure_tf(self, ra_deg: float, dec_deg: float) -> float:
        """
        ``dE/dOmega(tf)`` towards ``(ra_deg, dec_deg)``, in km² yr: the
        exposure per solid angle over the whole run, from the cache, or
        computed and stored there on the first call for the direction.
        """
        key = (ra_deg, dec_deg)
        if key not in self._max_exposure_cache:
            self._directional_exposure(Time([self.tf]), ra_deg, dec_deg)
        return self._max_exposure_cache[key]

    def max_norm_cumul_exposure(self, centre: np.ndarray) -> float:
        """
        Normalised cumulative directional exposure over the whole interval,
        i.e. :meth:`norm_cumul_exposure` evaluated at ``tf``.

        Being the largest value the normalised exposure reaches for this
        direction, it is what converts an expected number of events into a
        rate per unit exposure. ``dE/dOmega(tf)`` is cached per
        ``(RA, Dec)``, also when :meth:`norm_cumul_exposure` was called
        first for the same direction, so it is computed once.

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
        return self._directional_exposure_tf(ra_deg, dec_deg) / (
            self.observatory.max_area * self._t_obs_yr
        )

    def cumul_exposure(self, t: Time, window) -> np.ndarray | float:
        """
        Cumulative directional exposure towards ``window`` accumulated from
        ``t0`` to ``t``, in km² sr yr::

            E(t) = ∫_window dE/dOmega(t) dOmega

        where ``dE/dOmega(t)`` is the exposure per solid angle, in km² yr,
        defined in :meth:`norm_cumul_exposure`.  Holding it at its value at
        ``window.centre`` gives::

            E(t) = Omega_win * dE/dOmega(t, centre)

        with ``Omega_win = window.solid_angle`` in sr.  This holds while the
        zenith angle varies little across the window.

        Parameters
        ----------
        t : astropy.time.Time
            Scalar or array of evaluation times in ``[t0, tf]``.
        window : SkyWindow
            Region of sky; supplies ``centre`` and ``solid_angle``.

        Returns
        -------
        float or numpy.ndarray
            Cumulative directional exposure in km² sr yr.  A float is
            returned when ``t`` is scalar, otherwise an array shaped like
            ``t``.
        """
        t_arr, scalar_input = self._as_time_array(t)
        self._check_in_interval(t_arr)
        ra_deg, dec_deg = self._validate_centre(window.centre)

        out = window.solid_angle * self._directional_exposure(t_arr, ra_deg, dec_deg)
        return float(out[0]) if scalar_input else out

    def total_exposure(self) -> float:
        """
        Exposure over the whole sky and the whole interval, in km² sr yr.

        With ``P = 1``, integrating the array area out and swapping the two
        remaining integrals::

            E_sky = ∫_{t0}^{tf} A(t) [ ∫_sky H(theta_max - theta) cos(theta) dOmega ] dt

        At fixed time the map from equatorial to local coordinates is a
        rigid rotation, which preserves ``dOmega``, so the inner integral
        equals the same integral over the field-of-view cap in local
        coordinates — and that is the same at every instant::

            ∫_cap cos(theta) dOmega = 2 pi ∫_0^{theta_max} cos(theta) sin(theta) dtheta
                                    = pi sin²(theta_max)

        The time integral is then that of the area alone::

            E_sky = pi sin²(theta_max) ∫_{t0}^{tf} A(t) dt
                  = pi sin²(theta_max) <A> T_obs

        with ``<A>`` the time average of the area over the run, in km²
        (``A`` itself when the area is constant).

        Returns
        -------
        float
            Total exposure in km² sr yr.
        """
        return self._cap_cos_integral * self._mean_area_km2 * self._t_obs_yr

    def relative_window_exposure(self, window) -> float:
        """
        Exposure towards ``window`` relative to the whole sky, over the
        full interval ``[t0, tf]``::

            relative = E(window) / E_sky

        with ``E(window)`` from :meth:`cumul_exposure` at ``tf`` and
        ``E_sky`` from :meth:`total_exposure`::

            relative = Omega_win * dE/dOmega(tf, centre)
                       / (pi sin²(theta_max) <A> T_obs)

                     = Omega_win ∫ A(t) H(theta_max - theta(t)) cos(theta(t)) dt
                       / (pi sin²(theta_max) ∫ A(t) dt)

        When the area is constant, ``A`` comes out of both time integrals
        and cancels, leaving a purely geometric ratio::

            relative = Omega_win ∫ H(theta_max - theta(t)) cos(theta(t)) dt
                       / (pi sin²(theta_max) T_obs)

        Over a full-sky tiling these values sum to 1, up to the error of
        holding each window at its centre.

        Parameters
        ----------
        window : SkyWindow
            Region of sky; supplies ``centre`` and ``solid_angle``.

        Returns
        -------
        float
            Dimensionless, in ``[0, 1]``.
        """
        ra_deg, dec_deg = self._validate_centre(window.centre)
        return (
            window.solid_angle
            * self._directional_exposure_tf(ra_deg, dec_deg)
            / self.total_exposure()
        )

    # -------------------------------------------------------------------------
    # Exposure per solid angle and direction thinning
    # -------------------------------------------------------------------------

    def spatial_exposure(self, dec_deg: np.ndarray) -> np.ndarray:
        """
        Normalised exposure per solid angle over the whole interval, as a
        function of declination.

        The exposure per solid angle towards a direction is::

            dE/dOmega = ∫_{t0}^{tf} P(E, theta, phi, t, ...) A(t)
                            H(theta_max - theta(t)) cos(theta(t)) dt

        Its largest possible value, ``max(dE/dOmega) = A_max * T_obs``, is
        that of a direction held at the zenith for the whole interval.
        Dividing by it, with ``P = 1`` and ``A(t) = A_max``, over whole
        sidereal days (see ``TODO.md``), leaves a function of declination
        only::

            spatial_exposure(dec) = [cos(lat) cos(dec) sin(h_m)
                                     + h_m sin(lat) sin(dec)] / pi

            cos(h_m) = (cos(theta_max) - sin(lat) sin(dec))
                       / (cos(lat) cos(dec))

        ``h_m`` is the half-width, in hour angle, of the arc of the
        direction's daily circle inside the field of view. It is clipped
        to ``[0, pi]``: 0 for a direction that never enters, and at most
        pi, half a sidereal day, when the whole circle lies inside.

        The normalisation is that of :meth:`norm_cumul_exposure`, and the
        integral over the sky is ``pi sin²(theta_max)``, as in
        :meth:`relative_window_exposure`.

        Parameters
        ----------
        dec_deg : array-like
            Declinations in degrees, in ``[-90, 90]``.

        Returns
        -------
        numpy.ndarray
            Dimensionless values in ``[0, 1]``, shaped like ``dec_deg``.
        """
        dec_deg = np.asarray(dec_deg, dtype=float)
        if np.any(np.abs(dec_deg) > 90.0):
            raise ValueError("Dec must be in [-90, 90].")

        dec = np.deg2rad(dec_deg)
        a = math.sin(self._lat_rad) * np.sin(dec)
        b = math.cos(self._lat_rad) * np.cos(dec)
        c = self._cos_theta_max

        # At the poles b = 0 and the division is undefined, but there one
        # of the two clipped cases always holds, so `np.where` discards it.
        with np.errstate(divide="ignore", invalid="ignore"):
            h_m = np.arccos(np.clip((c - a) / b, -1.0, 1.0))
        h_m = np.where(a + b <= c, 0.0, np.where(a - b >= c, np.pi, h_m))

        return (b * np.sin(h_m) + h_m * a) / np.pi

    def detect_directions(
        self,
        ra: np.ndarray,
        dec: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply exposure thinning to candidate directions.

        Each candidate is kept with probability
        ``spatial_exposure(dec) / max_sky spatial_exposure``: a uniform
        ``u`` is drawn from ``self.rng`` and the direction survives when
        ``u`` is below that ratio. Applied to isotropic candidates, the
        survivors are distributed on the sky as the exposure per solid
        angle.

        Parameters
        ----------
        ra, dec : array-like
            Candidate directions in degrees, same shape.

        Returns
        -------
        ra, dec : numpy.ndarray
            The accepted directions, in degrees.
        """
        ra = np.asarray(ra, dtype=float)
        dec = np.asarray(dec, dtype=float)
        if ra.shape != dec.shape:
            raise ValueError("ra and dec must have the same shape.")

        p = self.spatial_exposure(dec) / self._max_spatial_exposure
        mask = self.rng.random(size=p.shape) < p
        return ra[mask], dec[mask]

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
