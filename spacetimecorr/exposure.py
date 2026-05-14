from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from astropy.time import Time

from .observatory import Observatory


class ExposureModel:
    """
    Directional exposure model for a fixed source direction.

    This class provides:
      - instantaneous acceptance a(t) in [0, 1]
      - cumulative directional exposure epsilon(t)
      - Bernoulli thinning of sampled event times

    Notes
    -----
    In the current geometric model:
        a(t) = max(0, cos theta(t))
    so it can be used directly as a detection probability.
    """

    SIDEREAL_DAY_SEC = 86164.0905

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
            Observatory whose latitude defines the local geometric acceptance.
        t0, tf : astropy.time.Time
            Start and end of the observation interval. Must satisfy ``tf > t0``.
        rng : numpy.random.Generator
            Random stream used by the Bernoulli thinning and exposure-space
            samplers. Typically obtained from :class:`RNGManager`.
        theta_max_deg : float, optional
            Maximum zenith angle in degrees for the acceptance cut.  Events
            with zenith angle ``theta > theta_max_deg`` are rejected (a = 0).
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

        # Cached sidereal time at t0 (function of observatory + t0 only).
        # Used by `_continuous_hour_angle`; precomputing it here avoids
        # rebuilding an astropy `Time` object and re-evaluating
        # `sidereal_time("mean")` on every call.
        self._sidereal_t0_rad = float(
            Time(self.t0, location=self.observatory.location)
            .sidereal_time("mean")
            .rad
        )

        # Cache for `max_directional_exposure`, keyed by (RA_deg, Dec_deg).
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
    # Instantaneous acceptance and thinning
    # -------------------------------------------------------------------------

    def instantaneous_acceptance(self, t: Time, centre: np.ndarray) -> np.ndarray | float:
        """
        Instantaneous geometric acceptance ``a(t)`` for a fixed sky direction.

        In the current model::

            a(t) = cos theta(t)   for cos(theta_max) <= cos theta(t) <= 1
                   0              otherwise

        where ``theta(t)`` is the local zenith angle of ``centre`` at the
        observatory at time ``t`` and ``theta_max`` is set at construction.
        The accepted region corresponds to ``theta(t) <= theta_max``, where
        ``a(t)`` ranges from ``cos(theta_max)`` (at the cut boundary) to 1
        (at the zenith).

        Parameters
        ----------
        t : astropy.time.Time
            Scalar or array of evaluation times. Must satisfy
            ``t0 <= t <= tf`` for every entry.
        centre : array-like of shape (2,)
            Sky direction ``[RA_deg, Dec_deg]``.

        Returns
        -------
        float or numpy.ndarray
            Acceptance value(s) in ``{0} ∪ [cos(theta_max), 1]``. A float is
            returned when ``t`` is scalar, otherwise an array of the same
            shape as ``t``.
        """
        t_arr, scalar_input = self._as_time_array(t)

        if np.any(t_arr < self.t0) or np.any(t_arr > self.tf):
            raise ValueError("All times must satisfy t0 <= t <= tf.")

        ra_deg, dec_deg = self._validate_centre(centre)

        dec_rad = np.deg2rad(dec_deg)
        lat_rad = np.deg2rad(self.observatory.latitude)

        h = self._continuous_hour_angle(t_arr, ra_deg)

        sin_lat = np.sin(lat_rad)
        cos_lat = np.cos(lat_rad)
        sin_dec = np.sin(dec_rad)
        cos_dec = np.cos(dec_rad)

        cos_theta = sin_lat * sin_dec + cos_lat * cos_dec * np.cos(h)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        out = np.where(cos_theta >= self._cos_theta_max, cos_theta, 0.0)

        return float(out[0]) if scalar_input else out
    
    def detection_probability(
        self,
        t: Time,
        centre: np.ndarray,
        efficiency=None,
    ) -> np.ndarray | float:
        """
        Detection probability ``p_det(t)`` for candidate event times.

        The probability is computed as::

            p_det(t) = a(t) * efficiency(t)

        where ``a(t)`` is the instantaneous geometric acceptance (see
        :meth:`instantaneous_acceptance`) and ``efficiency(t)`` is an
        optional time-dependent correction factor.  When no efficiency is
        provided, ``p_det(t) = a(t)``.

        Parameters
        ----------
        t : Time
            Candidate event times.
        centre : array-like
            [RA_deg, Dec_deg].
        efficiency : callable or None
            Optional time-dependent efficiency correction.  Must accept ``t``
            and return one value in ``[0, 1]`` per input time (same shape as
            ``t``).

        Returns
        -------
        array or float
            Detection probability in ``[0, 1]``.
        """
        a = np.asarray(self.instantaneous_acceptance(t, centre), dtype=float)

        if efficiency is None:
            p = a
        else:
            eff = np.asarray(efficiency(t), dtype=float)
            if eff.shape != a.shape:
                raise ValueError("efficiency(t) must return an array with the same shape as t.")
            if np.any(eff < 0.0) or np.any(eff > 1.0):
                raise ValueError("efficiency(t) must lie in [0, 1].")
            p = a * eff

        p = np.clip(p, 0.0, 1.0)

        if np.isscalar(p) or p.shape == ():
            return float(p)
        return p
    
    def acceptance_mask(
        self,
        t: Time,
        centre: np.ndarray,
        efficiency=None,
    ) -> np.ndarray | bool:
        """
        Draw a Bernoulli thinning mask for candidate event times.

        For each input time the detection probability is computed via
        :meth:`detection_probability` and a Bernoulli trial is drawn from
        ``self.rng``.

        Parameters
        ----------
        t : astropy.time.Time
            Scalar or array of candidate times.
        centre : array-like of shape (2,)
            Sky direction ``[RA_deg, Dec_deg]``.
        efficiency : callable or None, optional
            Optional time-dependent efficiency in ``[0, 1]``, see
            :meth:`detection_probability`.

        Returns
        -------
        bool or numpy.ndarray of bool
            Mask with the same shape as ``t``: ``True`` for accepted times.

        Notes
        -----
        Because this method consumes random draws from ``self.rng``, calling
        it directly and then asking for ``return_prob=True`` from
        :meth:`detect_times` would either redo the draw or recompute the
        probability. :meth:`detect_times` therefore inlines the same logic
        and is the preferred entry point for combined draws.
        """
        t_arr, scalar_input = self._as_time_array(t)
        p = np.asarray(self.detection_probability(t_arr, centre, efficiency=efficiency), dtype=float)
        mask = self.rng.random(size=p.shape) < p
        return bool(mask[0]) if scalar_input else mask
    
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

        Parameters
        ----------
        t : astropy.time.Time
            Candidate event times (scalar or array).
        centre : array-like of shape (2,)
            ``[RA_deg, Dec_deg]``.
        efficiency : callable or None
            Optional time-dependent efficiency in ``[0, 1]``.
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

        # Compute the detection probability once and reuse it for the
        # Bernoulli draw and for the optional return value, instead of
        # delegating to acceptance_mask (which would recompute it).
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
                    eps = self.cumulative_directional_exposure(
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
                else self.cumulative_directional_exposure(t_acc, centre=centre)
            )
            outputs.append(exp_acc)
        return tuple(outputs)
    
    # -------------------------------------------------------------------------
    # Cumulative directional exposure
    # -------------------------------------------------------------------------
    
    def cumulative_directional_exposure(
        self,
        t: Time,
        centre: np.ndarray,
    ) -> np.ndarray | float:
        """
        Exact cumulative directional exposure relative to ``self.t0``::

            epsilon(t) = ∫_{t0}^{t} max(0, cos(theta(u))) du

        Computed analytically using the periodic primitive of the integrand
        in the (continuous) hour-angle ``h``.

        The expression splits into three regimes depending on the relative
        sign of the geometric coefficients
        ``A = sin(lat) * sin(dec)`` and ``B = cos(lat) * cos(dec)``:

        With the zenith cut ``c = cos(theta_max)``, the integrand is
        ``cos theta(t)`` only when ``cos theta(t) >= c``, i.e. when the
        source is within the acceptance cone.  The three regimes become:

        - ``A + B <= c``  (always outside cut)  -> ``epsilon(t) = 0``,
        - ``A - B >= c``  (always inside cut)   -> the integrand is purely
          sinusoidal and integrates to a closed form,
        - otherwise (partial visibility)        -> the integrand is non-zero
          only when ``|h| < h_star`` modulo ``2π`` with
          ``h_star = arccos((c - A) / B)``; the integral is built piecewise
          per sidereal cycle.

        Setting ``c = 0`` (``theta_max = 90°``) recovers the original
        horizon-only formula.

        Parameters
        ----------
        t : astropy.time.Time
            Scalar or array of evaluation times in ``[t0, tf]``.
        centre : array-like of shape (2,)
            Sky direction ``[RA_deg, Dec_deg]``.

        Returns
        -------
        float or numpy.ndarray
            ``epsilon(t)`` in seconds (acceptance is dimensionless).
        """
        t_arr, scalar_input = self._as_time_array(t)

        if np.any(t_arr < self.t0) or np.any(t_arr > self.tf):
            raise ValueError("All times must satisfy t0 <= t <= tf.")
        
        ra_deg, dec_deg = self._validate_centre(centre)

        lat_rad = np.deg2rad(self.observatory.latitude)
        dec_rad = np.deg2rad(dec_deg)

        A = np.sin(lat_rad) * np.sin(dec_rad)
        B = np.cos(lat_rad) * np.cos(dec_rad)

        omega = 2.0 * np.pi / self.SIDEREAL_DAY_SEC
        two_pi = 2.0 * np.pi

        h = np.asarray(self._continuous_hour_angle(t_arr, ra_deg), dtype=float)
        h0 = float(self._continuous_hour_angle(Time([self.t0]), ra_deg)[0])

        c = self._cos_theta_max

        # Case 1: source never enters the acceptance cone
        if A + B <= c:
            out = np.zeros_like(h, dtype=float)

        # Case 2: source always inside the acceptance cone
        elif A - B >= c:
            out = (A * (h - h0) + B * (np.sin(h) - np.sin(h0))) / omega

        # Case 3: partial visibility — generalised cut angle
        else:
            h_star = np.arccos((c - A) / B)
            cycle_h = 2.0 * (A * h_star + B * np.sin(h_star))   # integral over one full cycle in h-space
            plateau = A * h_star + B * np.sin(h_star)

            def H(x: np.ndarray) -> np.ndarray:
                n = np.floor(x / two_pi)
                eta = x - two_pi * n   # eta in [0, 2π)

                out_h = n * cycle_h

                # The three masks cover [0, 2π) without overlap. The
                # boundary `eta == h_star` is included in `m1` (rising
                # edge) only, by virtue of `<` here vs. `>=` in `m2`;
                # similarly `eta == 2π - h_star` is included in `m3`
                # only. The integrand is continuous at both boundaries,
                # so this assignment is consistent.
                m1 = eta < h_star
                m2 = (eta >= h_star) & (eta < two_pi - h_star)
                m3 = eta >= two_pi - h_star

                out_h = out_h.astype(float)

                out_h[m1] += A * eta[m1] + B * np.sin(eta[m1])
                out_h[m2] += plateau
                out_h[m3] += cycle_h + A * (eta[m3] - two_pi) + B * np.sin(eta[m3])

                return out_h

            out = (H(h) - H(np.array([h0]))[0]) / omega

        return float(out[0]) if scalar_input else out

    def max_directional_exposure(self, centre: np.ndarray) -> float:
        """
        Return ``epsilon(tf)``, the maximum cumulative directional exposure
        accumulated over ``[t0, tf]`` for the direction ``centre``.

        Parameters
        ----------
        centre : array-like of shape (2,)
            Sky direction ``[RA_deg, Dec_deg]``.

        Returns
        -------
        float
            Total accumulated exposure (in seconds).
        """
        ra_deg, dec_deg = self._validate_centre(centre)
        key = (ra_deg, dec_deg)

        cached = self._max_exposure_cache.get(key)
        if cached is not None:
            return cached

        value = float(self.cumulative_directional_exposure(self.tf, centre))
        self._max_exposure_cache[key] = value
        return value

    # -------------------------------------------------------------------------
    # Relative directional exposure
    # -------------------------------------------------------------------------

    def relative_exposure(self, centre: np.ndarray) -> float:
        """
        Time-integrated relative directional exposure ``omega(delta)`` for a
        given sky direction, following the closed-form expression of
        Sommers (2001).

        The per-sidereal-cycle integral of ``cos(theta)`` over the visible
        portion of the sky reduces analytically to::

            omega(delta) ∝ A * h_star + B * sin(h_star)

        with the geometric coefficients::

            A      = sin(lat) * sin(dec)
            B      = cos(lat) * cos(dec)
            c      = cos(theta_max)
            h_star = arccos((c - A) / B)

        and three regimes selected by the relative magnitude of ``A``, ``B``
        and ``c``:

        - ``A + B <= c``  (always outside the cut)  -> ``omega = 0``,
        - ``A - B >= c``  (always inside the cut)   -> ``h_star = pi``,
          so ``omega = A * pi``,
        - otherwise (partial visibility)            -> the formula above
          with ``h_star = arccos((c - A) / B)``.

        Since ``omega`` is independent of RA (it averages over a sidereal
        cycle), only the declination of ``centre`` enters the computation.
        RA is validated for consistency with the rest of the API.

        Parameters
        ----------
        centre : array-like of shape (2,)
            Sky direction ``[RA_deg, Dec_deg]``.

        Returns
        -------
        float
            The relative directional exposure at ``centre``, as defined by
            the analytical Sommers expression.
        """
        _, dec_deg = self._validate_centre(centre)

        lat_rad = math.radians(self.observatory.latitude)
        dec_rad = math.radians(dec_deg)

        A = math.sin(lat_rad) * math.sin(dec_rad)
        B = math.cos(lat_rad) * math.cos(dec_rad)
        c = self._cos_theta_max

        if A + B <= c:
            return 0.0
        if A - B >= c:
            return float(A * math.pi)

        h_star = math.acos((c - A) / B)
        return float(A * h_star + B * math.sin(h_star))

    # -------------------------------------------------------------------------
    # Exposure-space sampling
    # -------------------------------------------------------------------------
    
    def sample_directional_exposure(
        self,
        n_events: int,
        expected_exposure_rate: float,
        max_dir_exposure: float,
        factor: int = 30,
    ) -> Tuple[np.ndarray, str]:
        """
        Generate sampled cumulative directional exposure values.

        This method assumes that events follow a Poisson process in
        *exposure space* with constant rate `expected_exposure_rate`. 
        Under this assumption, event exposure values are uniformly 
        distributed in [0, max_dir_exposure].

        The implementation oversamples the exposure interval by a
        multiplicative `factor` to avoid biasing the Poisson rate,
        sorts the sampled exposure values, and returns the first
        `n_events`.

        Parameters
        ----------
        n_events : int
            Number of exposure values to return (i.e., number of events
            in the target sample).

        expected_exposure_rate : float
            Event rate per unit cumulative exposure. Typically defined as
            parent_sample.n_events / max_dir_exposure.

        max_dir_exposure : float
            Maximum cumulative directional exposure epsilon(tf) for the chosen
            reference direction over the observation interval [t0, tf].

        factor : int, optional
            Oversampling factor used internally to generate a sufficiently
            large uniform exposure pool before selecting the first
            `n_events`. Normally does not need adjustment.

        Returns
        -------
        sample : np.ndarray of shape (n_events,)
            Sorted cumulative exposure values for each event.

        method_name : str
            Identifier string describing the sampling strategy.
        """
        
        if not isinstance(n_events, int) or isinstance(n_events, bool):
            raise TypeError("n_events must be an integer.")
        if n_events < 0:
            raise ValueError("n_events must be non-negative.")
        if not isinstance(factor, int) or isinstance(factor, bool):
            raise TypeError("factor must be an integer.")
        if factor <= 0:
            raise ValueError("factor must be > 0.")
        if not isinstance(expected_exposure_rate, (int, float)) or isinstance(expected_exposure_rate, bool):
            raise TypeError("expected_exposure_rate must be numeric.")
        if expected_exposure_rate <= 0:
            # A zero or negative *rate* is a setup error from the caller
            # (the rate is `expected_n / max_exposure`, which is meaningful
            # only when both are positive). We reject it loudly instead of
            # silently returning an empty sample.
            raise ValueError("expected_exposure_rate must be > 0.")
        if not isinstance(max_dir_exposure, (int, float)) or isinstance(max_dir_exposure, bool):
            raise TypeError("max_dir_exposure must be numeric.")
        if max_dir_exposure <= 0:
            # Zero (or negative) maximum directional exposure means the
            # source never enters the acceptance cone (A + B <= c).
            # Pipelines should be able to call this method without
            # special-casing that geometry, so we silently return an empty
            # sample.
            return np.empty(0, dtype=float), "free_maximum_exposure_method"

        mu = float(factor) * float(expected_exposure_rate) * float(max_dir_exposure)
        mu_expanded = int(math.floor(mu))

        if mu_expanded<= 0 or n_events == 0:
            return np.empty(0, dtype=float), "free_maximum_exposure_method"

        # Exposure interval length used for uniform sampling
        exposure_expanded = mu_expanded / float(expected_exposure_rate)

        # Draw uniform exposure values and return the first n_events in "time" order
        sample = np.sort(self.rng.uniform(0.0, exposure_expanded, size=mu_expanded))

        return sample[:n_events], "free_maximum_exposure_method"