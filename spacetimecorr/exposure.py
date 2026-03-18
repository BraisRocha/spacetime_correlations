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
    ):

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

        self.observatory = observatory
        self.t0 = t0
        self.tf = tf
        self.rng = rng

    # -------------------------------------------------------------------------
    # Private input / geometry helpers
    # -------------------------------------------------------------------------


    def _as_time_array(self, t: Time) -> tuple[Time, bool]:
        if not isinstance(t, Time):
            raise TypeError("Input must be an astropy.time.Time object.")
        scalar_input = bool(getattr(t, "isscalar", np.isscalar(t)))
        t_arr = t if not scalar_input else Time([t])
        return t_arr, scalar_input

    def _validate_centre(self, centre: np.ndarray) -> tuple[float, float]:
        c = np.asarray(centre, dtype=float)
        if c.size != 2:
            raise TypeError("centre must be array-like with 2 elements: [RA_deg, Dec_deg].")
        ra_deg, dec_deg = c.reshape(2,)
        return float(ra_deg), float(dec_deg)
    
    def _continuous_hour_angle(self, t: Time, ra_deg: float) -> np.ndarray:
        """
        Continuous hour angle in radians, referenced to t0.
        """
        ra_rad = np.deg2rad(ra_deg)

        t0_loc = Time(self.t0, location=self.observatory.location)
        h0 = float(t0_loc.sidereal_time("mean").rad - ra_rad)

        dt_sec = (t - self.t0).to_value("sec")
        return h0 + 2.0 * np.pi * dt_sec / self.SIDEREAL_DAY_SEC
    
    # -------------------------------------------------------------------------
    # Instantaneous acceptance and thinning
    # -------------------------------------------------------------------------

    def instantaneous_acceptance(self, t: Time, centre: np.ndarray) -> np.ndarray | float:
        """
        Instantaneous geometric acceptance a(t) in [0, 1].

        In the current model:
            a(t) = max(0, cos z(t))
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
        out = np.maximum(0.0, cos_theta)

        return float(out[0]) if scalar_input else out
    
    def detection_probability(
        self,
        t: Time,
        centre: np.ndarray,
        efficiency=None,
    ) -> np.ndarray | float:
        """
        Detection probability p_det(t).

        Parameters
        ----------
        t : Time
            Candidate event times.
        centre : array-like
            [RA_deg, Dec_deg].
        efficiency : callable or None
            Optional multiplicative factor. Must return values in [0, 1]
            with the same shape as t.
            Final probability is:
                p_det(t) = a(t) * efficiency(t)

        Returns
        -------
        array or float
            Detection probability in [0, 1].
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
        Bernoulli thinning mask for candidate event times.
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
        t : Time
            Candidate event times.
        centre : array-like
            [RA_deg, Dec_deg].
        efficiency : callable or None
            Optional extra time-dependent efficiency in [0, 1].
        return_mask, return_prob, return_exposure : bool
            Control extra outputs.

        Returns
        -------
        Time or tuple
            Accepted times, optionally with mask / probabilities / exposures.
        """

        t_arr, scalar_input = self._as_time_array(t)

        mask = self.acceptance_mask(t_arr, centre, efficiency=efficiency)
        t_acc = t_arr[mask]

        outputs = [t_acc]

        if return_mask:
            outputs.append(mask)

        if return_prob:
            p = np.asarray(
                self.detection_probability(t_arr, centre, efficiency=efficiency),
                dtype=float,
            )
            outputs.append(p)

        if return_exposure:
            exp_acc = (
                np.array([], dtype=float)
                if len(t_acc) == 0
                else self.cumulative_directional_exposure(t_acc, centre=centre)
            )
            outputs.append(exp_acc)

        if len(outputs) == 1:
            if scalar_input:
                return t_acc[0] if mask[0] else None
            return t_acc

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
        Exact cumulative directional exposure relative to self.t0:

            epsilon(t) = ∫_{t0}^{t} max(0, cos(theta(u))) du

        using the analytic periodic primitive.
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

        # Case 1: always invisible
        if A + B <= 0.0:
            out = np.zeros_like(h, dtype=float)

        # Case 2: always visible
        elif A - B >= 0.0:
            out = (A * (h - h0) + B * (np.sin(h) - np.sin(h0))) / omega

        # Case 3: partial visibility
        else:
            h_star = np.arccos(-A / B)
            cycle_h = 2.0 * (A * h_star + B * np.sin(h_star))   # integral over one full cycle in h-space
            plateau = A * h_star + B * np.sin(h_star)

            def H(x: np.ndarray) -> np.ndarray:
                n = np.floor(x / two_pi)
                eta = x - two_pi * n   # eta in [0, 2π)

                out_h = n * cycle_h

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
        Return epsilon(tf), interpreted as the maximum cumulative directional
        exposure over [t0, tf].
        """
        return float(self.cumulative_directional_exposure(self.tf, centre))
    
    # -------------------------------------------------------------------------
    # Exposure-space sampling
    # -------------------------------------------------------------------------
    
    def sample_directional_exposure(
        self,
        n_events: int,
        exp_rate_exposure: float,
        max_dir_exposure: float,
        factor: int = 30,
    ) -> Tuple[np.ndarray, str]:
        """
        Generate sampled cumulative directional exposure values.

        This method assumes that events follow a Poisson process in
        *exposure space* with constant rate `exp_rate_exposure`. 
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

        exp_rate_exposure : float
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
        
        if not isinstance(n_events, int) or isinstance(n_events, bool) or n_events < 0:
            raise TypeError("n_events must be a non-negative integer.")
        if not isinstance(factor, int) or isinstance(factor, bool) or factor <= 0:
            raise TypeError("factor must be a positive integer.")
        if not isinstance(exp_rate_exposure, (int, float)) or isinstance(exp_rate_exposure, bool):
            raise TypeError("exp_rate_exposure must be numeric.")
        if exp_rate_exposure <= 0:
            raise ValueError("exp_rate_exposure must be > 0.")
        if not isinstance(max_dir_exposure, (int, float)) or isinstance(max_dir_exposure, bool):
            raise TypeError("max_dir_exposure must be numeric.")
        if max_dir_exposure <= 0:
            return np.empty(0, dtype=float), "free_maximum_exposure_method"

        mu = float(factor) * float(exp_rate_exposure) * float(max_dir_exposure)
        mu_expanded = int(math.floor(mu))

        if mu_expanded<= 0 or n_events == 0:
            return np.empty(0, dtype=float), "free_maximum_exposure_method"

        # Exposure interval length used for uniform sampling
        exposure_expanded = mu_expanded / float(exp_rate_exposure)

        # Draw uniform exposure values and return the first n_events in "time" order
        sample = np.sort(self.rng.uniform(0.0, exposure_expanded, size=mu_expanded))

        return sample[:n_events], "free_maximum_exposure_method"