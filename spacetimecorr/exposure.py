from __future__ import annotations

from .observatory import Observatory

import numpy as np
import math
from astropy.time import Time
from typing import Tuple

class ExposureModel:
    """
    Model the cumulative directional exposure of an observatory.

    This class provides the mapping between observation time and
    cumulative directional exposure epsilon(t) for a given sky direction,
    assuming continuous operation and purely geometric acceptance.

    Parameters
    ----------
    observatory : Observatory
        Observatory location defining latitude, longitude and altitude.
    t0 : astropy.time.Time
        Start time of the observation window.
    tf : astropy.time.Time
        End time of the observation window (must be strictly later than t0).
    rng : numpy.random.Generator
        Random generator used for exposure-space sampling.

    Notes
    -----
    - The directional exposure epsilon(t) is computed from the standard
      hour-angle formulation for a fixed equatorial direction
      (RA, Dec) at a ground-based observatory.
    - epsilon(t) is defined relative to t0, such that epsilon(t0) = 0.
    - The implementation assumes uniform detector efficiency and
      full duty cycle (no downtime or weather effects).
    - The exposure sampling method operates in cumulative exposure
      space and can be used to simulate event times under a Poisson
      process with constant rate per unit exposure.
    """

    def __init__(
        self,
        observatory: Observatory,
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

    def to_directional_exposure(
        self,
        t: Time,                 # scalar or array Time
        centre: np.ndarray,      # [RA_deg, Dec_deg]
    ) -> np.ndarray | float:
        """
        Convert times into cumulative directional exposure epsilon(t).

        Parameters
        ----------
        t : astropy.time.Time
            Scalar or array time(s).
        centre : np.ndarray, shape (2,)
            [RA_deg, Dec_deg] of the window centre.

        Returns
        -------
        np.ndarray or float
            Directional exposure values; scalar if input is scalar.
        """
        if not isinstance(t, Time):
            raise TypeError("t must be an astropy.time.Time (scalar or array).")

        c = np.asarray(centre, dtype=float)
        if c.size != 2:
            raise TypeError("centre must be array-like with 2 elements: [RA_deg, Dec_deg].")
        ra_c, dec_c = c.reshape(2,)
        ra_c_rad = np.deg2rad(float(ra_c))
        dec_c_rad = np.deg2rad(float(dec_c))

        scalar_input = bool(getattr(t, "isscalar", np.isscalar(t)))
        t_arr = t if not scalar_input else Time([t])

        lat_rad = np.deg2rad(self.observatory.latitude)
        cosl0 = float(np.cos(lat_rad))
        sinl0 = float(np.sin(lat_rad))

        cosDec = float(np.cos(dec_c_rad))
        sinDec = float(np.sin(dec_c_rad))

        # Local sidereal time at t and at t0
        t_loc = t_arr.copy()
        t_loc.location = self.observatory.location
        lst = t_loc.sidereal_time("mean").rad
        h = lst - ra_c_rad

        t0_loc = self.t0.copy()
        t0_loc.location = self.observatory.location
        h0 = float(t0_loc.sidereal_time("mean").rad - ra_c_rad)

        # Ensure continuity for array inputs (important when crossing 2pi)
        if h.size > 1:
            h = np.unwrap(h)

        # Cumulative-style expression relative to t0 via h0
        dir_exposure = cosl0 * cosDec * (np.sin(h) - np.sin(h0)) + (h - h0) * sinl0 * sinDec

        return float(dir_exposure[0]) if scalar_input else dir_exposure
    
    @property
    def max_directional_exposure(self, centre: np.ndarray) -> float:
        """
        Return epsilon(tf), interpreted as maximum cumulative exposure over [t0, tf].
        """
        return float(self.to_directional_exposure(self.tf, centre))
    
    @property
    def min_directional_exposure(self, centre: np.ndarray) -> float:
        """
        Return epsilon(t0), interpreted as the minimum cumulative exposure over [t0, tf].
        """
        return float(self.to_directional_exposure(self.t0, centre))

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