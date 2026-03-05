from __future__ import annotations

from .event_sample import EventSample
from typing import Tuple
import numpy as np

"""This file contains multiple functions that apply statistical methods to the
analysis.
"""


def Lambda_estimator(
    sample: "EventSample"
    ) -> Tuple[float, float]:

    """Compute Lambda estimator and p-value (gamma survival function)."""
    if sample.has_exposure is None:
        raise RuntimeError("Directional exposure not set. Call a method to generate it first.")

    delta_exp = np.diff(np.sort(sample.dir_exposure))
    # TODO: exp_rate_exposure has to be an attribute of the sample
    Lambda = -np.sum(np.log(1.0 - np.exp(-delta_exp * sample.exp_rate_exposure)))
    p_value = scp.gamma.sf(Lambda, a=self.n_events - 1, scale=1.0)

    return Lambda, p_value