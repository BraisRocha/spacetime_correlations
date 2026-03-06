from __future__ import annotations

from typing import Tuple

import numpy as np
import scipy.stats as scp

from .event_sample import EventSample


def lambda_estimator(sample: EventSample) -> Tuple[float, float]:
    """
    Compute the Lambda test statistic and its p-value.

    The statistic is computed from the spacings of the sorted directional exposure
    values. The p-value is obtained from the Gamma survival function.

    Parameters
    ----------
    sample:
        EventSample instance with directional exposure already computed.

    Returns
    -------
    (lambda_stat, p_value):
        Lambda statistic and the corresponding p-value.
    """

    if not sample.has_exposure:
        raise RuntimeError(
            "Directional exposure not set. Call a method to generate it first."
        )

    if sample.n_events < 2:
        raise ValueError("Need at least 2 events to compute Delta exposure.")

    # Spacings of sorted exposure values
    delta_exp = np.diff(np.sort(sample.dir_exposure))

    # Computation of the Lambda estimator
    lambda_stat = float(-np.sum(np.log(1.0 - np.exp(-delta_exp * sample.exp_rate_exposure))))

    # Gamma survival function
    p_value = float(scp.gamma.sf(lambda_stat, a=sample.n_events - 1, scale=1.0))

    return lambda_stat, p_value

def theoretical_lambda_estimator(
    sample: EventSample,
    n_simulations: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate theoretical Lambda statistics and corresponding p-values.

    Parameters
    ----------
    sample
        EventSample instance used to determine the Gamma distribution parameters.
    n_simulations
        Number of simulated Lambda values to generate.

    Returns
    -------
    lambda_stat : np.ndarray
        Simulated Lambda statistics.
    p_values : np.ndarray
        Corresponding p-values computed from the Gamma survival function.
    """

    shape = sample.n_events - 1

    lambda_stat = scp.gamma.rvs(a=shape, scale=1.0, size=1)
    p_values = scp.gamma.sf(lambda_stat, a=shape, scale=1.0)

    return lambda_stat, p_values
