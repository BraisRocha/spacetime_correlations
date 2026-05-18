"""
spacetimecorr
=============

Tools for simulating and analysing spatiotemporal correlations in
ultra-high-energy cosmic-ray (UHECR) arrival directions.

The package exposes:

- ``Observatory`` / ``SkyWindow`` / ``ExposureModel``
    Geometry and exposure primitives.
- ``EventSample`` / ``Flare``
    Event-level data containers and flare generators.
- ``RNGManager``
    Reproducible, named random-number streams.
- ``lambda_*`` / ``lambda_estimator`` / ``empirical_p_values``
    Anisotropy estimators and their distributions.

Recommended usage
-----------------
Import the package as ``stc`` in scripts and notebooks::

    import spacetimecorr as stc
"""

from .observatory import Observatory
from .skywindow import SkyWindow
from .exposure import ExposureModel
from .event_sample import EventSample
from .flare import Flare
from .statistics import (
    lambda_conditional_pdf,
    lambda_conditional_cdf,
    lambda_conditional_sf,
    lambda_conditional_logsf,
    lambda_conditional_sigma,
    lambda_conditional_pvalue,
    lambda_conditional_pvalue_and_sigma,
    lambda_conditional_rvs,
    lambda_marginal_pdf,
    lambda_marginal_sf,
    lambda_marginal_logsf,
    lambda_marginal_sigma,
    lambda_marginal_pvalue,
    lambda_marginal_pvalue_and_sigma,
    lambda_marginal_rvs,
    lambda_estimator,
    empirical_p_values,
    plot_lambda_joint_heatmap,
    pvalue_to_sigma,
    sigma_to_pvalue,
)
from .rng import RNGManager

__all__ = [
    "Observatory",
    "SkyWindow",
    "ExposureModel",
    "EventSample",
    "Flare",
    "RNGManager",
    "lambda_conditional_pdf",
    "lambda_conditional_cdf",
    "lambda_conditional_sf",
    "lambda_conditional_logsf",
    "lambda_conditional_sigma",
    "lambda_conditional_pvalue",
    "lambda_conditional_pvalue_and_sigma",
    "lambda_conditional_rvs",
    "lambda_marginal_pdf",
    "lambda_marginal_sf",
    "lambda_marginal_logsf",
    "lambda_marginal_sigma",
    "lambda_marginal_pvalue",
    "lambda_marginal_pvalue_and_sigma",
    "lambda_marginal_rvs",
    "lambda_estimator",
    "empirical_p_values",
    "plot_lambda_joint_heatmap",
    "pvalue_to_sigma",
    "sigma_to_pvalue",
]
