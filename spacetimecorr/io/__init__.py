"""
Input/output helpers for ``spacetimecorr`` runs.

Provides utilities for creating uniquely-named output directories,
writing metadata files, configuring per-run loggers, and reading the
active area of the observatory.
"""

from .active_area import ActiveAreaProcessor, load_active_area
from .logs import setup_logger
from .output import make_run_dir, write_metadata
