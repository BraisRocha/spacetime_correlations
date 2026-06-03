"""
Input/output helpers for ``spacetimecorr`` runs.

Provides utilities for creating uniquely-named output directories,
writing metadata files, and configuring per-run loggers.
"""

from .logs import setup_logger
from .output import make_run_dir, write_metadata
