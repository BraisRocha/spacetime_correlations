"""
Input/output helpers for ``spacetimecorr`` runs.

Provides utilities for creating uniquely-named output directories,
writing metadata files, configuring per-run loggers, and reading the
active area of the observatory and the mock samples.
"""

from .active_area import ActiveAreaProcessor, load_active_area
from .logs import setup_logger
from .mocksamples import load_mock_sample
from .output import make_run_dir, write_metadata
