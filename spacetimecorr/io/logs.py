from pathlib import Path
import logging


def setup_logger(log_path: Path, name: str = "spacetimecorr") -> logging.Logger:
    """
    Return a logger that writes to ``log_path``.

    Any existing handlers on the logger are removed first, so that calling
    ``setup_logger`` repeatedly inside the same Python process (e.g. multiple
    Monte-Carlo runs) routes log records to the new file rather than to the
    file opened on the first call.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Drop any handlers attached by a previous call so we don't keep writing
    # to the previous run's log file.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Avoid double-emission via the root logger.
    logger.propagate = False

    return logger