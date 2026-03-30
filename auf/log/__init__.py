"""Logging and progress display utilities for AUF.

Provides a configured logger and a manual progress bar wrapper designed
for uplift modeling workflows.

Functions:
    get_logger: Returns a configured ``logging.Logger`` instance.

Classes:
    ManualProgressBar: Wrapper around ``tqdm`` with verbosity control.

Examples:
    >>> from auf.log import get_logger, ManualProgressBar
    >>> from sklift.models import SoloModel
    >>> from catboost import CatBoostClassifier

    >>> # Logging setup
    >>> logger = get_logger(verbosity=1)
    >>> logger.info("Starting training process")

    >>> # Progress tracking
    >>> with ManualProgressBar(total=100, description="Training") as bar:
    ...     for i in range(100):
    ...         # ... model training step ...
    ...         bar.update(1)

Notes:
    Verbosity levels: 0=WARNING, 1=INFO, 2=DEBUG.
    ManualProgressBar is disabled automatically when verbosity=0.
"""

from .log import ManualProgressBar, get_logger

__all__ = ["get_logger", "ManualProgressBar"]
