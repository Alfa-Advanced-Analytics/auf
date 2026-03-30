"""Predefined evaluation metrics for AUF.

This package exports pre-configured metric dictionaries ready for immediate
use in uplift experiments without need to define evaluation strategies.

Constants:
    METRICS: Dictionary of pre-configured uplift metrics.
    RANDOM_STATE: Fixed seed (42) for reproducibility across all models.
    BOOTSTRAP_REPEATS: Default number (300) for bootstrap confidence intervals.

Examples:
    >>> from auf.constants import METRICS
    >>> # Access metric by key and calculate
    >>> uplift = METRICS['uplift@10'](y_test, predictions, treatment_test)
    >>> qini = METRICS['qini_auc'](y_test, predictions, treatment_test)
"""

from .metrics import METRICS
from .numbers import BOOTSTRAP_REPEATS, RANDOM_STATE

__all__ = [
    "METRICS",
    "RANDOM_STATE",
    "BOOTSTRAP_REPEATS",
]
