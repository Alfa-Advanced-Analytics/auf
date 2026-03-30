"""Overfit detection utilities for model evaluation.

Provides functions to detect and penalize overfitting by comparing
training and validation metric scores.

Functions:
    overfit_abs: Negative absolute difference between metrics.
    overfit_metric_minus_metric_delta: Validation metric penalized by overfit.

Examples:
    >>> from auf.metrics.overfit import overfit_abs, overfit_metric_minus_metric_delta

    >>> train_score = 0.85
    >>> valid_score = 0.72

    >>> overfit_score = overfit_abs(valid_score, train_score)
    >>> penalized = overfit_metric_minus_metric_delta(valid_score, train_score)
"""


def overfit_abs(metric_valid: float, metric_train: float) -> float:
    """Compute negative absolute difference between metrics.

    Returns the negated absolute difference to penalize overfitting
    (higher values indicate less overfitting).

    Args:
        metric_valid: Metric score on validation set.
        metric_train: Metric score on training set.

    Returns:
        Negative absolute difference (closer to 0 is better).

    Examples:
        >>> from auf.metrics import overfit_abs
        >>> score = overfit_abs(metric_valid, metric_train)

    Notes:
        Useful for optimization where higher is better.
        Returns 0 when metrics are equal (no overfit).
    """
    return -abs(metric_valid - metric_train)


def overfit_metric_minus_metric_delta(
    metric_valid: float, metric_train: float
) -> float:
    """Compute validation metric penalized by overfit gap.

    Returns validation metric reduced by half the absolute difference
    between training and validation scores.

    Args:
        metric_valid: Metric score on validation set.
        metric_train: Metric score on training set.

    Returns:
        Penalized validation score.

    Examples:
        >>> from auf.metrics import overfit_metric_minus_metric_delta
        >>> score = overfit_metric_minus_metric_delta(metric_valid, metric_train)

    Notes:
        Balances between validation performance and generalization.
        Penalty is half the overfit gap: (train - valid) / 2.
    """
    return metric_valid - abs(metric_valid - metric_train) / 2
