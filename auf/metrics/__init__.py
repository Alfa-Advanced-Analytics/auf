"""Custom evaluation metrics for uplift modeling.

Provides metrics that extend scikit-uplift functionality with bin-weighted
averaging, stability checks, and overfit detection capabilities.

Functions:
    uplift_at_k: Computes uplift at top k percent observations by uplift.
    qini_auc_score_clip_at_k: Qini AUC score with relative uplift constraint.
    control_treatment_ones_ratios_at_k: F1-score of target ratios in top/bottom samples.
    abs_rel_uplift_growth_at_k: Relative growth of uplift in top vs bottom samples.
    bin_weighted_average_uplift: Bin-weighted average uplift score.
    calculate_control_target_averages: Control response rate deviation by bin.
    calculate_relative_uplift: Relative uplift deviation by bin.
    weighted_average_uplift_auc: Harmonic weighted AUC of control and uplift curves.
    overfit_abs: Absolute difference between train and validation metrics.
    overfit_metric_minus_metric_delta: Validation metric penalized by overfit gap.

Examples:
    >>> import numpy as np
    >>> from auf.metrics import uplift_at_k, qini_auc_score_clip_at_k
    >>> from auf.metrics import bin_weighted_average_uplift, overfit_abs

    >>> y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    >>> uplift = np.array([0.1, 0.4, 0.3, 0.8, 0.2, 0.5, 0.6, 0.7])
    >>> treatment = np.array([0, 1, 0, 1, 0, 1, 0, 1])

    >>> score = uplift_at_k(y_true, uplift, treatment, k=0.3)
    >>> overfit = overfit_abs(metric_valid=0.5, metric_train=0.7)

Notes:
    All metrics expect binary treatment vector (0=control, 1=treatment).
    Strategy parameter 'overall' uses global ranking, 'by_group' ranks within groups.
"""

from .averaged import (
    bin_weighted_average_uplift,
    calculate_control_target_averages,
    calculate_relative_uplift,
    weighted_average_uplift_auc,
)
from .by_top import (
    abs_rel_uplift_growth_at_k,
    control_treatment_ones_ratios_at_k,
    qini_auc_score_clip_at_k,
    uplift_at_k,
)
from .overfit import overfit_abs, overfit_metric_minus_metric_delta

__all__ = [
    "uplift_at_k",
    "qini_auc_score_clip_at_k",
    "control_treatment_ones_ratios_at_k",
    "abs_rel_uplift_growth_at_k",
    "bin_weighted_average_uplift",
    "calculate_control_target_averages",
    "calculate_relative_uplift",
    "weighted_average_uplift_auc",
    "overfit_abs",
    "overfit_metric_minus_metric_delta",
]
