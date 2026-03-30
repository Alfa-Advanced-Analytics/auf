"""Pre-configured uplift evaluation metrics with fixed parameters.

This module provides METRICS dictionary containing all pre-configured metric
functions for uplift evaluation. Each metric uses functools.partial to lock
optimal parameters (strategy, k-values, binning) for consistent experiments.

Metrics cover:
- uplift@k (absolute/relative) for k=5, 10, 15, .., 40 percentiles
- weighted uplift with 2-20 bins (stable/unstable variants)
- control/treatment target ratios
- uplift growth metrics (absolute, relative, combined)
- Qini AUC scores

Constants:
    METRICS: Master dictionary of all uplift metrics. Keys follow patterns:
        'uplift@10', 'weighted_uplift_5_bins', 'rel_uplift_growth@15',
        'control_treatment_ones_ratios@20', 'qini_auc', etc.

Examples:
    >>> from auf.constants.metrics import METRICS
    >>> uplift_10 = METRICS['uplift@10'](y_true, pred, treatment)
    >>> weighted = METRICS['weighted_uplift_5_bins'](y_true, pred, treatment)
    >>> qini = METRICS['qini_auc'](y_true, pred, treatment)

Notes:
    For custom parameters, use original functions from auf.metrics.
"""

from functools import partial
from itertools import chain

from sklift.metrics import qini_auc_score

from ..metrics import (
    abs_rel_uplift_growth_at_k,
    bin_weighted_average_uplift,
    control_treatment_ones_ratios_at_k,
    qini_auc_score_clip_at_k,
    uplift_at_k,
)

METRICS_UPLIFT_AT_K = dict(
    {
        f"uplift@{k:02d}": partial(
            uplift_at_k,
            strategy="overall",
            k=k / 100,
            output_transform=lambda x, y: x - y,
        )
        for k in range(5, 41, 5)
    }
)

METRICS_UPLIFT_REL_AT_K = dict(
    {
        f"uplift_rel@{k:02d}": partial(
            uplift_at_k,
            strategy="overall",
            k=k / 100,
            output_transform=lambda x, y: x / y - 1,
        )
        for k in range(5, 41, 5)
    }
)

METRICS_WEIGHTED_UPLIFT = dict(
    {
        f"weighted_uplift_{k}_bins": partial(
            bin_weighted_average_uplift,
            uplift_type="absolute",
            strategy="overall",
            bins=k,
            control_stability=False,
        )
        for k in range(2, 21)
    }
)

METRICS_WEIGHTED_UPLIFT_STABLE = dict(
    {
        f"weighted_uplift_{k}_bins_stable": partial(
            bin_weighted_average_uplift,
            uplift_type="absolute",
            strategy="overall",
            bins=k,
            control_stability=True,
        )
        for k in range(2, 21)
    }
)

METRICS_WEIGHTED_UPLIFT_REL = dict(
    {
        f"weighted_relative_uplift_{k}_bins": partial(
            bin_weighted_average_uplift,
            uplift_type="relative",
            strategy="overall",
            bins=k,
            control_stability=False,
        )
        for k in range(2, 21)
    }
)

METRICS_WEIGHTED_UPLIFT_REL_STABLE = dict(
    {
        f"weighted_relative_uplift_{k}_bins_stable": partial(
            bin_weighted_average_uplift,
            uplift_type="relative",
            strategy="overall",
            bins=k,
            control_stability=True,
        )
        for k in range(2, 21)
    }
)

METRICS_CONTROL_TREATMENT_TARGET_RATIOS_AT_K = dict(
    {
        f"control_treatment_ones_ratios@{k:02d}": partial(
            control_treatment_ones_ratios_at_k,
            strategy="overall",
            k=k / 100,
            target_type="both",
        )
        for k in range(5, 41, 5)
    }
)

METRICS_TREATMENT_TARGET_RATIOS_AT_K = dict(
    {
        f"treatment_ones_ratios@{k:02d}": partial(
            control_treatment_ones_ratios_at_k,
            strategy="overall",
            k=k / 100,
            target_type="treatment",
        )
        for k in range(5, 41, 5)
    }
)

METRICS_CONTROL_TARGET_RATIOS_AT_K = dict(
    {
        f"control_ones_ratios@{k:02d}": partial(
            control_treatment_ones_ratios_at_k,
            strategy="overall",
            k=k / 100,
            target_type="control",
        )
        for k in range(5, 41, 5)
    }
)

METRICS_ABS_UPLIFT_GROWTH_AT_K = dict(
    {
        f"abs_uplift_growth@{k:02d}": partial(
            abs_rel_uplift_growth_at_k,
            uplift_type="abs",
            strategy="overall",
            k=k / 100,
        )
        for k in range(5, 41, 5)
    }
)

METRICS_REL_UPLIFT_GROWTH_AT_K = dict(
    {
        f"rel_uplift_growth@{k:02d}": partial(
            abs_rel_uplift_growth_at_k,
            uplift_type="rel",
            strategy="overall",
            k=k / 100,
        )
        for k in range(5, 41, 5)
    }
)

METRICS_ABS_REL_UPLIFT_GROWTH_AT_K = dict(
    {
        f"abs_rel_uplift_growth@{k:02d}": partial(
            abs_rel_uplift_growth_at_k,
            uplift_type="both",
            strategy="overall",
            k=k / 100,
        )
        for k in range(5, 41, 5)
    }
)

METRICS_QINI_CLIPPED_AT_K = dict(
    {
        f"qini_clipped@{k:02d}": partial(
            qini_auc_score_clip_at_k,
            strategy="overall",
            k=k / 100,
            multiplier_threshold=0.9,
        )
        for k in range(5, 41, 5)
    }
)

METRICS_OTHERS = dict(
    {
        "qini_auc": qini_auc_score,
    }
)

METRICS = dict(
    list(
        chain(
            METRICS_UPLIFT_AT_K.items(),
            METRICS_UPLIFT_REL_AT_K.items(),
            METRICS_WEIGHTED_UPLIFT.items(),
            METRICS_WEIGHTED_UPLIFT_STABLE.items(),
            METRICS_WEIGHTED_UPLIFT_REL.items(),
            METRICS_WEIGHTED_UPLIFT_REL_STABLE.items(),
            METRICS_CONTROL_TREATMENT_TARGET_RATIOS_AT_K.items(),
            METRICS_CONTROL_TARGET_RATIOS_AT_K.items(),
            METRICS_TREATMENT_TARGET_RATIOS_AT_K.items(),
            METRICS_ABS_UPLIFT_GROWTH_AT_K.items(),
            METRICS_REL_UPLIFT_GROWTH_AT_K.items(),
            METRICS_ABS_REL_UPLIFT_GROWTH_AT_K.items(),
            METRICS_QINI_CLIPPED_AT_K.items(),
            METRICS_OTHERS.items(),
        )
    )
)
