"""Visualization utilities for uplift modeling analysis.

Provides plotting functions for model evaluation, feature analysis,
and client segmentation in uplift modeling workflows.

Functions:
    plot_uplift_by_percentile: Plot uplift scores and response rates
        by percentile bins.
    plot_uplift_top_vs_bottom: Compare response rates between top and
        bottom segments.
    plot_cumulative_target_ratio: Plot cumulative target ratio curves
        for treatment and control groups.
    plot_portrait_tree: Visualize client segments using decision tree
        on predicted uplift.
    plot_uplift_by_feature_bins: Plot uplift and observation counts
        by feature bins.

Examples:
    >>> from auf.plots import plot_uplift_by_percentile, plot_portrait_tree
    >>> import numpy as np

    >>> y_true = np.random.randint(0, 2, 1000)
    >>> uplift = np.random.randn(1000)
    >>> treatment = np.random.randint(0, 2, 1000)

    >>> axes = plot_uplift_by_percentile(
    ...     y_true=y_true,
    ...     uplift=uplift,
    ...     treatment=treatment,
    ...     bins=10
    ... )

Notes:
    All plotting functions support external axes for subplot integration.
    Uses matplotlib as the plotting backend with default style.
"""

from .plots import (
    plot_cumulative_target_ratio,
    plot_portrait_tree,
    plot_uplift_by_feature_bins,
    plot_uplift_by_percentile,
    plot_uplift_top_vs_bottom,
)

__all__ = [
    "plot_cumulative_target_ratio",
    "plot_uplift_by_feature_bins",
    "plot_portrait_tree",
    "plot_uplift_by_percentile",
    "plot_uplift_top_vs_bottom",
]
