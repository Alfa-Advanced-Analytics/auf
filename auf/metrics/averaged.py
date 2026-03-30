"""Bin-weighted and AUC-based uplift metrics.

Implements metrics that aggregate uplift across percentile bins with
various weighting strategies for stability and robustness.

Functions:
    bin_weighted_average_uplift: Weighted average uplift across bins.
    calculate_control_target_averages: Control response rate deviation.
    calculate_relative_uplift: Relative uplift deviation by bin.
    weighted_average_uplift_auc: Harmonic weighted AUC score.

Examples:
    >>> import numpy as np
    >>> from auf.metrics.averaged import bin_weighted_average_uplift

    >>> y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    >>> uplift = np.array([0.1, 0.4, 0.3, 0.8, 0.2, 0.5, 0.6, 0.7])
    >>> treatment = np.array([0, 1, 0, 1, 0, 1, 0, 1])

    >>> score = bin_weighted_average_uplift(y_true, uplift, treatment, bins=4)
"""

import typing as tp

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_consistent_length
from sklift.metrics import response_rate_by_percentile
from sklift.utils import check_is_binary


def bin_weighted_average_uplift(
    y_true: tp.Sequence[float],
    uplift: tp.Sequence[float],
    treatment: tp.Sequence[float],
    uplift_type: tp.Literal["absolute", "relative"] = "absolute",
    strategy: tp.Literal["overall", "by_group"] = "overall",
    bins: int = 10,
    control_stability: bool = False,
):
    """Compute bin-weighted average uplift score.

    Calculates uplift across percentile bins with harmonic weighting
    that reduces importance of lower-ranked bins.

    Args:
        y_true: True binary target values.
        uplift: Predicted uplift scores.
        treatment: Binary treatment labels (0=control, 1=treatment).
        uplift_type: Uplift calculation method.
            'absolute': treatment_rate - control_rate.
            'relative': treatment_rate / control_rate - 1.
            Defaults to 'absolute'.
        strategy: Ranking strategy for bin calculation.
            'overall': Global ranking across all samples.
            'by_group': Separate ranking within each group.
            Defaults to 'overall'.
        bins: Number of percentile bins. Defaults to 10.
        control_stability: If True, includes control stability penalty.
            Defaults to False.

    Returns:
        Bin-weighted average uplift score.

    Raises:
        ValueError: If strategy or uplift_type is invalid.
        ValueError: If bins is not a positive integer.
        ValueError: If bins >= number of samples.

    Examples:
        >>> import numpy as np
        >>> from auf.metrics import bin_weighted_average_uplift

        >>> y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        >>> uplift = np.array([0.1, 0.4, 0.3, 0.8, 0.2, 0.5, 0.6, 0.7])
        >>> treatment = np.array([0, 1, 0, 1, 0, 1, 0, 1])

        >>> score = bin_weighted_average_uplift(y_true, uplift, treatment, bins=4)

    Notes:
        Weights are inversely proportional to bin rank (1, 1/2, 1/3, ...).
        Control stability uses coefficient of variation as penalty term.
    """
    check_consistent_length(y_true, uplift, treatment)
    check_is_binary(treatment)
    check_is_binary(y_true)

    strategy_methods = ["overall", "by_group"]
    uplift_types = ["absolute", "relative"]

    n_samples = len(y_true)

    if strategy not in strategy_methods:
        raise ValueError(
            f"Response rate supports only calculating methods in {strategy_methods},"
            f" got {strategy}."
        )

    if uplift_type not in uplift_types:
        raise ValueError(
            f"Uplift supports only calculating methods in {uplift_types},"
            f" got {uplift_type}."
        )

    if not isinstance(bins, int) or bins <= 0:
        raise ValueError(
            f"Bins should be positive integer." f" Invalid value bins: {bins}"
        )

    if bins >= n_samples:
        raise ValueError(
            f"Number of bins = {bins} should be smaller than the length of y_true {n_samples}"
        )

    (
        bins_response_rate_trmnt,
        bins_variance_trmnt,
        bins_n_trmnt,
    ) = response_rate_by_percentile(
        y_true,
        uplift,
        treatment,
        group="treatment",
        strategy=strategy,
        bins=bins,
    )

    (
        bins_response_rate_ctrl,
        bins_variance_ctrl,
        bins_n_ctrl,
    ) = response_rate_by_percentile(
        y_true, uplift, treatment, group="control", strategy=strategy, bins=bins
    )

    if uplift_type == "absolute":
        uplift_scores = bins_response_rate_trmnt - bins_response_rate_ctrl
        weighted_avg_uplift = np.dot(1 / np.arange(1, 1 + bins), uplift_scores)

    else:
        uplift_scores = bins_response_rate_trmnt / bins_response_rate_ctrl - 1
        weighted_avg_uplift = np.dot(1 / np.arange(1, 1 + bins), uplift_scores)
        weighted_avg_uplift = (weighted_avg_uplift - min(uplift_scores)) / (
            max(uplift_scores) - min(uplift_scores)
        )

    if control_stability:
        response_rate_ctrl_stability = 1 - np.std(bins_response_rate_ctrl)
        weighted_avg_uplift = (
            2 * response_rate_ctrl_stability * weighted_avg_uplift
        ) / (response_rate_ctrl_stability + weighted_avg_uplift)

    return weighted_avg_uplift


def calculate_control_target_averages(df: pd.DataFrame, bins: int = 10):
    """Calculate control response rate deviation by cumulative bins.

    Computes the difference between cumulative control response rate
    in top bins and the overall control response rate.

    Args:
        df: DataFrame with 'target', 'treatment', and 'bin' columns.
        bins: Number of percentile bins. Defaults to 10.

    Returns:
        Series of differences for each cumulative bin threshold.

    Examples:
        >>> import pandas as pd
        >>> from auf.metrics.averaged import calculate_control_target_averages
        >>> df = pd.DataFrame({
        ...     'target': [0, 1, 0, 1, 0, 1],
        ...     'treatment': [0, 1, 0, 1, 0, 1],
        ...     'bin': [3, 3, 2, 2, 1, 1]
        ... })
        >>> diff = calculate_control_target_averages(df, bins=3)
    """
    bins_avg_cntrl_target = []
    bin_nums = list(range(1, bins + 1))

    for idx in range(1, bins):
        bins_avg_cntrl_target.append(
            df[(df["treatment"] == 0) & (df["bin"].isin(bin_nums[:idx]))][
                "target"
            ].mean()
        )

    avg_cntrl_target = df[df["treatment"] == 0]["target"].mean()
    bins_diff_cntrl_target = bins_avg_cntrl_target - avg_cntrl_target

    return bins_diff_cntrl_target


def calculate_relative_uplift(df: pd.DataFrame):
    """Calculate relative uplift deviation by bin.

    Computes the difference between relative uplift in each bin
    and the overall relative uplift.

    Args:
        df: DataFrame with 'target', 'treatment', and 'bin' columns.

    Returns:
        Series of relative uplift differences by bin.

    Examples:
        >>> import pandas as pd
        >>> from auf.metrics.averaged import calculate_relative_uplift
        >>> df = pd.DataFrame({
        ...     'target': [0, 1, 0, 1, 0, 1],
        ...     'treatment': [0, 1, 0, 1, 0, 1],
        ...     'bin': [2, 2, 1, 1, 1, 1]
        ... })
        >>> diff = calculate_relative_uplift(df)

    Notes:
        Relative uplift = treatment_rate / control_rate.
        Returns deviation from overall relative uplift per bin.
    """
    bins_mean_target = (
        df.groupby(["bin", "treatment"])["target"].mean().unstack()
    )
    bins_mean_target.columns = [
        "mean_target_treatment_0",
        "mean_target_treatment_1",
    ]

    bins_rel_uplift = (
        bins_mean_target["mean_target_treatment_1"]
        / bins_mean_target["mean_target_treatment_0"]
    )
    data_rel_uplift = (
        df[df["treatment"] == 1]["target"].mean()
        / df[df["treatment"] == 0]["target"].mean()
    )

    bins_rel_uplift_diff = bins_rel_uplift - data_rel_uplift

    return bins_rel_uplift_diff


def weighted_average_uplift_auc(
    y_true: tp.Sequence[float],
    uplift: tp.Sequence[float],
    treatment: tp.Sequence[float],
    bins: int = 10,
):
    """Compute harmonic weighted AUC of control and relative uplift curves.

    Calculates the harmonic mean of two AUC scores: control response rate
    deviation and relative uplift deviation across percentile bins.

    Args:
        y_true: True binary target values.
        uplift: Predicted uplift scores.
        treatment: Binary treatment labels (0=control, 1=treatment).
        bins: Number of percentile bins. Defaults to 10.

    Returns:
        Harmonic weighted AUC score.

    Examples:
        >>> import numpy as np
        >>> from auf.metrics import weighted_average_uplift_auc

        >>> y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        >>> uplift = np.array([0.1, 0.4, 0.3, 0.8, 0.2, 0.5, 0.6, 0.7])
        >>> treatment = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        >>> score = weighted_average_uplift_auc(y_true, uplift, treatment, bins=4)
    """
    check_consistent_length(y_true, uplift, treatment)
    check_is_binary(y_true)
    check_is_binary(treatment)

    percentiles = np.arange(0, 1 + 1 / bins, 1 / bins)
    df = pd.DataFrame(
        {"target": y_true, "uplift": uplift, "treatment": treatment}
    ).sort_values(by="uplift", ascending=True)
    df["bin"] = pd.cut(
        df.uplift,
        bins=df.uplift.quantile(percentiles).values,
        labels=False,
        duplicates="drop",
    )
    df["bin"].fillna(0, inplace=True)
    df["bin"] = bins - df["bin"]

    bins_diff_cntrl_target = calculate_control_target_averages(df, bins)
    bins_rel_uplift_diff = calculate_relative_uplift(df)

    ctrl_target_auc = np.trapz(bins_diff_cntrl_target, dx=1)
    rel_uplift_auc = np.trapz(bins_rel_uplift_diff, dx=1)
    weighted_auc_uplift = (
        2
        * (ctrl_target_auc * rel_uplift_auc)
        / (ctrl_target_auc + rel_uplift_auc)
    )

    return weighted_auc_uplift
