"""Plotting functions for uplift model evaluation and interpretation.

Provides visualization tools for analyzing uplift model performance,
feature effects, and client segmentation. All functions support
external matplotlib axes for flexible subplot layouts.

Functions:
    plot_uplift_by_percentile: Visualize uplift and response rates
        across percentile bins with error bars.
    plot_uplift_top_vs_bottom: Compare response rates between top
        and bottom uplift segments.
    plot_cumulative_target_ratio: Plot cumulative gain curves for
        treatment and control groups.
    plot_portrait_tree: Build interpretable decision tree visualization
        on predicted uplift values.
    plot_uplift_by_feature_bins: Analyze uplift distribution across
        feature value bins.

Examples:
    >>> from auf.plots import plot_uplift_by_percentile
    >>> import numpy as np

    >>> y_true = np.random.randint(0, 2, 1000)
    >>> uplift = np.random.randn(1000)
    >>> treatment = np.random.randint(0, 2, 1000)

    >>> axes = plot_uplift_by_percentile(
    ...     y_true=y_true,
    ...     uplift=uplift,
    ...     treatment=treatment,
    ...     strategy='overall',
    ...     bins=10
    ... )

Notes:
    All functions validate input consistency using sklearn's
    check_consistent_length.
    Treatment and target columns must be binary.
    Uses 'forestgreen' for treatment group and 'orange' for control
    group consistently across all plots.
"""

import typing as tp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.utils.validation import check_consistent_length
from sklift.metrics import uplift_by_percentile
from sklift.utils import check_is_binary

from ..constants import RANDOM_STATE

plt.style.use("default")


def plot_uplift_by_percentile(
    y_true,
    uplift,
    treatment,
    strategy="overall",
    kind="line",
    bins=10,
    string_percentiles=True,
    axes=None,
    draw_bars="both",
):
    """Plot uplift scores and response rates by percentile bins.

    Visualizes uplift model performance by dividing predictions into
    percentile bins and computing response rates for treatment and
    control groups within each bin.

    Args:
        y_true (1d array-like): Correct (true) binary target values.
        uplift (1d array-like): Predicted uplift, as returned by a model.
        treatment (1d array-like): Treatment labels (binary).
        strategy (str): Calculating strategy for uplift.
            'overall': Sort all data by uplift and calculate conversions
                on top k observations across both groups.
            'by_group': Calculate conversions separately in top k
                observations of each group.
            Defaults to 'overall'.
        kind (str): Type of plot to draw.
            'line': Line plot with error bars.
            'bar': Bar-style plot.
            Defaults to 'line'.
        bins (int): Number of percentile bins. Defaults to 10.
        string_percentiles (bool): If True, format x-axis labels as
            percentile ranges (e.g., '0-10'). If False, use numeric
            percentile values. Defaults to True.
        axes (matplotlib.axes.Axes): External axes for plotting.
            If None, creates new figure. Defaults to None.
        draw_bars (str): Which bar plots to draw when kind='bar'.
            'both': Draw both uplift and response rate bars.
            'uplift': Draw only uplift bars.
            'rates': Draw only response rate bars.
            Defaults to 'both'.

    Returns:
        matplotlib.axes.Axes: Axes object with the plot.

    Raises:
        ValueError: If strategy is not in ['overall', 'by_group'].
        ValueError: If kind is not in ['line', 'bar'].
        ValueError: If bins is not a positive integer.
        ValueError: If bins exceeds sample size.

    Examples:
        >>> import numpy as np
        >>> y_true = np.random.randint(0, 2, 1000)
        >>> uplift = np.random.randn(1000)
        >>> treatment = np.random.randint(0, 2, 1000)

        >>> axes = plot_uplift_by_percentile(
        ...     y_true=y_true,
        ...     uplift=uplift,
        ...     treatment=treatment,
        ...     strategy='overall',
        ...     kind='line',
        ...     bins=10
        ... )

    Notes:
        Extends sklift.metrics.uplift_by_percentile with custom axes
            support and selective bar plotting.
        Error bars represent standard deviation within each bin.
        Line plot fills area between treatment and control rates.
    """
    strategy_methods = ["overall", "by_group"]
    kind_methods = ["line", "bar"]

    check_consistent_length(y_true, uplift, treatment)
    check_is_binary(treatment)
    check_is_binary(y_true)
    n_samples = len(y_true)

    if strategy not in strategy_methods:
        raise ValueError(
            f"Response rate supports only calculating methods in {strategy_methods},"
            f" got {strategy}."
        )

    if kind not in kind_methods:
        raise ValueError(
            f"Function supports only types of plots in {kind_methods},"
            f" got {kind}."
        )

    if not isinstance(bins, int) or bins <= 0:
        raise ValueError(
            f"Bins should be positive integer. Invalid value bins: {bins}"
        )

    if bins >= n_samples:
        raise ValueError(
            f"Number of bins = {bins} should be smaller than the length of y_true {n_samples}"
        )

    if not isinstance(string_percentiles, bool):
        raise ValueError(
            "string_percentiles flag should be bool: True or False."
            f" Invalid value string_percentiles: {string_percentiles}"
        )

    df = uplift_by_percentile(
        y_true,
        uplift,
        treatment,
        strategy=strategy,
        std=True,
        total=False,
        bins=bins,
        string_percentiles=False,
    )

    percentiles = df.index[:bins].values.astype(float)

    response_rate_trmnt = df.loc[percentiles, "response_rate_treatment"].values
    std_trmnt = df.loc[percentiles, "std_treatment"].values

    response_rate_ctrl = df.loc[percentiles, "response_rate_control"].values
    std_ctrl = df.loc[percentiles, "std_control"].values

    uplift_score = df.loc[percentiles, "uplift"].values
    std_uplift = df.loc[percentiles, "std_uplift"].values

    check_consistent_length(
        percentiles,
        response_rate_trmnt,
        response_rate_ctrl,
        uplift_score,
        std_trmnt,
        std_ctrl,
        std_uplift,
    )

    if kind == "line":
        if axes is None:
            _, axes = plt.subplots(ncols=1, nrows=1, figsize=(8, 6))
        axes.errorbar(
            percentiles,
            response_rate_trmnt,
            yerr=std_trmnt,
            linewidth=2,
            color="forestgreen",
            label="treatment\nresponse rate",
        )
        axes.errorbar(
            percentiles,
            response_rate_ctrl,
            yerr=std_ctrl,
            linewidth=2,
            color="orange",
            label="control\nresponse rate",
        )
        axes.errorbar(
            percentiles,
            uplift_score,
            yerr=std_uplift,
            linewidth=2,
            color="red",
            label="uplift",
        )
        axes.fill_between(
            percentiles,
            response_rate_trmnt,
            response_rate_ctrl,
            alpha=0.1,
            color="red",
        )

        if np.amin(uplift_score) < 0:
            axes.axhline(y=0, color="black", linewidth=1)

        if string_percentiles:  # string percentiles for plotting
            percentiles_str = [f"0-{percentiles[0]:.0f}"] + [
                f"{percentiles[i]:.0f}-{percentiles[i + 1]:.0f}"
                for i in range(len(percentiles) - 1)
            ]
            axes.set_xticks(percentiles)
            axes.set_xticklabels(percentiles_str, rotation=45)
        else:
            axes.set_xticks(percentiles)

        axes.legend(loc="upper right")
        axes.set_title("Uplift by percentile\n")
        axes.set_xlabel("Percentile")
        axes.set_ylabel(
            "Uplift = treatment response rate - control response rate"
        )

    elif draw_bars == "both":  # kind == 'bar'
        delta = percentiles[0]

        # don't check axes if None : draw both plots on big figure
        fig, axes = plt.subplots(
            ncols=1, nrows=2, figsize=(8, 6), sharex=True, sharey=True
        )
        fig.text(
            0.04,
            0.5,
            "Uplift = treatment response rate - control response rate",
            va="center",
            ha="center",
            rotation="vertical",
        )

        axes[1].bar(
            np.array(percentiles) - delta / 6,
            response_rate_trmnt,
            delta / 3,
            yerr=std_trmnt,
            color="forestgreen",
            label="treatment\nresponse rate",
        )
        axes[1].bar(
            np.array(percentiles) + delta / 6,
            response_rate_ctrl,
            delta / 3,
            yerr=std_ctrl,
            color="orange",
            label="control\nresponse rate",
        )
        axes[0].bar(
            np.array(percentiles),
            uplift_score,
            delta / 1.5,
            yerr=std_uplift,
            color="red",
            label="uplift",
        )

        axes[0].legend(loc="upper right")
        axes[0].tick_params(axis="x", bottom=False)
        axes[0].axhline(y=0, color="black", linewidth=1)
        axes[0].set_title("Uplift by percentile\n")

        if string_percentiles:  # string percentiles for plotting
            percentiles_str = [f"0-{percentiles[0]:.0f}"] + [
                f"{percentiles[i]:.0f}-{percentiles[i + 1]:.0f}"
                for i in range(len(percentiles) - 1)
            ]
            axes[1].set_xticks(percentiles)
            axes[1].set_xticklabels(percentiles_str, rotation=45)

        else:
            axes[1].set_xticks(percentiles)

        axes[1].legend(loc="upper right")
        axes[1].axhline(y=0, color="black", linewidth=1)
        axes[1].set_xlabel("Percentile")
        axes[1].set_title("Response rate by percentile")

    elif draw_bars == "uplift":  # kind == 'bar'
        delta = percentiles[0]

        if axes is None:
            fig, axes = plt.subplots(
                ncols=1, nrows=1, figsize=(5, 5), sharex=True, sharey=True
            )
        fig.text(
            0.04,
            0.5,
            "Uplift = treatment response rate - control response rate",
            va="center",
            ha="center",
            rotation="vertical",
        )

        axes.bar(
            np.array(percentiles),
            uplift_score,
            delta / 1.5,
            yerr=std_uplift,
            color="red",
            label="uplift",
        )

        axes.legend(loc="upper right")
        axes.tick_params(axis="x", bottom=False)
        axes.axhline(y=0, color="black", linewidth=1)
        axes.set_title("Uplift by percentile\n")

    elif draw_bars == "rates":  # kind == 'bar'
        delta = percentiles[0]

        if axes is None:
            fig, axes = plt.subplots(
                ncols=1, nrows=1, figsize=(5, 5), sharex=True, sharey=True
            )
            fig.text(
                0.04,
                0.5,
                "Uplift = treatment response rate - control response rate",
                va="center",
                ha="center",
                rotation="vertical",
            )

        axes.bar(
            np.array(percentiles) - delta / 6,
            response_rate_trmnt,
            delta / 3,
            yerr=std_trmnt,
            color="forestgreen",
            label="treatment\nresponse rate",
        )
        axes.bar(
            np.array(percentiles) + delta / 6,
            response_rate_ctrl,
            delta / 3,
            yerr=std_ctrl,
            color="orange",
            label="control\nresponse rate",
        )

        if string_percentiles:  # string percentiles for plotting
            percentiles_str = [f"0-{percentiles[0]:.0f}"] + [
                f"{percentiles[i]:.0f}-{percentiles[i + 1]:.0f}"
                for i in range(len(percentiles) - 1)
            ]
            axes.set_xticks(percentiles)
            axes.set_xticklabels(percentiles_str, rotation=45)

        else:
            axes.set_xticks(percentiles)

        axes.legend(loc="upper right")
        axes.axhline(y=0, color="black", linewidth=1)
        axes.set_xlabel("Percentile")
        axes.set_title("Response rate by percentile")

    return axes


def plot_uplift_top_vs_bottom(
    y_true,
    uplift,
    treatment,
    top_ratio: float = 0.1,
    kind: str = "bar",
    axes=None,
):
    """Plot response rates for top and bottom uplift segments.

    Compares response rates between the highest and lowest uplift
    predictions to evaluate model discrimination ability.

    Args:
        y_true (1d array-like): Correct (true) binary target values.
        uplift (1d array-like): Predicted uplift, as returned by a model.
        treatment (1d array-like): Treatment labels (binary).
        top_ratio (float): Ratio of top segment size. Must be in (0, 1].
            Defaults to 0.1 (top 10%).
        kind (str): Type of plot to draw.
            'line': Line plot connecting top and bottom segments.
            'bar': Grouped bar chart.
            Defaults to 'bar'.
        axes (matplotlib.axes.Axes): External axes for plotting.
            If None, creates new figure. Defaults to None.

    Returns:
        matplotlib.axes.Axes: Axes object with the plot.

    Raises:
        ValueError: If top_ratio is not in (0, 1].
        ValueError: If kind is not in ['line', 'bar'].

    Examples:
        >>> import numpy as np
        >>> y_true = np.random.randint(0, 2, 1000)
        >>> uplift = np.random.randn(1000)
        >>> treatment = np.random.randint(0, 2, 1000)

        >>> axes = plot_uplift_top_vs_bottom(
        ...     y_true=y_true,
        ...     uplift=uplift,
        ...     treatment=treatment,
        ...     top_ratio=0.2,
        ...     kind='bar'
        ... )

    Notes:
        Top segment contains observations with highest uplift predictions.
        Bottom segment contains all remaining observations.
        Useful for quick model performance validation.
    """
    check_consistent_length(y_true, uplift, treatment)
    check_is_binary(treatment)
    check_is_binary(y_true)

    if not (0 < top_ratio <= 1):
        raise ValueError("top_ratio must be in (0, 1]")

    if kind not in ["line", "bar"]:
        raise ValueError(
            f"Function supports only types of plots in ['line', 'bar'], got {kind}."
        )

    df = pd.DataFrame(
        {"y_true": y_true, "uplift": uplift, "treatment": treatment}
    )
    df = df.sort_values("uplift", ascending=False).reset_index(drop=True)

    top_size = int(len(df) * top_ratio)
    top_df = df.iloc[:top_size]
    bottom_df = df.iloc[top_size:]

    def calc_rates(subset_df):
        trmnt = subset_df[subset_df["treatment"] == 1]["y_true"]
        ctrl = subset_df[subset_df["treatment"] == 0]["y_true"]
        return {
            "rate_treatment": trmnt.mean() if len(trmnt) > 0 else 0,
            "rate_control": ctrl.mean() if len(ctrl) > 0 else 0,
        }

    top_rates = calc_rates(top_df)
    bottom_rates = calc_rates(bottom_df)

    positions = [f"Top {top_ratio:.0%}", f"Bottom {1-top_ratio:.0%}"]
    trmnt_rates = [top_rates["rate_treatment"], bottom_rates["rate_treatment"]]
    ctrl_rates = [top_rates["rate_control"], bottom_rates["rate_control"]]

    if axes is None:
        if kind == "line":
            fig, axes = plt.subplots(figsize=(8, 6))
        else:
            fig, axes = plt.subplots(figsize=(8, 6))

    if kind == "line":
        x = [0, 1]
        axes.plot(
            x,
            trmnt_rates,
            "-o",
            linewidth=2,
            color="forestgreen",
            label="Treatment response rate",
        )
        axes.plot(
            x,
            ctrl_rates,
            "-o",
            linewidth=2,
            color="orange",
            label="Control response rate",
        )
        axes.fill_between(x, trmnt_rates, ctrl_rates, alpha=0.1, color="red")
        axes.set_xticks(x)
        axes.set_xticklabels(positions)

    elif kind == "bar":
        x = np.arange(len(positions))
        width = 0.35
        axes.bar(
            x - width / 2,
            trmnt_rates,
            width,
            label="Treatment response rate",
            color="forestgreen",
        )
        axes.bar(
            x + width / 2,
            ctrl_rates,
            width,
            label="Control response rate",
            color="orange",
        )
        axes.set_xticks(x)
        axes.set_xticklabels(positions)

    axes.set_xlabel("Sample segment")
    axes.set_ylabel("Response rate")
    axes.set_title("Response rates in top vs bottom segments")
    axes.legend()
    axes.axhline(y=0, color="black", linewidth=0.5)

    return axes


def plot_cumulative_target_ratio(
    y_true,
    uplift,
    treatment,
    ax=None,
    color_control="orange",
    color_treatment="forestgreen",
    linewidth=2,
    linestyle="-",
    random=True,
    label=None,
):
    """Plot cumulative target ratio curves for treatment and control.

    Visualizes the cumulative gain achieved by ordering samples by
    predicted uplift, showing the proportion of targets captured
    at each fraction of the sample.

    Args:
        y_true (1d array-like): Correct (true) binary target values.
        uplift (1d array-like): Predicted uplift, as returned by a model.
        treatment (1d array-like): Treatment labels (binary).
        ax (matplotlib.axes.Axes): External axes for plotting.
            If None, creates new figure. Defaults to None.
        color_control (str): Color for control group curve.
            Defaults to 'orange'.
        color_treatment (str): Color for treatment group curve.
            Defaults to 'forestgreen'.
        linewidth (int): Width of plot lines. Defaults to 2.
        linestyle (str): Style of plot lines. Defaults to '-'.
        random (bool): Whether to plot random selection baseline.
            Defaults to True.
        label (str): Label prefix for model curves. If None, uses
            'Control' and 'Treatment'. Defaults to None.

    Returns:
        matplotlib.axes.Axes: Axes object with the plot.

    Raises:
        ValueError: If either group contains zero positive targets.

    Examples:
        >>> import numpy as np
        >>> y_true = np.random.randint(0, 2, 1000)
        >>> uplift = np.random.randn(1000)
        >>> treatment = np.random.randint(0, 2, 1000)

        >>> ax = plot_cumulative_target_ratio(
        ...     y_true=y_true,
        ...     uplift=uplift,
        ...     treatment=treatment,
        ...     random=True
        ... )

    Notes:
        Samples are ordered by descending uplift prediction.
        Curves show cumulative proportion of positive targets captured.
        Random baseline represents uniform target distribution.
        Similar to cumulative gain charts for binary classification.
    """
    check_consistent_length(y_true, uplift, treatment)
    check_is_binary(treatment)
    check_is_binary(y_true)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    order = np.argsort(uplift, kind="mergesort")[::-1]
    y_true = np.array(y_true)[order]
    treatment = np.array(treatment)[order]

    n_samples = len(y_true)

    control_mask = treatment == 0
    treatment_mask = treatment == 1

    y_true_control = y_true[control_mask]
    y_true_treatment = y_true[treatment_mask]

    total_control = y_true_control.sum()
    total_treatment = y_true_treatment.sum()

    if total_control == 0 or total_treatment == 0:
        raise ValueError(
            "Both groups must contain at least one positive target"
        )

    x_data = np.linspace(0, 1, n_samples)

    cumsum_control = np.cumsum(y_true_control) / total_control
    cumsum_treatment = np.cumsum(y_true_treatment) / total_treatment

    control_x = np.linspace(0, 1, len(cumsum_control))
    treatment_x = np.linspace(0, 1, len(cumsum_treatment))

    control_curve = np.interp(x_data, control_x, cumsum_control)
    treatment_curve = np.interp(x_data, treatment_x, cumsum_treatment)

    ax.plot(
        x_data,
        control_curve,
        color=color_control,
        linewidth=linewidth,
        linestyle=linestyle,
        label="Control" if label is None else f"{label} (Control)",
    )

    ax.plot(
        x_data,
        treatment_curve,
        color=color_treatment,
        linewidth=linewidth,
        linestyle=linestyle,
        label="Treatment" if label is None else f"{label} (Treatment)",
    )

    if random:
        ax.plot(
            x_data,
            x_data,
            color="black",
            linewidth=1,
            linestyle="--",
            label="Random",
        )

    ax.set_xlabel("Fraction of sample (ordered by descending uplift)")
    ax.set_ylabel("Cumulative target ratio")
    ax.set_title("Cumulative target ratio curves")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    return ax


def plot_portrait_tree(
    x: pd.DataFrame,
    uplift: np.array,
    feature_names_dict: tp.Optional[tp.Dict[str, str]] = None,
    max_depth: int = 3,
    max_leaf_nodes: int = 6,
    min_samples_leaf: tp.Union[float, int] = 0.05,
    axes=None,
):
    """Build client portrait visualization using decision tree on uplift.

    Trains a decision tree regressor on original features with predicted
    uplift as target, then visualizes the tree to reveal segments with
    high and low uplift values.

    Args:
        x (pd.DataFrame): Features DataFrame used for uplift model training.
        uplift (1d array-like): Predicted uplift values from uplift model.
        feature_names_dict (Optional[Dict[str, str]]): Mapping from original
            feature names to display names. Keys are column names in x,
            values are labels to show in the tree. Defaults to None.
        max_depth (int): Maximum depth of the decision tree.
            Defaults to 3.
        max_leaf_nodes (int): Maximum number of leaf nodes in the tree.
            Defaults to 6.
        min_samples_leaf (Union[float, int]): Minimum samples required
            in a leaf node.
            If int, absolute count. If float, fraction of samples.
            Defaults to 0.05 (5% of samples).
        axes (matplotlib.axes.Axes): External axes for plotting.
            If None, creates new figure. Defaults to None.

    Returns:
        matplotlib.figure.Figure: Figure object containing the tree plot,
            or None if axes is provided.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from auf.plots import plot_portrait_tree

        >>> X = pd.DataFrame({
        ...     'age': np.random.randint(18, 70, 1000),
        ...     'income': np.random.randn(1000) * 10000 + 50000
        ... })
        >>> uplift = np.random.randn(1000)

        >>> fig = plot_portrait_tree(
        ...     x=X,
        ...     uplift=uplift,
        ...     max_depth=3,
        ...     min_samples_leaf=0.05
        ... )

    Notes:
        Uses sklearn.tree.DecisionTreeRegressor internally.
        Tree is fitted on original features, not encoded features.
        Useful for interpreting which feature values correspond to
            high or low uplift predictions.
        The tree reveals client segments for targeted marketing.
    """
    check_consistent_length(x, uplift)

    if isinstance(min_samples_leaf, float):
        min_samples_leaf = int(min_samples_leaf * len(x))

    if feature_names_dict:
        feats = [
            feature_names_dict[col] if col in feature_names_dict else col
            for col in x.columns
        ]
    else:
        feats = x.columns

    tree = DecisionTreeRegressor(
        max_depth=max_depth,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        random_state=RANDOM_STATE,
    )

    tree.fit(x, uplift)

    fig = None
    if axes is None:
        fig, axes = plt.subplots(1, 1, figsize=(12, 8))

    plot_tree(tree, filled=True, feature_names=feats, ax=axes)

    plt.title("Client portrait: uplift depending on model features")
    plt.tight_layout()
    plt.show()
    return fig


def plot_uplift_by_feature_bins(
    feature: tp.Sequence[float],
    treatment: tp.Sequence[float],
    target: tp.Sequence[float],
    feature_name: str,
    amount_of_bins: int = 7,
    round_const: int = 3,
    axes=None,
):
    """Plot uplift and observation counts by feature value bins.

    Creates a two-panel visualization showing observation counts and
    target rates by treatment group across binned feature values.

    Args:
        feature (1d array-like): Feature values to bin. Values outside
            the 2nd to 98th percentile are excluded from binning.
        treatment (1d array-like): Treatment labels (binary).
        target (1d array-like): Binary target values.
        feature_name (str): Display name for the feature in plot titles.
        amount_of_bins (int): Number of bins for numerical features.
            Defaults to 7.
        round_const (int): Decimal places for bin edge labels.
            Defaults to 3.
        axes (array-like): External axes for plotting. Must contain
            two axes for count and rate plots. If None, creates new
            figure. Defaults to None.

    Returns:
        matplotlib.figure.Figure: Figure object containing the plots,
            or None if axes is provided.

    Examples:
        >>> import numpy as np
        >>> from auf.plots import plot_uplift_by_feature_bins

        >>> feature = np.random.randn(1000) * 10 + 50
        >>> treatment = np.random.randint(0, 2, 1000)
        >>> target = np.random.randint(0, 2, 1000)

        >>> fig = plot_uplift_by_feature_bins(
        ...     feature=feature,
        ...     treatment=treatment,
        ...     target=target,
        ...     feature_name='customer_age',
        ...     amount_of_bins=5
        ... )

    Notes:
        For numerical features, bins are created between 2nd and 98th
            percentiles to exclude outliers.
        Missing values (NaN) are shown as a separate 'NAN' category.
        For categorical features with more than amount_of_bins unique
            values, less frequent categories are grouped into 'Other'.
        Left panel shows observation counts by treatment group.
        Right panel shows target rates (uplift) by treatment group.
    """
    check_consistent_length(feature, treatment, target)

    if feature_name is None:
        feature_name = "feature"
    df = pd.DataFrame(
        {feature_name: feature, "target": target, "treatment": treatment}
    )

    amount_of_nans = df[feature_name].isna().sum()

    if df[feature_name].dtype == "object":
        df[feature_name].fillna("NAN", inplace=True)

        category_counts = (
            df[feature_name][df[feature_name] != "NAN"]
            .value_counts()
            .nlargest(amount_of_bins)
            .index.tolist()
        )

        if df[feature_name].nunique() > amount_of_bins:
            category_counts.pop()
            category_counts.append("Other")

            df[feature_name] = df[feature_name].apply(
                lambda x: x if x in category_counts else "Other"
            )

        ordered_categories = ["NAN"] + category_counts

        df[feature_name] = pd.Series(
            pd.Categorical(
                df[feature_name], categories=ordered_categories, ordered=False
            )
        )

    else:
        if amount_of_bins > df[feature_name].dropna().unique().shape[0]:
            amount_of_bins = df[feature_name].dropna().nunique()

        bins = np.linspace(
            df[feature_name].quantile(q=0.02),
            df[feature_name].quantile(q=0.98),
            amount_of_bins,
        )
        df = df.loc[
            (df[feature_name] >= df[feature_name].quantile(q=0.02))
            & (df[feature_name] <= df[feature_name].quantile(q=0.98))
            | (df[feature_name].isna())
        ]

        feature_cat = pd.cut(
            x=df[feature_name],
            bins=bins,
            precision=round_const,
            duplicates="drop",
            include_lowest=True,
            ordered=True,
        )

        interval_labels = [
            f"{intv.left:.{round_const}f} - {intv.right:.{round_const}f}"
            for intv in feature_cat.cat.categories
        ]

        feature_cat = feature_cat.cat.rename_categories(interval_labels)

        if amount_of_nans > 0:
            feature_cat = feature_cat.cat.add_categories("NAN")
            feature_cat = feature_cat.fillna("NAN")
            interval_labels = ["NAN"] + interval_labels

        feature_cat = feature_cat.cat.reorder_categories(interval_labels)

        df[feature_name] = feature_cat

    grouped = (
        df.groupby([feature_name, "treatment"])
        .agg(count=("target", "size"), mean_target=("target", "mean"))
        .reset_index()
    )

    if df[feature_name].dtype == "object":
        grouped.sort_values(by="count", inplace=True, ascending=False)

        if amount_of_nans > 0:
            nan_mask = grouped[feature_name] == "NAN"
            grouped = pd.concat(
                [grouped[nan_mask], grouped[~nan_mask]], ignore_index=True
            )

        other_mask = grouped[feature_name] == "Other"
        grouped = pd.concat(
            [grouped[~other_mask], grouped[other_mask]], ignore_index=True
        )

    else:
        grouped[feature_name] = pd.Categorical(
            grouped[feature_name],
            categories=df[feature_name].cat.categories,
            ordered=True,
        )
        grouped = grouped.sort_values(feature_name)

    treatment_1 = grouped[grouped["treatment"] == 1]
    treatment_0 = grouped[grouped["treatment"] == 0]

    fig = None
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax1 = axes[0]
    width = 0.35
    x1 = np.arange(len(treatment_1[feature_name]))
    x0 = x1 + width

    ax1.bar(
        x1,
        treatment_1["count"],
        width,
        color="forestgreen",
        edgecolor="black",
        label="Treatment",
    )
    ax1.bar(
        x0,
        treatment_0["count"],
        width,
        color="orange",
        edgecolor="black",
        label="Control",
    )

    ax1.set_title("Treatment and control group sizes")
    ax1.set_ylabel("Number of observations")
    ax1.set_xticks(x1 + width / 2)
    ax1.set_xticklabels(treatment_1[feature_name])
    ax1.legend(loc="upper right")

    ax2 = axes[1]
    ax2.plot(
        treatment_1[feature_name],
        treatment_1["mean_target"],
        color="forestgreen",
        label="Treatment\ntarget rate",  # "Treatment 1",
        marker="o",
    )
    ax2.plot(
        treatment_0[feature_name],
        treatment_0["mean_target"],
        color="orange",
        label="Control\ntarget rate",  # "Treatment 0",
        marker="o",
    )
    ax2.set_title("Uplift")
    ax2.legend(loc="upper right")

    fig.suptitle(f"Uplift and observations count by\n{feature_name}\nbuckets")

    for ax in axes:
        ax.tick_params(axis="x", rotation=45)

    return fig
