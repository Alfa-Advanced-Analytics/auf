"""Model evaluation utilities for uplift modeling.

Provides a main evaluation function that computes metrics, generates
summary tables and plots for assessing uplift model performance on a
given dataset. The module integrates metric calculations from the
constants module with visualization tools from the plots module.

Functions:
    evaluate_model: Comprehensive model evaluation with configurable
        output types (tables, plots).

Examples:
    >>> from auf.pipeline.evaluation import evaluate_model
    >>> from auf.models import AufModel
    >>> from auf.data.preprocessing import Preprocessor

    >>> base_cols_mapper = {'id': 'id', 'treatment': 'trt', 'target': 'y'}
    >>> treatment_groups_mapper = {'control': 0, 'treatment': 1}

    >>> evaluate_model(
    ...     base_cols_mapper=base_cols_mapper,
    ...     treatment_groups_mapper=treatment_groups_mapper,
    ...     data=test_df,
    ...     preprocessor=preprocessor,
    ...     model=trained_model,
    ...     evaluation_types=['metrics_table', 'buckets_qini_plots'],
    ...     n_uplift_bins=10
    ... )

Notes:
    All evaluation types assume binary treatment (control vs treatment).
    The function displays outputs using IPython.display and matplotlib.
    For multi-treatment models, uplift is taken as the maximum across
    treatment groups.
"""

import typing as tp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from sklift.metrics import uplift_by_percentile
from sklift.viz import plot_qini_curve

from ..constants import METRICS
from ..data.preprocessing import Preprocessor
from ..models import AufModel
from ..plots import plot_cumulative_target_ratio, plot_uplift_by_percentile


def evaluate_model(
    base_cols_mapper: tp.Dict[str, str],
    treatment_groups_mapper: tp.Dict[str, str],
    data: pd.DataFrame,
    preprocessor: Preprocessor,
    model: AufModel,
    evaluation_types: tp.List[str] = [
        "target_info",
        "metrics_table",
        "buckets_table",
        "tops_table",
        "buckets_qini_plots",
        "target_ratios_plots",
    ],
    n_uplift_bins: int = 10,
):
    """Evaluate uplift model on a given dataset.

    Performs comprehensive model evaluation including target statistics,
    metrics calculation, bucket analysis, top-slice analysis and
    visualization plots. Outputs are displayed in the notebook.

    Args:
        base_cols_mapper: Mapping from unified column
            names (id, treatment, target, segm) to actual column names in
            the data.
        treatment_groups_mapper: Mapping from treatment
            group names to internal integer codes (0 for control, 1 for
            treatment).
        data: Dataset for evaluation. Must contain target
            and treatment columns.
        preprocessor: Fitted preprocessor instance for
            transforming features before prediction.
        model: Trained uplift model to evaluate.
        evaluation_types: List of evaluation types to
            perform. Available types are 'target_info', 'metrics_table',
            'buckets_table', 'tops_table', 'buckets_qini_plots',
            'target_ratios_plots'. Defaults to all types.
        n_uplift_bins: Number of bins for bucket-based analysis and
            plots. Defaults to 10.

    Returns:
        None

    Raises:
        AssertionError: If evaluation_types contains invalid values.
        AssertionError: If target or treatment columns are missing from
            data.
        AssertionError: If target contains values other than 0 and 1.
        AssertionError: If treatment_groups_mapper keys do not match
            unique treatment values in data.
        AssertionError: If treatment_groups_mapper values contain values
            other than 0 and 1.

    Notes:
        The function displays outputs using IPython.display.display() and
        matplotlib.pyplot.show(). For multi-treatment models, the maximum
        uplift across treatment groups is used for evaluation.
        Metrics are calculated using functions from the METRICS constant
        dictionary.
    """
    allowed_types = [
        "target_info",
        "metrics_table",
        "buckets_table",
        "tops_table",
        "buckets_qini_plots",
        "target_ratios_plots",
    ]
    wrong_types = list(set(evaluation_types) - set(allowed_types))
    assert not wrong_types, (
        f"Wrong evaluation types: {wrong_types}\n"
        f"Allowed types: {allowed_types}"
    )

    assert (
        base_cols_mapper["target"] in data.columns
    ), "Data must contain target"
    assert (
        base_cols_mapper["treatment"] in data.columns
    ), "Data must contain treatment"

    target = data[base_cols_mapper["target"]].values
    treatment = data[base_cols_mapper["treatment"]]

    assert set(treatment_groups_mapper.keys()) == set(treatment), (
        "Treatment mapping keys must be same as treatment unique values.\n"
        f"    Treatment mapping keys: {treatment_groups_mapper.keys()}\n"
        f"    Treatment unique values: {set(treatment)}\n"
    )

    treatment = treatment.map(treatment_groups_mapper).values

    assert set(target) == {
        0,
        1,
    }, f"Target must be only 0 and 1, found: {set(target)}"

    data = preprocessor.transform(data, inplace=False)
    uplift = model.predict(data, return_df=False)

    if model._is_multitreatment:
        uplift = uplift.max(1)
        control_name = treatment_groups_mapper["control"]
        treatment = (treatment != control_name).astype(int)

    if "target_info" in evaluation_types:
        target_info = pd.DataFrame({"target": target, "treatment": treatment})
        target_info = target_info.groupby("treatment")["target"].agg(
            ["mean", "sum", "count"]
        )
        target_info.columns = ["target_mean", "target_sum", "target_count"]
        display(target_info)

    if "metrics_table" in evaluation_types:
        metrics = [
            "qini_auc",
            "uplift@k",
            "uplift_rel@k",
            "treatment_ones_ratios@k",
            "control_ones_ratios@k",
        ]
        thresholds = list(range(5, 26, 5))
        metrics_info = pd.DataFrame(index=metrics, columns=thresholds)
        for metric_idx, metric_name in enumerate(metrics):
            for threshold_idx, threshold in enumerate(thresholds):
                if metric_name != "qini_auc":
                    metric_func = METRICS[metric_name[:-1] + f"{threshold:02d}"]
                else:
                    metric_func = METRICS[metric_name]
                metric_val = metric_func(target, uplift, treatment)
                metrics_info.iloc[metric_idx, threshold_idx] = metric_val
        display(metrics_info)

    if "buckets_table" in evaluation_types:
        buckets_info = uplift_by_percentile(
            target, uplift, treatment, bins=n_uplift_bins
        )
        buckets_info["rel_uplift, %"] = (
            buckets_info["response_rate_treatment"]
            / buckets_info["response_rate_control"]
            - 1
        ) * 100
        display(buckets_info)

    if "tops_table" in evaluation_types:
        target, uplift, treatment = (
            np.array(target),
            np.array(uplift),
            np.array(treatment),
        )

        order = np.argsort(uplift, kind="mergesort")[::-1]

        target = np.array(target)[order]
        uplift = np.array(uplift)[order]
        treatment = np.array(treatment)[order]

        tops = []
        top_response_rate_treatment = []
        top_response_rate_control = []
        bottom_response_rate_treatment = []
        bottom_response_rate_control = []
        final_response_rate = []
        top_target_ratio_treatment = []
        top_target_ratio_control = []

        for percent in range(5, 41, 5):
            tops.append(f"{percent}%")

            percent = percent / 100
            size = int(len(uplift) * percent)

            top_y, top_t = target[:size], treatment[:size]
            bot_y, bot_t = target[size:], treatment[size:]

            y_trmnt, y_cntrl = target[treatment == 1], target[treatment == 0]
            top_y_trmnt, bot_y_trmnt = top_y[top_t == 1], bot_y[bot_t == 1]
            top_y_cntrl, bot_y_cntrl = top_y[top_t == 0], bot_y[bot_t == 0]

            final_response_rate.append(
                top_y_trmnt.mean() * percent
                + bot_y_cntrl.mean() * (1 - percent)
            )

            top_response_rate_treatment.append(top_y_trmnt.mean())
            top_response_rate_control.append(top_y_cntrl.mean())

            bottom_response_rate_treatment.append(bot_y_trmnt.mean())
            bottom_response_rate_control.append(bot_y_cntrl.mean())

            top_target_ratio_treatment.append(top_y_trmnt.sum() / y_trmnt.sum())
            top_target_ratio_control.append(top_y_cntrl.sum() / y_cntrl.sum())

        tops_info = pd.DataFrame()
        tops_info["top"] = tops
        tops_info["final_response_rate"] = final_response_rate
        tops_info["top_target_ratio_treatment"] = top_target_ratio_treatment
        tops_info["top_target_ratio_control"] = top_target_ratio_control
        tops_info["top_response_rate_treatment"] = top_response_rate_treatment
        tops_info["top_response_rate_control"] = top_response_rate_control
        tops_info[
            "bottom_response_rate_treatment"
        ] = bottom_response_rate_treatment
        tops_info["bottom_response_rate_control"] = bottom_response_rate_control
        display(tops_info)

    if "buckets_qini_plots" in evaluation_types:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        plot_uplift_by_percentile(
            y_true=target,
            uplift=uplift,
            treatment=treatment,
            strategy="overall",
            kind="bar",
            bins=n_uplift_bins,
            string_percentiles=True,
            axes=axes[0],
            draw_bars="rates",
        )

        plot_qini_curve(
            y_true=target,
            uplift=uplift,
            treatment=treatment,
            random=True,
            perfect=False,
            ax=axes[1],
        )

        axes[0].set_title("Uplift by decile", fontsize=12)
        axes[1].set_title("Qini curve", fontsize=12)

        plt.tight_layout()
        plt.show()

    if "target_ratios_plots" in evaluation_types:
        fig, axes = plt.subplots(1, 1, figsize=(7, 7))

        plot_cumulative_target_ratio(
            y_true=target,
            uplift=uplift,
            treatment=treatment,
            ax=axes,
            color_control="orange",
            color_treatment="forestgreen",
            linewidth=2,
            linestyle="-",
            random=True,
            label=None,
        )

        plt.tight_layout()
        plt.show()
