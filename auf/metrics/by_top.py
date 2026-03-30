"""Top-k percent based uplift metrics.

Implements metrics that evaluate uplift model performance on top-k
or top-percentile segments of the ranked population.

Functions:
    uplift_at_k: Uplift score at top k percent observations by uplift.
    qini_auc_score_clip_at_k: Qini AUC with relative uplift constraint.
    control_treatment_ones_ratios_at_k: Target ratio F1-score at top and bottom.
    abs_rel_uplift_growth_at_k: Uplift growth ratio between top and bottom.

Examples:
    >>> import numpy as np
    >>> from auf.metrics.by_top import uplift_at_k

    >>> y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    >>> uplift = np.array([0.1, 0.4, 0.3, 0.8, 0.2, 0.5, 0.6, 0.7])
    >>> treatment = np.array([0, 1, 0, 1, 0, 1, 0, 1])

    >>> score = uplift_at_k(y_true, uplift, treatment, k=0.3)
"""

import typing as tp

import numpy as np
from sklearn.utils.validation import check_consistent_length
from sklift.metrics import qini_auc_score
from sklift.utils import check_is_binary


def uplift_at_k(
    y_true: tp.Sequence[float],
    uplift: tp.Sequence[float],
    treatment: tp.Sequence[float],
    strategy: tp.Literal["overall", "by_group"] = "overall",
    k: tp.Union[float, int] = 0.3,
    output_transform: tp.Callable = lambda x, y: x - y,
):
    """Compute uplift score at first k percent observations.

    Calculates uplift using only the top k percent observations ranked by
    predicted uplift, with customizable output transformation.

    Args:
        y_true: True binary target values.
        uplift: Predicted uplift scores.
        treatment: Binary treatment labels (0=control, 1=treatment).
        strategy: Ranking strategy.
            'overall': Global ranking across all samples.
            'by_group': Separate ranking within each group.
            Defaults to 'overall'.
        k: Proportion (float in (0, 1)) or absolute count (int) of top
            observations to include. Defaults to 0.3.
        output_transform: Function applied to (treatment_rate, control_rate).
            Defaults to treatment_rate - control_rate.

    Returns:
        Transformed uplift score at top k percent observations by uplift.

    Raises:
        ValueError: If strategy is invalid.
        ValueError: If k is out of valid range.

    Examples:
        >>> import numpy as np
        >>> from auf.metrics import uplift_at_k

        >>> y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        >>> uplift = np.array([0.1, 0.4, 0.3, 0.8, 0.2, 0.5, 0.6, 0.7])
        >>> treatment = np.array([0, 1, 0, 1, 0, 1, 0, 1])

        >>> score = uplift_at_k(y_true, uplift, treatment, k=0.3)

    Notes:
        Extends scikit-uplift implementation.
        For 'by_group' strategy, k applies separately to each group.
        Use output_transform=lambda x, y: x/y-1 for ranking by relative uplift.
    """
    check_consistent_length(y_true, uplift, treatment)
    check_is_binary(treatment)
    check_is_binary(y_true)
    y_true, uplift, treatment = (
        np.array(y_true),
        np.array(uplift),
        np.array(treatment),
    )

    strategy_methods = ["overall", "by_group"]
    if strategy not in strategy_methods:
        raise ValueError(
            f"Uplift score supports only calculating methods in {strategy_methods},"
            f" got {strategy}."
        )

    n_samples = len(y_true)
    order = np.argsort(uplift, kind="mergesort")[::-1]
    _, treatment_counts = np.unique(treatment, return_counts=True)
    n_samples_ctrl = treatment_counts[0]
    n_samples_trmnt = treatment_counts[1]

    k_type = np.asarray(k).dtype.kind

    if (
        k_type == "i"
        and (k >= n_samples or k <= 0)
        or k_type == "f"
        and (k <= 0 or k >= 1)
    ):
        raise ValueError(
            f"k={k} should be either positive and smaller"
            f" than the number of samples {n_samples} or a float in the "
            f"(0, 1) range"
        )

    if k_type not in ("i", "f"):
        raise ValueError(f"Invalid value for k: {k_type}")

    if strategy == "overall":
        if k_type == "f":
            n_size = int(n_samples * k)
        else:
            n_size = k

        score_ctrl = y_true[order][:n_size][
            treatment[order][:n_size] == 0
        ].mean()
        score_trmnt = y_true[order][:n_size][
            treatment[order][:n_size] == 1
        ].mean()

    else:
        if k_type == "f":
            n_ctrl = int((treatment == 0).sum() * k)
            n_trmnt = int((treatment == 1).sum() * k)

        else:
            n_ctrl = k
            n_trmnt = k

        if n_ctrl > n_samples_ctrl:
            raise ValueError(
                f"With k={k}, the number of the first k observations"
                " bigger than the number of samples"
                f"in the control group: {n_samples_ctrl}"
            )
        if n_trmnt > n_samples_trmnt:
            raise ValueError(
                f"With k={k}, the number of the first k observations"
                " bigger than the number of samples"
                f"in the treatment group: {n_samples_ctrl}"
            )

        score_ctrl = y_true[order][treatment[order] == 0][:n_ctrl].mean()
        score_trmnt = y_true[order][treatment[order] == 1][:n_trmnt].mean()

    return output_transform(score_trmnt, score_ctrl)


def qini_auc_score_clip_at_k(
    y_true: tp.Sequence[float],
    uplift: tp.Sequence[float],
    treatment: tp.Sequence[float],
    strategy: tp.Literal["overall", "by_group"] = "overall",
    k: tp.Union[float, int] = 0.2,
    multiplier_threshold: tp.Union[float, int] = 0.9,
):
    """Compute Qini AUC score with relative uplift constraint.

    Returns Qini AUC only if relative uplift at k exceeds threshold,
    otherwise returns -1 as penalty.

    Args:
        y_true: True binary target values.
        uplift: Predicted uplift scores.
        treatment: Binary treatment labels (0=control, 1=treatment).
        strategy: Ranking strategy. Defaults to 'overall'.
        k: Proportion for relative uplift calculation. Defaults to 0.2.
        multiplier_threshold: Multiplier threshold for relative uplift comparison.
            Defaults to 0.9.

    Returns:
        Qini AUC score if condition met, otherwise -1.

    Examples:
        >>> import numpy as np
        >>> from auf.metrics import qini_auc_score_clip_at_k

        >>> y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        >>> uplift = np.array([0.1, 0.4, 0.3, 0.8, 0.2, 0.5, 0.6, 0.7])
        >>> treatment = np.array([0, 1, 0, 1, 0, 1, 0, 1])

        >>> score = qini_auc_score_clip_at_k(y_true, uplift, treatment, k=0.3)

    Notes:
        Condition: rel_uplift_at_k > multiplier_threshold * rel_uplift_at_99%.
        Penalizes models with inconsistent uplift distribution.
    """
    rel_all = uplift_at_k(
        y_true, uplift, treatment, strategy, 0.99, lambda x, y: x / y - 1
    )
    rel_k = uplift_at_k(
        y_true, uplift, treatment, strategy, k, lambda x, y: x / y - 1
    )
    if rel_k <= multiplier_threshold * rel_all:
        return -1
    else:
        return qini_auc_score(y_true, uplift, treatment)


def control_treatment_ones_ratios_at_k(
    y_true: tp.Sequence[float],
    uplift: tp.Sequence[float],
    treatment: tp.Sequence[float],
    strategy: tp.Literal["overall", "by_group"] = "overall",
    k: tp.Union[float, int] = 0.3,
    target_type: tp.Literal["control", "treatment", "both"] = "both",
):
    """Compute target ratio score at k observations.

    Calculates ratios of treatment targets in top-k and control targets
    in bottom-(100-k), returning either individual ratios or their F1-score.

    Args:
        y_true: True binary target values.
        uplift: Predicted uplift scores.
        treatment: Binary treatment labels (0=control, 1=treatment).
        strategy: Ranking strategy. Defaults to 'overall'.
        k: Proportion or count of observations. Defaults to 0.3.
        target_type: Which ratio to return.
            'control': Control targets ratio in bottom.
            'treatment': Treatment targets ratio in top.
            'both': F1-score of both ratios.
            Defaults to 'both'.

    Returns:
        Target ratio score based on target_type.

    Raises:
        ValueError: If strategy is invalid.
        ValueError: If k is out of valid range.

    Examples:
        >>> import numpy as np
        >>> from auf.metrics import control_treatment_ones_ratios_at_k

        >>> y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        >>> uplift = np.array([0.1, 0.4, 0.3, 0.8, 0.2, 0.5, 0.6, 0.7])
        >>> treatment = np.array([0, 1, 0, 1, 0, 1, 0, 1])

        >>> score = control_treatment_ones_ratios_at_k(y_true, uplift, treatment, k=0.3)

    Notes:
        Treatment ratio: treatment targets in top / all treatment targets.
        Control ratio: control targets in bottom / all control targets.
        F1-score uses harmonic mean of both ratios.
    """
    check_consistent_length(y_true, uplift, treatment)
    check_is_binary(treatment)
    check_is_binary(y_true)
    y_true, uplift, treatment = (
        np.array(y_true),
        np.array(uplift),
        np.array(treatment),
    )

    strategy_methods = ["overall", "by_group"]
    if strategy not in strategy_methods:
        raise ValueError(
            f"Uplift score supports only calculating methods in {strategy_methods},"
            f" got {strategy}."
        )

    n_samples = len(y_true)
    order = np.argsort(uplift, kind="mergesort")[::-1]
    _, treatment_counts = np.unique(treatment, return_counts=True)
    n_samples_ctrl = treatment_counts[0]
    n_samples_trmnt = treatment_counts[1]

    k_type = np.asarray(k).dtype.kind

    if (
        k_type == "i"
        and (k >= n_samples or k <= 0)
        or k_type == "f"
        and (k <= 0 or k >= 1)
    ):
        raise ValueError(
            f"k={k} should be either positive and smaller"
            f" than the number of samples {n_samples} or a float in the "
            f"(0, 1) range"
        )

    if k_type not in ("i", "f"):
        raise ValueError(f"Invalid value for k: {k_type}")

    ones_ctrl = y_true[treatment == 0].sum()
    ones_trmnt = y_true[treatment == 1].sum()

    if strategy == "overall":
        if k_type == "f":
            n_size = int(n_samples * k)
        else:
            n_size = k

        down_ones_ctrl = y_true[order][n_size:][
            treatment[order][n_size:] == 0
        ].sum()
        top_ones_trmnt = y_true[order][:n_size][
            treatment[order][:n_size] == 1
        ].sum()

        score_ctrl = down_ones_ctrl / ones_ctrl
        score_trmnt = top_ones_trmnt / ones_trmnt

    else:
        if k_type == "f":
            n_ctrl = int((treatment == 0).sum() * k)
            n_trmnt = int((treatment == 1).sum() * k)
        else:
            n_ctrl = k
            n_trmnt = k

        if n_ctrl > n_samples_ctrl:
            raise ValueError(
                f"With k={k}, the number of the first k observations"
                " bigger than the number of samples"
                f"in the control group: {n_samples_ctrl}"
            )
        if n_trmnt > n_samples_trmnt:
            raise ValueError(
                f"With k={k}, the number of the first k observations"
                " bigger than the number of samples"
                f"in the treatment group: {n_samples_ctrl}"
            )

        down_ones_ctrl = y_true[order][treatment[order] == 0][n_ctrl:].sum()
        top_ones_trmnt = y_true[order][treatment[order] == 1][:n_trmnt].sum()

        score_ctrl = down_ones_ctrl / ones_ctrl
        score_trmnt = top_ones_trmnt / ones_trmnt

    if target_type == "control":
        score = score_ctrl
    elif target_type == "treatment":
        score = score_trmnt
    else:
        score = 2 * (score_ctrl * score_trmnt) / (score_ctrl + score_trmnt)
    return score


def abs_rel_uplift_growth_at_k(
    y_true: tp.Sequence[float],
    uplift: tp.Sequence[float],
    treatment: tp.Sequence[float],
    strategy: tp.Literal["overall", "by_group"] = "overall",
    k: tp.Union[float, int] = 0.3,
    uplift_type: tp.Literal["abs", "rel", "both"] = "both",
):
    """Compute uplift growth ratio between top and bottom samples.

    Calculates the ratio of uplift in top-k versus bottom-(100-k) samples,
    supporting both absolute and relative uplift measures.

    Args:
        y_true: True binary target values.
        uplift: Predicted uplift scores.
        treatment: Binary treatment labels (0=control, 1=treatment).
        strategy: Ranking strategy. Defaults to 'overall'.
        k: Proportion or count of observations. Defaults to 0.3.
        uplift_type: Which uplift measure to use.
            'abs': Absolute uplift growth ratio.
            'rel': Relative uplift growth ratio.
            'both': Sum of both ratios.
            Defaults to 'both'.

    Returns:
        Uplift growth ratio score.

    Raises:
        AssertionError: If uplift_type is invalid.
        ValueError: If strategy is invalid.
        ValueError: If k is out of valid range.

    Examples:
        >>> import numpy as np
        >>> from auf.metrics import abs_rel_uplift_growth_at_k

        >>> y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        >>> uplift = np.array([0.1, 0.4, 0.3, 0.8, 0.2, 0.5, 0.6, 0.7])
        >>> treatment = np.array([0, 1, 0, 1, 0, 1, 0, 1])

        >>> score = abs_rel_uplift_growth_at_k(y_true, uplift, treatment, k=0.3)

    Notes:
        Absolute uplift = treatment_rate - control_rate.
        Relative uplift = treatment_rate / control_rate - 1.
        Growth ratio = top_uplift / bottom_uplift.
    """
    assert uplift_type in ["abs", "rel", "both"], f"{uplift_type}"

    check_consistent_length(y_true, uplift, treatment)
    check_is_binary(treatment)
    check_is_binary(y_true)
    y_true, uplift, treatment = (
        np.array(y_true),
        np.array(uplift),
        np.array(treatment),
    )

    strategy_methods = ["overall", "by_group"]
    if strategy not in strategy_methods:
        raise ValueError(
            f"Uplift score supports only calculating methods in {strategy_methods},"
            f" got {strategy}."
        )

    n_samples = len(y_true)
    order = np.argsort(uplift, kind="mergesort")[::-1]
    _, treatment_counts = np.unique(treatment, return_counts=True)
    n_samples_ctrl = treatment_counts[0]
    n_samples_trmnt = treatment_counts[1]

    k_type = np.asarray(k).dtype.kind

    if (
        k_type == "i"
        and (k >= n_samples or k <= 0)
        or k_type == "f"
        and (k <= 0 or k >= 1)
    ):
        raise ValueError(
            f"k={k} should be either positive and smaller"
            f" than the number of samples {n_samples} or a float in the "
            f"(0, 1) range"
        )

    if k_type not in ("i", "f"):
        raise ValueError(f"Invalid value for k: {k_type}")

    if strategy == "overall":
        if k_type == "f":
            n_size = int(n_samples * k)
        else:
            n_size = k

        tr_ctrl_top = y_true[order][:n_size][
            treatment[order][:n_size] == 0
        ].mean()
        tr_trmnt_top = y_true[order][:n_size][
            treatment[order][:n_size] == 1
        ].mean()

        tr_ctrl_bottom = y_true[order][n_size:][
            treatment[order][n_size:] == 0
        ].mean()
        tr_trmnt_bottom = y_true[order][n_size:][
            treatment[order][n_size:] == 1
        ].mean()

    else:
        if k_type == "f":
            n_ctrl = int((treatment == 0).sum() * k)
            n_trmnt = int((treatment == 1).sum() * k)
        else:
            n_ctrl = k
            n_trmnt = k

        if n_ctrl > n_samples_ctrl:
            raise ValueError(
                f"With k={k}, the number of the first k observations"
                " bigger than the number of samples"
                f"in the control group: {n_samples_ctrl}"
            )
        if n_trmnt > n_samples_trmnt:
            raise ValueError(
                f"With k={k}, the number of the first k observations"
                " bigger than the number of samples"
                f"in the treatment group: {n_samples_ctrl}"
            )

        tr_ctrl_top = y_true[order][treatment[order] == 0][:n_ctrl].mean()
        tr_trmnt_top = y_true[order][treatment[order] == 1][:n_trmnt].mean()

        tr_ctrl_bottom = y_true[order][treatment[order] == 0][n_ctrl:].mean()
        tr_trmnt_bottom = y_true[order][treatment[order] == 1][n_trmnt:].mean()

    abs_uplift_growth = (tr_trmnt_top - tr_ctrl_top) / (
        tr_trmnt_bottom - tr_ctrl_bottom
    )
    rel_uplift_growth = (tr_trmnt_top / tr_ctrl_top - 1) / (
        tr_trmnt_bottom / tr_ctrl_bottom - 1
    )

    if uplift_type == "abs":
        score = abs_uplift_growth
    elif uplift_type == "rel":
        score = rel_uplift_growth
    else:
        score = abs_uplift_growth + rel_uplift_growth

    return score
