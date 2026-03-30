"""Data validation utilities for uplift modeling.

Provides functions to verify data quality, treatment balance, and detect
potential leaks between features and target/treatment variables.

Functions:
    check_leaks_v2: Detects features leaking target/treatment via CatBoost bootstrap.
    check_correlations: Identifies highly correlated feature pairs exceeding threshold.
    check_train_val_test_split: Validates treatment and target balance across splits.
    check_nans: Filters features with excessive missing values.
    check_too_less_unique_value: Filters features with insufficient unique values.
    process_too_much_categories: Collapses rare categories into '_others_'.
    check_bernoulli_dependence: Chi-square test for binary variable dependence.
    check_bernoulli_equal_means: Welch's t-test for binary sample mean equality.

Examples:
    >>> from auf.data.checks import check_leaks_v2, check_correlations
    >>> mapper = {'segm': 'segment', 'target': 'converted', 'treatment': 'campaign'}
    >>> leaks, clean, _ = check_leaks_v2(df, mapper, features, 'treatment')
    >>> if leaks: print(f"Found {len(leaks)} leaking features")
    >>> corr_pairs, filtered = check_correlations(df, features, max_abs_corr=0.95)

Notes:
    check_leaks_v2 uses CatBoostClassifier with bootstrap to assess feature importance.
    Returns empty lists when no issues detected.
"""

import typing as tp

import numpy as np
import pandas as pd
import scipy.stats as ss
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score

from ..constants import BOOTSTRAP_REPEATS, RANDOM_STATE
from ..log import get_logger

logger = get_logger(verbosity=1)


def check_bernoulli_dependence(x: np.array, y: np.array, alpha: float = 0.05):
    """Tests dependence between two binary variables using chi-squared test.

    Args:
        x: Binary array-like observations (0/1).
        y: Binary array-like observations (0/1).
        alpha: Significance level (0 < alpha < 1). Defaults to 0.05.

    Returns:
        dict: Test results with keys:
            - 'stat': Chi-squared statistic
            - 'critical_stat': Critical value at (1-alpha) quantile
            - 'dependent': Boolean decision (True if dependent)
            - 'pvalue': p-value

    Raises:
        AssertionError: If lengths differ, values not binary, or alpha invalid.

    Examples:
        >>> result = check_bernoulli_dependence([0,1,0,1], [0,0,1,1])
        >>> if result['dependent']:
        ...     print("Variables are dependent.")
    """
    assert len(x) == len(y), "Lengths of arrays x and y must be equal."
    assert set(np.unique(x)) == set(
        [0, 1]
    ), "Array x must contain 2 unique values 0 and 1 and only them."
    assert set(np.unique(y)) == set(
        [0, 1]
    ), "Array y must contain 2 unique values 0 and 1 and only them."
    assert (
        0.0 < alpha < 1.0
    ), "Significance level alpha must be from 0 to 1 (exclusively)."

    n = len(x)
    co_matr = np.array([[0, 0], [0, 0]])
    for i in [0, 1]:
        for j in [0, 1]:
            co_matr[i, j] = np.sum((x == i) * (y == j))

    stat_sum = 0.0
    for i in [0, 1]:
        for j in [0, 1]:
            x_i = np.sum(co_matr[i, :])
            y_j = np.sum(co_matr[:, j])
            stat_sum += co_matr[i, j] ** 2 / (x_i * y_j)

    stat = n * (stat_sum - 1)

    critical_stat = ss.chi2(df=1).ppf(1 - alpha)
    pvalue = 1 - ss.chi2(df=1).cdf(stat)

    return {
        "stat": stat,
        "critical_stat": critical_stat,
        "dependent": stat > critical_stat,
        "pvalue": pvalue,
    }


def check_bernoulli_equal_means(x: np.array, y: np.array, alpha: float = 0.05):
    """Tests equality of means between two binary samples using Welch's t-test.

    Args:
        x: Binary array-like observations (0/1).
        y: Binary array-like observations (0/1).
        alpha: Significance level. Defaults to 0.05.

    Returns:
        dict: Test results with keys:
            - 'stat': t-statistic
            - 'equals': Boolean decision (True if means equal)
            - 'pvalue': p-value

    Raises:
        AssertionError: If inputs invalid.

    Examples:
        >>> result = check_bernoulli_equal_means([0,1,0,1], [0,0,1,1])
        >>> if not result['equals']:
        ...     print("Means differ significantly.")
    """
    stat, pvalue = ss.ttest_ind(
        x,
        y,
        equal_var=False,
        nan_policy="propagate",
        alternative="two-sided",
    )

    return {
        "stat": stat,
        "equals": pvalue >= alpha,
        "pvalue": pvalue,
    }


def check_nans(
    df: pd.DataFrame, feature_cols: tp.List[str], max_nan_ratio: float = 0.95
):
    """Filters features exceeding maximum NaN ratio.

    Args:
        df: DataFrame with features.
        feature_cols: List of feature names to check.
        max_nan_ratio: Maximum allowable NaN ratio (0-1). Defaults to 0.95.

    Returns:
        list: Feature names with NaN ratio <= threshold.

    Examples:
        >>> valid_features = check_nans(df, features, max_nan_ratio=0.9)
    """
    return [f for f in feature_cols if df[f].isna().mean() <= max_nan_ratio]


def check_too_less_unique_value(df: pd.DataFrame, feature_cols: tp.List[str]):
    """Filters features with fewer than 2 unique values.

    Args:
        df: DataFrame with features.
        feature_cols: List of feature names to check.

    Returns:
        list: Feature names with >=2 unique values.

    Examples:
        >>> valid_features = check_too_less_unique_value(df, features)
    """
    return [f for f in feature_cols if df[f].nunique() >= 2]


def process_too_much_categories(
    df: pd.DataFrame, feature_cols: tp.List[str], max_categories_count: int = 20
):
    """Collapses rare categories into '_others_' in-place.

    Modifies df in-place. For categorical features (object dtype) with more than
    max_categories_count unique values, replaces least frequent categories with '_others_'.

    Args:
        df: DataFrame to modify.
        feature_cols: List of feature names to process.
        max_categories_count: Maximum number of unique values allowed. Defaults to 20.

    Examples:
        >>> process_too_much_categories(df, cat_features, max_categories_count=15)
    """
    for col in feature_cols:
        if (
            df[col].dtype == "object"
            and df[col].nunique() >= max_categories_count
        ):
            sorted_values_desc = df[col].value_counts().sort_values()[::-1]
            others = sorted_values_desc[
                max_categories_count - 1 :
            ].index.tolist()
            df.loc[df[col].isin(others), col] = "_others_"


def check_leaks_v2(
    df: pd.DataFrame,
    base_cols_mapper: tp.Dict[str, str],
    feature_cols: tp.List[str],
    col_to_check: str,
    alpha: float = 0.05,
    max_val_roc_auc: float = 0.55,
    early_stopping: int = None,
):
    """Detect leaking target/treatment features using CatBoost and bootstrap.

    Builds small CatBoost models to predict col_to_check from features.
    Uses bootstrap to estimate ROC-AUC distribution and identify
    features causing significant leaks.

    Args:
        df: DataFrame with features and target/treatment columns.
        base_cols_mapper: Mapping of standard column names to actual names.
        feature_cols: Features to check for leaks.
        col_to_check: Column to predict ('target' or 'treatment').
        alpha: Significance level for bootstrap quantile. Defaults to 0.05.
        max_val_roc_auc: Maximum allowable ROC-AUC (leak threshold). Defaults to 0.55.
        early_stopping: Consecutive stages to stop if no leak found. Defaults to None.

    Returns:
        tuple:
            - leaks_roc_aucs: List of (feature, roc_auc) for leaking features
            - not_leaks: List of clean feature names
            - all_features_roc_aucs: List of (feature, roc_auc) for all checked features

    Raises:
        AssertionError: If col_to_check invalid, alpha out of range, or max_val_roc_auc <=0.5.

    Examples:
        >>> mapper = {'segm': 'segment', 'target': 'converted', 'treatment': 'campaign'}
        >>> leaks_roc_aucs, not_leaks, all_features_roc_aucs = check_leaks_v2(
        ...     df, mapper, features, 'treatment', alpha=0.05
        ... )
    """
    assert col_to_check in [
        "target",
        "treatment",
    ], f"Parameter col_to_check must be in ['target', 'treatment'], but col_to_check='{col_to_check}'"
    assert (
        0.0 < alpha and alpha < 1.0
    ), f"Parameter alpha must be in [0.0, 1.0], but is {alpha:.2f}."
    assert (
        max_val_roc_auc > 0.5
    ), f"Parameter max_val_roc_auc must be greater than 0.5, but is {max_val_roc_auc:.2f}."

    features = feature_cols.copy()
    all_features_roc_aucs = []

    segm_col = base_cols_mapper["segm"]
    col_to_check = base_cols_mapper[col_to_check]

    # Split data into train and validation sets
    feature_df_train = df.loc[
        df[segm_col] == "train", features + [col_to_check]
    ]
    if len(feature_df_train) > 50_000:
        feature_df_train = feature_df_train.sample(
            n=max(50_000, int(feature_df_train.shape[0] * 0.2))
        )
    feature_df_val = df.loc[df[segm_col] == "val", features + [col_to_check]]

    # Define bootstrap sample size
    bootstrap_data_size = min(feature_df_val.shape[0], int(df.shape[0] * 0.1))
    if len(feature_df_val) > 15_000:
        bootstrap_data_size = max(15_000, bootstrap_data_size)
    else:
        bootstrap_data_size = feature_df_val.shape[0]

    # Main algorithm loop
    while len(features) > 0:
        # Split features into batches
        shuffled_features = np.random.choice(
            features, size=len(features), replace=False
        )
        BATCH_SIZE = min(250, len(shuffled_features))
        feature_batches = [
            shuffled_features[i : i + BATCH_SIZE]
            for i in range(0, len(shuffled_features), BATCH_SIZE)
        ]

        feature_df_val_sample = feature_df_val.sample(
            bootstrap_data_size, random_state=RANDOM_STATE
        )

        # Process each batch
        for batch_features in feature_batches:
            train_pool = Pool(
                feature_df_train[batch_features], feature_df_train[col_to_check]
            )
            checker = CatBoostClassifier(
                n_estimators=10,
                depth=2,
                learning_rate=0.1,
                silent=True,
                random_seed=RANDOM_STATE,
                cat_features=[
                    f for f in batch_features if df[f].dtype == "object"
                ],
                ignored_features=None,
            )

            checker.fit(train_pool)
            val_preds = checker.predict_proba(
                feature_df_val_sample[batch_features]
            )
            val_true = feature_df_val_sample[col_to_check].values

            roc_auc_kwargs = {"multi_class": "ovr", "average": "macro"}
            if val_preds.shape[1] == 2:
                val_preds = val_preds[:, 1]
                roc_auc_kwargs = {}

            bootstrap_roc_aucs = np.zeros(shape=BOOTSTRAP_REPEATS)
            rng = np.random.RandomState(seed=RANDOM_STATE)

            idxs = rng.choice(
                len(val_true), (BOOTSTRAP_REPEATS, len(val_true)), replace=True
            )
            bootstrap_roc_aucs = np.array(
                [
                    roc_auc_score(val_true[i], val_preds[i], **roc_auc_kwargs)
                    for i in idxs
                ]
            )

            q = np.quantile(bootstrap_roc_aucs, q=1 - alpha)
            q = max(q, 1 - q)  # revert labels if needed
            top_leaking_feature = batch_features[
                np.argmax(checker.feature_importances_)
            ]
            features.remove(top_leaking_feature)
            all_features_roc_aucs.append((top_leaking_feature, q))

            if early_stopping is not None:
                if early_stopping == 0:
                    break
                if q < max_val_roc_auc:
                    early_stopping -= 1

        if early_stopping is not None:
            if early_stopping == 0:
                break

    leaks_roc_aucs = [
        (f, q) for f, q in all_features_roc_aucs if q >= max_val_roc_auc
    ]
    not_leaks = [
        f for f, q in all_features_roc_aucs if q < max_val_roc_auc
    ] + features

    leaks_roc_aucs = sorted(leaks_roc_aucs, key=lambda p: -p[1])
    all_features_roc_aucs = sorted(all_features_roc_aucs, key=lambda p: -p[1])

    return leaks_roc_aucs, not_leaks, all_features_roc_aucs


def check_correlations(
    df: pd.DataFrame, feature_cols: tp.List[str], max_abs_corr: float = 0.95
):
    """Filters features with pairwise correlation exceeding threshold.

    Iteratively identifies and removes features from highly correlated pairs.
    Prefers to keep features that appear earlier in the list.

    Args:
        df: DataFrame with features.
        feature_cols: List of feature names to check.
        max_abs_corr: Maximum absolute correlation allowed (0-1). Defaults to 0.95.

    Returns:
        tuple:
            - too_correlated: List of (feature1, feature2) pairs exceeding threshold
            - features: Filtered list of feature names

    Raises:
        AssertionError: If max_abs_corr not in (0, 1).

    Examples:
        >>> too_correlated, features = check_correlations(
        ...     df, features, max_abs_corr=0.9
        ... )
    """
    assert (
        0.0 < max_abs_corr and max_abs_corr < 1.0
    ), f"Parameter max_abs_corr must be from 0 to 1, but is {max_abs_corr:.2f}"

    corr_matr = df[feature_cols].corr()

    too_correlated: list[tuple[str, str]] = []
    features = feature_cols.copy()
    something_deleted = True

    while something_deleted:
        features_to_remove = []
        something_deleted = False

        for f in features:
            if f in features_to_remove or df[f].dtype == "object":
                continue

            i = corr_matr.columns.get_loc(f)
            abs_corrs = corr_matr.iloc[i, i + 1 :].abs()
            bad_corrs_cols = list(abs_corrs[abs_corrs > max_abs_corr].index)
            bad_corrs_cols = [
                g
                for g in bad_corrs_cols
                if g not in features_to_remove and g in features
            ]
            too_correlated.extend([(f, g) for g in bad_corrs_cols])
            features_to_remove.extend(bad_corrs_cols)

        if features_to_remove:
            something_deleted = True
            features = [f for f in features if f not in features_to_remove]

    return too_correlated, features


def check_train_val_test_split(
    df: pd.DataFrame,
    segm_col: str,
    target_col: str,
    treatment_col: str,
    treatment_groups_mapper: tp.Dict[tp.Any, int],
):
    """Validates treatment and target balance across train/val/test splits.

    Checks three conditions using Welch's t-test:
    1. Treatment ratio equality across splits
    2. Target rate equality in treatment group across splits
    3. Target rate equality in control group across splits

    Prints warnings if any check fails.

    Args:
        df: DataFrame with data.
        segm_col: Column name indicating splits ('train', 'val', 'test').
        target_col: Binary target column name.
        treatment_col: Treatment column name.
        treatment_groups_mapper: Mapping from treatment values to 0/1.

    Raises:
        AssertionError: If required columns missing from df.

    Examples:
        >>> mapper = {'treatment_a': 1, 'control': 0}
        >>> check_train_val_test_split(
        ...     df, 'segment', 'converted', 'campaign', mapper
        ... )
    """
    for col in [segm_col, target_col, treatment_col]:
        assert col in df.columns

    segm = df[segm_col]
    target = df[target_col]
    treatment = df[treatment_col].map(treatment_groups_mapper)

    mask_train = (segm == "train").values
    mask_val = (segm == "val").values
    mask_test = (segm == "test").values

    # Check segments to have the same ratio of treatment = 1
    treatment_train = treatment[mask_train]
    treatment_val = treatment[mask_val]
    treatment_test = treatment[mask_test]

    result_train_val = check_bernoulli_equal_means(
        treatment_train, treatment_val, alpha=0.05
    )
    result_val_test = check_bernoulli_equal_means(
        treatment_val, treatment_test, alpha=0.05
    )

    if not result_train_val["equals"] or not result_val_test["equals"]:
        logger.info(
            "\n\033[1;31mTreatment ratio should be the same in train, val, test splits\033[0m"
        )

    # Check segments to have the same ratio of target = 1 in treatment group
    target_train_treatment = target[mask_train & (treatment == 1)]
    target_val_treatment = target[mask_val & (treatment == 1)]
    target_test_treatment = target[mask_test & (treatment == 1)]

    result_train_val = check_bernoulli_equal_means(
        target_train_treatment, target_val_treatment, alpha=0.05
    )
    result_val_test = check_bernoulli_equal_means(
        target_val_treatment, target_test_treatment, alpha=0.05
    )

    if not result_train_val["equals"] or not result_val_test["equals"]:
        logger.info(
            "\n\033[1;31mTarget rate ratio should be the same in train, val, test splits in treatment group\033[0m"
        )

    # Check segments to have the same ratio of target = 1 in control group
    target_train_control = target[mask_train & (treatment == 0)]
    target_val_control = target[mask_val & (treatment == 0)]
    target_test_control = target[mask_test & (treatment == 0)]

    result_train_val = check_bernoulli_equal_means(
        target_train_control, target_val_control, alpha=0.05
    )
    result_val_test = check_bernoulli_equal_means(
        target_val_control, target_test_control, alpha=0.05
    )

    if not result_train_val["equals"] or not result_val_test["equals"]:
        logger.info(
            "\n\033[1;31mTarget rate ratio should be the same in train, val, test splits in control group\033[0m"
        )
