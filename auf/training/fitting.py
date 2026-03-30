"""Model fitting and training utilities.

Provides functions to train uplift models with various configurations,
including multi-treatment support and feature selection.

Functions:
    fit_model: Train a model and return wrapped AufModel.
    generate_model_from_classes: Train models with optional hyperparameter
        optimization via search class.
    get_default_params_dict: Get default parameters for a model class.

Examples:
    >>> from auf.training import fit_model
    >>> from catboost import CatBoostClassifier

    >>> model = CatBoostClassifier(verbose=False)
    >>> wrapped = fit_model(
    ...     estimator=model,
    ...     df_train=train_df,
    ...     features=['f1', 'f2'],
    ...     target_col='target',
    ...     treatment_col='treatment'
    ... )

Notes:
    Supports SoloModel, TwoModels, AufTreeClassifier,
    AufRandomForestClassifier, and AufXLearner.
"""

import typing as tp

import pandas as pd
from sklearn.pipeline import Pipeline

from ..models import AufModel


def fit_model(
    estimator: Pipeline,
    df_train: pd.DataFrame,
    features: tp.List[str],
    target_col: str,
    treatment_col: str,
    uplift_type: tp.Optional[str] = "abs",
    treatment_groups: tp.Optional[tp.List[str]] = None,
):
    """Train model and return wrapped AufModel instance.

    Args:
        estimator: Model to train. Should be compatible with scikit-learn
            interface.
        df_train: Training data containing features and target.
        features: List of feature names for training.
        target_col: Name of the target column.
        treatment_col: Name of the treatment column.
        uplift_type: Uplift calculation type.
            'abs': treatment_rate - control_rate.
            'rel': treatment_rate / control_rate - 1.
            Defaults to 'abs'.
        treatment_groups: List of treatment groups for multi-treatment
            modeling. Defaults to None.

    Returns:
        Wrapped model containing trained estimator and metadata.

    Examples:
        >>> from auf.training import fit_model
        >>> from catboost import CatBoostClassifier

        >>> model = CatBoostClassifier(verbose=False)
        >>> wrapped = fit_model(model, train_df, ['f1', 'f2'], 'target', 'treatment')
    """
    auf_model = AufModel(
        estimator,
        type(estimator).__name__,
        features,
        uplift_prediction_type=None if treatment_groups else uplift_type,
        treatment_groups=treatment_groups if treatment_groups else None,
    )

    auf_model.fit(
        X=df_train[features],
        y=df_train[target_col],
        treatment=df_train[treatment_col],
    )

    return auf_model


def generate_model_from_classes(
    model_class: object,
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    df_test: pd.DataFrame,
    features: tp.Dict[str, tp.List[str]],
    target_col: str,
    treatment_col: str,
    feature_nums: tp.List[int],
    timeout: int,
    metric: object,
    search_class: object,
    overfit_metric: object = None,
    treatment_groups: tp.Optional[tp.List[str]] = None,
    training_mode: tp.Literal["light", "medium", "hard"] = "light",
):
    """Train models with optional hyperparameter optimization.

    Generates and trains models from specified class using provided
    datasets. Supports hyperparameter tuning via search_class.

    Args:
        model_class: Model class to instantiate and train.
        df_train: Training dataset with features and target.
        df_valid: Validation dataset for hyperparameter tuning.
        df_test: Test dataset for final evaluation.
        features: Dictionary mapping feature set names to feature lists.
        target_col: Name of the target column.
        treatment_col: Name of the treatment column.
        feature_nums: List of feature counts to use for training.
        timeout: Maximum time in seconds for hyperparameter search.
        metric: Metric function to optimize.
        use_default_params: If True, use default parameters.
            If False, perform hyperparameter tuning.
        search_class: Class for hyperparameter optimization.
        overfit_metric: Optional metric to penalize overfitting.
        treatment_groups: List of treatment groups for multi-treatment.
        training_mode: Hyperparameter search complexity mode.
            'light': Fast search with fewer iterations.
            'medium': Balanced search.
            'hard': Extensive search with more iterations.

    Returns:
        Dictionary mapping feature set names to lists of trained models.

    Raises:
        AssertionError: If feature_nums exceeds available feature count.

    Examples:
        >>> from auf.training import generate_model_from_classes
        >>> from auf.models import AufSoloModel
        >>> from auf.training import OptunaOptimizer

        >>> results = generate_model_from_classes(
        ...     model_class=AufSoloModel,
        ...     df_train=train,
        ...     df_valid=valid,
        ...     df_test=test,
        ...     features={'set1': ['f1', 'f2']},
        ...     target_col='target',
        ...     treatment_col='treatment',
        ...     feature_nums=[2],
        ...     timeout=60,
        ...     metric=qini_auc_score,
        ...     use_default_params=True,
        ...     search_class=OptunaOptimizer
        ... )

    Notes:
        For binary treatment, trains models for each uplift type
        (propensity, abs, rel) depending on model class.
    """
    fast_fit_result = dict()

    for key in features.keys():
        if feature_nums is None:
            feature_nums = [len(features[key])]

        result_n = list()

        for n in feature_nums:
            assert n <= len(features[key]), (
                f"Feature count {n} in feature_nums cannot exceed "
                f"available features {len(features[key])}"
            )

            if treatment_groups:
                finder_class = search_class(
                    df_train,
                    df_valid,
                    metric,
                    treatment_col,
                    target_col,
                    overfit_metric,
                    training_mode,
                )

                mc = finder_class.find_best_params(
                    model_class,
                    features[key][:n],
                    timeout,
                    treatment_groups=treatment_groups,
                )

                result_n.append(
                    fit_model(
                        mc,
                        df_train,
                        features[key][:n],
                        target_col,
                        treatment_col,
                        uplift_type=None,
                        treatment_groups=treatment_groups,
                    )
                )

                continue

            for uplift_type in ["propensity", "abs", "rel"]:
                if (
                    model_class.__name__ == "CatBoostClassifier"
                    and uplift_type != "propensity"
                ):
                    break

                if (
                    model_class.__name__ != "CatBoostClassifier"
                    and uplift_type == "propensity"
                ):
                    continue

                if (
                    uplift_type == "rel"
                    and model_class.__name__ == "AufXLearner"
                ):
                    break

                finder_class = search_class(
                    df_train,
                    df_valid,
                    metric,
                    treatment_col,
                    target_col,
                    overfit_metric,
                    training_mode,
                )

                mc = finder_class.find_best_params(
                    model_class, features[key][:n], timeout
                )

                result_n.append(
                    fit_model(
                        mc,
                        df_train,
                        features[key][:n],
                        target_col,
                        treatment_col,
                        uplift_type,
                        treatment_groups=None,
                    )
                )

        fast_fit_result[key] = result_n

    return fast_fit_result
