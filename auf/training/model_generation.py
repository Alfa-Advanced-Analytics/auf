"""Model generation utilities for hyperparameter optimization.

Provides functions to generate model instances with parameters
suggested by Optuna trials or specified explicitly.

Functions:
    generate_catboost_params: Generate CatBoost parameters from Optuna trial.
    generate_tree_params: Generate uplift tree parameters from Optuna trial.
    generate_forest_params: Generate uplift random forest parameters from
        Optuna trial.
    generate_model: Generate model instance with parameters from trial or dict.
    generate_multitreatment_model: Generate multi-treatment model instance
        with parameters.

Examples:
    >>> from auf.training.model_generation import generate_model
    >>> from auf.models import AufSoloModel
    >>>
    >>> model = generate_model(
    ...     trial=trial,
    ...     model_class=AufSoloModel,
    ...     training_mode='light'
    ... )

Notes:
    Parameter ranges depend on training_mode ('light', 'medium', 'hard'):
    - light: fewer iterations, larger learning rate, deeper trees.
    - medium: balanced search with moderate iteration counts.
    - hard: more iterations, smaller learning rate, shallower trees.
    Uses CatBoostClassifier and CatBoostRegressor as base learners.
"""

import typing as tp

from catboost import CatBoostClassifier, CatBoostRegressor

from ..constants import RANDOM_STATE


def generate_catboost_params(
    trial, training_mode: str, suffix: str
) -> tp.Dict[str, tp.Any]:
    """Generate CatBoost parameters from Optuna trial.

    Args:
        trial: Optuna trial object.
        training_mode: Search mode ('light', 'medium', 'hard').
        suffix: Suffix for parameter names.

    Returns:
        Dictionary of CatBoost parameters.

    Notes:
        Parameter ranges depend on training_mode:
        - light: iterations 50-200, depth 2-10, learning_rate 0.01-0.2
        - medium: iterations 50-500, depth 2-8, learning_rate 0.005-0.1
        - hard: iterations 50-1000, depth 2-6, learning_rate 0.0005-0.05
    """
    params = {
        "l2_leaf_reg": trial.suggest_float(
            "l2_leaf_reg" + suffix, 1e-3, 1e3, log=True
        ),
        "min_data_in_leaf": trial.suggest_int(
            "min_data_in_leaf" + suffix, 1e1, 1e3, log=True
        ),
        "random_strength": trial.suggest_float(
            "random_strength" + suffix, 1e-3, 1e3, log=True
        ),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type" + suffix, ["Bayesian", "Bernoulli", "MVS"]
        ),
        "grow_policy": trial.suggest_categorical(
            "grow_policy" + suffix, ["SymmetricTree", "Depthwise"]
        ),
        "early_stopping_rounds": 50,
        "random_state": RANDOM_STATE,
        "eval_metric": "AUC",
        "verbose": 0,
    }

    if params["bootstrap_type"] == "Bernoulli":
        params["subsample"] = trial.suggest_float("subsample" + suffix, 0.1, 1)
    elif params["bootstrap_type"] == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float(
            "bagging_temperature" + suffix, 0, 10
        )

    if training_mode == "light":
        params["iterations"] = trial.suggest_int(
            "iterations" + suffix, 50, 200, log=True
        )
        params["learning_rate"] = trial.suggest_float(
            "learning_rate" + suffix, 0.01, 0.2, log=True
        )
        params["depth"] = trial.suggest_int("depth" + suffix, 2, 10)

    elif training_mode == "medium":
        params["iterations"] = trial.suggest_int(
            "iterations" + suffix, 50, 500, log=True
        )
        params["learning_rate"] = trial.suggest_float(
            "learning_rate" + suffix, 0.005, 0.1, log=True
        )
        params["depth"] = trial.suggest_int("depth" + suffix, 2, 8)

    elif training_mode == "hard":
        params["iterations"] = trial.suggest_int(
            "iterations" + suffix, 50, 1000, log=True
        )
        params["learning_rate"] = trial.suggest_float(
            "learning_rate" + suffix, 0.0005, 0.05, log=True
        )
        params["depth"] = trial.suggest_int("depth" + suffix, 2, 6)

    return params


def generate_tree_params(
    trial, training_mode: str, use_multitreatment: bool = False
) -> tp.Dict[str, tp.Any]:
    """Generate uplift tree parameters from Optuna trial.

    Args:
        trial: Optuna trial object.
        training_mode: Search mode ('light', 'medium', 'hard').
        use_multitreatment: Whether to use multi-treatment parameters.

    Returns:
        Dictionary of tree parameters.

    Notes:
        Parameter ranges depend on training_mode:
        - light: depth 2-10, features in split 2-5
        - medium: depth 2-8, features in split 2-10
        - hard: depth 2-6, features in split 2-15
    """
    params = {
        "estimation_sample_size": trial.suggest_float(
            "estimation_sample_size", 0.1, 0.9
        ),
        "evaluationFunction": trial.suggest_categorical(
            "evaluationFunction",
            ["KL", "ED", "Chi", "CTS", "DDP", "IT", "CIT", "IDDP"]
            if not use_multitreatment
            else ["KL", "ED", "Chi", "CTS"],
        ),
    }

    if training_mode == "light":
        params["max_depth"] = trial.suggest_int("max_depth", 2, 5, log=True)
        params["max_features"] = trial.suggest_int("max_features", 2, 5)

    elif training_mode == "medium":
        params["max_depth"] = trial.suggest_int("max_depth", 2, 7, log=True)
        params["max_features"] = trial.suggest_int("max_features", 2, 10)

    elif training_mode == "hard":
        params["max_depth"] = trial.suggest_int("max_depth", 2, 11, log=True)
        params["max_features"] = trial.suggest_int("max_features", 2, 15)

    return params


def generate_forest_params(
    trial, training_mode: str, use_multitreatment: bool = False
) -> tp.Dict[str, tp.Any]:
    """Generate uplift random forest parameters from Optuna trial.

    Args:
        trial: Optuna trial object.
        training_mode: Search mode ('light', 'medium', 'hard').
        use_multitreatment: Whether to use multi-treatment parameters.

    Returns:
        Dictionary of forest parameters.

    Notes:
        Parameter ranges depend on training_mode:
        - light: estimators 10-100, depth 2-10, features in split 2-5
        - medium: estimators 10-200, depth 2-8, features in split 2-10
        - hard: estimators 10-400, depth 2-6, features in split 2-15
    """
    params = {
        "estimation_sample_size": trial.suggest_float(
            "estimation_sample_size", 0.1, 0.9
        ),
        "evaluationFunction": trial.suggest_categorical(
            "evaluationFunction",
            ["KL", "ED", "Chi", "CTS", "DDP", "IT", "CIT", "IDDP"]
            if not use_multitreatment
            else ["KL", "ED", "Chi", "CTS"],
        ),
    }

    if training_mode == "light":
        params["max_depth"] = trial.suggest_int("max_depth", 2, 5, log=True)
        params["max_features"] = trial.suggest_int("max_features", 2, 5)
        params["n_estimators"] = trial.suggest_int("n_estimators", 10, 100)

    elif training_mode == "medium":
        params["max_depth"] = trial.suggest_int("max_depth", 2, 7, log=True)
        params["max_features"] = trial.suggest_int("max_features", 2, 10)
        params["n_estimators"] = trial.suggest_int("n_estimators", 10, 200)

    elif training_mode == "hard":
        params["max_depth"] = trial.suggest_int("max_depth", 2, 11, log=True)
        params["max_features"] = trial.suggest_int("max_features", 2, 15)
        params["n_estimators"] = trial.suggest_int("n_estimators", 10, 400)

    return params


def generate_model(
    trial,
    model_class,
    params: tp.Optional[tp.Dict[str, tp.Any]] = None,
    training_mode: tp.Optional[str] = None,
):
    """Generate model instance with parameters from trial or dict.

    Args:
        trial: Optuna trial object (can be None if params provided).
        model_class: Model class to instantiate.
        params: Optional dictionary of parameters.
        training_mode: Search mode for parameter generation.

    Returns:
        Model instance with specified parameters.

    Raises:
        ValueError: If model_class is not supported.

    Examples:
        >>> model = generate_model(trial, AufSoloModel, training_mode='light')
    """
    model_class_name = model_class.__name__

    if model_class_name == "CatBoostClassifier":
        if params is None:
            params = generate_catboost_params(trial, training_mode, suffix="")
        else:
            params["early_stopping_rounds"] = 50
            params["random_state"] = RANDOM_STATE
            params["eval_metric"] = "AUC"
            params["verbose"] = 0
        model = CatBoostClassifier(**params)

    elif model_class_name == "SoloModel":
        if params is None:
            params = generate_catboost_params(trial, training_mode, suffix="")
            method = trial.suggest_categorical(
                "method", ["dummy", "treatment_interaction"]
            )
        else:
            method = params["method"]
            del params["method"]
            params["early_stopping_rounds"] = 50
            params["random_state"] = RANDOM_STATE
            params["eval_metric"] = "AUC"
            params["verbose"] = 0

        model = model_class(
            method=method, estimator=CatBoostClassifier(**params)
        )

    elif model_class_name == "TwoModels":
        if params is None:
            params1 = generate_catboost_params(trial, training_mode, suffix="")
            params2 = generate_catboost_params(trial, training_mode, suffix="2")
            method = trial.suggest_categorical(
                "method", ["vanilla", "ddr_control", "ddr_treatment"]
            )
        else:
            method = params["method"]
            del params["method"]

            params1 = {k: v for k, v in params.items() if k[-1] != "2"}
            params1["early_stopping_rounds"] = 50
            params1["random_state"] = RANDOM_STATE
            params1["eval_metric"] = "AUC"
            params1["verbose"] = 0

            params2 = {k[:-1]: v for k, v in params.items() if k[-1] == "2"}
            params2["early_stopping_rounds"] = 50
            params2["random_state"] = RANDOM_STATE
            params2["eval_metric"] = "AUC"
            params2["verbose"] = 0

        model = model_class(
            method=method,
            estimator_trmnt=CatBoostClassifier(**params1),
            estimator_ctrl=CatBoostClassifier(**params2),
        )

    elif model_class_name == "AufXLearner":
        if params is None:
            base_model_params = generate_catboost_params(
                trial, training_mode, suffix="1"
            )
            uplift_model_params = generate_catboost_params(
                trial, training_mode, suffix="2"
            )
            group_model_params = generate_catboost_params(
                trial, training_mode, suffix="3"
            )
        else:
            base_model_params = {
                k[:-1]: v for k, v in params.items() if k[-1] == "1"
            }
            base_model_params["early_stopping_rounds"] = 50
            base_model_params["random_state"] = RANDOM_STATE
            base_model_params["eval_metric"] = "AUC"
            base_model_params["verbose"] = 0

            uplift_model_params = {
                k[:-1]: v for k, v in params.items() if k[-1] == "2"
            }
            uplift_model_params["early_stopping_rounds"] = 50
            uplift_model_params["random_state"] = RANDOM_STATE
            uplift_model_params["eval_metric"] = "RMSE"
            uplift_model_params["verbose"] = 0

            group_model_params = {
                k[:-1]: v for k, v in params.items() if k[-1] == "3"
            }
            group_model_params["early_stopping_rounds"] = 50
            group_model_params["random_state"] = RANDOM_STATE
            group_model_params["eval_metric"] = "AUC"
            group_model_params["verbose"] = 0

        model = model_class(
            model=CatBoostClassifier(**base_model_params),
            uplift_model=CatBoostRegressor(**uplift_model_params),
            map_groups={"control": 0, "treatment": 1},
            group_model=CatBoostClassifier(**group_model_params),
        )

    elif model_class_name == "AufTreeClassifier":
        if params is None:
            params = generate_tree_params(trial, training_mode)
        model = model_class(**params, control_name="0")

    elif model_class_name == "AufRandomForestClassifier":
        if params is None:
            params = generate_forest_params(trial, training_mode)
        model = model_class(**params, control_name="0")

    return model


def generate_multitreatment_model(
    trial,
    model_class,
    params: tp.Optional[tp.Dict[str, tp.Any]] = None,
    training_mode: tp.Optional[str] = None,
):
    """Generate multi-treatment model instance with parameters.

    Args:
        trial: Optuna trial object (can be None if params provided).
        model_class: Model class to instantiate.
        params: Optional dictionary of parameters.
        training_mode: Search mode for parameter generation.

    Returns:
        Model instance with specified parameters.

    Raises:
        ValueError: If model_class is not supported.

    Examples:
        >>> model = generate_multitreatment_model(
        ...     trial, UpliftRandomForestClassifier, training_mode='light'
        ... )
    """
    model_class_name = model_class.__name__

    if model_class_name in ["BaseSClassifier", "BaseTClassifier"]:
        if params is None:
            params = generate_catboost_params(trial, training_mode, suffix="")
        else:
            params["early_stopping_rounds"] = 50
            params["random_state"] = RANDOM_STATE
            params["eval_metric"] = "AUC"
            params["verbose"] = 0

        model = model_class(learner=CatBoostClassifier(**params))

    elif model_class_name == "BaseXClassifier":
        if params is None:
            params1 = generate_catboost_params(trial, training_mode, suffix="")
            params2 = generate_catboost_params(trial, training_mode, suffix="2")
            params2["eval_metric"] = "RMSE"
        else:
            params1 = {k: v for k, v in params.items() if k[-1] != "2"}
            params1["early_stopping_rounds"] = 50
            params1["random_state"] = RANDOM_STATE
            params1["eval_metric"] = "AUC"
            params1["verbose"] = 0

            params2 = {k[:-1]: v for k, v in params.items() if k[-1] == "2"}
            params2["early_stopping_rounds"] = 50
            params2["random_state"] = RANDOM_STATE
            params2["eval_metric"] = "RMSE"
            params2["verbose"] = 0

        model = model_class(
            outcome_learner=CatBoostClassifier(**params1),
            effect_learner=CatBoostRegressor(**params2),
        )

    elif model_class_name == "UpliftTreeClassifier":
        if params is None:
            params = generate_tree_params(
                trial, training_mode, use_multitreatment=True
            )
        model = model_class(**params, control_name="control")

    elif model_class_name == "UpliftRandomForestClassifier":
        if params is None:
            params = generate_forest_params(
                trial, training_mode, use_multitreatment=True
            )
        model = model_class(**params, control_name="control")

    else:
        raise ValueError(f"Wrong model_class value: {model_class}")

    return model
