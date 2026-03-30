"""Model training and hyperparameter optimization utilities.

Provides functions for training uplift models with support for
hyperparameter tuning via Optuna and multi-treatment scenarios.

Functions:
    fit_model: Train a model and return wrapped AufModel instance.
    generate_model_from_classes: Train models with hyperparameter search.

Classes:
    OptunaOptimizer: Optuna-based hyperparameter optimizer.

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
    Supports both binary and multi-treatment uplift modeling.
    Integrates with Optuna for hyperparameter optimization.
"""

from .fitting import fit_model, generate_model_from_classes
from .gridsearch import OptunaOptimizer

__all__ = [
    "fit_model",
    "generate_model_from_classes",
    "OptunaOptimizer",
]
