"""Uplift modeling estimators and wrappers for AUF library.

Provides a collection of uplift models including S-Learner, X-Learner,
Tree-based and Forest-based classifiers, along with unified wrappers
for consistent prediction interface.

Classes:
    AufModel: Wrapper for binary and multitreamnt uplift models.
    AufSoloModel: S-Learner (Single Model) with multi-treatment support.
    AufXLearner: X-Learner with multi-treatment and propensity weighting.
    AufTreeClassifier: Single uplift decision tree classifier.
    AufRandomForestClassifier: Uplift random forest classifier.

Examples:
    >>> from auf.models import AufSoloModel, AufModel
    >>> from catboost import CatBoostClassifier

    >>> # Direct usage of AufSoloModel
    >>> model = AufSoloModel(estimator=CatBoostClassifier(verbose=False))
    >>> model.fit(X_train, y_train, treatment_train, control_group='control')
    >>> uplift = model.predict(X_test)

    >>> # Using AufModel wrapper
    >>> wrapped_model = AufModel(
    ...     model=model,
    ...     model_name='AufSoloModel',
    ...     features=['feature_1', 'feature_2'],
    ...     uplift_prediction_type='abs'
    ... )

Notes:
    All models follow scikit-learn convention with fit/predict interface.
    Models store control and treatment predictions in ctrl_preds_ and trmnt_preds_.
"""

from .auf_forest import AufRandomForestClassifier
from .auf_model import AufModel
from .auf_tree import AufTreeClassifier
from .auf_x_learner import AufXLearner

__all__ = [
    "AufTreeClassifier",
    "AufRandomForestClassifier",
    "AufXLearner",
    "AufModel",
]
