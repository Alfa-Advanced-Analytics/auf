"""Uplift Random Forest classifier implementation.

Wraps causalml's UpliftRandomForestClassifier with additional functionality
for control group probability estimation and feature name tracking.

Classes:
    AufRandomForestClassifier: Uplift random forest with control probability
        estimation and feature name tracking.

Examples:
    >>> from auf.models import AufRandomForestClassifier
    >>> import pandas as pd
    >>> import numpy as np

    >>> X = pd.DataFrame({'f1': np.random.rand(100), 'f2': np.random.rand(100)})
    >>> y = np.random.randint(0, 2, 100)
    >>> treatment = np.random.choice(['control', 'treatment'], 100)

    >>> model = AufRandomForestClassifier(
    ...     control_name='control',
    ...     n_estimators=10,
    ...     max_depth=3
    ... )
    >>> model.fit(X, y, treatment)
    >>> uplift = model.predict(X)

Notes:
    Uses CatBoostClassifier internally for control group predictions.
    Suppresses numpy divide warnings during fitting and prediction.
    Extends causalml.inference.tree.UpliftRandomForestClassifier.
"""

import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from causalml.inference.tree import UpliftRandomForestClassifier

from ..constants import RANDOM_STATE

warnings.filterwarnings("ignore")


class AufRandomForestClassifier(UpliftRandomForestClassifier):
    """Uplift Random Forest classifier with control probability estimation.

    Extends UpliftRandomForestClassifier with a base propensity model
    for control group predictions and feature name tracking.

    Attributes:
        params (dict): Model hyperparameters dictionary.
        base_model (CatBoostClassifier): Model for control group probability.
        feature_names_ (list): Names of features from training data.
        ctrl_preds_ (np.ndarray): Control group probabilities from predict.
        trmnt_preds_ (np.ndarray): Treatment group probabilities from predict.

    Examples:
        >>> from auf.models import AufRandomForestClassifier
        >>> import pandas as pd
        >>> import numpy as np

        >>> X = pd.DataFrame({'f1': np.random.rand(100), 'f2': np.random.rand(100)})
        >>> y = np.random.randint(0, 2, 100)
        >>> treatment = np.random.choice(['control', 'treatment'], 100)

        >>> model = AufRandomForestClassifier(
        ...     control_name='control',
        ...     n_estimators=10,
        ...     max_depth=3
        ... )
        >>> model.fit(X, y, treatment)
        >>> uplift = model.predict(X)

    Notes:
        Uses CatBoostClassifier internally for control group predictions.
        Suppresses numpy divide warnings during fitting and prediction.
    """

    def __init__(
        self,
        control_name,
        n_estimators=10,
        max_features=10,
        random_state=RANDOM_STATE,
        max_depth=5,
        min_samples_leaf=100,
        min_samples_treatment=10,
        n_reg=10,
        early_stopping_eval_diff_scale=1,
        evaluationFunction="KL",
        normalization=True,
        honesty=False,
        estimation_sample_size=0.5,
        n_jobs=-1,
        joblib_prefer: str = "threads",
    ):
        """Initialize AufRandomForestClassifier.

        Args:
            control_name: Name of the control group in treatment column.
            n_estimators: Number of trees in the forest. Defaults to 10.
            max_features: Number of features to consider for best split.
                Defaults to 10.
            random_state: Random seed for reproducibility.
            max_depth: Maximum depth of each tree. Defaults to 5.
            min_samples_leaf: Minimum samples required in a leaf.
                Defaults to 100.
            min_samples_treatment: Minimum samples per treatment in leaf.
                Defaults to 10.
            n_reg: Regularization parameter. Defaults to 10.
            early_stopping_eval_diff_scale: Early stopping scale factor.
                Defaults to 1.
            evaluationFunction: Evaluation metric ('KL', 'ED', 'Chi').
                Defaults to 'KL'.
            normalization: Whether to normalize scores. Defaults to True.
            honesty: Whether to use honest estimation. Defaults to False.
            estimation_sample_size: Fraction for estimation sample.
                Defaults to 0.5.
            n_jobs: Number of parallel jobs. Defaults to -1.
            joblib_prefer: Joblib backend preference. Defaults to 'threads'.
        """
        self.params = {
            "control_name": control_name,
            "n_estimators": n_estimators,
            "max_features": max_features,
            "random_state": random_state,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "min_samples_treatment": min_samples_treatment,
            "n_reg": n_reg,
            "early_stopping_eval_diff_scale": early_stopping_eval_diff_scale,
            "evaluationFunction": evaluationFunction,
            "normalization": normalization,
            "honesty": honesty,
            "estimation_sample_size": estimation_sample_size,
            "n_jobs": n_jobs,
            "joblib_prefer": joblib_prefer,
        }
        super().__init__(**self.params)
        self.base_model = CatBoostClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4, silent=True
        )
        self.ctrl_preds_ = None
        self.trmnt_preds_ = None

    def get_params(self):
        """Return model parameters dictionary.

        Returns:
            Dictionary of model hyperparameters.
        """
        return self.params

    def fit(self, X: pd.DataFrame, y: pd.Series, treatment: pd.Series):
        """Fit the uplift random forest model.

        Args:
            X: Training features DataFrame.
            y: Binary target values.
            treatment: Treatment group labels.

        Returns:
            Fitted model instance.

        Examples:
            >>> model.fit(X_train, y_train, treatment_train)
        """
        self.feature_names_ = X.columns.tolist()
        fltr = treatment.astype("str") == self.params["control_name"]
        self.base_model.fit(X=X[fltr], y=y[fltr])
        with np.errstate(divide="ignore", invalid="ignore"):
            super().fit(
                X=X.values, y=y.values, treatment=treatment.astype("str").values
            )
        return self

    def predict(self, X: pd.DataFrame):
        """Predict uplift scores for samples in X.

        Args:
            X: Feature DataFrame for prediction.

        Returns:
            Array of uplift predictions.

        Examples:
            >>> uplift = model.predict(X_test)

        Notes:
            Also populates ctrl_preds_ and trmnt_preds_ attributes.
        """
        self.ctrl_preds_ = self.base_model.predict_proba(X)[:, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            uplift = super().predict(X.values)
            if len(self.classes_) == 2:
                uplift = uplift.reshape(-1)
        self.trmnt_preds_ = self.ctrl_preds_ + uplift
        return uplift
