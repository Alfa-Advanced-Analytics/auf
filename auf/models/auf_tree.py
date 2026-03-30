"""Single uplift decision tree classifier implementation.

Wraps causalml's UpliftTreeClassifier with additional functionality for
feature name tracking and control/treatment probability extraction.

Classes:
    AufTreeClassifier: Single uplift decision tree with feature name tracking
        and probability extraction.

Examples:
    >>> from auf.models import AufTreeClassifier
    >>> import pandas as pd
    >>> import numpy as np

    >>> X = pd.DataFrame({'f1': np.random.rand(100), 'f2': np.random.rand(100)})
    >>> y = np.random.randint(0, 2, 100)
    >>> treatment = np.random.choice(['control', 'treatment'], 100)

    >>> model = AufTreeClassifier(
    ...     control_name='control',
    ...     max_depth=3,
    ...     min_samples_leaf=50
    ... )
    >>> model.fit(X, y, treatment)
    >>> uplift = model.predict(X)

Notes:
    Single tree variant of AufRandomForestClassifier.
    Useful for interpretability and feature importance analysis.
    Extends causalml.inference.tree.UpliftTreeClassifier.
"""

import warnings

import numpy as np
import pandas as pd
from causalml.inference.tree import UpliftTreeClassifier

from ..constants import RANDOM_STATE

warnings.filterwarnings("ignore")


class AufTreeClassifier(UpliftTreeClassifier):
    """Single uplift decision tree classifier.

    Wraps UpliftTreeClassifier with feature name tracking and
    control/treatment probability extraction.

    Attributes:
        params (dict): Model hyperparameters dictionary.
        feature_names_ (list): Names of features from training data.
        ctrl_preds_ (np.ndarray): Control group probabilities from predict.
        trmnt_preds_ (np.ndarray): Treatment group probabilities from predict.

    Examples:
        >>> from auf.models import AufTreeClassifier
        >>> import pandas as pd
        >>> import numpy as np

        >>> X = pd.DataFrame({'f1': np.random.rand(100), 'f2': np.random.rand(100)})
        >>> y = np.random.randint(0, 2, 100)
        >>> treatment = np.random.choice(['control', 'treatment'], 100)

        >>> model = AufTreeClassifier(
        ...     control_name='control',
        ...     max_depth=3,
        ...     min_samples_leaf=50
        ... )
        >>> model.fit(X, y, treatment)
        >>> uplift = model.predict(X)

    Notes:
        Single tree variant of AufRandomForestClassifier.
        Useful for interpretability and feature importance analysis.
    """

    def __init__(
        self,
        control_name,
        max_features=None,
        max_depth=3,
        min_samples_leaf=100,
        min_samples_treatment=10,
        n_reg=100,
        early_stopping_eval_diff_scale=1,
        evaluationFunction="KL",
        normalization=True,
        honesty=False,
        estimation_sample_size=0.5,
        random_state=RANDOM_STATE,
    ):
        """Initialize AufTreeClassifier.

        Args:
            control_name: Name of the control group in treatment column.
            max_features: Number of features to consider for best split.
                None means all features.
            max_depth: Maximum depth of the tree. Defaults to 3.
            min_samples_leaf: Minimum samples required in a leaf.
                Defaults to 100.
            min_samples_treatment: Minimum samples per treatment in leaf.
                Defaults to 10.
            n_reg: Regularization parameter. Defaults to 100.
            early_stopping_eval_diff_scale: Early stopping scale factor.
                Defaults to 1.
            evaluationFunction: Evaluation metric ('KL', 'ED', 'Chi').
                Defaults to 'KL'.
            normalization: Whether to normalize scores. Defaults to True.
            honesty: Whether to use honest estimation. Defaults to False.
            estimation_sample_size: Fraction for estimation sample.
                Defaults to 0.5.
            random_state: Random seed for reproducibility.
        """
        self.params = {
            "control_name": control_name,
            "max_features": max_features,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "min_samples_treatment": min_samples_treatment,
            "n_reg": n_reg,
            "early_stopping_eval_diff_scale": early_stopping_eval_diff_scale,
            "evaluationFunction": evaluationFunction,
            "normalization": normalization,
            "honesty": honesty,
            "estimation_sample_size": estimation_sample_size,
            "random_state": random_state,
        }

        super().__init__(**self.params)
        self.params["n_estimators"] = 1
        self.ctrl_preds_ = None
        self.trmnt_preds_ = None

    def get_params(self):
        """Return model parameters dictionary.

        Returns:
            Dictionary of model hyperparameters.
        """
        return self.params

    def fit(self, X: pd.DataFrame, y: pd.Series, treatment: pd.Series):
        """Fit the uplift decision tree.

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
            Array of uplift predictions (treatment - control).

        Examples:
            >>> uplift = model.predict(X_test)

        Notes:
            Also populates ctrl_preds_ and trmnt_preds_ attributes.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            uplift = super().predict(X.values)

        self.ctrl_preds_ = uplift[:, 0]
        self.trmnt_preds_ = uplift[:, 1]

        return uplift[:, 1] - uplift[:, 0]
