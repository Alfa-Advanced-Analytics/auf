"""X-Learner implementation with multi-treatment and propensity support.

Provides AufXLearner for uplift modeling using the X-Learner approach with
separate response and effect models. Supports propensity score weighting.

Classes:
    AufXLearner: X-Learner with multi-treatment support and optional
        propensity model for weighting effect predictions.

Examples:
    >>> from auf.models import AufXLearner
    >>> from catboost import CatBoostClassifier, CatBoostRegressor
    >>> import pandas as pd
    >>> import numpy as np

    >>> X = pd.DataFrame({'f1': np.random.rand(100), 'f2': np.random.rand(100)})
    >>> y = np.random.randint(0, 2, 100)
    >>> treatment = np.random.choice([0, 1], 100)

    >>> model = AufXLearner(
    ...     model=CatBoostClassifier(iterations=100, verbose=False),
    ...     uplift_model=CatBoostRegressor(iterations=50, verbose=False),
    ...     map_groups={'control': 0, 'treatment1': 1},
    ...     features=['f1', 'f2']
    ... )
    >>> model.fit(X, y, treatment)
    >>> uplift = model.predict(X)

Notes:
    Requires 'control' key in map_groups dictionary.
    Supports CatBoost models with cat_features parameter.
    Propensity weighting available via group_model parameter.
    Returns array for single treatment, dict for multiple treatments.
"""

import typing as tp

import numpy as np
import pandas as pd


class AufXLearner:
    """X-Learner with multi-treatment support and propensity weighting.

    Implements the X-Learner algorithm which trains separate response
    models for treatment and control groups, then trains effect models
    on the residuals. Supports propensity score weighting.

    Attributes:
        trmnt_preds_ (np.ndarray): Treatment group effect predictions.
        ctrl_preds_ (np.ndarray): Control group effect predictions.

    Examples:
        >>> from auf.models import AufXLearner
        >>> from catboost import CatBoostClassifier, CatBoostRegressor
        >>> import pandas as pd
        >>> import numpy as np

        >>> X = pd.DataFrame({'f1': np.random.rand(100), 'f2': np.random.rand(100)})
        >>> y = np.random.randint(0, 2, 100)
        >>> treatment = np.random.choice([0, 1], 100)

        >>> model = AufXLearner(
        ...     model=CatBoostClassifier(iterations=100, verbose=False),
        ...     uplift_model=CatBoostRegressor(iterations=50, verbose=False),
        ...     map_groups={'control': 0, 'treatment1': 1},
        ...     features=['f1', 'f2']
        ... )
        >>> model.fit(X, y, treatment)
        >>> uplift = model.predict(X)

    Notes:
        Requires 'control' key in map_groups dictionary.
        Supports CatBoost models with cat_features parameter.
        Propensity weighting available via group_model parameter.

    References:
        Kunzel, S. R., et al. (2019). Meta-learners for Estimating
        Heterogeneous Treatment Effects using Machine Learning.
        PNAS, 116(10), 4156-4165.
    """

    def __init__(
        self,
        model,
        uplift_model,
        map_groups=None,
        features=None,
        cat_features=None,
        group_model=None,
    ):
        """Initialize AufXLearner.

        Args:
            model: Estimator for response models (treatment/control).
            uplift_model: Estimator for effect models (regressor recommended).
            map_groups: Dictionary mapping group names to treatment values.
                Must contain 'control' key. Example:
                {'control': 0, 'treatment1': 1, 'treatment2': 2}.
            features: List of feature names for response models.
            cat_features: List of categorical feature names.
            group_model: Optional model for propensity score estimation.

        Raises:
            AssertionError: If 'control' not in map_groups keys.
        """
        self._model = {}
        self._uplift_model = {}
        self._params = {
            "model": model.copy(),
            "uplift_model": uplift_model.copy(),
        }

        if map_groups is None:
            map_groups = {"control": 0, "treatment1": 1, "treatment2": 2}

        if group_model is not None:
            self._params["group_model"] = group_model.copy()

        self._map_groups = map_groups
        self._features = features
        self._response_features = features
        self._cat_features = cat_features
        self._feature_importances = None
        self._group_model = group_model

        self.trmnt_preds_ = None
        self.ctrl_preds_ = None

        assert (
            "control" in self._map_groups.keys()
        ), "map_groups must contain 'control' key"

        for key in self._map_groups.keys():
            self._model[key] = model.copy()
            self._uplift_model[key] = {
                "control": uplift_model.copy(),
                "treatment": uplift_model.copy(),
            }

    def get_params(self):
        """Return model parameters dictionary.

        Returns:
            Dictionary of model hyperparameters.
        """
        return self._params

    def get_feature_importances(self):
        """Get aggregated feature importances.

        Returns:
            Dictionary mapping treatment groups to feature importance dicts.

        Examples:
            >>> importances = model.get_feature_importances()
        """
        if self._feature_importances is None:
            return self._get_feature_importances()
        return self._feature_importances

    def _get_feature_importances(self):
        self._feature_importances = {}

        for key in self._map_groups.keys():
            if key != "control":
                fi_cntrl = self._uplift_model[key][
                    "control"
                ].get_feature_importance(prettified=True)
                fi_treat = self._uplift_model[key][
                    "treatment"
                ].get_feature_importance(prettified=True)
                fi_cntrl_resp = self._model["control"].get_feature_importance(
                    prettified=True
                )
                fi_treat_resp = self._model[key].get_feature_importance(
                    prettified=True
                )

                fi = pd.merge(fi_cntrl, fi_treat, on="Feature Id", how="inner")
                fi_resp = pd.merge(
                    fi_cntrl_resp, fi_treat_resp, on="Feature Id", how="inner"
                )
                fi = pd.merge(fi, fi_resp, on="Feature Id", how="inner")

                drop_cols = [col for col in fi.columns if col != "Feature Id"]

                fi["Importances"] = fi[drop_cols].sum(axis=1)
                fi.Importances = fi.Importances.apply(lambda x: round(x, 3))

                fi.index = fi["Feature Id"]
                fi = fi.drop(["Feature Id"] + drop_cols, axis=1)
                fi = fi.to_dict()["Importances"]

                self._feature_importances[key] = fi

        return self._feature_importances

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        treatment: pd.Series,
        cost_dict: tp.Optional[tp.Dict[tp.Any, float]] = None,
        **kwargs,
    ):
        """Fit the X-Learner model.

        Args:
            X: Training features DataFrame.
            y: Binary target values.
            treatment: Treatment group labels.
            cost_dict: Dictionary mapping group names to cost/benefit values.
                Used for weighted uplift calculation.
            **kwargs: Additional parameters passed to model.fit().

        Returns:
            Fitted model instance.

        Raises:
            AssertionError: If cost_dict keys don't match map_groups keys.
            AssertionError: If X doesn't contain required features.

        Examples:
            >>> model.fit(X_train, y_train, treatment_train)

        Notes:
            Trains response models for each group, then effect models
            on residuals. Supports CatBoost-specific parameters.
        """
        if cost_dict is None:
            cost_dict = {}
            for group in self._map_groups:
                cost_dict[group] = 1

        assert (
            len(
                set(self._map_groups.keys()).symmetric_difference(
                    cost_dict.keys()
                )
            )
            == 0
        ), (
            f"cost_dict keys {cost_dict.keys()} don't match "
            f"map_groups keys {self._map_groups.keys()}"
        )

        if self._features is None:
            self._features = X.columns.tolist()

        if self._cat_features is None:
            self._cat_features = [
                f for f in self._features if X[f].dtype == "object"
            ]

        if self._response_features is None:
            self._response_features = X.columns.tolist()

        assert set(X.columns) == set(self._features) | set(
            self._response_features
        ), "X must contain all features from self._features and self._response_features"

        y = np.array(y)
        treatment = np.array(treatment)

        if self._group_model is not None:
            self._group_model = self._group_model.fit(
                X=X.loc[:, self._features],
                y=treatment,
                cat_features=self._cat_features,
                **kwargs,
            )

        # Train response models
        for key in self._map_groups.keys():
            treat_name = self._map_groups[key]
            mask = treatment == treat_name
            self._model[key] = self._model[key].fit(
                X=X.loc[mask, self._features],
                y=y[mask],
                cat_features=self._cat_features,
                **kwargs,
            )

        cntrl_name = self._map_groups["control"]
        cntrl_mask = treatment == cntrl_name
        cntrl_X_resp = X.loc[cntrl_mask, self._response_features]
        cntrl_X = X.loc[cntrl_mask, self._features]

        for key in self._map_groups.keys():
            if key != "control":
                trmnt_name = self._map_groups[key]

                sj = cost_dict[key]
                s0 = cost_dict["control"]

                # Fit uplift model for control group
                cntrl_probas = self._model[key].predict_proba(cntrl_X_resp)[
                    :, 1
                ]
                cntrl_uplift = sj * cntrl_probas - s0 * y[cntrl_mask]

                self._uplift_model[key]["control"] = self._uplift_model[key][
                    "control"
                ].fit(
                    X=cntrl_X,
                    y=cntrl_uplift,
                    cat_features=self._cat_features,
                    **kwargs,
                )

                # Fit uplift model for treatment group
                trmnt_mask = treatment == trmnt_name
                trmnt_X_resp = X.loc[trmnt_mask, self._response_features]
                trmnt_X = X.loc[trmnt_mask, self._features]

                trmnt_probas = self._model["control"].predict_proba(
                    trmnt_X_resp
                )[:, 1]
                trmnt_uplift = -s0 * trmnt_probas + sj * y[trmnt_mask]

                self._uplift_model[key]["treatment"] = self._uplift_model[key][
                    "treatment"
                ].fit(
                    X=trmnt_X,
                    y=trmnt_uplift,
                    cat_features=self._cat_features,
                    **kwargs,
                )

        return self

    def predict(self, X: pd.DataFrame, eps: float = 0.5):
        """Predict uplift scores.

        Args:
            X: Feature DataFrame for prediction.
            eps: Weight for control effect model vs treatment effect model.
                Only used if group_model is None. Defaults to 0.5.

        Returns:
            Array (single treatment) or dict (multiple treatments) of uplifts.

        Examples:
            >>> uplift = model.predict(X_test)
            >>> # Single treatment: returns array
            >>> # Multiple treatments: returns dict

        Notes:
            If group_model is provided, uses its predictions for weighting.
            Final uplift = eps * control_effect + (1 - eps) * treatment_effect.
        """
        uplifts = {}

        for key in self._map_groups.keys():
            if key != "control":
                if self._group_model is not None:
                    eps = self._group_model.predict_proba(X[self._features])[
                        :, 0
                    ]

                t0 = self._uplift_model[key]["control"].predict(
                    X[self._features]
                )
                t1 = self._uplift_model[key]["treatment"].predict(
                    X[self._features]
                )
                uplifts[key] = eps * t0 + (1 - eps) * t1

        if len(uplifts) == 1:
            treatment_group_name = [
                group for group in uplifts if group != "control"
            ][0]
            assert len(uplifts[treatment_group_name]) == X.shape[0], (
                f"Prediction length {len(uplifts[treatment_group_name])} "
                f"doesn't match X length {X.shape[0]}"
            )
            return uplifts[treatment_group_name]

        return uplifts
