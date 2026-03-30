"""Model wrapper for unified prediction interface.

Provides AufModel for binary and multi-treatment uplift prediction
scenarios with consistent API across different model types.

Classes:
    AufModel: Wrapper both for binary and multi-treatment uplift models.

Functions:


Examples:
    >>> from auf.models import AufModel, AufSoloModel
    >>> from catboost import CatBoostClassifier

    >>> uplift_model = AufSoloModel(CatBoostClassifier(verbose=False))
    >>> wrapped = AufModel(
    ...     model=uplift_model,
    ...     model_name='AufSoloModel',
    ...     features=['age', 'income', 'city'],
    ...     uplift_prediction_type='abs'
    ... )
    >>> wrapped.fit(X_train, y_train, treatment_train)
    >>> predictions = wrapped.predict(X_test)

Notes:
    Binary treatment uplift modeling supports CatBoostClassifier, SoloModel,
        TwoModels, AufXLearner, AufRandomForestClassifier, and AufTreeClassifier.
    Multitreatment uplift modeling supports BaseSClassifier, BaseTClassifier,
        BaseXClassifier, UpliftTreeClassifier, and UpliftRandomForestClassifier.
"""

import typing as tp

import numpy as np
import pandas as pd


class AufModel:
    """Unified wrapper for binary and multi-treatment uplift models.

    Provides a consistent interface for fitting, predicting, and
    extracting feature importances across different uplift model
    implementations.

    Attributes:
        _model (tp.Any): Wrapped model instance.
        _model_name (str): Model type identifier.
        _features (tp.List[str]): List of feature names used for prediction.
        _feature_importances (np.array): Cached feature importances.
        _uplift_prediction_type (str): 'abs' for absolute or 'rel' for
            relative uplift.
        _is_propensity (bool): True if model is a propensity model.
        _is_multitreatment (bool): True if model supports multiple treatments.
        _treatment_groups (tp.List[str]): List of treatment group names.
        _n_treatments (int): Number of treatment groups.

    Examples:
        >>> from auf.models import AufModel
        >>> from sklift.models import SoloModel
        >>> from catboost import CatBoostClassifier

        >>> estimator = CatBoostClassifier(verbose=False)
        >>> sm = SoloModel(estimator)
        >>> wrapped = AufModel(sm, 'SoloModel', features=['f1', 'f2'])
        >>> wrapped.fit(X_train, y_train, treatment_train)
        >>> scores = wrapped.predict(X_test)

    Notes:
        The wrapper automatically detects binary vs multi-treatment mode
        based on the model_name. Feature importances are aggregated across
        internal models for S/T/X classifiers.
    """

    _available_binary_models = [
        "CatBoostClassifier",
        "SoloModel",
        "TwoModels",
        "AufXLearner",
        "AufRandomForestClassifier",
        "AufTreeClassifier",
    ]

    _available_multi_models = [
        "BaseSClassifier",
        "BaseTClassifier",
        "BaseXClassifier",
        "UpliftTreeClassifier",
        "UpliftRandomForestClassifier",
    ]

    def __init__(
        self,
        model: tp.Any,
        model_name: str,
        features: tp.List[str],
        uplift_prediction_type: tp.Literal["abs", "rel"] = None,
        treatment_groups: tp.List[str] = None,
    ):
        """Initialize AufModel wrapper.

        Args:
            model: Fitted or unfitted model instance to wrap.
            model_name: Model type identifier. Must be in
                _available_binary_models or _available_multi_models.
            features: List of feature names for prediction.
            uplift_prediction_type: 'abs' for
                absolute uplift (treatment - control), 'rel' for relative
                uplift (treatment / control - 1). Defaults to None.
            treatment_groups: List of all treatment group
                names including control. Required for multi-treatment models.

        Returns:
            None

        Raises:
            AssertionError: If model_name is not in allowed list.
            AssertionError: If treatment_groups is missing for
                multi-treatment models.
        """
        assert (
            model_name in self._available_binary_models
            or model_name in self._available_multi_models
        ), f"{model_name} model name isn't allowed"

        self._model = model
        self._model_name = model_name
        self._features = features
        self._feature_importances = None

        self._uplift_prediction_type = uplift_prediction_type
        self._is_propensity = model_name == "CatBoostClassifier"
        self._is_multitreatment = model_name in self._available_multi_models

        self._n_treatments = None
        self._treatment_groups = None

        if self._is_multitreatment:
            assert treatment_groups
            self._treatment_groups = [
                x
                for x in sorted(list(treatment_groups))
                if x not in [0, "control"]
            ]
            self._n_treatments = len(self._treatment_groups)

    def fit(self, X, y, treatment):
        """Fit the wrapped model.

        Args:
            X: Training features DataFrame.
            y: Binary target values.
            treatment: Treatment group labels.

        Returns:
            AufModel: Fitted wrapper instance (self).

        Raises:
            AssertionError: If X columns do not match specified features.

        Examples:
            >>> wrapped.fit(X_train, y_train, treatment_train)

        Notes:
            For propensity models, treatment is ignored. For multi-treatment
            models, both DataFrame and array inputs are supported.
        """
        assert set(X.columns.tolist()) == set(self._features)
        if self._is_propensity:
            self._model.fit(X[self._features], y)
        elif self._is_multitreatment:
            try:
                self._model.fit(X=X[self._features], y=y, treatment=treatment)
            except (KeyError, TypeError, AttributeError):
                self._model.fit(
                    X=X[self._features].values,
                    y=y.values,
                    treatment=treatment.values,
                )
        else:
            self._model.fit(X=X[self._features], y=y, treatment=treatment)
        return self

    def _multitreatment_feature_importances(self, prettified=False):
        """Get aggregated feature importances.

        Args:
            prettified: If True, returns DataFrame with feature names.
                If False, returns raw importance array. Defaults to False.

        Returns:
            Feature importances array or DataFrame.

        Examples:
            >>> importances = wrapped.get_feature_importances(prettified=True)

        Notes:
            Aggregates importances across all internal models for
            S/T/X classifiers. For tree ensembles, returns directly.
        """
        features = self._features.copy()

        if self._model_name == "BaseSClassifier":
            importances = 0

            for model in self._model.models.values():
                importances += model.feature_importances_

            if "treatment" not in features:
                features = features + ["treatment"]

        elif self._model_name == "BaseTClassifier":
            importances = 0

            for model in self._model.models_c.values():
                importances += model.feature_importances_

            for model in self._model.models_t.values():
                importances += model.feature_importances_

        elif self._model_name == "BaseXClassifier":
            importances = 0

            for model in self._model.models_mu_c.values():
                importances += model.feature_importances_

            for model in self._model.models_mu_t.values():
                importances += model.feature_importances_

            for model in self._model.models_tau_c.values():
                importances += model.feature_importances_

            for model in self._model.models_tau_t.values():
                importances += model.feature_importances_

        elif self._model_name in [
            "UpliftTreeClassifier",
            "UpliftRandomForestClassifier",
        ]:
            importances = self._model.feature_importances_

        importances = importances / importances.sum() * 100
        self._feature_importances = importances

        if prettified:
            return pd.DataFrame(
                {"Feature Id": features, "Importance": importances}
            )

        return self._feature_importances

    def _binary_feature_importances(self, prettified=False):
        """Get aggregated feature importances.

        Args:
            prettified: If True, returns DataFrame with feature names.
                If False, returns raw importance array. Defaults to False.

        Returns:
            Feature importances array or DataFrame.

        Examples:
            >>> importances = wrapped.get_feature_importances(prettified=True)

        Notes:
            Aggregates importances across all internal models for
            S/T/X classifiers. For tree ensembles, returns directly.
        """
        if self._model_name == "TwoModels":
            feats_c = list(self._model.estimator_ctrl.feature_names_)
            imps_c = list(self._model.estimator_ctrl.feature_importances_)
            feats_imps_c = sorted(
                list(zip(feats_c, imps_c)), key=lambda p: p[0]
            )

            feats_t = list(self._model.estimator_trmnt.feature_names_)
            imps_t = list(self._model.estimator_trmnt.feature_importances_)
            feats_imps_t = sorted(
                list(zip(feats_t, imps_t)), key=lambda p: p[0]
            )

            feats_imps = {
                fc: ic + it
                for (fc, ic), (ft, it) in zip(feats_imps_c, feats_imps_t)
            }

        else:
            if hasattr(self._model, "estimator"):
                feats = np.array(self._model.estimator.feature_names_)
                imps = np.array(self._model.estimator.feature_importances_)
            elif self._model_name == "AufXLearner":
                feat_imps = self._model.get_feature_importances()["treatment"]
                feats = np.array([feat for (feat, imp) in feat_imps.items()])
                imps = np.array([imp for (feat, imp) in feat_imps.items()])
            else:
                feats = np.array(self._model.feature_names_)
                imps = np.array(self._model.feature_importances_)

            feats_imps = {f: i for f, i in zip(feats, imps)}

        importances = np.array([feats_imps[f] for f in self._features])
        importances = importances / importances.sum() * 100
        self._feature_importances = importances

        if prettified:
            return pd.DataFrame(
                {"Feature Id": self._features, "Importance": importances}
            )

        return self._feature_importances

    def get_feature_importances(self, prettified=False):
        """Get feature importances from the wrapped model.

        Args:
            prettified: If True, returns DataFrame with feature
                names. If False, returns raw importance array. Defaults to
                False.

        Returns:
            If prettified is True, dataframe with columns 'Feature Id' and 'Importance', otherwise feature importances array.

        Examples:
            >>> importances = wrapped.get_feature_importances(prettified=True)
            >>> print(importances.head())

        Notes:
            Automatically delegates to the appropriate method based on
            whether the model is binary or multi-treatment.
        """
        if self._is_multitreatment:
            return self._multitreatment_feature_importances(prettified)
        else:
            return self._binary_feature_importances(prettified)

    def _binary_predict(self, data: pd.DataFrame, return_df: bool = True):
        """Predict uplift scores or propensity.

        Args:
            data: Feature DataFrame for prediction.
            return_df: If True, returns DataFrame with scores appended.
                If False, returns only scores array. Defaults to True.

        Returns:
            DataFrame with scores or scores array, depending on return_df.

        Examples:
            >>> predictions = wrapped.predict(X_test, return_df=True)
            >>> scores = wrapped.predict(X_test, return_df=False)

        Notes:
            For uplift models, adds 'trmnt_preds' and 'ctrl_preds' columns
            when return_df is True.
        """
        result = dict()

        if self._is_propensity:
            result["scores"] = self._model.predict_proba(data[self._features])[
                :, 1
            ]
        else:
            result["scores"] = self._model.predict(
                data[self._features]
            ).reshape(-1)
            result["trmnt_preds"] = self._model.trmnt_preds_
            result["ctrl_preds"] = self._model.ctrl_preds_
            if self._uplift_prediction_type != "abs":
                result["scores"] = (
                    result["trmnt_preds"] / result["ctrl_preds"] - 1
                )

        if return_df:
            output = data.copy()
            output["score_raw"] = result["scores"]
            if not self._is_propensity:
                output["trmnt_preds"] = result["trmnt_preds"]
                output["ctrl_preds"] = result["ctrl_preds"]
        else:
            output = result["scores"]

        return output

    def _multi_predict(self, data: pd.DataFrame):
        """Predict uplift scores for all treatment groups.

        Args:
            data: Feature DataFrame for prediction.

        Returns:
            DataFrame with uplift scores for each treatment group.

        Examples:
            >>> predictions = wrapped.predict(X_test)
            >>> print(predictions.columns)  # ['treatment_a', 'treatment_b']

        Notes:
            Returns only treatment columns, excluding control.
        """
        if self._model_name in [
            "UpliftTreeClassifier",
            "UpliftRandomForestClassifier",
        ]:
            cate_pred_raw = self._model.predict(
                data[self._features].values.copy()
            )
            # Skip control column if needed
            if cate_pred_raw.shape[1] > self._n_treatments:
                cate_pred = cate_pred_raw[:, 1:]
            else:
                cate_pred = cate_pred_raw
        else:
            cate_pred = self._model.predict(
                X=data[self._features].values.copy(), p=None
            )

        output = pd.DataFrame(
            cate_pred, columns=self._treatment_groups, index=data.index
        )

        return output

    def predict(self, data: pd.DataFrame, return_df: bool = True):
        """Predict uplift scores or propensity.

        Args:
            data: Feature DataFrame for prediction.
            return_df: If True, returns DataFrame with scores.
                If False, returns only scores array. Only applicable for
                binary treatment models. Defaults to True.

        Returns:
            tp.Union[pd.DataFrame, np.ndarray]: Predictions. For binary
                treatment, returns DataFrame or array depending on
                return_df. For multi-treatment, returns DataFrame with
                treatment group columns.

        Raises:
            AssertionError: If data columns do not match features.

        Examples:
            >>> predictions = wrapped.predict(X_test)
            >>> scores = wrapped.predict(X_test, return_df=False)

        Notes:
            Automatically delegates to _binary_predict or _multi_predict
            based on model type.
        """
        if self._is_multitreatment:
            return self._multi_predict(data)
        else:
            return self._binary_predict(data, return_df)
