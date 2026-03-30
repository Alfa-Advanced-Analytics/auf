"""Permutation-based feature importance for uplift modeling.

Provides feature ranking by measuring the degradation in uplift metric
when each feature's values are randomly shuffled. Uses bootstrapping
to obtain robust importance estimates.

Classes:
    PermutationRanker: Ranks features by median uplift metric drop divided
        by standard deviation across bootstrap samples.

Examples:
    >>> from auf.feature_selection import PermutationRanker
    >>> from sklift.models import SoloModel
    >>> from catboost import CatBoostClassifier
    >>> from functools import partial
    >>> from sklift.metrics import uplift_at_k
    >>> ranker = PermutationRanker(
    ...     model_class=SoloModel,
    ...     model_params={'estimator': CatBoostClassifier(iterations=100)},
    ...     rng=np.random.RandomState(42),
    ...     bootstrap_repeats=50
    ... )
    >>> ranked_features, scores = ranker.run(
    ...     train_df, val_df, features, 'target', 'treatment',
    ...     metric=partial(uplift_at_k, k=0.3)
    ... )

Notes:
    Requires both training and validation datasets.
    Score = median(delta_uplift) / (std(delta_uplift) + epsilon).
    Higher score indicates more important feature.
    May underestimate importance of highly correlated features.
    Shuffling is performed once per feature; bootstrap evaluates on validation.
"""

import typing as tp

import numpy as np
import pandas as pd
from causalml.inference.tree import UpliftRandomForestClassifier
from sklift.models import SoloModel


class PermutationRanker:
    """Permutation-based feature importance for uplift modeling.

    Ranks features by measuring uplift metric degradation when feature
    values are shuffled. Uses bootstrapping to compute robust importance
    scores as median drop divided by standard deviation.

    Attributes:
        _model_class (Any): Model class to instantiate for ranking.
        _model_params (Dict[str, Any]): Parameters passed to model constructor.
        _rng (RandomState): Random number generator for reproducibility.
        _bootstrap_repeats (int): Number of bootstrap samples per feature.
        _name (str): Identifier for the ranker instance.
        _ranked_features (List[str]): Ordered feature names after run().
        _ranked_features_scores (List[float]): Corresponding permutation scores.

    Examples:
        >>> from auf.feature_selection import PermutationRanker
        >>> from sklift.models import SoloModel
        >>> from catboost import CatBoostClassifier
        >>> from sklift.metrics import qini_auc_score
        >>> ranker = PermutationRanker(
        ...     model_class=SoloModel,
        ...     model_params={'estimator': CatBoostClassifier(iterations=100)},
        ...     rng=np.random.RandomState(42),
        ...     bootstrap_repeats=50,
        ...     name='permutation_ranker'
        ... )
        >>> ranked_features, scores = ranker.run(
        ...     train_df, val_df, features, 'target', 'treatment',
        ...     metric=qini_auc_score
        ... )

    Notes:
        Requires both training and validation datasets.
        Score formula: median(delta_uplift) / (std(delta_uplift) + epsilon).
        Higher score indicates greater feature importance.
        May underestimate importance of highly correlated features due to
        information leakage from correlated features.
        Performs single shuffle per feature; bootstrap evaluates on validation.
        Trains model two times for each feature: with and without it.
    """

    def __init__(
        self,
        model_class: tp.Any,
        model_params: tp.Dict[str, tp.Any],
        rng: np.random.RandomState,
        bootstrap_repeats: int,
        name: str = "__empty_name__",
    ):
        """Initialize the PermutationRanker instance.

        Args:
            model_class: Model class to instantiate. Must implement
                'fit' method and at least one of 'predict' or 'predict_proba'.
                Supported types include SoloModel, UpliftRandomForestClassifier,
                and propensity models.
            model_params: Dictionary of parameters passed to model_class
                constructor.
            rng: numpy RandomState instance for reproducible bootstrap
                sampling and permutation.
            bootstrap_repeats: Number of bootstrap samples to compute
                median and standard deviation of metric degradation.
            name: Identifier for the ranker instance. Defaults to
                '__empty_name__'.

        Raises:
            ValueError: If model cannot be instantiated with provided params
                or lacks required methods (fit, predict/predict_proba).

        Examples:
            >>> from sklift.models import SoloModel
            >>> from catboost import CatBoostClassifier
            >>> ranker = PermutationRanker(
            ...     model_class=SoloModel,
            ...     model_params={'estimator': CatBoostClassifier()},
            ...     rng=np.random.RandomState(42),
            ...     bootstrap_repeats=50,
            ...     name='permutation_ranker'
            ... )
        """
        assert bootstrap_repeats > 0

        try:
            try_to_create_model = model_class(**model_params)
            assert hasattr(try_to_create_model, "fit")
            assert hasattr(try_to_create_model, "predict") or hasattr(
                try_to_create_model, "predict_proba"
            )
            del try_to_create_model
        except Exception as exc:
            raise ValueError(
                "Check params for creating model: model creation failed"
            ) from exc

        self._model_class = model_class
        self._model_params = model_params
        self._rng = rng
        self._bootstrap_repeats = bootstrap_repeats
        self._model: tp.Any = None
        self._name: str = name
        self._ranked_features: tp.List[str] = []
        self._ranked_features_scores: tp.List[float] = []

    def _model_fit(self, x_train, y_train, t_train):
        self._model = self._model_class(**self._model_params)
        if issubclass(self._model_class, SoloModel):
            self._model.fit(x_train, y_train, t_train)
        elif issubclass(self._model_class, UpliftRandomForestClassifier):
            self._model.fit(
                X=x_train.values,
                treatment=t_train.apply(
                    lambda t: "treatment" if t == 1 else "control"
                ).values,
                y=y_train.values,
            )
        else:
            self._model.fit(x_train, y_train)

    def _model_predict(self, x_val):
        if issubclass(
            self._model_class, (SoloModel, UpliftRandomForestClassifier)
        ):
            return self._model.predict(x_val).reshape(-1)
        else:
            return self._model.predict_proba(x_val)[:, 1].reshape(-1)

    def _get_feature_gain(
        self,
        metric,
        x_train,
        y_train,
        t_train,
        x_val,
        y_val,
        t_val,
        feature_to_check,
        n_test_repeats,
        permutation_idxs,
    ):
        self._model_fit(x_train, y_train, t_train)
        preds_usual = self._model_predict(x_val)

        values_copy = x_train.loc[:, feature_to_check].values.copy()
        x_train.loc[:, feature_to_check] = x_train.loc[
            :, feature_to_check
        ].values[permutation_idxs]
        self._model_fit(x_train, y_train, t_train)
        x_train.loc[:, feature_to_check] = values_copy
        preds_shuffled = self._model_predict(x_val)

        delta_uplifts = [0 for _ in range(n_test_repeats)]

        for i in range(n_test_repeats):
            idxs = self._rng.choice(
                range(x_val.shape[0]), size=x_val.shape[0], replace=True
            )
            quality_usual = metric(
                y_val.values[idxs], preds_usual[idxs], t_val.values[idxs]
            )
            quality_shuffled = metric(
                y_val.values[idxs], preds_shuffled[idxs], t_val.values[idxs]
            )
            delta_uplifts[i] = quality_usual - quality_shuffled

        std = np.std(delta_uplifts)
        median = np.median(delta_uplifts)

        epsilon = 1e-5
        feature_score = median / (std + epsilon)

        return median, std, feature_score

    def run(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        all_features: tp.List[str],
        target_col: str,
        treatment_col: str,
        metric: tp.Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    ) -> tp.Tuple[tp.List[str], tp.List[float]]:
        """Execute permutation-based feature importance estimation.

        Args:
            train_data: Training DataFrame with features, target, and treatment.
            val_data: Validation DataFrame with same structure as train_data.
            all_features: List of feature column names to rank.
            target_col: Name of the target column.
            treatment_col: Name of the treatment column.
            metric: Callable taking (y_true, predictions, treatment) and returning
                a float score. Typically an uplift metric like qini_auc_score.

        Returns:
            Tuple[List[str], List[float]]: A tuple containing:
                - ranked_features: Feature names ordered by descending importance.
                - ranked_scores: Corresponding permutation importance scores.

        Raises:
            AssertionError: If train_data and val_data have different columns,
                or if target_col/treatment_col are missing, or if features
                are not present in the data.

        Examples:
            >>> from functools import partial
            >>> ranker = PermutationRanker(
            ...     model_class=SoloModel,
            ...     model_params={'estimator': CatBoostClassifier()},
            ...     rng=np.random.RandomState(42),
            ...     bootstrap_repeats=50
            ... )
            >>> ranked_features, scores = ranker.run(
            ...     train_df, val_df, features, 'target', 'treatment',
            ...     metric=partial(uplift_at_k, k=0.3)
            ... )

        Notes:
            Score formula: median(delta_uplift) / (std(delta_uplift) + epsilon).
            Higher score indicates greater feature importance.
            Uses the same permutation indices for all features to reduce variance.
        """
        assert set(train_data.columns) == set(val_data.columns)
        assert target_col in train_data.columns
        assert treatment_col in train_data.columns
        assert (
            (train_data[treatment_col] == 0) | (train_data[treatment_col] == 1)
        ).all()
        assert (
            (val_data[treatment_col] == 0) | (val_data[treatment_col] == 1)
        ).all()
        assert set(all_features) & set(train_data.columns) == set(all_features)

        features = []
        features_scores = []

        permutation_idxs = self._rng.choice(
            range(train_data.shape[0]), size=train_data.shape[0], replace=False
        )
        x_train, y_train, t_train = (
            train_data.loc[:, all_features],
            train_data.loc[:, target_col],
            train_data.loc[:, treatment_col],
        )
        x_val, y_val, t_val = (
            val_data.loc[:, all_features],
            val_data.loc[:, target_col],
            val_data.loc[:, treatment_col],
        )

        for f in all_features:
            _, _, feature_score = self._get_feature_gain(
                metric,
                x_train,
                y_train,
                t_train,
                x_val,
                y_val,
                t_val,
                feature_to_check=f,
                n_test_repeats=self._bootstrap_repeats,
                permutation_idxs=permutation_idxs,
            )

            features.append(f)
            features_scores.append(feature_score)

        order = list(np.argsort(features_scores)[::-1])
        self._ranked_features = features[order]
        self._ranked_features_scores = features_scores[order]
        return self._ranked_features, self._ranked_features_scores

    def get_ranker_name(self) -> str:
        """Return the identifier of this ranker instance.

        Returns:
            The name string assigned during initialization.
        """
        return self._name

    def get_ranked_features(self) -> tp.List[str]:
        """Return the list of ranked feature names.

        Returns:
            List of feature names ordered by descending relevance.
            Empty if run() has not been called yet.
        """
        return self._ranked_features

    def get_ranked_features_scores(self) -> tp.List[float]:
        """Return the list of relevance scores for ranked features.

        Returns:
            List of scores corresponding to ranked_features order.
            Empty if run() has not been called yet.
        """
        return self._ranked_features_scores
