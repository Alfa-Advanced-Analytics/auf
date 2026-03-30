"""Drop-one-out feature importance for uplift modeling.

Provides simple feature ranking by training a model with all features,
then retraining with each feature removed and measuring the change
in uplift metric on validation data.

Classes:
    StraightforwardRanker: Ranks features by median uplift metric change
        when feature is excluded from the model.

Examples:
    >>> from auf.feature_selection import StraightforwardRanker
    >>> from sklift.models import SoloModel
    >>> from catboost import CatBoostClassifier
    >>> from sklift.metrics import qini_auc_score
    >>> ranker = StepwiseRanker(
    ...     model_class=SoloModel,
    ...     model_params={'estimator': CatBoostClassifier(iterations=100)},
    ...     rng=np.random.RandomState(42),
    ...     bootstrap_repeats=30
    ... )
    >>> ranked_features, scores = ranker.run(
    ...     train_df, val_df, features, 'target', 'treatment',
    ...     metric=qini_auc_score
    ... )

Notes:
    Requires both training and validation datasets.
    Score = median(uplift_full - uplift_without) / (std + epsilon).
    Positive score indicates feature contributes positively to model.
    May produce misleading results for highly correlated features.
    Does not perform feature selection, only ranking.
"""

import typing as tp

import numpy as np
import pandas as pd
from sklift.models import SoloModel, TwoModels

from ..models import AufRandomForestClassifier, AufTreeClassifier, AufXLearner


class StraightforwardRanker:
    """Drop-one-out feature importance for uplift modeling.

    Ranks features by training a model with all features, then measuring
    the change in uplift metric when each feature is removed. Uses
    bootstrapping to compute robust importance scores.

    Attributes:
        _model_class (Any): Model class to instantiate for ranking.
        _model_params (Dict[str, Any]): Parameters passed to model constructor.
        _rng (RandomState): Random number generator for reproducibility.
        _bootstrap_repeats (int): Number of bootstrap samples per candidate.
        _name (str): Identifier for the ranker instance.
        _ranked_features (List[str]): Ordered feature names after run().
        _ranked_features_scores (List[float]): Gain scores at each step.

    Examples:
        >>> from auf.feature_selection import StraightforwardRanker
        >>> from sklift.models import SoloModel
        >>> from catboost import CatBoostClassifier
        >>> from sklift.metrics import qini_auc_score
        >>> model = SoloModel(
        ...     estimator=CatBoostClassifier(iterations=100, verbose=False)
        ... )
        >>> ranker = StepwiseRanker(
        ...     model_class=SoloModel,
        ...     model_params={'estimator': CatBoostClassifier(iterations=100)},
        ...     rng=np.random.RandomState(42),
        ...     bootstrap_repeats=30
        ... )
        >>> ranked_features, scores = ranker.run(
        ...     train_df, val_df, features, 'target', 'treatment',
        ...     metric=qini_auc_score
        ... )

    Notes:
        Requires both training and validation datasets.
        Score = median(uplift_full - uplift_without) / (std + epsilon).
        Positive score indicates feature contributes positively.
        May produce misleading results for highly correlated features
        due to information redundancy.
        Does not perform feature selection; only produces ranking.
    """

    def __init__(
        self,
        model_class: tp.Any,
        model_params: tp.Dict[str, tp.Any],
        rng: np.random.RandomState,
        bootstrap_repeats: int,
        name: str = "__empty_name__",
    ):
        """Initialize the StraightforwardRanker instance.

        Args:
            model_class: Model class to instantiate. Must implement
                'fit' method and at least one of 'predict' or 'predict_proba'.
                Supported types include SoloModel, UpliftRandomForestClassifier,
                and propensity models.
            model_params: Dictionary of parameters passed to model_class
                constructor.
            rng: numpy RandomState instance for reproducible bootstrap sampling.
            bootstrap_repeats: Number of bootstrap samples to compute
                median and standard deviation of feature gain at each step.
            name: Identifier for the ranker instance. Defaults to
                '__empty_name__'.

        Raises:
            AttributeError: If uplift_model lacks required methods.

        Examples:
            >>> from sklift.models import SoloModel
            >>> from catboost import CatBoostClassifier
            >>> ranker = StraightforwardRanker(
            ...     model_class=SoloModel,
            ...     model_params={'estimator': CatBoostClassifier()},
            ...     rng=np.random.RandomState(42),
            ...     bootstrap_repeats=30,
            ...     name='stepwise_ranker'
            ... )

        Notes:
            Unlike other rankers, this class accepts an already instantiated
            model rather than a model class and parameters.
        """
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
        if issubclass(
            self._model_class, (AufTreeClassifier, AufRandomForestClassifier)
        ):
            self._model.fit(
                X=x_train.values,
                treatment=t_train.apply(
                    lambda t: "treatment" if t == 1 else "control"
                ).values,
                y=y_train.values,
            )
        elif issubclass(self._model_class, (SoloModel, TwoModels, AufXLearner)):
            self._model.fit(x_train, y_train, t_train)
        else:
            self._model.fit(x_train, y_train)

    def _model_predict(self, x_val):
        if issubclass(
            self._model_class,
            (
                SoloModel,
                TwoModels,
                AufXLearner,
                AufTreeClassifier,
                AufRandomForestClassifier,
            ),
        ):
            return self._model.predict(x_val).reshape(-1)
        else:
            return self._model.predict_proba(x_val)[:, 1].reshape(-1)

    def _get_feature_gain(
        self,
        metric,
        train_data,
        val_data,
        target_col: str,
        treatment_col: str,
        features,
        feature_to_check,
        n_test_repeats=1000,
    ):
        x_train, y_train, t_train = (
            train_data[features],
            train_data[target_col],
            train_data[treatment_col],
        )
        x_val, y_val, t_val = (
            val_data[features],
            val_data[target_col],
            val_data[treatment_col],
        )

        self._model_fit(x_train, y_train, t_train)
        preds_full = self._model_predict(x_val)

        self._model.fit(
            x_train[[f for f in features if f != feature_to_check]],
            y_train,
            t_train,
        )
        preds_without = self._model_predict(
            x_val[[f for f in features if f != feature_to_check]]
        )

        delta_uplifts = [0 for _ in range(n_test_repeats)]

        for i in range(n_test_repeats):
            idxs = self._rng.choice(
                range(x_val.shape[0]), size=x_val.shape[0], replace=True
            )
            quality_full = metric(
                y_val.values[idxs], preds_full[idxs], t_val.values[idxs]
            )
            quality_without = metric(
                y_val.values[idxs], preds_without[idxs], t_val.values[idxs]
            )
            delta_uplifts[i] = quality_full - quality_without

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
        """Execute drop-one-out feature importance estimation.

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
                - ranked_scores: Corresponding importance scores.

        Raises:
            AssertionError: If train_data and val_data have different columns.

        Examples:
            >>> model = SoloModel(estimator=CatBoostClassifier(verbose=False))
            >>> ranker = StraightforwardRanker(uplift_model=model)
            >>> ranked_features, scores = ranker.run(
            ...     train_df, val_df, features, 'target', 'treatment',
            ...     metric=qini_auc_score
            ... )

        Notes:
            Score = median(uplift_full - uplift_without) / (std + epsilon).
            Uses fixed 1000 bootstrap repeats internally.
            Positive score indicates feature contributes positively to model.
            May produce misleading results for highly correlated features.
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

        for f in all_features:
            median, std, feature_score = self._get_feature_gain(
                metric,
                train_data,
                val_data,
                target_col,
                treatment_col,
                all_features,
                feature_to_check=f,
                n_test_repeats=self._bootstrap_repeats,
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
