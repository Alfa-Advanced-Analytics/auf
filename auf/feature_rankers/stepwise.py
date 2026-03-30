"""Forward stepwise feature selection for uplift modeling.

Provides greedy feature selection that iteratively adds the feature
providing the largest improvement in uplift metric on validation data.
Uses bootstrapping to obtain robust gain estimates.

Classes:
    StepwiseRanker: Ranks features by sequential forward selection
        optimizing an uplift metric.

Examples:
    >>> from auf.feature_selection import StepwiseRanker
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
    At each step, selects feature with highest median gain / std ratio.
    Score represents improvement when feature was added to the model.
    Works with any uplift model implementing fit/predict interface.
"""

import typing as tp

import numpy as np
import pandas as pd
from causalml.inference.tree import UpliftRandomForestClassifier
from sklift.models import SoloModel, TwoModels

from ..models import AufRandomForestClassifier, AufTreeClassifier, AufXLearner


class StepwiseRanker:
    """Forward stepwise feature selection for uplift modeling.

    Iteratively adds features that provide the largest improvement in
    uplift metric on validation data. Uses bootstrapping to obtain
    robust gain estimates at each selection step.

    Attributes:
        _model_class (Any): Model class to instantiate for ranking.
        _model_params (Dict[str, Any]): Parameters passed to model constructor.
        _rng (RandomState): Random number generator for reproducibility.
        _bootstrap_repeats (int): Number of bootstrap samples per candidate.
        _name (str): Identifier for the ranker instance.
        _ranked_features (List[str]): Ordered feature names after run().
        _ranked_features_scores (List[float]): Gain scores at each step.

    Examples:
        >>> from auf.feature_selection import StepwiseRanker
        >>> from sklift.models import SoloModel
        >>> from catboost import CatBoostClassifier
        >>> from sklift.metrics import qini_auc_score
        >>> ranker = StepwiseRanker(
        ...     model_class=SoloModel,
        ...     model_params={'estimator': CatBoostClassifier(iterations=100)},
        ...     rng=np.random.RandomState(42),
        ...     bootstrap_repeats=30,
        ...     name='stepwise_ranker'
        ... )
        >>> ranked_features, scores = ranker.run(
        ...     train_df, val_df, features, 'target', 'treatment',
        ...     metric=qini_auc_score
        ... )

    Notes:
        Requires both training and validation datasets.
        At each step, selects feature maximizing median_gain / std ratio.
        Score represents incremental improvement when feature was added.
        Works with any model implementing fit/predict interface.
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
        """Initialize the StepwiseRanker instance.

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
            ValueError: If model cannot be instantiated with provided params
                or lacks required methods (fit, predict/predict_proba).

        Examples:
            >>> from sklift.models import SoloModel
            >>> from catboost import CatBoostClassifier
            >>> ranker = StepwiseRanker(
            ...     model_class=SoloModel,
            ...     model_params={'estimator': CatBoostClassifier()},
            ...     rng=np.random.RandomState(42),
            ...     bootstrap_repeats=30,
            ...     name='stepwise_ranker'
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
            self._model_class, (SoloModel, UpliftRandomForestClassifier)
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
        current_features,
        feature_to_add,
        n_test_repeats,
    ):
        x_train, y_train, t_train = (
            train_data[current_features + [feature_to_add]],
            train_data[target_col],
            train_data[treatment_col],
        )
        x_val, y_val, t_val = (
            val_data[current_features + [feature_to_add]],
            val_data[target_col],
            val_data[treatment_col],
        )

        self._model_fit(x_train, y_train, t_train)
        preds_full = self._model_predict(x_val)

        if current_features:
            self._model_fit(x_train[current_features], y_train, t_train)
            preds_without = self._model_predict(x_val[current_features])
        else:
            preds_without = np.zeros_like(preds_full)

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
        """Execute forward stepwise feature selection.

        Args:
            train_data: Training DataFrame with features, target, and treatment.
            val_data: Validation DataFrame with same structure as train_data.
            all_features: List of candidate feature column names to evaluate.
            target_col: Name of the target column.
            treatment_col: Name of the treatment column.
            metric: Callable taking (y_true, predictions, treatment) and returning
                a float score. Used to evaluate feature contribution at each step.

        Returns:
            Tuple[List[str], List[float]]: A tuple containing:
                - ranked_features: Feature names in order of selection.
                - ranked_scores: Gain scores corresponding to each selection step.

        Raises:
            AssertionError: If train_data and val_data have different columns.

        Examples:
            >>> ranker = StepwiseRanker(
            ...     model_class=SoloModel,
            ...     model_params={'estimator': CatBoostClassifier()},
            ...     rng=np.random.RandomState(42),
            ...     bootstrap_repeats=30
            ... )
            >>> ranked_features, scores = ranker.run(
            ...     train_df, val_df, features, 'target', 'treatment',
            ...     metric=qini_auc_score
            ... )

        Notes:
            At each step, selects the feature with highest median gain / std ratio.
            Score represents incremental improvement when feature was added.
            Computational complexity is O(n_features^2) model fits.
            Features are added greedily; does not guarantee global optimum.
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

        candidate_features = all_features.copy()
        features = []
        features_scores = []

        steps = 0
        while steps < len(all_features):
            steps += 1

            best_feature = ""
            best_score = None

            for f in candidate_features:
                median, std, features_score = self._get_feature_gain(
                    metric,
                    train_data,
                    val_data,
                    target_col,
                    treatment_col,
                    current_features=features,
                    feature_to_add=f,
                    n_test_repeats=self._bootstrap_repeats,
                )

                if best_score is None or features_score > best_score:
                    best_feature = f
                    best_score = features_score

            candidate_features.remove(best_feature)
            features.append(best_feature)
            features_scores.append(best_score)

        self._ranked_features = features
        self._ranked_features_scores = features_scores
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
