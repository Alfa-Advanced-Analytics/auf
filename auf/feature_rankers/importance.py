"""Model-based feature ranking using feature importance scores.

Provides feature selection strategies that leverage model's internal
feature_importances_ attribute to rank features by their predictive power.

Classes:
    ImportanceRanker: Ranks features using model-derived importance scores
        with three sorting modes: 'at_once', 'iterative', and 'rfe'.

Examples:
    >>> from auf.feature_selection import ImportanceRanker
    >>> from sklift.models import SoloModel
    >>> from catboost import CatBoostClassifier

    >>> # Single-pass ranking
    >>> ranker = ImportanceRanker(
    ...     model_class=SoloModel,
    ...     model_params={'estimator': CatBoostClassifier(iterations=100)},
    ...     sorting_mode='at_once'
    ... )
    >>> ranked_features, scores = ranker.run(df, features, 'target', 'treatment')

    >>> # Recursive feature elimination
    >>> rfe_ranker = ImportanceRanker(
    ...     model_class=SoloModel,
    ...     model_params={'estimator': CatBoostClassifier()},
    ...     sorting_mode='rfe'
    ... )
    >>> ranked_features, scores = rfe_ranker.run(df, features, 'target', 'treatment')

Notes:
    Works with SoloModel (S-learner), UpliftRandomForestClassifier, and
    propensity models exposing feature_importances_ attribute.
    All sorting modes operate only on training data.
    'iterative' mode drops 10% worst features per iteration.
    'rfe' mode drops one worst feature per iteration.
"""

import typing as tp

import pandas as pd
from causalml.inference.tree import UpliftRandomForestClassifier
from sklift.models import SoloModel


class ImportanceRanker:
    """Model-based feature ranking using feature importance scores.

    Extracts feature_importances_ from uplift or propensity models and
    ranks features with three sorting strategies: single-pass, iterative,
    or recursive feature elimination.

    Attributes:
        _model_class (Any): Model class to instantiate for ranking.
        _model_params (Dict[str, Any]): Parameters passed to model constructor.
        _sorting_mode (str): Ranking strategy ('at_once', 'iterative', 'rfe').
        _name (str): Identifier for the ranker instance.
        _ranked_features (List[str]): Ordered feature names after run().
        _ranked_features_scores (List[float]): Corresponding importance scores.

    Examples:
        >>> from auf.feature_selection import ImportanceRanker
        >>> from sklift.models import SoloModel
        >>> from catboost import CatBoostClassifier
        >>> ranker = ImportanceRanker(
        ...     model_class=SoloModel,
        ...     model_params={'estimator': CatBoostClassifier(iterations=100)},
        ...     sorting_mode='rfe',
        ...     name='importance_rfe'
        ... )
        >>> ranked_features, scores = ranker.run(
        ...     df, features, 'target', 'treatment'
        ... )

    Notes:
        Supports SoloModel (S-learner), UpliftRandomForestClassifier, and
        any propensity model with feature_importances_ attribute.
        'at_once': single model fit, direct ordering by importance.
        'iterative': removes 10% worst features per iteration.
        'rfe': removes one worst feature per iteration.
        All modes operate only on training data.
    """

    def __init__(
        self,
        model_class: tp.Any,
        model_params: tp.Dict[str, tp.Any],
        sorting_mode: str,
        name: str = "__empty_name__",
    ):
        """Initializes the ImportanceRanker instance.

        Args:
            model_class: Model class to instantiate. Must implement
                'fit' method and at least one of 'predict' or 'predict_proba'.
                Supported types include SoloModel, UpliftRandomForestClassifier,
                and propensity models with feature_importances_.
            model_params: Dictionary of parameters passed to model_class
                constructor.
            sorting_mode: Feature ranking strategy. One of:
                - 'at_once': Single model fit, direct ordering.
                - 'iterative': Remove 10% worst features per iteration.
                - 'rfe': Remove one worst feature per iteration (RFE).
            name: Identifier for the ranker instance. Defaults to
                '__empty_name__'.

        Raises:
            AssertionError: If sorting_mode is not one of the supported options.
            ValueError: If model cannot be instantiated with provided params
                or lacks required methods (fit, predict/predict_proba).

        Examples:
            >>> from sklift.models import SoloModel
            >>> from catboost import CatBoostClassifier
            >>> ranker = ImportanceRanker(
            ...     model_class=SoloModel,
            ...     model_params={'estimator': CatBoostClassifier()},
            ...     sorting_mode='rfe',
            ...     name='importance_rfe'
            ... )
        """
        assert sorting_mode in ["at_once", "iterative", "rfe"]

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
        self._sorting_mode = sorting_mode
        self._name: str = name
        self._ranked_features: tp.List[str] = []
        self._ranked_features_scores: tp.List[float] = []

    def _run_propensity(
        self, data: pd.DataFrame, features: tp.List[str], target_col: str
    ) -> tp.Tuple[tp.List[str], tp.List[float]]:
        model = self._model_class(**self._model_params)
        model.fit(data[features], data[target_col])

        importances = model.feature_importances_
        features_info = list(zip(importances, features))
        features_info = sorted(features_info, key=lambda x: -x[0])
        importances, ranked_features = map(list, zip(*features_info))

        return ranked_features, importances

    def _run_s_learner(
        self,
        data: pd.DataFrame,
        features: tp.List[str],
        target_col: str,
        treatment_col: str,
    ) -> tp.Tuple[tp.List[str], tp.List[float]]:
        model = self._model_class(**self._model_params)
        model.fit(data[features], data[target_col], data[treatment_col])

        importances = model.estimator.feature_importances_
        feature_names = model.estimator.feature_names_
        features_info = list(zip(importances, feature_names))
        features_info = [
            p for p in features_info if p[1] not in [treatment_col, "treatment"]
        ]
        features_info = sorted(features_info, key=lambda x: -x[0])
        importances, ranked_features = map(list, zip(*features_info))

        return ranked_features, importances

    def _run_uplift_forest(
        self,
        data: pd.DataFrame,
        features: tp.List[str],
        target_col: str,
        treatment_col: str,
    ) -> tp.Tuple[tp.List[str], tp.List[float]]:
        model = self._model_class(**self._model_params)
        model.fit(
            X=data[features].values,
            treatment=data[treatment_col]
            .apply(lambda t: "treatment" if t == 1 else "control")
            .values,
            y=data[target_col].values,
        )

        importances = model.feature_importances_
        features_info = list(zip(importances, features))
        features_info = sorted(features_info, key=lambda x: -x[0])
        importances, ranked_features = map(list, zip(*features_info))

        return ranked_features, importances

    def _run(
        self,
        data: pd.DataFrame,
        features: tp.List[str],
        target_col: str,
        treatment_col: str,
    ) -> tp.Tuple[tp.List[str], tp.List[float]]:
        ranked_features: tp.List[str] = []
        importances: tp.List[float] = []

        if issubclass(self._model_class, SoloModel):
            ranked_features, importances = self._run_s_learner(
                data, features, target_col, treatment_col
            )
        elif issubclass(self._model_class, UpliftRandomForestClassifier):
            ranked_features, importances = self._run_uplift_forest(
                data, features, target_col, treatment_col
            )
        else:
            ranked_features, importances = self._run_propensity(
                data, features, target_col
            )

        return ranked_features, importances

    def run(
        self,
        data: pd.DataFrame,
        features: tp.List[str],
        target_col: str,
        treatment_col: str,
    ) -> tp.Tuple[tp.List[str], tp.List[float]]:
        """Execute model-based feature ranking using importance scores.

        Args:
            data: DataFrame containing features, target, and treatment columns.
            features: List of feature column names to rank.
            target_col: Name of the target column.
            treatment_col: Name of the treatment column.

        Returns:
            Tuple[List[str], List[float]]: A tuple containing:
                - ranked_features: Feature names ordered by descending importance.
                - ranked_scores: Corresponding importance scores.

        Examples:
            >>> ranker = ImportanceRanker(
            ...     model_class=SoloModel,
            ...     model_params={'estimator': CatBoostClassifier()},
            ...     sorting_mode='rfe'
            ... )
            >>> ranked_features, scores = ranker.run(df, features, 'target', 'treatment')

        Notes:
            Ranking strategy depends on sorting_mode set during initialization:
                - 'at_once': Single model fit, features ordered by importance.
                - 'iterative': Repeatedly removes 10% worst features.
                - 'rfe': Removes one worst feature per iteration.
            Results are stored in internal attributes after execution.
        """
        assert target_col in data.columns
        assert not treatment_col or treatment_col in data.columns
        assert set(features) & set(data.columns) == set(features)

        ranked_features: tp.List[str] = features.copy()
        importances: tp.List[float] = [0.0 for f in ranked_features]

        if self._sorting_mode == "at_once":
            ranked_features, importances = self._run(
                data, ranked_features, target_col, treatment_col
            )
        elif self._sorting_mode == "iterative":
            n_top = len(features)
            while n_top:
                feats, imps = self._run(
                    data, ranked_features[:n_top], target_col, treatment_col
                )
                ranked_features[:n_top] = feats
                importances[:n_top] = imps
                n_top = int(n_top * 0.9)
        else:  # self._sorting_mode == 'rfe'
            for n_top in range(len(features), 0, -1):
                feats, imps = self._run(
                    data, ranked_features[:n_top], target_col, treatment_col
                )
                ranked_features[:n_top] = feats
                importances[:n_top] = imps

        self._ranked_features = ranked_features.copy()
        self._ranked_features_scores = importances.copy()
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
