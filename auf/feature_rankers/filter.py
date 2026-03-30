"""Filter-based feature ranking for uplift modeling.

Provides univariate feature selection methods that evaluate each feature
independently using statistical divergence measures between treatment
and control groups.

Classes:
    FilterRanker: Ranks features using F-test, likelihood ratio, KL divergence,
        Euclidean distance, or Chi-square statistics using causalml package.

Examples:
    >>> from auf.feature_selection import FilterRanker
    >>> ranker = FilterRanker(method='KL', bins=10)
    >>> ranked_features, scores = ranker.run(df, features, 'target', 'treatment')
    >>> print(f"Top feature: {ranked_features[0]} with score {scores[0]}")

Notes:
    Filter methods process each feature independently without model training.
    Bin-based methods (KL, ED, Chi) discretize continuous features.
    Requires treatment column with binary values (0=control, 1=treatment).
    Uses causalml.feature_selection.FilterSelect internally.
"""

import typing as tp

import pandas as pd
from causalml.feature_selection import FilterSelect


class FilterRanker:
    """Univariate filter-based feature ranking for uplift modeling.

    Ranks features independently using statistical divergence measures
    between treatment and control groups without training a model.

    Attributes:
        _name (str): Identifier for the ranker instance.
        _method (str): Selection method ('F', 'LR', 'KL', 'ED', 'Chi').
        _bins (int): Number of bins for discretization in bin-based methods.
        _ranked_features (List[str]): Ordered feature names after run().
        _ranked_features_scores (List[float]): Corresponding relevance scores.

    Examples:
        >>> from auf.feature_selection import FilterRanker
        >>> ranker = FilterRanker(method='KL', bins=10, name='filter_kl')
        >>> ranked_features, scores = ranker.run(
        ...     df, features, 'target', 'treatment'
        ... )
        >>> print(f"Top feature: {ranker[0]}, score: {scores[0]}")

    Notes:
        Works only with training data; no validation set required.
        Bin-based methods (KL, ED, Chi) discretize continuous features.
        F-test and LR are standard statistical tests for feature relevance.
        Uses causalml.feature_selection.FilterSelect internally.
    """

    def __init__(self, method: str, bins: int, name: str = "__empty_name__"):
        """Initialize the FilterRanker instance.

        Args:
            method: Selection method to rank features. One of:
                - 'F': F-test for feature relevance.
                - 'LR': Likelihood ratio test.
                - 'KL': KL divergence (bin-based).
                - 'ED': Euclidean distance (bin-based).
                - 'Chi': Chi-square statistic (bin-based).
            bins: Number of bins for discretization. Used only with
                bin-based methods ('KL', 'ED', 'Chi').
            name: Identifier for the ranker instance. Defaults to
                '__empty_name__'.

        Raises:
            AssertionError: If method is not one of the supported options.

        Examples:
            >>> ranker = FilterRanker(method='KL', bins=10, name='my_ranker')
            >>> ranker = FilterRanker(method='F', bins=0)
        """
        assert method in ["F", "LR", "KL", "ED", "Chi"]
        self._name: str = name
        self._method: str = method
        self._bins: int = bins
        self._ranked_features: tp.List[str] = []
        self._ranked_features_scores: tp.List[float] = []

    def run(
        self,
        data: pd.DataFrame,
        features: tp.List[str],
        target_col: str,
        treatment_col: str,
    ) -> tp.Tuple[tp.List[str], tp.List[float]]:
        """Execute filter-based feature ranking on the provided dataset.

        Args:
            data: DataFrame containing features, target, and treatment columns.
            features: List of feature column names to rank.
            target_col: Name of the binary target column (0/1).
            treatment_col: Name of the binary treatment column (0=control, 1=treatment).

        Returns:
            Tuple[List[str], List[float]]: A tuple containing:
                - ranked_features: Feature names ordered by descending relevance.
                - ranked_scores: Corresponding relevance scores.

        Raises:
            AssertionError: If 'treatment_group_key' column already exists in data,
                or if treatment column contains values other than 0 or 1.
            Exception: Re-raises any exception from the underlying FilterSelect.

        Examples:
            >>> ranker = FilterRanker(method='KL', bins=10)
            >>> ranked_features, scores = ranker.run(df, ['feat1', 'feat2'], 'target', 'treatment')

        Notes:
            Temporarily creates 'treatment_group_key' column during execution.
            The column is removed before the method returns or raises an exception.
        """
        assert "treatment_group_key" not in data.columns
        assert target_col in data.columns
        assert treatment_col in data.columns
        assert ((data[treatment_col] == 0) | (data[treatment_col] == 1)).all()
        assert set(features) & set(data.columns) == set(features)

        try:
            data["treatment_group_key"] = data[treatment_col].apply(
                lambda x: "control" if x == 0 else "treatment"
            )

            selector = FilterSelect()

            result = selector.filter_D(
                data=data,
                features=features,
                y_name=target_col,
                n_bins=self._bins,
                method=self._method,
                null_impute="mean",
                experiment_group_column="treatment_group_key",
                control_group="control",
            )

        finally:
            data.drop("treatment_group_key", axis=1, inplace=True)

        self._ranked_features = list(result.feature.values)
        self._ranked_features_scores = list(result.score.values)
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
