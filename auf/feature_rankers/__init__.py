"""Feature ranking strategies for uplift modeling.

Provides 5 ranking approaches for uplift modeling with binary treatment. The
approacehs have different computational trade-offs and data requirements.

Classes:
    FilterRanker: Univariate filter-based ranking (F-test, KL, Chi, etc.).
    ImportanceRanker: Model-based ranking using feature_importances_ attribute.
    PermutationRanker: Permutation importance with bootstrapped scoring.
    StepwiseRanker: Forward stepwise selection optimizing uplift metric.
    StraightforwardRanker: Ranking based on metric growth after adding feature.

Examples:
    >>> from auf.feature_selection import FilterRanker, ImportanceRanker

    >>> # Filter-based ranking (fast, no model training)
    >>> filter_ranker = FilterRanker(method='KL', bins=10)
    >>> ranked_features, scores = filter_ranker.run(df, features, 'target', 'treatment')

    >>> # Model-based ranking with recursive feature elimination
    >>> from sklift.models import SoloModel
    >>> from catboost import CatBoostClassifier
    >>> importance_ranker = ImportanceRanker(
    ...     model_class=SoloModel,
    ...     model_params={'estimator': CatBoostClassifier()},
    ...     sorting_mode='rfe'
    ... )
    >>> ranked_features, scores = importance_ranker.run(df, features, 'target', 'treatment')

Notes:
    FilterRanker works only with train data and requires no model.
    ImportanceRanker works only with train data and requires model to have feature_importances_ attribute.
    Other rankers require both train and validation data.
"""

from .filter import FilterRanker
from .importance import ImportanceRanker
from .permutation import PermutationRanker
from .stepwise import StepwiseRanker
from .straightforward import StraightforwardRanker

__all__ = [
    "FilterRanker",
    "ImportanceRanker",
    "PermutationRanker",
    "StepwiseRanker",
    "StraightforwardRanker",
]
