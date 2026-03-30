"""Uplift-aware data preprocessing pipeline.

Provides MissingValueHandler, CategoryEncoder and main Preprocessor class
for end-to-end data preparation in uplift modeling.

Classes:
    Preprocessor: Main preprocessing pipeline combining missing value handling and encoding.
    MissingValueHandler: Handles missing values with min/mean/median/max strategies.
    CategoryEncoder: Encodes categoricals using label/target/uplift methods.

Examples:
    >>> from auf.data.preprocessing import Preprocessor
    >>> preprocessor = Preprocessor(
    ...     num_fill_strategy='mean',
    ...     encoding_method='target'
    ... )
    >>> preprocessor.fit(X_train, features, y_train, treatment_train)
    >>> X_processed = preprocessor.transform(X_test)
    >>> original = preprocessor.inversed_transform(X_processed)

Notes:
    Preprocessor automatically detects numerical vs categorical features.
    Use keep_features() to narrow down feature set after fitting.
    CategoryEncoder supports label, target and uplift encoding strategies.
"""

import typing as tp

import numpy as np
import pandas as pd

from ..log import get_logger

logger = get_logger(verbosity=1)


class MissingValueHandler:
    """Missing value imputer for numerical and categorical features.

    Supports multiple imputation strategies for numerical features and
    constant fill for categorical features. Preserves imputation values
    for inverse transformation.

    Attributes:
        _num_fill_strategy (str): Strategy for numerical imputation
            ('min', 'mean', 'median', 'max').
        _cat_fill_value (str): Constant value for categorical missing values.
        _num_features (List[str]): List of numerical feature names.
        _cat_features (List[str]): List of categorical feature names.
        _num_fill_values (Dict[str, float]): Mapping numerical features to fill values.
        _cat_fill_values (Dict[str, str]): Mapping categorical features to fill values.
        _is_fitted (bool): True if fit() was called successfully.

    Examples:
        >>> handler = MissingValueHandler(num_fill_strategy='mean', cat_fill_value='__none__')
        >>> handler.fit(X_train, num_features=['age', 'income'], cat_features=['city'])
        >>> X_filled = handler.transform(X_test)
        >>> X_original = handler.inversed_transform(X_filled)

    Notes:
        'min' strategy fills with (min - 1), 'max' with (max + 1).
        Inverse transform uses np.isclose for numerical value matching.
    """

    def __init__(
        self, num_fill_strategy: str = "mean", cat_fill_value: str = "__none__"
    ):
        """Initialize MissingValueHandler.

        Args:
            num_fill_strategy: Strategy for filling missing values in numerical
                features. Must be one of: 'min', 'mean', 'median', 'max'.
                Defaults to 'mean'.
            cat_fill_value: Value for filling missing values in categorical
                features. Defaults to '__none__'.

        Raises:
            AssertionError: If num_fill_strategy is not one of the valid options.
        """
        assert num_fill_strategy in [
            "min",
            "mean",
            "median",
            "max",
        ], "Value of num_fill_strategy must be one of: 'min', 'mean', 'median', 'max'."

        self._num_fill_strategy: str = num_fill_strategy
        self._num_features: tp.List[str] = []
        self._num_fill_values: tp.Dict[str, float] = {}

        self._cat_fill_value: str = cat_fill_value
        self._cat_features: tp.List[str] = []
        self._cat_fill_values: tp.Dict[str, str] = {}

        self._is_fitted: bool = False

    def fit(
        self,
        X: pd.DataFrame,
        num_features: tp.List[str],
        cat_features: tp.List[str],
    ) -> "MissingValueHandler":
        """Fit MissingValueHandler on training data.

        Computes fill values for numerical features based on the selected strategy
        and stores categorical fill values for later transformation.

        Args:
            X: DataFrame with training data.
            num_features: List of numerical feature names to process.
            cat_features: List of categorical feature names to process.

        Returns:
            self: Fitted MissingValueHandler instance.

        Raises:
            AssertionError: If X does not contain all specified features.

        Examples:
            >>> handler = MissingValueHandler(num_fill_strategy='mean')
            >>> handler.fit(X_train, num_features=['age'], cat_features=['city'])
        """
        assert not set(num_features).difference(
            set(X.columns)
        ), "X must contain all specified numerical features"
        assert not set(cat_features).difference(
            set(X.columns)
        ), "X must contain all specified categorical features"

        self._num_features = num_features
        self._cat_features = cat_features

        for feature in num_features:
            if self._num_fill_strategy == "min":
                self._num_fill_values[feature] = X[feature].min() - 1
            elif self._num_fill_strategy == "mean":
                self._num_fill_values[feature] = X[feature].mean()
            elif self._num_fill_strategy == "median":
                self._num_fill_values[feature] = X[feature].median()
            else:
                self._num_fill_values[feature] = X[feature].max() + 1

        for feature in cat_features:
            self._cat_fill_values[feature] = self._cat_fill_value

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame, inplace: bool = True) -> pd.DataFrame:
        """Transform DataFrame by filling missing values.

        Applies learned fill values to numerical features and constant fill
        to categorical features.

        Args:
            X: DataFrame with data to transform.
            inplace: If False, returns a transformed copy. Defaults to True.

        Returns:
            DataFrame with missing values filled.

        Raises:
            AssertionError: If handler is not fitted.
            AssertionError: If X does not contain all features from fit().

        Examples:
            >>> X_filled = handler.transform(X_test, inplace=False)
        """
        assert self._is_fitted, "MissingValueHandler must be fitted"
        assert not set(self._num_features).difference(
            set(X.columns)
        ), "X must contain all numerical features passed in fit"
        assert not set(self._cat_features).difference(
            set(X.columns)
        ), "X must contain all categorical features passed in fit"

        if not inplace:
            X = X.copy()

        for feature, value in self._num_fill_values.items():
            X[feature] = X[feature].fillna(value)

        for feature, value in self._cat_fill_values.items():
            X[feature] = X[feature].fillna(value)

        return X

    def keep_features(
        self, num_features: tp.List[str], cat_features: tp.List[str]
    ):
        """Keep only specified features in fitted handler.

        Removes fill values for features not in the provided lists.
        Useful for narrowing down feature set after initial fitting.

        Args:
            num_features: List of numerical feature names to keep.
            cat_features: List of categorical feature names to keep.

        Returns:
            self: Handler with narrowed feature set.

        Raises:
            AssertionError: If handler is not fitted.
            AssertionError: If unknown features are passed.

        Examples:
            >>> handler.keep_features(num_features=['age'], cat_features=['city'])
        """
        assert (
            self._is_fitted
        ), "MissingValueHandler is not fitted. Call fit() first."
        assert not set(num_features).difference(
            set(self._num_features)
        ), "Unknown numerical features provided"
        assert not set(cat_features).difference(
            set(self._cat_features)
        ), "Unknown categorical features provided"

        self._num_features = [
            f for f in self._num_features if f in num_features
        ]
        self._cat_features = [
            f for f in self._cat_features if f in cat_features
        ]

        self._num_fill_values = {
            f: self._num_fill_values[f] for f in num_features
        }
        self._cat_fill_values = {
            f: self._cat_fill_values[f] for f in cat_features
        }
        return self

    def inversed_transform(
        self, X: pd.DataFrame, inplace: bool = True
    ) -> pd.DataFrame:
        """Restore missing values in transformed DataFrame.

        Replaces fill values with None to recover original missing value pattern.
        Uses np.isclose for numerical value matching to handle floating-point precision.

        Args:
            X: DataFrame with transformed data.
            inplace: If False, returns a transformed copy. Defaults to True.

        Returns:
            DataFrame with fill values replaced by None.

        Raises:
            AssertionError: If handler is not fitted.
            AssertionError: If X does not contain all features from fit().

        Examples:
            >>> X_original = handler.inversed_transform(X_filled)

        Notes:
            Numerical matching uses relative tolerance of 1e-10.
        """
        assert (
            self._is_fitted
        ), "MissingValueHandler is not fitted. Call fit() before inversed_transform()."
        assert not set(self._num_features).difference(
            set(X.columns)
        ), "X must contain all numerical features from fit()"
        assert not set(self._cat_features).difference(
            set(X.columns)
        ), "X must contain all categorical features from fit()"

        if not inplace:
            X = X.copy()

        for feature, value in self._num_fill_values.items():
            nan_mask = np.isclose(X[feature], value, rtol=1e-10)
            X.loc[nan_mask, feature] = None

        for feature, value in self._cat_fill_values.items():
            nan_mask = X[feature] == value
            X.loc[nan_mask, feature] = None

        return X


class CategoryEncoder:
    """Categorical feature encoder.

    Supports three encoding methods:
        - 'label': Frequency-based label encoding.
        - 'target': Target encoding with smoothing for rare categories.
        - 'uplift': Uplift encoding using treatment/control mean difference.

    Handles rare categories by collapsing to '__other__' and missing values
    as a separate '__none__' category with extreme encoding values.

    Attributes:
        _encoding_method (str): Current encoding method ('label', 'target', 'uplift').
        _max_top_categories_cnt (int): Maximum categories to encode separately.
        _other_category (str): Value for rare categories outside top-N.
        _missing_category (str): Value representing missing categories.
        _smoothing (float): Smoothing parameter for target/uplift encoding.
        _min_samples_leaf (int): Minimum samples for computing category statistics.
        _verbose (bool): If True, prints processing information.
        _cat_features (List[str]): List of categorical feature names.
        _top_categories (Dict[str, List]): Mapping features to their top category lists.
        _category_to_label (Dict[str, Dict]): Mapping features to category-to-label dicts.
        _label_to_category (Dict[str, Dict]): Mapping features to label-to-category dicts.
        _is_fitted (bool): True if fit() was called successfully.

    Examples:
        >>> encoder = CategoryEncoder(encoding_method='target', max_top_categories_cnt=10)
        >>> encoder.fit(X_train, y=y_train, cat_features=['city', 'product'])
        >>> X_encoded = encoder.transform(X_test)
        >>> X_decoded = encoder.inversed_transform(X_encoded)

    Notes:
        For 'target' and 'uplift' encoding, y is required.
        For 'uplift' encoding, treatment vector is required.
        Missing values must be filled before fitting (use MissingValueHandler).
    """

    def __init__(
        self,
        encoding_method: str = "target",
        max_top_categories_cnt: int = 10,
        other_category: str = "__other__",
        missing_category: str = "__none__",
        smoothing: float = 1.0,
        min_samples_leaf: int = 1,
        verbose: bool = False,
    ):
        """Initialize CategoryEncoder.

        Args:
            encoding_method: Encoding method to use. Must be one of:
                'label' (frequency-based), 'target' (target encoding),
                'uplift' (uplift encoding). Defaults to 'target'.
            max_top_categories_cnt: Maximum number of top categories to encode
                separately per feature. Rare categories collapse to other_category.
                Defaults to 10.
            other_category: Value for rare categories outside top-N.
                Defaults to '__other__'.
            missing_category: Value representing missing categories.
                Defaults to '__none__'.
            smoothing: Smoothing parameter for target/uplift encoding.
                Higher values give more weight to global statistics.
                Defaults to 1.0.
            min_samples_leaf: Minimum samples required to compute category
                statistics independently. Categories with fewer samples use
                smoothed global statistics. Defaults to 1.
            verbose: If True, prints processing information during fit.
                Defaults to False.

        Raises:
            ValueError: If encoding_method is not one of ['label', 'target', 'uplift'].
            ValueError: If max_top_categories_cnt < 1.
            ValueError: If smoothing < 0.
            ValueError: If min_samples_leaf < 1.
        """
        self._encoding_method = encoding_method
        self._max_top_categories_cnt = max_top_categories_cnt
        self._other_category = other_category
        self._missing_category = missing_category
        self._smoothing = smoothing
        self._min_samples_leaf = min_samples_leaf
        self._verbose = verbose
        self._is_fitted = False
        self._cat_features: tp.List[str] = []
        self._top_categories = {}
        self._category_to_label = {}
        self._label_to_category = {}

        valid_methods = ["label", "target", "uplift"]
        if encoding_method not in valid_methods:
            raise ValueError(
                f"Invalid encoding method: {encoding_method}. Must be one of: {valid_methods}"
            )

        if max_top_categories_cnt < 1:
            raise ValueError(
                "Value of max_top_categories_cnt must be not less than 1"
            )

        if smoothing < 0:
            raise ValueError("Value of smoothing must be not less than 0")

        if min_samples_leaf < 1:
            raise ValueError(
                "Value of min_samples_leaf must be not less than 1"
            )

    def fit(
        self,
        X: pd.DataFrame,
        y: tp.Optional[pd.Series] = None,
        treatment: tp.Optional[pd.Series] = None,
        cat_features: tp.List[str] = None,
    ) -> "CategoryEncoder":
        """Fit CategoryEncoder on training data.

        Learns category-to-label mappings based on the selected encoding method.
        For 'target' and 'uplift' methods, computes statistics using target
        and treatment vectors.

        Args:
            X: DataFrame with categorical features.
            y: Target variable. Required for 'target' and 'uplift' encoding.
            treatment: Treatment vector (0=control, 1=treatment).
                Required for 'uplift' encoding only.
            cat_features: List of categorical feature names. If None, automatically
                detects object and category dtype columns.

        Returns:
            self: Fitted CategoryEncoder instance.

        Raises:
            ValueError: If cat_features contain columns not in X.
            ValueError: If y is None for 'target' or 'uplift' encoding.
            ValueError: If treatment is None for 'uplift' encoding.
            ValueError: If X, y, or treatment have mismatched lengths.
            ValueError: If missing values detected in categorical features.
            ValueError: If control or treatment group is empty (uplift only).

        Examples:
            >>> encoder = CategoryEncoder(encoding_method='target')
            >>> encoder.fit(X_train, y=y_train, cat_features=['city'])
        """
        if cat_features is None:
            cat_features = X.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

        if not cat_features:
            self._is_fitted = True
            self._cat_features = []
            return self

        missing_features = set(cat_features) - set(X.columns)
        if missing_features:
            raise ValueError(f"Features not found in data: {missing_features}")

        if self._encoding_method in ["target", "uplift"] and y is None:
            raise ValueError(
                "Parameter y is required for target/uplift encoding."
            )

        if self._encoding_method == "uplift" and treatment is None:
            raise ValueError(
                "Parameter treatment is required for uplift encoding."
            )

        if y is not None and len(X) != len(y):
            raise ValueError("X and y must have the same length.")

        if treatment is not None and len(X) != len(treatment):
            raise ValueError("X and treatment must have the same length.")

        missing_mask = X[cat_features].isna()
        if missing_mask.any().any():
            raise ValueError(
                f"Missing values detected in categorical features. Fill them with '{self._missing_category}' first."
            )

        self._cat_features = cat_features

        # Global statistics
        global_mean = y.mean() if y is not None else None
        global_cntrl = None
        global_trmnt = None

        if self._encoding_method == "uplift":
            control_mask = treatment == 0
            treatment_mask = treatment != 0

            if not control_mask.any():
                raise ValueError("Control group is empty.")
            if not treatment_mask.any():
                raise ValueError("Treatment group is empty.")

            global_cntrl = y[control_mask].mean()
            global_trmnt = y[treatment_mask].mean()
            global_uplift = global_trmnt - global_cntrl

            if self._verbose:
                logger.info(f"Global mean uplift: {global_uplift:.4f}")

        for feature in cat_features:
            if self._verbose:
                logger.info(f"\nCheck feature: {feature}")

            # Top categories without specials
            mask = X[feature] != self._missing_category
            freq = X.loc[mask, feature].value_counts()
            top_categories = freq.nlargest(
                self._max_top_categories_cnt
            ).index.tolist()

            # Include special categories
            top_categories = list(
                set(top_categories)
                | {self._other_category, self._missing_category}
            )
            self._top_categories[feature] = top_categories

            # Compute per-category statistics
            value_map = {}

            for category in top_categories:
                if category in [self._other_category, self._missing_category]:
                    continue

                mask = X[feature] == category
                count = mask.sum()

                if self._encoding_method == "label":
                    value_map[category] = freq[category]

                elif self._encoding_method == "target" and y is not None:
                    if count >= self._min_samples_leaf:
                        value_map[category] = y[mask].mean()
                    else:  # smoothing for rare categories
                        value_map[category] = (
                            y[mask].sum() + global_mean * self._smoothing
                        ) / (count + self._smoothing)

                elif (
                    self._encoding_method == "uplift"
                    and y is not None
                    and treatment is not None
                ):
                    if count >= self._min_samples_leaf:
                        cat_mask_cntrl = mask & (treatment == 0)
                        cat_mask_trmnt = mask & (treatment != 0)

                        mean_cntrl = (
                            y[cat_mask_cntrl].mean()
                            if cat_mask_cntrl.any()
                            else global_cntrl
                        )

                        mean_trmnt = (
                            y[cat_mask_trmnt].mean()
                            if cat_mask_trmnt.any()
                            else global_trmnt
                        )

                        value_map[category] = mean_trmnt - mean_cntrl
                    else:
                        value_map[category] = global_uplift

            value_map[self._other_category] = float("-inf")
            value_map[self._missing_category] = float("inf")

            sorted_categories = sorted(
                value_map.keys(),
                key=lambda x: (
                    value_map[x],
                    str(x),
                ),
            )

            self._category_to_label[feature] = {
                cat: idx for idx, cat in enumerate(sorted_categories)
            }
            self._label_to_category[feature] = {
                idx: cat for idx, cat in enumerate(sorted_categories)
            }

            if self._verbose:
                logger.info(
                    f"Unique categories count: {len(sorted_categories)}"
                )
                logger.info(
                    f"Mapping examples: {list(self._category_to_label[feature].items())[:5]}"
                )

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame, inplace: bool = True) -> pd.DataFrame:
        """Transform DataFrame by encoding categorical features.

        Converts categorical values to numerical labels using learned mappings.
        Rare categories not in top-N are replaced with other_category label.

        Args:
            X: DataFrame with categorical features to encode.
            inplace: If False, returns a transformed copy. Defaults to True.

        Returns:
            DataFrame with categorical features encoded as numerical labels.

        Raises:
            RuntimeError: If encoder is not fitted.
            ValueError: If feature from fit() is missing in X.

        Examples:
            >>> X_encoded = encoder.transform(X_test, inplace=False)
        """
        if not self._is_fitted:
            raise RuntimeError(
                "CategoryEncoder is not fitted. Call fit() before transform()."
            )

        if not inplace:
            X = X.copy()

        if not self._cat_features:
            return X

        for feature in self._cat_features:
            if feature not in X.columns:
                raise ValueError(f"Feature '{feature}' not found in data.")

            feature_values = X[feature].copy()
            top_mask = feature_values.isin(self._top_categories[feature])
            feature_values[~top_mask] = self._other_category
            X[feature] = feature_values.map(self._category_to_label[feature])

        return X

    def keep_features(self, cat_features: tp.List[str]) -> "CategoryEncoder":
        """Keep only specified features in fitted encoder.

        Removes category mappings for features not in the provided list.
        Useful for narrowing down feature set after initial fitting.

        Args:
            cat_features: List of categorical feature names to keep.

        Returns:
            self: Encoder with narrowed feature set.

        Raises:
            RuntimeError: If encoder is not fitted.
            ValueError: If unknown features are passed.

        Examples:
            >>> encoder.keep_features(cat_features=['city', 'product'])
        """
        if not self._is_fitted:
            raise RuntimeError(
                "CategoryEncoder is not fitted. Call fit() first."
            )

        unknown_features = set(cat_features) - set(self._cat_features)
        if unknown_features:
            raise ValueError(f"Unknown features: {unknown_features}.")

        self._cat_features = [
            f for f in self._cat_features if f in cat_features
        ]
        self._top_categories = {
            f: self._top_categories[f] for f in self._cat_features
        }
        self._category_to_label = {
            f: self._category_to_label[f] for f in self._cat_features
        }
        self._label_to_category = {
            f: self._label_to_category[f] for f in self._cat_features
        }

        return self

    def inversed_transform(
        self, X: pd.DataFrame, inplace: bool = True
    ) -> pd.DataFrame:
        """Restore original categories from encoded labels.

        Converts numerical labels back to original category values using
        learned reverse mappings.

        Args:
            X: DataFrame with encoded categorical features.
            inplace: If False, returns a transformed copy. Defaults to True.

        Returns:
            DataFrame with categorical features restored to original values.

        Raises:
            RuntimeError: If encoder is not fitted.
            ValueError: If feature from fit() is missing in X.

        Examples:
            >>> X_decoded = encoder.inversed_transform(X_encoded)

        Notes:
            Unknown labels are replaced with other_category value.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "CategoryEncoder is not fitted. Call fit() before inversed_transform()."
            )

        if not inplace:
            X = X.copy()

        if not self._cat_features:
            return X

        for feature in self._cat_features:
            if feature not in X.columns:
                raise ValueError(f"Feature '{feature}' not found in data.")

            X[feature] = X[feature].apply(
                lambda label: self._label_to_category[feature].get(
                    label, self._other_category
                )
            )

        return X


class Preprocessor:
    """End-to-end preprocessing pipeline for uplift modeling.

    Combines missing value handling and categorical encoding into a single
    sklearn-like interface with fit/transform pattern. Automatically detects
    numerical and categorical feature types.

    Attributes:
        _num_fill_strategy (str): Strategy for numerical imputation
            ('min', 'mean', 'median', 'max').
        _cat_fill_value (str): Constant value for categorical missing values.
        _encoding_method (str): Encoding method ('label', 'target', 'uplift').
        _max_top_categories_cnt (int): Maximum categories to encode separately.
        _other_category (str): Value for rare categories outside top-N.
        _num_features (List[str]): List of detected numerical feature names.
        _cat_features (List[str]): List of detected categorical feature names.
        _missing_handler (MissingValueHandler): Internal missing value handler.
        _category_encoder (CategoryEncoder): Internal categorical encoder.
        _is_fitted (bool): True if fit() was called successfully.

    Examples:
        >>> preprocessor = Preprocessor(
        ...     num_fill_strategy='mean',
        ...     encoding_method='uplift'
        ... )
        >>> preprocessor.fit(X_train, features, y_train, treatment_train)
        >>> X_processed = preprocessor.transform(X_test)
        >>> X_original = preprocessor.inversed_transform(X_processed)

    Notes:
        Automatically detects feature types via float casting attempt.
        For 'target' encoding, y is required in fit().
        For 'uplift' encoding, both y and treatment are required in fit().
        Use keep_features() to remove features from fitted preprocessor.
    """

    def __init__(
        self,
        num_fill_strategy: str = "min",
        cat_fill_value: str = "__none__",
        encoding_method: str = "target",
        max_top_categories_cnt: int = 10,
        other_category: str = "__other__",
    ):
        """Initialize Preprocessor.

        Args:
            num_fill_strategy: Strategy for filling missing values in numerical
                features. Must be one of: 'min', 'mean', 'median', 'max'.
                Defaults to 'min'.
            cat_fill_value: Value for filling missing values in categorical
                features. Defaults to '__none__'.
            encoding_method: Encoding method for categorical features.
                Must be one of: 'label', 'target', 'uplift'.
                Defaults to 'target'.
            max_top_categories_cnt: Maximum number of top categories to encode
                separately per feature. Defaults to 10.
            other_category: Value for rare categories outside top-N.
                Defaults to '__other__'.

        Notes:
            Feature types (numerical vs categorical) are detected automatically
            during fit() via float casting attempt.
        """
        self._num_features: tp.List[str] = []
        self._cat_features: tp.List[str] = []

        self._other_category: str = other_category
        self._max_top_categories_cnt: int = max_top_categories_cnt
        self._encoding_method: str = encoding_method
        self._category_encoder: CategoryEncoder = None

        self._num_fill_strategy: str = num_fill_strategy
        self._cat_fill_value: str = cat_fill_value
        self._missing_handler: MissingValueHandler = None

        self._is_fitted: bool = False

    def _get_num_cat_features(self, X, features):
        self._num_features = []
        self._cat_features = []
        for f in features:
            try:
                casted = X[f].astype(float)
                del casted
                self._num_features.append(f)
            except (ValueError, TypeError):
                self._cat_features.append(f)

    def _cast_num_cat_features(self, X):
        for f in self._num_features:
            nan_mask = X[f].isna()
            X[f] = X[f].astype(float)
            X.loc[nan_mask, f] = None

        for f in self._cat_features:
            nan_mask = X[f].isna()
            X[f] = X[f].astype(str)
            X.loc[nan_mask, f] = None

    def fit(
        self,
        X: pd.DataFrame,
        features: tp.List[str] = None,
        y: pd.Series = None,
        treatment: pd.Series = None,
    ) -> "Preprocessor":
        """Fit Preprocessor on training data.

        Automatically detects numerical and categorical feature types,
        then fits internal MissingValueHandler and CategoryEncoder.

        Args:
            X: DataFrame with training data.
            features: List of feature names to preprocess. If None, uses all columns.
            y: Target variable. Required for 'target' and 'uplift' encoding.
            treatment: Treatment vector (0=control, 1=treatment).
                Required for 'uplift' encoding only.

        Returns:
            self: Fitted Preprocessor instance.

        Raises:
            AssertionError: If X does not contain all specified features.
            AssertionError: If y is None for 'target' or 'uplift' encoding.
            AssertionError: If treatment is None for 'uplift' encoding.

        Examples:
            >>> preprocessor = Preprocessor(encoding_method='uplift')
            >>> preprocessor.fit(X_train, features, y_train, treatment_train)

        Notes:
            Feature types are detected automatically via float casting attempt.
        """
        assert not set(features).difference(
            set(X.columns)
        ), "X must contain all specified features."

        if self._encoding_method in ["target", "uplift"] and y is None:
            raise AssertionError(
                "Parameter y is required for target or uplift encoding."
            )
        if self._encoding_method not in ["target", "uplift"] and y is not None:
            raise AssertionError(
                "Parameter y should be None for label encoding."
            )

        assert not (
            treatment is None and self._encoding_method == "uplift"
        ), "Parameter treatment is required for uplift encoding."

        X = X.copy()
        self._get_num_cat_features(X, features)
        self._cast_num_cat_features(X)

        self._missing_handler = MissingValueHandler(
            self._num_fill_strategy, self._cat_fill_value
        )
        self._missing_handler.fit(X, self._num_features, self._cat_features)
        X = self._missing_handler.transform(X, inplace=True)

        self._category_encoder = CategoryEncoder(
            encoding_method=self._encoding_method,
            max_top_categories_cnt=self._max_top_categories_cnt,
            other_category=self._other_category,
            missing_category=self._cat_fill_value,
        )
        self._category_encoder.fit(
            X, y, treatment=treatment, cat_features=self._cat_features
        )
        del X

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame, inplace: bool = True) -> pd.DataFrame:
        """Transform DataFrame using fitted handlers and encoder.

        Applies missing value imputation and categorical encoding in sequence.

        Args:
            X: DataFrame with data to transform.
            inplace: If False, returns a transformed copy. Defaults to True.

        Returns:
            DataFrame with preprocessed features.

        Raises:
            AssertionError: If preprocessor is not fitted.
            AssertionError: If X does not contain all features from fit().

        Examples:
            >>> X_processed = preprocessor.transform(X_test, inplace=False)
        """
        assert (
            self._is_fitted
        ), "Preprocessor is not fitted. Call fit() before transform()."
        assert not set(self._num_features).difference(
            set(X.columns)
        ), "X must contain all numerical features from fit()."
        assert not set(self._cat_features).difference(
            set(X.columns)
        ), "X must contain all categorical features from fit()."

        if not inplace:
            X = X.copy()

        self._cast_num_cat_features(X)
        X = self._missing_handler.transform(X, inplace=True)
        X = self._category_encoder.transform(X, inplace=True)

        return X

    def keep_features(self, features: tp.List[str]):
        """Keep only specified features in fitted preprocessor.

        Propagates feature selection to internal MissingValueHandler
        and CategoryEncoder. Useful for narrowing down feature set
        after initial fitting.

        Args:
            features: List of feature names to keep.

        Returns:
            self: Preprocessor with narrowed feature set.

        Raises:
            AssertionError: If preprocessor is not fitted.
            AssertionError: If duplicate features are passed.
            AssertionError: If unknown features are passed.

        Examples:
            >>> preprocessor.keep_features(features=['age', 'city', 'income'])
        """
        assert self._is_fitted, "Preprocessor is not fitted. Call fit() first."
        assert len(features) == len(
            set(features)
        ), "Duplicate features provided."
        assert not set(features).difference(
            set(self._num_features) | set(self._cat_features)
        ), "Unknown features provided."

        self._num_features = [f for f in self._num_features if f in features]
        self._cat_features = [f for f in self._cat_features if f in features]

        self._missing_handler.keep_features(
            num_features=self._num_features, cat_features=self._cat_features
        )

        self._category_encoder.keep_features(cat_features=self._cat_features)

        return self

    def inversed_transform(
        self, X: pd.DataFrame, inplace: bool = True
    ) -> pd.DataFrame:
        """Restore original data from preprocessed DataFrame.

        Reverses categorical encoding and missing value imputation to
        recover the original data structure.

        Args:
            X: DataFrame with preprocessed data.
            inplace: If False, returns a transformed copy. Defaults to True.

        Returns:
            DataFrame with original categorical values and missing value pattern.

        Raises:
            AssertionError: If preprocessor is not fitted.
            AssertionError: If X does not contain all features from fit().

        Examples:
            >>> X_original = preprocessor.inversed_transform(X_processed)

        Notes:
            Inverse transform is applied in reverse order: encoding first,
            then missing value restoration.
        """
        assert (
            self._is_fitted
        ), "Preprocessor is not fitted. Call fit() before inversed_transform()."
        assert not set(self._num_features).difference(
            set(X.columns)
        ), "X must contain all numerical features from fit()."
        assert not set(self._cat_features).difference(
            set(X.columns)
        ), "X must contain all categorical features from fit()."

        if not inplace:
            X = X.copy()

        X = self._category_encoder.inversed_transform(X, inplace=True)
        X = self._missing_handler.inversed_transform(X, inplace=True)

        return X
