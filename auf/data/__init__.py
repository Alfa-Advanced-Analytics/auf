"""Data processing utilities for AUF.

Provides data validation, preprocessing and splitting functionality
specifically designed for uplift modeling tasks with treatment/control
balance preservation.

Functions:
    check_bernoulli_dependence: Tests dependency between two binary flags.
    check_bernoulli_equal_means: Compares means of two binary distributions.
    check_nans: Filters features exceeding NaN ratio threshold.
    check_too_less_unique_value: Removes features with less than two unique values.
    process_too_much_categories: Collapses rare categories into "__others__".
    check_leaks_v2: Detects treatment/target leaks using bootstrap.
    check_correlations: Identifies correlated feature pairs above threshold.
    check_train_val_test_split: Validates treatment/target balance across splits.
    train_val_test_split: Creates stratified train/val/test splits with treatment/target balance.

Classes:
    MissingValueHandler: Imputes missing values with min/mean/median/max strategies.
    CategoryEncoder: Encodes categoricals via label/target/uplift methods.
    Preprocessor: Main pipeline combining missing handling and encoding.

Examples:
    >>> from auf.data import check_nans, check_leaks_v2, Preprocessor, train_val_test_split

    >>> # Data validation
    >>> features = check_nans(df, features, max_nan_ratio=0.95)
    >>> leaks, clean_features, _ = check_leaks_v2(df, mapper, features, 'treatment')

    >>> # Preprocessing
    >>> preprocessor = Preprocessor(
    ...     num_fill_strategy='mean',
    ...     encoding_method='target'
    ... ).fit(X_train, features, y_train, treatment_train)
    >>> X_processed = preprocessor.transform(X_test)

    >>> # Splitting
    >>> train_idx, val_idx, test_idx = train_val_test_split(
    ...     df, size_ratios=[0.6, 0.2, 0.2], stratify_cols=['treatment', 'target']
    ... )

Notes:
    All validation functions expect binary treatment vector (0/1).
    CategoryEncoder supports 'label', 'target' and 'uplift' encoding strategies.
    Preprocessor automatically detects numeric vs categorical feature types.
    Use deepcopy() before modifying any Preprocessor configuration post-fit.
"""

from .checks import (
    check_bernoulli_dependence,
    check_bernoulli_equal_means,
    check_correlations,
    check_leaks_v2,
    check_nans,
    check_too_less_unique_value,
    check_train_val_test_split,
    process_too_much_categories,
)
from .preprocessing import CategoryEncoder, MissingValueHandler, Preprocessor
from .split import train_val_test_split

__all__ = [
    "check_bernoulli_dependence",
    "check_bernoulli_equal_means",
    "check_nans",
    "check_too_less_unique_value",
    "process_too_much_categories",
    "check_leaks_v2",
    "check_correlations",
    "check_train_val_test_split",
    "CategoryEncoder",
    "MissingValueHandler",
    "Preprocessor",
    "train_val_test_split",
]
