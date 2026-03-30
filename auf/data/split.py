"""Data splitting utilities for uplift modeling.

Provides stratified train/validation/test splits preserving treatment/control
distributions across datasets.

Functions:
    train_val_test_split: Creates three-way split with stratification by treatment/target.

Examples:
    >>> from auf.data.split import train_val_test_split
    >>> train_idx, val_idx, test_idx = train_val_test_split(
    ...     df, size_ratios=[0.6, 0.2, 0.2], stratify_cols=['treatment', 'target']
    ... )
    >>> df_train, df_val, df_test = df[train_idx], df[val_idx], df[test_idx]
"""

import typing as tp

import pandas as pd
from sklearn.model_selection import train_test_split

from ..constants import RANDOM_STATE


def train_val_test_split(
    df: pd.DataFrame,
    size_ratios: tp.List[float] = [0.6, 0.2, 0.2],
    stratify_cols: tp.List[str] = ["target", "treatment"],
):
    """Creates stratified train/validation/test splits.

    Performs two-stage splitting with stratification to maintain
    treatment and target balance across all resulting datasets.

    Args:
        df: Input DataFrame with features, target and treatment columns.
        size_ratios: Ratios [train, val, test] summing to 1.0.
            Defaults to [0.6, 0.2, 0.2].
        stratify_cols: Columns for stratification.
            Defaults to ['target', 'treatment'].

    Returns:
        tuple: Three Index objects (train_idx, val_idx, test_idx).

    Examples:
        >>> idx_train, idx_val, idx_test = train_val_test_split(
        ...     df, size_ratios=[0.7, 0.15, 0.15]
        ... )

    Notes:
        Fixed random_state=RANDOM_STATE for reproducibility.
    """
    df_train_idx, df_val_test_idx = train_test_split(
        df.index,
        train_size=size_ratios[0],
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=df[stratify_cols],
    )

    df_val_idx, df_test_idx = train_test_split(
        df_val_test_idx,
        test_size=size_ratios[2] / (1 - size_ratios[0]),
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=df.loc[df_val_test_idx, stratify_cols],
    )

    return df_train_idx, df_val_idx, df_test_idx
