"""Uplift score calibration utilities.

Provides UpliftCalibrator class for adjusting raw uplift predictions to
match observed uplift values in each score bucket. This technique helps
to correct model bias and improve the interpretability of uplift scores.

Classes:
    UpliftCalibrator: Bucket-level calibrator that stores coefficients
        for each uplift score range.

Examples:
    >>> from auf.pipeline.calibration import UpliftCalibrator
    >>> from auf.models import AufModel
    >>> import numpy as np
    >>> import pandas as pd

    >>> # Assume 'model' is a trained AufModel instance
    >>> calibrator = UpliftCalibrator()
    >>> calibrator.fit(val_data, model, bins=10)

    >>> raw_uplift = model.predict(test_data[model._features])
    >>> calibrated_uplift = calibrator.predict(raw_uplift)

Notes:
    Calibration is performed on a validation dataset separate from the
    training set. The calibrator supports both absolute and relative
    uplift prediction types. Can be applied only for binary-tratment case.
"""

import typing as tp

import numpy as np
import pandas as pd
from sklift.metrics import uplift_by_percentile

from ..models import AufModel


class UpliftCalibrator:
    """Uplift bucket-level calibrator for binary-treatment case.

    Stores calibration coefficients for every uplift score bucket.
    Calibration adjusts raw uplift predictions to align with observed
    uplift values within discrete score ranges.

    How calibration fitting works:
        1. Use model to get uplift predictions for calibration dataset.
        2. Split this dataset by predicted uplift buckets.
        3. In each bucket calculate mean real and mean predicted uplifts.
        4. Calibration coefficient for this bucket is the ratio of mean
           real to mean predicted uplift.

    How calibration prediction works:
        1. Predict uplift with the same model.
        2. Determine uplift score bucket for every object.
        3. Multiply uplift score by corresponding coefficient for every
           object.

    Attributes:
        _borders (np.array): Array of upper borders for each bucket.
            The last element is always infinity.
        _coeffs (np.array): Array of calibration coefficients for each
            bucket. Each coefficient is the ratio of mean real uplift to
            mean predicted uplift in the bucket.

    Examples:
        >>> calibrator = UpliftCalibrator()
        >>> calibrator.fit(val_data, model, bins=10)
        >>> calibrated = calibrator.predict(raw_uplift_scores)

    Notes:
        The calibrator assumes binary treatment (control vs treatment).
        For relative uplift type, coefficients are based on
        (treatment_rate / control_rate - 1).
    """

    def __init__(self):
        """Initialize UpliftCalibrator.

        Sets internal attributes to None. They are populated during the
        fit() method call.

        Returns:
            None
        """
        self._borders: np.array = None
        self._coeffs: np.array = None

    def _check_base_columns(
        self,
        data: pd.DataFrame,
        base_cols_mapper: tp.Dict[str, str],
        treatment_groups_mapper: tp.Dict[tp.Any, int],
    ):
        assert np.all(
            [
                (base_col == "segm") or (col is None) or (col in data.columns)
                for base_col, col in base_cols_mapper.items()
            ]
        )
        treatment_col = base_cols_mapper["treatment"]
        assert set(treatment_groups_mapper.keys()) == set(
            data[treatment_col].values
        )
        assert set(treatment_groups_mapper.values()) == set([0, 1])

    def _check_features(self, data: pd.DataFrame, features: tp.List[str]):
        assert all([col in data.columns for col in features])

    def fit(
        self,
        data: pd.DataFrame,
        model: AufModel,
        base_cols_mapper: tp.Dict[str, str] = {
            "id": "id",
            "treatment": "treatment",
            "target": "target",
            "segm": "segm",
        },
        treatment_groups_mapper: tp.Dict[tp.Any, int] = {0: 0, 1: 1},
        bins: int = 10,
    ):
        """Fit the calibrator on a validation dataset.

        Calculates calibration coefficients for each uplift bucket based
        on the ratio of observed uplift to predicted uplift.

        Args:
            data: Sample for calibrating which must be
                preprocessed at first. Usually a validation dataset.
            model: Trained uplift model to calibrate.
            base_cols_mapper: Mapping for main sample
                column names. Defaults to {"id": "id", "treatment":
                "treatment", "target": "target", "segm": "segm"}.
            treatment_groups_mapper: Mapping for
                treatment group names. Defaults to {0: 0, 1: 1}.
            bins: Number of bins for calibrating. Defaults to 10.

        Returns:
            UpliftCalibrator: Fitted calibrator instance (self).

        Raises:
            AssertionError: If base columns or features are missing from
                data, or if treatment groups are invalid.

        Notes:
            The method uses sklift.metrics.uplift_by_percentile to
            calculate observed uplift in each bucket. For absolute uplift
            type, observed uplift is the difference between treatment and
            control response rates. For relative uplift type, it is the
            ratio minus one.
        """
        self._check_base_columns(
            data, base_cols_mapper, treatment_groups_mapper
        )
        self._check_features(data, model._features)

        target = data[base_cols_mapper["target"]]
        treatment = data[base_cols_mapper["treatment"]].map(
            treatment_groups_mapper
        )
        uplift = model.predict(data[model._features])["score_raw"].values

        buckets = np.array_split(np.sort(uplift), bins)
        mean_uplift_preds = np.array([np.mean(bucket) for bucket in buckets])
        self._borders = np.array([np.max(bucket) for bucket in buckets])
        self._borders[-1] = np.inf

        buckets_info = uplift_by_percentile(
            target, uplift, treatment, bins=bins
        )

        if model._uplift_prediction_type == "abs":
            mean_uplifts = buckets_info["uplift"].values[::-1]
        else:
            treat_rates = buckets_info["response_rate_treatment"].values[::-1]
            cntrl_rates = buckets_info["response_rate_control"].values[::-1]
            mean_uplifts = treat_rates / cntrl_rates - 1

        self._coeffs = mean_uplifts / mean_uplift_preds

        return self

    def predict(self, uplift: np.array):
        """Apply calibration to raw uplift scores.

        Multiplies each uplift score by the coefficient corresponding to
        its bucket.

        Args:
            uplift: Raw uplift scores to calibrate.

        Returns:
            np.array: Calibrated uplift scores.

        Notes:
            Bucket assignment is performed using np.digitize with the
            borders stored during fit(). The right=True parameter means
            that buckets include the right border.
        """
        idxs = np.digitize(uplift, self._borders, right=True)
        calibrated_uplift = uplift * self._coeffs[idxs]
        return calibrated_uplift
