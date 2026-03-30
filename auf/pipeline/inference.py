"""Model inference utilities for uplift modeling.

Provides UpliftInference class for loading trained artifacts (preprocessor,
model, calibrator) and generating uplift predictions on new data. Supports
loading artifacts either directly from Python objects or from an MLflow
run.

Classes:
    UpliftInference: Main inference class that wraps preprocessing,
        prediction and calibration into a single interface.

Functions:


Examples:
    >>> from auf.pipeline.inference import UpliftInference
    >>> import pandas as pd
    >>>
    >>> # Load from MLflow run
    >>> inference = UpliftInference(run_id='xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    >>> uplift = inference.predict(new_data)
    >>>
    >>> # Or use directly with trained objects
    >>> inference = UpliftInference(
    ...     preprocessor=preprocessor,
    ...     model=model,
    ...     calibrator=calibrator
    ... )
    >>> uplift = inference.predict(new_data, return_df=True)

Notes:
    The inference class automatically handles multi-treatment models by
    loading the appropriate model wrapper from MLflow. For binary
    treatment models, calibration is applied if a calibrator artifact
    exists.
"""

import logging
import pickle
import tempfile
import typing as tp

import mlflow
import pandas as pd

from ..data.preprocessing import Preprocessor
from ..models import AufModel
from ..pipeline.calibration import UpliftCalibrator


class UpliftInference:
    """Full uplift inference pipeline.

    Stores preprocessor, model and calibrator objects and provides a
    unified interface for generating uplift predictions on new data.

    How it works:
        1. Load preprocessor, model and calibrator from MLflow if
           run_id is specified.
        2. If run_id is not provided, use manually passed components.
        3. Apply preprocessing, model prediction and optional calibration.

    Attributes:
        _preprocessor (Preprocessor): Fitted preprocessor for feature
            transformation. None if loading from MLflow.
        _model (AufModel | mlflow.pyfunc.PyFuncModel): Trained uplift
            model or MLflow model wrapper.
        _features (tp.List[str]): List of features required for
            prediction. Populated during predict() call.
        _calibrator (UpliftCalibrator): Fitted calibrator for score
            adjustment. None for multi-treatment models or if not
            available.
        _run_id (str): MLflow run identifier for artifact loading.
            None if using direct object references.

    Examples:
        >>> # Load from MLflow
        >>> inference = UpliftInference(run_id='abc123')
        >>> uplift = inference.predict(df_features)

        >>> # Use with direct objects
        >>> inference = UpliftInference(
        ...     preprocessor=preprocessor,
        ...     model=model,
        ...     calibrator=calibrator
        ... )
        >>> uplift = inference.predict(df_features)

    Notes:
        When loading from MLflow, the model is wrapped as an MLflow
        pyfunc model. Use unwrap_python_model() to access the underlying
        AufModel if needed. Calibration is only applied for binary
        treatment models.
    """

    def __init__(
        self,
        preprocessor: Preprocessor = None,
        model: AufModel = None,
        calibrator: UpliftCalibrator = None,
        run_id: str = None,
    ):
        """Initialize inference class object.

        Args:
            preprocessor: Fitted preprocessor instance.
                Required if run_id is None. Defaults to None.
            model: Trained uplift model. Required if run_id
                is None. Defaults to None.
            calibrator: Fitted calibrator instance.
                Optional, used only for binary treatment models.
                Defaults to None.
            run_id: MLflow run identifier to load artifacts from.
                If provided, preprocessor, model and calibrator arguments
                must be None. Defaults to None.

        Returns:
            None

        Raises:
            AssertionError: If run_id is provided and any of preprocessor,
                model or calibrator is not None.
            AssertionError: If run_id is None and preprocessor or model
                is not provided.
        """
        if run_id is not None:
            assert preprocessor is None and model is None and calibrator is None
        else:
            assert preprocessor is not None and model is not None

        self._preprocessor: Preprocessor = preprocessor
        self._model: AufModel = model
        self._is_multitreatment: bool = (
            model._is_multitreatment if model else False
        )
        self._features: tp.List[str] = None
        self._calibrator: UpliftCalibrator = calibrator
        self._run_id: str = run_id

        self._load_artifacts()

    def _check_features(self, data: pd.DataFrame):
        if self._run_id is None:
            self._features = self._model._features
        else:
            self._features = (
                self._model.unwrap_python_model()._auf_model._features
            )
        assert all([col in data.columns for col in self._features])

    def _load_pickle(self, tmp_dir, artifact_type):
        try:
            local_path = mlflow.artifacts.download_artifacts(
                run_id=self._run_id,
                artifact_path=f"5_best_model_artifacts/{artifact_type}.pkl",
                dst_path=tmp_dir,
            )

            with open(local_path, "rb") as f:
                return pickle.load(f)
        except (
            mlflow.exceptions.MlflowException,
            FileNotFoundError,
            pickle.UnpicklingError,
            OSError,
        ) as e:
            logging.error(f"Ошибка при загрузке артефакта {artifact_type}: {e}")
            return None

    def _load_artifacts(self):
        if self._run_id is None:
            return

        model_uri = f"runs:/{self._run_id}/5_best_model_artifacts"
        self._model = mlflow.pyfunc.load_model(model_uri)

        self._is_multitreatment = (
            self._model.unwrap_python_model()._auf_model._is_multitreatment
        )
        self._is_multitreatment

        if not self._is_multitreatment:
            with tempfile.TemporaryDirectory() as tmp_dir:
                self._calibrator = self._load_pickle(tmp_dir, "calibrator")

    def predict(self, data: pd.DataFrame, return_df: bool = True):
        """Predict uplift scores for new data.

        Applies preprocessing, model prediction and optional calibration
        to generate uplift scores.

        Args:
            data: Input data containing features for
                prediction. Must contain all required features.
            return_df: If True, returns a DataFrame with original
                data plus prediction columns. If False, returns only the
                uplift scores as a Series. Defaults to True.

        Returns:
                If return_df is True, returns a DataFrame with columns
                'score_raw' (calibrated uplift) and optionally 'trmnt_preds',
                'ctrl_preds' for TwoModels. If return_df is False, returns
                a Series with uplift scores.

        Raises:
            AssertionError: If required features are missing from data.

        Notes:
            When loading from MLflow, the preprocessor is applied
            internally by the model wrapper. For direct object usage,
            the preprocessor is applied explicitly before prediction.
            Calibration is applied only if a calibrator is available.
        """
        self._check_features(data)

        if self._run_id:
            uplift_df = self._model.unwrap_python_model().predict(data)
        else:
            transformed_data = self._preprocessor.transform(
                data[self._features], inplace=False
            )
            uplift_df = self._model.predict(transformed_data)

        if self._calibrator is not None:
            uplift_df["score_raw"] = self._calibrator.predict(
                uplift_df["score_raw"]
            )

        if self._is_multitreatment:
            return uplift_df

        if not return_df:
            uplift = uplift_df["score_raw"]
        else:
            uplift = data.copy()
            if "trmnt_preds" in uplift_df.columns:
                uplift["trmnt_preds"] = uplift_df["trmnt_preds"]
                uplift["ctrl_preds"] = uplift_df["ctrl_preds"]
            uplift["score_raw"] = uplift_df["score_raw"]

        return uplift
