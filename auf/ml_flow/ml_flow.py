"""MLflow logging utilities and model wrapper implementation.

Implements a pyfunc wrapper for AUF uplift models and utility functions
for logging various artifacts to MLflow tracking server.

Classes:
    MlflowWrapper: MLflow pyfunc wrapper for AUF uplift models.

Functions:
    get_or_create_experiment: Get or create experiment by name.
    generate_run: Create new run within experiment.
    save_json: Log dictionary as JSON artifact.
    save_dataframe_html: Log DataFrame as HTML artifact.
    save_figure: Log matplotlib figure as PNG.
    save_pdf_figures: Log PDF file as artifact.
    save_params_dict: Log parameters dictionary.
    save_metrics: Log metrics dictionary.
    save_model: Log model as pyfunc.
    save_pickle: Log object as pickle artifact.

Examples:
    >>> from auf.ml_flow import MlflowWrapper, save_model
    >>> from auf.data import Preprocessor
    >>> from auf.models import SoloModel
    >>> from catboost import CatBoostClassifier

    >>> preprocessor = Preprocessor()
    >>> model = SoloModel(estimator=CatBoostClassifier())
    >>> wrapper = MlflowWrapper(preprocessor, model)
"""

import json
import os
import pickle
import tempfile
import typing as tp

import mlflow
import pandas as pd
from mlflow.pyfunc import PythonModel

from ..data.preprocessing import Preprocessor
from ..log import get_logger
from ..models import AufModel

logger = get_logger(verbosity=1)


def skip_none_input(func):
    """Implement decorator to skip function execution if any argument is None.

    Args:
        func: Function to wrap.

    Returns:
        Wrapped function that returns None if any argument is None.

    Examples:
        >>> @skip_none_input
        ... def save_data(data, path):
        ...     return f"Saved to {path}"
        >>> save_data(None, "path.txt")  # Returns None
        >>> save_data("data", "path.txt")  # Returns "Saved to path.txt"

    Notes:
        Checks both positional and keyword arguments for None values.
    """

    def wrapper(*args, **kwargs):
        if any(arg is None for arg in args) or any(
            value is None for value in kwargs.values()
        ):
            return None
        return func(*args, **kwargs)

    return wrapper


class MlflowWrapper(PythonModel):
    """MLflow pyfunc wrapper for AUF uplift models.

    Wraps a Preprocessor and AufModel into a single pyfunc-compatible
    model for MLflow logging and serving.

    Attributes:
        _preprocessor (Preprocessor): Data preprocessor instance.
        _auf_model (AufModel): Uplift model instance.

    Examples:
        >>> from auf.ml_flow import MlflowWrapper
        >>> from auf.data import Preprocessor
        >>> from auf.models import SoloModel
        >>> from catboost import CatBoostClassifier

        >>> preprocessor = Preprocessor(num_fill_strategy='mean')
        >>> model = SoloModel(estimator=CatBoostClassifier(verbose=False))
        >>> wrapper = MlflowWrapper(preprocessor, model)

    Notes:
        Implements MLflow pyfunc interface for model serving.
        Preprocessor transform is applied before model inference.
    """

    def __init__(self, preprocessor: Preprocessor, auf_model: AufModel):
        """Initialize MlflowWrapper.

        Args:
            preprocessor: Fitted Preprocessor instance for data transformation.
            auf_model: Fitted AufModel instance for uplift prediction.
        """
        self._preprocessor: Preprocessor = preprocessor
        self._auf_model: AufModel = auf_model

    def load_context(self, context):
        """Load model context from MLflow artifact.

        Args:
            context: MLflow context containing artifact paths.
        """
        pass

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing transformation to input DataFrame.

        Args:
            df: Input DataFrame with raw features.

        Returns:
            DataFrame with preprocessed features.

        Examples:
            >>> processed = wrapper.preprocess(raw_df)
        """
        df[self._auf_model._features] = self._preprocessor.transform(
            df[self._auf_model._features], inplace=False
        )
        return df

    def inference(self, df: pd.DataFrame) -> pd.Series:
        """Run model inference on preprocessed data.

        Args:
            df: Preprocessed DataFrame with features.

        Returns:
            DataFrame with uplift scores and optional treatment/control predictions.

        Examples:
            >>> scores = wrapper.inference(processed_df)
        """
        uplift_df = self._auf_model.predict(df, return_df=True)

        if self._auf_model._is_multitreatment:
            return uplift_df

        uplift = df.copy()

        if "trmnt_preds" in uplift_df.columns:
            uplift["trmnt_preds"] = uplift_df["trmnt_preds"]
            uplift["ctrl_preds"] = uplift_df["ctrl_preds"]

        uplift["score_raw"] = uplift_df["score_raw"]

        return uplift

    def predict(self, model_input):
        """Full prediction pipeline with preprocessing and inference.

        Args:
            model_input: Input DataFrame with raw features.

        Returns:
            DataFrame with uplift predictions and scores.

        Examples:
            >>> from auf.ml_flow import MlflowWrapper
            >>> predictions = wrapper.predict(input_df)

        Notes:
            Implements MLflow pyfunc predict interface.
        """
        processed_df = self.preprocess(model_input)
        score = self.inference(processed_df)
        return score


@skip_none_input
def get_or_create_experiment(experiment_name):
    """Get existing or create new MLflow experiment.

    Args:
        experiment_name: Name of the experiment to get or create.

    Returns:
        Experiment ID as string.

    Examples:
        >>> from auf.ml_flow import get_or_create_experiment
        >>> experiment_id = get_or_create_experiment("uplift_experiment")

    Notes:
        Prints experiment ID to stdout.
    """
    current_experiment = mlflow.get_experiment_by_name(experiment_name)
    if current_experiment is None:
        current_experiment = mlflow.create_experiment(experiment_name)
        logger.info(
            f"Experiment '{experiment_name}' created with ID: {current_experiment}"
        )
    else:
        current_experiment = dict(current_experiment)["experiment_id"]
        logger.info(
            f"Experiment '{experiment_name}' already exists with ID: {current_experiment}"
        )
    return current_experiment


@skip_none_input
def generate_run(
    experiment_name: str,
    experiment_id: str,
    run_name: str,
    description: tp.Optional[str] = None,
):
    """Create a new MLflow run within an experiment.

    Args:
        experiment_name: Name of the experiment.
        experiment_id: ID of the experiment.
        run_name: Name for the new run.
        description: Optional description for the run.

    Returns:
        Run ID as string.

    Examples:
        >>> from auf.ml_flow import generate_run
        >>> run_id = generate_run(
        ...     experiment_name="uplift_experiment",
        ...     experiment_id="1",
        ...     run_name="model_v1",
        ...     description="First model version"
        ... )

    Notes:
        Prints run ID to stdout.
    """
    with mlflow.start_run(
        run_name=run_name, experiment_id=experiment_id, description=description
    ) as run:
        run_id = run.info.run_id
    logger.info(f"RunID {run_id}")
    return run_id


@skip_none_input
def save_json(data, name, artifact_path, run_id):
    """Log dictionary as JSON artifact to MLflow.

    Args:
        data: Dictionary to save as JSON.
        name: Filename without extension.
        artifact_path: Directory path within MLflow artifacts.
        run_id: MLflow run ID to log artifact to.

    Examples:
        >>> from auf.ml_flow import save_json
        >>> save_json(
        ...     data={"feature_count": 10, "model_type": "solo"},
        ...     name="model_info",
        ...     artifact_path="metadata",
        ...     run_id="abc123"
        ... )
    """
    with mlflow.start_run(run_id=run_id) as _:
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_file_path = os.path.join(temp_dir, f"{name}.json")
            with open(custom_file_path, "w") as json_file:
                json.dump(data, json_file, indent=4)
            mlflow.log_artifact(custom_file_path, artifact_path=artifact_path)


@skip_none_input
def save_dataframe_html(df, name, artifact_path, run_id):
    """Log DataFrame as HTML artifact to MLflow.

    Converts DataFrame to HTML with proper formatting and logs as artifact.

    Args:
        df: DataFrame to save as HTML.
        name: Filename without extension.
        artifact_path: Directory path within MLflow artifacts.
        run_id: MLflow run ID to log artifact to.

    Examples:
        >>> from auf.ml_flow import save_dataframe_html
        >>> import pandas as pd
        >>> df = pd.DataFrame({"feature": ["a", "b"], "importance": [0.5, 0.3]})
        >>> save_dataframe_html(df, "feature_importance", "artifacts", "run_123")

    Notes:
        Newline characters in string columns are replaced with HTML <br> tags.
    """
    with mlflow.start_run(run_id=run_id) as _:
        df = df.copy()
        for col in df.select_dtypes(include=["object"]):
            df[col] = df[col].astype(str).str.replace("\n", "<br>", regex=False)
        html_content = (
            """
        <!DOCTYPE html>
        <html>
        <head>
            <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
        </head>
        <body>
        """
            + df.to_html(escape=False)
            + """
        </body>
        </html>
        """
        )

        mlflow.log_text(html_content, artifact_path + f"/{name}.html")


@skip_none_input
def save_figure(fig, name, artifact_path, run_id):
    """Log matplotlib figure as PNG artifact to MLflow.

    Args:
        fig: Matplotlib figure object to save.
        name: Filename without extension.
        artifact_path: Directory path within MLflow artifacts.
        run_id: MLflow run ID to log artifact to.

    Examples:
        >>> from auf.ml_flow import save_figure
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3], [1, 4, 9])
        >>> save_figure(fig, "uplift_curve", "plots", "run_123")

    Notes:
        Figure is saved with bbox_inches='tight' to prevent clipping.
    """
    with mlflow.start_run(run_id=run_id) as _:
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_file_path = os.path.join(temp_dir, f"{name}.png")
            fig.savefig(custom_file_path, bbox_inches="tight")
            mlflow.log_artifact(custom_file_path, artifact_path=artifact_path)


@skip_none_input
def save_pdf_figures(file_path, artifact_path, run_id):
    """Log PDF file as artifact to MLflow.

    Args:
        file_path: Local path to PDF file.
        artifact_path: Directory path within MLflow artifacts.
        run_id: MLflow run ID to log artifact to.

    Examples:
        >>> from auf.ml_flow import save_pdf_figures
        >>> save_pdf_figures(
        ...     file_path="/tmp/report.pdf",
        ...     artifact_path="reports",
        ...     run_id="run_123"
        ... )
    """
    with mlflow.start_run(run_id=run_id) as _:
        mlflow.log_artifact(file_path, artifact_path=artifact_path)


@skip_none_input
def save_params_dict(params, run_id):
    """Log dictionary of parameters to MLflow run.

    Args:
        params: Dictionary of parameter names and values.
        run_id: MLflow run ID to log parameters to.

    Examples:
        >>> from auf.ml_flow import save_params_dict
        >>> save_params_dict(
        ...     params={"learning_rate": 0.1, "iterations": 100, "depth": 6},
        ...     run_id="run_123"
        ... )

    Notes:
        Parameter values must be serializable (str, int, float, or bool).
    """
    with mlflow.start_run(run_id=run_id) as _:
        mlflow.log_params(params)


@skip_none_input
def save_metrics(metrics, run_id):
    """Log dictionary of metrics to MLflow run.

    Args:
        metrics: Dictionary of metric names and values.
        run_id: MLflow run ID to log metrics to.

    Examples:
        >>> from auf.ml_flow import save_metrics
        >>> save_metrics(
        ...     metrics={"qini_auc": 0.15, "uplift_at_30": 0.08, "accuracy": 0.72},
        ...     run_id="run_123"
        ... )

    Notes:
        Metric values must be numeric (int or float).
    """
    with mlflow.start_run(run_id=run_id) as _:
        mlflow.log_metrics(metrics)


@skip_none_input
def save_model(model, artifact_path, run_id, experiment_name):
    """Log MlflowWrapper model as pyfunc to MLflow.

    Args:
        model: MlflowWrapper instance to log.
        artifact_path: Directory path within MLflow artifacts.
        run_id: MLflow run ID to log model to.
        experiment_name: Name for model registration in MLflow registry.

    Examples:
        >>> from auf.ml_flow import MlflowWrapper, save_model
        >>> from auf.data import Preprocessor
        >>> from auf.models import SoloModel
        >>> from catboost import CatBoostClassifier

        >>> preprocessor = Preprocessor()
        >>> auf_model = SoloModel(estimator=CatBoostClassifier())
        >>> wrapper = MlflowWrapper(preprocessor, auf_model)
        >>> save_model(wrapper, "model", "run_123", "uplift_experiment")

    Notes:
        Model is logged as MLflow pyfunc and registered in model registry.
    """
    with mlflow.start_run(run_id=run_id) as _:
        mlflow.pyfunc.log_model(
            python_model=model,
            artifact_path=artifact_path,
            registered_model_name=experiment_name,
        )


@skip_none_input
def save_pickle(obj, name, artifact_path, run_id):
    """Log arbitrary Python object as pickle artifact to MLflow.

    Args:
        obj: Python object to pickle and log.
        name: Filename (with or without .pkl extension).
        artifact_path: Directory path within MLflow artifacts.
        run_id: MLflow run ID to log artifact to.

    Examples:
        >>> from auf.ml_flow import save_pickle
        >>> feature_list = ["age", "income", "city"]
        >>> save_pickle(feature_list, "features", "artifacts", "run_123")

    Notes:
        .pkl extension is added automatically if not present.
        Uses highest pickle protocol for serialization.
    """
    with mlflow.start_run(run_id=run_id) as _:
        with tempfile.TemporaryDirectory() as temp_dir:
            name = name[:-4] if name.endswith(".pkl") else name
            path = os.path.join(temp_dir, f"{name}.pkl")
            with open(path, "wb") as file:
                pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)
            mlflow.log_artifact(
                path,
                artifact_path=artifact_path,
            )
