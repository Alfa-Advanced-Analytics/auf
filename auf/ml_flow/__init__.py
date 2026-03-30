"""MLflow integration utilities for AUF library.

Provides helper functions and model wrappers for logging uplift models,
artifacts, and metrics to MLflow tracking server.

Functions:
    get_or_create_experiment: Get or create MLflow experiment by name.
    generate_run: Create a new MLflow run within an experiment.
    save_dataframe_html: Log DataFrame as HTML artifact.
    save_figure: Log matplotlib figure as PNG artifact.
    save_json: Log dictionary as JSON artifact.
    save_metrics: Log dictionary of metrics to MLflow.
    save_model: Log AUF model as MLflow pyfunc model.
    save_model_preprocess: Log model with preprocessor artifacts.
    save_params_dict: Log dictionary of parameters to MLflow.
    save_pdf_figures: Log PDF file as artifact.
    save_pickle: Log arbitrary Python object as pickle artifact.

Classes:
    MlflowWrapper: MLflow pyfunc wrapper for AUF uplift models.

Examples:
    >>> from auf.ml_flow import get_or_create_experiment, generate_run, save_metrics
    >>> from auf.ml_flow import MlflowWrapper, save_model

    >>> experiment_id = get_or_create_experiment("uplift_experiment")
    >>> run_id = generate_run("uplift_experiment", experiment_id, "model_v1")
    >>> save_metrics({"qini_auc": 0.15, "uplift@30%": 0.08}, run_id)

Notes:
    All logging functions accept None arguments and skip logging gracefully.
    MLflow tracking URI should be configured before using these utilities.
"""

from .ml_flow import (
    MlflowWrapper,
    generate_run,
    get_or_create_experiment,
    save_dataframe_html,
    save_figure,
    save_json,
    save_metrics,
    save_model,
    save_params_dict,
    save_pdf_figures,
    save_pickle,
)

__all__ = [
    "MlflowWrapper",
    "generate_run",
    "get_or_create_experiment",
    "save_dataframe_html",
    "save_figure",
    "save_json",
    "save_metrics",
    "save_model",
    "save_params_dict",
    "save_pdf_figures",
    "save_pickle",
]
