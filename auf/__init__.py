"""AUF - Automated Uplift Framework.

A comprehensive Python library for automated uplift modeling that provides
end-to-end workflow from data preprocessing to model deployment. AUF
integrates best practices from scikit-learn, CatBoost, scikit-uplift and
MLflow into a unified, production-ready pipeline.

Main Components:
    pipeline: High-level pipeline orchestration for automated model
        training, evaluation, and selection.
    models:  wrapper interface for various uplift model types
        including binary and multi-treatment scenarios.
    data: Data preprocessing utilities including missing value handling,
        categorical encoding, and feature engineering.
    metrics: Custom evaluation metrics for uplift modeling with
        bin-weighted averaging and overfit detection.
    ml_flow: MLflow integration utilities for experiment tracking and
        model versioning.
    evaluation: Model evaluation utilities for computing metrics,
        generating summary tables and visualization plots.
    inference: Production-ready inference utilities for model serving.

Key Features:
    * Automatic feature selection and ranking with multiple methods
    * Treatment leak detection and removal
    * Bootstrap-based model evaluation and selection
    * Support for binary and multi-treatment uplift modeling
    * Integrated calibration for uplift scores
    * Comprehensive PDF reporting and visualization
    * MLflow experiment tracking with local artifact mirroring

Modules:
    pipeline: UpliftPipeline, UpliftInference, UpliftCalibrator
    models: AufModel, AufSoloModel, AufXLearner
    data: Preprocessor, MissingValueHandler, CategoryEncoder
    metrics: uplift_at_k, qini_auc_score, bin_weighted_average_uplift
    evaluation: evaluate_model
    ml_flow: MlflowWrapper, save_model, save_metrics

Examples:
    >>> from auf.pipeline import UpliftPipeline
    >>> from auf.models import AufModel
    >>> from auf.data import Preprocessor
    >>> from auf.evaluation import evaluate_model

    >>> # Initialize and run the pipeline
    >>> pipeline = UpliftPipeline(
    ...     task_name_mlflow='uplift_experiment',
    ...     run_name='baseline_model',
    ...     verbosity=2
    ... )
    >>> pipeline.load_sample(df, base_cols_mapper, treatment_groups_mapper)
    >>> preprocessor, model, calibrator = pipeline.run()

    >>> # Evaluate model on test data
    >>> evaluate_model(
    ...     base_cols_mapper=base_cols_mapper,
    ...     treatment_groups_mapper=treatment_groups_mapper,
    ...     data=test_df,
    ...     preprocessor=preprocessor,
    ...     model=model,
    ...     evaluation_types=['metrics_table', 'buckets_qini_plots']
    ... )

    >>> # Make predictions
    >>> from auf.pipeline.inference import UpliftInference
    >>> inference = UpliftInference(preprocessor, model, calibrator)
    >>> uplift_scores = inference.predict(new_data)

Notes:
    AUF requires CatBoost, scikit-learn, scikit-uplift, MLflow, and
    matplotlib as core dependencies. For multi-treatment scenarios,
    additional classifiers from causalml are supported.
"""

from . import (
    constants,
    data,
    feature_rankers,
    metrics,
    ml_flow,
    models,
    pipeline,
    plots,
    training,
)

__version__ = "2.0.0"
__all__ = [
    "constants",
    "data",
    "feature_rankers",
    "metrics",
    "ml_flow",
    "models",
    "pipeline",
    "plots",
    "training",
]
