"""Optuna-based hyperparameter optimization.

Provides OptunaOptimizer class for finding optimal hyperparameters
for uplift models using various metrics.

Classes:
    OptunaOptimizer: Hyperparameter optimizer using Optuna with TPE sampler.

Examples:
    >>> from auf.training import OptunaOptimizer
    >>> from sklift.metrics import qini_auc_score
    >>> from auf.models import AufSoloModel

    >>> optimizer = OptunaOptimizer(
    ...     df_train=train,
    ...     df_valid=valid,
    ...     metric=qini_auc_score,
    ...     treatment_col='treatment',
    ...     target_col='target',
    ...     overfit_metric=None,
    ...     training_mode='light'
    ... )
    >>> best_model = optimizer.find_best_params(
    ...     model_class=AufSoloModel,
    ...     features=['f1', 'f2'],
    ...     timeout=60
    ... )

Notes:
    Supports both binary and multi-treatment scenarios.
    Uses TPESampler for optimization with fixed random seed.
    Supported model classes: CatBoostClassifier, SoloModel, TwoModels,
    AufXLearner, AufTreeClassifier, AufRandomForestClassifier,
    BaseSClassifier, BaseTClassifier, BaseXClassifier,
    UpliftTreeClassifier, UpliftRandomForestClassifier.
"""

import typing as tp

import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from ..constants import RANDOM_STATE
from ..models import AufModel
from .model_generation import generate_model, generate_multitreatment_model


class OptunaOptimizer:
    """Hyperparameter optimizer using Optuna.

    Finds optimal hyperparameters for a given model class by
    maximizing a user-supplied metric.

    Attributes:
        df_trn (pd.DataFrame): Training dataset.
        df_val (pd.DataFrame): Validation dataset.
        metric: Metric function to optimize.
        treatment_col (str): Treatment column name.
        target_col (str): Target column name.
        overfit_metric: Optional overfit penalty metric.
        training_mode (str): Search complexity mode.

    Examples:
        >>> from auf.training import OptunaOptimizer
        >>> from sklift.metrics import qini_auc_score
        >>>
        >>> optimizer = OptunaOptimizer(
        ...     df_train=train,
        ...     df_valid=valid,
        ...     metric=qini_auc_score,
        ...     treatment_col='treatment',
        ...     target_col='target',
        ...     overfit_metric=None,
        ...     training_mode='light'
        ... )
        >>> best_model = optimizer.find_best_params(
        ...     model_class=AufSoloModel,
        ...     features=['f1', 'f2'],
        ...     timeout=60
        ... )

    Notes:
        Supports both binary and multi-treatment scenarios.
        Uses TPESampler for optimization with fixed random seed.
    """

    def __init__(
        self,
        df_train: pd.DataFrame,
        df_valid: pd.DataFrame,
        metric: object,
        treatment_col: str,
        target_col: str,
        overfit_metric: object,
        training_mode: str,
    ):
        """Initialize OptunaOptimizer.

        Args:
            df_train: Training dataset.
            df_valid: Validation dataset.
            metric: Metric function to maximize.
            treatment_col: Name of treatment column.
            target_col: Name of target column.
            overfit_metric: Optional function to compute overfit penalty.
            training_mode: Search mode ('light', 'medium', 'hard').
        """
        self.df_trn = df_train
        self.df_val = df_valid
        self.metric = metric
        self.treatment_col = treatment_col
        self.target_col = target_col
        self.overfit_metric = overfit_metric
        self.treatment_groups = None
        self.training_mode = training_mode

    def multitreatment_objective(self, trial):
        """Objective function for multi-treatment optimization.

        Args:
            trial: Optuna trial object.

        Returns:
            Metric value to optimize.

        Notes:
            Computes weighted average metric across treatment groups.
        """
        # def multi_metric(y_true, treatment, uplift):
        #     control_name = 0 if 0 in self.treatment_groups else "control"
        #     groups_metrics = []
        #     groups_weights = []
        #     for group_name in self.treatment_groups:
        #         if group_name == control_name:
        #             continue
        #         mask = treatment.isin([group_name, control_name])
        #         y, t, u = y_true[mask], treatment[mask], uplift[group_name][mask]
        #         value = self.metric(y_true=y, uplift=u, treatment=(t != control_name).astype(int))
        #         groups_metrics.append(value)
        #         groups_weights.append(mask.sum())
        #     return np.average(groups_metrics, weights=groups_weights)

        def multi_metric(y_true, treatment, uplift):
            control_name = 0 if 0 in self.treatment_groups else "control"
            return self.metric(
                y_true=y_true,
                uplift=uplift.max(axis=1),
                treatment=(treatment != control_name).astype(int),
            )

        model = generate_multitreatment_model(
            trial, self.model_class, training_mode=self.training_mode
        )

        auf_model = AufModel(
            model=model,
            model_name=self.model_class.__name__,
            features=self.features,
            treatment_groups=self.treatment_groups,
        )

        auf_model.fit(
            X=self.df_trn[self.features],
            y=self.df_trn[self.target_col],
            treatment=self.df_trn[self.treatment_col],
        )

        uplift_val = auf_model.predict(self.df_val[self.features])

        metric_val = multi_metric(
            y_true=self.df_val[self.target_col],
            uplift=uplift_val,
            treatment=self.df_val[self.treatment_col],
        )

        if self.overfit_metric is not None:
            uplift_trn = auf_model.predict(self.df_trn[self.features])

            metric_trn = multi_metric(
                y_true=self.df_trn[self.target_col],
                uplift=uplift_trn,
                treatment=self.df_trn[self.treatment_col],
            )

            return self.overfit_metric(metric_val, metric_trn)

        return metric_val

    def objective(self, trial):
        """Objective function for binary treatment optimization.

        Args:
            trial: Optuna trial object.

        Returns:
            Metric value to optimize.
        """
        model = generate_model(
            trial, self.model_class, training_mode=self.training_mode
        )

        auf_model = AufModel(
            model=model,
            model_name=self.model_class.__name__,
            features=self.features,
            uplift_prediction_type="abs",  # train to predict absolute uplift only
        )

        auf_model.fit(
            X=self.df_trn[self.features],
            y=self.df_trn[self.target_col],
            treatment=self.df_trn[self.treatment_col],
        )

        if self.model_class.__name__ != "CatBoostClassifier":
            uplift_val = auf_model.predict(
                self.df_val[self.features], return_df=False
            )

            metric_val = self.metric(
                y_true=self.df_val[self.target_col],
                uplift=uplift_val,
                treatment=self.df_val[self.treatment_col],
            )

            if self.overfit_metric is not None:
                uplift_trn = auf_model.predict(
                    self.df_trn[self.features], return_df=False
                )

                metric_trn = self.metric(
                    y_true=self.df_trn[self.target_col],
                    uplift=uplift_trn,
                    treatment=self.df_trn[self.treatment_col],
                )

        else:
            score_val = auf_model.predict(
                self.df_val[self.features], return_df=False
            )

            metric_val = roc_auc_score(
                y_true=self.df_val[self.target_col],
                y_score=score_val,
            )

            if self.overfit_metric is not None:
                score_trn = auf_model.predict(
                    self.df_trn[self.features], return_df=False
                )

                metric_trn = roc_auc_score(
                    y_true=self.df_trn[self.target_col],
                    y_score=score_trn,
                )

        if self.overfit_metric is not None:
            return self.overfit_metric(metric_val, metric_trn)

        return metric_val

    def find_best_params(
        self,
        model_class: Pipeline,
        features: tp.List[str],
        timeout: int,
        treatment_groups: tp.Optional[tp.List[str]] = None,
    ):
        """Find optimal hyperparameters for model class.

        Args:
            model_class: Model class to optimize.
            features: List of feature names.
            timeout: Maximum optimization time in seconds.
            treatment_groups: List of treatment groups for multi-treatment.
                Defaults to None.

        Returns:
            Model instance with optimal hyperparameters.

        Raises:
            ValueError: If model_class is not supported.

        Examples:
            >>> best_model = optimizer.find_best_params(
            ...     model_class=AufSoloModel,
            ...     features=['f1', 'f2'],
            ...     timeout=60
            ... )

        Notes:
            Supported model classes: CatBoostClassifier, SoloModel,
            TwoModels, AufXLearner, AufTreeClassifier,
            AufRandomForestClassifier, BaseSClassifier, BaseTClassifier,
            BaseXClassifier, UpliftTreeClassifier, UpliftRandomForestClassifier.
        """
        self.treatment_groups = treatment_groups

        if model_class.__name__ not in [
            "CatBoostClassifier",
            "SoloModel",
            "TwoModels",
            "AufXLearner",
            "AufTreeClassifier",
            "AufRandomForestClassifier",
            "BaseSClassifier",
            "BaseTClassifier",
            "BaseXClassifier",
            "UpliftTreeClassifier",
            "UpliftRandomForestClassifier",
        ]:
            raise ValueError(f"Unsupported model class: {model_class.__name__}")

        self.model_class = model_class
        self.features = features
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        if self.treatment_groups:
            study.optimize(
                self.multitreatment_objective, timeout=timeout, n_jobs=1
            )
            model = generate_multitreatment_model(
                None, model_class, params=study.best_params
            )
        else:
            study.optimize(self.objective, timeout=timeout, n_jobs=1)
            model = generate_model(None, model_class, params=study.best_params)
        return model
