"""End-to-end uplift modeling pipeline.

Provides a high‑level, end‑to‑end workflow for uplift (treatment‑effect)
modeling.  The pipeline orchestrates data loading, validation, feature
engineering, model training (with optional Optuna hyper‑parameter search),
evaluation, selection of the best model and generation of a comprehensive
PDF report.  All artefacts (pre‑processor, trained models, metrics,
figures) are stored in an MLflow experiment.

Classes:
    UpliftPipeline: Main pipeline class that manages the complete
        uplift modeling lifecycle from data ingestion to model selection.


Examples:
    >>> from auf.pipeline import UpliftPipeline
    >>> pipeline = UpliftPipeline(
    ...     task_name_mlflow='uplift_experiment',
    ...     run_name='run_01',
    ...     verbosity=2,
    ... )
    >>> pipeline.load_sample(df, base_cols_mapper, treatment_groups_mapper)
    >>> pipeline.run()

Notes:
    The pipeline works in two modes: binary‑treatment (control vs single
    treatment) and multi‑treatment (control + several treatments). The mode
    is detected automatically from the supplied treatment_groups_mapper.
    All intermediate artefacts are saved under the run's mlflow_artifacts
    directory. Logging is performed both to the console and to MLflow.
"""

import itertools
import logging
import os
import typing as tp
from dataclasses import dataclass
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
from catboost import CatBoostClassifier
from causalml.inference.meta import (
    BaseSClassifier,
    BaseTClassifier,
    BaseXClassifier,
)
from causalml.inference.tree import (
    UpliftRandomForestClassifier,
    UpliftTreeClassifier,
)
from IPython.display import Markdown, display
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import roc_auc_score
from sklift.metrics import qini_auc_score, uplift_by_percentile
from sklift.models import SoloModel, TwoModels
from sklift.viz import plot_qini_curve
from tqdm.contrib.logging import logging_redirect_tqdm

from ..constants import BOOTSTRAP_REPEATS, METRICS, RANDOM_STATE
from ..data.checks import (
    check_bernoulli_equal_means,
    check_correlations,
    check_leaks_v2,
    check_nans,
    check_train_val_test_split,
    process_too_much_categories,
)
from ..data.preprocessing import Preprocessor
from ..data.split import train_val_test_split
from ..feature_rankers import (
    FilterRanker,
    ImportanceRanker,
    PermutationRanker,
    StepwiseRanker,
    StraightforwardRanker,
)
from ..log import ManualProgressBar, get_logger
from ..metrics import overfit_metric_minus_metric_delta
from ..ml_flow import (
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
from ..models import (
    AufModel,
    AufRandomForestClassifier,
    AufTreeClassifier,
    AufXLearner,
)
from ..plots import (
    plot_cumulative_target_ratio,
    plot_portrait_tree,
    plot_uplift_by_feature_bins,
    plot_uplift_by_percentile,
    plot_uplift_top_vs_bottom,
)
from ..training.fitting import fit_model, generate_model_from_classes
from ..training.gridsearch import OptunaOptimizer
from .calibration import UpliftCalibrator


@dataclass
class ModelResult:
    """
    "Data class for model results saving.
    """

    auf_model: AufModel
    median_test_metric: float


class UpliftPipeline:
    """Automatic uplift‑modeling pipeline.

    Manages the complete life‑cycle of an uplift‑modeling project,
    from data ingestion to model selection and reporting.  All
    intermediate results are stored in the instance attributes and
    logged to an MLflow run.

    Attributes:
        _df: The loaded and pre‑processed dataset.
        _feature_cols: List of features that are currently
            usable for modeling.
        _base_cols_mapper: Mapping of unified column names (id,
            treatment, target, segm) to real column names in the source
            data.
        _treatment_groups_mapper: Mapping of user‑defined treatment
            identifiers to internal integer codes.
        _removed_features: Dictionary that records
            why each feature was removed.
        _ranked_candidates: Ranked feature lists produced by each ranker.
        _train_results: Trained models together with their bootstrap‑estimated test metrics.
        _use_multitreatment: True if the problem contains more than
            two treatment groups.
        _preprocessor: Fitted preprocessing object.
        _calibrator: Calibrator fitted on the validation
            set (binary‑treatment only).

    Examples:
        >>> from auf.pipeline import UpliftPipeline
        >>> pipeline = UpliftPipeline(
        ...     task_name_mlflow='uplift_experiment',
        ...     run_name='run_01',
        ...     verbosity=2,
        ... )
        >>> pipeline.load_sample(df, base_cols_mapper, treatment_groups_mapper)
        >>> pipeline.run()

    Notes:
        The pipeline automatically detects binary vs multi‑treatment mode.
    """

    _base_cols = ["id", "treatment", "target", "segm"]

    def __init__(
        self,
        print_doc: bool = True,
        task_name_mlflow: str = None,
        run_id: str = None,
        run_name: str = None,
        run_description: str = "auf",
        logger: logging.Logger = None,
        verbosity: int = 1,
    ):
        """Initialize the pipeline and create (or reuse) an MLflow run.

        Args:
            print_doc: If True and the code is executed inside a
                Jupyter notebook, the class docstring is displayed as
                Markdown.
            task_name_mlflow: Name of the MLflow experiment.
                If None no experiment is created.
            run_id: Existing MLflow run identifier. If None
                a new run is started (only when task_name_mlflow is
                provided).
            run_name: Human‑readable name for the run. If
                None a timestamp is used.
            run_description: Description stored in MLflow for the
                run.
            logger: Custom logger instance. If
                None a logger is created with the specified verbosity.
            verbosity: Logging level (0=quiet, 1=default, 2=verbose).

        Returns:
            None
        """
        self._df: pd.DataFrame = None
        self._feature_cols: tp.List[str] = None
        self._base_cols_mapper: tp.Dict[str, str] = None
        self._treatment_groups_mapper: tp.Dict[tp.Any, int] = None
        self._treatment_groups_mapper_inv: tp.Dict[tp.Any, int] = None
        self._treatment_groups: tp.List[tp.Any] = None
        self._use_multitreatment: bool = False

        self.verbosity = verbosity
        self.logger: logging.Logger = (
            logger if logger else get_logger(verbosity)
        )

        self._experiment_name = task_name_mlflow
        self._experiment_id = get_or_create_experiment(self._experiment_name)
        self._run_name = (
            run_name
            if run_name is not None
            else datetime.now().strftime("%Y-%m-%d %H:%M")
        )
        self._run_description = run_description
        if run_id is None and task_name_mlflow is not None:
            self._run_id = generate_run(
                self._experiment_name,
                self._experiment_id,
                self._run_name,
                self._run_description,
            )
        else:
            self._run_id = run_id

        self._preprocessor: Preprocessor = None
        self._calibrator: UpliftCalibrator = None

        self._removed_features: tp.Dict[str, str] = {}
        self._feature_cols_treatment_leaks_roc_aucs: tp.List[str] = None
        self._feature_cols_treatment_roc_aucs: tp.List[
            tp.Tuple[str, float]
        ] = None

        self._ranked_candidates: tp.Dict[str, tp.List[str]] = None

        self._train_metric: tp.Callable[
            [np.array, np.array, np.array], float
        ] = None
        self._train_results: tp.Dict[
            str, tp.Dict[str, tp.List[ModelResult]]
        ] = dict()

        self._use_default_run: bool = False

        if print_doc:
            try:
                shell = get_ipython().__class__.__name__
                if shell == "ZMQInteractiveShell":
                    display(Markdown(self.__doc__))

            except NameError:
                pass

    def _check_base_column_names(self, df, base_cols_mapper):
        for col in self._base_cols:
            if col not in base_cols_mapper:
                raise AssertionError(
                    f"Specify None or value for base_cols_mapper['{col}']"
                )

            if base_cols_mapper[col] is None:
                if col in ["id", "treatment", "target"]:
                    raise AssertionError(
                        f"Value of base_cols_mapper['{col}'] must not be None"
                    )
                continue

            if base_cols_mapper[col] not in df.columns:
                raise AssertionError(
                    f"Value of base_cols_mapper['{col}'] must be a dataframe column"
                )

        self._base_cols_mapper = base_cols_mapper

    def _check_base_column_values(self, df, treatment_groups_mapper):
        target_col = self._base_cols_mapper["target"]
        treatment_col = self._base_cols_mapper["treatment"]
        segm_col = self._base_cols_mapper["segm"]

        for base_col, col in self._base_cols_mapper.items():
            if col is not None:
                if df[col].isna().any():
                    raise AssertionError(
                        f"'{col}' column must not contain missed values"
                    )

        if set(treatment_groups_mapper.keys()) != set(
            np.unique(df[treatment_col])
        ):
            raise AssertionError(
                "'treatment_groups_mapper' must contain all treatment groups as keys and only"
            )

        self._treatment_groups_mapper = treatment_groups_mapper
        self._treatment_groups_mapper_inv = {
            v: k for k, v in self._treatment_groups_mapper.items()
        }
        self._treatment_groups = list(treatment_groups_mapper.keys())

        self._use_multitreatment = len(self._treatment_groups) > 2

        if self._use_multitreatment:
            if "control" not in treatment_groups_mapper:
                raise AssertionError(
                    "When using multitreatment 'control' group key must appear in treatment_groups_mapper"
                )
        else:
            if set(treatment_groups_mapper.values()) != set([0, 1]):
                raise AssertionError(
                    "In binary treatment case 'treatment_groups_mapper' must contain both 0 and 1 and only"
                )

        if set(np.unique(df[target_col])) != set([0, 1]):
            raise AssertionError(
                f"'{target_col}' column must contain both 0 and 1 and only"
            )

        if segm_col is not None and set(np.unique(df[segm_col])) != {
            "train",
            "val",
            "test",
        }:
            raise AssertionError(
                f"'{segm_col}' column must contain all of ['train', 'val', 'test']"
            )

    def _check_take_rate_differ(self, df):
        target_col = self._base_cols_mapper["target"]
        treatment_col = self._base_cols_mapper["treatment"]

        target_ratio_diff = None

        for treatment_group in self._treatment_groups_mapper.keys():
            if self._use_multitreatment:
                if treatment_group == "control":
                    continue

                control_group = "control"

                data = df.loc[
                    df[treatment_col].isin([treatment_group, "control"]),
                    [target_col, treatment_col],
                ]

                treatment_flag = data[treatment_col].map(
                    {"control": 0, treatment_group: 1}
                )
            else:
                data = df
                control_group = self._treatment_groups_mapper_inv[0]
                treatment_group = self._treatment_groups_mapper_inv[1]
                treatment_flag = df[treatment_col].map(
                    self._treatment_groups_mapper
                )

            target_treatment = data.loc[treatment_flag == 1, target_col]
            target_control = data.loc[treatment_flag == 0, target_col]

            result = check_bernoulli_equal_means(
                target_treatment, target_control, alpha=0.05
            )

            self.logger.info(
                f"Difference in target rates for '{treatment_group}' and '{control_group}' groups:\n"
                f"{'':<4}pvalue = {result['pvalue']:.3f}\n"
                f"{'':<4}treatment target rate : {target_treatment.mean():.3f}\n"
                f"{'':<4}control target rate   : {target_control.mean():.3f}"
            )

            if result["equals"]:
                self.logger.warning(
                    "Target rates in control and treatment groups must have statistically significant difference for uplift modeling\n"
                )

            tmp_target_ratio_diff = pd.DataFrame(
                {
                    "treatment group": treatment_group,
                    "treatment target rate": round(target_treatment.mean(), 3),
                    "control target rate": round(target_control.mean(), 3),
                    "pvalue": round(result["pvalue"], 3),
                    "equals": [result["equals"]],
                }
            )

            if target_ratio_diff is None:
                target_ratio_diff = tmp_target_ratio_diff
            else:
                target_ratio_diff = pd.concat(
                    [target_ratio_diff, tmp_target_ratio_diff], axis=0
                )

            if not self._use_multitreatment:
                break

        display(target_ratio_diff)

        save_dataframe_html(
            target_ratio_diff,
            "3_take_rate_diff",
            "1_data",
            self._run_id,
        )

    def show_take_rate_info(self):
        """
        Show tables with take rate statistics.
        """
        target_col = self._base_cols_mapper["target"]
        treatment_col = self._base_cols_mapper["treatment"]
        segm_col = self._base_cols_mapper["segm"]

        info = self._df.groupby([segm_col, treatment_col])[target_col].agg(
            ["mean", "sum", "count"]
        )
        info.columns = ["target_mean", "target_sum", "target_count"]
        save_dataframe_html(info, "3_take_rate_info", "1_data", self._run_id)
        display(info)

    def _check_train_val_test_split(self, df):
        segm_col = self._base_cols_mapper["segm"]
        target_col = self._base_cols_mapper["target"]
        treatment_col = self._base_cols_mapper["treatment"]

        if self._use_multitreatment:
            for treatment_group in self._treatment_groups_mapper.keys():
                if treatment_group == "control":
                    continue

                data = df.loc[
                    df[treatment_col].isin([treatment_group, "control"])
                ]

                treatment_groups_mapper = {"control": 0, treatment_group: 1}

                check_train_val_test_split(
                    data,
                    segm_col,
                    target_col,
                    treatment_col,
                    treatment_groups_mapper,
                )
        else:
            check_train_val_test_split(
                df,
                segm_col,
                target_col,
                treatment_col,
                self._treatment_groups_mapper,
            )

    def _default_train_val_test_split(self, df):
        df_train_idx, df_val_idx, df_test_idx = train_val_test_split(
            df,
            size_ratios=[0.6, 0.2, 0.2],
            stratify_cols=[
                self._base_cols_mapper["target"],
                self._base_cols_mapper["treatment"],
            ],
        )

        df["segm"] = "train"
        self._base_cols_mapper["segm"] = "segm"
        df.loc[df.index.isin(df_val_idx), "segm"] = "val"
        df.loc[df.index.isin(df_test_idx), "segm"] = "test"

    def _get_available_features(self):
        removed_features = list(
            itertools.chain(*self._removed_features.values())
        )
        removed_features += list(
            itertools.chain(
                self._base_cols_mapper.keys(), self._base_cols_mapper.values()
            )
        )
        return [f for f in self._feature_cols if f not in removed_features]

    def _format_feature_name_by_limit(self, feature_name: str, limit: int = 40):
        words = feature_name.split()
        lines = []
        current_line = ""

        for word in words:
            if current_line:
                if len(current_line) + 1 + len(word) <= limit:
                    current_line += " " + word
                else:
                    lines.append(current_line)
                    current_line = word
            else:
                current_line = word

        if current_line:
            lines.append(current_line)

        return "\n".join(lines)

    def load_sample(
        self,
        df: pd.DataFrame,
        base_cols_mapper: tp.Dict[str, str] = {
            "id": "id",
            "treatment": "treatment",
            "target": "target",
            "segm": "segm",
        },
        treatment_groups_mapper: tp.Dict[tp.Any, int] = {0: 0, 1: 1},
        feature_names: tp.Optional[tp.Dict[str, str]] = None,
    ):
        """Load and check a dataset and run preprocessing.

        Args:
            df: Raw data.
            base_cols_mapper: Mapping from unified column names to
                actual column names in df.
            treatment_groups_mapper: Mapping from user‑defined
                treatment identifiers to internal integer codes.
            feature_names: Optional dictionary {feature:
                pretty_name} used for labeling plots. If None feature
                names are left unchanged.

        Returns:
            None

        Raises:
            AssertionError: If required columns are missing or mappings
                are invalid.
        """
        assert isinstance(
            df, pd.DataFrame
        ), "Parameter 'df' must be a pandas dataframe"
        self._run_id = (
            self._run_id
            if self._run_id is not None
            else generate_run(
                self._experiment_name,
                self._experiment_id,
                self._run_name,
                self._run_description,
            )
        )

        mappers = {
            "base_cols_mapper": base_cols_mapper,
            "treatment_groups_mapper": treatment_groups_mapper,
        }
        save_json(mappers, "1_mappings", "1_data", self._run_id)

        assert (
            len(set(df.columns)) == df.shape[1]
        ), "Dataframe df must have unique column names"
        self._check_base_column_names(df, base_cols_mapper)
        self._check_base_column_values(df, treatment_groups_mapper)
        self._check_take_rate_differ(df)

        if base_cols_mapper["segm"] is not None:
            self._check_train_val_test_split(df)
        else:
            self._default_train_val_test_split(df)

        self._train_val_test_map = df[
            [self._base_cols_mapper["id"], self._base_cols_mapper["segm"]]
        ]

        self._df = df
        self._feature_cols = df.columns.tolist()

        if feature_names is not None:
            feature_names = {
                feat: self._format_feature_name_by_limit(name)
                for feat, name in feature_names.items()
            }
        else:
            feature_names = dict()

        self._feature_names = feature_names

        self._feature_cols = self._get_available_features()
        self._removed_features["all values missed"] = [
            f for f in self._feature_cols if self._df[f].isna().all()
        ]
        self._feature_cols = self._get_available_features()
        self._removed_features["only 1 unique value"] = [
            f for f in self._feature_cols if self._df[f].nunique() == 1
        ]
        self._feature_cols = self._get_available_features()

        self.logger.info("Preprocess features")
        self._preprocessor = self.get_preprocessor()
        self._df = self._preprocessor.transform(self._df)

        self.logger.info("Sample was succesfully loaded!")
        self.show_take_rate_info()

    def check_treatment_leaks(
        self,
        max_val_roc_auc_treatment: float = 0.55,
        early_stopping: int = None,
        check_only_available_features: bool = True,
    ):
        """Detect features with significant treatment leak.

        Args:
            max_val_roc_auc: Upper bound for acceptable ROC‑AUC
                on the validation set.
            early_stopping: Number of iterations without
                improvement after which training stops.
            check_only_available_features: If True only features
                that have survived previous filtering steps are examined.

        Returns:
            None
        """
        if not self._use_default_run:
            self.logger.info("Check feature leaks for treatment column.")

        if check_only_available_features:
            all_feature_cols = self._get_available_features()
        else:
            all_feature_cols = self._feature_cols

        if not self._use_default_run:
            self.logger.info(f"{len(all_feature_cols):7} features in total")

        if not self._use_default_run:
            self.logger.info("Analyze potential leaks of treatment")
        (
            treatment_leaks_roc_aucs,
            treatment_not_leaks,
            treatment_roc_aucs,
        ) = check_leaks_v2(
            self._df,
            self._base_cols_mapper,
            all_feature_cols,
            col_to_check="treatment",
            alpha=0.05,
            max_val_roc_auc=max_val_roc_auc_treatment,
            early_stopping=early_stopping,
        )

        treatment_leaks = pd.DataFrame(
            treatment_leaks_roc_aucs, columns=["feature", "roc_auc"]
        )
        save_dataframe_html(
            treatment_leaks,
            "1_treatment_leaks",
            "2_feature_selection",
            self._run_id,
        )

        self._removed_features["treatment leaks"] = [
            f for f, roc_auc in treatment_leaks_roc_aucs
        ]

        if not self._use_default_run:
            if len(treatment_leaks_roc_aucs) > 0:
                self.logger.info(
                    f"TOP leaking features ({len(treatment_leaks_roc_aucs)} found, but at most 5 are printed):"
                )
                for f, roc_auc in treatment_leaks_roc_aucs[:5]:
                    self.logger.info(f"   {f:20} --> ROC-AUC = {roc_auc:.3f}")
            else:
                self.logger.info("No leaking features were detected.")

        self._feature_cols_treatment_leaks_roc_aucs = (
            treatment_leaks_roc_aucs.copy()
        )
        self._feature_cols_treatment_roc_aucs = treatment_roc_aucs.copy()

    def show_selected_features_stat(self):
        """
        Show number of current numerical and categorical features.
        """
        selected_features = self._get_available_features()
        cat_feats = [
            f for f in selected_features if self._df[f].dtype == "object"
        ]
        num_feats = [f for f in selected_features if f not in cat_feats]
        self.logger.info(
            f"Currently selected for modeling: {len(selected_features)}"
        )
        self.logger.info(f"{'':<4}{len(num_feats):5} numerical features")
        self.logger.info(f"{'':<4}{len(cat_feats):5} categorical features")

    def _preselect_features_candidates_binary_treatment(
        self, n_features: int = 300, method: str = "filter"
    ):
        """Save a long-list of features which have potential benefit.

        Note:
            - feature preselction needs to be very fast and simple
            - features that have potential for the task may be rarely filled or be categorical
            - save them in any case and work with only well filled numerical features here

        Args:
            method (str, optional): method for feature selection (by importance of SoloModel with
                CatBoostClassifier as base model of bin-based filter method). Default is "filter"
            n_features (int, optional): number of feature to use in further pipeline steps. Default is 200

        Raises:
            ValueError: if method is not in ["importance", "filter"]

        Returns: None
        """
        target_col = self._base_cols_mapper["target"]
        treatment_col = self._base_cols_mapper["treatment"]

        treatment_map = self._treatment_groups_mapper
        treatment_map_inv = {
            v: k for k, v in self._treatment_groups_mapper.items()
        }

        self._df[treatment_col] = self._df[treatment_col].map(treatment_map)

        try:
            # repeated runs of preselection need to be independent
            if "preselection" in self._removed_features:
                del self._removed_features["preselection"]

            ctb_params = {
                "iterations": 200,
                "max_depth": 4,
                "learning_rate": 0.05,
                "cat_features": [
                    f
                    for f in self._get_available_features()
                    if self._df[f].dtype == "object"
                ],
            }

            model_params = {
                "estimator": CatBoostClassifier(
                    **ctb_params, silent=True, random_state=RANDOM_STATE
                ),
                "method": "dummy",
            }

            if method == "importance":
                ranker = ImportanceRanker(SoloModel, model_params, "at_once")
                ranked_features, ranked_importances = ranker.run(
                    self._df,
                    self._get_available_features(),
                    target_col,
                    treatment_col,
                )

                zero_importance_features = [
                    f
                    for f, imp in zip(ranked_features, ranked_importances)
                    if not imp > 0 and f not in ["treatment", treatment_col]
                ]

                if len(zero_importance_features) > 0:
                    ctb_params = {
                        "iterations": 200,
                        "max_depth": 4,
                        "learning_rate": 0.05,
                        "cat_features": [
                            f
                            for f in zero_importance_features
                            if self._df[f].dtype == "object"
                        ],
                    }

                    model_params = {
                        "estimator": CatBoostClassifier(
                            **ctb_params, silent=True, random_state=RANDOM_STATE
                        ),
                        "method": "dummy",
                    }

                    ranker = ImportanceRanker(
                        SoloModel, model_params, "at_once"
                    )
                    zero_ranked_features, zero_ranked_importances = ranker.run(
                        self._df,
                        zero_importance_features,
                        target_col,
                        treatment_col,
                    )

                    features_imps = zip(ranked_features, ranked_importances)
                    features_imps = [p for p in features_imps if p[1] > 0]
                    ranked_features, ranked_importances = map(
                        list, zip(*features_imps)
                    )

                    ranked_features += zero_ranked_features
                    ranked_importances += zero_ranked_importances

            elif method == "filter":
                filled_num_features = [
                    f
                    for f in self._get_available_features()
                    if self._df[f].dtype != "object"
                    and self._df[f].isna().mean() < 0.9
                ]

                ranker = FilterRanker(method="KL", bins=10)
                ranked_features, ranked_importances = ranker.run(
                    self._df, filled_num_features, target_col, treatment_col
                )

                other_features = [
                    f
                    for f in self._get_available_features()
                    if f not in filled_num_features
                ]
                ranker = ImportanceRanker(SoloModel, model_params, "at_once")
                other_ranked_features, other_ranked_importances = ranker.run(
                    self._df, other_features, target_col, treatment_col
                )

                ranked_features = (
                    ranked_features[: n_features // 2]
                    + other_ranked_features[: n_features - n_features // 2]
                    + ranked_features[n_features // 2 :]
                    + other_ranked_features[n_features - n_features // 2 :]
                )

            else:
                raise ValueError(
                    f"'method' parameter must be either 'importance' or 'filter', but is {method}"
                )
        finally:
            self._df[treatment_col] = self._df[treatment_col].map(
                treatment_map_inv
            )

        self._removed_features["preselection"] = ranked_features[
            n_features:
        ].copy()

        save_json(
            ranked_features[n_features:],
            "2_preselect_feature_candidates",
            "2_feature_selection",
            self._run_id,
        )

    def _preselect_features_candidates_multitreatment(
        self, n_features: int = 300
    ):
        """Saves a long-list of features which have potential benefit.

        Note:
            - feature preselction needs to be very fast and simple
            - features that have potential for the task may be rarely filled or be categorical

        Args:
            n_features (int, optional): number of feature to use in further pipeline steps. Default is 200

        Returns: None
        """
        target_col = self._base_cols_mapper["target"]
        treatment_col = self._base_cols_mapper["treatment"]

        treatment_map = self._treatment_groups_mapper
        treatment_map_inv = {
            v: k for k, v in self._treatment_groups_mapper.items()
        }

        self._df[treatment_col] = self._df[treatment_col].map(treatment_map)

        try:
            # repeated runs of preselection need to be independent
            if "preselection" in self._removed_features:
                del self._removed_features["preselection"]

            model_params = {
                "iterations": 200,
                "max_depth": 4,
                "learning_rate": 0.05,
                "cat_features": [
                    f
                    for f in self._get_available_features()
                    if self._df[f].dtype == "object"
                ],
                "silent": True,
                "random_state": RANDOM_STATE,
            }

            ranker = ImportanceRanker(
                CatBoostClassifier, model_params, "at_once"
            )
            ranked_features, ranked_importances = ranker.run(
                self._df,
                self._get_available_features() + [treatment_col],
                target_col,
                "",
            )

            features_info = list(zip(ranked_importances, ranked_features))
            features_info = [
                p
                for p in features_info
                if p[1] not in [treatment_col, "treatment"]
            ]
            ranked_importances, ranked_features = map(list, zip(*features_info))

            n_important = len([imp for imp in ranked_importances if imp > 0])

            if n_important < n_features:
                zero_importance_features = [
                    f
                    for f, imp in zip(ranked_features, ranked_importances)
                    if not imp > 0 and f not in ["treatment", treatment_col]
                ]

                model_params["cat_features"] = [
                    f
                    for f in zero_importance_features
                    if self._df[f].dtype == "object"
                ]

                ranker = ImportanceRanker(
                    CatBoostClassifier, model_params, "at_once"
                )
                zero_ranked_features, zero_ranked_importances = ranker.run(
                    self._df,
                    zero_importance_features,
                    target_col,
                    "",
                )

                features_imps = zip(ranked_features, ranked_importances)
                features_imps = [p for p in features_imps if p[1] > 0]
                features_imps = [
                    p
                    for p in features_imps
                    if p[0] not in [treatment_col, "treatment"]
                ]
                ranked_features, ranked_importances = map(
                    list, zip(*features_imps)
                )

                ranked_features += zero_ranked_features
                ranked_importances += zero_ranked_importances

        finally:
            self._df[treatment_col] = self._df[treatment_col].map(
                treatment_map_inv
            )

        self._removed_features["preselection"] = ranked_features[
            n_features:
        ].copy()

        save_json(
            ranked_features[n_features:],
            "2_preselect_feature_candidates",
            "2_feature_selection",
            self._run_id,
        )

    def preselect_features_candidates(
        self, n_features: int = 300, method: str = "filter"
    ):
        """Run features pre‑selection step.

        Args:
            n_features: Number of candidate features to retain.
            method: Used only for the binary‑treatment case;
                ignored for multi‑treatment.

        Returns:
            None
        """
        if self._use_multitreatment:
            self._preselect_features_candidates_multitreatment(n_features)
        else:
            self._preselect_features_candidates_binary_treatment(
                n_features, method
            )

    def check_feature_values(
        self,
        max_nan_ratio: float = 0.95,
        max_categories_count: int = 20,
    ):
        """Check NaNs and unique values for each feature.

        Args:
            max_nan_ratio: maximum allowed percentage of missed values. Default is 0.95
            max_categories_count: maximum allowed number of unique values for
                categorical features. Default is 20

        Returns: None
        """
        if not self._use_default_run:
            self.logger.info("Simple feature values checks.")

        all_feature_cols = self._get_available_features()

        if not self._use_default_run:
            self.logger.info(f"{len(all_feature_cols):7} features in total")

        filled_feature = check_nans(
            self._df, all_feature_cols, max_nan_ratio=max_nan_ratio
        )
        remove_features = [
            col for col in all_feature_cols if col not in filled_feature
        ]
        self._removed_features["too much nans"] = remove_features

        if not self._use_default_run:
            self.logger.info(
                f"{len(filled_feature):7} features with less than {int(100 * max_nan_ratio)}% nans"
            )

        process_too_much_categories(
            self._df,
            all_feature_cols,
            max_categories_count=max_categories_count,
        )

        if not self._use_default_run:
            self.logger.info(
                "Process categorical features with too much unique values."
            )
            self.logger.info(
                f"{'':<4}all they have now no more than {max_categories_count} categories."
            )

    def check_correlated_features(
        self,
        max_abs_corr: float = 0.9,
        check_only_available_features: bool = True,
    ):
        """Remove correlated features by threshold.

        Args:
            max_abs_corr: maximum allowed level of features correlation. Default is 0.9
            check_only_available_features: whether to use only filtered features. Default is True

        Returns: None
        """
        if check_only_available_features:
            too_correlated, clean_feature_cols = check_correlations(
                self._df,
                self._get_available_features(),
                max_abs_corr=max_abs_corr,
            )
        else:
            too_correlated, clean_feature_cols = check_correlations(
                self._df, self._feature_cols, max_abs_corr=max_abs_corr
            )

        remove_features = [g for f, g in too_correlated]
        save_json(
            remove_features,
            "3_check_correlated_features",
            "2_feature_selection",
            self._run_id,
        )
        self._removed_features["too correlated"] = remove_features.copy()

    def show_removed_features_with_reasons(self):
        """
        Print number of features which should be removed by some reason.
        """
        self.logger.info("Number of features removed for each reason:")
        for reason, removed_features in self._removed_features.items():
            self.logger.info(f"{'':<4}{len(removed_features):5} : {reason}")

        save_json(
            self._removed_features,
            "4_deleted_features_with_reason",
            "2_feature_selection",
            self._run_id,
        )

    def get_removed_features_by_reason(self, reason: str):
        """
        Return features which should be removed by some reason.
        """
        return self._removed_features[reason]

    def plot_treatment_leaks(
        self, top_k: int = None, features: tp.List[str] = None
    ):
        """Plot probability density function by group for each feature.

        Plot distributions for features with top_k highest ROC-AUC in predicting
        treatment Distributions are displayed for all treatment groups for every
        of selected features
        """
        assert not (
            top_k is None and features is None
        ), "Specify only one set of features for analyzing"
        assert not (
            top_k is not None and features is not None
        ), "Specify only one set of features for analyzing"

        if features is not None:
            for f in features:
                assert (
                    f in self._feature_cols
                ), f"features to remove must be in feature list, but '{f}' isn't"

        features_roc_aucs = self._feature_cols_treatment_roc_aucs

        if features is not None:
            features_roc_aucs_copy = features_roc_aucs.copy()
            features_roc_aucs = []
            for col in features:
                for f, roc_auc in features_roc_aucs_copy:
                    if f == col:
                        features_roc_aucs.append((col, roc_auc))

                if not features_roc_aucs or features_roc_aucs[-1][0] != col:
                    features_roc_aucs.append((col, None))

        else:
            features_roc_aucs = features_roc_aucs[:top_k]

        n_rows, n_cols = (len(features_roc_aucs) + 2) // 3, min(
            len(features_roc_aucs), 3
        )
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4)
        )

        for idx, (feature, roc_auc) in enumerate(features_roc_aucs):
            plt.subplot(n_rows, n_cols, 1 + idx)

            values = self._df.loc[~self._df[feature].isna(), feature]
            q01, q99 = np.quantile(values, q=[0.01, 0.99])

            mask = (q01 < self._df[feature]) & (self._df[feature] < q99)
            sns.kdeplot(
                data=self._df.loc[
                    mask, [feature, self._base_cols_mapper["treatment"]]
                ],
                x=feature,
                hue=self._base_cols_mapper["treatment"],  # col_to_check
                ax=plt.gca(),
                palette=list(plt.cm.Reds(np.linspace(0, 1, 20))[[6, 14]]),
                common_norm=False,
            )

            if roc_auc is not None:
                plt.title(
                    f"Validation ROC-AUC\nquantile(95%) = {roc_auc:.3f}",
                    fontsize=14,
                )
            else:
                plt.title(
                    "Feature wasn't considered\nas leak of treatment",
                    fontsize=14,
                )

            plt.xlabel(feature, fontsize=14)
            plt.ylabel("Density", fontsize=14)

        plt.tight_layout()
        save_figure(
            fig, "1_treatment_leaks", "2_feature_selection", self._run_id
        )
        plt.show()

    def remove_features(self, features: tp.List[str]):
        """Remove features from the list of used features for a custom reason.

        Args:
            features: list of features to remove

        Returns: None
        """
        for f in features:
            assert (
                f in self._feature_cols
            ), f"features to remove must be in feature list, but '{f}' isn't"

        if "custom blacklist" not in self._removed_features:
            self._removed_features["custom blacklist"] = []

        self._removed_features["custom blacklist"].extend(features)

    def rank_features_candidates(
        self,
        ranker_types: tp.List[str] = None,
        opt_metric: tp.Callable[
            [np.array, np.array, np.array], np.array
        ] = qini_auc_score,
    ):
        """Rank features using one or several ranking methods.

        Args:
            ranker_types: List of ranker identifiers to
                run. If None the first two available rankers are used.
            opt_metric: Metric used for permutation‑based rankers.

        Returns:
            None
        """
        if self._use_multitreatment:
            available_rankers = ["importance_s_learner", "importance_forest"]
        else:
            available_rankers = [
                "filter",
                "importance",
                "permutation",
                "stepwise",
                "straightforward",
            ]

        if ranker_types is None:
            ranker_types = available_rankers[:2]

        assert set(ranker_types).issubset(set(available_rankers)), (
            f"Available ranker_types are {available_rankers}.\n"
            f"Received ranker_types are {ranker_types}.\n"
        )

        segm_col = self._base_cols_mapper["segm"]
        target_col = self._base_cols_mapper["target"]
        treatment_col = self._base_cols_mapper["treatment"]

        treatment_map = self._treatment_groups_mapper
        treatment_map_inv = {
            v: k for k, v in self._treatment_groups_mapper.items()
        }

        if not self._use_multitreatment:
            self._df[treatment_col] = self._df[treatment_col].map(treatment_map)

        catboost_params = {
            "iterations": 200,
            "max_depth": 4,
            "learning_rate": 0.05,
            "cat_features": [
                f
                for f in self._get_available_features()
                if self._df[f].dtype == "object"
            ],
            "random_seed": RANDOM_STATE,
            "silent": True,
        }

        s_learner_params = {
            "estimator": CatBoostClassifier(**catboost_params),
            "method": "dummy",
        }

        self._ranked_candidates = dict()

        with logging_redirect_tqdm():
            progress = ManualProgressBar(
                total=100,
                description="Ranking features",
                verbosity=self.verbosity,
            )
            update_perc = int(100 / len(ranker_types))
            try:
                if "filter" in ranker_types:
                    ranker = FilterRanker(method="KL", bins=10)

                    ranked_features, ranked_importances = ranker.run(
                        self._df,
                        self._get_available_features(),
                        target_col,
                        treatment_col,
                    )
                    progress.update(update_perc)

                    self._ranked_candidates["filter"] = ranked_features.copy()

                if "importance" in ranker_types:
                    ranker = ImportanceRanker(
                        model_class=SoloModel,
                        model_params=s_learner_params,
                        sorting_mode="iterative",
                    )

                    ranked_features, ranked_importances = ranker.run(
                        self._df.loc[self._df[segm_col] != "test"],
                        self._get_available_features(),
                        target_col,
                        treatment_col,
                    )
                    progress.update(update_perc)

                    self._ranked_candidates[
                        "importance"
                    ] = ranked_features.copy()

                if "permutation" in ranker_types:
                    rng = np.random.RandomState(seed=RANDOM_STATE)
                    ranker = PermutationRanker(
                        SoloModel, s_learner_params, rng, BOOTSTRAP_REPEATS
                    )

                    ranked_features, ranked_importances = ranker.run(
                        self._df.loc[self._df[segm_col] == "train"],
                        self._df.loc[self._df[segm_col] == "val"],
                        self._get_available_features(),
                        target_col,
                        treatment_col,
                        metric=opt_metric,
                    )
                    progress.update(update_perc)

                    self._ranked_candidates[
                        "permutation"
                    ] = ranked_features.copy()

                if "stepwise" in ranker_types:
                    rng = np.random.RandomState(seed=RANDOM_STATE)
                    ranker = StepwiseRanker(
                        SoloModel, s_learner_params, rng, BOOTSTRAP_REPEATS
                    )

                    ranked_features, ranked_importances = ranker.run(
                        self._df.loc[self._df[segm_col] == "train"],
                        self._df.loc[self._df[segm_col] == "val"],
                        self._get_available_features(),
                        target_col,
                        treatment_col,
                        metric=opt_metric,
                    )
                    progress.update(update_perc)

                    self._ranked_candidates["stepwise"] = ranked_features.copy()

                if "straightforward" in ranker_types:
                    rng = np.random.RandomState(seed=RANDOM_STATE)
                    ranker = StraightforwardRanker(
                        SoloModel, s_learner_params, rng, BOOTSTRAP_REPEATS
                    )

                    ranked_features, ranked_importances = ranker.run(
                        self._df.loc[self._df[segm_col] == "train"],
                        self._df.loc[self._df[segm_col] == "val"],
                        self._get_available_features(),
                        target_col,
                        treatment_col,
                        metric=opt_metric,
                    )
                    progress.update(update_perc)

                    self._ranked_candidates["stepwise"] = ranked_features.copy()

                if "importance_s_learner" in ranker_types:
                    ranker = ImportanceRanker(
                        CatBoostClassifier, catboost_params, "iterative"
                    )

                    self._df[treatment_col] = self._df[treatment_col].map(
                        treatment_map
                    )

                    ranked_features, ranked_importances = ranker.run(
                        self._df.loc[self._df[segm_col] != "test"],
                        self._get_available_features() + [treatment_col],
                        target_col,
                        "",
                    )

                    self._df[treatment_col] = self._df[treatment_col].map(
                        treatment_map_inv
                    )

                    features_info = list(
                        zip(ranked_importances, ranked_features)
                    )
                    features_info = [
                        p
                        for p in features_info
                        if p[1] not in [treatment_col, "treatment"]
                    ]
                    ranked_importances, ranked_features = map(
                        list, zip(*features_info)
                    )

                    progress.update(update_perc)

                    self._ranked_candidates[
                        "importance_s_learner"
                    ] = ranked_features.copy()

                if "importance_forest" in ranker_types:
                    forest_params = {
                        "control_name": "control",
                        "n_estimators": 200,
                        "max_features": 15,
                        "random_state": RANDOM_STATE,
                        "max_depth": 4,
                    }

                    ranker = ImportanceRanker(
                        UpliftRandomForestClassifier, forest_params, "at_once"
                    )

                    ranked_features, ranked_importances = ranker.run(
                        self._df.loc[self._df[segm_col] != "test"],
                        self._get_available_features(),
                        target_col,
                        treatment_col,
                    )

                    features_info = list(
                        zip(ranked_importances, ranked_features)
                    )
                    features_info = [
                        p
                        for p in features_info
                        if p[1] not in [treatment_col, "treatment"]
                    ]
                    ranked_importances, ranked_features = map(
                        list, zip(*features_info)
                    )

                    progress.update(update_perc)

                    self._ranked_candidates[
                        "importance_forest"
                    ] = ranked_features.copy()

                save_json(
                    self._ranked_candidates,
                    "1_ranked_features_candidates",
                    "3_feature_ranking",
                    self._run_id,
                )

                all_candidates = list()
                for cands in self._ranked_candidates.values():
                    all_candidates.extend(cands)

                # compare top ranked features
                TOP_TO_COMPARE = 20
                top_ranked_candidates = pd.DataFrame(
                    self._ranked_candidates
                ).head(TOP_TO_COMPARE)

                def bold_common_values(val, common_values):
                    if val in common_values:
                        return f"<b>{val}</b>"
                    return str(val)

                ranked_candidates_names = top_ranked_candidates.columns
                common_candidates = set(
                    top_ranked_candidates[ranked_candidates_names[0]]
                )
                for candidate in ranked_candidates_names[1:]:
                    common_candidates.intersection_update(
                        top_ranked_candidates[candidate]
                    )

                html_styled_top = top_ranked_candidates.applymap(
                    lambda x: bold_common_values(x, common_candidates)
                )

                save_dataframe_html(
                    html_styled_top,
                    "3_rankers_top_feats_comparison",
                    "3_feature_ranking",
                    self._run_id,
                )

            except Exception as e:
                self.logger.warning("Exception was raised:", str(e))
                self._ranked_candidates = dict()
                raise e

            finally:
                if not self._use_multitreatment:
                    self._df[treatment_col] = self._df[treatment_col].map(
                        treatment_map_inv
                    )
                progress.close()

    def train_models(
        self,
        classes: tp.List[str] = None,
        features: tp.List[str] = None,
        feature_nums: tp.Union[tp.List[int], tp.Dict[str, tp.List[int]]] = [
            20,
            35,
            50,
            100,
        ],
        metric: object = None,
        timeout_estimator: tp.Union[int, tp.Dict[str, int]] = 60 * 3,
        search_class=OptunaOptimizer,
        overfit_metric: object = overfit_metric_minus_metric_delta,
        training_mode: tp.Literal["light", "medium", "hard"] = "light",
    ):
        """Train a collection of uplift models.

        Args:
            classes: Model class names to train.
            features: List of feature names to use.
            models: Custom list of instantiated models with a display name.
            feature_nums: Either a list of feature counts or a dict mapping model names to lists.
            use_default_params: If True default model parameters
                are used.
            metric: Metric for Optuna optimisation.
            timeout_estimator: Maximum training time (seconds) per estimator.
            search_class: Optimiser class.
            overfit_metric: Optional over‑fit penalty for Optuna.
            training_mode: Search intensity.

        Returns:
            None
        """
        assert classes, "'classes' must be not empty"

        assert not (
            features is None and self._ranked_candidates is None
        ), "parameter features is None, specify it or call rank_features_candidates() method"

        if isinstance(feature_nums, dict):
            assert set(classes) == set(
                feature_nums.keys()
            ), "if feature_nums is dict, feature_nums for all classes for model traning must be specified"
        else:
            feature_nums = {
                class_name: feature_nums.copy() for class_name in classes
            }

        if isinstance(timeout_estimator, dict):
            assert set(classes) == set(
                timeout_estimator.keys()
            ), "if timeout_estimator is dict, timeout_estimator for all classes for model traning must be specified"
        else:
            timeout_estimator = {
                class_name: timeout_estimator for class_name in classes
            }

        assert training_mode in ["light", "medium", "hard"]

        if self._use_multitreatment:
            cls_map = {
                "BaseSClassifier": BaseSClassifier,
                "BaseTClassifier": BaseTClassifier,
                "BaseXClassifier": BaseXClassifier,
                "UpliftTreeClassifier": UpliftTreeClassifier,
                "UpliftRandomForestClassifier": UpliftRandomForestClassifier,
            }

            cls_time_map = {
                "BaseSClassifier": 1,
                "BaseTClassifier": 2,
                "BaseXClassifier": 4,
                "UpliftTreeClassifier": 10,
                "UpliftRandomForestClassifier": 30,
            }
        else:
            cls_map = {
                "CatBoostClassifier": CatBoostClassifier,
                "SoloModel": SoloModel,
                "TwoModels": TwoModels,
                "AufRandomForestClassifier": AufRandomForestClassifier,
                "AufTreeClassifier": AufTreeClassifier,
                "AufXLearner": AufXLearner,
            }

            cls_time_map = {
                "CatBoostClassifier": 1,
                "SoloModel": 1,
                "TwoModels": 2,
                "AufXLearner": 4,
                "AufTreeClassifier": 10,
                "AufRandomForestClassifier": 30,
            }

        if classes is not None:
            for cls_name in classes:
                if cls_name not in cls_map:
                    raise ValueError(
                        "Check correctness of passed class names: "
                        + f"{cls_name} not found"
                    )
            times = [cls_time_map[cls_name] for cls_name in classes]
            classes = [cls_map[cls_name] for cls_name in classes]

        self._train_metric = metric

        model_info = dict()  # make method calls independent

        segm_col = self._base_cols_mapper["segm"]
        target_col = self._base_cols_mapper["target"]
        treatment_col = self._base_cols_mapper["treatment"]

        treatment_map = self._treatment_groups_mapper
        treatment_map_inv = {
            v: k for k, v in self._treatment_groups_mapper.items()
        }

        if not self._use_multitreatment:
            self._df[treatment_col] = self._df[treatment_col].map(treatment_map)

        try:
            df_train_mask = self._df[segm_col] == "train"
            df_val_mask = self._df[segm_col] == "val"
            df_test_mask = self._df[segm_col] == "test"

            if features is None:
                features = self._ranked_candidates
            else:
                features = {"custom_features": features}

            for class_name in feature_nums:
                if max(feature_nums[class_name]) > len(
                    self._get_available_features()
                ):
                    self.logger.warning(
                        "feature_nums must contain numbers no greater than number of available features. Start filtering it."
                    )
                    break

            for class_name in feature_nums:
                feature_nums[class_name] = sorted(
                    [
                        x
                        for x in feature_nums[class_name]
                        if x <= len(self._get_available_features())
                    ]
                )

            with logging_redirect_tqdm():
                progress = ManualProgressBar(
                    total=sum(times),
                    description="Training models",
                    verbosity=self.verbosity,
                )

                for idx, cls_model in enumerate(classes):
                    self.logger.info(f"{cls_model.__name__} training started")

                    progress.update_description(
                        f"Training models. {cls_model.__name__} training"
                    )

                    treatment_groups = None

                    if self._use_multitreatment:
                        if cls_model in [
                            BaseSClassifier,
                            BaseTClassifier,
                            BaseXClassifier,
                        ]:
                            self._df[treatment_col] = self._df[
                                treatment_col
                            ].map(treatment_map)

                        treatment_groups = sorted(
                            list(self._df[treatment_col].unique())
                        )

                    cls_results = generate_model_from_classes(
                        cls_model,
                        self._df.loc[df_train_mask],
                        self._df.loc[df_val_mask],
                        self._df.loc[df_test_mask],
                        features,
                        target_col,
                        treatment_col,
                        feature_nums[cls_model.__name__],
                        timeout_estimator[cls_model.__name__],
                        metric,
                        search_class=search_class,
                        overfit_metric=overfit_metric,
                        treatment_groups=treatment_groups,
                        training_mode=training_mode,
                    )

                    if self._use_multitreatment:
                        if cls_model in [
                            BaseSClassifier,
                            BaseTClassifier,
                            BaseXClassifier,
                        ]:
                            self._df[treatment_col] = self._df[
                                treatment_col
                            ].map(treatment_map_inv)

                    for ranker, results in cls_results.items():
                        for jdx in range(len(results)):
                            results[jdx] = ModelResult(
                                auf_model=results[jdx],
                                median_test_metric=None,
                            )

                    model_info[cls_model.__name__] = cls_results
                    progress.update(times[idx])
                    self.logger.info(
                        f"{cls_model.__name__} successfully trained"
                    )

                progress.update_description("Training models")
                progress.close()

        except Exception as e:
            self.logger.warning("Exception was raised:", str(e))
            self._train_results = dict()
            raise e

        finally:
            if set(self._df[treatment_col].unique()) != set(
                treatment_map.keys()
            ):
                self._df[treatment_col] = self._df[treatment_col].map(
                    treatment_map_inv
                )

        self._train_results = model_info

    def _get_median_test_metrics(
        self, metric: tp.Callable[[np.array, np.array, np.array], float]
    ):
        treatment_groups = None

        if self._use_multitreatment:
            treatment_groups = self._treatment_groups_mapper.keys()

            def multi_metric(y_true, uplift, treatment):
                control_name = 0 if 0 in treatment_groups else "control"
                return metric(
                    y_true=y_true,
                    uplift=uplift.max(axis=1),
                    treatment=(treatment != control_name).astype(int),
                )

        segm_col = self._base_cols_mapper["segm"]
        target_col = self._base_cols_mapper["target"]
        treatment_col = self._base_cols_mapper["treatment"]

        treatment_map = self._treatment_groups_mapper
        treatment_map_inv = {
            v: k for k, v in self._treatment_groups_mapper.items()
        }

        if not self._use_multitreatment:
            self._df[treatment_col] = self._df[treatment_col].map(treatment_map)

        mask_test = self._df[segm_col] == "test"
        x_test = self._df.loc[mask_test, self._get_available_features()]
        y_test = self._df.loc[mask_test, target_col].values
        t_test = self._df.loc[mask_test, treatment_col].values

        for model_name, rankers in self._train_results.items():
            if model_name == "baseline":
                continue

            if self._use_multitreatment and model_name in [
                "BaseSClassifier",
                "BaseTClassifier",
                "BaseXClassifier",
            ]:
                treatment_groups = self._treatment_groups_mapper.values()
                self._df[treatment_col] = self._df[treatment_col].map(
                    treatment_map
                )
                t_test = self._df.loc[mask_test, treatment_col].values
                t_test = (t_test != treatment_map["control"]).astype(int)

            for ranker_method, current_results in rankers.items():
                for result in current_results:
                    if self._use_multitreatment:
                        uplift_test = result.auf_model.predict(
                            x_test[result.auf_model._features]
                        )
                    else:
                        uplift_test = result.auf_model.predict(
                            x_test[result.auf_model._features], return_df=False
                        )

                    metric_values = np.zeros(shape=BOOTSTRAP_REPEATS)
                    rng = np.random.RandomState(seed=RANDOM_STATE)

                    for it in range(BOOTSTRAP_REPEATS):
                        cnt = mask_test.sum()
                        idxs = rng.choice(range(cnt), size=cnt, replace=True)
                        if self._use_multitreatment:
                            y, u, t = (
                                y_test[idxs],
                                uplift_test.iloc[idxs],
                                t_test[idxs],
                            )
                            metric_values[it] = multi_metric(y, u, t)
                        else:
                            y, u, t = (
                                y_test[idxs],
                                uplift_test[idxs],
                                t_test[idxs],
                            )
                            metric_values[it] = metric(y, u, t)

                    result.median_test_metric = np.median(metric_values)

            if self._use_multitreatment and model_name in [
                "BaseSClassifier",
                "BaseTClassifier",
                "BaseXClassifier",
            ]:
                treatment_groups = self._treatment_groups_mapper.keys()
                self._df[treatment_col] = self._df[treatment_col].map(
                    treatment_map_inv
                )
                t_test = self._df.loc[mask_test, treatment_col].values

        if not self._use_multitreatment:
            self._df[treatment_col] = self._df[treatment_col].map(
                treatment_map_inv
            )

    def get_result(
        self,
        metric: tp.Callable[[np.array, np.array, np.array], float],
        n_max_features: int = None,
        rating: int = None,
    ):
        """Select the best model according to a metric on the test set.

        Args:
            metric: Metric used for ranking.
            n_max_features: Upper bound on the number of
                features a model may use.
            rating: Zero‑based index of the desired model
                after sorting. If None the best model is selected
                automatically.

        Returns:
            tuple: (model_class_name, ranker_method, ModelResult).
        """
        assert (
            n_max_features is not None
        ), "n_max_features is None, but should be either set by user or by default value in caller"

        self._get_median_test_metrics(metric)

        results = []

        for model_name, rankers in self._train_results.items():
            if model_name == "baseline":
                continue

            for ranker_method, current_results in rankers.items():
                for result in current_results:
                    if len(result.auf_model._features) <= n_max_features:
                        results.append((model_name, ranker_method, result))

        results = sorted(results, key=lambda p: -1 * p[2].median_test_metric)

        if rating is None:
            results = [
                p
                for p in results
                if p[2].median_test_metric
                >= 0.95 * results[0][2].median_test_metric
            ]
            min_n_features = min(len(p[2].features) for p in results)
            results = [
                p for p in results if len(p[2].features) == min_n_features
            ]
            results = sorted(
                results, key=lambda p: p[0]
            )  # e.g. "SoloModel" < "TwoModels" < "UpliftRandomForest"
            model_class_name, ranker_method, result = results[0]
        else:
            model_class_name, ranker_method, result = results[rating]

        return model_class_name, ranker_method, result

    def _get_trn_vld_tst_predictions(self, auf_model: AufModel):
        segm_col = self._base_cols_mapper["segm"]

        mask_trn = self._df[segm_col] == "train"
        mask_val = self._df[segm_col] == "val"
        mask_tst = self._df[segm_col] == "test"

        if self._use_multitreatment:
            uplift_trn = auf_model.predict(self._df.loc[mask_trn])
            uplift_val = auf_model.predict(self._df.loc[mask_val])
            uplift_tst = auf_model.predict(self._df.loc[mask_tst])
        else:
            uplift_trn = auf_model.predict(
                self._df.loc[mask_trn], return_df=False
            )
            uplift_val = auf_model.predict(
                self._df.loc[mask_val], return_df=False
            )
            uplift_tst = auf_model.predict(
                self._df.loc[mask_tst], return_df=False
            )

        return uplift_trn, uplift_val, uplift_tst

    def _modify_catboost_params_dict(self, params: tp.Dict[str, float]):
        assert "depth" in params or "max_depth" in params
        assert "iterations" in params or "n_estimators" in params

        if "depth" in params:
            params["max_depth"] = params["depth"]
        else:
            params["depth"] = params["max_depth"]

        if "n_estimators" in params:
            params["iterations"] = params["n_estimators"]
        else:
            params["n_estimators"] = params["iterations"]

    def show_metrics_table(
        self,
        metrics_names: tp.List[str] = None,
        round_digits: int = 3,
        show_segments: tp.List[str] = ["train", "val", "test"],
    ):
        """Return table with all requested metrics for every trained model.

        Args:
            metrics_names: List of metric identifiers.
            round_digits: Number of decimal places for rounding.
            show_segments: Sub‑sets of data to include.

        Returns:
            pd.DataFrame: DataFrame with a multi‑index.
        """
        assert metrics_names is None or all(
            [
                any([name == metric_name for name in METRICS])
                for metric_name in metrics_names
            ]
        ), (
            "Specify correct metric names or metric functions itself, already available metrics are:\n\t"
            + "\n\t".join(list(METRICS.keys()))
        )

        if metrics_names is None:
            use_at_k = list(range(5, 41, 5))
            metrics_names = []
            for name in METRICS.keys():
                if "@" in name and int(name.split("@")[-1]) not in use_at_k:
                    continue
                if (
                    "_bins" in name
                    and int(name.split("_bins")[0].split("_")[-1])
                    not in use_at_k
                ):
                    continue
                metrics_names.append(name)

        assert (
            show_segments
            and set(show_segments) - set({"train", "val", "test"}) == set()
        ), "show_segments parameter must be not empty and contain only labels from {'train', 'val', 'test'}"

        segm_col = self._base_cols_mapper["segm"]
        target_col = self._base_cols_mapper["target"]
        treatment_col = self._base_cols_mapper["treatment"]

        target = self._df[target_col]
        treatment = self._df[treatment_col].copy()

        treatment_map = self._treatment_groups_mapper
        treatment_map_inv = {
            v: k for k, v in self._treatment_groups_mapper.items()
        }

        if not self._use_multitreatment:
            treatment = treatment.map(treatment_map)

        treatment_groups = sorted(list(treatment_map.keys()))
        control_group = 0 if 0 in treatment_groups else "control"

        mask_trn = self._df[segm_col] == "train"
        mask_val = self._df[segm_col] == "val"
        mask_tst = self._df[segm_col] == "test"

        y_trn, y_val, y_tst = (
            target[mask_trn],
            target[mask_val],
            target[mask_tst],
        )
        t_trn, t_val, t_tst = (
            treatment[mask_trn],
            treatment[mask_val],
            treatment[mask_tst],
        )

        metrics_df = None

        for model_name, rankers in self._train_results.items():
            if self._use_multitreatment:
                if model_name in [
                    "BaseSClassifier",
                    "BaseTClassifier",
                    "BaseXClassifier",
                ]:
                    treatment = treatment.map(treatment_map)
                    t_trn, t_val, t_tst = (
                        t_trn.map(treatment_map),
                        t_val.map(treatment_map),
                        t_tst.map(treatment_map),
                    )
                    treatment_groups = sorted(list(treatment_map.values()))
                    control_group = 0 if 0 in treatment_groups else "control"

            for ranker_method, current_results in rankers.items():
                for result in current_results:
                    (
                        uplift_trn,
                        uplift_val,
                        uplift_tst,
                    ) = self._get_trn_vld_tst_predictions(result.auf_model)

                    n_features = len(result.auf_model._features)

                    uplift_type = None
                    if not self._use_multitreatment:
                        uplift_type = result.auf_model._uplift_prediction_type

                    for (
                        segm_name,
                        segm_target,
                        segm_treatment,
                        segm_uplift,
                    ) in zip(
                        ["train", "val", "test"],
                        [y_trn, y_val, y_tst],
                        [t_trn, t_val, t_tst],
                        [uplift_trn, uplift_val, uplift_tst],
                    ):
                        if segm_name not in show_segments:
                            continue

                        if self._use_multitreatment:
                            metrics_info_row = {}
                        else:
                            metrics_info_row = {
                                "uplift_type": uplift_type,
                            }

                        metrics_info_row.update(
                            {
                                "model_name": model_name,
                                "ranker_method": ranker_method,
                                "n_features": n_features,
                                "segm": segm_name,
                            }
                        )

                        for name, function in METRICS.items():
                            if name in metrics_names:
                                if self._use_multitreatment:
                                    score = function(
                                        y_true=segm_target,
                                        treatment=(
                                            segm_treatment != control_group
                                        ).astype(int),
                                        uplift=segm_uplift.max(axis=1),
                                    )
                                else:
                                    score = function(
                                        y_true=segm_target,
                                        treatment=segm_treatment,
                                        uplift=segm_uplift,
                                    )
                                metrics_info_row[name] = np.round(
                                    score, round_digits
                                )

                        if metrics_df is None:
                            metrics_df = pd.DataFrame(
                                columns=list(metrics_info_row.keys())
                            )

                        metrics_info_row_df = pd.DataFrame([metrics_info_row])
                        metrics_df = pd.concat(
                            [metrics_df, metrics_info_row_df], ignore_index=True
                        )

            if self._use_multitreatment:
                if model_name in [
                    "BaseSClassifier",
                    "BaseTClassifier",
                    "BaseXClassifier",
                ]:
                    treatment = treatment.map(treatment_map_inv)
                    t_trn, t_val, t_tst = (
                        t_trn.map(treatment_map_inv),
                        t_val.map(treatment_map_inv),
                        t_tst.map(treatment_map_inv),
                    )
                    treatment_groups = sorted(list(treatment_map.keys()))
                    control_group = 0 if 0 in treatment_groups else "control"

        index_cols = ["model_name", "ranker_method", "n_features"]
        if not self._use_multitreatment:
            index_cols.append("uplift_type")

        metrics_df.set_index(index_cols, inplace=True)

        return metrics_df

    def plot_results(
        self,
        metrics_df: pd.DataFrame,
        model_class_name: str,
        ranker_method: str,
        auf_model: AufModel,
        n_uplift_bins: int,
    ):
        """Generate a comprehensive report for the selected model.

        Args:
            metrics_df: DataFrame returned by show_metrics_table.
            model_class_name: Name of the model class to visualise.
            ranker_method: Ranker that produced the feature set.
            auf_model: Trained uplift model.
            n_uplift_bins: Number of bins for percentile‑based plots.

        Returns:
            None
        """
        segm_col = self._base_cols_mapper["segm"]
        target_col = self._base_cols_mapper["target"]
        treatment_col = self._base_cols_mapper["treatment"]

        target = self._df[target_col].copy()
        treatment = self._df[treatment_col].copy()

        treatment_map = self._treatment_groups_mapper

        if not self._use_multitreatment:
            treatment = treatment.map(treatment_map)

        mask_trn = self._df[segm_col] == "train"
        mask_val = self._df[segm_col] == "val"
        mask_tst = self._df[segm_col] == "test"

        y_trn, y_val, y_tst = (
            target[mask_trn],
            target[mask_val],
            target[mask_tst],
        )
        t_trn, t_val, t_tst = (
            treatment[mask_trn],
            treatment[mask_val],
            treatment[mask_tst],
        )

        uplift_trn, uplift_val, uplift_tst = self._get_trn_vld_tst_predictions(
            auf_model
        )

        if self._use_multitreatment:
            treatment_groups = sorted(list(treatment_map.keys()))
            control_group = 0 if 0 in treatment_groups else "control"
            t_trn = (t_trn != control_group).astype(int)
            t_val = (t_val != control_group).astype(int)
            t_tst = (t_tst != control_group).astype(int)
            uplift_trn, uplift_val, uplift_tst = (
                uplift_trn.max(axis=1),
                uplift_val.max(axis=1),
                uplift_tst.max(axis=1),
            )

        if self._use_multitreatment:
            metric_index = (
                model_class_name,
                ranker_method,
                len(auf_model._features),
            )
        else:
            metric_index = (
                model_class_name,
                ranker_method,
                len(auf_model._features),
                auf_model._uplift_prediction_type,
            )
        model_metrics_df = metrics_df.loc[
            metrics_df.index == metric_index,
            [
                "segm",
                "uplift@10",
                "uplift_rel@10",
                "uplift@15",
                "uplift_rel@15",
                "uplift@20",
                "uplift_rel@20",
                "qini_auc",
            ],
        ]

        model_metrics_df = model_metrics_df.sort_values(by=["segm"])
        model_metrics_df = model_metrics_df.iloc[[1, 2, 0]]
        display(model_metrics_df)
        save_dataframe_html(
            model_metrics_df,
            "2_best_model_main_metrics",
            "4_modeling_results",
            self._run_id,
        )

        metrics_dict = {}
        for metric_name in [
            "uplift@10",
            "uplift_rel@10",
            "uplift@15",
            "uplift_rel@15",
            "uplift@20",
            "uplift_rel@20",
            "qini_auc",
        ]:
            for segm in ["train", "val", "test"]:
                metrics_dict[
                    f'{metric_name.replace("@", "_at_")}_{segm}'
                ] = model_metrics_df.loc[
                    model_metrics_df["segm"] == segm, metric_name
                ].values[
                    0
                ]

        save_metrics(metrics_dict, self._run_id)

        uplift_buckets_info = self.show_uplift_by_bucket(
            auf_model=auf_model,
            show_segment="test",
            n_uplift_bins=n_uplift_bins,
        )
        display(uplift_buckets_info)
        save_dataframe_html(
            uplift_buckets_info,
            "3_best_model_test_k_bins",
            "4_modeling_results",
            self._run_id,
        )

        uplift_tops_info = self.show_uplift_by_top(
            auf_model=auf_model, show_segment="test"
        )
        display(uplift_tops_info)
        save_dataframe_html(
            uplift_tops_info,
            "3_best_model_test_k_tops",
            "4_modeling_results",
            self._run_id,
        )

        if not self._use_multitreatment:
            uplift_type = {
                "abs": "absolute",
                "absolute": "absolute",
                "rel": "relative",
                "relative": "relative",
                "prop": "propensity",
                "propensity": "propensity",
            }[auf_model._uplift_prediction_type]
        else:
            uplift_type = None

        # diagram for train/val/test samples
        figure_name = os.path.join(
            "mlflow_artifacts", "2_metrics_by_segment_type.pdf"
        )
        os.makedirs(os.path.dirname(figure_name), exist_ok=True)
        with PdfPages(figure_name) as pdf:
            for row, y, t, u, segm in zip(
                [0, 1, 2],
                [y_trn, y_val, y_tst],
                [t_trn, t_val, t_tst],
                [uplift_trn, uplift_val, uplift_tst],
                ["Train", "Val", "Test"],
            ):
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                if uplift_type:
                    fig.suptitle(
                        f"Uplift type : {uplift_type}\n\n{segm} sample",
                        fontsize=14,
                    )
                else:
                    fig.suptitle(f"{segm} sample", fontsize=14)

                plot_uplift_by_percentile(
                    y_true=y,
                    uplift=u,
                    treatment=t,
                    strategy="overall",
                    kind="bar",
                    bins=n_uplift_bins,
                    string_percentiles=True,
                    axes=axes[0],
                    draw_bars="rates",
                )

                plot_qini_curve(
                    y_true=y,
                    uplift=u,
                    treatment=t,
                    random=True,
                    perfect=False,
                    ax=axes[1],
                )

                axes[0].set_title("Uplift by decile", fontsize=12)
                axes[1].set_title("Qini curve", fontsize=12)

                plt.tight_layout()
                pdf.savefig()
                plt.show()

        save_pdf_figures(figure_name, "4_modeling_results", self._run_id)

        # diagram for top vs bottom
        figure_name = os.path.join(
            "mlflow_artifacts", "2_metrics_by_test_top_bottom.pdf"
        )
        os.makedirs(os.path.dirname(figure_name), exist_ok=True)
        with PdfPages(figure_name) as pdf:
            for row, y, t, u, segm in zip(
                [0, 1, 2],
                [y_trn, y_val, y_tst],
                [t_trn, t_val, t_tst],
                [uplift_trn, uplift_val, uplift_tst],
                ["Train", "Val", "Test"],
            ):
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                if uplift_type:
                    fig.suptitle(
                        f"Uplift type : {uplift_type}\n\n{segm} sample",
                        fontsize=14,
                    )
                else:
                    fig.suptitle(f"{segm} sample", fontsize=14)

                for idx, top_ratio in enumerate([0.1, 0.2, 0.3]):
                    plot_uplift_top_vs_bottom(
                        y_true=y,
                        uplift=u,
                        treatment=t,
                        top_ratio=top_ratio,
                        kind="bar",
                        axes=axes[idx],
                    )

                    axes[idx].set_title(
                        f"Top {int(top_ratio * 100)}%", fontsize=12
                    )

                plt.tight_layout()
                pdf.savefig()
                plt.show()

        save_pdf_figures(figure_name, "4_modeling_results", self._run_id)

        # target ratio for treatment/control groups
        figure_name = os.path.join(
            "mlflow_artifacts", "2_target_ratios_by_segment_type.pdf"
        )
        os.makedirs(os.path.dirname(figure_name), exist_ok=True)
        with PdfPages(figure_name) as pdf:
            for row, y, t, u, segm in zip(
                [0, 1, 2],
                [y_trn, y_val, y_tst],
                [t_trn, t_val, t_tst],
                [uplift_trn, uplift_val, uplift_tst],
                ["Train", "Val", "Test"],
            ):
                fig, axes = plt.subplots(1, 1, figsize=(7, 7))
                if uplift_type:
                    fig.suptitle(
                        f"Uplift type : {uplift_type}\n\n{segm} sample",
                        fontsize=14,
                    )
                else:
                    fig.suptitle(f"{segm} sample", fontsize=14)

                plot_cumulative_target_ratio(
                    y_true=y,
                    uplift=u,
                    treatment=t,
                    ax=axes,
                    color_control="orange",
                    color_treatment="forestgreen",
                    linewidth=2,
                    linestyle="-",
                    random=True,
                    label=None,
                )

                plt.tight_layout()
                pdf.savefig()
                plt.show()

        save_pdf_figures(figure_name, "4_modeling_results", self._run_id)

    def plot_feature_importances(
        self,
        auf_model: AufModel,
        feature_names: tp.Optional[tp.Dict[str, str]] = None,
    ):
        """Plot the top‑10 feature importances of a model.

        Args:
            auf_model: Wrapped uplift model.
            feature_names: Optional mapping {feature_id: pretty_name}.

        Returns:
            pd.DataFrame: DataFrame with feature importances.
        """
        imps_df = auf_model.get_feature_importances(prettified=True)
        feats = imps_df["Feature Id"]
        imps = imps_df["Importance"]

        mask = np.vectorize(lambda x: "treatment" not in x)(feats)
        feats, imps = feats[mask], imps[mask]

        feats_imps = pd.DataFrame({"f": feats, "i": imps}).sort_values(
            by=["i"], ascending=False
        )
        feats_imps = feats_imps.iloc[:10].reset_index(drop=True)

        if self._feature_names:
            feats_imps.insert(
                1,
                "d",
                [
                    self._feature_names.get(feat, feat)
                    for feat in feats_imps["f"]
                ],
            )
        col_to_plot = "d" if self._feature_names else "f"

        fig, axes = plt.subplots(1, 1, figsize=(6, 6))

        plt.subplot(111)
        plt.title("TOP-10 feature importances")
        sns.barplot(x=feats_imps["i"], y=feats_imps[col_to_plot])
        plt.xlabel("feature importance")
        plt.ylabel("feature name")

        feats_imps = feats_imps.rename(
            columns={"d": "description", "i": "importance", "f": "feature"}
        ).round(4)

        save_figure(
            fig,
            "4_feature_importance_top_10",
            "4_modeling_results",
            self._run_id,
        )
        save_dataframe_html(
            feats_imps,
            "4_feature_importance_top_10",
            "4_modeling_results",
            self._run_id,
        )

        return feats_imps

    def show_uplift_by_bucket(
        self,
        auf_model: AufModel,
        show_segment: str = "test",
        n_uplift_bins: int = 10,
    ):
        """Display and return a table with uplift statistics for each bucket.

        Build per-bucket table for sample using model predictions and display it,
        after that return it.

        Args:
            auf_model: Trained uplift model.
            show_segment: One of 'train', 'val' or 'test'.
            n_uplift_bins: Number of buckets.

        Returns:
            pd.DataFrame: DataFrame with bucket‑level statistics.
        """
        assert show_segment in ["train", "val", "test"]

        mask = self._df[self._base_cols_mapper["segm"]] == show_segment

        target = self._df.loc[mask, self._base_cols_mapper["target"]]
        treatment = self._df.loc[mask, self._base_cols_mapper["treatment"]]

        if not self._use_multitreatment:
            treatment = treatment.map(self._treatment_groups_mapper)

        uplift_trn, uplift_val, uplift_tst = self._get_trn_vld_tst_predictions(
            auf_model
        )

        if self._use_multitreatment:
            treatment_map = self._treatment_groups_mapper
            treatment_groups = sorted(list(treatment_map.keys()))
            control_group = 0 if 0 in treatment_groups else "control"
            treatment = (treatment != control_group).astype(int)
            uplift_trn, uplift_val, uplift_tst = (
                uplift_trn.max(axis=1),
                uplift_val.max(axis=1),
                uplift_tst.max(axis=1),
            )

        idx = [
            i for i in range(3) if ["train", "val", "test"][i] == show_segment
        ][0]
        uplift = [uplift_trn, uplift_val, uplift_tst][idx]

        buckets_info = uplift_by_percentile(
            target, uplift, treatment, bins=n_uplift_bins
        )
        buckets_info["rel_uplift, %"] = (
            buckets_info["response_rate_treatment"]
            / buckets_info["response_rate_control"]
            - 1
        ) * 100

        target, uplift, treatment = (
            np.array(target),
            np.array(uplift),
            np.array(treatment),
        )

        order = np.argsort(uplift, kind="mergesort")[::-1]

        # score border for the bin
        uplift_bin = np.array_split(np.array(uplift)[order], n_uplift_bins)
        uplift_min = np.array([np.min(u) for u in uplift_bin])
        buckets_info["min_score"] = uplift_min

        # ratio of control targets in each bucket
        buckets_info["control_target_ratio, %"] = (
            (buckets_info["response_rate_control"] * buckets_info["n_control"])
            / (
                buckets_info["response_rate_control"]
                * buckets_info["n_control"]
            ).sum()
        ) * 100

        # ratio of treatment targets in each bucket
        buckets_info["treatment_target_ratio, %"] = (
            (
                buckets_info["response_rate_treatment"]
                * buckets_info["n_treatment"]
            )
            / (
                buckets_info["response_rate_treatment"]
                * buckets_info["n_treatment"]
            ).sum()
        ) * 100
        return buckets_info

    def show_uplift_by_top(
        self,
        auf_model: AufModel,
        show_segment: str = "test",
    ):
        """Plots uplift statistics for different top slices.

        Plot uplift statistics for different top percent of sample
        int cumuative way, return table with numerical values used for plots.

        Args:
            auf_model: Trained uplift model.
            show_segment: One of 'train', 'val' or 'test'.

        Returns:
            pd.DataFrame: DataFrame with top‑slice statistics.
        """
        assert show_segment in ["train", "val", "test"]

        mask = self._df[self._base_cols_mapper["segm"]] == show_segment

        target = self._df.loc[mask, self._base_cols_mapper["target"]]
        treatment = self._df.loc[mask, self._base_cols_mapper["treatment"]]

        if not self._use_multitreatment:
            treatment = treatment.map(self._treatment_groups_mapper)

        uplift_trn, uplift_val, uplift_tst = self._get_trn_vld_tst_predictions(
            auf_model
        )

        if self._use_multitreatment:
            treatment_map = self._treatment_groups_mapper
            treatment_groups = sorted(list(treatment_map.keys()))
            control_group = 0 if 0 in treatment_groups else "control"
            treatment = (treatment != control_group).astype(int)
            uplift_trn, uplift_val, uplift_tst = (
                uplift_trn.max(axis=1),
                uplift_val.max(axis=1),
                uplift_tst.max(axis=1),
            )

        idx = [
            i for i in range(3) if ["train", "val", "test"][i] == show_segment
        ][0]
        uplift = [uplift_trn, uplift_val, uplift_tst][idx]

        target, uplift, treatment = (
            np.array(target),
            np.array(uplift),
            np.array(treatment),
        )

        order = np.argsort(uplift, kind="mergesort")[::-1]

        target = target[order]
        uplift = uplift[order]
        treatment = treatment[order]

        tops = []
        top_response_rate_treatment = []
        top_response_rate_control = []
        bottom_response_rate_treatment = []
        bottom_response_rate_control = []
        final_response_rate = []
        top_target_ratio_treatment = []
        top_target_ratio_control = []

        for percent in range(5, 41, 5):
            tops.append(f"{percent}%")

            percent = percent / 100
            size = int(len(uplift) * percent)

            top_y, top_t = target[:size], treatment[:size]
            bot_y, bot_t = target[size:], treatment[size:]

            y_trmnt, y_cntrl = target[treatment == 1], target[treatment == 0]
            top_y_trmnt, bot_y_trmnt = top_y[top_t == 1], bot_y[bot_t == 1]
            top_y_cntrl, bot_y_cntrl = top_y[top_t == 0], bot_y[bot_t == 0]

            final_response_rate.append(
                top_y_trmnt.mean() * percent
                + bot_y_cntrl.mean() * (1 - percent)
            )

            top_response_rate_treatment.append(top_y_trmnt.mean())
            top_response_rate_control.append(top_y_cntrl.mean())

            bottom_response_rate_treatment.append(bot_y_trmnt.mean())
            bottom_response_rate_control.append(bot_y_cntrl.mean())

            top_target_ratio_treatment.append(top_y_trmnt.sum() / y_trmnt.sum())
            top_target_ratio_control.append(top_y_cntrl.sum() / y_cntrl.sum())

        tops_info = pd.DataFrame()
        tops_info["top"] = tops
        tops_info["final_response_rate"] = final_response_rate
        tops_info["top_target_ratio_treatment"] = top_target_ratio_treatment
        tops_info["top_target_ratio_control"] = top_target_ratio_control
        tops_info["top_response_rate_treatment"] = top_response_rate_treatment
        tops_info["top_response_rate_control"] = top_response_rate_control
        tops_info[
            "bottom_response_rate_treatment"
        ] = bottom_response_rate_treatment
        tops_info["bottom_response_rate_control"] = bottom_response_rate_control
        return tops_info

    def train_propensity_baseline(
        self,
        features: tp.List[str] = None,
        n_propensity_features: int = 100,
        metric: object = roc_auc_score,
        timeout_estimator: int = 60 * 3,
        search_class=OptunaOptimizer,
        overfit_metric: object = overfit_metric_minus_metric_delta,
        training_mode: tp.Literal["light", "medium", "hard"] = "light",
    ):
        """Train simple Catboost without treatment and save it.

        Args:
            features: List of features to be used for training propensity models.
            n_propensity_features: number of feature to select for propensity baseline. Defaults to 100.
            metric: optimization function for optune class.
            timeout_estimator: Time for fitting one estimator.
            search_class: Hyperparameters search class.
            overfit_metric: Optuna class optimization metric regularization.
            training_mode: Hyperparameters search mode.

        Returns: None

        Examples:
            >>> pipeline.train_propensity_baseline(
            ...     features=['age', 'gender', 'tail'],
            ...     metric=accuracy_score,
            ...     timeout_estimator=20
            ... )
        """
        if features is None:
            features = self._df.columns.tolist()
            # filter base columns
            features = [
                f for f in features if f not in self._base_cols_mapper.keys()
            ]
            features = [
                f for f in features if f not in self._base_cols_mapper.values()
            ]
            # filter features with all values missed
            features = [f for f in features if not self._df[f].isna().all()]
            # filter features with only 1 unique value
            features = [f for f in features if self._df[f].nunique() > 1]

        assert features, "'features' must be not empty"

        assert set(features).issubset(
            set(self._df.columns)
        ), f"{len(set(features) - set(self._df.columns))} признаков из features нет в данных."

        segm_col = self._base_cols_mapper["segm"]
        target_col = self._base_cols_mapper["target"]
        treatment_col = self._base_cols_mapper["treatment"]

        control_group_name = self._treatment_groups_mapper_inv[0]
        df_control_mask = self._df[treatment_col] == control_group_name

        df_train_mask = (self._df[segm_col] == "train") & df_control_mask
        df_val_mask = (self._df[segm_col] == "val") & df_control_mask

        features_selector = CatBoostClassifier(
            iterations=400,
            max_depth=4,
            learning_rate=0.03,
            random_state=RANDOM_STATE,
            silent=True,
            cat_features=[f for f in features if self._df[f].dtype == "object"],
        )

        features_selector.fit(
            self._df.loc[df_train_mask | df_val_mask, features],
            self._df.loc[df_train_mask | df_val_mask, target_col],
        )

        importances = features_selector.get_feature_importance(prettified=True)
        selected_features = importances["Feature Id"][
            :n_propensity_features
        ].tolist()

        finder_class = OptunaOptimizer(
            self._df.loc[df_train_mask],
            self._df.loc[df_val_mask],
            metric,
            treatment_col,
            target_col,
            overfit_metric,
            training_mode,
        )

        model = finder_class.find_best_params(
            CatBoostClassifier, selected_features, timeout_estimator
        )

        baseline_model = fit_model(
            model,
            self._df.loc[df_train_mask],
            selected_features,
            target_col,
            treatment_col,
            uplift_type="propensity",
            treatment_groups=None,
        )

        baseline_result = ModelResult(
            auf_model=baseline_model, median_test_metric=None
        )

        self._train_results["baseline"] = {"propensity": [baseline_result]}

    def compare_with_propensity_baseline(
        self,
        full_metrics_df: pd.DataFrame,
        best_uplift_model: AufModel,
        metrics_names: tp.Optional[tp.List[str]] = None,
    ) -> tp.Dict[str, tp.Any]:
        """Compare the best uplift model with a propensity‑score baseline.

        Args:
            full_metrics_df: Dataframe with all metrics for train, validation and test samples.
            best_uplift_model: Trained uplift model.
            metrics_names: List of metric names to be
                calculated for the propensity model.

        Raises:
            AssertionError: If the supplied models are not fitted or
                required columns are missing.
        """
        segm_col = self._base_cols_mapper["segm"]
        target_col = self._base_cols_mapper["target"]
        treatment_col = self._base_cols_mapper["treatment"]

        control_group_name = self._treatment_groups_mapper_inv[0]
        df_control_mask = self._df[treatment_col] == control_group_name

        mask_test = (self._df[segm_col] == "test") & df_control_mask
        y_test = self._df.loc[mask_test, target_col].values

        propensity_model = self._train_results["baseline"]["propensity"][
            0
        ].auf_model

        metric_index = (
            "baseline",
            "propensity",
            len(propensity_model._features),
            "propensity",
        )

        propensity_metrics = full_metrics_df.loc[
            full_metrics_df.index == metric_index,
            ["segm"] + metrics_names,
        ]

        propensity_metrics = propensity_metrics.sort_values(by=["segm"])
        propensity_metrics = propensity_metrics.iloc[[1, 2, 0]]
        display(propensity_metrics)

        save_dataframe_html(
            propensity_metrics,
            "7_propensity_baseline_metrics",
            "4_modeling_results",
            self._run_id,
        )

        prop_pred = propensity_model.predict(
            self._df.loc[mask_test], return_df=False
        )
        uplift_pred = best_uplift_model.predict(
            self._df.loc[mask_test], return_df=False
        )

        uplift_pred_norm = (uplift_pred - uplift_pred.min()) / (
            uplift_pred.max() - uplift_pred.min()
        )
        uplift_auc_on_control = roc_auc_score(y_test, uplift_pred_norm)

        correlations_df = pd.DataFrame(
            [
                {
                    "method": "pearson",
                    "coefficient": np.corrcoef(prop_pred, uplift_pred)[0, 1],
                },
                {
                    "method": "spearman",
                    "coefficient": ss.spearmanr(
                        prop_pred, uplift_pred
                    ).statistic,
                },
                {
                    "method": "kendall",
                    "coefficient": ss.kendalltau(
                        prop_pred, uplift_pred
                    ).statistic,
                },
            ]
        )

        display(correlations_df)
        
        save_dataframe_html(
            correlations_df,
            "7_propensity_uplift_correlations",
            "4_modeling_results",
            self._run_id,
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(prop_pred, uplift_pred, alpha=0.6, s=20)
        ax.set_xlabel("Propensity score (control model)")
        ax.set_ylabel("Uplift score (best uplift model)")
        ax.set_title("Propensity vs. Uplift (control test subset)")
        plt.grid(True, linestyle="--", alpha=0.5)

        save_figure(
            fig,
            "7_propensity_vs_uplift_scatter",
            "4_modeling_results",
            self._run_id,
        )

        result = {
            "uplift_auc_on_control": uplift_auc_on_control,
            "correlations": correlations_df.to_dict(orient="records"),
        }

        self.logger.info(
            "Propensity vs. uplift comparison completed. "
            f"ROC AUC of uplift model on control group: {uplift_auc_on_control:.4f}"
        )
        self.logger.info(
            "Correlation coefficients: "
            + ", ".join(
                f"{row['method']}={row['coefficient']:.3f}"
                for _, row in correlations_df.iterrows()
            )
        )

        save_json(
            result,
            "7_compare_with_propensity_result",
            "4_modeling_results",
            self._run_id,
        )

    def get_preprocessor(
        self,
        numerical_method: str = "min",
        max_categories: int = 4,
        encoder_type: str = "target",
    ):
        """Create and fit a Preprocessor instance.

        Args:
            numerical_method (str): Strategy for filling missing numeric
                values.
            max_categories (int): Maximum number of top categories to
                keep.
            encoder_type (str): Encoding method for categorical features.

        Returns:
            Preprocessor: Fitted preprocessing object.
        """
        segm_col = self._base_cols_mapper["segm"]
        target_col = self._base_cols_mapper["target"]
        df_train_mask = self._df[segm_col] == "train"
        self._feature_cols = self._get_available_features()

        self._preprocessor = Preprocessor(
            num_fill_strategy=numerical_method,
            cat_fill_value="__none__",
            encoding_method=encoder_type,
            max_top_categories_cnt=max_categories,
            other_category="__other__",
        )
        self._preprocessor.fit(
            X=self._df.loc[df_train_mask, self._feature_cols],
            features=self._feature_cols,
            y=self._df.loc[df_train_mask, target_col],
        )
        return self._preprocessor

    def get_calibrator(self, model: AufModel, bins: int = 10):
        """Fit an UpliftCalibrator on the validation set.

        Args:
            model: Trained uplift model to calibrate.
            bins: Number of probability bins used during calibration.

        Returns:
            UpliftCalibrator: Fitted calibrator instance.
        """
        self._calibrator = UpliftCalibrator()
        segm_col = self._base_cols_mapper["segm"]
        df_val_mask = self._df[segm_col] == "val"
        self._calibrator.fit(
            self._df.loc[df_val_mask],
            model,
            self._base_cols_mapper,
            self._treatment_groups_mapper,
            bins=bins,
        )
        return self._calibrator

    def run(
        self,
        n_propensity_features: int = 50,
        max_val_roc_auc_treatment: float = 0.55,
        early_stopping: int = 10,
        n_features_candidates: int = 200,
        max_abs_feature_correlation: float = 0.95,
        classes_for_train: tp.List[str] = [
            "SoloModel",
            "TwoModels",
            "AufXLearner",
        ],
        feature_nums: tp.Union[tp.List[int], tp.Dict[str, tp.List[int]]] = [
            20,
            35,
            50,
            100,
        ],
        timeout_estimator: tp.Union[int, tp.Dict[str, int]] = 60 * 3,
        training_mode: tp.Literal["light", "medium", "hard"] = "light",
        opt_metric: str = "qini_auc",
        n_min_features: int = 5,
        n_max_features: int = None,
        n_uplift_bins: int = 10,
    ):
        """Execute the full uplift‑modeling pipeline (except data loading).

        Args:
            n_propensity_features: Number of feature to select for training
                propensity baseline model.
            max_val_roc_auc_treatment: Upper bound for ROC‑AUC when checking
                treatment leaks.
            early_stopping: Early stopping patience for the CatBoost model
                detection of treatment leak.
            n_features_candidates: Number of top features to keep
                after the fast pre‑selection step.
            max_abs_feature_correlation: Correlation threshold
                for feature removal.
            classes_for_train: List of model class names to
                train.
            feature_nums: Either a list
                of feature counts or a dict mapping model names to lists.
            timeout_estimator: Maximum training
                time (seconds) per estimator.
            training_mode: Search
                intensity for Optuna.
            opt_metric: Metric identifier from the global METRICS
                dict used for model selection and optuna search.
            n_min_features: Minimum number of features a model may
                use.
            n_max_features: Maximum number of features a
                model may use. If None all available features are allowed.
            n_uplift_bins: Number of bins for percentile‑based
                visualisations.

        Returns:
            tuple: (preprocessor, best_auf_model, calibrator) for binary
                treatment pipelines; for multi‑treatment pipelines returns
                (preprocessor, best_auf_model).

        Raises:
            AssertionError: If any of the validation checks fails.
        """
        run_parameters = {
            "n_propensity_features": n_propensity_features,
            "max_val_roc_auc_treatment": max_val_roc_auc_treatment,
            "early_stopping": early_stopping,
            "n_features_candidates": n_features_candidates,
            "max_abs_feature_correlation": max_abs_feature_correlation,
            "classes_for_train": classes_for_train,
            "feature_nums": feature_nums,
            "timeout_estimator": timeout_estimator,
            "training_mode": training_mode,
            "opt_metric": opt_metric,
            "n_min_features": n_min_features,
            "n_max_features": n_max_features,
            "n_uplift_bins": n_uplift_bins,
        }
        save_json(
            run_parameters, "parameters", "0_pipeline_setting", self._run_id
        )

        self._use_default_run = True

        self.logger.info(
            "Start with cleaning feature list: remove leaks, unimportant features and so on."
        )

        self.check_treatment_leaks(
            max_val_roc_auc_treatment=max_val_roc_auc_treatment,
            early_stopping=early_stopping,
        )
        self.logger.info(
            f"Number of features after cleaning: {len(self._get_available_features())}"
        )

        if self._use_multitreatment:
            self.preselect_features_candidates(n_features_candidates)
        else:
            self.preselect_features_candidates(
                n_features_candidates, "importance"
            )

        self.check_correlated_features(
            max_abs_feature_correlation, check_only_available_features=True
        )

        self.show_removed_features_with_reasons()

        self.logger.info(
            "Rank filtered feature list by different kinds of importance."
        )

        self.rank_features_candidates()

        self.logger.info(
            "Train models using different number of top features from every sort method."
        )

        opt_metric = METRICS[opt_metric]

        if n_max_features is None:
            n_max_features = len(self._get_available_features())

        if isinstance(feature_nums, list):
            feature_nums = sorted(
                [
                    x
                    for x in feature_nums
                    if n_min_features <= x <= n_max_features
                ]
            )
        elif isinstance(feature_nums, dict):
            for model_name in feature_nums:
                feature_nums[model_name] = sorted(
                    [
                        x
                        for x in feature_nums[model_name]
                        if n_min_features <= x <= n_max_features
                    ]
                )

        self.train_models(
            classes=classes_for_train,
            features=None,
            feature_nums=feature_nums,
            metric=opt_metric,
            timeout_estimator=timeout_estimator,
            training_mode=training_mode,
        )

        if not self._use_multitreatment:
            self.logger.info("Train propensity baseline.")
            self.train_propensity_baseline(
                features=None,
                n_propensity_features=n_propensity_features,
                metric=roc_auc_score,
                timeout_estimator=(
                    timeout_estimator
                    if isinstance(timeout_estimator, int)
                    else max(timeout_estimator.values())
                ),
                search_class=OptunaOptimizer,
                training_mode=training_mode,
            )

        self.logger.info("Find the best model.")

        model_class_name, ranker_method, best_result = self.get_result(
            metric=opt_metric, n_max_features=n_max_features, rating=0
        )
        (
            best_model,
            best_model_name,
            best_ranker_name,
            best_n_features,
        ) = (
            best_result.auf_model._model,
            model_class_name,
            ranker_method,
            len(best_result.auf_model._features),
        )

        if not self._use_multitreatment:
            best_uplift_type = best_result.auf_model._uplift_prediction_type

        self.logger.info("Best model description:")
        self.logger.info(f"{'':<4}{'feature ranker':<21}: {best_ranker_name}")
        self.logger.info(f"{'':<4}{'features count':<21}: {best_n_features}")
        self.logger.info(f"{'':<4}{'model class':<21}: {best_model_name}")

        if not self._use_multitreatment:
            self.logger.info(f"{'':<4}{'uplift type':<21}: {best_uplift_type}")

        best_auf_model = best_result.auf_model
        best_auf_model_wrapped = MlflowWrapper(
            self._preprocessor, best_auf_model
        )

        # need to keep only needed features for best model
        self._preprocessor.keep_features(best_result.auf_model._features)
        save_pickle(
            self._preprocessor,
            "preprocessor",
            "5_best_model_artifacts",
            self._run_id,
        )

        save_model(
            best_auf_model_wrapped,
            "5_best_model_artifacts",
            self._run_id,
            self._experiment_name,
        )

        if not self._use_multitreatment:
            if best_model_name == "TwoModels":
                ctrl_params = best_model.estimator_ctrl.get_params()
                trmnt_params = best_model.estimator_trmnt.get_params()
                self._modify_catboost_params_dict(ctrl_params)
                self._modify_catboost_params_dict(trmnt_params)
                self.logger.info(f"{'':<4}control model parameters:")
                self.logger.info(
                    f"{'':<8}{'iterations':<17}: {ctrl_params['iterations']}"
                )
                self.logger.info(
                    f"{'':<8}{'max_depth':<17}: {ctrl_params['max_depth']}"
                )
                self.logger.info(
                    f"{'':<8}{'learning_rate':<17}: {ctrl_params['learning_rate']}"
                )
                self.logger.info(f"{'':<4}treatment model parameters:")
                self.logger.info(
                    f"{'':<8}{'iterations':<17}: {trmnt_params['iterations']}"
                )
                self.logger.info(
                    f"{'':<8}{'max_depth':<17}: {trmnt_params['max_depth']}"
                )
                self.logger.info(
                    f"{'':<8}{'learning_rate':<17}: {trmnt_params['learning_rate']}"
                )
            elif best_model_name == "SoloModel":
                params = best_model.estimator.get_params()
                self._modify_catboost_params_dict(params)
                self.logger.info(f"{'':<4}model parameters:")
                self.logger.info(
                    f"{'':<8}{'iterations':<17}: {params['iterations']}"
                )
                self.logger.info(
                    f"{'':<8}{'max_depth':<17}: {params['max_depth']}"
                )
                if "learning_rate" in params:
                    self.logger.info(
                        f"{'':<8}{'learning_rate':<17}: {params['learning_rate']}"
                    )
                else:
                    self.logger.info(
                        f"{'':<8}{'evaluationFunction':<17}{params['evaluationFunction']}"
                    )
            elif best_model_name == "AufXLearner":
                params = best_model.get_params()
                model_params = params["model"].get_params()
                uplift_model_params = params["uplift_model"].get_params()
                group_model_params = params["group_model"].get_params()

                self._modify_catboost_params_dict(model_params)
                self._modify_catboost_params_dict(uplift_model_params)
                self._modify_catboost_params_dict(group_model_params)

                self.logger.info(f"{'':<4}(1 step) model parameters:")
                self.logger.info(
                    f"{'':<8}{'iterations':<17}: {model_params['iterations']}"
                )
                self.logger.info(
                    f"{'':<8}{'max_depth':<17}: {model_params['max_depth']}"
                )
                self.logger.info(
                    f"{'':<8}{'learning_rate':<17}: {model_params['learning_rate']}"
                )

                self.logger.info(f"{'':<4}(2 step) uplift model parameters:")
                self.logger.info(
                    f"{'':<8}{'iterations':<17}: {uplift_model_params['iterations']}"
                )
                self.logger.info(
                    f"{'':<8}{'max_depth':<17}: {uplift_model_params['max_depth']}"
                )
                self.logger.info(
                    f"{'':<8}{'learning_rate':<17}: {uplift_model_params['learning_rate']}"
                )

                self.logger.info(f"{'':<4}(3 step) group model parameters:")
                self.logger.info(
                    f"{'':<8}{'iterations':<17}: {group_model_params['iterations']}"
                )
                self.logger.info(
                    f"{'':<8}{'max_depth':<17}: {group_model_params['max_depth']}"
                )
                self.logger.info(
                    f"{'':<8}{'learning_rate':<17}: {group_model_params['learning_rate']}"
                )
            else:
                params = best_model.get_params()
                self.logger.info(f"{'':<4}model parameters:")
                for name, value in params.items():
                    self.logger.info(f"{'':<8}{name}{'':<4}{value}")

        if self._use_multitreatment:
            params = dict()
        else:
            params = best_model.get_params()
            params["best_uplift_type"] = best_uplift_type

        params["best_model_name"] = best_model_name
        params["best_ranker_name"] = best_ranker_name
        params["best_n_features"] = best_n_features
        save_params_dict(params, self._run_id)

        self.logger.info("Best model quality table:")
        full_metrics_df = self.show_metrics_table(
            metrics_names=[
                "uplift@10",
                "uplift_rel@10",
                "uplift@15",
                "uplift_rel@15",
                "uplift@20",
                "uplift_rel@20",
                "qini_auc",
                "qini_clipped@20",
            ]
        )
        self._full_metrics_df = full_metrics_df
        save_dataframe_html(
            full_metrics_df,
            "1_all_models_all_metrics",
            "4_modeling_results",
            self._run_id,
        )

        self.plot_feature_importances(auf_model=best_auf_model)

        self.plot_results(
            full_metrics_df,
            model_class_name,
            ranker_method,
            best_auf_model,
            n_uplift_bins,
        )

        if not self._use_multitreatment:
            self.compare_with_propensity_baseline(
                full_metrics_df,
                best_auf_model,
                [
                    "uplift@10",
                    "uplift_rel@10",
                    "uplift@15",
                    "uplift_rel@15",
                    "uplift@20",
                    "uplift_rel@20",
                    "qini_auc",
                    "qini_clipped@20",
                ],
            )

        features = self._ranked_candidates[best_ranker_name][:best_n_features]
        save_json(
            features, "model_features", "5_best_model_artifacts", self._run_id
        )

        figure_name = os.path.join(
            "mlflow_artifacts", "5_uplift_by_feature_bins_top_10_features.pdf"
        )
        os.makedirs(os.path.dirname(figure_name), exist_ok=True)

        self._df = self._preprocessor.inversed_transform(self._df, inplace=True)

        with PdfPages(figure_name) as pdf:
            for i, f in enumerate(features[:10]):
                plot_uplift_by_feature_bins(
                    self._df[f],
                    self._df[self._base_cols_mapper["treatment"]].map(
                        self._treatment_groups_mapper
                    ),
                    self._df[self._base_cols_mapper["target"]],
                    f"{self._feature_names.get(f, f)}",
                    amount_of_bins=6,
                )

                plt.tight_layout()
                pdf.savefig()
                plt.show()
        save_pdf_figures(figure_name, "4_modeling_results", self._run_id)

        # for uplift prediction by best_auf_model
        self._df = self._preprocessor.transform(self._df)

        fig, axes = plt.subplots(1, 1, figsize=(12, 8))
        test_mask = self._df[self._base_cols_mapper["segm"]] == "test"

        if self._use_multitreatment:
            best_auf_model_preds = best_auf_model.predict(
                self._df.loc[test_mask, best_result.auf_model._features]
            )
        else:
            best_auf_model_preds = best_auf_model.predict(
                self._df.loc[test_mask, best_result.auf_model._features],
                return_df=False,
            )

        plot_portrait_tree(
            x=self._df.loc[test_mask, best_result.auf_model._features],
            uplift=best_auf_model_preds,
            feature_names_dict=self._feature_names
            if self._feature_names
            else None,
            max_depth=2,
            axes=axes,
        )
        plt.tight_layout()
        save_figure(
            fig, "6_client_portrait_tree", "4_modeling_results", self._run_id
        )
        plt.show()

        self._use_default_run = False

        if not self._use_multitreatment:
            self.get_calibrator(best_auf_model, bins=10)
            save_pickle(
                self._calibrator,
                "calibrator",
                "5_best_model_artifacts",
                self._run_id,
            )
            return self._preprocessor, best_auf_model, self._calibrator

        return self._preprocessor, best_auf_model
