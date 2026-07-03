from __future__ import annotations

import argparse
import logging
import pickle
import sys
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import skew
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedGroupKFold,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("credit_score_pipeline")


# Configure
# ==============================================================================
@dataclass(frozen=True)
class FeatureConfig:
    # Definisi kolom, aturan validasi domain, dan fitur rekayasa (hasil EDA v3)

    target_col: str = "Credit_Score"
    group_col: str = "Customer_ID"

    id_cols: Tuple[str, ...] = ("ID", "Customer_ID", "SSN", "Name", "Month")

    numeric_text_cols: Tuple[str, ...] = (
        "Age", "Annual_Income", "Num_of_Loan", "Num_of_Delayed_Payment",
        "Changed_Credit_Limit", "Outstanding_Debt", "Amount_invested_monthly",
        "Monthly_Balance",
    )

    numeric_impute_cols: Tuple[str, ...] = (
        "Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
        "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", "Num_of_Delayed_Payment",
        "Changed_Credit_Limit", "Num_Credit_Inquiries", "Outstanding_Debt",
        "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance",
        "Credit_History_Age_Months",
    )

    categorical_impute_cols: Tuple[str, ...] = ("Occupation", "Credit_Mix", "Payment_Behaviour")

    valid_ranges: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "Age": (14, 100),
        "Annual_Income": (0, 2_000_000),
        "Num_Bank_Accounts": (0, 20),
        "Num_Credit_Card": (0, 20),
        "Interest_Rate": (0, 40),
        "Num_of_Loan": (0, 15),
        "Num_of_Delayed_Payment": (0, 60),
        "Num_Credit_Inquiries": (0, 50),
        "Delay_from_due_date": (-10, 100),
        "Total_EMI_per_month": (0, 10000),
        "Amount_invested_monthly": (0, 9999),
    })

    categorical_placeholder_map: Dict[str, str] = field(default_factory=lambda: {
        "Occupation": "_______",
        "Credit_Mix": "_",
        "Payment_Behaviour": "!@9#%8",
    })

    # Fitur interaksi domain-spesifik 
    engineered_ratio_cols: Tuple[str, ...] = (
        "Debt_to_Income", "Credit_Utilization_Intensity", "Free_Cash_Flow",
    )
    debt_to_income_clip_max: float = 50.0

    class_order: Tuple[str, ...] = ("Poor", "Standard", "Good")
    minority_class: str = "Good"

    @property
    def numeric_features(self) -> List[str]:
        return (
            list(self.numeric_impute_cols)
            + ["Num_Loan_Types", "Credit_Utilization_Ratio"]
            + list(self.engineered_ratio_cols)
        )

    @property
    def categorical_features(self) -> List[str]:
        return list(self.categorical_impute_cols) + ["Payment_of_Min_Amount"]


@dataclass(frozen=True)
class TrainingConfig:
    # Konfigurasi proses training/eksperimen (bukan domain data)

    random_state: int = 42
    test_size: float = 0.2
    n_cv_splits: int = 3
    n_iter_search: int = 10
    primary_metric: str = "f1_macro"


@dataclass(frozen=True)
class MLflowConfig:
    tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "credit_score_experiment"
    registered_model_name: str = "credit_score_classifier"


# Data Cleaning Utility (stateless)
# ==============================================================================
class DataCleaningUtils:
    # stateless static methods

    @staticmethod
    def to_numeric_clean(series: pd.Series) -> pd.Series:
        return pd.to_numeric(
            series.astype(str).str.replace("_", "", regex=False).str.strip(),
            errors="coerce",
        )

    @staticmethod
    def parse_credit_history_age(series: pd.Series) -> pd.Series:
        extracted = series.astype(str).str.extract(r"(\d+)\s*Years?\s*and\s*(\d+)\s*Months?")
        years = pd.to_numeric(extracted[0], errors="coerce")
        months = pd.to_numeric(extracted[1], errors="coerce")
        return years * 12 + months

    @staticmethod
    def count_loan_types(series: pd.Series) -> pd.Series:
        s = series.astype(str)
        is_specified = series.notna() & (s.str.strip().str.lower() != "not specified")
        counts = s.str.count(",") + 1
        return pd.Series(np.where(is_specified, counts, 0), index=series.index)

    @staticmethod
    def mode_or_nan(s: pd.Series) -> Any:
        m = s.mode(dropna=True)
        return m.iloc[0] if len(m) else np.nan


# Custom Transformer: CreditDataCleaner
# ==============================================================================
class CreditDataCleaner(BaseEstimator, TransformerMixin):

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config

    def _resolve_config(self) -> FeatureConfig:
        return self.config if self.config is not None else FeatureConfig()

    def _basic_clean(self, X: pd.DataFrame) -> pd.DataFrame:
        cfg = self._resolve_config()
        df = X.copy()

        for col in cfg.numeric_text_cols:
            df[col] = DataCleaningUtils.to_numeric_clean(df[col])

        for col, (lo, hi) in cfg.valid_ranges.items():
            df.loc[~df[col].between(lo, hi), col] = np.nan

        for col, placeholder in cfg.categorical_placeholder_map.items():
            df[col] = df[col].replace(placeholder, np.nan)

        df["Credit_History_Age_Months"] = DataCleaningUtils.parse_credit_history_age(
            df["Credit_History_Age"]
        )
        df["Num_Loan_Types"] = DataCleaningUtils.count_loan_types(df["Type_of_Loan"])
        df = df.drop(columns=["Credit_History_Age", "Type_of_Loan"])
        return df

    def _add_engineered_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self._resolve_config()
        df["Debt_to_Income"] = (
            df["Outstanding_Debt"] / (df["Annual_Income"] + 1.0)
        ).clip(upper=cfg.debt_to_income_clip_max)
        df["Credit_Utilization_Intensity"] = df["Credit_Utilization_Ratio"] * df["Num_Credit_Card"]
        df["Free_Cash_Flow"] = (
            df["Monthly_Inhand_Salary"] - df["Total_EMI_per_month"] - df["Amount_invested_monthly"]
        )
        return df

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "CreditDataCleaner":
        cfg = self._resolve_config()
        df = self._basic_clean(X)

        self.global_medians_ = {c: df[c].median() for c in cfg.numeric_impute_cols}
        self.global_modes_ = {c: DataCleaningUtils.mode_or_nan(df[c]) for c in cfg.categorical_impute_cols}
        self.feature_columns_ = cfg.numeric_features + cfg.categorical_features
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cfg = self._resolve_config()
        df = self._basic_clean(X)
        has_group = cfg.group_col in df.columns

        for col in cfg.numeric_impute_cols:
            if has_group:
                grp_median = df.groupby(cfg.group_col)[col].transform("median")
                df[col] = df[col].fillna(grp_median)
            df[col] = df[col].fillna(self.global_medians_[col])

        for col in cfg.categorical_impute_cols:
            if has_group:
                df[col] = df.groupby(cfg.group_col)[col].transform(
                    lambda s: s.fillna(DataCleaningUtils.mode_or_nan(s))
                )
            df[col] = df[col].fillna(self.global_modes_[col])

        df = self._add_engineered_ratios(df)

        drop_cols = [c for c in cfg.id_cols if c in df.columns]
        df = df.drop(columns=drop_cols)
        return df[self.feature_columns_]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(self.feature_columns_)



# Preprocess
# ==============================================================================
class PreprocessorFactory:
    # Imputasi pengaman + `RobustScaler` (numerik) + `OneHotEncoder` (kategorikal).

    @staticmethod
    def build(config: FeatureConfig) -> ColumnTransformer:
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ])
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        return ColumnTransformer(transformers=[
            ("num", numeric_transformer, config.numeric_features),
            ("cat", categorical_transformer, config.categorical_features),
        ])


# XGBoost Wrapper (label string in/out, API konsisten dgn model sklearn lain)
# ==============================================================================
class XGBWithLabelEncoding(BaseEstimator, ClassifierMixin):

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.1,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBWithLabelEncoding":
        self.label_encoder_ = LabelEncoder()
        y_enc = self.label_encoder_.fit_transform(y)
        self.model_ = XGBClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.learning_rate, subsample=self.subsample,
            colsample_bytree=self.colsample_bytree, random_state=self.random_state,
            n_jobs=1, eval_metric="mlogloss",
        )
        self.model_.fit(X, y_enc)
        self.classes_ = self.label_encoder_.classes_
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.label_encoder_.inverse_transform(self.model_.predict(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.model_.feature_importances_


# Model Specification (Abstraction + Inheritance + Polymorphism)
# ==============================================================================
class BaseModelSpec(ABC):

    supports_class_weight: bool = False

    def __init__(self, random_state: int):
        self._random_state = random_state

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def build_estimator(self, use_class_weight: bool = True, **params: Any) -> BaseEstimator: ...

    @abstractmethod
    def param_distributions(self) -> Dict[str, List[Any]]: ...


class LogisticRegressionSpec(BaseModelSpec):
    name = "Logistic Regression"  # type: ignore[assignment]
    supports_class_weight = True

    def build_estimator(self, use_class_weight: bool = True, **params: Any) -> BaseEstimator:
        defaults: Dict[str, Any] = dict(max_iter=2000, random_state=self._random_state)
        if use_class_weight:
            defaults["class_weight"] = "balanced"
        defaults.update(params)
        return LogisticRegression(**defaults)

    def param_distributions(self) -> Dict[str, List[Any]]:
        return {}


class KNearestNeighborsSpec(BaseModelSpec):
    name = "K-Nearest Neighbors"  # type: ignore[assignment]

    def build_estimator(self, use_class_weight: bool = True, **params: Any) -> BaseEstimator:
        defaults: Dict[str, Any] = dict(n_neighbors=15)
        defaults.update(params)
        return KNeighborsClassifier(**defaults)

    def param_distributions(self) -> Dict[str, List[Any]]:
        return {}


class DecisionTreeSpec(BaseModelSpec):
    name = "Decision Tree"  # type: ignore[assignment]
    supports_class_weight = True

    def build_estimator(self, use_class_weight: bool = True, **params: Any) -> BaseEstimator:
        defaults: Dict[str, Any] = dict(max_depth=12, random_state=self._random_state)
        if use_class_weight:
            defaults["class_weight"] = "balanced"
        defaults.update(params)
        return DecisionTreeClassifier(**defaults)

    def param_distributions(self) -> Dict[str, List[Any]]:
        return {}


class RandomForestSpec(BaseModelSpec):
    name = "Random Forest"  # type: ignore[assignment]
    supports_class_weight = True

    def build_estimator(self, use_class_weight: bool = True, **params: Any) -> BaseEstimator:
        defaults: Dict[str, Any] = dict(
            n_estimators=100, max_depth=14, n_jobs=1, random_state=self._random_state,
        )
        if use_class_weight:
            defaults["class_weight"] = "balanced"
        defaults.update(params)
        return RandomForestClassifier(**defaults)

    def param_distributions(self) -> Dict[str, List[Any]]:
        return {
            "classifier__n_estimators": [100, 150, 200, 260],
            "classifier__max_depth": [10, 16, 22, 28, None],
            "classifier__min_samples_split": [2, 5, 10, 15],
            "classifier__min_samples_leaf": [1, 2, 4, 6],
            "classifier__max_features": ["sqrt", "log2", 0.5],
        }


class GradientBoostingSpec(BaseModelSpec):
    name = "Gradient Boosting"  # type: ignore[assignment]

    def build_estimator(self, use_class_weight: bool = True, **params: Any) -> BaseEstimator:
        defaults: Dict[str, Any] = dict(n_estimators=70, max_depth=3, random_state=self._random_state)
        defaults.update(params)
        return GradientBoostingClassifier(**defaults)

    def param_distributions(self) -> Dict[str, List[Any]]:
        return {
            "classifier__n_estimators": [60, 100, 140, 180],
            "classifier__max_depth": [2, 3, 4, 5],
            "classifier__learning_rate": [0.02, 0.05, 0.1, 0.15],
            "classifier__subsample": [0.7, 0.85, 1.0],
        }


class XGBoostSpec(BaseModelSpec):
    name = "XGBoost"  # type: ignore[assignment]

    def build_estimator(self, use_class_weight: bool = True, **params: Any) -> BaseEstimator:
        defaults: Dict[str, Any] = dict(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=self._random_state,
        )
        defaults.update(params)
        return XGBWithLabelEncoding(**defaults)

    def param_distributions(self) -> Dict[str, List[Any]]:
        return {
            "classifier__n_estimators": [80, 120, 160, 220],
            "classifier__max_depth": [3, 4, 5, 6],
            "classifier__learning_rate": [0.02, 0.05, 0.1, 0.2],
            "classifier__subsample": [0.7, 0.85, 1.0],
            "classifier__colsample_bytree": [0.7, 0.85, 1.0],
        }


class ModelRegistry:
    # Daftar pusat kandidat algoritma (Open/Closed Principle)

    _SPEC_CLASSES = (
        LogisticRegressionSpec,
        KNearestNeighborsSpec,
        DecisionTreeSpec,
        RandomForestSpec,
        GradientBoostingSpec,
        XGBoostSpec,
    )

    def __init__(self, random_state: int):
        self._specs = [cls(random_state) for cls in self._SPEC_CLASSES]

    def all(self) -> List[BaseModelSpec]:
        return list(self._specs)

    def get(self, name: str) -> BaseModelSpec:
        for spec in self._specs:
            if spec.name == name:
                return spec
        raise KeyError(f"Model '{name}' tidak terdaftar di ModelRegistry.")


# Pipeline (cleaner + preprocessor + [smote] + classifier)
# ==============================================================================
class CreditPipelineFactory:
    """Merangkai `CreditDataCleaner` + `ColumnTransformer` + classifier.

    `build()` menghasilkan `sklearn.pipeline.Pipeline` biasa.
    `build_with_smote()` menghasilkan `imblearn.pipeline.Pipeline`, di mana
    tahap SMOTE hanya aktif saat `.fit()` (di setiap fold/di data latih) dan
    otomatis dilewati saat `.predict()`/`.transform()`, sehingga oversampling
    tidak pernah bocor ke data validasi maupun data uji.
    """

    def __init__(self, feature_config: FeatureConfig):
        self._feature_config = feature_config

    def build(self, estimator: BaseEstimator) -> Pipeline:
        return Pipeline(steps=[
            ("cleaner", CreditDataCleaner(config=self._feature_config)),
            ("preprocessor", PreprocessorFactory.build(self._feature_config)),
            ("classifier", estimator),
        ])

    def build_with_smote(self, estimator: BaseEstimator, random_state: int = 42) -> ImbPipeline:
        return ImbPipeline(steps=[
            ("cleaner", CreditDataCleaner(config=self._feature_config)),
            ("preprocessor", PreprocessorFactory.build(self._feature_config)),
            ("smote", SMOTE(random_state=random_state)),
            ("classifier", estimator),
        ])


# MLflow Wrapper
# ==============================================================================
class MLflowExperimentTracker:
    # Encapsulation seluruh interaksi dengan MLflow

    def __init__(self, config: MLflowConfig):
        self._config = config
        mlflow.set_tracking_uri(config.tracking_uri)
        mlflow.set_experiment(config.experiment_name)

    def start_run(self, run_name: str, nested: bool = False, tags: Optional[Dict[str, str]] = None):
        return mlflow.start_run(run_name=run_name, nested=nested, tags=tags or {})

    @staticmethod
    def log_params(params: Dict[str, Any]) -> None:
        safe_params = {k: str(v)[:250] for k, v in params.items()}
        mlflow.log_params(safe_params)

    @staticmethod
    def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)

    @staticmethod
    def log_text(text: str, artifact_file: str) -> None:
        mlflow.log_text(text, artifact_file)

    @staticmethod
    def log_dict(data: Dict[str, Any], artifact_file: str) -> None:
        mlflow.log_dict(data, artifact_file)

    @staticmethod
    def log_sklearn_model(
        model: Pipeline,
        artifact_path: str,
        registered_model_name: Optional[str] = None,
        input_example: Optional[pd.DataFrame] = None,
    ) -> None:
        # cloudpickle (bukan default "skops") karena pipeline memuat class
        # custom yang didefinisikan di modul ini (`CreditDataCleaner`,
        # `XGBWithLabelEncoding`) yang tidak dikenal skops.
        mlflow.sklearn.log_model(
            sk_model=model,
            name=artifact_path,
            registered_model_name=registered_model_name,
            input_example=input_example,
            serialization_format="cloudpickle",
        )

    def end_run(self) -> None:
        mlflow.end_run()


# Laporan
# ==============================================================================
class DataQualityReporter:

    def __init__(self, feature_config: FeatureConfig):
        self._fc = feature_config

    def missingness_report(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = df.isnull().sum()
        pct = (missing / len(df) * 100).round(2)
        return pd.DataFrame({"missing_count": missing, "missing_pct": pct}).query("missing_count > 0")

    def skewness_report(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = ["Annual_Income", "Outstanding_Debt", "Total_EMI_per_month",
                "Amount_invested_monthly", "Monthly_Balance", "Monthly_Inhand_Salary"]
        rows = []
        for c in cols:
            cleaned = DataCleaningUtils.to_numeric_clean(df[c]) if df[c].dtype == object else df[c]
            rows.append({"feature": c, "skewness": float(skew(cleaned.dropna()))})
        return pd.DataFrame(rows).sort_values("skewness", ascending=False)


# Data Splitting
# ==============================================================================
class CustomerAwareSplitter:
    """Membagi data latih/uji pada level nasabah (`Customer_ID`), bukan baris."""

    def __init__(self, feature_config: FeatureConfig, training_config: TrainingConfig):
        self._fc = feature_config
        self._tc = training_config

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        customer_target = df.groupby(self._fc.group_col)[self._fc.target_col].agg(
            lambda s: s.mode().iloc[0]
        )
        train_customers, _ = train_test_split(
            customer_target.index,
            test_size=self._tc.test_size,
            stratify=customer_target.values,
            random_state=self._tc.random_state,
        )
        train_mask = df[self._fc.group_col].isin(train_customers)

        X = df.drop(columns=[self._fc.target_col])
        y = df[self._fc.target_col]

        X_train, X_test = X[train_mask.values], X[~train_mask.values]
        y_train, y_test = y[train_mask.values], y[~train_mask.values]
        groups_train = X_train[self._fc.group_col]

        logger.info(
            "Split selesai | latih: %s baris / %s nasabah | uji: %s baris / %s nasabah",
            X_train.shape[0], train_customers.nunique(),
            X_test.shape[0], customer_target.index.nunique() - train_customers.nunique(),
        )
        return X_train, X_test, y_train, y_test, groups_train


# Algorithm Comparasion
# ==============================================================================
class ModelExperimentRunner:

    SCORING = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
    }

    def __init__(
        self,
        pipeline_factory: CreditPipelineFactory,
        training_config: TrainingConfig,
        tracker: MLflowExperimentTracker,
    ):
        self._pipeline_factory = pipeline_factory
        self._tc = training_config
        self._tracker = tracker

    def run(
        self,
        specs: List[BaseModelSpec],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        groups_train: pd.Series,
    ) -> pd.DataFrame:
        cv = StratifiedGroupKFold(
            n_splits=self._tc.n_cv_splits, shuffle=True, random_state=self._tc.random_state
        )

        results: List[Dict[str, Any]] = []
        for spec in specs:
            logger.info("Menjalankan cross-validation untuk model: %s", spec.name)
            pipe = self._pipeline_factory.build(spec.build_estimator())

            with self._tracker.start_run(run_name=f"cv_{spec.name}", nested=True, tags={"stage": "model_comparison"}):
                cv_result = cross_validate(
                    pipe, X_train, y_train, cv=cv, groups=groups_train,
                    scoring=self.SCORING, n_jobs=1,
                )
                metrics = {
                    "cv_accuracy_mean": cv_result["test_accuracy"].mean(),
                    "cv_f1_macro_mean": cv_result["test_f1_macro"].mean(),
                    "cv_f1_macro_std": cv_result["test_f1_macro"].std(),
                    "cv_precision_macro_mean": cv_result["test_precision_macro"].mean(),
                    "cv_recall_macro_mean": cv_result["test_recall_macro"].mean(),
                }
                self._tracker.log_params({
                    "model_name": spec.name,
                    "cv_strategy": "StratifiedGroupKFold",
                    "n_splits": self._tc.n_cv_splits,
                })
                self._tracker.log_metrics(metrics)

            results.append({"Model": spec.name, **metrics})
            logger.info("%s -> F1-macro CV: %.4f", spec.name, metrics["cv_f1_macro_mean"])

        results_df = pd.DataFrame(results).sort_values("cv_f1_macro_mean", ascending=False).reset_index(drop=True)
        return results_df


# Hyperparameter Tuning
# ==============================================================================
class HyperparameterTuner:

    def __init__(
        self,
        pipeline_factory: CreditPipelineFactory,
        training_config: TrainingConfig,
        tracker: MLflowExperimentTracker,
    ):
        self._pipeline_factory = pipeline_factory
        self._tc = training_config
        self._tracker = tracker

    def tune(
        self,
        spec: BaseModelSpec,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        groups_train: pd.Series,
    ) -> Tuple[Pipeline, Dict[str, Any]]:
        param_dist = spec.param_distributions()
        pipe = self._pipeline_factory.build(spec.build_estimator())

        if not param_dist:
            logger.info("%s tidak memiliki ruang hyperparameter terdaftar; fit langsung tanpa tuning.", spec.name)
            pipe.fit(X_train, y_train)
            return pipe, {}

        cv = StratifiedGroupKFold(
            n_splits=self._tc.n_cv_splits, shuffle=True, random_state=self._tc.random_state
        )
        search = RandomizedSearchCV(
            pipe,
            param_distributions=param_dist,
            n_iter=self._tc.n_iter_search,
            cv=cv,
            scoring=self._tc.primary_metric,
            n_jobs=1,
            random_state=self._tc.random_state,
            verbose=0,
        )

        with self._tracker.start_run(run_name=f"tuning_{spec.name}", nested=True, tags={"stage": "hyperparameter_tuning"}):
            search.fit(X_train, y_train, groups=groups_train)
            self._tracker.log_params({"model_name": spec.name, **search.best_params_})
            self._tracker.log_metrics({"best_cv_f1_macro": search.best_score_})
            logger.info("Hyperparameter terbaik (%s): %s", spec.name, search.best_params_)
            logger.info("F1-macro CV terbaik: %.4f", search.best_score_)

        return search.best_estimator_, search.best_params_


# Handling Class Imbalance (class_weight vs SMOTE)
# ==============================================================================
class ImbalanceExperimentRunner:

    def __init__(
        self,
        pipeline_factory: CreditPipelineFactory,
        training_config: TrainingConfig,
        feature_config: FeatureConfig,
        tracker: MLflowExperimentTracker,
    ):
        self._pipeline_factory = pipeline_factory
        self._tc = training_config
        self._fc = feature_config
        self._tracker = tracker

    @staticmethod
    def _clean_params(best_params: Dict[str, Any]) -> Dict[str, Any]:
        return {k.replace("classifier__", ""): v for k, v in best_params.items()}

    def run(
        self,
        spec: BaseModelSpec,
        best_params: Dict[str, Any],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        groups_train: pd.Series,
    ) -> Dict[str, Any]:
        clean_params = self._clean_params(best_params)
        cv = StratifiedGroupKFold(
            n_splits=self._tc.n_cv_splits, shuffle=True, random_state=self._tc.random_state
        )

        baseline_estimator = spec.build_estimator(use_class_weight=True, **clean_params)
        baseline_pipe = self._pipeline_factory.build(baseline_estimator)

        smote_estimator = spec.build_estimator(use_class_weight=False, **clean_params)
        smote_pipe = self._pipeline_factory.build_with_smote(
            smote_estimator, random_state=self._tc.random_state
        )

        minority = self._fc.minority_class
        scoring = {
            "f1_macro": "f1_macro",
            "f1_good": lambda est, X, y: f1_score(y, est.predict(X), labels=[minority], average="macro"),
            "precision_good": lambda est, X, y: precision_score(y, est.predict(X), labels=[minority], average="macro", zero_division=0),
            "recall_good": lambda est, X, y: recall_score(y, est.predict(X), labels=[minority], average="macro", zero_division=0),
        }

        results: List[Dict[str, Any]] = []
        for label, pipe in [("baseline_class_weight", baseline_pipe), ("smote", smote_pipe)]:
            logger.info("Menjalankan eksperimen imbalance: %s", label)
            with self._tracker.start_run(run_name=f"imbalance_{label}", nested=True, tags={"stage": "imbalance_handling"}):
                cv_res = cross_validate(
                    pipe, X_train, y_train, cv=cv, groups=groups_train, scoring=scoring, n_jobs=1
                )
                metrics = {
                    "f1_macro": cv_res["test_f1_macro"].mean(),
                    "f1_good": cv_res["test_f1_good"].mean(),
                    "precision_good": cv_res["test_precision_good"].mean(),
                    "recall_good": cv_res["test_recall_good"].mean(),
                }
                self._tracker.log_params({"strategy": label, "model_name": spec.name})
                self._tracker.log_metrics(metrics)
            results.append({"strategy": label, **metrics})
            logger.info(
                "%s -> F1-macro: %.4f | F1-Good: %.4f", label, metrics["f1_macro"], metrics["f1_good"]
            )

        results_df = pd.DataFrame(results)
        best_row = results_df.sort_values("f1_good", ascending=False).iloc[0]
        best_strategy = str(best_row["strategy"])
        logger.info("Strategi imbalance terpilih (berdasarkan F1-Good tertinggi): %s", best_strategy)

        return {"results_df": results_df, "best_strategy": best_strategy}


# Model Evaluation
# ==============================================================================
class ModelEvaluator:

    def __init__(self, feature_config: FeatureConfig, tracker: MLflowExperimentTracker):
        self._fc = feature_config
        self._tracker = tracker

    def evaluate(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        y_pred = model.predict(X_test)
        order = list(self._fc.class_order)

        metrics = {
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_f1_macro": f1_score(y_test, y_pred, average="macro"),
            "test_precision_macro": precision_score(y_test, y_pred, average="macro"),
            "test_recall_macro": recall_score(y_test, y_pred, average="macro"),
        }

        report_text = classification_report(y_test, y_pred, labels=order, target_names=order)
        cm = confusion_matrix(y_test, y_pred, labels=order)
        cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in order], columns=[f"pred_{c}" for c in order])

        self._tracker.log_metrics(metrics)
        self._tracker.log_text(report_text, "classification_report.txt")
        self._tracker.log_text(cm_df.to_csv(), "confusion_matrix.csv")

        logger.info("Evaluasi data uji selesai:")
        for k, v in metrics.items():
            logger.info("  %-22s : %.4f", k, v)
        logger.info("\n%s", report_text)

        return {"metrics": metrics, "classification_report": report_text, "confusion_matrix": cm_df, "y_pred": y_pred}


# Threshold Tuning for Class Minority ("Good")
# ==============================================================================
class MinorityClassThresholdTuner:

    def __init__(self, feature_config: FeatureConfig, tracker: MLflowExperimentTracker):
        self._fc = feature_config
        self._tracker = tracker

    def tune(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        minority = self._fc.minority_class
        y_binary = (y_test == minority).astype(int)
        proba = model.predict_proba(X_test)
        class_idx = list(model.classes_).index(minority)
        proba_minority = proba[:, class_idx]

        precisions, recalls, thresholds = precision_recall_curve(y_binary, proba_minority)
        f1_scores = np.divide(
            2 * precisions * recalls, precisions + recalls,
            out=np.zeros_like(precisions), where=(precisions + recalls) != 0,
        )
        best_idx = int(np.argmax(f1_scores[:-1]))
        best_threshold = float(thresholds[best_idx])

        metrics = {
            f"{minority.lower()}_threshold_best": best_threshold,
            f"{minority.lower()}_threshold_f1": float(f1_scores[best_idx]),
            f"{minority.lower()}_threshold_precision": float(precisions[best_idx]),
            f"{minority.lower()}_threshold_recall": float(recalls[best_idx]),
        }
        self._tracker.log_metrics(metrics)
        logger.info(
            "Threshold terbaik kelas %s: %.3f (F1=%.4f, Precision=%.4f, Recall=%.4f)",
            minority, best_threshold, f1_scores[best_idx], precisions[best_idx], recalls[best_idx],
        )
        return {"best_threshold": best_threshold, **metrics}


# Artifact
# ==============================================================================
class ArtifactManager:

    def __init__(self, feature_config: FeatureConfig):
        self._fc = feature_config

    def build_artifact(
        self,
        model,
        raw_feature_columns: List[str],
        good_class_threshold: float,
    ) -> Dict[str, Any]:
        return {
            "pipeline": model,
            "raw_feature_columns": raw_feature_columns,
            "numeric_features": self._fc.numeric_features,
            "categorical_features": self._fc.categorical_features,
            "target_classes": list(self._fc.class_order),
            "valid_ranges": dict(self._fc.valid_ranges),
            "good_class_threshold": good_class_threshold,
            "minority_class": self._fc.minority_class,
        }

    def save(self, artifact: Dict[str, Any], output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(artifact, f)
        logger.info("Artefak model disimpan di: %s", output_path.resolve())
        return output_path


# Main
# ==============================================================================
class CreditScoreTrainingPipeline:

    def __init__(
        self,
        data_path: Path,
        feature_config: Optional[FeatureConfig] = None,
        training_config: Optional[TrainingConfig] = None,
        mlflow_config: Optional[MLflowConfig] = None,
        output_model_path: Path = Path("models/credit_score_model.pkl"),
    ):
        self.data_path = Path(data_path)
        self.feature_config = feature_config or FeatureConfig()
        self.training_config = training_config or TrainingConfig()
        self.mlflow_config = mlflow_config or MLflowConfig()
        self.output_model_path = Path(output_model_path)

        self.tracker = MLflowExperimentTracker(self.mlflow_config)
        self.pipeline_factory = CreditPipelineFactory(self.feature_config)
        self.splitter = CustomerAwareSplitter(self.feature_config, self.training_config)
        self.experiment_runner = ModelExperimentRunner(self.pipeline_factory, self.training_config, self.tracker)
        self.tuner = HyperparameterTuner(self.pipeline_factory, self.training_config, self.tracker)
        self.imbalance_runner = ImbalanceExperimentRunner(
            self.pipeline_factory, self.training_config, self.feature_config, self.tracker
        )
        self.evaluator = ModelEvaluator(self.feature_config, self.tracker)
        self.threshold_tuner = MinorityClassThresholdTuner(self.feature_config, self.tracker)
        self.data_quality_reporter = DataQualityReporter(self.feature_config)
        self.artifact_manager = ArtifactManager(self.feature_config)

    def load_data(self) -> pd.DataFrame:
        logger.info("Memuat data dari: %s", self.data_path)
        df = pd.read_csv(self.data_path, index_col=0)
        logger.info("Ukuran data mentah: %s", df.shape)
        return df

    def _rebuild_final_estimator(
        self, spec: BaseModelSpec, best_params: Dict[str, Any], best_strategy: str
    ):
        """Membangun ulang pipeline final sesuai strategi imbalance terpilih."""
        clean_params = ImbalanceExperimentRunner._clean_params(best_params)
        if best_strategy == "smote":
            estimator = spec.build_estimator(use_class_weight=False, **clean_params)
            return self.pipeline_factory.build_with_smote(estimator, random_state=self.training_config.random_state)
        estimator = spec.build_estimator(use_class_weight=True, **clean_params)
        return self.pipeline_factory.build(estimator)

    def run(self) -> Dict[str, Any]:
        df_raw = self.load_data()

        # Laporan kualitas data ---
        missing_report = self.data_quality_reporter.missingness_report(df_raw)
        skew_report = self.data_quality_reporter.skewness_report(df_raw)
        logger.info("Fitur finansial paling condong (skewness):\n%s", skew_report.to_string(index=False))

        X_train, X_test, y_train, y_test, groups_train = self.splitter.split(df_raw)

        registry = ModelRegistry(self.training_config.random_state)

        with self.tracker.start_run(run_name="credit_score_training_pipeline", tags={"project": "credit_scoring"}):
            self.tracker.log_params({
                "n_train_rows": X_train.shape[0],
                "n_test_rows": X_test.shape[0],
                "test_size": self.training_config.test_size,
                "cv_splits": self.training_config.n_cv_splits,
                "random_state": self.training_config.random_state,
                "candidate_models": ", ".join(s.name for s in registry.all()),
                "scaler": "RobustScaler",
            })
            self.tracker.log_text(missing_report.to_csv(), "missingness_report.csv")
            self.tracker.log_text(skew_report.to_csv(index=False), "skewness_report.csv")

            # --- Algorithm Comprarasion ---
            comparison_df = self.experiment_runner.run(registry.all(), X_train, y_train, groups_train)
            self.tracker.log_text(comparison_df.to_csv(index=False), "model_comparison.csv")
            logger.info("\n%s", comparison_df.to_string(index=False))

            best_model_name = comparison_df.iloc[0]["Model"]
            best_spec = registry.get(best_model_name)
            logger.info("Model terbaik dari tahap perbandingan: %s", best_model_name)
            self.tracker.log_params({"selected_model": best_model_name})

            # --- Hyperparamater Tuning on Best Model ---
            _, best_params = self.tuner.tune(best_spec, X_train, y_train, groups_train)

            # ---  class_weight vs SMOTE ---
            imbalance_result = self.imbalance_runner.run(best_spec, best_params, X_train, y_train, groups_train)
            self.tracker.log_text(imbalance_result["results_df"].to_csv(index=False), "imbalance_experiment.csv")
            self.tracker.log_params({"imbalance_strategy": imbalance_result["best_strategy"]})

            # --- FInal Model Fitting ---
            final_model = self._rebuild_final_estimator(
                best_spec, best_params, imbalance_result["best_strategy"]
            )
            logger.info("Melatih model final (strategi: %s) pada seluruh data latih...", imbalance_result["best_strategy"])
            final_model.fit(X_train, y_train)

            # --- Evaluation ---
            evaluation = self.evaluator.evaluate(final_model, X_test, y_test)

            # --- Threshold on Class Monority (Good) ---
            threshold_result = self.threshold_tuner.tune(final_model, X_test, y_test)

            # --- Log Model to  MLflow + registry ---
            self.tracker.log_sklearn_model(
                model=final_model,
                artifact_path="model",
                registered_model_name=self.mlflow_config.registered_model_name,
                input_example=X_train.head(3),
            )

            # --- Artifact ---
            artifact = self.artifact_manager.build_artifact(
                model=final_model,
                raw_feature_columns=list(X_train.columns),
                good_class_threshold=threshold_result["best_threshold"],
            )
            saved_path = self.artifact_manager.save(artifact, self.output_model_path)

        return {
            "best_model_name": best_model_name,
            "imbalance_strategy": imbalance_result["best_strategy"],
            "comparison_df": comparison_df,
            "final_model": final_model,
            "evaluation": evaluation,
            "threshold_result": threshold_result,
            "artifact_path": saved_path,
        }


# CLI Entry Point
# ==============================================================================
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training pipeline model Credit Score (v3) dengan MLflow tracking.")
    parser.add_argument("--data-path", type=str, default="data_D.csv", help="Path ke file data_D.csv")
    parser.add_argument("--output-model-path", type=str, default="models/credit_score_model.pkl",
                         help="Path output artefak pickle")
    parser.add_argument("--mlflow-tracking-uri", type=str, default="sqlite:///mlflow.db", help="MLflow tracking URI")
    parser.add_argument("--experiment-name", type=str, default="credit_score_experiment", help="Nama MLflow experiment")
    parser.add_argument("--registered-model-name", type=str, default="credit_score_classifier",
                         help="Nama model pada MLflow Model Registry")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proporsi nasabah untuk data uji")
    parser.add_argument("--cv-splits", type=int, default=3, help="Jumlah fold StratifiedGroupKFold")
    parser.add_argument("--n-iter-search", type=int, default=10, help="Jumlah iterasi RandomizedSearchCV")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    training_config = TrainingConfig(
        random_state=args.random_state,
        test_size=args.test_size,
        n_cv_splits=args.cv_splits,
        n_iter_search=args.n_iter_search,
    )
    mlflow_config = MLflowConfig(
        tracking_uri=args.mlflow_tracking_uri,
        experiment_name=args.experiment_name,
        registered_model_name=args.registered_model_name,
    )

    pipeline = CreditScoreTrainingPipeline(
        data_path=Path(args.data_path),
        training_config=training_config,
        mlflow_config=mlflow_config,
        output_model_path=Path(args.output_model_path),
    )

    try:
        result = pipeline.run()
    except Exception:
        logger.exception("Training pipeline gagal dijalankan.")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("TRAINING SELESAI")
    logger.info("Model terpilih         : %s", result["best_model_name"])
    logger.info("Strategi imbalance      : %s", result["imbalance_strategy"])
    logger.info("F1-macro (data uji)     : %.4f", result["evaluation"]["metrics"]["test_f1_macro"])
    logger.info("Threshold kelas Good    : %.3f", result["threshold_result"]["best_threshold"])
    logger.info("Artefak tersimpan di    : %s", result["artifact_path"].resolve())
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
