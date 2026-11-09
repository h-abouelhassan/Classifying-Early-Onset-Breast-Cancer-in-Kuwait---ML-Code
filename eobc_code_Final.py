"""
EOBC Kuwait reproducible analysis.

Generated from EOBC_Kuwait_Reproducible_Analysis.ipynb.
Patient-level KCR data are not included. Configure EOBCConfig.csv_path to an authorized local copy.
Requires Python 3.11.x and R 4.4.x with the mice package for inferential multiple imputation.
"""



# EOBC Kuwait — Reproducible Analysis


# Analysis overview


# 0) Imports + settings

# 0) Imports + settings
# =========================
import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# Plotting
import matplotlib.pyplot as plt

try:
    from IPython.display import display
except ImportError:
    def display(obj):
        print(obj)

# Core sklearn
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    balanced_accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, brier_score_loss
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance

# Inferential regression
try:
    import statsmodels.api as sm
    from scipy import stats
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# Optional SHAP for ML interpretation
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# =========================

# Environment version checks
# =========================
import sys
import sklearn
import xgboost

print("Python:", sys.version.split()[0])
print("scikit-learn:", sklearn.__version__)
print("XGBoost:", xgboost.__version__)
print("SHAP:", shap.__version__ if HAS_SHAP else "not installed")


# 1) Configuration

# 1) Configuration
# =========================
@dataclass
class EOBCConfig:
    csv_path: str = "early-onset_dta.csv"
    output_dir: str = "eobc_outputs_final"
    target: str = "early_onset45"

    predictors: List[str] = field(default_factory=lambda: [
        "ydod",
        "hospital_grp_num",
        "ethnicity4",
        "gov_num",
        "religion_grp3",
        "edu_level",
        "smoke_bi",
        "job_employed",
        "famhx_bi",
        "comorb_YesNo",
        "behav",
        "topo_cat",
        "stagedx_num",
        "grade_num",
        "morpho3",
    ])

    # Auxiliary variables support imputation only and are excluded
    # from the final predictive models.
    aux_impute_candidates: List[str] = field(default_factory=lambda: [
        "surgery",
        "radio",
        "chemo",
        "hormone",
        "immuno",
    ])

    numeric_cols: List[str] = field(default_factory=lambda: [
        "ydod",
    ])

    # Tumor grade is ordinal; tumor morphology is nominal.
    ordinal_cols: List[str] = field(default_factory=lambda: [
        "grade_num",
    ])

    # Development split retained for final fitted models, sensitivity
    # analyses, model interpretation, and illustrative plots.
    test_size: float = 0.25

    # Inner CV used in the original development split.
    n_splits: int = 5

    n_bootstrap: int = 2000
    min_recall: float = 0.70

    models_to_run: List[str] = field(
        default_factory=lambda: [
            "Logit",
            "RF",
            "SVM_RBF",
            "XGB",
        ]
    )

    # Repeated nested internal validation.
    nested_outer_folds: int = 5
    nested_outer_repeats: int = 5
    nested_inner_folds: int = 3
    nested_seed_start: int = 4001

    # Reduced randomized-search sizes because tuning is repeated
    # independently inside every outer-training fold.
    nested_n_iter_logit: int = 10
    nested_n_iter_rf: int = 12
    nested_n_iter_svm: int = 12
    nested_n_iter_xgb: int = 20

    # Mixed-type MICE/FCS settings for inferential logistic regression.
    mice_m: int = 20
    mice_max_iter: int = 5
    force_recompute_mice: bool = True

    permutation_repeats: int = 30
    shap_sample_size: int = 750


CONFIG = EOBCConfig()
os.makedirs(
    CONFIG.output_dir,
    exist_ok=True,
)

# =========================


# 2) Utility transformers

# 2) Utility transformers
# =========================
def _to_numeric_or_catcodes(s: pd.Series) -> pd.Series:
    """Convert a Series to numeric; otherwise encode categories while preserving NaN."""
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")

    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.notna().any():
        return numeric

    categorical = pd.Categorical(s)
    codes = pd.Series(categorical.codes, index=s.index).astype(float)
    codes[codes < 0] = np.nan
    return codes


def _extract_ordinal_1to3(
    s: pd.Series,
) -> pd.Series:
    """
    Convert tumor-grade codes and labels to ordered numeric values:

        Grade I   -> 1
        Grade II  -> 2
        Grade III -> 3

    Missing, unknown, and unrecognized values remain NaN.
    """
    def convert_grade(value):
        if pd.isna(value):
            return np.nan

        # Handle genuine numeric codes first.
        try:
            numeric = float(value)

            if numeric in {1.0, 2.0, 3.0}:
                return numeric
        except (TypeError, ValueError):
            pass

        text = (
            str(value)
            .strip()
            .lower()
        )

        if text in {
            "",
            "nan",
            "none",
            "unknown",
            "missing",
            "not known",
            "not available",
            "not applicable",
        }:
            return np.nan

        # Check Grade III before Grade II and Grade I because
        # shorter Roman-numeral strings occur inside longer ones.
        if (
            "grade iii" in text
            or "grade 3" in text
            or "poorly differentiated" in text
            or "poor diff" in text
            or text in {"iii", "3"}
        ):
            return 3.0

        if (
            "grade ii" in text
            or "grade 2" in text
            or "moderately differentiated" in text
            or "moderate diff" in text
            or "mod diff" in text
            or text in {"ii", "2"}
        ):
            return 2.0

        if (
            "grade i" in text
            or "grade 1" in text
            or "well differentiated" in text
            or "well diff" in text
            or text in {"i", "1"}
        ):
            return 1.0

        return np.nan

    return s.map(
        convert_grade
    ).astype(float)


class TargetedKNNImputer(BaseEstimator, TransformerMixin):
    """
    Leakage-safe targeted KNN imputation for tumor grade only.

    Tumor morphology is excluded because it is nominal. Auxiliary treatment
    variables are one-hot encoded before entering the KNN distance calculation.
    Auxiliary variables are removed downstream by the ColumnTransformer.
    """

    def __init__(self, target_cols=None, aux_cols=None, k=7):
        self.target_cols = target_cols if target_cols is not None else ["grade_num"]
        self.aux_cols = aux_cols if aux_cols is not None else []
        self.k = k

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()

        self.available_target_cols_ = [
            c for c in self.target_cols if c in X.columns
        ]
        self.available_aux_cols_ = [
            c for c in self.aux_cols if c in X.columns
        ]

        if not self.available_target_cols_:
            self.imputer_ = None
            return self

        Z = pd.DataFrame(index=X.index)

        for c in self.available_target_cols_:
            Z[c] = _extract_ordinal_1to3(X[c])
        for c in self.available_target_cols_:
            observed_values = (
                Z[c]
                .dropna()
                .unique()
            )

            if len(observed_values) == 0:
                raise ValueError(
                    f"Targeted KNN preprocessing converted every "
                    f"observed value of {c} to missing. "
                    "Check the tumor-grade coding and mapping."
                )

            invalid_values = set(
                observed_values
            ).difference(
                {1.0, 2.0, 3.0}
            )

            if invalid_values:
                raise ValueError(
                    f"Unexpected ordered values for {c}: "
                    f"{sorted(invalid_values)}"
                )

        if self.available_aux_cols_:
            aux = X[self.available_aux_cols_].copy()
            aux = aux.where(aux.notna(), "Unknown_or_missing").astype(str)

            self.aux_encoder_ = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            )
            aux_encoded = self.aux_encoder_.fit_transform(aux)
            aux_names = self.aux_encoder_.get_feature_names_out(
                self.available_aux_cols_
            )
            aux_df = pd.DataFrame(
                aux_encoded,
                columns=aux_names,
                index=X.index,
            )
            Z = pd.concat([Z, aux_df], axis=1)
        else:
            self.aux_encoder_ = None

        self.knn_columns_ = Z.columns.tolist()

        for c in Z.columns:
            if Z[c].isna().all():
                Z[c] = 0.0

        self.imputer_ = KNNImputer(
            n_neighbors=self.k,
            weights="distance",
        )
        self.imputer_.fit(Z)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        if getattr(self, "imputer_", None) is None:
            return X

        Z = pd.DataFrame(index=X.index)

        for c in self.available_target_cols_:
            Z[c] = _extract_ordinal_1to3(X[c])

        if self.available_aux_cols_:
            aux = X[self.available_aux_cols_].copy()
            aux = aux.where(aux.notna(), "Unknown_or_missing").astype(str)

            aux_encoded = self.aux_encoder_.transform(aux)
            aux_names = self.aux_encoder_.get_feature_names_out(
                self.available_aux_cols_
            )
            aux_df = pd.DataFrame(
                aux_encoded,
                columns=aux_names,
                index=X.index,
            )
            Z = pd.concat([Z, aux_df], axis=1)

        Z = Z.reindex(columns=self.knn_columns_, fill_value=0.0)

        imputed = pd.DataFrame(
            self.imputer_.transform(Z),
            columns=self.knn_columns_,
            index=X.index,
        )

        for c in self.available_target_cols_:
            X[c] = np.clip(np.rint(imputed[c]), 1, 3)

        # Morphology is untouched and remains nominal.
        return X


class ColumnKeeper(BaseEstimator, TransformerMixin):
    """Keep selected columns and return a DataFrame."""

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        self.columns_ = [
            c for c in self.columns if c in pd.DataFrame(X).columns
        ]
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        return X[self.columns_]


# =========================


# 3) Data loading + checks

# 3) Data loading + checks
# =========================
def load_data(config: EOBCConfig) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    df = pd.read_csv(config.csv_path)
    if config.target not in df.columns:
        raise ValueError(f"Target column {config.target!r} not found.")

    df[config.target] = pd.to_numeric(df[config.target], errors="coerce")
    df = df.dropna(subset=[config.target]).copy()
    df[config.target] = df[config.target].astype(int)

    observed_classes = set(df[config.target].dropna().unique())
    if not observed_classes.issubset({0, 1}):
        raise ValueError(
            "Outcome must be binary 0/1; "
            f"found {sorted(observed_classes)}"
        )

    predictors = [c for c in config.predictors if c in df.columns]
    missing_predictors = sorted(set(config.predictors) - set(predictors))
    if missing_predictors:
        print("Warning: missing configured predictors:", missing_predictors)

    aux_cols = [c for c in config.aux_impute_candidates if c in df.columns]
    all_x_cols = predictors + [c for c in aux_cols if c not in predictors]

    X = df[all_x_cols].copy()
    y = df[config.target].copy()

    missingness = X[predictors].isna().mean().sort_values(ascending=False).rename("missing_fraction")
    missingness.to_csv(os.path.join(config.output_dir, "missingness_predictors.csv"))
    print("Rows/cols:", df.shape)
    print("Outcome counts:\n", y.value_counts())
    print("EOBC prevalence:", round(y.mean() * 100, 2), "%")
    print("Predictors used:", predictors)
    print("Auxiliary imputation columns only:", aux_cols)
    return X, y, predictors, aux_cols


# =========================


# 4) Preprocessing builders

# 4) Preprocessing builders
# =========================
def make_preprocessor(
    config: EOBCConfig,
    predictors: List[str],
    imputation_strategy: str,
    ordinal_as_categorical: bool = False
):
    """
    Leakage-safe preprocessing fitted inside the training/CV pipeline.

    imputation_strategy:
      - 'registry_default': categorical missing coded as Unknown; numeric/ordinal imputed with indicators.
      - 'simple': conventional most_frequent/median imputation without explicit Unknown categories.

    ordinal_as_categorical:
      - False when targeted KNN is used upstream, because tumor grade is converted to ordered numeric values.
      - True when targeted KNN is off, because raw tumor-grade labels should be one-hot encoded rather than forced into numeric scaling.
    """
    numeric_cols = [c for c in config.numeric_cols if c in predictors]
    ordinal_cols = [] if ordinal_as_categorical else [c for c in config.ordinal_cols if c in predictors]
    categorical_cols = [c for c in predictors if c not in numeric_cols + ordinal_cols]

    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler())
    ])

    # This step is safe when targeted KNN is used because target cols are already numeric;
    # it also protects against numeric-looking strings.
    ordinal_pipeline = Pipeline([
        ("to_numeric", FunctionTransformer(lambda X: pd.DataFrame(X).apply(lambda s: pd.to_numeric(s, errors="coerce")), validate=False)),
        ("impute", SimpleImputer(strategy="most_frequent", add_indicator=True)),
        ("scale", StandardScaler())
    ])

    if imputation_strategy == "registry_default":
        categorical_pipeline = Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="Unknown_or_missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
    elif imputation_strategy == "simple":
        categorical_pipeline = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
    else:
        raise ValueError("imputation_strategy must be 'registry_default' or 'simple'.")

    transformers = [
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ]
    if ordinal_cols:
        transformers.insert(1, ("ord", ordinal_pipeline, ordinal_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True
    )
    return preprocessor, numeric_cols, ordinal_cols, categorical_cols


# =========================


# 5) Model builders

# 5) Model builders
# =========================
def make_models(
    config: EOBCConfig,
    predictors: List[str],
    aux_cols: List[str],
    imputation_strategy: str = "registry_default",
    use_targeted_knn: bool = True,
    models_to_run: Optional[List[str]] = None
) -> Dict[str, Pipeline]:
    # If targeted KNN is disabled, tumor grade is treated categorically to avoid arbitrary numeric coding of raw registry labels.
    preprocessor, _, _, _ = make_preprocessor(
        config,
        predictors,
        imputation_strategy,
        ordinal_as_categorical=not use_targeted_knn
    )

    steps_prefix = []
    if use_targeted_knn:
        steps_prefix.append(("targeted_knn", TargetedKNNImputer(target_cols=config.ordinal_cols, aux_cols=aux_cols, k=7)))
    steps_prefix.extend([
        ("keep_predictors", ColumnKeeper(predictors)),
        ("prep", preprocessor),
    ])

    models = {
        "Logit": LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs"),
        "RF": RandomForestClassifier(
            n_estimators=700, max_features="sqrt", min_samples_leaf=5,
            class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=1
        ),
        "SVM_RBF": SVC(kernel="rbf", probability=False, class_weight="balanced", random_state=RANDOM_STATE),
    }

    if not HAS_XGB:
        raise ImportError(
            "XGBoost is required for the final EOBC analysis. "
            "Install a pinned xgboost version before running the notebook."
        )

    models["XGB"] = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=1,
        scale_pos_weight=1.0,
    )

    if models_to_run is None:
        models_to_run = config.models_to_run
    models = {k: v for k, v in models.items() if k in models_to_run}

    return {name: Pipeline(steps_prefix + [("clf", clf)]) for name, clf in models.items()}


def get_param_distributions(model_name: str):
    if model_name == "Logit":
        return {"clf__C": np.logspace(-3, 2, 30)}
    if model_name == "RF":
        return {
            "clf__n_estimators": [400, 700, 1000],
            "clf__max_depth": [None, 3, 5, 8, 12],
            "clf__min_samples_leaf": [1, 3, 5, 10, 20],
            "clf__max_features": ["sqrt", "log2", 0.5],
        }
    if model_name == "SVM_RBF":
        return {"clf__C": np.logspace(-2, 2, 30), "clf__gamma": np.logspace(-4, 0, 30)}
    if model_name == "XGB":
        return {
            "clf__n_estimators": np.arange(300, 1000, 100),
            "clf__learning_rate": np.linspace(0.01, 0.08, 20),
            "clf__max_depth": [2, 3, 4, 5],
            "clf__min_child_weight": [1, 3, 5, 8, 12],
            "clf__subsample": [0.7, 0.8, 0.9, 1.0],
            "clf__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "clf__reg_lambda": np.linspace(0.2, 3.0, 20),
        }
    return None


# =========================


# 6) Thresholding + evaluation

# 6) Thresholding + evaluation
# =============================

def _predict_proba_positive(
    model,
    X,
):
    """
    Return probabilities for the positive class.
    """
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(
            model.predict_proba(X),
            dtype=float,
        )

        if probabilities.ndim != 2 or probabilities.shape[1] != 2:
            raise ValueError(
                "Expected predict_proba() to return an "
                "observations-by-two-classes array."
            )

        return probabilities[:, 1]

    if hasattr(model, "decision_function"):
        scores = np.asarray(
            model.decision_function(X),
            dtype=float,
        )

        return 1.0 / (
            1.0
            + np.exp(
                -np.clip(
                    scores,
                    -30,
                    30,
                )
            )
        )

    raise ValueError(
        "Model has neither predict_proba() nor decision_function()."
    )


def make_sigmoid_calibrator(
    estimator,
    random_state=RANDOM_STATE,
):
    """
    Create a reproducible cross-validated Platt calibrator.

    Out-of-fold training predictions are used to estimate the sigmoid
    calibration function. With ensemble=False, one final base estimator
    is fitted on the complete available training set instead of averaging
    predictions from several fold-specific estimators.
    """
    calibration_cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=int(random_state),
    )

    return CalibratedClassifierCV(
        estimator=clone(estimator),
        method="sigmoid",
        cv=calibration_cv,
        ensemble=False,
        n_jobs=1,
    )


def best_threshold_precision_at_recall(
    y_true,
    prob,
    min_recall=0.70,
):
    """
    Select the threshold that maximizes precision while maintaining
    the prespecified minimum recall.

    If no tested threshold satisfies the recall constraint, the
    threshold maximizing F1 is used as a fallback.
    """
    y_true = np.asarray(y_true, dtype=int)
    prob = np.asarray(prob, dtype=float)

    thresholds = np.unique(
        np.quantile(
            prob,
            np.linspace(0.01, 0.99, 199),
        )
    )

    best_threshold = 0.5
    best_precision = -np.inf

    for threshold in thresholds:
        predictions = (prob >= threshold).astype(int)

        recall = recall_score(
            y_true,
            predictions,
            zero_division=0,
        )

        if recall >= min_recall:
            precision = precision_score(
                y_true,
                predictions,
                zero_division=0,
            )

            if precision > best_precision:
                best_threshold = threshold
                best_precision = precision

    if best_precision == -np.inf:
        f1_values = [
            f1_score(
                y_true,
                (prob >= threshold).astype(int),
                zero_division=0,
            )
            for threshold in thresholds
        ]

        best_threshold = thresholds[int(np.argmax(f1_values))]

    return float(best_threshold)


def compute_metrics(
    y_true,
    prob,
    threshold,
):
    """
    Calculate discrimination, probabilistic accuracy, and
    threshold-dependent classification metrics.
    """
    y_true = np.asarray(y_true, dtype=int)
    prob = np.asarray(prob, dtype=float)
    predictions = (prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "pr_auc": float(average_precision_score(y_true, prob)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predictions)
        ),
        "sensitivity": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "specificity": float(specificity),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "brier_score": float(
            brier_score_loss(y_true, prob)
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def bootstrap_metric_cis(
    y_true,
    prob,
    threshold,
    n_bootstrap=2000,
    seed=RANDOM_STATE,
):
    """
    Estimate percentile bootstrap confidence intervals for model
    performance metrics.
    """
    rng = np.random.default_rng(seed)

    y_true = np.asarray(y_true, dtype=int)
    prob = np.asarray(prob, dtype=float)
    n_observations = len(y_true)

    bootstrap_rows = []

    for _ in range(int(n_bootstrap)):
        indices = rng.integers(
            0,
            n_observations,
            size=n_observations,
        )

        y_bootstrap = y_true[indices]

        if np.unique(y_bootstrap).size < 2:
            continue

        bootstrap_rows.append(
            compute_metrics(
                y_true=y_bootstrap,
                prob=prob[indices],
                threshold=threshold,
            )
        )

    if not bootstrap_rows:
        raise RuntimeError(
            "No valid bootstrap samples were generated."
        )

    bootstrap_results = pd.DataFrame(bootstrap_rows)
    confidence_intervals = {}

    metric_columns = [
        "roc_auc",
        "pr_auc",
        "accuracy",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "brier_score",
    ]

    for metric in metric_columns:
        confidence_intervals[f"{metric}_ci_low"] = float(
            bootstrap_results[metric].quantile(0.025)
        )
        confidence_intervals[f"{metric}_ci_high"] = float(
            bootstrap_results[metric].quantile(0.975)
        )

    confidence_intervals["n_bootstrap_completed"] = int(
        len(bootstrap_results)
    )

    return confidence_intervals


# 7) One scenario runner

# 7) One scenario runner
# =========================
def tune_model(name, model, X_train, y_train, config):
    param_dist = get_param_distributions(name)

    if param_dist is None:
        model.fit(X_train, y_train)
        return model, {"tuned": False}

    if name == "XGB":
        y_train_series = pd.Series(y_train)
        n_negative = int((y_train_series == 0).sum())
        n_positive = int((y_train_series == 1).sum())

        if n_positive == 0:
            raise ValueError("XGBoost training data contain no positive EOBC cases.")

        imbalance_ratio = n_negative / n_positive
        scale_candidates = sorted({
            1.0,
            round(0.75 * imbalance_ratio, 4),
            round(imbalance_ratio, 4),
            round(1.25 * imbalance_ratio, 4),
            round(1.50 * imbalance_ratio, 4),
        })

        param_dist = dict(param_dist)
        param_dist["clf__scale_pos_weight"] = scale_candidates

        print("XGBoost scale_pos_weight candidates:", scale_candidates)
    else:
        imbalance_ratio = None
        scale_candidates = None

    cv = StratifiedKFold(
        n_splits=config.n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    n_iter = 25 if name in [
        "Logit", "RF", "SVM_RBF"
    ] else 50

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="balanced_accuracy",
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=0,
        refit=True,
    )

    search.fit(X_train, y_train)

    result = {
        "tuned": True,
        "best_params": search.best_params_,
        "best_cv_bal_acc": search.best_score_,
    }

    if name == "XGB":
        result["training_imbalance_ratio"] = imbalance_ratio
        result["scale_pos_weight_candidates"] = scale_candidates

    return search.best_estimator_, result


def evaluate_model(
    name,
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    config,
):
    """
    Calibrate a tuned predictive model using cross-validated Platt
    scaling within the training data, select the operating threshold
    from training probabilities, and evaluate the unchanged calibrated
    model and threshold on the independent test set.
    """
    calibrated = make_sigmoid_calibrator(
        estimator=model,
        random_state=RANDOM_STATE,
    )

    calibrated.fit(
        X_train,
        y_train,
    )

    # Training-derived probabilities are used only for threshold
    # selection. The independent test set is not involved.
    prob_train = _predict_proba_positive(
        calibrated,
        X_train,
    )

    threshold = best_threshold_precision_at_recall(
        y_true=y_train,
        prob=prob_train,
        min_recall=config.min_recall,
    )

    training_metrics = compute_metrics(
        y_true=y_train,
        prob=prob_train,
        threshold=threshold,
    )

    # Apply the calibrated model and fixed threshold unchanged to the
    # independent held-out test set.
    prob_test = _predict_proba_positive(
        calibrated,
        X_test,
    )

    metrics = compute_metrics(
        y_true=y_test,
        prob=prob_test,
        threshold=threshold,
    )

    confidence_intervals = bootstrap_metric_cis(
        y_true=y_test,
        prob=prob_test,
        threshold=threshold,
        n_bootstrap=config.n_bootstrap,
        seed=RANDOM_STATE,
    )

    metrics.update(confidence_intervals)

    metrics["training_sensitivity_at_threshold"] = float(
        training_metrics["sensitivity"]
    )
    metrics["training_precision_at_threshold"] = float(
        training_metrics["precision"]
    )
    metrics["training_f1_at_threshold"] = float(
        training_metrics["f1"]
    )
    metrics["training_balanced_accuracy_at_threshold"] = float(
        training_metrics["balanced_accuracy"]
    )
    metrics["threshold_selection_min_recall"] = float(
        config.min_recall
    )
    metrics["calibration_method"] = (
        "Platt sigmoid; 3-fold shuffled CV; ensemble=False"
    )
    metrics["model"] = name

    return calibrated, prob_test, metrics


def run_scenario(
    scenario_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    predictors: List[str],
    aux_cols: List[str],
    config,
    imputation_strategy: str = "registry_default",
    use_targeted_knn: bool = True,
    complete_case_grade_morpho: bool = False,
    models_to_run: Optional[List[str]] = None,
):
    print(f"\n===== Scenario: {scenario_name} =====")

    predictors = [c for c in predictors if c in X.columns]
    aux_cols = [c for c in aux_cols if c in X.columns]

    X_s = X[
        predictors + [c for c in aux_cols if c not in predictors]
    ].copy()
    y_s = y.copy()

    if complete_case_grade_morpho:
        cc_cols = [
            c for c in ["grade_num", "morpho3"] if c in X_s.columns
        ]
        mask = X_s[cc_cols].notna().all(axis=1)
        X_s = X_s.loc[mask].copy()
        y_s = y_s.loc[mask].copy()
        print("Complete-case rows retained:", len(y_s))

    X_train, X_test, y_train, y_test = train_test_split(
        X_s,
        y_s,
        test_size=config.test_size,
        stratify=y_s,
        random_state=RANDOM_STATE,
    )

    models = make_models(
        config,
        predictors,
        aux_cols,
        imputation_strategy,
        use_targeted_knn,
        models_to_run=models_to_run,
    )

    all_rows = []
    model_artifacts = {}
    tuning_info = {}

    for name, model in models.items():
        print("Fitting", name)

        best_model, info = tune_model(
            name,
            model,
            X_train,
            y_train,
            config,
        )

        calibrated, prob_test, metrics = evaluate_model(
            name,
            best_model,
            X_train,
            y_train,
            X_test,
            y_test,
            config,
        )

        metrics["scenario"] = scenario_name
        metrics["n_total"] = len(y_s)
        metrics["n_train"] = len(y_train)
        metrics["n_test"] = len(y_test)
        metrics["imputation_strategy"] = imputation_strategy
        metrics["targeted_knn"] = use_targeted_knn
        metrics["complete_case_grade_morpho"] = complete_case_grade_morpho

        all_rows.append(metrics)
        model_artifacts[name] = calibrated
        tuning_info[name] = info

    results = pd.DataFrame(all_rows).sort_values(
        ["balanced_accuracy", "roc_auc"],
        ascending=False,
    )

    results.to_csv(
        os.path.join(
            config.output_dir,
            f"results_{scenario_name}.csv",
        ),
        index=False,
    )

    with open(
        os.path.join(
            config.output_dir,
            f"tuning_{scenario_name}.json",
        ),
        "w",
    ) as f:
        json.dump(tuning_info, f, indent=2, default=str)

    return results, model_artifacts


def run_named_scenario(
    scen_name: str,
    models_to_run: Optional[List[str]] = None,
):
    X, y, predictors, aux_cols = load_data(CONFIG)
    scenarios = build_scenarios(CONFIG, predictors)

    if scen_name not in scenarios:
        raise ValueError(
            f"Unknown scenario: {scen_name}. "
            f"Available: {list(scenarios)}"
        )

    params = scenarios[scen_name]

    return run_scenario(
        scenario_name=scen_name,
        X=X,
        y=y,
        predictors=params["predictors"],
        aux_cols=aux_cols,
        config=CONFIG,
        imputation_strategy=params["imputation_strategy"],
        use_targeted_knn=params["use_targeted_knn"],
        complete_case_grade_morpho=params[
            "complete_case_grade_morpho"
        ],
        models_to_run=models_to_run,
    )


# =========================

# 7B) Repeated nested stratified cross-validation
# =================================================
# Primary internal-validation framework:
#   - Outer CV: five stratified folds repeated five times.
#   - Inner CV: three stratified folds for hyperparameter tuning.
#
# All preprocessing, targeted imputation, model tuning, probability
# calibration, and threshold selection are performed without access
# to the corresponding outer-test fold.

NESTED_CV_MODELS = [
    "Logit",
    "RF",
    "SVM_RBF",
    "XGB",
]


def _nested_search_iterations(
    model_name,
    config,
):
    """
    Return the randomized-search size used inside each outer fold.
    """
    iteration_map = {
        "Logit": config.nested_n_iter_logit,
        "RF": config.nested_n_iter_rf,
        "SVM_RBF": config.nested_n_iter_svm,
        "XGB": config.nested_n_iter_xgb,
    }

    if model_name not in iteration_map:
        raise ValueError(
            f"Unsupported nested-CV model: {model_name}"
        )

    return int(
        iteration_map[model_name]
    )


def build_nested_search(
    model_name,
    estimator,
    y_train,
    config,
    random_state,
):
    """
    Construct one inner randomized hyperparameter search.

    The complete estimator is a leakage-safe sklearn Pipeline, so all
    preprocessing and imputation steps are fitted independently within
    each inner-training fold.
    """
    param_distributions = get_param_distributions(
        model_name
    )

    if param_distributions is None:
        return None

    param_distributions = dict(
        param_distributions
    )

    if model_name == "XGB":
        y_train_series = pd.Series(
            np.asarray(
                y_train,
                dtype=int,
            )
        )

        n_negative = int(
            (y_train_series == 0).sum()
        )

        n_positive = int(
            (y_train_series == 1).sum()
        )

        if n_positive == 0:
            raise ValueError(
                "An outer-training fold contains no EOBC cases."
            )

        imbalance_ratio = (
            n_negative / n_positive
        )

        param_distributions[
            "clf__scale_pos_weight"
        ] = sorted({
            1.0,
            round(
                0.75 * imbalance_ratio,
                4,
            ),
            round(
                imbalance_ratio,
                4,
            ),
            round(
                1.25 * imbalance_ratio,
                4,
            ),
            round(
                1.50 * imbalance_ratio,
                4,
            ),
        })

    inner_cv = StratifiedKFold(
        n_splits=int(
            config.nested_inner_folds
        ),
        shuffle=True,
        random_state=int(
            random_state
        ),
    )

    return RandomizedSearchCV(
        estimator=clone(
            estimator
        ),
        param_distributions=
            param_distributions,
        n_iter=_nested_search_iterations(
            model_name=model_name,
            config=config,
        ),
        scoring="balanced_accuracy",
        cv=inner_cv,
        refit=True,
        random_state=int(
            random_state
        ),
        n_jobs=-1,
        verbose=0,
        return_train_score=False,
    )


def _outer_training_threshold_probabilities(
    tuned_estimator,
    X_train,
    y_train,
    config,
    random_state,
):
    """
    Generate out-of-fold calibrated probabilities within one outer
    training fold for threshold selection.

    This keeps threshold optimization independent of the fitted model's
    in-sample probabilities.
    """
    threshold_cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=int(
            random_state
        ),
    )

    oof_probabilities = np.full(
        len(y_train),
        np.nan,
        dtype=float,
    )

    X_train_reset = X_train.reset_index(
        drop=True
    )

    y_train_reset = pd.Series(
        np.asarray(
            y_train,
            dtype=int,
        )
    ).reset_index(
        drop=True
    )

    for threshold_fold, (
        subtrain_indices,
        validation_indices,
    ) in enumerate(
        threshold_cv.split(
            X_train_reset,
            y_train_reset,
        ),
        start=1,
    ):
        calibration_seed = (
            int(random_state)
            + threshold_fold
        )

        threshold_model = make_sigmoid_calibrator(
            estimator=clone(
                tuned_estimator
            ),
            random_state=
                calibration_seed,
        )

        threshold_model.fit(
            X_train_reset.iloc[
                subtrain_indices
            ],
            y_train_reset.iloc[
                subtrain_indices
            ],
        )

        oof_probabilities[
            validation_indices
        ] = _predict_proba_positive(
            threshold_model,
            X_train_reset.iloc[
                validation_indices
            ],
        )

    if np.isnan(
        oof_probabilities
    ).any():
        raise RuntimeError(
            "Threshold-selection OOF probabilities are incomplete."
        )

    return oof_probabilities


def run_repeated_nested_cv(
    X,
    y,
    predictors,
    aux_cols,
    config,
    model_names=None,
    overwrite=False,
):
    """
    Run repeated nested stratified cross-validation.

    Hyperparameter tuning is repeated independently within every outer
    training fold. Each outer-test fold remains untouched until final
    evaluation.

    Checkpoints are saved after every completed outer fold.
    """
    if model_names is None:
        model_names = list(
            NESTED_CV_MODELS
        )

    X = X.reset_index(
        drop=True
    ).copy()

    y = pd.Series(
        np.asarray(
            y,
            dtype=int,
        ),
        name=getattr(
            y,
            "name",
            "outcome",
        ),
    ).reset_index(
        drop=True
    )

    if len(X) != len(y):
        raise ValueError(
            "X and y must contain the same number of observations."
        )

    fold_results_path = os.path.join(
        config.output_dir,
        "nested_cv_fold_metrics.csv",
    )

    prediction_path = os.path.join(
        config.output_dir,
        "nested_cv_oof_predictions.csv",
    )

    tuning_path = os.path.join(
        config.output_dir,
        "nested_cv_tuning_results.csv",
    )

    checkpoint_paths = [
        fold_results_path,
        prediction_path,
        tuning_path,
    ]

    if overwrite:
        for path in checkpoint_paths:
            if os.path.exists(path):
                os.remove(path)

                print(
                    "Deleted previous nested-CV checkpoint:",
                    path,
                )

    if (
        os.path.exists(fold_results_path)
        and not overwrite
    ):
        fold_results = pd.read_csv(
            fold_results_path
        )
    else:
        fold_results = pd.DataFrame()

    if (
        os.path.exists(prediction_path)
        and not overwrite
    ):
        oof_predictions = pd.read_csv(
            prediction_path
        )
    else:
        oof_predictions = pd.DataFrame()

    if (
        os.path.exists(tuning_path)
        and not overwrite
    ):
        tuning_results = pd.read_csv(
            tuning_path
        )
    else:
        tuning_results = pd.DataFrame()

    expected_models = set(
        model_names
    )

    total_outer_folds = (
        int(config.nested_outer_folds)
        * int(config.nested_outer_repeats)
    )

    for repeat_index in range(
        int(
            config.nested_outer_repeats
        )
    ):
        repeat_number = (
            repeat_index + 1
        )

        repeat_seed = (
            int(
                config.nested_seed_start
            )
            + repeat_index
        )

        outer_cv = StratifiedKFold(
            n_splits=int(
                config.nested_outer_folds
            ),
            shuffle=True,
            random_state=
                repeat_seed,
        )

        for fold_number, (
            train_indices,
            test_indices,
        ) in enumerate(
            outer_cv.split(
                X,
                y,
            ),
            start=1,
        ):
            existing_models = set()

            if not fold_results.empty:
                existing_rows = fold_results.loc[
                    fold_results[
                        "repeat"
                    ].eq(
                        repeat_number
                    )
                    & fold_results[
                        "fold"
                    ].eq(
                        fold_number
                    )
                    & fold_results[
                        "repeat_seed"
                    ].eq(
                        repeat_seed
                    )
                ]

                existing_models = set(
                    existing_rows[
                        "model"
                    ].tolist()
                )

            if expected_models.issubset(
                existing_models
            ):
                print(
                    f"Repeat {repeat_number}, "
                    f"fold {fold_number}: already complete."
                )
                continue

            # Remove partial rows for this fold before recomputing it.
            if not fold_results.empty:
                fold_results = fold_results.loc[
                    ~(
                        fold_results[
                            "repeat"
                        ].eq(
                            repeat_number
                        )
                        & fold_results[
                            "fold"
                        ].eq(
                            fold_number
                        )
                        & fold_results[
                            "repeat_seed"
                        ].eq(
                            repeat_seed
                        )
                    )
                ].copy()

            if not oof_predictions.empty:
                oof_predictions = oof_predictions.loc[
                    ~(
                        oof_predictions[
                            "repeat"
                        ].eq(
                            repeat_number
                        )
                        & oof_predictions[
                            "fold"
                        ].eq(
                            fold_number
                        )
                        & oof_predictions[
                            "repeat_seed"
                        ].eq(
                            repeat_seed
                        )
                    )
                ].copy()

            if not tuning_results.empty:
                tuning_results = tuning_results.loc[
                    ~(
                        tuning_results[
                            "repeat"
                        ].eq(
                            repeat_number
                        )
                        & tuning_results[
                            "fold"
                        ].eq(
                            fold_number
                        )
                        & tuning_results[
                            "repeat_seed"
                        ].eq(
                            repeat_seed
                        )
                    )
                ].copy()

            X_train = X.iloc[
                train_indices
            ].copy()

            X_test = X.iloc[
                test_indices
            ].copy()

            y_train = y.iloc[
                train_indices
            ].copy()

            y_test = y.iloc[
                test_indices
            ].copy()

            models = make_models(
                config=config,
                predictors=predictors,
                aux_cols=aux_cols,
                imputation_strategy=
                    "registry_default",
                use_targeted_knn=True,
                models_to_run=
                    model_names,
            )

            outer_fold_index = (
                repeat_index
                * int(
                    config.nested_outer_folds
                )
                + fold_number
            )

            fold_rows = []
            prediction_frames = []
            tuning_rows = []

            for model_position, (
                model_name,
                base_estimator,
            ) in enumerate(
                models.items(),
                start=1,
            ):
                print(
                    f"Nested outer fold "
                    f"{outer_fold_index}/"
                    f"{total_outer_folds} "
                    f"| repeat={repeat_number} "
                    f"| fold={fold_number} "
                    f"| model={model_name}"
                )

                inner_seed = (
                    repeat_seed
                    * 1000
                    + fold_number
                    * 10
                    + model_position
                )

                search = build_nested_search(
                    model_name=
                        model_name,
                    estimator=
                        base_estimator,
                    y_train=
                        y_train,
                    config=config,
                    random_state=
                        inner_seed,
                )

                if search is None:
                    tuned_estimator = clone(
                        base_estimator
                    )

                    tuned_estimator.fit(
                        X_train,
                        y_train,
                    )

                    best_params = {}
                    best_inner_score = np.nan
                else:
                    search.fit(
                        X_train,
                        y_train,
                    )

                    tuned_estimator = clone(
                        search.best_estimator_
                    )

                    best_params = dict(
                        search.best_params_
                    )

                    best_inner_score = float(
                        search.best_score_
                    )

                threshold_probabilities = (
                    _outer_training_threshold_probabilities(
                        tuned_estimator=
                            tuned_estimator,
                        X_train=
                            X_train,
                        y_train=
                            y_train,
                        config=config,
                        random_state=
                            inner_seed + 20000,
                    )
                )

                threshold = (
                    best_threshold_precision_at_recall(
                        y_true=
                            y_train,
                        prob=
                            threshold_probabilities,
                        min_recall=
                            config.min_recall,
                    )
                )

                threshold_training_metrics = (
                    compute_metrics(
                        y_true=
                            y_train,
                        prob=
                            threshold_probabilities,
                        threshold=
                            threshold,
                    )
                )

                final_calibrated_model = (
                    make_sigmoid_calibrator(
                        estimator=
                            tuned_estimator,
                        random_state=
                            inner_seed + 50000,
                    )
                )

                final_calibrated_model.fit(
                    X_train,
                    y_train,
                )

                test_probabilities = (
                    _predict_proba_positive(
                        final_calibrated_model,
                        X_test,
                    )
                )

                test_metrics = compute_metrics(
                    y_true=
                        y_test,
                    prob=
                        test_probabilities,
                    threshold=
                        threshold,
                )

                predicted_labels = (
                    test_probabilities
                    >= threshold
                ).astype(int)

                fold_rows.append({
                    "repeat":
                        int(repeat_number),
                    "repeat_seed":
                        int(repeat_seed),
                    "fold":
                        int(fold_number),
                    "model":
                        model_name,
                    "n_total":
                        int(len(y)),
                    "n_train":
                        int(len(y_train)),
                    "n_test":
                        int(len(y_test)),
                    "test_eobc_n":
                        int(
                            np.sum(
                                y_test.to_numpy()
                                == 1
                            )
                        ),
                    "test_eobc_prevalence":
                        float(
                            np.mean(y_test)
                        ),
                    "threshold":
                        float(threshold),
                    "training_sensitivity_at_threshold":
                        float(
                            threshold_training_metrics[
                                "sensitivity"
                            ]
                        ),
                    "training_precision_at_threshold":
                        float(
                            threshold_training_metrics[
                                "precision"
                            ]
                        ),
                    "roc_auc":
                        float(
                            test_metrics[
                                "roc_auc"
                            ]
                        ),
                    "pr_auc":
                        float(
                            test_metrics[
                                "pr_auc"
                            ]
                        ),
                    "accuracy":
                        float(
                            test_metrics[
                                "accuracy"
                            ]
                        ),
                    "balanced_accuracy":
                        float(
                            test_metrics[
                                "balanced_accuracy"
                            ]
                        ),
                    "sensitivity":
                        float(
                            test_metrics[
                                "sensitivity"
                            ]
                        ),
                    "specificity":
                        float(
                            test_metrics[
                                "specificity"
                            ]
                        ),
                    "precision":
                        float(
                            test_metrics[
                                "precision"
                            ]
                        ),
                    "f1":
                        float(
                            test_metrics[
                                "f1"
                            ]
                        ),
                    "brier_score":
                        float(
                            test_metrics[
                                "brier_score"
                            ]
                        ),
                    "best_inner_balanced_accuracy":
                        best_inner_score,
                    "calibration_method":
                        (
                            "Platt sigmoid; "
                            "3-fold shuffled CV; "
                            "ensemble=False"
                        ),
                })

                prediction_frames.append(
                    pd.DataFrame({
                        "repeat":
                            int(repeat_number),
                        "repeat_seed":
                            int(repeat_seed),
                        "fold":
                            int(fold_number),
                        "model":
                            model_name,
                        "observation_id":
                            test_indices.astype(
                                int
                            ),
                        "y_true":
                            y_test.to_numpy(
                                dtype=int
                            ),
                        "probability":
                            np.asarray(
                                test_probabilities,
                                dtype=float,
                            ),
                        "threshold":
                            float(threshold),
                        "predicted_label":
                            np.asarray(
                                predicted_labels,
                                dtype=int,
                            ),
                    })
                )

                tuning_rows.append({
                    "repeat":
                        int(repeat_number),
                    "repeat_seed":
                        int(repeat_seed),
                    "fold":
                        int(fold_number),
                    "model":
                        model_name,
                    "best_inner_balanced_accuracy":
                        best_inner_score,
                    "best_params":
                        json.dumps(
                            best_params,
                            default=str,
                            sort_keys=True,
                        ),
                })

            fold_results = pd.concat(
                [
                    fold_results,
                    pd.DataFrame(
                        fold_rows
                    ),
                ],
                ignore_index=True,
            )

            oof_predictions = pd.concat(
                [
                    oof_predictions,
                    pd.concat(
                        prediction_frames,
                        ignore_index=True,
                    ),
                ],
                ignore_index=True,
            )

            tuning_results = pd.concat(
                [
                    tuning_results,
                    pd.DataFrame(
                        tuning_rows
                    ),
                ],
                ignore_index=True,
            )

            fold_results = (
                fold_results
                .sort_values(
                    [
                        "repeat",
                        "fold",
                        "model",
                    ]
                )
                .reset_index(
                    drop=True
                )
            )

            oof_predictions = (
                oof_predictions
                .sort_values(
                    [
                        "repeat",
                        "fold",
                        "model",
                        "observation_id",
                    ]
                )
                .reset_index(
                    drop=True
                )
            )

            tuning_results = (
                tuning_results
                .sort_values(
                    [
                        "repeat",
                        "fold",
                        "model",
                    ]
                )
                .reset_index(
                    drop=True
                )
            )

            fold_results.to_csv(
                fold_results_path,
                index=False,
            )

            oof_predictions.to_csv(
                prediction_path,
                index=False,
            )

            tuning_results.to_csv(
                tuning_path,
                index=False,
            )

            print(
                "Nested-CV checkpoint saved after "
                f"repeat {repeat_number}, "
                f"fold {fold_number}."
            )

    return (
        fold_results,
        oof_predictions,
        tuning_results,
    )


def summarize_repeated_nested_cv(
    fold_results,
    output_dir,
):
    """
    Summarize nested-CV performance across paired outer folds.

    The summary reports descriptive variability across outer folds.
    Percentile intervals are empirical distributions, not conventional
    independent-sample confidence intervals.
    """
    required_columns = {
        "repeat",
        "fold",
        "model",
        "roc_auc",
        "pr_auc",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "brier_score",
    }

    missing_columns = required_columns.difference(
        fold_results.columns
    )

    if missing_columns:
        raise ValueError(
            "Nested-CV results are missing columns: "
            f"{sorted(missing_columns)}"
        )

    metric_columns = [
        "roc_auc",
        "pr_auc",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "brier_score",
    ]

    summary_rows = []

    for model_name, model_data in (
        fold_results.groupby(
            "model",
            sort=False,
        )
    ):
        row = {
            "model":
                model_name,
            "n_outer_folds":
                int(
                    len(model_data)
                ),
            "n_repeats":
                int(
                    model_data[
                        "repeat"
                    ].nunique()
                ),
            "folds_per_repeat":
                int(
                    model_data[
                        "fold"
                    ].nunique()
                ),
        }

        for metric in metric_columns:
            values = (
                model_data[metric]
                .dropna()
                .astype(float)
            )

            row[
                f"{metric}_mean"
            ] = float(
                values.mean()
            )

            row[
                f"{metric}_sd"
            ] = float(
                values.std(
                    ddof=1
                )
            )

            row[
                f"{metric}_median"
            ] = float(
                values.median()
            )

            row[
                f"{metric}_min"
            ] = float(
                values.min()
            )

            row[
                f"{metric}_max"
            ] = float(
                values.max()
            )

            row[
                f"{metric}_p025"
            ] = float(
                values.quantile(
                    0.025
                )
            )

            row[
                f"{metric}_p975"
            ] = float(
                values.quantile(
                    0.975
                )
            )

        summary_rows.append(
            row
        )

    model_summary = (
        pd.DataFrame(
            summary_rows
        )
        .sort_values(
            "roc_auc_mean",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    wide_auc = fold_results.pivot_table(
        index=[
            "repeat",
            "fold",
        ],
        columns="model",
        values="roc_auc",
        aggfunc="first",
    )

    if "Logit" not in wide_auc.columns:
        raise ValueError(
            "Logistic regression is required as the reference model."
        )

    difference_rows = []

    for model_name in [
        column
        for column in wide_auc.columns
        if column != "Logit"
    ]:
        paired = wide_auc[
            [
                "Logit",
                model_name,
            ]
        ].dropna()

        differences = (
            paired[model_name]
            - paired["Logit"]
        )

        difference_rows.append({
            "comparison_model":
                model_name,
            "reference_model":
                "Logit",
            "n_paired_outer_folds":
                int(
                    len(differences)
                ),
            "mean_delta_auc":
                float(
                    differences.mean()
                ),
            "sd_delta_auc":
                float(
                    differences.std(
                        ddof=1
                    )
                ),
            "median_delta_auc":
                float(
                    differences.median()
                ),
            "minimum_delta_auc":
                float(
                    differences.min()
                ),
            "maximum_delta_auc":
                float(
                    differences.max()
                ),
            "delta_auc_p025":
                float(
                    differences.quantile(
                        0.025
                    )
                ),
            "delta_auc_p975":
                float(
                    differences.quantile(
                        0.975
                    )
                ),
            "proportion_outer_folds_above_logit":
                float(
                    np.mean(
                        differences > 0
                    )
                ),
            "proportion_outer_folds_equal_logit":
                float(
                    np.mean(
                        differences == 0
                    )
                ),
            "proportion_outer_folds_below_logit":
                float(
                    np.mean(
                        differences < 0
                    )
                ),
        })

    paired_differences = (
        pd.DataFrame(
            difference_rows
        )
        .sort_values(
            "mean_delta_auc",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    model_summary_path = os.path.join(
        output_dir,
        "nested_cv_model_summary.csv",
    )

    paired_differences_path = os.path.join(
        output_dir,
        "nested_cv_auc_differences_vs_logit.csv",
    )

    model_summary.to_csv(
        model_summary_path,
        index=False,
    )

    paired_differences.to_csv(
        paired_differences_path,
        index=False,
    )

    print(
        "Saved nested-CV model summary:",
        model_summary_path,
    )

    print(
        "Saved nested-CV paired AUC differences:",
        paired_differences_path,
    )

    return (
        model_summary,
        paired_differences,
    )


def plot_nested_cv_auc_distributions(
    fold_results,
    output_dir,
):
    """
    Save a boxplot of outer-fold ROC-AUC values.
    """
    requested_order = [
        "Logit",
        "RF",
        "SVM_RBF",
        "XGB",
    ]

    available_order = [
        model_name
        for model_name in requested_order
        if model_name
        in fold_results[
            "model"
        ].unique()
    ]

    plot_values = [
        fold_results.loc[
            fold_results[
                "model"
            ].eq(
                model_name
            ),
            "roc_auc",
        ].to_numpy()
        for model_name in available_order
    ]

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.boxplot(
        plot_values,
        labels=available_order,
        showmeans=True,
    )

    ax.set_ylabel(
        "Outer-fold ROC-AUC"
    )

    ax.set_xlabel(
        "Model"
    )

    ax.set_title(
        "ROC-AUC across repeated nested cross-validation"
    )

    plt.tight_layout()

    output_path = os.path.join(
        output_dir,
        "nested_cv_auc_distributions.png",
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        output_path.replace(
            ".png",
            ".pdf",
        ),
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        "Saved nested-CV AUC plot:",
        output_path,
    )

    return output_path


# 8) Sensitivity scenarios

# 8) Sensitivity scenarios
# =========================
def build_scenarios(config: EOBCConfig, all_predictors: List[str]):
    # Domains. Edit if your conceptual grouping differs.
    sociodemo_history = [
        "ydod", "hospital_grp_num", "ethnicity4", "gov_num", "religion_grp3",
        "edu_level", "smoke_bi", "job_employed", "famhx_bi", "comorb_YesNo"
    ]
    tumor_clinical = ["behav", "topo_cat", "stagedx_num", "grade_num", "morpho3"]

    scenarios = {
        "primary_full_registry_default": {
            "predictors": all_predictors,
            "imputation_strategy": "registry_default",
            "use_targeted_knn": True,
            "complete_case_grade_morpho": False,
        },
        "no_employment": {
            "predictors": [c for c in all_predictors if c != "job_employed"],
            "imputation_strategy": "registry_default",
            "use_targeted_knn": True,
            "complete_case_grade_morpho": False,
        },
        "no_education": {
            "predictors": [c for c in all_predictors if c != "edu_level"],
            "imputation_strategy": "registry_default",
            "use_targeted_knn": True,
            "complete_case_grade_morpho": False,
        },
        "no_employment_no_education": {
            "predictors": [c for c in all_predictors if c not in ["job_employed", "edu_level"]],
            "imputation_strategy": "registry_default",
            "use_targeted_knn": True,
            "complete_case_grade_morpho": False,
        },
        "sociodemo_history_only": {
            "predictors": [c for c in sociodemo_history if c in all_predictors],
            "imputation_strategy": "registry_default",
            "use_targeted_knn": False,
            "complete_case_grade_morpho": False,
        },
        "tumor_clinical_only": {
            "predictors": [c for c in tumor_clinical if c in all_predictors],
            "imputation_strategy": "registry_default",
            "use_targeted_knn": True,
            "complete_case_grade_morpho": False,
        },
        "simple_imputation_no_missing_category": {
            "predictors": all_predictors,
            "imputation_strategy": "simple",
            "use_targeted_knn": True,
            "complete_case_grade_morpho": False,
        },
        "no_targeted_knn_registry_default": {
            "predictors": all_predictors,
            "imputation_strategy": "registry_default",
            "use_targeted_knn": False,
            "complete_case_grade_morpho": False,
        },
        "complete_case_grade_morpho": {
            "predictors": all_predictors,
            "imputation_strategy": "registry_default",
            "use_targeted_knn": False,
            "complete_case_grade_morpho": True,
        },
    }
    return scenarios


# =========================


# 9) Optional: permutation importance for best model in primary scenario

# 9) Optional: permutation importance for best model in primary scenario
# =========================
def save_permutation_importance(best_model, X_test, y_test, predictors, output_path):
    # Permute raw columns. Works because the full pipeline accepts raw DataFrames.
    scorer = "balanced_accuracy"
    pi = permutation_importance(
        best_model, X_test, y_test,
        n_repeats=20,
        random_state=RANDOM_STATE,
        scoring=scorer,
        n_jobs=-1
    )
    out = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": pi.importances_mean,
        "importance_sd": pi.importances_std,
    }).query("feature in @predictors").sort_values("importance_mean", ascending=False)
    out.to_csv(output_path, index=False)
    return out


# =========================


# 10) Final inferential regression framework: MICE logistic regression

# 10) Stable R mice/FCS logistic regression
# =========================
# This implementation uses the established R mice package through Rscript.
# To avoid the instability and long runtime of multinomial logistic imputation,
# incomplete categorical variables are imputed with CART within the MICE/FCS
# framework. CART preserves nominal categories, handles nonlinearities and
# interactions, and does not impose artificial numeric ordering on morphology.
#
# Required R packages (install once in a Python cell):
# !Rscript -e 'install.packages(c("mice", "broom"), repos="https://cloud.r-project.org")'

import shutil
import subprocess
from pathlib import Path


GRADE_CANONICAL_LABELS = {
    1: "Grade I",
    2: "Grade II",
    3: "Grade III",
}

MORPHO_CANONICAL_LABELS = {
    1: "Infiltrating duct carcinoma",
    2: "Lobular carcinoma",
    3: "Mixed/other, including rare morphologies",
}


def _canonicalize_grade(series):
    def convert(value):
        if pd.isna(value):
            return np.nan

        try:
            numeric = int(float(value))
            if numeric in GRADE_CANONICAL_LABELS:
                return GRADE_CANONICAL_LABELS[numeric]
        except (TypeError, ValueError):
            pass

        text = str(value).strip().lower()

        if text in {
            "", "nan", "none", "unknown", "missing",
            "not known", "not available", "not applicable",
        }:
            return np.nan

        if (
            "grade iii" in text
            or "grade 3" in text
            or "poor diff" in text
            or "poorly differentiated" in text
            or text in {"iii", "3"}
        ):
            return GRADE_CANONICAL_LABELS[3]

        if (
            "grade ii" in text
            or "grade 2" in text
            or "mod diff" in text
            or "moderately differentiated" in text
            or text in {"ii", "2"}
        ):
            return GRADE_CANONICAL_LABELS[2]

        if (
            "grade i" in text
            or "grade 1" in text
            or "well diff" in text
            or "well differentiated" in text
            or text in {"i", "1"}
        ):
            return GRADE_CANONICAL_LABELS[1]

        return np.nan

    return series.map(convert).astype("object")


def _canonicalize_morphology(series):
    def convert(value):
        if pd.isna(value):
            return np.nan

        try:
            numeric = int(float(value))
            if numeric in MORPHO_CANONICAL_LABELS:
                return MORPHO_CANONICAL_LABELS[numeric]
        except (TypeError, ValueError):
            pass

        text = str(value).strip().lower()

        if text in {
            "", "nan", "none", "unknown", "missing",
            "not known", "not available", "not applicable",
        }:
            return np.nan

        if "duct" in text:
            return MORPHO_CANONICAL_LABELS[1]

        if "lobular" in text:
            return MORPHO_CANONICAL_LABELS[2]

        if "mixed" in text or "other" in text or "rare" in text:
            return MORPHO_CANONICAL_LABELS[3]

        return np.nan

    return series.map(convert).astype("object")


def prepare_r_mice_dataframe(X, y, predictors, aux_cols):
    predictors = [c for c in predictors if c in X.columns]
    aux_cols = [c for c in aux_cols if c in X.columns]

    imputation_cols = predictors + [
        c for c in aux_cols if c not in predictors
    ]

    data = X[imputation_cols].copy()
    data["early_onset45"] = y.astype(int).values

    if "ydod" in data.columns:
        data["ydod"] = pd.to_numeric(data["ydod"], errors="coerce")

    if "grade_num" in data.columns:
        data["grade_num"] = _canonicalize_grade(data["grade_num"])

    if "morpho3" in data.columns:
        data["morpho3"] = _canonicalize_morphology(data["morpho3"])

    print("\nR mice grade mapping:")
    print(data["grade_num"].value_counts(dropna=False))

    print("\nR mice morphology mapping:")
    print(data["morpho3"].value_counts(dropna=False))

    if data["grade_num"].notna().sum() == 0:
        raise ValueError("Tumor-grade recoding produced no observed values.")

    if data["morpho3"].notna().sum() == 0:
        raise ValueError("Tumor-morphology recoding produced no observed values.")

    return data, predictors, aux_cols


def _r_quote(value):
    value = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{value}"'


def _build_r_mice_script(
    input_csv,
    output_or_csv,
    output_reference_csv,
    output_diagnostics_pdf,
    output_logged_events_csv,
    output_mids_rds,
    predictors,
    m,
    maxit,
    seed,
):
    predictor_vector = ", ".join(_r_quote(x) for x in predictors)

    return f'''
options(stringsAsFactors = FALSE)
options(warn = 1)

required_packages <- c("mice", "broom")
missing_packages <- required_packages[
    !vapply(required_packages, requireNamespace, quietly = TRUE, FUN.VALUE = logical(1))
]

if (length(missing_packages) > 0) {{
    stop(
        paste0(
            "Missing R package(s): ",
            paste(missing_packages, collapse = ", "),
            ". Install once with: install.packages(c('mice','broom'))"
        )
    )
}}

library(mice)

input_csv <- {_r_quote(input_csv)}
output_or_csv <- {_r_quote(output_or_csv)}
output_reference_csv <- {_r_quote(output_reference_csv)}
output_diagnostics_pdf <- {_r_quote(output_diagnostics_pdf)}
output_logged_events_csv <- {_r_quote(output_logged_events_csv)}
output_mids_rds <- {_r_quote(output_mids_rds)}

final_predictors <- c({predictor_vector})

dat <- read.csv(
    input_csv,
    na.strings = c("", "NA", "NaN"),
    check.names = FALSE
)

if (!"early_onset45" %in% names(dat)) {{
    stop("The EOBC outcome is absent from the prepared MICE dataset.")
}}

dat$early_onset45 <- as.integer(dat$early_onset45)

if ("ydod" %in% names(dat)) {{
    dat$ydod <- as.numeric(dat$ydod)
}}

categorical_variables <- setdiff(
    names(dat),
    c("ydod", "early_onset45")
)

for (variable in categorical_variables) {{
    dat[[variable]] <- factor(dat[[variable]])
}}

if ("grade_num" %in% names(dat)) {{
    dat$grade_num <- factor(
        dat$grade_num,
        levels = c("Grade I", "Grade II", "Grade III"),
        ordered = FALSE
    )
}}

if ("morpho3" %in% names(dat)) {{
    dat$morpho3 <- factor(
        dat$morpho3,
        levels = c(
            "Infiltrating duct carcinoma",
            "Lobular carcinoma",
            "Mixed/other, including rare morphologies"
        ),
        ordered = FALSE
    )
}}

reference_candidates <- list(
    hospital_grp_num = c("KCCC"),
    ethnicity4 = c("Arab Kuwaiti"),
    gov_num = c("Ahmadi"),
    religion_grp3 = c("Muslim"),
    edu_level = c("High"),
    smoke_bi = c("Never Smoked", "Never smoked"),
    job_employed = c("Employed"),
    famhx_bi = c("No", "Absent"),
    comorb_YesNo = c("Has comorbidity"),
    behav = c("Malignant"),
    topo_cat = c(
        "Breast, NOS & rare sites",
        "Breast, NOS + rare sites",
        "Breast, NOS + rare sites (central, nipple, axillary tail, overlapping)"
    ),
    stagedx_num = c("Stage (0-I)", "Stage 0-I", "Stage 0–I"),
    grade_num = c("Grade III"),
    morpho3 = c("Infiltrating duct carcinoma")
)

resolve_reference <- function(variable, candidates, data) {{
    available <- levels(factor(data[[variable]]))
    available_lower <- tolower(trimws(available))

    for (candidate in candidates) {{
        candidate_lower <- tolower(trimws(candidate))
        exact_index <- which(available_lower == candidate_lower)
        if (length(exact_index) == 1) return(available[exact_index])
    }}

    for (candidate in candidates) {{
        candidate_lower <- tolower(trimws(candidate))
        partial_index <- which(grepl(candidate_lower, available_lower, fixed = TRUE))
        if (length(partial_index) == 1) return(available[partial_index])
    }}

    stop(
        paste0(
            "Could not resolve reference category for ", variable,
            ". Available levels: ", paste(available, collapse = " | ")
        )
    )
}}

resolved_references <- list()

for (variable in names(reference_candidates)) {{
    if (!variable %in% names(dat)) next

    reference <- resolve_reference(
        variable,
        reference_candidates[[variable]],
        dat
    )

    resolved_references[[variable]] <- reference
    dat[[variable]] <- relevel(
        factor(dat[[variable]], ordered = FALSE),
        ref = reference
    )
}}

initial <- mice(
    dat,
    m = 1,
    maxit = 0,
    printFlag = FALSE
)

method <- initial$method
predictor_matrix <- initial$predictorMatrix

method["early_onset45"] <- ""
predictor_matrix["early_onset45", ] <- 0
diag(predictor_matrix) <- 0

incomplete_variables <- names(dat)[colSums(is.na(dat)) > 0]

for (variable in names(dat)) {{
    if (!anyNA(dat[[variable]])) {{
        method[variable] <- ""
    }} else if (variable == "ydod") {{
        method[variable] <- "pmm"
    }} else {{
        method[variable] <- "cart"
    }}
}}

for (variable in incomplete_variables) {{
    if (variable != "early_onset45") {{
        predictor_matrix[variable, "early_onset45"] <- 1
    }}
}}

cat("\\nMICE methods used:\\n")
print(method)

cat("\\nIncomplete variables:\\n")
print(incomplete_variables)

set.seed({int(seed)})

if (file.exists(output_mids_rds)) {{
    cat("\nLoading cached MICE object:", output_mids_rds, "\n")
    imputed <- readRDS(output_mids_rds)

    if (imputed$m != {int(m)}) {{
        stop(
            paste0(
                "Cached MICE object has m=", imputed$m,
                " but the current configuration requests m={int(m)}. ",
                "Delete the cached RDS file and rerun."
            )
        )
    }}
}} else {{
    imputed <- mice(
        dat,
        m = {int(m)},
        maxit = {int(maxit)},
        method = method,
        predictorMatrix = predictor_matrix,
        seed = {int(seed)},
        printFlag = TRUE,
        remove.collinear = TRUE,
        remove.constant = TRUE
    )

    saveRDS(imputed, output_mids_rds)
    cat("\nSaved cached MICE object:", output_mids_rds, "\n")
}}

pdf(output_diagnostics_pdf, width = 11, height = 8.5)
plot(imputed)
dev.off()

if (is.null(imputed$loggedEvents)) {{
    write.csv(
        data.frame(message = "No logged events"),
        output_logged_events_csv,
        row.names = FALSE
    )
}} else {{
    write.csv(
        imputed$loggedEvents,
        output_logged_events_csv,
        row.names = FALSE
    )
}}

completed_long <- complete(
    imputed,
    action = "long",
    include = FALSE
)

completed_list <- split(completed_long, completed_long$.imp)

fit_list <- lapply(
    completed_list,
    function(completed) {{
        completed$.imp <- NULL
        completed$.id <- NULL

        for (variable in names(resolved_references)) {{
            completed[[variable]] <- relevel(
                factor(completed[[variable]], ordered = FALSE),
                ref = resolved_references[[variable]]
            )
        }}

        model_formula <- as.formula(
            paste(
                "early_onset45 ~",
                paste(final_predictors, collapse = " + ")
            )
        )

        glm(
            model_formula,
            data = completed,
            family = binomial()
        )
    }}
)

model_fits <- as.mira(fit_list)
pooled_model <- pool(model_fits)

pooled_summary <- summary(
    pooled_model,
    conf.int = TRUE,
    exponentiate = FALSE
)

pooled_summary <- pooled_summary[
    pooled_summary$term != "(Intercept)",
]

standardize_term <- function(term) {{
    term <- as.character(term)

    if (identical(term, "ydod")) {{
        return(as.character(term))
    }}

    ordered_predictors <- as.character(
        final_predictors[
            order(nchar(final_predictors), decreasing = TRUE)
        ]
    )

    for (variable in ordered_predictors) {{
        variable <- as.character(variable)

        if (startsWith(term, variable)) {{
            category <- substring(term, nchar(variable) + 1)

            if (nchar(category) > 0) {{
                return(as.character(paste0(variable, "_", category)))
            }}
        }}
    }}

    return(as.character(term))
}}

pooled_summary$term <- as.character(pooled_summary$term)

standardized_terms <- vapply(
    X = pooled_summary$term,
    FUN = standardize_term,
    FUN.VALUE = character(1),
    USE.NAMES = FALSE
)

pooled_output <- data.frame(
    term = standardized_terms,
    log_or = pooled_summary$estimate,
    se = pooled_summary$std.error,
    z = pooled_summary$statistic,
    p_value = pooled_summary$p.value,
    ci_low_log = pooled_summary$`2.5 %`,
    ci_high_log = pooled_summary$`97.5 %`,
    OR = exp(pooled_summary$estimate),
    CI_low = exp(pooled_summary$`2.5 %`),
    CI_high = exp(pooled_summary$`97.5 %`),
    m_imputations = {int(m)},
    stringsAsFactors = FALSE
)

pooled_output <- pooled_output[order(pooled_output$p_value), ]

write.csv(
    pooled_output,
    output_or_csv,
    row.names = FALSE
)

reference_output <- data.frame(
    variable = names(resolved_references),
    reference_category = unlist(resolved_references, use.names = FALSE),
    stringsAsFactors = FALSE
)

write.csv(
    reference_output,
    output_reference_csv,
    row.names = FALSE
)

cat("\\nSaved pooled OR table:", output_or_csv, "\\n")
cat("Saved reference categories:", output_reference_csv, "\\n")
cat("Saved convergence diagnostics:", output_diagnostics_pdf, "\\n")
cat("Saved logged events:", output_logged_events_csv, "\\n")
cat("Cached MICE object:", output_mids_rds, "\\n")
'''


def run_r_mice_logistic_regression(
    X,
    y,
    predictors,
    aux_cols,
    config,
    output_prefix="mice_logistic_primary_full",
):
    rscript_path = shutil.which("Rscript")

    if rscript_path is None:
        raise RuntimeError(
            "Rscript was not found. Confirm that `!Rscript --version` works "
            "inside a Python notebook cell."
        )

    prepared, predictors, aux_cols = prepare_r_mice_dataframe(
        X=X,
        y=y,
        predictors=predictors,
        aux_cols=aux_cols,
    )

    work_dir = Path(config.output_dir) / "r_mice_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    input_csv = work_dir / "mice_input.csv"
    r_script_path = work_dir / "run_mice_analysis.R"

    pooled_csv = (
        Path(config.output_dir)
        / f"{output_prefix}_pooled_ORs.csv"
    )
    reference_csv = (
        Path(config.output_dir)
        / f"{output_prefix}_reference_categories.csv"
    )
    diagnostics_pdf = (
        Path(config.output_dir)
        / f"{output_prefix}_convergence_diagnostics.pdf"
    )
    logged_events_csv = (
        Path(config.output_dir)
        / f"{output_prefix}_logged_events.csv"
    )
    mids_rds = (
        Path(config.output_dir)
        / (
            f"{output_prefix}_m{config.mice_m}_"
            f"iter{config.mice_max_iter}_mids.rds"
        )
    )

    prepared.to_csv(
        input_csv,
        index=False,
        na_rep="NA",
    )

    r_script = _build_r_mice_script(
        input_csv=input_csv.resolve(),
        output_or_csv=pooled_csv.resolve(),
        output_reference_csv=reference_csv.resolve(),
        output_diagnostics_pdf=diagnostics_pdf.resolve(),
        output_logged_events_csv=logged_events_csv.resolve(),
        output_mids_rds=mids_rds.resolve(),
        predictors=predictors,
        m=config.mice_m,
        maxit=config.mice_max_iter,
        seed=RANDOM_STATE,
    )

    r_script_path.write_text(
        r_script,
        encoding="utf-8",
    )

    print("\nRunning stable R mice analysis...")
    print("R executable:", rscript_path)
    print("Imputations:", config.mice_m)
    print("Iterations:", config.mice_max_iter)
    print("Method for incomplete categorical variables: CART")

    process = subprocess.run(
        [rscript_path, str(r_script_path.resolve())],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    print("\n========== R OUTPUT ==========")
    print(process.stdout or "[No standard output]")

    print("\n========== R WARNINGS/ERRORS ==========")
    print(process.stderr or "[No warnings or errors]")

    if process.returncode != 0:
        raise RuntimeError(
            "R mice failed.\n\n"
            f"Return code: {process.returncode}\n\n"
            f"R output:\n{process.stdout}\n\n"
            f"R warnings/errors:\n{process.stderr}"
        )

    required_outputs = [
        pooled_csv,
        reference_csv,
        diagnostics_pdf,
        logged_events_csv,
        mids_rds,
    ]

    missing_outputs = [
        str(path) for path in required_outputs if not path.exists()
    ]

    if missing_outputs:
        raise FileNotFoundError(
            "R finished but did not create these expected outputs: "
            + ", ".join(missing_outputs)
        )

    pooled_output = pd.read_csv(pooled_csv)
    reference_output = pd.read_csv(reference_csv)
    logged_events = pd.read_csv(logged_events_csv)

    print("\nSaved pooled OR table:", pooled_csv)
    print("Saved reference categories:", reference_csv)
    print("Saved convergence diagnostics:", diagnostics_pdf)
    print("Saved logged events:", logged_events_csv)
    print("Cached MICE object:", mids_rds)

    return pooled_output, reference_output, logged_events


# 11) Interpretation outputs: logistic regression + XGB

# 11) XGBoost interpretation functions
# =========================

def save_primary_xgb_interpretation(
    primary_artifacts,
    X,
    y,
    predictors,
    aux_cols,
    config,
):
    """
    Calculate raw-variable permutation importance for the calibrated
    XGBoost model from the primary full-registry scenario.

    Permutation importance is calculated using the held-out test set and
    the complete calibrated pipeline, so preprocessing and calibration
    remain part of the evaluated prediction process.
    """
    if primary_artifacts is None or "XGB" not in primary_artifacts:
        raise ValueError(
            "XGBoost artifact not found. Run the "
            "primary_full_registry_default scenario first and retain "
            "PRIMARY_ARTIFACTS."
        )

    X_model = X[
        predictors
        + [c for c in aux_cols if c not in predictors]
    ].copy()

    _, X_test, _, y_test = train_test_split(
        X_model,
        y,
        test_size=config.test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    xgb_model = primary_artifacts["XGB"]

    xgb_pi_path = os.path.join(
        config.output_dir,
        "xgb_primary_permutation_importance_raw_features.csv",
    )

    xgb_pi = save_permutation_importance(
        best_model=xgb_model,
        X_test=X_test,
        y_test=y_test,
        predictors=predictors,
        output_path=xgb_pi_path,
    )

    # Diagnostic plot in the main output directory.
    top = xgb_pi.head(20).iloc[::-1].copy()

    fig, ax = plt.subplots(
        figsize=(8, max(5, 0.32 * len(top) + 1.5))
    )

    ax.barh(
        top["feature"],
        top["importance_mean"],
    )

    ax.set_xlabel(
        "Permutation importance: mean decrease in balanced accuracy"
    )
    ax.set_title(
        "XGBoost interpretation: raw-feature permutation importance"
    )

    plt.tight_layout()

    figure_path = os.path.join(
        config.output_dir,
        "xgb_primary_permutation_importance_raw_features.png",
    )

    fig.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(
        "Saved XGBoost permutation importance table:",
        xgb_pi_path,
    )
    print(
        "Saved XGBoost permutation importance plot:",
        figure_path,
    )

    return xgb_pi


# =========================


# Execution


# 12) Sectioned main execution

# 12A) Load data and define scenarios
# =========================
X, y, predictors, aux_cols = load_data(CONFIG)
SCENARIOS = build_scenarios(CONFIG, predictors)

print("Available scenarios:")
for k in SCENARIOS:
    print(" -", k)

print("\nModels to run:", CONFIG.models_to_run)
print("Output directory:", CONFIG.output_dir)

# 12B) Run primary full model
# =========================
PRIMARY_RESULTS, PRIMARY_ARTIFACTS = run_named_scenario(
    "primary_full_registry_default",
    models_to_run=CONFIG.models_to_run
)
PRIMARY_RESULTS

# 12B-1) Paired bootstrap differences in test-set ROC-AUC
# =======================================================

X, y, predictors, aux_cols = load_data(CONFIG)

X_primary = X[
    predictors
    + [c for c in aux_cols if c not in predictors]
].copy()

(
    X_train_auc,
    X_test_auc,
    y_train_auc,
    y_test_auc,
) = train_test_split(
    X_primary,
    y,
    test_size=CONFIG.test_size,
    stratify=y,
    random_state=RANDOM_STATE,
)

# Obtain predictions from the exact same held-out observations.
test_probabilities = {
    model_name: _predict_proba_positive(
        fitted_model,
        X_test_auc,
    )
    for model_name, fitted_model
    in PRIMARY_ARTIFACTS.items()
}

if "Logit" not in test_probabilities:
    raise ValueError(
        "Logistic regression was not found in PRIMARY_ARTIFACTS."
    )


def paired_bootstrap_auc_difference(
    y_true,
    reference_prob,
    comparison_prob,
    n_bootstrap=2000,
    seed=42,
):
    """
    Estimate the paired test-set difference in ROC-AUC:
    comparison model minus reference model.
    """
    y_true = np.asarray(y_true)
    reference_prob = np.asarray(reference_prob)
    comparison_prob = np.asarray(comparison_prob)

    if not (
        len(y_true)
        == len(reference_prob)
        == len(comparison_prob)
    ):
        raise ValueError(
            "Outcome and probability arrays must have equal lengths."
        )

    observed_reference_auc = roc_auc_score(
        y_true,
        reference_prob,
    )
    observed_comparison_auc = roc_auc_score(
        y_true,
        comparison_prob,
    )

    observed_delta = (
        observed_comparison_auc
        - observed_reference_auc
    )

    rng = np.random.default_rng(seed)
    n = len(y_true)
    bootstrap_deltas = []

    for _ in range(n_bootstrap):
        indices = rng.integers(
            0,
            n,
            size=n,
        )

        y_boot = y_true[indices]

        if np.unique(y_boot).size < 2:
            continue

        reference_auc = roc_auc_score(
            y_boot,
            reference_prob[indices],
        )
        comparison_auc = roc_auc_score(
            y_boot,
            comparison_prob[indices],
        )

        bootstrap_deltas.append(
            comparison_auc - reference_auc
        )

    bootstrap_deltas = np.asarray(
        bootstrap_deltas,
        dtype=float,
    )

    ci_low, ci_high = np.quantile(
        bootstrap_deltas,
        [0.025, 0.975],
    )

    # Two-sided bootstrap p-value for delta = 0.
    proportion_nonpositive = (
        np.sum(bootstrap_deltas <= 0) + 1
    ) / (len(bootstrap_deltas) + 1)

    proportion_nonnegative = (
        np.sum(bootstrap_deltas >= 0) + 1
    ) / (len(bootstrap_deltas) + 1)

    p_value = min(
        1.0,
        2 * min(
            proportion_nonpositive,
            proportion_nonnegative,
        ),
    )

    return {
        "reference_auc": observed_reference_auc,
        "comparison_auc": observed_comparison_auc,
        "delta_auc_vs_logit": observed_delta,
        "delta_auc_ci_low": ci_low,
        "delta_auc_ci_high": ci_high,
        "paired_bootstrap_p_value": p_value,
        "n_test": len(y_true),
        "n_bootstrap_completed": len(
            bootstrap_deltas
        ),
    }


paired_auc_rows = []

for model_name, probabilities in test_probabilities.items():
    if model_name == "Logit":
        continue

    comparison = paired_bootstrap_auc_difference(
        y_true=y_test_auc,
        reference_prob=test_probabilities["Logit"],
        comparison_prob=probabilities,
        n_bootstrap=2000,
        seed=RANDOM_STATE,
    )

    comparison["reference_model"] = "Logit"
    comparison["comparison_model"] = model_name
    paired_auc_rows.append(comparison)

paired_auc_results = (
    pd.DataFrame(paired_auc_rows)
    [
        [
            "comparison_model",
            "reference_model",
            "comparison_auc",
            "reference_auc",
            "delta_auc_vs_logit",
            "delta_auc_ci_low",
            "delta_auc_ci_high",
            "paired_bootstrap_p_value",
            "n_test",
            "n_bootstrap_completed",
        ]
    ]
    .sort_values(
        "delta_auc_vs_logit",
        ascending=False,
    )
    .reset_index(drop=True)
)

paired_auc_path = os.path.join(
    CONFIG.output_dir,
    "primary_paired_bootstrap_auc_differences.csv",
)

paired_auc_results.to_csv(
    paired_auc_path,
    index=False,
)

print(
    "Saved paired AUC comparisons:",
    paired_auc_path,
)

display(paired_auc_results)

# 12B-2) Test-set calibration intercept, slope, and plots
# =======================================================

from sklearn.calibration import calibration_curve

PUB_DIR = os.path.join(
    CONFIG.output_dir,
    "publication_outputs",
)
os.makedirs(PUB_DIR, exist_ok=True)


def estimate_calibration_parameters(
    y_true,
    probabilities,
):
    """
    Estimate calibration intercept and slope on the held-out test set.

    Ideal values:
      intercept = 0
      slope = 1
    """
    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    epsilon = 1e-6
    clipped_probabilities = np.clip(
        probabilities,
        epsilon,
        1 - epsilon,
    )

    logit_probabilities = np.log(
        clipped_probabilities
        / (1 - clipped_probabilities)
    )

    design = sm.add_constant(
        logit_probabilities,
        has_constant="add",
    )

    calibration_model = sm.GLM(
        y_true,
        design,
        family=sm.families.Binomial(),
    ).fit()

    return {
        "calibration_intercept": float(
            calibration_model.params[0]
        ),
        "calibration_intercept_ci_low": float(
            calibration_model.conf_int()[0, 0]
        ),
        "calibration_intercept_ci_high": float(
            calibration_model.conf_int()[0, 1]
        ),
        "calibration_slope": float(
            calibration_model.params[1]
        ),
        "calibration_slope_ci_low": float(
            calibration_model.conf_int()[1, 0]
        ),
        "calibration_slope_ci_high": float(
            calibration_model.conf_int()[1, 1]
        ),
        "brier_score": float(
            brier_score_loss(
                y_true,
                probabilities,
            )
        ),
    }


CALIBRATION_MODELS = [
    "Logit",
    "RF",
    "SVM_RBF",
    "XGB",
]

missing_calibration_models = [
    model_name
    for model_name in CALIBRATION_MODELS
    if model_name not in test_probabilities
]
if missing_calibration_models:
    raise ValueError(
        "Missing retained models for calibration: "
        f"{missing_calibration_models}"
    )

calibration_rows = []

fig, ax = plt.subplots(
    figsize=(7, 7)
)

for model_name in CALIBRATION_MODELS:
    probabilities = test_probabilities[model_name]
    calibration_result = (
        estimate_calibration_parameters(
            y_test_auc,
            probabilities,
        )
    )

    calibration_result["model"] = model_name
    calibration_result["n_test"] = len(y_test_auc)
    calibration_rows.append(
        calibration_result
    )

    observed_fraction, predicted_mean = (
        calibration_curve(
            y_test_auc,
            probabilities,
            n_bins=10,
            strategy="quantile",
        )
    )

    ax.plot(
        predicted_mean,
        observed_fraction,
        marker="o",
        label=model_name,
    )

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1,
    label="Perfect calibration",
)

ax.set_xlabel(
    "Mean predicted probability"
)
ax.set_ylabel(
    "Observed EOBC proportion"
)
ax.set_title(
    "Calibration of final EOBC classification models"
)
ax.legend()
plt.tight_layout()

calibration_plot_path = os.path.join(
    PUB_DIR,
    "primary_test_set_calibration_plot.png",
)

fig.savefig(
    calibration_plot_path,
    dpi=300,
    bbox_inches="tight",
)
fig.savefig(
    calibration_plot_path.replace(
        ".png",
        ".pdf",
    ),
    bbox_inches="tight",
)
plt.close(fig)

calibration_results = (
    pd.DataFrame(calibration_rows)
    [
        [
            "model",
            "n_test",
            "calibration_intercept",
            "calibration_intercept_ci_low",
            "calibration_intercept_ci_high",
            "calibration_slope",
            "calibration_slope_ci_low",
            "calibration_slope_ci_high",
            "brier_score",
        ]
    ]
    .sort_values("model")
    .reset_index(drop=True)
)

calibration_table_path = os.path.join(
    PUB_DIR,
    "primary_test_set_calibration_metrics.csv",
)

calibration_results.to_csv(
    calibration_table_path,
    index=False,
)

print(
    "Saved calibration metrics:",
    calibration_table_path,
)
print(
    "Saved calibration plot:",
    calibration_plot_path,
)

display(calibration_results)

# 12B-3) Run repeated nested stratified cross-validation
# ======================================================

X, y, predictors, aux_cols = load_data(
    CONFIG
)

X_nested_cv = X[
    predictors
    + [
        column
        for column in aux_cols
        if column not in predictors
    ]
].copy()

(
    NESTED_CV_FOLD_RESULTS,
    NESTED_CV_OOF_PREDICTIONS,
    NESTED_CV_TUNING_RESULTS,
) = run_repeated_nested_cv(
    X=X_nested_cv,
    y=y,
    predictors=predictors,
    aux_cols=aux_cols,
    config=CONFIG,
    model_names=[
        "Logit",
        "RF",
        "SVM_RBF",
        "XGB",
    ],
    overwrite=True,
)

(
    NESTED_CV_SUMMARY,
    NESTED_CV_AUC_DIFFERENCES,
) = summarize_repeated_nested_cv(
    fold_results=
        NESTED_CV_FOLD_RESULTS,
    output_dir=
        CONFIG.output_dir,
)

NESTED_CV_AUC_PLOT = (
    plot_nested_cv_auc_distributions(
        fold_results=
            NESTED_CV_FOLD_RESULTS,
        output_dir=
            CONFIG.output_dir,
    )
)

print(
    "\nCompleted nested outer folds:",
    len(
        NESTED_CV_FOLD_RESULTS
    ),
)

print(
    "\nNested-CV model summary:"
)

display(
    NESTED_CV_SUMMARY[
        [
            "model",
            "n_outer_folds",
            "n_repeats",
            "folds_per_repeat",
            "roc_auc_mean",
            "roc_auc_sd",
            "roc_auc_median",
            "roc_auc_min",
            "roc_auc_max",
            "pr_auc_mean",
            "balanced_accuracy_mean",
            "sensitivity_mean",
            "specificity_mean",
            "brier_score_mean",
        ]
    ]
)

print(
    "\nPaired outer-fold ROC-AUC differences "
    "versus logistic regression:"
)

display(
    NESTED_CV_AUC_DIFFERENCES
)

# 12C) Run sensitivity: no employment
# =========================
NO_EMPLOYMENT_RESULTS, _ = run_named_scenario(
    "no_employment",
    models_to_run=CONFIG.models_to_run
)
NO_EMPLOYMENT_RESULTS

# 12D) Run sensitivity: no education
# =========================
NO_EDUCATION_RESULTS, _ = run_named_scenario(
    "no_education",
    models_to_run=CONFIG.models_to_run
)
NO_EDUCATION_RESULTS

# 12E) Run sensitivity: no employment + no education
# =========================
NO_EMPLOYMENT_NO_EDUCATION_RESULTS, _ = run_named_scenario(
    "no_employment_no_education",
    models_to_run=CONFIG.models_to_run
)
NO_EMPLOYMENT_NO_EDUCATION_RESULTS

# 12F) Run domain model: sociodemographic/history only
# =========================
SOCIODEMO_HISTORY_RESULTS, _ = run_named_scenario(
    "sociodemo_history_only",
    models_to_run=CONFIG.models_to_run
)
SOCIODEMO_HISTORY_RESULTS

# 12G) Run domain model: tumor/clinical only
# =========================
TUMOR_CLINICAL_RESULTS, _ = run_named_scenario(
    "tumor_clinical_only",
    models_to_run=CONFIG.models_to_run
)
TUMOR_CLINICAL_RESULTS

# 12H) Run missing-data sensitivity: simple imputation/no explicit missing category
# =========================
SIMPLE_IMPUTATION_RESULTS, _ = run_named_scenario(
    "simple_imputation_no_missing_category",
    models_to_run=CONFIG.models_to_run
)
SIMPLE_IMPUTATION_RESULTS

# 12I) missing-data sensitivity: no targeted KNN
# =========================
# This scenario omits targeted KNN imputation and treats tumor grade categorically; tumor morphology remains nominal in all analyses.
# It is optional. Run only if you want a direct check that targeted KNN is not driving the findings.
NO_TARGETED_KNN_RESULTS, _ = run_named_scenario(
    "no_targeted_knn_registry_default",
    models_to_run=CONFIG.models_to_run
)
NO_TARGETED_KNN_RESULTS

# 12J) missing-data sensitivity: complete-case grade/morphology
# =========================
# This can be useful as a conservative sensitivity analysis, but it may reduce sample size.
COMPLETE_CASE_RESULTS, _ = run_named_scenario(
    "complete_case_grade_morpho",
    models_to_run=CONFIG.models_to_run
)
COMPLETE_CASE_RESULTS

# 12K) Combine completed scenario result files
# =========================
def combine_completed_results(config: EOBCConfig):
    files = [
        f for f in os.listdir(config.output_dir)
        if f.startswith("results_") and f.endswith(".csv")
    ]
    if not files:
        raise FileNotFoundError("No results_*.csv files found in output directory.")
    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(os.path.join(config.output_dir, f)))
        except Exception as e:
            print("Skipped", f, "because", repr(e))
    combined = pd.concat(frames, ignore_index=True)
    combined_path = os.path.join(config.output_dir, "ALL_MODEL_RESULTS_COMPLETED_SCENARIOS.csv")
    combined.to_csv(combined_path, index=False)

    best_by_scenario = (
        combined.sort_values(["scenario", "balanced_accuracy", "roc_auc"], ascending=[True, False, False])
        .groupby("scenario")
        .head(1)
    )
    best_path = os.path.join(config.output_dir, "BEST_MODEL_BY_COMPLETED_SCENARIO.csv")
    best_by_scenario.to_csv(best_path, index=False)

    print("Saved:", combined_path)
    print("Saved:", best_path)
    return combined, best_by_scenario

COMBINED_RESULTS, BEST_BY_SCENARIO = combine_completed_results(CONFIG)
BEST_BY_SCENARIO

# 12L) Run stable R mice and pooled logistic regression
# =========================
# Validation run:
#   CONFIG.mice_m = 5
#   CONFIG.mice_max_iter = 5
#
# Final run after checking outputs and convergence:
#   CONFIG.mice_m = 20
#   CONFIG.mice_max_iter = 5

X, y, predictors, aux_cols = load_data(CONFIG)

(
    MICE_OR_TABLE,
    MICE_REFERENCE_TABLE,
    MICE_LOGGED_EVENTS,
) = run_r_mice_logistic_regression(
    X=X,
    y=y,
    predictors=predictors,
    aux_cols=aux_cols,
    config=CONFIG,
    output_prefix="mice_logistic_primary_full",
)

print("\nReference categories used:")
display(MICE_REFERENCE_TABLE)

print("\nPooled adjusted associations:")
display(MICE_OR_TABLE.head(40))

print("\nR mice logged events:")
display(MICE_LOGGED_EVENTS.head(50))

# 12M) Run XGBoost interpretation after the primary model
# =========================
# Requires PRIMARY_ARTIFACTS from Section 12B.

X, y, predictors, aux_cols = load_data(CONFIG)

XGB_IMPORTANCE = save_primary_xgb_interpretation(
    primary_artifacts=PRIMARY_ARTIFACTS,
    X=X,
    y=y,
    predictors=predictors,
    aux_cols=aux_cols,
    config=CONFIG,
)

XGB_IMPORTANCE.head(20)


# 13) Publication tables and interpretation figures

# 13A) Publication-output settings and clean labels
# =========================
# This block only controls final tables/figures. It does not refit models.

PUBLICATION_MODELS = ["Logit", "RF", "SVM_RBF", "XGB"]

SCENARIO_LABELS = {
    "primary_full_registry_default": "Primary full registry model",
    "no_employment": "Excluding employment",
    "no_education": "Excluding education",
    "no_employment_no_education": "Excluding employment and education",
    "sociodemo_history_only": "Sociodemographic/history variables only",
    "tumor_clinical_only": "Tumor/clinical variables only",
    "simple_imputation_no_missing_category": "Simple-imputation sensitivity",
    "no_targeted_knn_registry_default": "Without targeted KNN imputation for tumor grade",
    "complete_case_grade_morpho": "Complete-case grade/morphology sensitivity",
}

MODEL_LABELS = {
    "Logit": "Logistic regression",
    "RF": "Random forest",
    "SVM_RBF": "Support vector machine",
    "XGB": "Extreme gradient boosting",
    "NB": "Naive Bayes",
    "MLP": "Multilayer perceptron",
}

FEATURE_LABELS = {
    "ydod": "Year of diagnosis",
    "hospital_grp_num": "Hospital at diagnosis",
    "ethnicity4": "Ethnicity",
    "gov_num": "Governorate",
    "religion_grp3": "Religion",
    "edu_level": "Education level",
    "smoke_bi": "Smoking status",
    "job_employed": "Employment status",
    "famhx_bi": "Family history",
    "comorb_YesNo": "Comorbidity status",
    "behav": "Tumor behavior",
    "topo_cat": "Tumor topography",
    "stagedx_num": "Stage at diagnosis",
    "grade_num": "Tumor grade",
    "morpho3": "Tumor morphology",
}

TERM_PREFIX_LABELS = {
    "comorb_YesNo": "Comorbidity",
    "job_employed": "Employment status",
    "ethnicity4": "Ethnicity",
    "edu_level": "Education level",
    "behav": "Tumor behavior",
    "ydod": "Year of diagnosis",
    "gov_num": "Governorate",
    "hospital_grp_num": "Hospital at diagnosis",
    "grade_num": "Tumor grade",
    "topo_cat": "Tumor topography",
    "morpho3": "Tumor morphology",
    "smoke_bi": "Smoking status",
    "religion_grp3": "Religion",
    "stagedx_num": "Stage at diagnosis",
    "famhx_bi": "Family history",
}

def clean_feature_label(x: str) -> str:
    """Clean raw feature names for tables/plots."""
    if pd.isna(x):
        return x
    x = str(x)
    return FEATURE_LABELS.get(x, x.replace("_", " "))

def clean_or_term(term: str) -> str:
    """Clean logistic-regression terms while preserving real category labels."""
    if pd.isna(term):
        return term

    term = str(term)

    if term == "ydod":
        return "Year of diagnosis (per year)"

    if term.startswith("morpho3_"):
        category = term.replace("morpho3_", "", 1)
        return f"Tumor morphology: {category}"

    if term.startswith("grade_num_"):
        category = term.replace("grade_num_", "", 1)
        return f"Tumor grade: {category}"

    for prefix in sorted(
        TERM_PREFIX_LABELS,
        key=len,
        reverse=True,
    ):
        if term.startswith(prefix + "_"):
            category = term[len(prefix) + 1:]
            category = category.replace("_", " ")
            return (
                f"{TERM_PREFIX_LABELS[prefix]}: "
                f"{category}"
            )

        if term == prefix:
            return TERM_PREFIX_LABELS[prefix]

    return term.replace("_", " ")

def format_ci(est, lo, hi, digits=3):
    return f"{est:.{digits}f} ({lo:.{digits}f}–{hi:.{digits}f})"

def format_p(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"

def ensure_publication_dir(config=CONFIG):
    pub_dir = os.path.join(config.output_dir, "publication_outputs")
    os.makedirs(pub_dir, exist_ok=True)
    return pub_dir

PUB_DIR = ensure_publication_dir(CONFIG)
PUB_DIR

# 13B) Main model-comparison table:
# repeated nested cross-validation
# =========================================================

PUB_DIR = os.path.join(
    CONFIG.output_dir,
    "publication_outputs",
)

os.makedirs(
    PUB_DIR,
    exist_ok=True,
)

nested_summary_path = os.path.join(
    CONFIG.output_dir,
    "nested_cv_model_summary.csv",
)

if not os.path.exists(
    nested_summary_path
):
    raise FileNotFoundError(
        f"Cannot find {nested_summary_path}. "
        "Run the repeated nested-CV analysis and its "
        "summary block before running this cell."
    )

nested_results = pd.read_csv(
    nested_summary_path
)

required_columns = {
    "model",
    "n_outer_folds",
    "n_repeats",
    "folds_per_repeat",
    "roc_auc_mean",
    "roc_auc_sd",
    "pr_auc_mean",
    "pr_auc_sd",
    "balanced_accuracy_mean",
    "balanced_accuracy_sd",
    "sensitivity_mean",
    "sensitivity_sd",
    "specificity_mean",
    "specificity_sd",
    "f1_mean",
    "f1_sd",
    "brier_score_mean",
    "brier_score_sd",
}

missing_columns = (
    required_columns
    .difference(
        nested_results.columns
    )
)

if missing_columns:
    raise ValueError(
        "The nested-CV summary is missing required columns: "
        f"{sorted(missing_columns)}"
    )

model_labels = {
    "Logit": "Logistic regression",
    "RF": "Random forest",
    "SVM_RBF": "SVM-RBF",
    "XGB": "XGBoost",
}

model_order = {
    "Logit": 0,
    "RF": 1,
    "SVM_RBF": 2,
    "XGB": 3,
}


def format_mean_sd(
    mean_value,
    sd_value,
    digits=3,
):
    """
    Format nested-CV results as mean ± SD.
    """
    if (
        pd.isna(mean_value)
        or pd.isna(sd_value)
    ):
        return "—"

    return (
        f"{mean_value:.{digits}f} "
        f"± {sd_value:.{digits}f}"
    )


main_model_table = (
    nested_results
    .copy()
)

main_model_table["Model"] = (
    main_model_table["model"]
    .map(model_labels)
    .fillna(
        main_model_table["model"]
    )
)

main_model_table[
    "Internal validation"
] = main_model_table.apply(
    lambda row: (
        f"{int(row['folds_per_repeat'])}-fold nested CV × "
        f"{int(row['n_repeats'])} repeats "
        f"({int(row['n_outer_folds'])} outer folds)"
    ),
    axis=1,
)

main_model_table[
    "ROC-AUC, mean ± SD"
] = main_model_table.apply(
    lambda row: format_mean_sd(
        row["roc_auc_mean"],
        row["roc_auc_sd"],
    ),
    axis=1,
)

main_model_table[
    "PR-AUC, mean ± SD"
] = main_model_table.apply(
    lambda row: format_mean_sd(
        row["pr_auc_mean"],
        row["pr_auc_sd"],
    ),
    axis=1,
)

main_model_table[
    "Balanced accuracy, mean ± SD"
] = main_model_table.apply(
    lambda row: format_mean_sd(
        row[
            "balanced_accuracy_mean"
        ],
        row[
            "balanced_accuracy_sd"
        ],
    ),
    axis=1,
)

main_model_table[
    "Sensitivity, mean ± SD"
] = main_model_table.apply(
    lambda row: format_mean_sd(
        row["sensitivity_mean"],
        row["sensitivity_sd"],
    ),
    axis=1,
)

main_model_table[
    "Specificity, mean ± SD"
] = main_model_table.apply(
    lambda row: format_mean_sd(
        row["specificity_mean"],
        row["specificity_sd"],
    ),
    axis=1,
)

main_model_table[
    "F1 score, mean ± SD"
] = main_model_table.apply(
    lambda row: format_mean_sd(
        row["f1_mean"],
        row["f1_sd"],
    ),
    axis=1,
)

main_model_table[
    "Brier score, mean ± SD"
] = main_model_table.apply(
    lambda row: format_mean_sd(
        row["brier_score_mean"],
        row["brier_score_sd"],
    ),
    axis=1,
)

main_model_table[
    "_model_order"
] = (
    main_model_table["model"]
    .map(model_order)
    .fillna(99)
)

main_model_table = (
    main_model_table[
        [
            "Model",
            "Internal validation",
            "ROC-AUC, mean ± SD",
            "PR-AUC, mean ± SD",
            "Balanced accuracy, mean ± SD",
            "Sensitivity, mean ± SD",
            "Specificity, mean ± SD",
            "F1 score, mean ± SD",
            "Brier score, mean ± SD",
            "_model_order",
        ]
    ]
    .sort_values(
        "_model_order"
    )
    .drop(
        columns="_model_order"
    )
    .reset_index(
        drop=True
    )
)

main_model_table_path = os.path.join(
    PUB_DIR,
    "table_main_nested_cv_model_comparison.csv",
)

main_model_table.to_csv(
    main_model_table_path,
    index=False,
)

print(
    "Saved main nested-CV model-comparison table:",
    main_model_table_path,
)

display(
    main_model_table
)

# 13C) Sensitivity-analysis summary tables
# =========================
# Output 1: Best-performing retained model per scenario.
# Output 2: Logistic regression + XGBoost per scenario for Table 4.
#
# This block is self-contained and can be run after a kernel restart.

import os
import pandas as pd


# --------------------------------------------------
# Output directory
# --------------------------------------------------
PUB_DIR = os.path.join(
    CONFIG.output_dir,
    "publication_outputs",
)

os.makedirs(
    PUB_DIR,
    exist_ok=True,
)


# --------------------------------------------------
# Reload completed scenario results
# --------------------------------------------------
all_results_path = os.path.join(
    CONFIG.output_dir,
    "ALL_MODEL_RESULTS_COMPLETED_SCENARIOS.csv",
)

if not os.path.exists(all_results_path):
    raise FileNotFoundError(
        f"Cannot find {all_results_path}. "
        "The completed scenario results must exist before generating "
        "the sensitivity tables."
    )

all_results = pd.read_csv(
    all_results_path
)

print(
    "Loaded completed scenario results:",
    all_results_path,
)

print(
    "Rows loaded:",
    len(all_results),
)


# --------------------------------------------------
# Local publication labels
# --------------------------------------------------
SCENARIO_LABELS_LOCAL = {
    "primary_full_registry_default":
        "Primary full registry model",

    "no_employment":
        "Excluding employment",

    "no_education":
        "Excluding education",

    "no_employment_no_education":
        "Excluding employment and education",

    "sociodemo_history_only":
        "Sociodemographic/history variables only",

    "tumor_clinical_only":
        "Tumor/clinical variables only",

    "simple_imputation_no_missing_category":
        "Simple-imputation sensitivity",

    "no_targeted_knn_registry_default":
        "Without targeted KNN imputation for tumor grade",

    "complete_case_grade_morpho":
        "Complete-case grade/morphology sensitivity",
}


MODEL_LABELS_LOCAL = {
    "Logit": "LR",
    "RF": "RF",
    "SVM_RBF": "SVM-RBF",
    "XGB": "XGBoost",
    "NB": "NB",
}


# --------------------------------------------------
# Scenarios and models retained for publication
# --------------------------------------------------
retained_scenarios = [
    "primary_full_registry_default",
    "no_employment",
    "no_education",
    "no_employment_no_education",
    "sociodemo_history_only",
    "tumor_clinical_only",
    "simple_imputation_no_missing_category",
    "no_targeted_knn_registry_default",
    "complete_case_grade_morpho",
]

publication_models = [
    "Logit",
    "RF",
    "SVM_RBF",
    "XGB",
]


# --------------------------------------------------
# Validate required columns
# --------------------------------------------------
required_columns = {
    "scenario",
    "model",
    "roc_auc",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "brier_score",
    "n_total",
}

missing_columns = required_columns.difference(
    all_results.columns
)

if missing_columns:
    raise ValueError(
        "The completed-results file is missing required columns: "
        f"{sorted(missing_columns)}\n"
        f"Available columns: {all_results.columns.tolist()}"
    )


# --------------------------------------------------
# Restrict to retained analyses
# --------------------------------------------------
sens = (
    all_results[
        all_results["scenario"].isin(
            retained_scenarios
        )
        & all_results["model"].isin(
            publication_models
        )
    ]
    .copy()
)

if sens.empty:
    raise ValueError(
        "No eligible sensitivity-analysis results were found "
        "in the completed-results file."
    )

sens["Scenario"] = (
    sens["scenario"]
    .map(SCENARIO_LABELS_LOCAL)
    .fillna(sens["scenario"])
)

sens["Model"] = (
    sens["model"]
    .map(MODEL_LABELS_LOCAL)
    .fillna(sens["model"])
)


# --------------------------------------------------
# Output 1: Best-performing model per scenario
# --------------------------------------------------
# Ranking:
# 1. balanced accuracy
# 2. ROC-AUC as tie-breaker

best_sens = (
    sens
    .sort_values(
        [
            "scenario",
            "balanced_accuracy",
            "roc_auc",
        ],
        ascending=[
            True,
            False,
            False,
        ],
    )
    .groupby(
        "scenario",
        as_index=False,
        sort=False,
    )
    .head(1)
    .copy()
)

best_sens["ROC-AUC"] = (
    best_sens["roc_auc"]
    .round(3)
)

best_sens["Balanced accuracy"] = (
    best_sens["balanced_accuracy"]
    .round(3)
)

best_sens["Sensitivity"] = (
    best_sens["sensitivity"]
    .round(3)
)

best_sens["Specificity"] = (
    best_sens["specificity"]
    .round(3)
)

best_sens["Brier score"] = (
    best_sens["brier_score"]
    .round(3)
)

best_sens["n"] = (
    best_sens["n_total"]
    .astype(int)
)

scenario_order = {
    scenario: position
    for position, scenario in enumerate(
        retained_scenarios
    )
}

best_sens["_scenario_order"] = (
    best_sens["scenario"]
    .map(scenario_order)
)

best_sens_table = (
    best_sens[
        [
            "Scenario",
            "n",
            "Model",
            "ROC-AUC",
            "Balanced accuracy",
            "Sensitivity",
            "Specificity",
            "Brier score",
            "_scenario_order",
        ]
    ]
    .sort_values(
        "_scenario_order"
    )
    .drop(
        columns="_scenario_order"
    )
    .reset_index(drop=True)
)

best_sens_path = os.path.join(
    PUB_DIR,
    "table_sensitivity_best_model_by_scenario.csv",
)

best_sens_table.to_csv(
    best_sens_path,
    index=False,
)


# --------------------------------------------------
# Output 2: LR and XGBoost across scenarios
# --------------------------------------------------
logit_xgb = (
    sens[
        sens["model"].isin(
            [
                "Logit",
                "XGB",
            ]
        )
    ]
    .copy()
)

if logit_xgb.empty:
    raise ValueError(
        "No LR or XGBoost sensitivity results were found."
    )

pair_counts = (
    logit_xgb
    .groupby("scenario")["model"]
    .nunique()
)

incomplete_scenarios = (
    pair_counts[
        pair_counts < 2
    ]
    .index
    .tolist()
)

if incomplete_scenarios:
    print(
        "Warning: LR and XGBoost were not both available for:",
        incomplete_scenarios,
    )


logit_xgb["ROC-AUC"] = (
    logit_xgb["roc_auc"]
    .round(3)
)

logit_xgb["Balanced accuracy"] = (
    logit_xgb["balanced_accuracy"]
    .round(3)
)

logit_xgb["Sensitivity"] = (
    logit_xgb["sensitivity"]
    .round(3)
)

logit_xgb["Specificity"] = (
    logit_xgb["specificity"]
    .round(3)
)

logit_xgb["Brier score"] = (
    logit_xgb["brier_score"]
    .round(3)
)

logit_xgb["n"] = (
    logit_xgb["n_total"]
    .astype(int)
)

logit_xgb["_scenario_order"] = (
    logit_xgb["scenario"]
    .map(scenario_order)
)

model_order = {
    "Logit": 0,
    "XGB": 1,
}

logit_xgb["_model_order"] = (
    logit_xgb["model"]
    .map(model_order)
)

logit_xgb_table = (
    logit_xgb[
        [
            "Scenario",
            "n",
            "Model",
            "ROC-AUC",
            "Balanced accuracy",
            "Sensitivity",
            "Specificity",
            "Brier score",
            "_scenario_order",
            "_model_order",
        ]
    ]
    .sort_values(
        [
            "_scenario_order",
            "_model_order",
        ]
    )
    .drop(
        columns=[
            "_scenario_order",
            "_model_order",
        ]
    )
    .reset_index(drop=True)
)

logit_xgb_path = os.path.join(
    PUB_DIR,
    "table_sensitivity_logit_xgb_by_scenario.csv",
)

logit_xgb_table.to_csv(
    logit_xgb_path,
    index=False,
)


# --------------------------------------------------
# Confirm outputs
# --------------------------------------------------
print(
    "\nSaved best-model sensitivity table:",
    os.path.abspath(best_sens_path),
)

print(
    "Saved LR–XGBoost sensitivity table:",
    os.path.abspath(logit_xgb_path),
)

print(
    "\nFiles exist:",
    os.path.exists(best_sens_path),
    os.path.exists(logit_xgb_path),
)

display(
    logit_xgb_table
)


# 13D) Full supplementary performance table for retained models only
# =========================

supp_perf = sens.copy()
supp_perf["ROC-AUC (95% CI)"] = supp_perf.apply(lambda r: format_ci(r["roc_auc"], r["roc_auc_ci_low"], r["roc_auc_ci_high"]), axis=1)
supp_perf["Balanced accuracy (95% CI)"] = supp_perf.apply(lambda r: format_ci(r["balanced_accuracy"], r["balanced_accuracy_ci_low"], r["balanced_accuracy_ci_high"]), axis=1)
supp_perf["Sensitivity (95% CI)"] = supp_perf.apply(lambda r: format_ci(r["sensitivity"], r["sensitivity_ci_low"], r["sensitivity_ci_high"]), axis=1)
supp_perf["Specificity (95% CI)"] = supp_perf.apply(lambda r: format_ci(r["specificity"], r["specificity_ci_low"], r["specificity_ci_high"]), axis=1)
supp_perf["F1 score (95% CI)"] = supp_perf.apply(lambda r: format_ci(r["f1"], r["f1_ci_low"], r["f1_ci_high"]), axis=1)
supp_perf["Brier score (95% CI)"] = supp_perf.apply(lambda r: format_ci(r["brier_score"], r["brier_score_ci_low"], r["brier_score_ci_high"]), axis=1)

supp_cols = [
    "Scenario", "Model", "n_total", "n_train", "n_test", "threshold",
    "ROC-AUC (95% CI)", "Balanced accuracy (95% CI)",
    "Sensitivity (95% CI)", "Specificity (95% CI)",
    "F1 score (95% CI)", "Brier score (95% CI)",
    "imputation_strategy", "targeted_knn", "complete_case_grade_morpho"
]
supp_perf_table = supp_perf[supp_cols].rename(columns={
    "n_total": "n total", "n_train": "n train", "n_test": "n test", "threshold": "threshold"
})
supp_perf_path = os.path.join(PUB_DIR, "supplementary_model_performance_retained_models.csv")
supp_perf_table.to_csv(supp_perf_path, index=False)
supp_perf_table.head()

# 13E) Clean full MICE-pooled OR table and publication-ready forest plot
# =========================

mice_path = os.path.join(
    CONFIG.output_dir,
    "mice_logistic_primary_full_pooled_ORs.csv"
)

if not os.path.exists(mice_path):
    raise FileNotFoundError(
        f"Cannot find {mice_path}. Run 12L first."
    )

# Publication output folder
PUB_DIR = os.path.join(CONFIG.output_dir, "publication_outputs")
os.makedirs(PUB_DIR, exist_ok=True)

# -------------------------
# Local formatting helpers
# -------------------------
def format_ci_local(
    estimate,
    ci_low,
    ci_high,
    digits=2,
):
    return (
        f"{estimate:.{digits}f} "
        f"({ci_low:.{digits}f}–{ci_high:.{digits}f})"
    )


def format_p_local(p):
    if pd.isna(p):
        return ""

    if p < 0.001:
        return "<0.001"

    return f"{p:.3f}"


# -------------------------
# Read and clean pooled MICE output
# -------------------------
mice_or = pd.read_csv(mice_path).copy()

required_columns = {
    "term",
    "OR",
    "CI_low",
    "CI_high",
    "p_value",
    "m_imputations",
}

missing_columns = required_columns.difference(
    mice_or.columns
)

if missing_columns:
    raise ValueError(
        "The pooled MICE file is missing required columns: "
        f"{sorted(missing_columns)}\n"
        f"Available columns: {mice_or.columns.tolist()}"
    )

mice_or["Variable/category"] = (
    mice_or["term"]
    .apply(clean_or_term)
)

mice_or["Adjusted OR (95% CI)"] = mice_or.apply(
    lambda row: format_ci_local(
        row["OR"],
        row["CI_low"],
        row["CI_high"],
        digits=2,
    ),
    axis=1,
)

mice_or["p-value"] = (
    mice_or["p_value"]
    .apply(format_p_local)
)

# False-discovery-rate adjustment across reported regression terms
from statsmodels.stats.multitest import multipletests

valid_p_mask = mice_or["p_value"].notna()

mice_or["p_value_fdr_bh"] = np.nan
mice_or["significant_fdr_0_05"] = False

(
    reject_fdr,
    adjusted_p_values,
    _,
    _,
) = multipletests(
    mice_or.loc[
        valid_p_mask,
        "p_value",
    ],
    alpha=0.05,
    method="fdr_bh",
)

mice_or.loc[
    valid_p_mask,
    "p_value_fdr_bh",
] = adjusted_p_values

mice_or.loc[
    valid_p_mask,
    "significant_fdr_0_05",
] = reject_fdr

mice_or_clean = (
    mice_or[
        [
            "term",
            "Variable/category",
            "OR",
            "CI_low",
            "CI_high",
            "p_value",
            "Adjusted OR (95% CI)",
            "p-value",
            "m_imputations",
            "p_value_fdr_bh",
            "significant_fdr_0_05"
        ]
    ]
    .sort_values(
        "p_value",
        ascending=True,
    )
    .reset_index(drop=True)
)

mice_or_table_path = os.path.join(
    PUB_DIR,
    "table_mice_pooled_adjusted_ORs_full.csv"
)

mice_or_clean.to_csv(
    mice_or_table_path,
    index=False,
)

fdr_summary_path = os.path.join(
    PUB_DIR,
    "table_mice_pooled_ORs_with_FDR_adjustment.csv",
)

mice_or_clean.to_csv(
    fdr_summary_path,
    index=False,
)

print(
    "Saved FDR-adjusted MICE results:",
    fdr_summary_path,
)

# -------------------------
# Publication forest plot
# -------------------------
def plot_mice_forest_publication(
    or_df,
    out_path,
    top_n=22,
    include_all_significant=True,
    linear_scale=True,
    show_estimates=True,
):
    df = or_df.copy()

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna(
        subset=[
            "OR",
            "CI_low",
            "CI_high",
            "p_value",
        ]
    )

    df = df[
        (df["OR"] > 0)
        & (df["CI_low"] > 0)
        & (df["CI_high"] > 0)
    ].copy()

    df["abs_log_or"] = np.abs(
        np.log(df["OR"])
    )

    if include_all_significant:
        significant = df[
            df["significant_fdr_0_05"]
        ].copy()

        remaining_slots = max(
            0,
            top_n - len(significant),
        )

        nonsignificant = (
            df[
                ~df["significant_fdr_0_05"]
            ]
            .sort_values(
                "abs_log_or",
                ascending=False,
            )
            .head(remaining_slots)
        )

        plot_df = (
            pd.concat(
                [
                    significant,
                    nonsignificant,
                ],
                axis=0,
            )
            .drop_duplicates("term")
        )

        if len(plot_df) > top_n:
            plot_df = (
                plot_df
                .sort_values(
                    "p_value",
                    ascending=True,
                )
                .head(top_n)
            )

    else:
        plot_df = (
            df
            .sort_values(
                "abs_log_or",
                ascending=False,
            )
            .head(top_n)
        )

    plot_df = (
        plot_df
        .sort_values(
            "OR",
            ascending=True,
        )
        .copy()
    )

    y_positions = np.arange(
        len(plot_df)
    )

    fig, ax = plt.subplots(
        figsize=(
            10,
            max(
                6,
                0.38 * len(plot_df) + 1.5,
            ),
        )
    )

    ax.errorbar(
        plot_df["OR"],
        y_positions,
        xerr=[
            plot_df["OR"] - plot_df["CI_low"],
            plot_df["CI_high"] - plot_df["OR"],
        ],
        fmt="o",
        capsize=3,
    )

    ax.axvline(
        1.0,
        linestyle="--",
        linewidth=1,
    )

    ax.set_yticks(
        y_positions
    )

    ax.set_yticklabels(
        plot_df["Variable/category"]
    )

    if linear_scale:
        ax.set_xscale("linear")
        ax.set_xlabel(
            "Adjusted odds ratio for EOBC classification"
        )

        max_ci = float(
            plot_df["CI_high"].max()
        )

        x_upper = min(
            max_ci + 0.6,
            5.0,
        )

        ax.set_xlim(
            0,
            max(
                x_upper,
                1.5,
            ),
        )

    else:
        ax.set_xscale("log")
        ax.set_xlabel(
            "Adjusted odds ratio for EOBC classification "
            "(log scale)"
        )

    ax.set_title(
        "MICE-pooled multivariable logistic regression"
    )

    if show_estimates:
        x_min, x_max = ax.get_xlim()

        text_offset = (
            x_max - x_min
        ) * 0.015

        for position, (_, row) in enumerate(
            plot_df.iterrows()
        ):
            x_text = (
                row["OR"]
                + text_offset
            )

            if x_text > x_max * 0.96:
                x_text = (
                    row["OR"]
                    - (x_max - x_min) * 0.08
                )
                horizontal_alignment = "right"
            else:
                horizontal_alignment = "left"

            ax.text(
                x_text,
                position,
                f"{row['OR']:.2f}",
                va="center",
                ha=horizontal_alignment,
                fontsize=8,
            )

    plt.tight_layout()

    fig.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        out_path.replace(
            ".png",
            ".pdf",
        ),
        bbox_inches="tight",
    )

    plt.close(fig)

    return plot_df


mice_forest_plot_path = os.path.join(
    PUB_DIR,
    "figure_mice_logistic_forest_publication.png"
)

mice_forest_plot_data = (
    plot_mice_forest_publication(
        mice_or,
        mice_forest_plot_path,
        top_n=22,
        include_all_significant=True,
        linear_scale=True,
        show_estimates=True,
    )
)

mice_forest_plot_data.to_csv(
    os.path.join(
        PUB_DIR,
        "figure_mice_logistic_forest_publication_data.csv"
    ),
    index=False,
)

print(
    "Saved cleaned MICE OR table:",
    mice_or_table_path,
)

print(
    "Saved publication forest plot:",
    mice_forest_plot_path,
)

display(
    mice_or_clean.head(20)
)

# 13F) XGBoost raw-feature permutation importance table and plot
# =========================

xgb_pi_path = os.path.join(
    CONFIG.output_dir,
    "xgb_primary_permutation_importance_raw_features.csv",
)

if not os.path.exists(xgb_pi_path):
    raise FileNotFoundError(
        f"Cannot find {xgb_pi_path}. Run Section 12M first."
    )

xgb_pi = pd.read_csv(xgb_pi_path).copy()

xgb_pi["Feature"] = xgb_pi["feature"].apply(
    clean_feature_label
)
xgb_pi["Importance"] = xgb_pi["importance_mean"]
xgb_pi["SD"] = xgb_pi["importance_sd"]

xgb_pi_clean = (
    xgb_pi[
        [
            "feature",
            "Feature",
            "Importance",
            "SD",
        ]
    ]
    .sort_values(
        "Importance",
        ascending=False,
    )
    .reset_index(drop=True)
)

xgb_pi_clean_path = os.path.join(
    PUB_DIR,
    "table_xgb_permutation_importance_raw_features.csv",
)

xgb_pi_clean.to_csv(
    xgb_pi_clean_path,
    index=False,
)


def plot_xgb_permutation_publication(
    importance_df,
    out_path,
    top_n=12,
):
    plot_df = (
        importance_df
        .sort_values(
            "Importance",
            ascending=False,
        )
        .head(top_n)
        .iloc[::-1]
        .copy()
    )

    fig, ax = plt.subplots(
        figsize=(
            8,
            max(4.5, 0.38 * len(plot_df) + 1.2),
        )
    )

    ax.barh(
        plot_df["Feature"],
        plot_df["Importance"],
    )

    ax.set_xlabel(
        "Mean decrease in balanced accuracy"
    )
    ax.set_title(
        "XGBoost permutation importance"
    )

    plt.tight_layout()

    fig.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        out_path.replace(".png", ".pdf"),
        bbox_inches="tight",
    )
    plt.close(fig)

    return plot_df


xgb_pi_plot_path = os.path.join(
    PUB_DIR,
    "figure_xgb_permutation_importance_publication.png",
)

xgb_pi_plot_data = plot_xgb_permutation_publication(
    xgb_pi_clean,
    xgb_pi_plot_path,
    top_n=12,
)

xgb_pi_plot_data.to_csv(
    os.path.join(
        PUB_DIR,
        "figure_xgb_permutation_importance_publication_data.csv",
    ),
    index=False,
)

print(
    "Saved XGBoost permutation-importance table:",
    xgb_pi_clean_path,
)
print(
    "Saved XGBoost permutation-importance plot:",
    xgb_pi_plot_path,
)

xgb_pi_clean.head(15)

# 13G) XGBoost SHAP helpers for transformed features
# ===================================================
# This section explains the calibrated XGBoost model from the primary
# full-registry scenario. SHAP values are calculated separately for each
# fitted calibration-fold XGBoost estimator and then combined after
# alignment by transformed feature name.


def _get_calibrated_base_pipelines(
    artifacts,
    model_name="XGB",
):
    """
    Recover the fitted base predictive pipeline from a calibrated model.

    The final calibration framework uses ensemble=False, so one final
    base estimator is fitted on the complete training dataset after
    cross-validated calibration. A list is returned for compatibility
    with downstream SHAP code.
    """
    if artifacts is None or model_name not in artifacts:
        raise ValueError(
            f"{model_name} was not found in PRIMARY_ARTIFACTS. "
            "Run the primary scenario first."
        )

    calibrated_model = artifacts[model_name]

    if not hasattr(
        calibrated_model,
        "calibrated_classifiers_",
    ):
        raise ValueError(
            "Expected a fitted CalibratedClassifierCV object."
        )

    pipelines = []

    for calibrated_classifier in (
        calibrated_model.calibrated_classifiers_
    ):
        fitted_estimator = getattr(
            calibrated_classifier,
            "estimator",
            None,
        )

        if fitted_estimator is None:
            fitted_estimator = getattr(
                calibrated_classifier,
                "base_estimator",
                None,
            )

        if fitted_estimator is None:
            raise ValueError(
                "Could not recover the fitted base estimator "
                "from CalibratedClassifierCV."
            )

        pipelines.append(fitted_estimator)

    if not pipelines:
        raise ValueError(
            "No fitted XGBoost pipeline was recovered."
        )

    if len(pipelines) != 1:
        print(
            "Warning: expected one fitted base estimator with "
            f"ensemble=False, but recovered {len(pipelines)}."
        )

    return pipelines


def _clean_transformed_feature_name(
    x,
):
    """
    Convert transformed sklearn feature names into publication labels.
    """
    x = str(x)

    for prefix in [
        "num__",
        "ord__",
        "cat__",
        "remainder__",
        "prep__",
    ]:
        x = x.replace(prefix, "")

    x = x.replace(
        "missingindicator_",
        "",
    )

    x = x.replace(
        "_nan",
        "_Unknown or missing",
    )

    local_feature_labels = {
        "ydod": "Year of diagnosis",
        "hospital_grp_num": "Hospital at diagnosis",
        "ethnicity4": "Ethnicity",
        "gov_num": "Governorate",
        "religion_grp3": "Religion",
        "edu_level": "Education level",
        "smoke_bi": "Smoking status",
        "job_employed": "Employment status",
        "famhx_bi": "Family history",
        "comorb_YesNo": "Comorbidity status",
        "behav": "Tumor behavior",
        "topo_cat": "Tumor topography",
        "stagedx_num": "Stage at diagnosis",
        "grade_num": "Tumor grade",
        "morpho3": "Tumor morphology",
    }

    for raw, label in sorted(
        local_feature_labels.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if x == raw:
            return label

        if x.startswith(raw + "_"):
            category = (
                x[len(raw) + 1:]
                .replace("_", " ")
            )

            if category in {
                (
                    "Breast, NOS + rare sites "
                    "(central, nipple, axillary tail, overlapping)"
                ),
                "Breast, NOS + rare sites",
                "Breast, NOS & rare sites",
            }:
                category = (
                    "Breast, NOS & rare sites"
                )

            if category in {
                "Unknown or missing",
                "Unknown",
                "Missing",
            }:
                category = (
                    "Unknown or missing"
                )

            return (
                f"{label}: {category}"
            )

    return x.replace("_", " ")


def _manual_feature_names_from_preprocessor(
    preprocessor,
):
    """
    Recover transformed feature names when sklearn cannot generate them
    automatically.
    """
    names = []

    for (
        name,
        transformer,
        columns,
    ) in preprocessor.transformers_:
        if (
            name == "remainder"
            and transformer == "drop"
        ):
            continue

        if columns is None:
            continue

        columns = list(columns)

        final_transformer = transformer

        if isinstance(
            transformer,
            Pipeline,
        ):
            final_transformer = (
                transformer.steps[-1][1]
            )

        if hasattr(
            final_transformer,
            "get_feature_names_out",
        ):
            try:
                transformed_names = (
                    final_transformer
                    .get_feature_names_out(
                        columns
                    )
                )

                names.extend(
                    list(transformed_names)
                )

                continue
            except Exception:
                pass

        names.extend(columns)

    return names


def _get_transformed_feature_names(
    base_pipeline,
):
    """
    Extract transformed feature names from the fitted ColumnTransformer.
    """
    preprocessing = (
        base_pipeline.named_steps.get(
            "prep"
        )
    )

    if preprocessing is None:
        raise ValueError(
            "Could not locate the fitted 'prep' step "
            "inside the XGBoost pipeline."
        )

    try:
        feature_names = list(
            preprocessing
            .get_feature_names_out()
        )
    except Exception:
        feature_names = (
            _manual_feature_names_from_preprocessor(
                preprocessing
            )
        )

    return feature_names


def _normalise_binary_shap_values(
    shap_output,
):
    """
    Convert different SHAP binary-classification output structures into
    a two-dimensional observations-by-features array.
    """
    if hasattr(
        shap_output,
        "values",
    ):
        shap_output = (
            shap_output.values
        )

    if isinstance(
        shap_output,
        list,
    ):
        if len(shap_output) == 2:
            return np.asarray(
                shap_output[1]
            )

        return np.asarray(
            shap_output[0]
        )

    shap_array = np.asarray(
        shap_output
    )

    if shap_array.ndim == 3:
        # observations × features × classes
        if shap_array.shape[2] == 2:
            return shap_array[:, :, 1]

        # classes × observations × features
        if shap_array.shape[0] == 2:
            return shap_array[1, :, :]

    if shap_array.ndim != 2:
        raise ValueError(
            "Unexpected SHAP array shape: "
            f"{shap_array.shape}"
        )

    return shap_array


def compute_xgb_shap_transformed(
    primary_artifacts,
    X,
    y,
    predictors,
    aux_cols,
    config,
    sample_size=None,
):
    """
    Calculate transformed-feature SHAP values for the calibrated primary
    XGBoost model.

    Signed SHAP contributions are averaged across calibration-fold base
    estimators for visualization. Global mean absolute SHAP importance is
    calculated separately within each estimator before averaging, preventing
    cancellation of positive and negative contributions across folds.
    """
    if not HAS_SHAP:
        raise ImportError(
            "The shap package is not installed. "
            "Install it with: pip install shap"
        )

    sample_size = (
        sample_size
        if sample_size is not None
        else config.shap_sample_size
    )

    model_columns = (
        predictors
        + [
            c
            for c in aux_cols
            if c not in predictors
        ]
    )

    X_model = X[
        model_columns
    ].copy()

    (
        _,
        X_test,
        _,
        y_test,
    ) = train_test_split(
        X_model,
        y,
        test_size=config.test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    sample_n = min(
        int(sample_size),
        len(X_test),
    )

    X_sample = X_test.sample(
        n=sample_n,
        random_state=RANDOM_STATE,
    ).copy()

    base_pipelines = (
        _get_calibrated_base_pipelines(
            primary_artifacts,
            model_name="XGB",
        )
    )

    shap_frames = []
    transformed_frames = []

    expected_clean_names = None
    original_feature_names = None

    for (
        fold_number,
        base_pipeline,
    ) in enumerate(
        base_pipelines,
        start=1,
    ):
        if (
            "clf"
            not in base_pipeline.named_steps
        ):
            raise ValueError(
                "The recovered XGBoost pipeline does not "
                "contain a 'clf' step."
            )

        preprocessing_pipeline = (
            Pipeline(
                base_pipeline.steps[:-1]
            )
        )

        xgb_classifier = (
            base_pipeline
            .named_steps["clf"]
        )

        X_transformed = (
            preprocessing_pipeline
            .transform(
                X_sample
            )
        )

        if hasattr(
            X_transformed,
            "toarray",
        ):
            X_transformed = (
                X_transformed.toarray()
            )

        X_transformed = np.asarray(
            X_transformed,
            dtype=float,
        )

        feature_names = (
            _get_transformed_feature_names(
                base_pipeline
            )
        )

        if (
            len(feature_names) != X_transformed.shape[1]
        ):
            raise ValueError(
                "The number of transformed feature names "
                "does not match the transformed matrix width. "
                f"Names: {len(feature_names)}; "
                f"matrix columns: {X_transformed.shape[1]}"
            )

        clean_names = [
            _clean_transformed_feature_name(
                name
            )
            for name in feature_names
        ]

        clean_name_series = pd.Series(
            clean_names
        )

        duplicated_names = (
            clean_name_series[
                clean_name_series.duplicated(
                    keep=False
                )
            ]
            .unique()
            .tolist()
        )

        if duplicated_names:
            raise ValueError(
                "Duplicate cleaned SHAP feature names "
                "were produced: "
                f"{sorted(duplicated_names)}"
            )

        X_transformed_df = pd.DataFrame(
            X_transformed,
            columns=clean_names,
            index=X_sample.index,
        )

        explainer = shap.TreeExplainer(
            xgb_classifier
        )

        raw_shap_values = (
            explainer.shap_values(
                X_transformed_df
            )
        )

        fold_shap_values = (
            _normalise_binary_shap_values(
                raw_shap_values
            )
        )

        if (
            fold_shap_values.shape != X_transformed_df.shape
        ):
            raise ValueError(
                "SHAP shape mismatch in calibration "
                f"fold {fold_number}: "
                f"SHAP {fold_shap_values.shape}, "
                f"features {X_transformed_df.shape}"
            )

        fold_shap_df = pd.DataFrame(
            fold_shap_values,
            columns=clean_names,
            index=X_sample.index,
        )

        if expected_clean_names is None:
            expected_clean_names = list(
                clean_names
            )

            original_feature_names = list(
                feature_names
            )

        else:
            missing_names = (
                set(expected_clean_names)
                - set(clean_names)
            )

            extra_names = (
                set(clean_names)
                - set(expected_clean_names)
            )

            if (
                missing_names
                or extra_names
            ):
                raise ValueError(
                    "Transformed feature sets differed "
                    "across calibration folds. "
                    f"Missing: {sorted(missing_names)}; "
                    f"extra: {sorted(extra_names)}"
                )

            fold_shap_df = (
                fold_shap_df[
                    expected_clean_names
                ]
            )

            X_transformed_df = (
                X_transformed_df[
                    expected_clean_names
                ]
            )

        shap_frames.append(
            fold_shap_df
        )

        transformed_frames.append(
            X_transformed_df
        )

    if not shap_frames:
        raise ValueError(
            "No SHAP values were produced."
        )

    # Average signed SHAP values across calibration-fold estimators.
    # These values are retained for the beeswarm visualization.
    averaged_shap = np.mean(
        np.stack(
            [
                frame.to_numpy()
                for frame in shap_frames
            ],
            axis=0,
        ),
        axis=0,
    )

    # Average transformed values for consistent beeswarm coloring.
    averaged_transformed = np.mean(
        np.stack(
            [
                frame.to_numpy()
                for frame
                in transformed_frames
            ],
            axis=0,
        ),
        axis=0,
    )

    averaged_shap_df = pd.DataFrame(
        averaged_shap,
        columns=expected_clean_names,
        index=X_sample.index,
    )

    averaged_transformed_df = pd.DataFrame(
        averaged_transformed,
        columns=expected_clean_names,
        index=X_sample.index,
    )

    # Calculate mean absolute SHAP importance inside each fitted
    # calibration-fold estimator before averaging across estimators.
    # This avoids cancellation of signed contributions across folds.
    fold_mean_abs_shap = np.vstack(
        [
            np.abs(
                frame.to_numpy()
            ).mean(axis=0)
            for frame in shap_frames
        ]
    )

    mean_abs_shap_across_folds = (
        fold_mean_abs_shap.mean(
            axis=0
        )
    )

    if len(shap_frames) > 1:
        sd_mean_abs_shap_across_folds = (
            fold_mean_abs_shap.std(
                axis=0,
                ddof=1,
            )
        )
    else:
        sd_mean_abs_shap_across_folds = (
            np.zeros(
                fold_mean_abs_shap.shape[1],
                dtype=float,
            )
        )

    shap_importance = (
        pd.DataFrame({
            "feature":
                original_feature_names,
            "clean_feature":
                expected_clean_names,
            "mean_abs_shap":
                mean_abs_shap_across_folds,
            "sd_mean_abs_shap_across_folds":
                sd_mean_abs_shap_across_folds,
        })
        .sort_values(
            "mean_abs_shap",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return {
        "X_trans":
            averaged_transformed_df,
        "X_sample":
            X_sample,
        "y_sample":
            y_test.loc[
                X_sample.index
            ].copy(),
        "shap_values":
            averaged_shap_df.to_numpy(),
        "feature_names":
            list(
                original_feature_names
            ),
        "clean_feature_names":
            list(
                expected_clean_names
            ),
        "shap_importance":
            shap_importance,
        "n_calibration_estimators":
            len(base_pipelines),
        "sample_size":
            len(X_sample),
    }

# 13H) Run XGBoost SHAP analysis
# =========================

X, y, predictors, aux_cols = load_data(CONFIG)

XGB_SHAP_RESULTS = compute_xgb_shap_transformed(
    primary_artifacts=PRIMARY_ARTIFACTS,
    X=X,
    y=y,
    predictors=predictors,
    aux_cols=aux_cols,
    config=CONFIG,
    sample_size=CONFIG.shap_sample_size,  # 750
)

print(
    "SHAP observations used:",
    len(XGB_SHAP_RESULTS["X_sample"]),
)

xgb_shap_importance = (
    XGB_SHAP_RESULTS["shap_importance"]
    .copy()
)

xgb_shap_table_path = os.path.join(
    PUB_DIR,
    "table_xgb_shap_mean_abs_transformed_features.csv",
)

xgb_shap_importance.to_csv(
    xgb_shap_table_path,
    index=False,
)

# -------------------------
# SHAP importance bar plot
# -------------------------
top_xgb_shap = (
    xgb_shap_importance
    .head(20)
    .iloc[::-1]
    .copy()
)

fig, ax = plt.subplots(
    figsize=(
        9,
        max(
            6,
            0.34 * len(top_xgb_shap) + 1.4,
        ),
    )
)

ax.barh(
    top_xgb_shap["clean_feature"],
    top_xgb_shap["mean_abs_shap"],
)

ax.set_xlabel(
    "Mean absolute SHAP value"
)
ax.set_title(
    "XGBoost SHAP importance"
)

plt.tight_layout()

xgb_shap_bar_path = os.path.join(
    PUB_DIR,
    "figure_xgb_shap_bar_transformed_features.png",
)

fig.savefig(
    xgb_shap_bar_path,
    dpi=300,
    bbox_inches="tight",
)
fig.savefig(
    xgb_shap_bar_path.replace(
        ".png",
        ".pdf",
    ),
    bbox_inches="tight",
)
plt.close(fig)

# -------------------------
# SHAP beeswarm plot
# -------------------------
try:
    X_shap_df = XGB_SHAP_RESULTS["X_trans"]

    plt.figure()

    shap.summary_plot(
        XGB_SHAP_RESULTS["shap_values"],
        X_shap_df,
        max_display=20,
        show=False,
    )

    xgb_shap_beeswarm_path = os.path.join(
        PUB_DIR,
        "figure_xgb_shap_beeswarm_transformed_features.png",
    )

    plt.savefig(
        xgb_shap_beeswarm_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        xgb_shap_beeswarm_path.replace(
            ".png",
            ".pdf",
        ),
        bbox_inches="tight",
    )
    plt.close()

    print(
        "Saved XGBoost SHAP beeswarm:",
        xgb_shap_beeswarm_path,
    )

except Exception as exc:
    print(
        "The XGBoost SHAP beeswarm failed, but the "
        "importance table and bar plot were saved."
    )
    print(
        "Error:",
        repr(exc),
    )

print(
    "Saved XGBoost SHAP table:",
    xgb_shap_table_path,
)
print(
    "Number of calibrated XGBoost estimators interpreted:",
    XGB_SHAP_RESULTS[
        "n_calibration_estimators"
    ],
)

xgb_shap_importance.head(20)

# 13H-1) Aggregate transformed-feature SHAP values
#         to the original predictor-variable level
# =================================================

if "xgb_shap_importance" not in globals():
    raise NameError(
        "xgb_shap_importance is not available. "
        "Run Section 13H first."
    )

if "PUB_DIR" not in globals():
    PUB_DIR = os.path.join(
        CONFIG.output_dir,
        "publication_outputs",
    )
    os.makedirs(
        PUB_DIR,
        exist_ok=True,
    )


def get_original_variable_from_clean_feature(
    clean_feature,
):
    """
    Map a publication-ready transformed-feature label back to its
    original registry predictor.

    Examples
    --------
    'Ethnicity: Arab Kuwaiti' -> 'Ethnicity'
    'Employment status: Employed' -> 'Employment status'
    'Year of diagnosis' -> 'Year of diagnosis'
    'Tumor grade' -> 'Tumor grade'
    """
    clean_feature = str(
        clean_feature
    ).strip()

    original_variables = [
        "Year of diagnosis",
        "Hospital at diagnosis",
        "Ethnicity",
        "Governorate",
        "Religion",
        "Education level",
        "Smoking status",
        "Employment status",
        "Family history",
        "Comorbidity status",
        "Tumor behavior",
        "Tumor topography",
        "Stage at diagnosis",
        "Tumor grade",
        "Tumor morphology",
    ]

    # Exact match is needed for unexpanded numeric/ordinal features,
    # including year of diagnosis and tumor grade.
    if clean_feature in original_variables:
        return clean_feature

    # One-hot encoded categorical features use:
    # "Original variable: category".
    for variable in sorted(
        original_variables,
        key=len,
        reverse=True,
    ):
        if clean_feature.startswith(
            variable + ":"
        ):
            return variable

    raise ValueError(
        "Could not map transformed SHAP feature back to an "
        f"original variable: {clean_feature!r}"
    )


variable_level_shap = (
    xgb_shap_importance[
        [
            "clean_feature",
            "mean_abs_shap",
        ]
    ]
    .copy()
)

variable_level_shap[
    "Original variable"
] = variable_level_shap[
    "clean_feature"
].map(
    get_original_variable_from_clean_feature
)

# Sum the mean absolute SHAP contributions of all transformed
# features belonging to each original predictor, as requested
# for the variable-level interpretation.
variable_level_shap = (
    variable_level_shap
    .groupby(
        "Original variable",
        as_index=False,
        sort=False,
    )[
        "mean_abs_shap"
    ]
    .sum()
    .rename(
        columns={
            "mean_abs_shap":
                "Aggregated mean absolute SHAP value"
        }
    )
    .sort_values(
        "Aggregated mean absolute SHAP value",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)

# Internal consistency check:
# aggregation must preserve the total feature-level importance.
feature_level_total = float(
    xgb_shap_importance[
        "mean_abs_shap"
    ].sum()
)

variable_level_total = float(
    variable_level_shap[
        "Aggregated mean absolute SHAP value"
    ].sum()
)

if not np.isclose(
    feature_level_total,
    variable_level_total,
    rtol=1e-10,
    atol=1e-12,
):
    raise RuntimeError(
        "Variable-level SHAP aggregation did not preserve "
        "the total mean absolute SHAP value. "
        f"Feature-level total={feature_level_total}; "
        f"variable-level total={variable_level_total}."
    )

# Keep full precision in the archived CSV.
variable_level_shap_path = os.path.join(
    PUB_DIR,
    "table_xgb_shap_aggregated_original_variables.csv",
)

variable_level_shap.to_csv(
    variable_level_shap_path,
    index=False,
)

# Create a separate manuscript-ready version rounded to six
# decimals. You can display three decimals in Word if preferred,
# but retaining six here prevents small variables from all
# appearing as zero.
variable_level_shap_manuscript = (
    variable_level_shap.copy()
)

variable_level_shap_manuscript[
    "Aggregated mean absolute SHAP value"
] = (
    variable_level_shap_manuscript[
        "Aggregated mean absolute SHAP value"
    ]
    .round(6)
)

variable_level_shap_manuscript_path = os.path.join(
    PUB_DIR,
    "table_S3_variable_level_SHAP_manuscript_ready.csv",
)

variable_level_shap_manuscript.to_csv(
    variable_level_shap_manuscript_path,
    index=False,
)

print(
    "Saved full-precision variable-level SHAP table:",
    variable_level_shap_path,
)

print(
    "Saved manuscript-ready variable-level SHAP table:",
    variable_level_shap_manuscript_path,
)

print(
    "Feature-level SHAP total:",
    feature_level_total,
)

print(
    "Variable-level SHAP total:",
    variable_level_total,
)

display(
    variable_level_shap_manuscript
)

# =========================
# 13I) Table 1: Baseline characteristics by EOBC status (observed data only)
# =========================

from scipy.stats import chi2_contingency, ttest_ind

# --- Load full observed analytic dataset ---
X, y, predictors, aux_cols = load_data(CONFIG)

# Keep only variables needed for Table 1
table1_vars = [
    "ydod",
    "hospital_grp_num",
    "ethnicity4",
    "gov_num",
    "religion_grp3",
    "edu_level",
    "smoke_bi",
    "job_employed",
    "famhx_bi",
    "comorb_YesNo",
    "behav",
    "topo_cat",
    "stagedx_num",
    "grade_num",
    "morpho3",
]

df_table1 = X[table1_vars].copy()
df_table1["early_onset45"] = y.values

# EOBC labels for readability
df_table1["EOBC_group"] = np.where(df_table1["early_onset45"] == 1, "EOBC (<45)", "Later-onset (≥45)")

# --- Clean labels for Table 1 ---
# Use your existing FEATURE_LABELS / value-label maps if already defined.
# If not, these fallbacks still work.

TABLE1_VAR_LABELS = {
    "ydod": "Year of diagnosis",
    "hospital_grp_num": "Hospital at diagnosis",
    "ethnicity4": "Ethnicity",
    "gov_num": "Governorate",
    "religion_grp3": "Religion",
    "edu_level": "Education level",
    "smoke_bi": "Smoking status",
    "job_employed": "Employment status",
    "famhx_bi": "Family history",
    "comorb_YesNo": "Comorbidity status",
    "behav": "Tumor behavior",
    "topo_cat": "Tumor topography",
    "stagedx_num": "Stage at diagnosis",
    "grade_num": "Tumor grade",
    "morpho3": "Tumor morphology",
}

# Optional category relabeling if you already have cleaner maps in your notebook
TABLE1_CATEGORY_MAPS = {
    # Example:
    # "comorb_YesNo": {"No comorbidity": "No", "Has comorbidity": "Yes"},
}

def fmt_n_pct(n, denom):
    if denom == 0 or pd.isna(denom):
        return "0 (0.0)"
    return f"{int(n)} ({100*n/denom:.1f})"

def fmt_mean_sd(series):
    series = pd.to_numeric(series, errors="coerce").dropna()
    if len(series) == 0:
        return "NA"
    return f"{series.mean():.2f} ± {series.std(ddof=1):.2f}"

def format_p_table1(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"

# Denominators for percentages
n_total = len(df_table1)
n_eobc = (df_table1["early_onset45"] == 1).sum()
n_late = (df_table1["early_onset45"] == 0).sum()

table1_rows = []

# -------------------------
# Continuous variable: Year of diagnosis
# -------------------------
var = "ydod"
overall_vals = pd.to_numeric(df_table1[var], errors="coerce")
eobc_vals = pd.to_numeric(df_table1.loc[df_table1["early_onset45"] == 1, var], errors="coerce")
late_vals = pd.to_numeric(df_table1.loc[df_table1["early_onset45"] == 0, var], errors="coerce")

# Welch t-test
try:
    p_val = ttest_ind(
        eobc_vals.dropna(),
        late_vals.dropna(),
        equal_var=False,
        nan_policy="omit"
    ).pvalue
except Exception:
    p_val = np.nan

table1_rows.append({
    "Characteristic": TABLE1_VAR_LABELS.get(var, var),
    "Category": "",
    "Overall": fmt_mean_sd(overall_vals),
    "EOBC (<45)": fmt_mean_sd(eobc_vals),
    "Later-onset (≥45)": fmt_mean_sd(late_vals),
    "p-value": format_p_table1(p_val),
    "sort_order": 0,
})

# -------------------------
# Categorical variables
# -------------------------
cat_vars = [v for v in table1_vars if v != "ydod"]

sort_counter = 1
for var in cat_vars:
    var_label = TABLE1_VAR_LABELS.get(var, var)

    # Keep observed categories exactly as in data, including Unknown if present
    tmp = df_table1[[var, "early_onset45"]].copy()
    tmp[var] = tmp[var].astype("object")

    # Explicit missing label if true NaN exists
    tmp[var] = tmp[var].fillna("Missing")

    # Apply optional relabeling
    if var in TABLE1_CATEGORY_MAPS:
        tmp[var] = tmp[var].replace(TABLE1_CATEGORY_MAPS[var])

    # Chi-square p-value
    try:
        ct = pd.crosstab(tmp[var], tmp["early_onset45"])
        if ct.shape[0] > 1 and ct.shape[1] == 2:
            p_val = chi2_contingency(ct)[1]
        else:
            p_val = np.nan
    except Exception:
        p_val = np.nan

    # Add header row for variable
    table1_rows.append({
        "Characteristic": var_label,
        "Category": "",
        "Overall": "",
        "EOBC (<45)": "",
        "Later-onset (≥45)": "",
        "p-value": format_p_table1(p_val),
        "sort_order": sort_counter,
    })
    sort_counter += 0.001

    # Counts per category
    cats = list(pd.Series(tmp[var].dropna().unique()).tolist())

    # Optional: sort categories with Unknown/Missing last
    def _cat_sort_key(x):
        x_str = str(x).lower()
        if x_str in ["unknown", "unknown_or_missing", "missing", "not known"]:
            return (1, str(x))
        return (0, str(x))
    cats = sorted(cats, key=_cat_sort_key)

    for cat in cats:
        n_all = (tmp[var] == cat).sum()
        n_e = ((tmp[var] == cat) & (tmp["early_onset45"] == 1)).sum()
        n_l = ((tmp[var] == cat) & (tmp["early_onset45"] == 0)).sum()

        table1_rows.append({
            "Characteristic": "",
            "Category": str(cat),
            "Overall": fmt_n_pct(n_all, n_total),
            "EOBC (<45)": fmt_n_pct(n_e, n_eobc),
            "Later-onset (≥45)": fmt_n_pct(n_l, n_late),
            "p-value": "",
            "sort_order": sort_counter,
        })
        sort_counter += 0.001

    sort_counter = int(sort_counter) + 1

table1_df = pd.DataFrame(table1_rows).sort_values("sort_order").drop(columns="sort_order")

# Save raw table
table1_path = os.path.join(PUB_DIR, "table1_baseline_characteristics_by_eobc.csv")
table1_df.to_csv(table1_path, index=False)

print("Saved Table 1 to:", table1_path)
table1_df.head(40)

# ============================================================
# 13J) Figure 1: ROC curves for primary predictive models
#      Final held-out test set
# ============================================================

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np


# ------------------------------------------------------------
# Reconstruct the exact primary held-out test set
# ------------------------------------------------------------
X, y, predictors, aux_cols = load_data(CONFIG)

X_model = X[
    predictors
    + [c for c in aux_cols if c not in predictors]
].copy()

(
    X_train_roc,
    X_test_roc,
    y_train_roc,
    y_test_roc,
) = train_test_split(
    X_model,
    y,
    test_size=CONFIG.test_size,
    stratify=y,
    random_state=RANDOM_STATE,
)


# ------------------------------------------------------------
# Models and publication labels
# ------------------------------------------------------------
MODEL_LABELS_ROC = {
    "Logit": "LR",
    "RF": "RF",
    "SVM_RBF": "SVM-RBF",
    "XGB": "XGBoost",
}

MODELS_TO_PLOT_ROC = [
    "Logit",
    "RF",
    "SVM_RBF",
    "XGB",
]


# ------------------------------------------------------------
# Generate ROC curves directly from final fitted models
# ------------------------------------------------------------
roc_rows = []
auc_rows = []

fig, ax = plt.subplots(figsize=(7.5, 6.5))

for model_key in MODELS_TO_PLOT_ROC:

    if model_key not in PRIMARY_ARTIFACTS:
        raise KeyError(
            f"{model_key} was not found in PRIMARY_ARTIFACTS. "
            "Run the final primary-model fitting block first."
        )

    fitted_model = PRIMARY_ARTIFACTS[model_key]

    # Use the same probability extraction helper as the
    # final held-out performance analysis.
    y_prob = _predict_proba_positive(
        fitted_model,
        X_test_roc,
    )

    # Calculate ROC curve and held-out ROC-AUC
    fpr, tpr, thresholds = roc_curve(
        y_test_roc,
        y_prob,
    )

    auc_value = roc_auc_score(
        y_test_roc,
        y_prob,
    )

    model_label = MODEL_LABELS_ROC[model_key]

    # Plot ROC curve
    ax.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{model_label} (AUC = {auc_value:.3f})",
    )

    # Save AUC for verification
    auc_rows.append({
        "model": model_label,
        "roc_auc": auc_value,
    })

    # Save complete ROC coordinates
    roc_rows.append(
        pd.DataFrame({
            "model": model_label,
            "fpr": fpr,
            "tpr": tpr,
            "threshold": thresholds,
            "roc_auc": auc_value,
        })
    )


# ------------------------------------------------------------
# No-skill reference line
# ------------------------------------------------------------
ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1,
    label="No-skill (AUC = 0.500)",
)


# ------------------------------------------------------------
# Figure formatting
# ------------------------------------------------------------
ax.set_xlabel(
    "1 - Specificity (False positive rate)"
)

ax.set_ylabel(
    "Sensitivity (True positive rate)"
)

ax.set_title(
    "ROC curves for primary EOBC classification models"
)

ax.legend(
    loc="lower right",
    frameon=True,
)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.grid(alpha=0.25)

plt.tight_layout()


# ------------------------------------------------------------
# Save publication figure
# ------------------------------------------------------------
roc_plot_path = os.path.join(
    PUB_DIR,
    "figure_1_roc_curves_primary_models.png",
)

fig.savefig(
    roc_plot_path,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    roc_plot_path.replace(".png", ".pdf"),
    bbox_inches="tight",
)

plt.show()
plt.close(fig)


# ------------------------------------------------------------
# Save ROC coordinates
# ------------------------------------------------------------
roc_curve_table = pd.concat(
    roc_rows,
    axis=0,
    ignore_index=True,
)

roc_curve_table_path = os.path.join(
    PUB_DIR,
    "figure_1_roc_curves_primary_models_data.csv",
)

roc_curve_table.to_csv(
    roc_curve_table_path,
    index=False,
)


# ------------------------------------------------------------
# Explicit numerical verification
# ------------------------------------------------------------
auc_check = pd.DataFrame(
    auc_rows
).sort_values(
    "roc_auc",
    ascending=False,
).reset_index(drop=True)

print("\nHeld-out ROC-AUC values used in Figure 1:")
display(auc_check)

print("\nRounded values shown in Figure 1:")
for _, row in auc_check.iterrows():
    print(
        f"{row['model']}: "
        f"{row['roc_auc']:.3f}"
    )

print("\nSaved Figure 1:")
print(roc_plot_path)

print("\nSaved ROC coordinates:")
print(roc_curve_table_path)

# 13K) Nested cross-validation publication tables
# =================================================

if "NESTED_CV_SUMMARY" not in globals():
    summary_path = os.path.join(
        CONFIG.output_dir,
        "nested_cv_model_summary.csv",
    )

    paired_path = os.path.join(
        CONFIG.output_dir,
        "nested_cv_auc_differences_vs_logit.csv",
    )

    missing_paths = [
        path
        for path in [
            summary_path,
            paired_path,
        ]
        if not os.path.exists(path)
    ]

    if missing_paths:
        raise FileNotFoundError(
            "Nested-CV outputs were not found. "
            "Run Section 12B-3 first. Missing: "
            f"{missing_paths}"
        )

    NESTED_CV_SUMMARY = pd.read_csv(
        summary_path
    )

    NESTED_CV_AUC_DIFFERENCES = pd.read_csv(
        paired_path
    )


nested_model_labels = {
    "Logit": "Logistic regression",
    "RF": "Random forest",
    "SVM_RBF": "SVM-RBF",
    "XGB": "XGBoost",
}


nested_cv_table = (
    NESTED_CV_SUMMARY[
        [
            "model",
            "n_outer_folds",
            "n_repeats",
            "folds_per_repeat",
            "roc_auc_mean",
            "roc_auc_sd",
            "pr_auc_mean",
            "balanced_accuracy_mean",
            "sensitivity_mean",
            "specificity_mean",
            "f1_mean",
            "brier_score_mean",
        ]
    ]
    .copy()
)

nested_cv_table[
    "Model"
] = (
    nested_cv_table[
        "model"
    ]
    .map(
        nested_model_labels
    )
    .fillna(
        nested_cv_table[
            "model"
        ]
    )
)

nested_cv_table[
    "Validation"
] = nested_cv_table.apply(
    lambda row: (
        f"{int(row['folds_per_repeat'])}-fold outer CV × "
        f"{int(row['n_repeats'])} repeats "
        f"({int(row['n_outer_folds'])} outer folds)"
    ),
    axis=1,
)

nested_cv_table[
    "ROC-AUC, mean ± SD"
] = nested_cv_table.apply(
    lambda row: (
        f"{row['roc_auc_mean']:.3f} "
        f"± {row['roc_auc_sd']:.3f}"
    ),
    axis=1,
)

for source_column, display_column in [
    (
        "pr_auc_mean",
        "Mean PR-AUC",
    ),
    (
        "balanced_accuracy_mean",
        "Mean balanced accuracy",
    ),
    (
        "sensitivity_mean",
        "Mean sensitivity",
    ),
    (
        "specificity_mean",
        "Mean specificity",
    ),
    (
        "f1_mean",
        "Mean F1 score",
    ),
    (
        "brier_score_mean",
        "Mean Brier score",
    ),
]:
    nested_cv_table[
        display_column
    ] = nested_cv_table[
        source_column
    ].round(3)

nested_cv_table = nested_cv_table[
    [
        "Model",
        "Validation",
        "ROC-AUC, mean ± SD",
        "Mean PR-AUC",
        "Mean balanced accuracy",
        "Mean sensitivity",
        "Mean specificity",
        "Mean F1 score",
        "Mean Brier score",
    ]
].sort_values(
    "Model"
).reset_index(
    drop=True
)

nested_cv_table_path = os.path.join(
    PUB_DIR,
    "table_nested_cv_model_performance.csv",
)

nested_cv_table.to_csv(
    nested_cv_table_path,
    index=False,
)


paired_nested_table = (
    NESTED_CV_AUC_DIFFERENCES.copy()
)

paired_nested_table[
    "Comparison"
] = paired_nested_table.apply(
    lambda row: (
        f"{nested_model_labels.get(row['comparison_model'], row['comparison_model'])} "
        "vs logistic regression"
    ),
    axis=1,
)

paired_nested_table[
    "Mean ΔROC-AUC"
] = paired_nested_table[
    "mean_delta_auc"
].round(4)

paired_nested_table[
    "SD"
] = paired_nested_table[
    "sd_delta_auc"
].round(4)

paired_nested_table[
    "Empirical 2.5th–97.5th percentile interval"
] = paired_nested_table.apply(
    lambda row: (
        f"{row['delta_auc_p025']:.4f} to "
        f"{row['delta_auc_p975']:.4f}"
    ),
    axis=1,
)

paired_nested_table[
    "Outer folds above logistic regression"
] = paired_nested_table.apply(
    lambda row: (
        f"{int(round(row['proportion_outer_folds_above_logit'] * row['n_paired_outer_folds']))}"
        f"/{int(row['n_paired_outer_folds'])} "
        f"({100 * row['proportion_outer_folds_above_logit']:.0f}%)"
    ),
    axis=1,
)

paired_nested_table = paired_nested_table[
    [
        "Comparison",
        "Mean ΔROC-AUC",
        "SD",
        "Empirical 2.5th–97.5th percentile interval",
        "Outer folds above logistic regression",
    ]
].reset_index(
    drop=True
)

paired_nested_table_path = os.path.join(
    PUB_DIR,
    "table_nested_cv_auc_differences_vs_logit.csv",
)

paired_nested_table.to_csv(
    paired_nested_table_path,
    index=False,
)

print(
    "Saved nested-CV performance table:",
    nested_cv_table_path,
)

print(
    "Saved paired nested-CV comparison table:",
    paired_nested_table_path,
)

display(
    nested_cv_table
)

display(
    paired_nested_table
)


# 13L) Publication-output manifest
# =========================
# This collects all generated table/figure paths for easy manuscript tracking.

manifest_rows = []
for root, _, files in os.walk(PUB_DIR):
    for f in files:
        if f.lower().endswith((".csv", ".png", ".pdf", ".svg")):
            manifest_rows.append({
                "file": f,
                "path": os.path.join(root, f),
                "type": os.path.splitext(f)[1].replace(".", "").lower()
            })

manifest = pd.DataFrame(manifest_rows).sort_values(["type", "file"])
manifest_path = os.path.join(PUB_DIR, "publication_outputs_manifest.csv")
manifest.to_csv(manifest_path, index=False)
manifest

# ============================================================
# 14) Sensitivity analysis:
#      alternative EOBC definition (<40 years)
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sensitivity-analysis output directory
REVIEWER_DIR = os.path.join(
    CONFIG.output_dir,
    "reviewer_age_cutoff_sensitivity"
)
os.makedirs(REVIEWER_DIR, exist_ok=True)

# Load original analytic dataset
df_age = pd.read_csv(CONFIG.csv_path)

# ------------------------------------------------------------
# Validate raw age-at-diagnosis variable
# ------------------------------------------------------------

if "age" not in df_age.columns:
    raise KeyError("Expected raw age variable 'age' was not found.")

df_age["age"] = pd.to_numeric(
    df_age["age"],
    errors="coerce"
)

print("Age-at-diagnosis summary:")
print(df_age["age"].describe())

print(
    f"\nMissing age values: "
    f"{df_age['age'].isna().sum():,} "
    f"({100 * df_age['age'].isna().mean():.2f}%)"
)

# Inspect implausible values before defining the outcome
implausible_age = df_age.loc[
    df_age["age"].notna()
    & ((df_age["age"] < 0) | (df_age["age"] > 120)),
    "age"
]

print(
    f"Implausible age values (<0 or >120): "
    f"{len(implausible_age):,}"
)

# ------------------------------------------------------------
# Construct alternative EOBC definition
# EOBC = diagnosis before age 40
# ------------------------------------------------------------

df_age["early_onset40"] = np.where(
    df_age["age"].notna(),
    (df_age["age"] < 40).astype(int),
    np.nan
)

print("\n<40 EOBC outcome distribution:")
print(
    df_age["early_onset40"]
    .value_counts(dropna=False)
    .sort_index()
)

print(
    f"\nEOBC prevalence using <40 years: "
    f"{100 * df_age['early_onset40'].mean():.2f}%"
)


# 14) Alternative EOBC definition (<40 years)

# ============================================================
# 14B) Audit the <40 versus <45 EOBC definitions
# ============================================================

# Restrict descriptive comparison to records with known age
age_valid = df_age.loc[
    df_age["age"].notna()
].copy()

# Reconstruct the original <45 definition directly from age
age_valid["early_onset45_reconstructed"] = (
    age_valid["age"] < 45
).astype(int)

# ------------------------------------------------------------
# Validate reconstructed <45 outcome against existing outcome
# ------------------------------------------------------------

if "early_onset45" not in age_valid.columns:
    raise KeyError(
        "Existing primary outcome 'early_onset45' was not found."
    )

existing_45 = pd.to_numeric(
    age_valid["early_onset45"],
    errors="coerce"
)

comparison_mask = existing_45.notna()

n_disagree = (
    existing_45.loc[comparison_mask].astype(int)
    != age_valid.loc[
        comparison_mask,
        "early_onset45_reconstructed"
    ]
).sum()

agreement = 1 - (
    n_disagree / comparison_mask.sum()
)

print(
    f"Agreement between existing and reconstructed "
    f"<45 outcome: {agreement:.2%}"
)

print(
    f"Number of discordant records: {n_disagree:,}"
)

# ------------------------------------------------------------
# Create reviewer-relevant age groups
# ------------------------------------------------------------

age_valid["age_cutoff_group"] = np.select(
    [
        age_valid["age"] < 40,
        (age_valid["age"] >= 40)
        & (age_valid["age"] < 45),
        age_valid["age"] >= 45
    ],
    [
        "<40 years",
        "40–44 years",
        "≥45 years"
    ],
    default="Missing"
)

group_order = [
    "<40 years",
    "40–44 years",
    "≥45 years"
]

age_cutoff_table = (
    age_valid["age_cutoff_group"]
    .value_counts()
    .reindex(group_order)
    .rename("n")
    .to_frame()
)

age_cutoff_table["percent_total"] = (
    100
    * age_cutoff_table["n"]
    / len(age_valid)
)

# Number classified as EOBC under each definition
n_under40 = (age_valid["age"] < 40).sum()
n_under45 = (age_valid["age"] < 45).sum()
n_40_44 = (
    (age_valid["age"] >= 40)
    & (age_valid["age"] < 45)
).sum()

pct_40_44_of_original_eobc = (
    100 * n_40_44 / n_under45
)

print("\nAge distribution relevant to cutoff sensitivity:")
display(age_cutoff_table.round(2))

print(f"\nEOBC cases using <40: {n_under40:,}")
print(f"EOBC cases using <45: {n_under45:,}")
print(f"Cases aged 40–44:      {n_40_44:,}")

print(
    f"\nCases aged 40–44 constitute "
    f"{pct_40_44_of_original_eobc:.2f}% "
    f"of all cases classified as EOBC under the <45 definition."
)

# Export
age_cutoff_table.to_csv(
    os.path.join(
        REVIEWER_DIR,
        "age_cutoff_distribution.csv"
    )
)

# ============================================================
# 14C) Age-at-diagnosis distribution with alternative cutoffs
# ============================================================

fig, ax = plt.subplots(figsize=(9, 6))

# Two-year age bins
bin_start = np.floor(age_valid["age"].min())
bin_end = np.ceil(age_valid["age"].max()) + 2

ax.hist(
    age_valid["age"],
    bins=np.arange(bin_start, bin_end, 2),
    edgecolor="black",
    alpha=0.75
)

# Alternative and primary EOBC cutoffs
ax.axvline(
    40,
    linestyle="--",
    linewidth=2,
    label="Alternative cutoff (<40 years)"
)

ax.axvline(
    45,
    linestyle=":",
    linewidth=2,
    label="Primary cutoff (<45 years)"
)

ax.set_xlabel("Age at breast cancer diagnosis (years)")
ax.set_ylabel("Number of cases")
ax.set_title(
    "Distribution of age at diagnosis in the analytic cohort"
)

ax.legend(frameon=False)

fig.tight_layout()

fig.savefig(
    os.path.join(
        REVIEWER_DIR,
        "age_distribution_40_45_cutoffs.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# 14D) Prepare isolated <40 sensitivity-analysis dataset
# ============================================================

from copy import deepcopy

# Keep the primary analysis completely untouched
df_40 = df_age.copy()

# Exclude records without a valid age because the alternative
# outcome cannot be defined for them
df_40 = df_40.loc[
    df_40["age"].notna()
].copy()

# Ensure binary integer outcome
df_40["early_onset40"] = (
    df_40["age"] < 40
).astype(int)

# Save reviewer-specific analytic dataset
reviewer_dataset_path = os.path.join(
    REVIEWER_DIR,
    "analytic_dataset_under40.csv"
)

df_40.to_csv(
    reviewer_dataset_path,
    index=False
)

# Create independent configuration
CONFIG_40 = deepcopy(CONFIG)

CONFIG_40.csv_path = reviewer_dataset_path
CONFIG_40.target = "early_onset40"

CONFIG_40.output_dir = os.path.join(
    REVIEWER_DIR,
    "under40_models"
)

os.makedirs(
    CONFIG_40.output_dir,
    exist_ok=True
)

print("Sensitivity dataset created successfully.")
print(f"N = {len(df_40):,}")
print(f"Outcome = {CONFIG_40.target}")
print(f"EOBC (<40) = {df_40['early_onset40'].sum():,}")
print(
    f"EOBC prevalence = "
    f"{100 * df_40['early_onset40'].mean():.2f}%"
)
print(f"Output directory: {CONFIG_40.output_dir}")

# ============================================================
# 14E) Final validation before running <40 models
# ============================================================

# Load using the same function used by the primary analysis
X40, y40, predictors40, aux_cols40 = load_data(
    CONFIG_40
)

print("=" * 60)
print("EOBC <40 SENSITIVITY ANALYSIS — PRE-MODEL CHECK")
print("=" * 60)

print(f"\nAnalytic N: {len(y40):,}")

print("\nOutcome counts:")
print(
    y40.value_counts()
    .sort_index()
    .rename(index={
        0: "Later-onset (≥40)",
        1: "EOBC (<40)"
    })
)

print(
    f"\nEOBC (<40) prevalence: "
    f"{100 * y40.mean():.2f}%"
)

# Compare with primary <45 outcome
primary_45 = pd.to_numeric(
    df_40["early_onset45"],
    errors="coerce"
)

print(
    f"Primary EOBC (<45) prevalence: "
    f"{100 * primary_45.mean():.2f}%"
)

print(
    f"Absolute prevalence difference: "
    f"{100 * (primary_45.mean() - y40.mean()):.2f} "
    f"percentage points"
)

# Verify predictor set
print(
    "\nPredictor set identical to primary analysis:",
    predictors40 == CONFIG.predictors
)

if predictors40 != CONFIG.predictors:
    print("\nWARNING: Predictor sets differ.")
    print("Under-40 predictors:", predictors40)
    print("Primary predictors:", CONFIG.predictors)

# Verify no outcome leakage
assert "age" not in predictors40, (
    "ERROR: raw age is present among predictors. "
    "Age defines the outcome and must not be used as a predictor."
)

assert "early_onset45" not in predictors40, (
    "ERROR: the original <45 outcome is present among predictors."
)

assert "early_onset40" not in predictors40, (
    "ERROR: the <40 outcome is present among predictors."
)

print("\nOutcome-leakage checks: PASSED")

print("\nReady for <40 predictive sensitivity analysis.")

# ============================================================
# 14F) Reproduce primary 75/25 split using <40 outcome
# ============================================================

# Construct the modelling matrix exactly as in the primary analysis:
# predictors + auxiliary variables used only to support imputation
X40_model = X40[
    predictors40
    + [
        c for c in aux_cols40
        if c not in predictors40
    ]
].copy()

# Use the exact same split specification as the primary analysis
(
    X40_train,
    X40_test,
    y40_train,
    y40_test,
) = train_test_split(
    X40_model,
    y40,
    test_size=CONFIG_40.test_size,
    stratify=y40,
    random_state=RANDOM_STATE,
)

print("=" * 60)
print("EOBC <40 SENSITIVITY ANALYSIS — DEVELOPMENT SPLIT")
print("=" * 60)

print(f"\nFull analytic cohort: {len(X40_model):,}")
print(f"Training set: {len(X40_train):,}")
print(f"Held-out test set: {len(X40_test):,}")

outcome_distribution_40 = pd.DataFrame({
    "Training": y40_train.value_counts().sort_index(),
    "Test": y40_test.value_counts().sort_index(),
}).rename(
    index={
        0: "Later-onset (≥40)",
        1: "EOBC (<40)"
    }
)

print("\nOutcome distribution:")
display(outcome_distribution_40)

print("\nEOBC prevalence:")
print(f"Training: {y40_train.mean():.3%}")
print(f"Test:     {y40_test.mean():.3%}")

# Verify stratification worked as expected
print(
    "\nAbsolute prevalence difference between train and test:",
    f"{abs(y40_train.mean() - y40_test.mean()):.4%}"
)

# Verify dimensions
print(f"\nNumber of modelling columns: {X40_model.shape[1]}")
print(
    f"Expected: {len(predictors40)} predictors + "
    f"{len([c for c in aux_cols40 if c not in predictors40])} "
    f"auxiliary imputation columns"
)

# ============================================================
# 14G) Primary held-out predictive analysis using <40 EOBC
#      definition
#
# Purpose:
# Reproduce the primary <45 predictive modelling framework
# unchanged, with the ONLY substantive change being the
# alternative EOBC outcome definition (<40 years).
# ============================================================

print("=" * 70)
print("EOBC <40 SENSITIVITY ANALYSIS — PRIMARY HELD-OUT MODELS")
print("=" * 70)

# ------------------------------------------------------------
# Safety checks before model fitting
# ------------------------------------------------------------

assert CONFIG_40.target == "early_onset40"

assert predictors40 == CONFIG.predictors, (
    "Predictor set differs from the primary analysis."
)

assert len(y40) == 8429, (
    "Unexpected change in analytic sample size."
)

assert int(y40.sum()) == 1359, (
    "Unexpected number of <40 EOBC cases."
)


UNDER40_RESULTS, UNDER40_ARTIFACTS = run_scenario(
    scenario_name="under40_primary_registry_default",
    X=X40,
    y=y40,
    predictors=predictors40,
    aux_cols=aux_cols40,
    config=CONFIG_40,
    imputation_strategy="registry_default",
    use_targeted_knn=True,
    complete_case_grade_morpho=False,
    models_to_run=CONFIG_40.models_to_run,
)

# ------------------------------------------------------------
# Display results
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("UNDER-40 HELD-OUT MODEL RESULTS")
print("=" * 70)

display(UNDER40_RESULTS)

print(
    "\nResults saved to:",
    os.path.join(
        CONFIG_40.output_dir,
        "results_under40_primary_registry_default.csv",
    )
)

print(
    "Tuning information saved to:",
    os.path.join(
        CONFIG_40.output_dir,
        "tuning_under40_primary_registry_default.json",
    )
)

# ============================================================
# 14I-1) Prepare MICE input for <40 cutoff sensitivity
# ============================================================

import os
import pandas as pd
import numpy as np

# Existing <45 MICE input created by the primary analysis
PRIMARY_MICE_INPUT = os.path.join(
    CONFIG.output_dir,
    "r_mice_work",
    "mice_input.csv"
)

# Reviewer-specific <40 MICE working directory
MICE40_DIR = os.path.join(
    REVIEWER_DIR,
    "mice_under40"
)
os.makedirs(MICE40_DIR, exist_ok=True)

MICE40_INPUT = os.path.join(
    MICE40_DIR,
    "mice_input_under40.csv"
)

# Load the exact prepared dataset used for the original MICE analysis
mice40 = pd.read_csv(
    PRIMARY_MICE_INPUT,
    low_memory=False
)

print("Original MICE input shape:", mice40.shape)

# ------------------------------------------------------------
# Replace the original outcome with the alternative <40 outcome
# ------------------------------------------------------------

if "early_onset45" not in mice40.columns:
    raise KeyError(
        "Expected 'early_onset45' in the original MICE input."
    )

# The prepared MICE file may not contain raw age, so align the
# alternative outcome from the validated analytic dataset.
if len(mice40) != len(df_40):
    raise ValueError(
        "MICE input and validated analytic dataset have "
        "different row counts. Do not proceed."
    )

# Remove primary outcome and insert alternative outcome
mice40 = mice40.drop(
    columns=["early_onset45"]
)

mice40.insert(
    0,
    "early_onset40",
    df_40["early_onset40"].to_numpy(dtype=int)
)

# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------

assert len(mice40) == 8429
assert mice40["early_onset40"].sum() == 1359
assert mice40["early_onset40"].isna().sum() == 0
assert "early_onset45" not in mice40.columns

expected_predictors = [
    "ydod",
    "hospital_grp_num",
    "ethnicity4",
    "gov_num",
    "religion_grp3",
    "edu_level",
    "smoke_bi",
    "job_employed",
    "famhx_bi",
    "comorb_YesNo",
    "behav",
    "topo_cat",
    "stagedx_num",
    "grade_num",
    "morpho3",
]

missing_predictors = [
    x for x in expected_predictors
    if x not in mice40.columns
]

if missing_predictors:
    raise KeyError(
        f"Missing MICE predictors: {missing_predictors}"
    )

print("\n<40 MICE outcome:")
print(mice40["early_onset40"].value_counts())

print(
    f"\nEOBC prevalence: "
    f"{100*mice40['early_onset40'].mean():.2f}%"
)

print(
    "\nAll 15 inferential predictors present:",
    len(missing_predictors) == 0
)

print(
    "Original <45 outcome removed:",
    "early_onset45" not in mice40.columns
)

# Export
mice40.to_csv(
    MICE40_INPUT,
    index=False
)

print(
    "\nSaved <40 MICE input to:",
    MICE40_INPUT
)

# ============================================================
# 14I-2) <40 sensitivity analysis:
#        R mice + pooled multivariable logistic regression
#
# Uses the SAME Python -> Rscript workflow as the primary <45
# analysis. Only the outcome definition is changed.
# ============================================================

import copy
import os
import pandas as pd


# ------------------------------------------------------------
# 1. Create reviewer-specific configuration
# ------------------------------------------------------------

CONFIG_MICE40 = copy.deepcopy(CONFIG)

CONFIG_MICE40.output_dir = os.path.join(
    REVIEWER_DIR,
    "mice_under40",
)

os.makedirs(
    CONFIG_MICE40.output_dir,
    exist_ok=True,
)


# ------------------------------------------------------------
# 2. Prepare X and y exactly as required by the existing
#    run_r_mice_logistic_regression() function
# ------------------------------------------------------------

# Same predictors and auxiliary imputation variables as primary
X_mice40 = X40.copy()

y_mice40 = y40.astype(int).copy()

predictors_mice40 = predictors40.copy()

aux_cols_mice40 = aux_cols40.copy()


# ------------------------------------------------------------
# 3. Safety checks
# ------------------------------------------------------------

assert len(X_mice40) == 8429
assert len(y_mice40) == 8429

assert int(y_mice40.sum()) == 1359

assert abs(
    y_mice40.mean() - (1359 / 8429)
) < 1e-12

assert predictors_mice40 == CONFIG.predictors

assert "age" not in predictors_mice40
assert "early_onset45" not in predictors_mice40
assert "early_onset40" not in predictors_mice40

print("Safety checks PASSED")
print()
print("N:", len(y_mice40))
print(
    "EOBC <40 cases:",
    int(y_mice40.sum()),
)
print(
    "EOBC <40 prevalence:",
    f"{100 * y_mice40.mean():.2f}%"
)
print(
    "Number of inferential predictors:",
    len(predictors_mice40),
)
print(
    "Auxiliary imputation variables:",
    aux_cols_mice40,
)


# ------------------------------------------------------------
# 4. IMPORTANT:
#    The existing helper hard-codes the generated R outcome
#    name as 'early_onset45'.
#
#    That is OK here.
#
#    y_mice40 contains the NEW <40 outcome. The label used
#    internally by R is merely a variable name.
#
#    Therefore:
#
#       R variable called "early_onset45"
#             actually contains y = age < 40
#
#    This lets us reuse the EXACT validated primary MICE
#    implementation without modifying its statistical logic.
# ------------------------------------------------------------


# ------------------------------------------------------------
# 5. Run EXACT same R-mice pipeline used for primary <45
# ------------------------------------------------------------

(
    MICE40_OR_TABLE,
    MICE40_REFERENCE_TABLE,
    MICE40_LOGGED_EVENTS,
) = run_r_mice_logistic_regression(
    X=X_mice40,
    y=y_mice40,
    predictors=predictors_mice40,
    aux_cols=aux_cols_mice40,
    config=CONFIG_MICE40,
    output_prefix="mice_logistic_under40",
)


# ------------------------------------------------------------
# 6. Display results
# ------------------------------------------------------------

print("\n========================================")
print("EOBC <40 MICE ANALYSIS COMPLETE")
print("========================================")

print("\nReference categories used:")
display(
    MICE40_REFERENCE_TABLE
)

print("\nPooled adjusted associations:")
display(
    MICE40_OR_TABLE.head(40)
)

print("\nR mice logged events:")
display(
    MICE40_LOGGED_EVENTS.head(50)
)


# ------------------------------------------------------------
# 7. Final validation
# ------------------------------------------------------------

print("\nFinal validation:")

print(
    "Number of pooled coefficients:",
    len(MICE40_OR_TABLE),
)

if "m_imputations" in MICE40_OR_TABLE.columns:
    print(
        "Number of imputations:",
        sorted(
            MICE40_OR_TABLE[
                "m_imputations"
            ].dropna().unique()
        ),
    )

print(
    "Output directory:",
    CONFIG_MICE40.output_dir,
)

# ============================================================
# 14J) Compare MICE-pooled adjusted associations:
#      primary EOBC <45 vs sensitivity EOBC <40
# ============================================================

import os
import numpy as np
import pandas as pd


# ------------------------------------------------------------
# 1. Locate original <45 MICE pooled-OR table
# ------------------------------------------------------------

candidate_files = [
    "table_mice_pooled_adjusted_ORs_full.csv",
    "table_mice_pooled_adjusted_ORs_full(20261107-015202).csv",
]

primary45_path = None

for candidate in candidate_files:
    if os.path.exists(candidate):
        primary45_path = candidate
        break

# Also search the primary output directory
if primary45_path is None:
    for root, dirs, files in os.walk(CONFIG.output_dir):
        for file in files:
            if (
                file.lower().endswith(".csv")
                and "mice" in file.lower()
                and "pooled" in file.lower()
                and "adjusted" in file.lower()
                and "full" in file.lower()
            ):
                primary45_path = os.path.join(root, file)
                break

        if primary45_path is not None:
            break

if primary45_path is None:
    raise FileNotFoundError(
        "Could not locate the original <45 MICE pooled-OR table. "
        "Place 'table_mice_pooled_adjusted_ORs_full"
        "(20261107-015202).csv' in the notebook directory "
        "and rerun this cell."
    )

print("Using <45 MICE table:")
print(primary45_path)


# ------------------------------------------------------------
# 2. Load <45 and copy <40 results
# ------------------------------------------------------------

mice45 = pd.read_csv(primary45_path)
mice40 = MICE40_OR_TABLE.copy()

print("\n<45 table shape:", mice45.shape)
print("<40 table shape:", mice40.shape)

print("\n<45 columns:")
print(mice45.columns.tolist())

print("\n<40 columns:")
print(mice40.columns.tolist())


# ------------------------------------------------------------
# 3. Standardize common column names
# ------------------------------------------------------------

def standardize_mice_table(df):

    df = df.copy()

    # Standardize possible column-name variants
    rename_map = {
        "estimate": "log_or",
        "std.error": "se",
        "p.value": "p_value",
        "CI_lower": "CI_low",
        "CI_upper": "CI_high",
        "conf.low": "CI_low",
        "conf.high": "CI_high",
    }

    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    # --------------------------------------------------------
    # Reconstruct log-OR if the exported table contains only OR
    # --------------------------------------------------------

    if "log_or" not in df.columns:

        if "OR" not in df.columns:
            raise ValueError(
                "Neither log_or nor OR is available, so the "
                "coefficient cannot be reconstructed."
            )

        if (pd.to_numeric(df["OR"], errors="coerce") <= 0).any():
            raise ValueError(
                "Invalid OR <= 0 detected; cannot take log."
            )

        df["log_or"] = np.log(
            pd.to_numeric(
                df["OR"],
                errors="raise"
            )
        )

        print(
            "Reconstructed log_or from OR using np.log(OR)."
        )

    # Make required numeric columns explicitly numeric
    numeric_columns = [
        "log_or",
        "OR",
        "CI_low",
        "CI_high",
        "p_value",
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="raise"
            )

    required = [
        "term",
        "log_or",
        "p_value",
        "OR",
        "CI_low",
        "CI_high",
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    return df


mice45 = standardize_mice_table(mice45)
mice40 = standardize_mice_table(mice40)

print("\nStandardization successful.")

print(
    "<45 coefficients:",
    len(mice45)
)

print(
    "<40 coefficients:",
    len(mice40)
)

print(
    "\n<45 log-OR reconstruction check:"
)

check45 = np.allclose(
    np.exp(mice45["log_or"]),
    mice45["OR"],
    rtol=1e-10,
    atol=1e-12,
)

print(
    "exp(log_or) reproduces OR:",
    check45
)

assert check45


# ------------------------------------------------------------
# 4. Verify coefficient sets match exactly
# ------------------------------------------------------------

terms45 = set(mice45["term"])
terms40 = set(mice40["term"])

only45 = sorted(terms45 - terms40)
only40 = sorted(terms40 - terms45)

print("\nTerms only in <45:")
print(only45)

print("\nTerms only in <40:")
print(only40)

if only45 or only40:
    raise ValueError(
        "Coefficient sets differ between <45 and <40. "
        "Do not compare until this is resolved."
    )

print("\nCoefficient sets match exactly: PASSED")


# ------------------------------------------------------------
# 5. Select and rename columns
# ------------------------------------------------------------

keep = [
    "term",
    "log_or",
    "p_value",
    "OR",
    "CI_low",
    "CI_high",
]

m45 = mice45[keep].copy().rename(
    columns={
        "log_or": "logOR_45",
        "p_value": "p_45",
        "OR": "OR_45",
        "CI_low": "CI_low_45",
        "CI_high": "CI_high_45",
    }
)

m40 = mice40[keep].copy().rename(
    columns={
        "log_or": "logOR_40",
        "p_value": "p_40",
        "OR": "OR_40",
        "CI_low": "CI_low_40",
        "CI_high": "CI_high_40",
    }
)


# ------------------------------------------------------------
# 6. Merge
# ------------------------------------------------------------

comparison = m45.merge(
    m40,
    on="term",
    how="inner",
    validate="one_to_one",
)

assert len(comparison) == len(m45) == len(m40)


# ------------------------------------------------------------
# 7. BH-FDR adjustment separately within each definition
# ------------------------------------------------------------

comparison["FDR_45"] = np.nan
comparison["FDR_40"] = np.nan

# Exclude intercept if present
test_mask = (
    ~comparison["term"]
    .astype(str)
    .str.lower()
    .isin(["(intercept)", "intercept"])
)

comparison.loc[test_mask, "FDR_45"] = (
    comparison.loc[test_mask, "p_45"]
    .pipe(
        lambda x: pd.Series(
            __import__("statsmodels.stats.multitest",
                       fromlist=["multipletests"])
            .multipletests(
                x,
                method="fdr_bh"
            )[1],
            index=x.index,
        )
    )
)

comparison.loc[test_mask, "FDR_40"] = (
    comparison.loc[test_mask, "p_40"]
    .pipe(
        lambda x: pd.Series(
            __import__("statsmodels.stats.multitest",
                       fromlist=["multipletests"])
            .multipletests(
                x,
                method="fdr_bh"
            )[1],
            index=x.index,
        )
    )
)


# ------------------------------------------------------------
# 8. Quantify effect-size changes
# ------------------------------------------------------------

comparison["delta_logOR_40_minus_45"] = (
    comparison["logOR_40"]
    - comparison["logOR_45"]
)

comparison["abs_delta_logOR"] = (
    comparison["delta_logOR_40_minus_45"].abs()
)

# Ratio of ORs is more interpretable than raw OR subtraction
comparison["OR_ratio_40_vs_45"] = (
    comparison["OR_40"]
    / comparison["OR_45"]
)

comparison["percent_OR_change"] = (
    (comparison["OR_ratio_40_vs_45"] - 1)
    * 100
)


# ------------------------------------------------------------
# 9. Directional agreement
# ------------------------------------------------------------

comparison["same_direction"] = (
    np.sign(comparison["logOR_45"])
    ==
    np.sign(comparison["logOR_40"])
)


# ------------------------------------------------------------
# 10. Raw and FDR significance status
# ------------------------------------------------------------

comparison["sig_raw_45"] = (
    comparison["p_45"] < 0.05
)

comparison["sig_raw_40"] = (
    comparison["p_40"] < 0.05
)

comparison["sig_FDR_45"] = (
    comparison["FDR_45"] < 0.05
)

comparison["sig_FDR_40"] = (
    comparison["FDR_40"] < 0.05
)

comparison["raw_sig_changed"] = (
    comparison["sig_raw_45"]
    != comparison["sig_raw_40"]
)

comparison["FDR_sig_changed"] = (
    comparison["sig_FDR_45"]
    != comparison["sig_FDR_40"]
)


# ------------------------------------------------------------
# 11. Does each CI exclude the null?
# ------------------------------------------------------------

comparison["CI_excludes_1_45"] = (
    (comparison["CI_low_45"] > 1)
    |
    (comparison["CI_high_45"] < 1)
)

comparison["CI_excludes_1_40"] = (
    (comparison["CI_low_40"] > 1)
    |
    (comparison["CI_high_40"] < 1)
)


# ------------------------------------------------------------
# 12. Sort by largest effect-size change
# ------------------------------------------------------------

comparison = comparison.sort_values(
    "abs_delta_logOR",
    ascending=False,
).reset_index(drop=True)


# ------------------------------------------------------------
# 13. Save complete comparison
# ------------------------------------------------------------

comparison_path = os.path.join(
    REVIEWER_DIR,
    "mice_under40",
    "comparison_MICE_ORs_under45_vs_under40.csv",
)

comparison.to_csv(
    comparison_path,
    index=False,
)


# ------------------------------------------------------------
# 14. Summary statistics
# ------------------------------------------------------------

non_intercept = comparison[
    ~comparison["term"]
    .astype(str)
    .str.lower()
    .isin(["(intercept)", "intercept"])
].copy()

print("\n========================================")
print("<45 vs <40 MICE ROBUSTNESS SUMMARY")
print("========================================")

print(
    "\nCoefficients compared:",
    len(non_intercept),
)

print(
    "Same direction:",
    f"{non_intercept['same_direction'].sum()}"
    f"/{len(non_intercept)}"
)

print(
    "Direction changes:",
    int(
        (~non_intercept["same_direction"]).sum()
    ),
)

print(
    "\nRaw p<0.05 under <45:",
    int(non_intercept["sig_raw_45"].sum()),
)

print(
    "Raw p<0.05 under <40:",
    int(non_intercept["sig_raw_40"].sum()),
)

print(
    "Raw significance-status changes:",
    int(non_intercept["raw_sig_changed"].sum()),
)

print(
    "\nFDR-significant under <45:",
    int(non_intercept["sig_FDR_45"].sum()),
)

print(
    "FDR-significant under <40:",
    int(non_intercept["sig_FDR_40"].sum()),
)

print(
    "FDR significance-status changes:",
    int(non_intercept["FDR_sig_changed"].sum()),
)

print(
    "\nMedian absolute change in log-OR:",
    round(
        non_intercept[
            "abs_delta_logOR"
        ].median(),
        3,
    ),
)

print(
    "Maximum absolute change in log-OR:",
    round(
        non_intercept[
            "abs_delta_logOR"
        ].max(),
        3,
    ),
)


# ------------------------------------------------------------
# 15. Display largest changes
# ------------------------------------------------------------

display_columns = [
    "term",
    "OR_45",
    "CI_low_45",
    "CI_high_45",
    "OR_40",
    "CI_low_40",
    "CI_high_40",
    "delta_logOR_40_minus_45",
    "percent_OR_change",
    "same_direction",
    "p_45",
    "p_40",
    "FDR_45",
    "FDR_40",
]

print("\nLargest effect-size changes:")
display(
    non_intercept[
        display_columns
    ].head(15)
)


# ------------------------------------------------------------
# 16. Specifically show direction/significance changes
# ------------------------------------------------------------

changed = non_intercept[
    (~non_intercept["same_direction"])
    |
    non_intercept["raw_sig_changed"]
    |
    non_intercept["FDR_sig_changed"]
].copy()

print(
    "\nCoefficients with direction and/or "
    "significance-status changes:"
)

if len(changed) == 0:
    print("None.")
else:
    display(
        changed[display_columns]
    )

print(
    "\nFull comparison saved to:",
    comparison_path,
)
