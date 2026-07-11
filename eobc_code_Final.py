#!/usr/bin/env python3
"""
Early-Onset Breast Cancer Analysis Pipeline

This script was generated from the final validated Jupyter notebook used for
the study. The patient-level Kuwait Cancer Registry dataset is not included.

Before running:
1. Update EOBCConfig.csv_path to point to the authorized local dataset.
2. Install the Python dependencies from requirements.txt.
3. Install R and the R packages mice, broom, future, and furrr for the pooled
   multiple-imputation analysis.

The workflow includes data preparation, targeted imputation, model tuning,
probability calibration, test-set evaluation, sensitivity analyses, pooled
MICE logistic regression, XGBoost permutation importance, SHAP interpretation,
and publication-output generation.
"""

# =========================

# 0) Imports + settings
# =========================
import os
import json
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# Plotting
import matplotlib.pyplot as plt

# Core sklearn
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    balanced_accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, brier_score_loss
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, BayesianRidge
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
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
warnings.filterwarnings("ignore")

# =========================

# 1) Configuration
# =========================
@dataclass
class EOBCConfig:
    csv_path: str = "early-onset_dta.csv"
    output_dir: str = "eobc_outputs_v3"
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

    # Auxiliary variables support imputation only and are excluded from final models.
    aux_impute_candidates: List[str] = field(default_factory=lambda: [
        "surgery", "radio", "chemo", "hormone", "immuno"
    ])

    numeric_cols: List[str] = field(default_factory=lambda: ["ydod"])

    # Grade is ordered; morphology is nominal and therefore excluded here.
    ordinal_cols: List[str] = field(default_factory=lambda: ["grade_num"])

    test_size: float = 0.25
    n_splits: int = 5
    n_bootstrap: int = 500
    min_recall: float = 0.70

    models_to_run: List[str] = field(
        default_factory=lambda: ["Logit", "RF", "SVM_RBF", "XGB", "NB"]
    )
    include_mlp: bool = False

    # Mixed-type MICE/FCS settings for inferential logistic regression.
    mice_m: int = 20
    mice_max_iter: int = 5
    mice_sample_posterior: bool = True

    permutation_repeats: int = 30
    shap_sample_size: int = 750

CONFIG = EOBCConfig()
os.makedirs(CONFIG.output_dir, exist_ok=True)

# =========================

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


def _extract_ordinal_1to3(s: pd.Series) -> pd.Series:
    """Extract ordered grade values 1, 2, or 3 from codes or labels."""
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")

    extracted = s.astype(str).str.extract(r"([123])", expand=False)
    extracted = pd.to_numeric(extracted, errors="coerce")
    if extracted.notna().any():
        return extracted

    return pd.to_numeric(s, errors="coerce")


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
# =========================
def load_data(config: EOBCConfig) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    df = pd.read_csv(config.csv_path)
    if config.target not in df.columns:
        raise ValueError(f"Target column {config.target!r} not found.")

    df[config.target] = pd.to_numeric(df[config.target], errors="coerce")
    df = df.dropna(subset=[config.target]).copy()
    df[config.target] = df[config.target].astype(int)

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
      - False when targeted KNN is used upstream, because grade/morphology are converted to numeric 1/2/3.
      - True when targeted KNN is off, because raw grade/morphology labels should be one-hot encoded,
        not forced into numeric scaling.
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
# =========================
def make_models(
    config: EOBCConfig,
    predictors: List[str],
    aux_cols: List[str],
    imputation_strategy: str = "registry_default",
    use_targeted_knn: bool = True,
    models_to_run: Optional[List[str]] = None
) -> Dict[str, Pipeline]:
    # If targeted KNN is disabled, grade/morphology remain categorical to prevent
    # errors and avoid arbitrary numeric coding of raw registry labels.
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
            class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "SVM_RBF": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE),
        "NB": GaussianNB(),
    }

    if config.include_mlp:
        models["MLP"] = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
                                      alpha=1e-4, max_iter=1200, random_state=RANDOM_STATE)

    if HAS_XGB:
        models["XGB"] = XGBClassifier(
            objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", random_state=RANDOM_STATE, n_jobs=-1,
            scale_pos_weight=1.0
        )
    else:
        models["HGB"] = HistGradientBoostingClassifier(random_state=RANDOM_STATE, learning_rate=0.05, max_iter=300)

    if models_to_run is None:
        models_to_run = config.models_to_run
    models = {k: v for k, v in models.items() if k in models_to_run or (k == "HGB" and "XGB" in models_to_run and not HAS_XGB)}

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
    if model_name == "MLP":
        return {
            "clf__hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)],
            "clf__alpha": np.logspace(-5, -2, 10),
            "clf__learning_rate_init": np.logspace(-4, -2, 10),
        }
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
    if model_name == "HGB":
        return {
            "clf__learning_rate": np.linspace(0.01, 0.1, 10),
            "clf__max_leaf_nodes": [15, 31, 63],
            "clf__l2_regularization": np.logspace(-4, 0, 10),
            "clf__max_iter": [200, 300, 500],
        }
    return None


# =========================

# 6) Thresholding + evaluation
# =========================
def _predict_proba_positive(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return 1 / (1 + np.exp(-scores))
    raise ValueError("Model has neither predict_proba nor decision_function.")


def best_threshold_precision_at_recall(y_true, prob, min_recall=0.70):
    thresholds = np.unique(np.quantile(prob, np.linspace(0.01, 0.99, 199)))
    best_thr, best_precision = 0.5, -np.inf
    for thr in thresholds:
        pred = (prob >= thr).astype(int)
        rec = recall_score(y_true, pred, zero_division=0)
        if rec >= min_recall:
            prec = precision_score(y_true, pred, zero_division=0)
            if prec > best_precision:
                best_thr, best_precision = thr, prec
    if best_precision == -np.inf:
        # fallback: maximize F1
        f1s = []
        for thr in thresholds:
            f1s.append(f1_score(y_true, (prob >= thr).astype(int), zero_division=0))
        best_thr = thresholds[int(np.argmax(f1s))]
    return float(best_thr)


def compute_metrics(y_true, prob, threshold):
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "threshold": float(threshold),
        "roc_auc": roc_auc_score(y_true, prob),
        "pr_auc": average_precision_score(y_true, prob),
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "sensitivity": recall_score(y_true, pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "precision": precision_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "brier_score": brier_score_loss(y_true, prob),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
    }


def bootstrap_metric_cis(y_true, prob, threshold, n_bootstrap=500, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)
    metrics = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        # skip bootstrap samples with one class only
        if len(np.unique(y_true[idx])) < 2:
            continue
        metrics.append(compute_metrics(y_true[idx], prob[idx], threshold))
    boot = pd.DataFrame(metrics)
    cis = {}
    for col in ["roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "brier_score"]:
        cis[f"{col}_ci_low"] = boot[col].quantile(0.025)
        cis[f"{col}_ci_high"] = boot[col].quantile(0.975)
    return cis


# =========================

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
        "Logit", "RF", "SVM_RBF", "MLP", "HGB"
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


def evaluate_model(name, model, X_train, y_train, X_test, y_test, config):
    calibrated = CalibratedClassifierCV(
        estimator=model,
        method="sigmoid",
        cv=3,
    )
    calibrated.fit(X_train, y_train)

    prob_train = _predict_proba_positive(calibrated, X_train)
    threshold = best_threshold_precision_at_recall(
        y_train,
        prob_train,
        min_recall=config.min_recall,
    )

    prob_test = _predict_proba_positive(calibrated, X_test)
    metrics = compute_metrics(y_test, prob_test, threshold)
    ci = bootstrap_metric_cis(
        y_test,
        prob_test,
        threshold,
        n_bootstrap=config.n_bootstrap,
        seed=RANDOM_STATE,
    )
    metrics.update(ci)
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
# This scenario treats grade/morphology as categorical variables instead of targeted KNN-imputed numeric variables.
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
print(MICE_REFERENCE_TABLE)

print("\nPooled adjusted associations:")
print(MICE_OR_TABLE.head(40))

print("\nR mice logged events:")
print(MICE_LOGGED_EVENTS.head(50))

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


# 13B) Main model-comparison table: primary full model only
# =========================
# Output: Table-ready CSV for main manuscript model comparison.

all_results_path = os.path.join(CONFIG.output_dir, "ALL_MODEL_RESULTS_COMPLETED_SCENARIOS.csv")
if not os.path.exists(all_results_path):
    raise FileNotFoundError(f"Cannot find {all_results_path}. Run scenario cells and 12K first.")

all_results = pd.read_csv(all_results_path)

primary_main = (
    all_results
    .query("scenario == 'primary_full_registry_default' and model in @PUBLICATION_MODELS")
    .copy()
)

primary_main["Model"] = primary_main["model"].map(MODEL_LABELS)
primary_main["ROC-AUC (95% CI)"] = primary_main.apply(lambda r: format_ci(r["roc_auc"], r["roc_auc_ci_low"], r["roc_auc_ci_high"]), axis=1)
primary_main["PR-AUC (95% CI)"] = primary_main.apply(lambda r: format_ci(r["pr_auc"], r["pr_auc_ci_low"], r["pr_auc_ci_high"]), axis=1)
primary_main["Balanced accuracy (95% CI)"] = primary_main.apply(lambda r: format_ci(r["balanced_accuracy"], r["balanced_accuracy_ci_low"], r["balanced_accuracy_ci_high"]), axis=1)
primary_main["Sensitivity (95% CI)"] = primary_main.apply(lambda r: format_ci(r["sensitivity"], r["sensitivity_ci_low"], r["sensitivity_ci_high"]), axis=1)
primary_main["Specificity (95% CI)"] = primary_main.apply(lambda r: format_ci(r["specificity"], r["specificity_ci_low"], r["specificity_ci_high"]), axis=1)
primary_main["F1 score (95% CI)"] = primary_main.apply(lambda r: format_ci(r["f1"], r["f1_ci_low"], r["f1_ci_high"]), axis=1)
primary_main["Brier score (95% CI)"] = primary_main.apply(lambda r: format_ci(r["brier_score"], r["brier_score_ci_low"], r["brier_score_ci_high"]), axis=1)
primary_main["Threshold"] = primary_main["threshold"].round(3)
primary_main["n test"] = primary_main["n_test"].astype(int)

main_cols = [
    "Model", "n test", "Threshold",
    "ROC-AUC (95% CI)", "PR-AUC (95% CI)", "Balanced accuracy (95% CI)",
    "Sensitivity (95% CI)", "Specificity (95% CI)", "F1 score (95% CI)", "Brier score (95% CI)"
]
main_model_table = primary_main[main_cols].sort_values("Model")
main_model_table_path = os.path.join(PUB_DIR, "table_main_primary_model_comparison.csv")
main_model_table.to_csv(main_model_table_path, index=False)
main_model_table

# Reload completed model results after a kernel restart
# =========================

all_results_path = os.path.join(
    CONFIG.output_dir,
    "ALL_MODEL_RESULTS_COMPLETED_SCENARIOS.csv",
)

if not os.path.exists(all_results_path):
    raise FileNotFoundError(
        f"Cannot find {all_results_path}. "
        "The sensitivity scenarios must be rerun because the combined "
        "results file is missing."
    )

all_results = pd.read_csv(
    all_results_path
)

print(
    "Reloaded completed model results:",
    all_results_path,
)

print(
    "Rows loaded:",
    len(all_results),
)

print(
    "Scenarios available:",
    sorted(all_results["scenario"].unique())
)

print(
    "Models available:",
    sorted(all_results["model"].unique())
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

print(
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

# =========================
# Clean OR term labels for publication tables/plots
# =========================

MORPHO_LABELS = {
    "1": "Infiltrating duct carcinoma",
    "2": "Lobular carcinoma",
    "3": "Mixed/Other (+rare)",
}

def clean_or_term(term):
    term = str(term)

    # Continuous variable
    if term == "ydod":
        return "Year of diagnosis (per year)"

    # Specific morphology fix
    if term.startswith("morpho3_"):
        code = term.replace("morpho3_", "")
        label = MORPHO_LABELS.get(code, code)
        return f"Tumor morphology: {label}"

    # Specific grade fixes
    if term == "grade_num_1":
        return "Tumor grade: Grade I"
    if term == "grade_num_2":
        return "Tumor grade: Grade II"
    if term == "grade_num_3":
        return "Tumor grade: Grade III"

    # General variable label map
    variable_prefix_map = {
        "hospital_grp_num": "Hospital at diagnosis",
        "ethnicity4": "Ethnicity",
        "gov_num": "Governorate",
        "religion_grp3": "Religion",
        "edu_level": "Education level",
        "smoke_bi": "Smoking status",
        "job_employed": "Employment status",
        "famhx_bi": "Family history",
        "comorb_YesNo": "Comorbidity",
        "behav": "Tumor behavior",
        "topo_cat": "Tumor topography",
        "stagedx_num": "Stage at diagnosis",
    }

    for raw_var, clean_var in sorted(variable_prefix_map.items(), key=lambda x: len(x[0]), reverse=True):
        if term.startswith(raw_var + "_"):
            category = term.replace(raw_var + "_", "", 1)
            category = category.replace("_", " ")
            return f"{clean_var}: {category}"

    return term.replace("_", " ")

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
PUB_DIR = os.path.join(CONFIG.output_dir, "Publication")
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
            df["p_value"] < 0.05
        ].copy()

        remaining_slots = max(
            0,
            top_n - len(significant),
        )

        nonsignificant = (
            df[
                df["p_value"] >= 0.05
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

print(
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
# =========================
# This section explains the calibrated XGBoost model from the primary
# full-registry scenario. SHAP values are calculated separately for each
# fitted calibration-fold XGBoost estimator and then averaged, avoiding
# dependence on one arbitrarily selected fold.


def _get_calibrated_base_pipelines(
    artifacts,
    model_name="XGB",
):
    """
    Recover fitted base pipelines from CalibratedClassifierCV.

    CalibratedClassifierCV fits one base estimator within each calibration
    fold. All available fitted pipelines are returned.
    """
    if artifacts is None or model_name not in artifacts:
        raise ValueError(
            f"{model_name} was not found in PRIMARY_ARTIFACTS. "
            "Run the primary scenario first."
        )

    calibrated = artifacts[model_name]

    if not hasattr(
        calibrated,
        "calibrated_classifiers_",
    ):
        raise ValueError(
            "Expected a fitted CalibratedClassifierCV object "
            "with calibrated_classifiers_."
        )

    pipelines = []

    for calibrated_fold in calibrated.calibrated_classifiers_:
        if hasattr(calibrated_fold, "estimator"):
            pipelines.append(
                calibrated_fold.estimator
            )
        elif hasattr(
            calibrated_fold,
            "base_estimator",
        ):
            pipelines.append(
                calibrated_fold.base_estimator
            )
        else:
            raise ValueError(
                "Could not recover a calibrated base estimator."
            )

    if not pipelines:
        raise ValueError(
            "No calibrated base XGBoost pipelines were recovered."
        )

    return pipelines


def _clean_transformed_feature_name(x):
    x = str(x)

    # Remove preprocessing prefixes.
    for prefix in [
        "num__",
        "ord__",
        "cat__",
        "remainder__",
        "prep__",
    ]:
        x = x.replace(prefix, "")

    x = x.replace("missingindicator_", "")
    x = x.replace("_nan", "_Unknown or missing")

    # Raw variable names to publication labels.
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
            category = x[len(raw) + 1:].replace("_", " ")

            # Shorten the long topography label for publication figures.
            if category in {
                "Breast, NOS + rare sites (central, nipple, axillary tail, overlapping)",
                "Breast, NOS + rare sites",
                "Breast, NOS & rare sites",
            }:
                category = "Breast, NOS & rare sites"

            # Standardize missing-category wording.
            if category in {
                "Unknown or missing",
                "Unknown",
                "Missing",
            }:
                category = "Unknown or missing"

            return f"{label}: {category}"

    return x.replace("_", " ")


def _manual_feature_names_from_preprocessor(
    preprocessor,
):
    """
    Recover transformed feature names when sklearn cannot generate them
    automatically.
    """
    names = []

    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder" and transformer == "drop":
            continue

        if columns is None:
            continue

        columns = list(columns)

        final_transformer = transformer

        if isinstance(transformer, Pipeline):
            final_transformer = transformer.steps[-1][1]

        if hasattr(
            final_transformer,
            "get_feature_names_out",
        ):
            try:
                transformed_names = (
                    final_transformer
                    .get_feature_names_out(columns)
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
    preprocessing = base_pipeline.named_steps.get(
        "prep"
    )

    if preprocessing is None:
        raise ValueError(
            "Could not locate the fitted 'prep' step "
            "inside the XGBoost pipeline."
        )

    try:
        feature_names = list(
            preprocessing.get_feature_names_out()
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
    if hasattr(shap_output, "values"):
        shap_output = shap_output.values

    if isinstance(shap_output, list):
        if len(shap_output) == 2:
            return np.asarray(shap_output[1])

        return np.asarray(shap_output[0])

    shap_array = np.asarray(shap_output)

    if shap_array.ndim == 3:
        # Shape may be observations × features × classes.
        if shap_array.shape[2] == 2:
            return shap_array[:, :, 1]

        # Alternative classes × observations × features layout.
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

    SHAP values are calculated for every fitted calibration-fold base
    estimator and averaged after alignment by transformed feature name.
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

    X_sample = X_test.sample(
        n=min(sample_size, len(X_test)),
        random_state=RANDOM_STATE,
    ).copy()

    base_pipelines = _get_calibrated_base_pipelines(
        primary_artifacts,
        model_name="XGB",
    )

    shap_frames = []
    transformed_frames = []

    expected_clean_names = None
    original_feature_names = None

    for fold_number, base_pipeline in enumerate(
        base_pipelines,
        start=1,
    ):
        if "clf" not in base_pipeline.named_steps:
            raise ValueError(
                "The recovered XGBoost pipeline does not contain "
                "a 'clf' step."
            )

        preprocessing_pipeline = Pipeline(
            base_pipeline.steps[:-1]
        )

        xgb_classifier = base_pipeline.named_steps[
            "clf"
        ]

        X_transformed = (
            preprocessing_pipeline.transform(
                X_sample
            )
        )

        if hasattr(
            X_transformed,
            "toarray",
        ):
            X_transformed = X_transformed.toarray()

        X_transformed = np.asarray(
            X_transformed,
            dtype=float,
        )

        feature_names = (
            _get_transformed_feature_names(
                base_pipeline
            )
        )

        if len(feature_names) != X_transformed.shape[1]:
            raise ValueError(
                "The number of transformed feature names does not "
                "match the transformed matrix width. "
                f"Names: {len(feature_names)}; "
                f"matrix columns: {X_transformed.shape[1]}"
            )

        clean_names = [
            _clean_transformed_feature_name(name)
            for name in feature_names
        ]

        if len(set(clean_names)) != len(clean_names):
            duplicates = pd.Series(
                clean_names
            )[
                pd.Series(clean_names).duplicated(
                    keep=False
                )
            ].tolist()

            raise ValueError(
                "Duplicate cleaned SHAP feature names were produced: "
                f"{sorted(set(duplicates))}"
            )

        X_transformed_df = pd.DataFrame(
            X_transformed,
            columns=clean_names,
            index=X_sample.index,
        )

        explainer = shap.TreeExplainer(
            xgb_classifier
        )

        raw_shap_values = explainer.shap_values(
            X_transformed_df
        )

        fold_shap_values = (
            _normalise_binary_shap_values(
                raw_shap_values
            )
        )

        if fold_shap_values.shape != X_transformed_df.shape:
            raise ValueError(
                f"SHAP shape mismatch in calibration fold "
                f"{fold_number}: SHAP {fold_shap_values.shape}, "
                f"features {X_transformed_df.shape}"
            )

        fold_shap_df = pd.DataFrame(
            fold_shap_values,
            columns=clean_names,
            index=X_sample.index,
        )

        if expected_clean_names is None:
            expected_clean_names = clean_names
            original_feature_names = feature_names
        else:
            missing = set(
                expected_clean_names
            ) - set(clean_names)

            extra = set(clean_names) - set(
                expected_clean_names
            )

            if missing or extra:
                raise ValueError(
                    "Transformed feature sets differed across "
                    "calibration folds. "
                    f"Missing: {sorted(missing)}; "
                    f"extra: {sorted(extra)}"
                )

            fold_shap_df = fold_shap_df[
                expected_clean_names
            ]

            X_transformed_df = X_transformed_df[
                expected_clean_names
            ]

        shap_frames.append(
            fold_shap_df
        )
        transformed_frames.append(
            X_transformed_df
        )

    # Average contributions across calibrated base estimators.
    averaged_shap = sum(
        frame.to_numpy()
        for frame in shap_frames
    ) / len(shap_frames)

    # Average transformed values for consistent beeswarm colouring.
    averaged_transformed = sum(
        frame.to_numpy()
        for frame in transformed_frames
    ) / len(transformed_frames)

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

    shap_importance = pd.DataFrame({
        "feature": original_feature_names,
        "clean_feature": expected_clean_names,
        "mean_abs_shap": np.abs(
            averaged_shap_df.to_numpy()
        ).mean(axis=0),
    }).sort_values(
        "mean_abs_shap",
        ascending=False,
    )

    return {
        "X_trans": averaged_transformed_df,
        "X_sample": X_sample,
        "shap_values": averaged_shap_df.to_numpy(),
        "feature_names": list(
            original_feature_names
        ),
        "clean_feature_names": list(
            expected_clean_names
        ),
        "shap_importance": shap_importance,
        "n_calibration_estimators": len(
            base_pipelines
        ),
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
    sample_size=500,
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

# =========================
# 13J) Supplementary Figure: ROC curves for primary predictive models
# =========================

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

# Load observed data
X, y, predictors, aux_cols = load_data(CONFIG)

# Recreate the same stratified test split used in the primary modeling pipeline
X_model = X[predictors + [c for c in aux_cols if c not in predictors]].copy()

_, X_test, _, y_test = train_test_split(
    X_model,
    y,
    test_size=CONFIG.test_size,
    stratify=y,
    random_state=RANDOM_STATE
)

# Professional model labels
MODEL_LABELS_ROC = {
    "Logit": "LR",
    "RF": "RF",
    "SVM_RBF": "SVM-RBF",
    "XGB": "XGBoost",
}

MODELS_TO_PLOT_ROC = ["Logit", "RF", "SVM_RBF", "XGB"]

roc_rows = []

fig, ax = plt.subplots(figsize=(7.5, 6.5))

for model_key in MODELS_TO_PLOT_ROC:
    if model_key not in PRIMARY_ARTIFACTS:
        print(f"Skipping {model_key}: not found in PRIMARY_ARTIFACTS")
        continue

    model = PRIMARY_ARTIFACTS[model_key]

    y_prob = model.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    auc_value = roc_auc_score(y_test, y_prob)

    label = MODEL_LABELS_ROC.get(model_key, model_key)

    ax.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{label} (AUC = {auc_value:.3f})"
    )

    tmp = pd.DataFrame({
        "model": label,
        "fpr": fpr,
        "tpr": tpr,
        "threshold": thresholds,
        "roc_auc": auc_value
    })
    roc_rows.append(tmp)

# No-skill reference line
ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1,
    label="No-skill (AUC = 0.500)"
)

ax.set_xlabel("1 - Specificity (False positive rate)")
ax.set_ylabel("Sensitivity (True positive rate)")
ax.set_title("ROC curves for primary EOBC classification models")
ax.legend(loc="lower right", frameon=True)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.grid(alpha=0.25)

plt.tight_layout()

roc_plot_path = os.path.join(PUB_DIR, "figure_S1_roc_curves_primary_models.png")
fig.savefig(roc_plot_path, dpi=300, bbox_inches="tight")
fig.savefig(roc_plot_path.replace(".png", ".pdf"), bbox_inches="tight")
plt.close(fig)

# Save ROC curve coordinates
roc_curve_table = pd.concat(roc_rows, axis=0, ignore_index=True)
roc_curve_table_path = os.path.join(PUB_DIR, "figure_S1_roc_curves_primary_models_data.csv")
roc_curve_table.to_csv(roc_curve_table_path, index=False)

print("Saved ROC plot:", roc_plot_path)
print("Saved ROC curve data:", roc_curve_table_path)


# 13K) Publication-output manifest
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
