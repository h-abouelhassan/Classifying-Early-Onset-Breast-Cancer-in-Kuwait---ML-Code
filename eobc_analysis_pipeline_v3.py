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

    # Core predictors used in the original abstract/pipeline.
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

    # Treatment variables are auxiliary only for targeted imputation of grade/morphology.
    # They must not enter the final predictor matrix.
    aux_impute_candidates: List[str] = field(default_factory=lambda: [
        "surgery", "radio", "chemo", "hormone", "immuno"
    ])

    # Model-role definitions.
    # ydod is continuous. Grade/morphology are targeted-imputation variables.
    # When targeted KNN is OFF, grade/morphology are handled as categorical variables
    # to avoid arbitrary numeric scaling of labels such as "Grade II (Mod diff)".
    numeric_cols: List[str] = field(default_factory=lambda: ["ydod"])
    ordinal_cols: List[str] = field(default_factory=lambda: ["grade_num", "morpho3"])

    test_size: float = 0.25
    n_splits: int = 5
    n_bootstrap: int = 500
    min_recall: float = 0.70

    # Keep the main model set focused. MLP is available but excluded by default
    # because it was slow and did not improve performance in the diagnostic run.
    models_to_run: List[str] = field(default_factory=lambda: ["Logit", "RF", "SVM_RBF", "XGB", "NB"])
    include_mlp: bool = False

    # MICE-style inferential logistic regression settings
    mice_m: int = 20
    mice_max_iter: int = 20
    mice_sample_posterior: bool = True

    # RF interpretation settings
    permutation_repeats: int = 30
    shap_sample_size: int = 750

CONFIG = EOBCConfig()
os.makedirs(CONFIG.output_dir, exist_ok=True)

# =========================

# 2) Utility transformers
# =========================
def _to_numeric_or_catcodes(s: pd.Series) -> pd.Series:
    """Convert a Series to numeric; if impossible, encode categories as numeric codes preserving NaN."""
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.notna().any():
        return s_num
    cat = pd.Categorical(s)
    codes = pd.Series(cat.codes, index=s.index).astype(float)
    codes[codes < 0] = np.nan
    return codes


def _extract_ordinal_1to3(s: pd.Series) -> pd.Series:
    """Extract ordinal values 1, 2, 3 from strings if possible; fallback to numeric/category codes."""
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    extracted = s.astype(str).str.extract(r"([123])", expand=False)
    extracted = pd.to_numeric(extracted, errors="coerce")
    if extracted.notna().any():
        return extracted
    return _to_numeric_or_catcodes(s)


class TargetedKNNImputer(BaseEstimator, TransformerMixin):
    """
    Leakage-safe targeted KNN imputer for grade_num and morpho3.

    It uses only variables available in X_train during fit(). Treatment variables may be used as
    auxiliary imputation variables but are removed downstream by the ColumnTransformer because
    they are not part of predictor_cols.
    """
    def __init__(self, target_cols=None, aux_cols=None, k=7):
        self.target_cols = target_cols if target_cols is not None else ["grade_num", "morpho3"]
        self.aux_cols = aux_cols if aux_cols is not None else []
        self.k = k

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.available_target_cols_ = [c for c in self.target_cols if c in X.columns]
        self.available_aux_cols_ = [c for c in self.aux_cols if c in X.columns]
        self.impute_cols_ = self.available_target_cols_ + self.available_aux_cols_

        # If nothing to impute, no-op.
        if not self.available_target_cols_:
            self.imputer_ = None
            return self

        Z = pd.DataFrame(index=X.index)
        for c in self.impute_cols_:
            if c in self.available_target_cols_:
                Z[c] = _extract_ordinal_1to3(X[c])
            else:
                Z[c] = _to_numeric_or_catcodes(X[c])

        self.col_medians_ = Z.median(numeric_only=True).to_dict()
        # KNNImputer cannot handle all-NaN columns. Fill all-NaN aux columns with 0.
        for c in Z.columns:
            if Z[c].isna().all():
                Z[c] = 0.0
                self.col_medians_[c] = 0.0

        self.imputer_ = KNNImputer(n_neighbors=self.k, weights="distance")
        self.imputer_.fit(Z)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        if getattr(self, "imputer_", None) is None:
            return X

        Z = pd.DataFrame(index=X.index)
        for c in self.impute_cols_:
            if c in self.available_target_cols_:
                Z[c] = _extract_ordinal_1to3(X[c])
            else:
                Z[c] = _to_numeric_or_catcodes(X[c])
            if Z[c].isna().all():
                Z[c] = self.col_medians_.get(c, 0.0)

        Z_imp = pd.DataFrame(self.imputer_.transform(Z), columns=self.impute_cols_, index=X.index)
        for c in self.available_target_cols_:
            # Round grade/morphology back to ordinal 1/2/3.
            X[c] = np.clip(np.rint(Z_imp[c]), 1, 3)
        return X


class ColumnKeeper(BaseEstimator, TransformerMixin):
    """Keep selected columns and return DataFrame. Useful inside pipelines."""
    def __init__(self, columns):
        self.columns = columns
    def fit(self, X, y=None):
        self.columns_ = [c for c in self.columns if c in pd.DataFrame(X).columns]
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

# 6) Thresholding & evaluation
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
def tune_model(name, model, X_train, y_train, config: EOBCConfig):
    param_dist = get_param_distributions(name)
    if param_dist is None:
        model.fit(X_train, y_train)
        return model, {"tuned": False}

    cv = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=RANDOM_STATE)
    n_iter = 25 if name in ["Logit", "RF", "SVM_RBF", "MLP", "HGB"] else 50
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="balanced_accuracy",
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=0,
        refit=True
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, {"tuned": True, "best_params": search.best_params_, "best_cv_bal_acc": search.best_score_}


def evaluate_model(name, model, X_train, y_train, X_test, y_test, config: EOBCConfig):
    # Calibrate using training data only. This refits internally using CV on train set.
    calibrated = CalibratedClassifierCV(estimator=model, method="sigmoid", cv=3)
    calibrated.fit(X_train, y_train)

    prob_train = _predict_proba_positive(calibrated, X_train)
    threshold = best_threshold_precision_at_recall(y_train, prob_train, min_recall=config.min_recall)

    prob_test = _predict_proba_positive(calibrated, X_test)
    metrics = compute_metrics(y_test, prob_test, threshold)
    ci = bootstrap_metric_cis(y_test, prob_test, threshold, n_bootstrap=config.n_bootstrap, seed=RANDOM_STATE)
    metrics.update(ci)
    metrics["model"] = name
    return calibrated, prob_test, metrics


def run_scenario(
    scenario_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    predictors: List[str],
    aux_cols: List[str],
    config: EOBCConfig,
    imputation_strategy: str = "registry_default",
    use_targeted_knn: bool = True,
    complete_case_grade_morpho: bool = False,
    models_to_run: Optional[List[str]] = None,
):
    print(f"\n===== Scenario: {scenario_name} =====")
    predictors = [c for c in predictors if c in X.columns]
    aux_cols = [c for c in aux_cols if c in X.columns]
    X_s = X[predictors + [c for c in aux_cols if c not in predictors]].copy()
    y_s = y.copy()

    if complete_case_grade_morpho:
        cc_cols = [c for c in ["grade_num", "morpho3"] if c in X_s.columns]
        mask = X_s[cc_cols].notna().all(axis=1)
        X_s, y_s = X_s.loc[mask].copy(), y_s.loc[mask].copy()
        print("Complete-case rows retained:", len(y_s))

    X_train, X_test, y_train, y_test = train_test_split(
        X_s, y_s, test_size=config.test_size, stratify=y_s,
        random_state=RANDOM_STATE
    )

    models = make_models(config, predictors, aux_cols, imputation_strategy, use_targeted_knn, models_to_run=models_to_run)
    all_rows = []
    model_artifacts = {}
    tuning_info = {}

    for name, model in models.items():
        print("Fitting", name)
        best_model, info = tune_model(name, model, X_train, y_train, config)
        calibrated, prob_test, metrics = evaluate_model(name, best_model, X_train, y_train, X_test, y_test, config)
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

    res = pd.DataFrame(all_rows).sort_values(["balanced_accuracy", "roc_auc"], ascending=False)
    res.to_csv(os.path.join(config.output_dir, f"results_{scenario_name}.csv"), index=False)
    with open(os.path.join(config.output_dir, f"tuning_{scenario_name}.json"), "w") as f:
        json.dump(tuning_info, f, indent=2, default=str)
    return res, model_artifacts


def run_named_scenario(scen_name: str, models_to_run: Optional[List[str]] = None):
    """
    Convenience function for the sectioned execution cells below.
    It reloads the data and runs one scenario only.
    """
    X, y, predictors, aux_cols = load_data(CONFIG)
    scenarios = build_scenarios(CONFIG, predictors)
    if scen_name not in scenarios:
        raise ValueError(f"Unknown scenario: {scen_name}. Available: {list(scenarios)}")
    params = scenarios[scen_name]
    res, artifacts = run_scenario(
        scenario_name=scen_name,
        X=X, y=y,
        predictors=params["predictors"],
        aux_cols=aux_cols,
        config=CONFIG,
        imputation_strategy=params["imputation_strategy"],
        use_targeted_knn=params["use_targeted_knn"],
        complete_case_grade_morpho=params["complete_case_grade_morpho"],
        models_to_run=models_to_run,
    )
    return res, artifacts


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

# 9) Permutation importance for best model in primary scenario
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

# 10) MICE logistic regression for inferential adjusted ORs
# =========================
def _build_code_maps(X: pd.DataFrame, predictors: List[str], config: EOBCConfig):
    """Build stable category-code maps for all non-continuous predictors."""
    maps = {}
    continuous = set(config.numeric_cols)
    for c in predictors:
        if c in continuous:
            continue
        s = X[c]
        if c in config.ordinal_cols:
            vals = sorted(pd.Series(_extract_ordinal_1to3(s)).dropna().unique().tolist())
            # fallback if extraction fails
            if not vals:
                vals = sorted(pd.Categorical(s.dropna()).categories.tolist())
        else:
            vals = sorted(pd.Categorical(s.dropna().astype(str)).categories.tolist())
        maps[c] = {v: i for i, v in enumerate(vals)}
    return maps


def _encode_for_mice(X: pd.DataFrame, predictors: List[str], config: EOBCConfig, maps: Dict):
    """Encode predictors numerically for IterativeImputer. Missing values remain NaN."""
    Z = pd.DataFrame(index=X.index)
    for c in predictors:
        if c in config.numeric_cols:
            Z[c] = pd.to_numeric(X[c], errors="coerce")
        elif c in config.ordinal_cols:
            # Use extracted 1/2/3 values where possible.
            vals = _extract_ordinal_1to3(X[c])
            if vals.notna().any():
                Z[c] = vals
                # Make map for valid extracted values if needed.
                maps.setdefault(c, {v: v for v in sorted(vals.dropna().unique())})
            else:
                Z[c] = X[c].map(maps[c]).astype(float)
        else:
            Z[c] = X[c].astype(str).where(X[c].notna(), np.nan).map(maps[c]).astype(float)
    return Z


def _decode_mice_imputed(Z_imp: pd.DataFrame, X_original: pd.DataFrame, predictors: List[str], config: EOBCConfig, maps: Dict):
    """
    Decode one MICE-imputed numeric dataset back into analysis-ready mixed-type data.
    Continuous variables remain continuous. Non-continuous variables are rounded/clipped to valid codes.
    """
    X_imp = pd.DataFrame(index=Z_imp.index)
    for c in predictors:
        if c in config.numeric_cols:
            X_imp[c] = Z_imp[c]
        elif c in config.ordinal_cols:
            valid = sorted(pd.Series(list(maps.get(c, {}).values())).dropna().unique().tolist())
            if not valid:
                valid = [1, 2, 3]
            lo, hi = min(valid), max(valid)
            X_imp[c] = np.clip(np.rint(Z_imp[c]), lo, hi).astype(int).astype(str)
        else:
            inv = {v: k for k, v in maps[c].items()}
            lo, hi = min(inv.keys()), max(inv.keys())
            codes = np.clip(np.rint(Z_imp[c]), lo, hi).astype(int)
            X_imp[c] = pd.Series(codes, index=Z_imp.index).map(inv).astype(str)
    return X_imp


def _make_logit_design(X_imp: pd.DataFrame, predictors: List[str], config: EOBCConfig):
    """
    Create a statsmodels design matrix.
    Continuous variables are kept numeric; all other predictors are dummy-coded.
    """
    continuous = [c for c in config.numeric_cols if c in predictors]
    categorical = [c for c in predictors if c not in continuous]

    parts = []
    if continuous:
        parts.append(X_imp[continuous].apply(pd.to_numeric, errors="coerce"))

    if categorical:
        cat_df = X_imp[categorical].astype(str)
        dummies = pd.get_dummies(cat_df, columns=categorical, drop_first=True, dtype=float)
        parts.append(dummies)

    X_design = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=X_imp.index)
    X_design = sm.add_constant(X_design, has_constant="add")
    return X_design.astype(float)


def _fit_logit_statsmodels(X_design: pd.DataFrame, y: pd.Series):
    """Fit logistic GLM. GLM Binomial is more stable than Logit for occasional separation."""
    model = sm.GLM(y.astype(float), X_design, family=sm.families.Binomial())
    fit = model.fit(maxiter=200, disp=False)
    return fit.params, fit.cov_params()


def run_mice_logistic_regression(
    X: pd.DataFrame,
    y: pd.Series,
    predictors: List[str],
    config: EOBCConfig,
    output_prefix: str = "mice_logistic_full"
):
    if not HAS_STATSMODELS:
        raise ImportError("statsmodels is required for MICE logistic regression output.")

    predictors = [c for c in predictors if c in X.columns]
    X0 = X[predictors].copy()
    y0 = y.copy().astype(int)

    maps = _build_code_maps(X0, predictors, config)
    Z = _encode_for_mice(X0, predictors, config, maps)

    imputer = IterativeImputer(
        estimator=BayesianRidge(),
        max_iter=config.mice_max_iter,
        sample_posterior=config.mice_sample_posterior,
        initial_strategy="most_frequent",
        random_state=RANDOM_STATE,
        skip_complete=True
    )

    params_list = []
    cov_list = []
    design_columns = None
    imputed_designs = []

    # Fit/transform separately for each imputation using different seeds to reflect imputation uncertainty.
    for m in range(config.mice_m):
        print(f"MICE imputation {m+1}/{config.mice_m}")
        imp = clone(imputer)
        imp.set_params(random_state=RANDOM_STATE + m)
        Z_imp_arr = imp.fit_transform(Z)
        Z_imp = pd.DataFrame(Z_imp_arr, columns=Z.columns, index=Z.index)
        X_imp = _decode_mice_imputed(Z_imp, X0, predictors, config, maps)
        X_design = _make_logit_design(X_imp, predictors, config)
        imputed_designs.append(X_design)
        design_columns = sorted(set(design_columns or []).union(X_design.columns))

    # Align all design matrices to identical dummy columns before fitting/pooling.
    design_columns = ["const"] + sorted([c for c in design_columns if c != "const"])
    for X_design in imputed_designs:
        X_aligned = X_design.reindex(columns=design_columns, fill_value=0.0)
        params, cov = _fit_logit_statsmodels(X_aligned, y0)
        params_list.append(params.reindex(design_columns))
        cov_list.append(cov.reindex(index=design_columns, columns=design_columns, fill_value=0.0))

    Q = pd.concat(params_list, axis=1)
    beta_bar = Q.mean(axis=1)

    # Rubin's rules
    U_bar = sum(cov_list) / len(cov_list)
    B = Q.var(axis=1, ddof=1)
    total_var = pd.Series(np.diag(U_bar), index=beta_bar.index) + (1 + 1 / config.mice_m) * B
    se = np.sqrt(total_var)

    z = beta_bar / se
    p = 2 * (1 - stats.norm.cdf(np.abs(z)))
    ci_low = beta_bar - 1.96 * se
    ci_high = beta_bar + 1.96 * se

    out = pd.DataFrame({
        "term": beta_bar.index,
        "log_or": beta_bar.values,
        "se": se.values,
        "z": z.values,
        "p_value": p,
        "ci_low_log": ci_low.values,
        "ci_high_log": ci_high.values,
        "OR": np.exp(beta_bar.values),
        "CI_low": np.exp(ci_low.values),
        "CI_high": np.exp(ci_high.values),
        "m_imputations": config.mice_m,
    })
    out = out[out["term"] != "const"].sort_values("p_value")
    csv_path = os.path.join(config.output_dir, f"{output_prefix}_pooled_ORs.csv")
    out.to_csv(csv_path, index=False)
    print("Saved pooled MICE OR table:", csv_path)
    return out


def plot_mice_or_forest(or_df: pd.DataFrame, output_path: str, max_terms: int = 30):
    """
    Forest plot for MICE logistic regression ORs.
    Plots the top terms by p-value. For manuscript figures, manually select clinically important terms later.
    """
    plot_df = or_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["OR", "CI_low", "CI_high"]).copy()
    plot_df = plot_df.head(max_terms).iloc[::-1]

    fig_height = max(6, 0.32 * len(plot_df) + 2)
    fig, ax = plt.subplots(figsize=(9, fig_height))
    y_pos = np.arange(len(plot_df))
    ax.errorbar(
        plot_df["OR"], y_pos,
        xerr=[plot_df["OR"] - plot_df["CI_low"], plot_df["CI_high"] - plot_df["OR"]],
        fmt="o", capsize=3
    )
    ax.axvline(1.0, linestyle="--", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["term"])
    ax.set_xscale("log")
    ax.set_xlabel("Adjusted odds ratio (log scale)")
    ax.set_title("MICE-pooled multivariable logistic regression")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved MICE OR forest plot:", output_path)


# =========================

# 11) RF interpretation functions
# =========================
def save_primary_rf_interpretation(
    primary_artifacts: Dict[str, object],
    X: pd.DataFrame,
    y: pd.Series,
    predictors: List[str],
    aux_cols: List[str],
    config: EOBCConfig,
):
    """
    Save RF permutation importance using the calibrated RF pipeline from the primary scenario.
    Optional SHAP output is attempted if shap is installed and the calibrated estimator internals are accessible.
    """
    if primary_artifacts is None or "RF" not in primary_artifacts:
        raise ValueError("RF artifact not found. Run the primary_full_registry_default scenario first and keep PRIMARY_ARTIFACTS.")

    X_model = X[predictors + [c for c in aux_cols if c not in predictors]].copy()
    _, X_test, _, y_test = train_test_split(
        X_model, y, test_size=config.test_size, stratify=y, random_state=RANDOM_STATE
    )

    rf_model = primary_artifacts["RF"]

    # Raw-column permutation importance. This is robust and works with the full pipeline.
    rf_pi_path = os.path.join(config.output_dir, "rf_primary_permutation_importance_raw_features.csv")
    rf_pi = save_permutation_importance(
        rf_model, X_test, y_test, predictors, rf_pi_path
    )

    # Plot top raw-feature permutation importances.
    top = rf_pi.head(20).iloc[::-1]
    fig_height = max(5, 0.32 * len(top) + 1.5)
    fig, ax = plt.subplots(figsize=(8, fig_height))
    ax.barh(top["feature"], top["importance_mean"])
    ax.set_xlabel("Permutation importance: mean decrease in balanced accuracy")
    ax.set_title("Random forest interpretation: raw-feature permutation importance")
    plt.tight_layout()
    fig_path = os.path.join(config.output_dir, "rf_primary_permutation_importance_raw_features.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved RF permutation importance plot:", fig_path)

    # Optional SHAP bar plot on transformed features.
    # This is exploratory/supplementary because calibrated CV stores multiple fitted estimators.
    if HAS_SHAP:
        try:
            cal0 = rf_model.calibrated_classifiers_[0]
            base_pipeline = cal0.estimator
            prep_pipe = Pipeline(base_pipeline.steps[:-1])
            rf_clf = base_pipeline.named_steps["clf"]

            X_sample = X_test.sample(min(config.shap_sample_size, len(X_test)), random_state=RANDOM_STATE)
            X_trans = prep_pipe.transform(X_sample)

            # Recover transformed feature names if possible.
            try:
                feature_names = base_pipeline.named_steps["prep"].get_feature_names_out()
            except Exception:
                feature_names = [f"feature_{i}" for i in range(X_trans.shape[1])]

            explainer = shap.TreeExplainer(rf_clf)
            shap_values = explainer.shap_values(X_trans)
            # Binary classifiers may return a list; use positive-class SHAP values.
            sv = shap_values[1] if isinstance(shap_values, list) else shap_values

            shap_abs = np.abs(sv).mean(axis=0)
            shap_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": shap_abs}).sort_values("mean_abs_shap", ascending=False)
            shap_csv = os.path.join(config.output_dir, "rf_primary_shap_mean_abs_transformed_features.csv")
            shap_df.to_csv(shap_csv, index=False)

            top_shap = shap_df.head(25).iloc[::-1]
            fig, ax = plt.subplots(figsize=(9, max(6, 0.32 * len(top_shap) + 1.5)))
            ax.barh(top_shap["feature"], top_shap["mean_abs_shap"])
            ax.set_xlabel("Mean absolute SHAP value")
            ax.set_title("Random forest SHAP importance, transformed features")
            plt.tight_layout()
            shap_fig = os.path.join(config.output_dir, "rf_primary_shap_mean_abs_transformed_features.png")
            fig.savefig(shap_fig, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print("Saved RF SHAP outputs:", shap_csv, shap_fig)
        except Exception as e:
            print("SHAP output skipped:", repr(e))
    else:
        print("SHAP not installed; skipped SHAP outputs.")

    return rf_pi


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

# 12I) Optional missing-data sensitivity: no targeted KNN
# =========================

NO_TARGETED_KNN_RESULTS, _ = run_named_scenario(
    "no_targeted_knn_registry_default",
    models_to_run=CONFIG.models_to_run
)
NO_TARGETED_KNN_RESULTS

# 12J) Optional missing-data sensitivity: complete-case grade/morphology
# =========================

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

# 12L) Run final MICE logistic regression for adjusted ORs
# =========================

X, y, predictors, aux_cols = load_data(CONFIG)
MICE_OR_TABLE = run_mice_logistic_regression(
    X=X,
    y=y,
    predictors=predictors,
    config=CONFIG,
    output_prefix="mice_logistic_primary_full"
)

plot_mice_or_forest(
    MICE_OR_TABLE,
    os.path.join(CONFIG.output_dir, "mice_logistic_primary_full_forest_top30.png"),
    max_terms=30
)

MICE_OR_TABLE.head(30)

# 12M) Run RF interpretation after the primary model has been run
# =========================
# Requires PRIMARY_ARTIFACTS from cell 12B.
X, y, predictors, aux_cols = load_data(CONFIG)
RF_IMPORTANCE = save_primary_rf_interpretation(
    primary_artifacts=PRIMARY_ARTIFACTS,
    X=X,
    y=y,
    predictors=predictors,
    aux_cols=aux_cols,
    config=CONFIG,
)
RF_IMPORTANCE.head(20)


# 13A) Publication-output settings and clean labels
# =========================

PUBLICATION_MODELS = ["Logit", "RF", "SVM_RBF", "XGB"]

SCENARIO_LABELS = {
    "primary_full_registry_default": "Primary full registry model",
    "no_employment": "Excluding employment",
    "no_education": "Excluding education",
    "no_employment_no_education": "Excluding employment and education",
    "sociodemo_history_only": "Sociodemographic/history variables only",
    "tumor_clinical_only": "Tumor/clinical variables only",
    "simple_imputation_no_missing_category": "Simple-imputation sensitivity",
    "no_targeted_knn_registry_default": "No targeted KNN sensitivity",
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
    """Clean one-hot/logistic term labels, preserving category information."""
    if pd.isna(term):
        return term
    term = str(term)
    if term == "ydod":
        return "Year of diagnosis (per year)"
    # Match the longest known prefix first to avoid partial matches.
    for prefix in sorted(TERM_PREFIX_LABELS, key=len, reverse=True):
        if term.startswith(prefix + "_"):
            cat = term[len(prefix)+1:]
            cat = cat.replace("_", " ")
            return f"{TERM_PREFIX_LABELS[prefix]}: {cat}"
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


# 13C) Sensitivity-analysis summary table
# =========================

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

sens = (
    all_results
    .query("scenario in @retained_scenarios and model in @PUBLICATION_MODELS")
    .copy()
)
sens["Scenario"] = sens["scenario"].map(SCENARIO_LABELS).fillna(sens["scenario"])
sens["Model"] = sens["model"].map(MODEL_LABELS).fillna(sens["model"])

# Best model per scenario by balanced accuracy, with ROC-AUC as tie-breaker.
best_sens = (
    sens.sort_values(["scenario", "balanced_accuracy", "roc_auc"], ascending=[True, False, False])
    .groupby("scenario", as_index=False)
    .head(1)
    .copy()
)
best_sens["ROC-AUC"] = best_sens["roc_auc"].round(3)
best_sens["Balanced accuracy"] = best_sens["balanced_accuracy"].round(3)
best_sens["Sensitivity"] = best_sens["sensitivity"].round(3)
best_sens["Specificity"] = best_sens["specificity"].round(3)
best_sens["Brier score"] = best_sens["brier_score"].round(3)
best_sens["n"] = best_sens["n_total"].astype(int)

best_sens_table = best_sens[[
    "Scenario", "n", "Model", "ROC-AUC", "Balanced accuracy", "Sensitivity", "Specificity", "Brier score"
]]
best_sens_path = os.path.join(PUB_DIR, "table_sensitivity_best_model_by_scenario.csv")
best_sens_table.to_csv(best_sens_path, index=False)

# Logit + RF comparison across scenarios.
logit_rf = sens.query("model in ['Logit', 'RF']").copy()
logit_rf["ROC-AUC"] = logit_rf["roc_auc"].round(3)
logit_rf["Balanced accuracy"] = logit_rf["balanced_accuracy"].round(3)
logit_rf["Sensitivity"] = logit_rf["sensitivity"].round(3)
logit_rf["Specificity"] = logit_rf["specificity"].round(3)
logit_rf["Brier score"] = logit_rf["brier_score"].round(3)
logit_rf["n"] = logit_rf["n_total"].astype(int)

logit_rf_table = logit_rf[[
    "Scenario", "n", "Model", "ROC-AUC", "Balanced accuracy", "Sensitivity", "Specificity", "Brier score"
]].sort_values(["Scenario", "Model"])
logit_rf_path = os.path.join(PUB_DIR, "table_sensitivity_logit_rf_by_scenario.csv")
logit_rf_table.to_csv(logit_rf_path, index=False)

best_sens_table


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

mice_path = os.path.join(CONFIG.output_dir, "mice_logistic_primary_full_pooled_ORs.csv")
if not os.path.exists(mice_path):
    raise FileNotFoundError(f"Cannot find {mice_path}. Run 12L first.")

mice_or = pd.read_csv(mice_path).copy()
mice_or["Variable/category"] = mice_or["term"].apply(clean_or_term)
mice_or["Adjusted OR (95% CI)"] = mice_or.apply(
    lambda r: format_ci(r["OR"], r["CI_low"], r["CI_high"], digits=2),
    axis=1
)
mice_or["p-value"] = mice_or["p_value"].apply(format_p)

mice_or_clean = mice_or[[
    "term",
    "Variable/category",
    "OR",
    "CI_low",
    "CI_high",
    "p_value",
    "Adjusted OR (95% CI)",
    "p-value",
    "m_imputations"
]].sort_values("p_value")

mice_or_table_path = os.path.join(PUB_DIR, "table_mice_pooled_adjusted_ORs_full.csv")
mice_or_clean.to_csv(mice_or_table_path, index=False)


def plot_mice_forest_publication(
    or_df,
    out_path,
    top_n=22,
    include_all_significant=True,
    linear_scale=True,
    show_estimates=True
):
    df = or_df.copy()
    df["abs_log_or"] = np.abs(np.log(df["OR"]))

    if include_all_significant:
        sig = df[df["p_value"] < 0.05].copy()
        nonsig = (
            df[df["p_value"] >= 0.05]
            .sort_values("abs_log_or", ascending=False)
            .head(max(0, top_n - len(sig)))
        )
        plot_df = pd.concat([sig, nonsig], axis=0).drop_duplicates("term")

        if len(plot_df) > top_n:
            plot_df = plot_df.sort_values("p_value").head(top_n)
    else:
        plot_df = df.sort_values("abs_log_or", ascending=False).head(top_n)

    # Sort by OR for clean visual separation.
    plot_df = plot_df.sort_values("OR", ascending=True).copy()
    y = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(10, max(6, 0.38 * len(plot_df) + 1.5)))

    ax.errorbar(
        plot_df["OR"],
        y,
        xerr=[
            plot_df["OR"] - plot_df["CI_low"],
            plot_df["CI_high"] - plot_df["OR"]
        ],
        fmt="o",
        capsize=3
    )

    ax.axvline(1.0, linestyle="--", linewidth=1)

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["Variable/category"])

    if linear_scale:
        ax.set_xscale("linear")
        ax.set_xlabel("Adjusted odds ratio for EOBC classification")
        max_ci = plot_df["CI_high"].max()
        ax.set_xlim(0, min(max_ci + 0.6, 5))
    else:
        ax.set_xscale("log")
        ax.set_xlabel("Adjusted odds ratio for EOBC classification (log scale)")

    ax.set_title("MICE-pooled multivariable logistic regression")

    if show_estimates:
        x_min, x_max = ax.get_xlim()
        text_offset = (x_max - x_min) * 0.015

        for pos, (_, row) in enumerate(plot_df.iterrows()):
            x_text = row["OR"] + text_offset

            # Keep text inside plot boundary.
            if x_text > x_max * 0.96:
                x_text = row["OR"] - (x_max - x_min) * 0.08
                ha = "right"
            else:
                ha = "left"

            ax.text(
                x_text,
                pos,
                f"{row['OR']:.2f}",
                va="center",
                ha=ha,
                fontsize=8
            )

    plt.tight_layout()

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)

    return plot_df


mice_forest_plot_path = os.path.join(PUB_DIR, "figure_mice_logistic_forest_publication.png")
mice_forest_plot_data = plot_mice_forest_publication(
    mice_or,
    mice_forest_plot_path,
    top_n=22,
    include_all_significant=True,
    linear_scale=True,
    show_estimates=True
)

mice_forest_plot_data.to_csv(
    os.path.join(PUB_DIR, "figure_mice_logistic_forest_publication_data.csv"),
    index=False
)

mice_or_clean.head(20)


# 13F) Clean RF raw-feature permutation importance table and bar plot
# =========================

rf_pi_path = os.path.join(CONFIG.output_dir, "rf_primary_permutation_importance_raw_features.csv")
if not os.path.exists(rf_pi_path):
    raise FileNotFoundError(f"Cannot find {rf_pi_path}. Run 12M first.")

rf_pi = pd.read_csv(rf_pi_path).copy()
rf_pi["Feature"] = rf_pi["feature"].apply(clean_feature_label)
rf_pi["Importance"] = rf_pi["importance_mean"]
rf_pi["SD"] = rf_pi["importance_sd"]

rf_pi_clean = rf_pi[["feature", "Feature", "Importance", "SD"]].sort_values("Importance", ascending=False)
rf_pi_clean_path = os.path.join(PUB_DIR, "table_rf_permutation_importance_raw_features.csv")
rf_pi_clean.to_csv(rf_pi_clean_path, index=False)

def plot_rf_permutation_publication(rf_df, out_path, top_n=12):
    plot_df = rf_df.sort_values("Importance", ascending=False).head(top_n).iloc[::-1].copy()
    fig, ax = plt.subplots(figsize=(8, max(4.5, 0.38 * len(plot_df) + 1.2)))
    ax.barh(plot_df["Feature"], plot_df["Importance"])
    ax.set_xlabel("Mean decrease in balanced accuracy")
    ax.set_title("Random forest permutation importance")
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    return plot_df

rf_pi_plot_path = os.path.join(PUB_DIR, "figure_rf_permutation_importance_publication.png")
rf_pi_plot_data = plot_rf_permutation_publication(rf_pi_clean, rf_pi_plot_path, top_n=12)
rf_pi_plot_data.to_csv(os.path.join(PUB_DIR, "figure_rf_permutation_importance_publication_data.csv"), index=False)

rf_pi_clean.head(15)

# 13G) RF SHAP helpers: transformed one-hot features
# =========================

def _get_calibrated_pipeline_for_model(artifacts, model_name="RF", fold_index=0):
    if artifacts is None or model_name not in artifacts:
        raise ValueError(f"{model_name} not found in artifacts. Run the primary scenario first.")
    calibrated = artifacts[model_name]
    if not hasattr(calibrated, "calibrated_classifiers_"):
        raise ValueError("Expected CalibratedClassifierCV object with calibrated_classifiers_.")
    cal = calibrated.calibrated_classifiers_[fold_index]
    if hasattr(cal, "estimator"):
        return cal.estimator
    if hasattr(cal, "base_estimator"):
        return cal.base_estimator
    raise ValueError("Could not recover the calibrated base estimator.")


def _clean_transformed_feature_name(x: str) -> str:
    x = str(x)

    # Remove common sklearn transformer prefixes.
    for prefix in ["num__", "ord__", "cat__", "remainder__", "prep__"]:
        x = x.replace(prefix, "")

    # Clean sklearn one-hot names.
    x = x.replace("missingindicator_", "")
    x = x.replace("_nan", ": Missing")

    for raw, label in sorted(FEATURE_LABELS.items(), key=lambda kv: len(kv[0]), reverse=True):
        if x == raw:
            return label
        if x.startswith(raw + "_"):
            cat = x[len(raw) + 1:].replace("_", " ")
            return f"{label}: {cat}"

    return x.replace("_", " ")


def _manual_feature_names_from_preprocessor(preprocessor, input_features):
    """
    Robust fallback for feature names when get_feature_names_out() fails.
    Handles common ColumnTransformer patterns.
    """
    names = []

    for name, transformer, cols in preprocessor.transformers_:
        if name == "remainder" and transformer == "drop":
            continue

        if cols is None:
            continue

        cols = list(cols)

        # If transformer is a Pipeline, inspect final step.
        final_transformer = transformer
        if isinstance(transformer, Pipeline):
            final_transformer = transformer.steps[-1][1]

        # OneHotEncoder case.
        if hasattr(final_transformer, "get_feature_names_out"):
            try:
                out = final_transformer.get_feature_names_out(cols)
                names.extend(list(out))
                continue
            except Exception:
                pass

        names.extend(cols)

    return names


def _get_transformed_feature_names(base_pipeline, predictors, aux_cols):
    """
    Extract feature names from the fitted preprocessing part of the RF pipeline.
    This prevents SHAP plots from showing generic feature_0, feature_1 labels.
    """
    input_features = predictors + [c for c in aux_cols if c not in predictors]

    prep = base_pipeline.named_steps.get("prep", None)

    if prep is None:
        for _, step in base_pipeline.steps:
            if hasattr(step, "get_feature_names_out") or hasattr(step, "transformers_"):
                prep = step
                break

    if prep is None:
        raise ValueError("Could not locate preprocessing step in RF pipeline.")

    try:
        feature_names = prep.get_feature_names_out(input_features)
        feature_names = list(feature_names)
    except Exception:
        feature_names = _manual_feature_names_from_preprocessor(prep, input_features)

    return feature_names


def compute_rf_shap_transformed(
    primary_artifacts,
    X: pd.DataFrame,
    y: pd.Series,
    predictors: List[str],
    aux_cols: List[str],
    config: EOBCConfig,
    sample_size: Optional[int] = None,
    fold_index: int = 0,
):
    if not HAS_SHAP:
        raise ImportError("shap is not installed in this environment. Install with: pip install shap")

    sample_size = sample_size or config.shap_sample_size

    X_model = X[predictors + [c for c in aux_cols if c not in predictors]].copy()
    _, X_test, _, y_test = train_test_split(
        X_model, y, test_size=config.test_size, stratify=y, random_state=RANDOM_STATE
    )
    X_sample = X_test.sample(min(sample_size, len(X_test)), random_state=RANDOM_STATE).copy()

    base_pipeline = _get_calibrated_pipeline_for_model(primary_artifacts, "RF", fold_index=fold_index)

    # Pipeline excluding classifier = preprocessing pipeline.
    prep_pipe = Pipeline(base_pipeline.steps[:-1])
    rf_clf = base_pipeline.named_steps["clf"]

    X_trans = prep_pipe.transform(X_sample)
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()

    feature_names = _get_transformed_feature_names(base_pipeline, predictors, aux_cols)

    if len(feature_names) != X_trans.shape[1]:
        print("WARNING: feature-name count does not match transformed matrix width.")
        print("Number of feature names:", len(feature_names))
        print("Transformed matrix columns:", X_trans.shape[1])
        feature_names = [f"transformed_feature_{i}" for i in range(X_trans.shape[1])]

    clean_names = [_clean_transformed_feature_name(f) for f in feature_names]

    X_trans_df = pd.DataFrame(X_trans, columns=clean_names, index=X_sample.index)

    explainer = shap.TreeExplainer(rf_clf)
    shap_values = explainer.shap_values(X_trans_df)

    # Binary RF can return list[class] or array[n, p, class]. Use EOBC/positive-class values.
    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif getattr(shap_values, "ndim", None) == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

    shap_df = pd.DataFrame({
        "feature": feature_names,
        "clean_feature": clean_names,
        "mean_abs_shap": np.abs(sv).mean(axis=0)
    }).sort_values("mean_abs_shap", ascending=False)

    return {
        "X_trans": X_trans_df,
        "X_sample": X_sample,
        "shap_values": sv,
        "feature_names": list(feature_names),
        "clean_feature_names": clean_names,
        "shap_importance": shap_df,
    }



# 13H) Run RF SHAP: one-hot/transformed feature bar plot + beeswarm plot
# =========================
X, y, predictors, aux_cols = load_data(CONFIG)

SHAP_RESULTS = compute_rf_shap_transformed(
    primary_artifacts=PRIMARY_ARTIFACTS,
    X=X,
    y=y,
    predictors=predictors,
    aux_cols=aux_cols,
    config=CONFIG,
    sample_size=300,   # increase to 500-1000 if runtime is acceptable
    fold_index=0
)

shap_imp = SHAP_RESULTS["shap_importance"].copy()
shap_imp.to_csv(os.path.join(PUB_DIR, "table_rf_shap_mean_abs_onehot_features.csv"), index=False)

# SHAP bar plot, top transformed features.
top_shap = shap_imp.head(20).iloc[::-1].copy()
fig, ax = plt.subplots(figsize=(9, max(6, 0.34 * len(top_shap) + 1.4)))
ax.barh(top_shap["clean_feature"], top_shap["mean_abs_shap"])
ax.set_xlabel("Mean absolute SHAP value")
ax.set_title("Random forest SHAP importance")
plt.tight_layout()
shap_bar_path = os.path.join(PUB_DIR, "figure_rf_shap_bar_onehot_features.png")
fig.savefig(shap_bar_path, dpi=300, bbox_inches="tight")
fig.savefig(shap_bar_path.replace(".png", ".pdf"), bbox_inches="tight")
plt.close(fig)

# SHAP beeswarm summary plot.
try:
    X_shap_df = SHAP_RESULTS["X_trans"]
    plt.figure()
    shap.summary_plot(
        SHAP_RESULTS["shap_values"],
        X_shap_df,
        max_display=20,
        show=False
    )
    shap_beeswarm_path = os.path.join(PUB_DIR, "figure_rf_shap_beeswarm_onehot_features.png")
    plt.savefig(shap_beeswarm_path, dpi=300, bbox_inches="tight")
    plt.savefig(shap_beeswarm_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print("Saved SHAP beeswarm:", shap_beeswarm_path)
except Exception as e:
    print("SHAP beeswarm failed, but SHAP importance table/bar plot were saved.")
    print("Error:", repr(e))


# 13I) SHAP dependence/scatter plots for key transformed features
# =========================
import re

KEY_SHAP_FEATURE_SEARCH_TERMS = [
    "Employment status",
    "Comorbidity",
    "Education level",
    "Year of diagnosis",
]

def plot_shap_dependence_like(results, search_terms, out_dir=PUB_DIR, max_plots=4):
    shap_df = results["shap_importance"].copy()
    X_shap_df = results["X_trans"]  # already a labeled DataFrame from updated 13G
    sv = results["shap_values"]

    made = []

    for term in search_terms:
        matches = shap_df[shap_df["clean_feature"].str.contains(term, case=False, regex=False)]
        if matches.empty:
            print(f"No SHAP feature matched: {term}")
            continue

        feat = matches.iloc[0]["clean_feature"]

        if feat not in X_shap_df.columns:
            print(f"Matched feature not found in SHAP matrix columns: {feat}")
            continue

        j = list(X_shap_df.columns).index(feat)

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.scatter(X_shap_df.iloc[:, j], sv[:, j], alpha=0.35, s=16)
        ax.axhline(0, linestyle="--", linewidth=1)
        ax.set_xlabel(feat)
        ax.set_ylabel("SHAP value for EOBC classification")
        ax.set_title(f"SHAP dependence: {feat}")
        plt.tight_layout()

        safe = re.sub(r"[^A-Za-z0-9]+", "_", feat).strip("_")[:80]
        out_path = os.path.join(out_dir, f"figure_shap_dependence_{safe}.png")

        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
        plt.close(fig)

        made.append(out_path)

        if len(made) >= max_plots:
            break

    return made

SHAP_DEPENDENCE_FILES = plot_shap_dependence_like(SHAP_RESULTS, KEY_SHAP_FEATURE_SEARCH_TERMS)


# 13J) Publication-output manifest
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
# manifest  # notebook display removed
