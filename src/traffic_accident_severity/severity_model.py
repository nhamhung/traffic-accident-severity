"""The production model: CatBoost + a tuned Random Forest, averaged, with a tuned cut-off.

Why this model (measured with `scripts/evaluate.py`, 5-fold cross-validation
grouped by accident, repeated 3 times):

- CatBoost handles the 22 categorical columns natively (no one-hot), which
  beat the previous one-hot Random Forest clearly on every metric.
- Averaging it with a tuned one-hot Random Forest (deeper trees, 30% of
  features per split) ranks severe accidents better still (PR-AUC 0.314 ->
  0.323, ROC-AUC 0.686 -> 0.698): the two models make different mistakes.
- The cut-off for calling an accident "Severe" is chosen to maximise macro-F1
  on out-of-fold predictions, not left at 0.5: with only ~15% Severe
  accidents, a calibrated model rarely says "more likely Severe than not".
- The CatBoost half explains each prediction in terms of the original
  columns ("darkness, no street lighting"), not one-hot fragments, which is
  what the app shows students.

Rows of one accident: the dataset records many accidents once per vehicle
(same time, place, weather, collision type). `accident_groups()` finds them
so cross-validation never puts part of an accident in training and the rest
in testing, which would flatter the scores.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from . import config
from .features import build_feature_pipeline

# Columns that together identify one accident (one row is written per vehicle).
ACCIDENT_KEY = [
    "Time", "Day_of_week", "Area_accident_occured", "Light_conditions", "Weather_conditions",
    "Road_surface_conditions", "Type_of_collision", "Number_of_vehicles_involved", "Number_of_casualties",
]
CATEGORICAL = [c for c in config.RAW_FEATURE_COLS if c not in (config.TIME_COL, "Number_of_vehicles_involved")]
MODEL_FEATURES = ["hour"] + ["Number_of_vehicles_involved"] + CATEGORICAL

CATBOOST_PARAMS = {"iterations": 1500, "learning_rate": 0.03, "depth": 8, "l2_leaf_reg": 5}
FOREST_PARAMS = {"n_estimators": 300, "min_samples_leaf": 2, "max_features": 0.3, "class_weight": "balanced_subsample"}
MISSING = "missing"


def accident_groups(df: pd.DataFrame) -> np.ndarray:
    """An id per accident: rows sharing every ACCIDENT_KEY column."""
    return df.groupby(ACCIDENT_KEY, dropna=False).ngroup().to_numpy()


def to_model_frame(X: pd.DataFrame) -> pd.DataFrame:
    """Raw feature columns -> CatBoost input: hour as a number, categories as strings."""
    out = pd.DataFrame(index=X.index)
    time = pd.to_datetime(X[config.TIME_COL], format="%H:%M:%S", errors="coerce")
    out["hour"] = time.dt.hour + time.dt.minute / 60
    out["Number_of_vehicles_involved"] = pd.to_numeric(X["Number_of_vehicles_involved"], errors="coerce")
    for col in CATEGORICAL:
        out[col] = X[col].astype(object).where(X[col].notna(), MISSING).astype(str)
    return out[MODEL_FEATURES]


def best_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    """The cut-off that maximises macro-F1."""
    grid = np.linspace(0.05, 0.95, 181)
    return float(grid[int(np.argmax([f1_score(y_true, proba >= t, average="macro") for t in grid]))])


class SeverityModel(BaseEstimator, ClassifierMixin):
    """Severity classifier: CatBoost, optionally averaged with a Random Forest,
    with its own decision threshold.

    `fit` takes the raw feature frame and string labels ("Severe" / "Not
    Severe"); `predict_proba` columns follow `classes_` like any sklearn model.
    """

    def __init__(self, threshold: float = 0.5, params: dict | None = None, forest: bool = True,
                 random_state: int = config.RANDOM_SEED):
        self.threshold = threshold
        self.params = params
        self.forest = forest
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y) -> "SeverityModel":
        from catboost import CatBoostClassifier

        y = np.asarray(y)
        self.classes_ = np.array(config.TARGET_CLASSES)
        self.model_ = CatBoostClassifier(
            **{**CATBOOST_PARAMS, **(self.params or {})}, cat_features=CATEGORICAL,
            random_seed=self.random_state, verbose=0, allow_writing_files=False, thread_count=-1,
        )
        self.model_.fit(to_model_frame(X), (y == "Severe").astype(int))
        self.forest_ = None
        if self.forest:
            from sklearn.pipeline import Pipeline

            steps = list(build_feature_pipeline().steps) + [("model", RandomForestClassifier(
                **FOREST_PARAMS, n_jobs=-1, random_state=self.random_state))]
            self.forest_ = Pipeline(steps).fit(X, (y == "Severe").astype(int))
        return self

    def catboost_probability(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(to_model_frame(X))[:, 1]

    def severe_probability(self, X: pd.DataFrame) -> np.ndarray:
        p = self.catboost_probability(X)
        if self.forest_ is not None:
            p = 0.5 * p + 0.5 * self.forest_.predict_proba(X)[:, 1]
        return p

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = self.severe_probability(X)
        return np.column_stack([1 - p, p])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.where(self.severe_probability(X) >= self.threshold, "Severe", "Not Severe")

    def explain(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-row contribution of each raw column to the CatBoost half's Severe
        score (SHAP, in log-odds; positive = pushes towards Severe). Columns:
        MODEL_FEATURES. The forest half, on one-hot columns, isn't explained."""
        from catboost import Pool

        frame = to_model_frame(X)
        shap = self.model_.get_feature_importance(Pool(frame, cat_features=CATEGORICAL), type="ShapValues")
        return pd.DataFrame(shap[:, :-1], columns=MODEL_FEATURES, index=X.index)


@dataclass
class CrossValidationResult:
    oof: np.ndarray                      # out-of-fold P(Severe), averaged over repeats
    folds: pd.DataFrame                  # one row of metrics per test fold
    thresholds: list = field(default_factory=list)

    def summary(self) -> dict:
        return {k: float(self.folds[k].mean()) for k in self.folds} | {
            "f1_tuned_sd": float(self.folds["f1_tuned"].std())}


def cross_validate(X: pd.DataFrame, y: pd.Series, groups: np.ndarray, make_model, repeats: int = 3,
                   n_splits: int = 5, log=print) -> CrossValidationResult:
    """Grouped, stratified K-fold, repeated. The macro-F1 cut-off for each
    test fold is chosen from an inner grouped 3-fold on its training part
    only, so the reported "tuned" F1 never peeks at the test fold."""
    y_bin = (np.asarray(y) == "Severe").astype(int)
    oof = np.zeros(len(y_bin))
    rows = []
    for r in range(repeats):
        outer = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_SEED + r)
        for k, (tr, te) in enumerate(outer.split(X, y_bin, groups)):
            fitted = make_model().fit(X.iloc[tr], np.asarray(y)[tr])
            p = fitted.severe_probability(X.iloc[te]) if hasattr(fitted, "severe_probability") \
                else fitted.predict_proba(X.iloc[te])[:, list(fitted.classes_).index("Severe")]
            oof[te] += p / repeats
            inner_p = np.zeros(len(tr))
            inner = StratifiedGroupKFold(3, shuffle=True, random_state=r * 10 + k)
            for itr, ite in inner.split(tr, y_bin[tr], groups[tr]):
                m = make_model().fit(X.iloc[tr[itr]], np.asarray(y)[tr[itr]])
                inner_p[ite] = m.severe_probability(X.iloc[tr[ite]]) if hasattr(m, "severe_probability") \
                    else m.predict_proba(X.iloc[tr[ite]])[:, list(m.classes_).index("Severe")]
            t = best_threshold(y_bin[tr], inner_p)
            rows.append({"f1_default": f1_score(y_bin[te], p >= 0.5, average="macro"),
                         "f1_tuned": f1_score(y_bin[te], p >= t, average="macro"), "threshold": t,
                         "pr_auc": average_precision_score(y_bin[te], p), "roc_auc": roc_auc_score(y_bin[te], p),
                         "recall_severe": float(((p >= t) & (y_bin[te] == 1)).sum() / max(1, y_bin[te].sum())),
                         "precision_severe": float(((p >= t) & (y_bin[te] == 1)).sum() / max(1, (p >= t).sum()))})
            log(f"  repeat {r + 1}/{repeats}, fold {k + 1}/{n_splits}: macro-F1 {rows[-1]['f1_tuned']:.3f}")
    return CrossValidationResult(oof=oof, folds=pd.DataFrame(rows), thresholds=[r["threshold"] for r in rows])
