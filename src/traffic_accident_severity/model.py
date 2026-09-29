"""Model pipeline construction, training, evaluation, and persistence.

Task: predict whether an accident is **Severe** (Serious or Fatal
injury) or **Not Severe** (Slight injury) from pre-accident risk factors
only (see `config.py`). `data.split_features_target` does the raw-3-class
-to-binary collapse; everything downstream of it just sees a binary `y`.
Still imbalanced (~15.4% Severe) but far more workable than the raw
3-class problem, where Fatal injury alone is only ~1.3% of rows. Macro-F1
(not accuracy) is used throughout — a model that never predicts "Severe"
at all would still score ~84.6% accuracy.

Pipelines are built with `imblearn.pipeline.Pipeline` (not sklearn's)
so `build_pipeline(..., resample=True)` can insert ADASYN as a step
without a separate code path — a no-op outside of `.fit()`, so synthetic
samples never leak into a validation score or a served prediction.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate

from . import config
from .features import build_feature_pipeline


def default_estimator() -> BaseEstimator:
    from lightgbm import LGBMClassifier

    return LGBMClassifier(n_estimators=300, random_state=config.RANDOM_SEED, verbosity=-1)


def logistic_estimator() -> BaseEstimator:
    return LogisticRegression(max_iter=1000, random_state=config.RANDOM_SEED)


def random_forest_estimator() -> BaseEstimator:
    return RandomForestClassifier(
        n_estimators=300, max_depth=12, random_state=config.RANDOM_SEED, n_jobs=-1
    )


def balanced_random_forest_estimator() -> BaseEstimator:
    """The same Random Forest, but with `class_weight="balanced"` —
    reweights the loss inversely to class frequency instead of
    resampling rows, a cheaper alternative to ADASYN worth comparing
    directly rather than assuming either wins.
    """
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        class_weight="balanced",
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
    )


def xgboost_estimator() -> BaseEstimator:
    from xgboost import XGBClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.base import ClassifierMixin

    class _StringLabelXGBClassifier(BaseEstimator, ClassifierMixin):
        """XGBoost's multiclass objective requires integer-encoded `y` —
        see academic_success's identical wrapper for the same issue."""

        def __init__(self, n_estimators: int = 300, random_state: int | None = None):
            self.n_estimators = n_estimators
            self.random_state = random_state

        def fit(self, X, y):
            self._label_encoder = LabelEncoder().fit(y)
            self._model = XGBClassifier(
                n_estimators=self.n_estimators, random_state=self.random_state, eval_metric="mlogloss"
            )
            self._model.fit(X, self._label_encoder.transform(y))
            self.classes_ = self._label_encoder.classes_
            return self

        def predict(self, X):
            return self._label_encoder.inverse_transform(self._model.predict(X))

        def predict_proba(self, X):
            return self._model.predict_proba(X)

    return _StringLabelXGBClassifier(n_estimators=300, random_state=config.RANDOM_SEED)


MODEL_FACTORIES: dict[str, "callable[[], BaseEstimator]"] = {
    "Logistic Regression": logistic_estimator,
    "Random Forest": random_forest_estimator,
    "Random Forest (balanced)": balanced_random_forest_estimator,
    "LightGBM": default_estimator,
    "XGBoost": xgboost_estimator,
}


def build_pipeline(estimator: BaseEstimator | None = None, resample: bool = False) -> Pipeline:
    """`resample=True` inserts ADASYN right after preprocessing. With the
    binary Severe target (~15.4% positive, ~1,901 rows), ADASYN's default
    `n_neighbors=5` runs fine — verified directly; this needed lowering to
    3 back when the target was the raw 3-class column and Fatal injury
    alone was only ~1.3% of rows.
    """
    pipeline = build_feature_pipeline()
    steps = list(pipeline.steps)
    if resample:
        steps.append(("resample", ADASYN(random_state=config.RANDOM_SEED)))
    steps.append(("model", estimator if estimator is not None else default_estimator()))
    return Pipeline(steps=steps)


def cross_validate_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    estimator: BaseEstimator | None = None,
    cv: int = 5,
    resample: bool = False,
) -> dict:
    """Stratified-by-default cross_validate; returns mean/std accuracy
    and macro-F1.
    """
    pipeline = build_pipeline(estimator, resample=resample)
    scores = cross_validate(
        pipeline, X, y, cv=cv, scoring=["accuracy", "f1_macro"], return_train_score=False
    )
    return {
        "accuracy_mean": scores["test_accuracy"].mean(),
        "accuracy_std": scores["test_accuracy"].std(),
        "f1_macro_mean": scores["test_f1_macro"].mean(),
        "f1_macro_std": scores["test_f1_macro"].std(),
    }


def train_pipeline(
    X: pd.DataFrame, y: pd.Series, estimator: BaseEstimator | None = None, resample: bool = False
) -> Pipeline:
    """Fit a fresh pipeline on the full given data."""
    pipeline = build_pipeline(estimator, resample=resample)
    pipeline.fit(X, y)
    return pipeline


def save_pipeline(pipeline: Pipeline, path: Path = config.MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_pipeline(path: Path = config.MODEL_PATH) -> Pipeline:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Train a model first: `python scripts/train.py`."
        )
    return joblib.load(path)


def cross_val_severe_probabilities(
    X: pd.DataFrame, y: pd.Series, estimator: BaseEstimator | None = None, cv: int = 5
) -> np.ndarray:
    """Out-of-fold predicted probability of the "Severe" class for every
    row, via `cross_val_predict` — needed for a precision/recall
    threshold sweep, which a single train/test split doesn't have enough
    rows to make trustworthy given how few Severe examples there are.

    Relies on `predict_proba`'s columns being ordered by `sorted(y.unique())`
    (sklearn's convention) — asserted directly rather than assumed, since
    getting this silently backwards would flip precision and recall.
    """
    classes = sorted(y.unique())
    assert classes == config.TARGET_CLASSES, f"Unexpected classes {classes}, expected {config.TARGET_CLASSES}"
    severe_index = classes.index("Severe")

    pipeline = build_pipeline(estimator if estimator is not None else balanced_random_forest_estimator())
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=config.RANDOM_SEED)
    proba = cross_val_predict(pipeline, X, y, cv=splitter, method="predict_proba")
    return proba[:, severe_index]


def severity_threshold_sweep(y: pd.Series, proba: np.ndarray, target_recalls: list[float]) -> pd.DataFrame:
    """For each target recall level (recall of the "Severe" class), find
    the threshold that achieves it and report the precision paid for it —
    the real tradeoff of pushing the model to catch more Severe accidents.
    """
    y_numeric = (y == "Severe").astype(int)
    precision, recall, thresholds = precision_recall_curve(y_numeric, proba)
    rows = []
    for target in target_recalls:
        idx = int(np.argmin(np.abs(recall[:-1] - target)))
        rows.append(
            {
                "target_recall": target,
                "threshold": thresholds[idx],
                "precision": precision[idx],
                "recall": recall[idx],
            }
        )
    return pd.DataFrame(rows)
