"""Feature engineering and the shared preprocessing pipeline.

Single source of truth for turning the raw RTA columns into model-ready
features. The notebook, `scripts/train.py`, and the Streamlit app all
call `build_feature_pipeline()` (wrapped inside the fitted pipeline
saved to `models/model.joblib`) so none of them can silently diverge.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Derives `hour` (0-23) from the raw `Time` string (`"17:02:00"`),
    plus a cyclical `hour_sin`/`hour_cos` encoding — hour 23 and hour 0
    are one hour apart, not "far away" from each other, the same
    circular-feature reasoning the Spotify project applies to musical
    key and the CO2 Emissions project applies to week-of-year.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        hour = pd.to_datetime(df[config.TIME_COL], format="%H:%M:%S", errors="coerce").dt.hour
        df["hour"] = hour
        angle = 2 * np.pi * hour / 24
        df["hour_sin"] = np.sin(angle)
        df["hour_cos"] = np.cos(angle)
        df = df.drop(columns=[config.TIME_COL])
        return df

    def get_feature_names_out(self, input_features=None):
        base = [f for f in (input_features or []) if f != config.TIME_COL]
        return np.array(base + ["hour", "hour_sin", "hour_cos"])


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Collapses low-frequency categories (below `min_frequency`) into a
    shared sentinel, learned from training data only. Same technique as
    academic_success's transformer of the same name, applied here to
    `Cause_of_accident`/`Type_of_vehicle`/`Area_accident_occured`.
    """

    def __init__(self, columns: list[str] | None = None, min_frequency: float = 0.01):
        self.columns = columns
        self.min_frequency = min_frequency

    def fit(self, X: pd.DataFrame, y=None) -> "RareCategoryGrouper":
        columns = self.columns if self.columns is not None else config.RARE_GROUPED_COLS
        self.kept_categories_: dict[str, set] = {}
        for col in columns:
            freqs = X[col].value_counts(normalize=True)
            self.kept_categories_[col] = set(freqs[freqs >= self.min_frequency].index)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for col, kept in self.kept_categories_.items():
            df[col] = df[col].where(df[col].isin(kept), "rare")
        return df

    def get_feature_names_out(self, input_features=None):
        return np.array(list(input_features) if input_features is not None else [])


def build_preprocessor() -> ColumnTransformer:
    """Numeric columns: median-impute, then standard-scale. Categorical
    columns: most-frequent-impute (several columns are missing 20-36% of
    values — verified directly), then one-hot encode with unknown
    categories mapped to all-zero rather than raising at inference time.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, config.NUMERIC_COLS),
            ("categorical", categorical_pipeline, config.CATEGORICAL_COLS),
        ]
    )


def build_feature_pipeline() -> Pipeline:
    """FeatureEngineer + RareCategoryGrouper + preprocessor, without a
    final estimator.
    """
    return Pipeline(
        steps=[
            ("engineer", FeatureEngineer()),
            ("group_rare", RareCategoryGrouper()),
            ("preprocess", build_preprocessor()),
        ]
    )
