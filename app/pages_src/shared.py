"""Cached data/model loaders shared across every page."""

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from traffic_accident_severity import config, data, interpretability, model  # noqa: E402


def _configure_kaggle_credentials() -> None:
    """Wire Kaggle API credentials from Streamlit secrets into the
    environment variables the `kaggle` package reads, so a deployment
    without a pre-baked Docker image (e.g. Streamlit Community Cloud)
    can fetch the dataset automatically on first load — see
    `data._download_from_kaggle`. A no-op if real environment variables
    are already set (e.g. running locally) or no `[kaggle]` secret is
    configured (falls back to `~/.kaggle/kaggle.json` if present, or to
    the manual-download error message if not).
    """
    if os.environ.get("KAGGLE_API_TOKEN") or (
        os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    ):
        return
    try:
        token = st.secrets.get("KAGGLE_API_TOKEN")
        if token:
            os.environ["KAGGLE_API_TOKEN"] = token
            return
    except Exception:
        pass
    try:
        os.environ["KAGGLE_USERNAME"] = st.secrets["kaggle"]["username"]
        os.environ["KAGGLE_KEY"] = st.secrets["kaggle"]["key"]
    except Exception:
        pass


_configure_kaggle_credentials()


@st.cache_resource
def get_pipeline():
    return model.load_pipeline()


@st.cache_data
def get_accidents_df() -> pd.DataFrame:
    return data.load_accidents()


@st.cache_data
def get_default_row() -> dict:
    """Most-frequent value per raw feature column — the "average"
    accident record a Predict-page reset button falls back to.
    """
    df = get_accidents_df()
    return {col: df[col].mode(dropna=True).iloc[0] for col in config.RAW_FEATURE_COLS}


@st.cache_data(show_spinner="Computing SHAP values (first load only)...")
def get_shap_explanation(sample_size: int = 500):
    pipeline = get_pipeline()
    df = get_accidents_df()
    X, _ = data.split_features_target(df)
    explanation, X_transformed = interpretability.compute_shap_values(
        pipeline, X, max_samples=sample_size
    )
    return explanation, X_transformed
