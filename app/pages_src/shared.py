"""Cached data/model loaders shared across every page."""

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from traffic_accident_severity import config, data, interpretability, model  # noqa: E402


@st.cache_resource
def get_pipeline():
    return model.load_pipeline()


@st.cache_data
def get_accidents_df() -> pd.DataFrame:
    if config.RAW_CSV.exists():
        return data.load_accidents()

    token = os.getenv("KAGGLE_API_TOKEN")
    if not token:
        try:
            token = st.secrets.get("KAGGLE_API_TOKEN")
        except (FileNotFoundError, KeyError):
            token = None

    try:
        downloaded = data.download_accidents(api_token=token)
        return pd.read_csv(downloaded)
    except (FileNotFoundError, RuntimeError):
        st.error(
            "The accident dataset is unavailable. Add `KAGGLE_API_TOKEN` to "
            "Streamlit secrets after confirming access to the documented Kaggle dataset."
        )
        st.caption("See docs/SETUP_AND_DEPLOYMENT.md for recovery steps.")
        st.stop()


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
