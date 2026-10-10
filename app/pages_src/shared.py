"""Cached loaders and small display helpers shared by every page.

Every number the app shows about the full dataset or the model's accuracy
comes from `results/` (written by `scripts/evaluate.py` on all 12,316 rows),
so a deployment that only ships the 1,200-row sample still shows the real
figures. The sample is used only to load example accidents.
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import joblib  # noqa: E402

from traffic_accident_severity import config, data, vi  # noqa: E402

RESULTS = ROOT / "results"
SEVERE_COLOUR = "#d1495b"
SAFE_COLOUR = "#2b59c3"
GREY = "#8d99ae"


def _configure_kaggle_credentials() -> None:
    """Optional: wire Kaggle credentials from Streamlit secrets so a
    deployment can use the full dataset (USE_FULL_KAGGLE_DATA). Without
    them the bundled sample is used, which is all the app needs."""
    try:
        if st.secrets.get("USE_FULL_KAGGLE_DATA"):
            os.environ["USE_FULL_KAGGLE_DATA"] = "true"
    except Exception:
        pass
    if os.environ.get("KAGGLE_API_TOKEN") or (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
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
def get_model():
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(f"Chưa có mô hình ({config.MODEL_PATH}). Hãy chạy: python scripts/train.py")
    return joblib.load(config.MODEL_PATH)


@st.cache_data
def get_accidents_df() -> pd.DataFrame:
    return data.load_accidents()


@st.cache_data
def get_result(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / name)


@st.cache_data
def get_json(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


@st.cache_data
def get_default_row() -> dict:
    """The most common value of every column: a "typical" accident."""
    df = get_accidents_df()
    return {col: df[col].mode(dropna=True).iloc[0] for col in config.RAW_FEATURE_COLS}


def options(col: str) -> list:
    """Every value a column can take (from the model's own training data summary)."""
    return get_json("data_facts.json")["values"][col]


def pct(x: float, digits: int = 0) -> str:
    return f"{100 * x:.{digits}f}%".replace(".", ",")


def num(x: float, digits: int = 0) -> str:
    """Vietnamese number format: 12.316 and 0,61."""
    s = f"{x:,.{digits}f}"
    return s.replace(",", "_").replace(".", ",").replace("_", ".")


def glossary(term: str, text: str) -> None:
    with st.expander(f"📖 {term} là gì?"):
        st.markdown(text)


def source_note() -> None:
    st.caption("Dữ liệu: hồ sơ tai nạn giao thông của cảnh sát Addis Ababa (Ethiopia), 2017–2020, "
               "12.316 bản ghi. Mô hình và phân tích: dự án này.")


__all__ = ["config", "vi"]
