"""Dataset Overview page: what's in the Addis Ababa RTA dataset before
any modeling — severity balance, missingness, and time-of-day patterns.
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from . import shared
from traffic_accident_severity import config


def render():
    st.title("📊 Dataset Overview")
    st.caption(
        "12,316 real accident records from Addis Ababa sub-city police "
        "logs, 2017-2020 — chosen for its contextual relevance to "
        "Vietnam's own mixed-vehicle, developing-country traffic "
        "environment over larger but car-centric Western datasets."
    )

    df = shared.get_accidents_df()

    c1, c2, c3 = st.columns(3)
    c1.metric("Accident records", f"{len(df):,}")
    c2.metric("Feature columns used", f"{len(config.RAW_FEATURE_COLS)}")
    c3.metric("Casualty-outcome columns excluded", f"{len(config.CASUALTY_OUTCOME_COLS)}")

    st.subheader("Severity is heavily imbalanced")
    st.caption(
        "The raw 3-class breakdown, verified directly against the CSV — "
        "Fatal injury alone is only ~1.3% of rows. This is why macro-F1 "
        "(not accuracy) is used throughout, and why class-imbalance "
        "handling gets its own comparison on the Model Insights page."
    )
    severity_counts = df[config.TARGET_COL].value_counts(normalize=True).reset_index()
    severity_counts.columns = ["severity", "share"]
    st.plotly_chart(px.bar(severity_counts, x="severity", y="share", title="Raw target class balance"), width="stretch")

    st.caption(
        "The model itself doesn't predict these 3 raw classes directly — "
        "see below."
    )
    severe_counts = (
        df[config.TARGET_COL]
        .isin(config.SEVERE_CLASSES)
        .map({True: "Severe", False: "Not Severe"})
        .value_counts(normalize=True)
        .reset_index()
    )
    severe_counts.columns = ["severity", "share"]
    st.plotly_chart(
        px.bar(severe_counts, x="severity", y="share", title="What the model actually predicts (binary)"),
        width="stretch",
    )
    st.caption(
        "Serious + Fatal injury are collapsed into one \"Severe\" class "
        "(~15.4% of rows). A 3-class model could barely detect Fatal "
        "injuries at all (~3% recall, cross-validated) — this binary "
        "reframing raised recall for the outcome that matters most to "
        "~35%, measured directly rather than assumed. See `config.py`'s "
        "module docstring for the full reasoning."
    )

    st.subheader("When do accidents happen?")
    # `Time` isn't consistently zero-padded (verified directly: some rows
    # are e.g. "1:15:00", not "01:15:00") — naive string-slicing breaks on
    # those; parsing as a real time handles it correctly, same as
    # `FeatureEngineer`.
    hour = pd.to_datetime(df[config.TIME_COL], format="%H:%M:%S", errors="coerce").dt.hour
    st.plotly_chart(
        px.histogram(x=hour, nbins=24, title="Accidents by hour of day", labels={"x": "Hour"}),
        width="stretch",
    )

    st.subheader("Missing data (real, not synthetic)")
    st.caption(
        "Several columns are missing a substantial share of values — "
        "handled with most-frequent imputation (categorical) rather than "
        "dropped, since dropping would discard a third of the dataset "
        "for some columns."
    )
    missing = (df[config.RAW_FEATURE_COLS].isna().mean() * 100).sort_values(ascending=False)
    missing = missing[missing > 0].reset_index()
    missing.columns = ["column", "missing_pct"]
    st.plotly_chart(px.bar(missing, x="column", y="missing_pct", title="% missing by column"), width="stretch")
