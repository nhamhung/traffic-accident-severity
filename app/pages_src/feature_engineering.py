"""Feature Engineering page: what `FeatureEngineer`/`RareCategoryGrouper`
derive, and why — plus the central data-understanding decision this
project is built around (excluding casualty-outcome columns).
"""

import plotly.express as px
import streamlit as st

from . import shared
from traffic_accident_severity import config
from traffic_accident_severity.features import FeatureEngineer


def render():
    st.title("🔧 Feature Engineering")
    st.caption("The real engineering decisions behind this project's feature set.")

    df = shared.get_accidents_df()
    engineered = FeatureEngineer().fit_transform(df[config.RAW_FEATURE_COLS])

    st.subheader("1. Excluding casualty-outcome columns")
    st.markdown(
        "The raw data mixes **pre-accident risk factors** (driver, vehicle, "
        "road, weather — knowable *before* a crash) with **post-accident "
        "casualty-outcome columns** (who got hurt, how badly — recorded "
        "*after*). Using the latter as features would be circular and "
        "useless for real prevention: a system that needs to already know "
        "how badly someone was hurt to predict how badly someone was hurt "
        "isn't actionable."
    )
    excluded = ", ".join(f"`{c}`" for c in config.CASUALTY_OUTCOME_COLS)
    st.caption(f"Excluded: {excluded}")

    st.subheader("2. Circular hour-of-day encoding")
    st.markdown(
        "`Time` (e.g. `\"17:02:00\"`) is parsed into `hour` (0-23), then "
        "encoded as a point on a 24-hour circle (`hour_sin`/`hour_cos`) — "
        "hour 23 and hour 0 are one hour apart, not \"far away\" from each "
        "other, the same reasoning the Spotify project applies to musical "
        "key and the CO2 Emissions project applies to week-of-year."
    )
    fig = px.scatter_polar(
        engineered.sample(min(2000, len(engineered)), random_state=config.RANDOM_SEED),
        r=[1] * min(2000, len(engineered)),
        theta=engineered["hour"].sample(min(2000, len(engineered)), random_state=config.RANDOM_SEED) * 15,
        title="Each accident's hour, placed on a 24-hour circle (15° per hour)",
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("3. Grouping rare categories")
    st.markdown(
        "`Cause_of_accident` (20 distinct values), `Type_of_vehicle` (17), "
        "and `Area_accident_occured` (14) each have a long tail of "
        "rare codes. `RareCategoryGrouper` collapses any code below 1% "
        "frequency into a shared \"rare\" bucket before one-hot encoding — "
        "the same technique academic_success uses for occupation/"
        "nationality codes."
    )
    col = st.selectbox("Column", config.RARE_GROUPED_COLS)
    counts = df[col].value_counts(normalize=True).reset_index()
    counts.columns = [col, "share"]
    st.plotly_chart(px.bar(counts, x=col, y="share", title=f"{col} frequency"), width="stretch")
