"""Predict page: load a real accident record (or start from dataset
averages), tweak the handful of fields that matter most, and see the
model's predicted severity update.

Only pre-accident risk-factor fields are shown — never the casualty-
outcome columns (`Casualty_severity`, `Number_of_casualties`, ...),
which the model was never trained on (see `config.py`'s module
docstring for why). Fields most likely to matter (by SHAP importance)
are promoted to the top, instead of asking anyone to fill in 24 fields
with no sense of which ones count.
"""

import pandas as pd
import streamlit as st

from . import shared
from traffic_accident_severity import config

PROMOTED_FIELDS = [
    "Cause_of_accident",
    "Number_of_vehicles_involved",
    "Light_conditions",
    "Weather_conditions",
    "Type_of_collision",
    "Road_surface_conditions",
]


def _field_key(col: str) -> str:
    return f"field_{col}"


def _init_field_state(col: str, default_value):
    key = _field_key(col)
    if key not in st.session_state:
        st.session_state[key] = default_value


def _load_random_accident():
    df = shared.get_accidents_df()
    row = df.sample(1).iloc[0]
    for col in config.RAW_FEATURE_COLS:
        st.session_state[_field_key(col)] = row[col]
    st.session_state["hour_pick"] = int(pd.to_datetime(row[config.TIME_COL], format="%H:%M:%S").hour)
    st.session_state["loaded_accident_features"] = {
        col: st.session_state[_field_key(col)] for col in config.RAW_FEATURE_COLS
    }
    st.session_state["loaded_accident_features"][config.TIME_COL] = (
        f"{st.session_state['hour_pick']:02d}:00:00"
    )
    raw_severity = row[config.TARGET_COL]
    st.session_state["actual_severity_raw"] = raw_severity
    st.session_state["actual_severity"] = "Severe" if raw_severity in config.SEVERE_CLASSES else "Not Severe"
    st.session_state["loaded_a_record"] = True


def _reset_to_average():
    defaults = shared.get_default_row()
    for col in config.RAW_FEATURE_COLS:
        st.session_state[_field_key(col)] = defaults[col]
    st.session_state["hour_pick"] = int(pd.to_datetime(defaults[config.TIME_COL], format="%H:%M:%S").hour)
    st.session_state.pop("actual_severity", None)
    st.session_state.pop("actual_severity_raw", None)
    st.session_state.pop("loaded_accident_features", None)
    st.session_state["loaded_a_record"] = False


def _select(col: str, options: list, defaults: dict):
    _init_field_state(col, defaults[col])
    # A widget with a `key` already present in session_state must not also
    # receive `index` (Streamlit treats that as two conflicting sources of
    # truth) — so any out-of-`options` value (e.g. a NaN from a real
    # record, or a missing default) is sanitized into session_state
    # *before* the widget is created, rather than passed as `index`.
    if st.session_state[_field_key(col)] not in options:
        st.session_state[_field_key(col)] = options[0]
    st.selectbox(col, options=options, key=_field_key(col))


def _matches_loaded_accident(current: dict, reference: dict | None = None) -> bool:
    """Only a completely unchanged historical row has valid ground truth."""
    if reference is None:
        reference = st.session_state.get("loaded_accident_features")
    if not reference or set(reference) != set(config.RAW_FEATURE_COLS):
        return False
    for col in config.RAW_FEATURE_COLS:
        current_value, reference_value = current[col], reference[col]
        if pd.isna(current_value) and pd.isna(reference_value):
            continue
        if current_value != reference_value:
            return False
    return True


def render():
    st.title("🎯 Predict Accident Severity")
    st.caption(
        "Predicts whether an accident with these pre-crash conditions is "
        "likely to be **Severe** (a serious or fatal injury) or **Not "
        "Severe** (a slight injury) — using only information available "
        "*before* the crash (driver, vehicle, road, weather, collision "
        "dynamics), never casualty outcomes recorded afterward."
    )

    try:
        pipeline = shared.get_pipeline()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    df = shared.get_accidents_df()
    defaults = shared.get_default_row()

    col1, col2 = st.columns(2)
    with col1:
        st.button("🎲 Load a random real accident", on_click=_load_random_accident, width="stretch")
    with col2:
        st.button("↺ Reset to dataset average", on_click=_reset_to_average, width="stretch")

    if st.session_state.get("loaded_a_record"):
        st.info(
            "Loaded a real accident record. Its recorded severity is shown only "
            "while every field remains unchanged."
        )

    st.subheader("The fields that matter most")
    st.caption("Promoted to the top based on this project's SHAP analysis — see Model Insights.")

    c1, c2 = st.columns(2)
    with c1:
        _select("Cause_of_accident", sorted(df["Cause_of_accident"].dropna().unique().tolist()), defaults)
        _select("Type_of_collision", sorted(df["Type_of_collision"].dropna().unique().tolist()), defaults)
        _select("Road_surface_conditions", sorted(df["Road_surface_conditions"].dropna().unique().tolist()), defaults)
    with c2:
        _select("Light_conditions", sorted(df["Light_conditions"].dropna().unique().tolist()), defaults)
        _select("Weather_conditions", sorted(df["Weather_conditions"].dropna().unique().tolist()), defaults)
        _init_field_state("Number_of_vehicles_involved", int(defaults["Number_of_vehicles_involved"]))
        st.number_input(
            "Number_of_vehicles_involved", min_value=1, max_value=10, step=1,
            key=_field_key("Number_of_vehicles_involved"),
        )

    _init_field_state("hour_pick", 12)
    st.slider("Hour of day", min_value=0, max_value=23, key="hour_pick")

    with st.expander("Other details (driver, vehicle, road, junction)"):
        st.markdown("**Driver**")
        d1, d2, d3 = st.columns(3)
        with d1:
            _select("Age_band_of_driver", sorted(df["Age_band_of_driver"].dropna().unique().tolist()), defaults)
            _select("Sex_of_driver", sorted(df["Sex_of_driver"].dropna().unique().tolist()), defaults)
        with d2:
            _select("Educational_level", sorted(df["Educational_level"].dropna().unique().tolist()), defaults)
            _select("Vehicle_driver_relation", sorted(df["Vehicle_driver_relation"].dropna().unique().tolist()), defaults)
        with d3:
            _select("Driving_experience", sorted(df["Driving_experience"].dropna().unique().tolist()), defaults)
            _select("Day_of_week", sorted(df["Day_of_week"].dropna().unique().tolist()), defaults)

        st.markdown("**Vehicle**")
        v1, v2 = st.columns(2)
        with v1:
            _select("Type_of_vehicle", sorted(df["Type_of_vehicle"].dropna().unique().tolist()), defaults)
            _select("Owner_of_vehicle", sorted(df["Owner_of_vehicle"].dropna().unique().tolist()), defaults)
        with v2:
            _select("Service_year_of_vehicle", sorted(df["Service_year_of_vehicle"].dropna().unique().tolist()), defaults)
            _select("Defect_of_vehicle", sorted(df["Defect_of_vehicle"].dropna().unique().tolist()), defaults)

        st.markdown("**Road**")
        r1, r2, r3 = st.columns(3)
        with r1:
            _select("Area_accident_occured", sorted(df["Area_accident_occured"].dropna().unique().tolist()), defaults)
            _select("Lanes_or_Medians", sorted(df["Lanes_or_Medians"].dropna().unique().tolist()), defaults)
        with r2:
            _select("Road_allignment", sorted(df["Road_allignment"].dropna().unique().tolist()), defaults)
            _select("Types_of_Junction", sorted(df["Types_of_Junction"].dropna().unique().tolist()), defaults)
        with r3:
            _select("Road_surface_type", sorted(df["Road_surface_type"].dropna().unique().tolist()), defaults)
            _select("Vehicle_movement", sorted(df["Vehicle_movement"].dropna().unique().tolist()), defaults)
        _select("Pedestrian_movement", sorted(df["Pedestrian_movement"].dropna().unique().tolist()), defaults)

    if st.button("Predict severity", type="primary"):
        row = {col: st.session_state[_field_key(col)] for col in config.RAW_FEATURE_COLS}
        row[config.TIME_COL] = f"{st.session_state['hour_pick']:02d}:00:00"
        X = pd.DataFrame([row])[config.RAW_FEATURE_COLS]

        prediction = pipeline.predict(X)[0]
        proba = pipeline.predict_proba(X)[0]
        classes = pipeline.named_steps["model"].classes_

        actual = st.session_state.get("actual_severity")
        actual_raw = st.session_state.get("actual_severity_raw")
        if actual is not None and _matches_loaded_accident(row):
            match = "✅ matches the recorded outcome" if actual == prediction else "❌ differs from the recorded outcome"
            st.subheader(
                f"Prediction: **{prediction}**  |  Recorded outcome: **{actual}** "
                f"(originally recorded as *{actual_raw}*) ({match})"
            )
            st.caption("This comparison is for one unchanged historical row; it is not proof that the model is always correct.")
        else:
            st.subheader(f"Prediction: **{prediction}**")
            if actual is not None:
                st.info("The loaded accident's fields were edited, so its original recorded severity no longer applies and is not compared.")

        proba_df = pd.DataFrame({"Severity": classes, "Probability": proba}).sort_values(
            "Probability", ascending=False
        )
        st.bar_chart(proba_df.set_index("Severity"))
        st.dataframe(proba_df, hide_index=True)
