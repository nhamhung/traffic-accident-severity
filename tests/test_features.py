"""Tests for feature engineering, using a small synthetic panel instead
of the real Addis Ababa RTA data.
"""

import numpy as np
import pandas as pd

from traffic_accident_severity import config
from traffic_accident_severity.features import FeatureEngineer, RareCategoryGrouper, build_feature_pipeline


def make_synthetic_panel(n_rows: int = 300, seed: int = 0) -> pd.DataFrame:
    """One row per accident record, with all raw feature columns plus
    the target — mirrors the real schema.
    """
    rng = np.random.default_rng(seed)
    hours = rng.integers(0, 24, n_rows)
    times = [f"{h:02d}:{m:02d}:00" for h, m in zip(hours, rng.integers(0, 60, n_rows))]

    df = pd.DataFrame(
        {
            config.TIME_COL: times,
            "Day_of_week": rng.choice(["Monday", "Tuesday", "Sunday"], n_rows),
            "Age_band_of_driver": rng.choice(["18-30", "31-50", "Unknown"], n_rows),
            "Sex_of_driver": rng.choice(["Male", "Female"], n_rows),
            "Educational_level": rng.choice(["Junior high school", "Above high school", None], n_rows),
            "Vehicle_driver_relation": rng.choice(["Employee", "Owner"], n_rows),
            "Driving_experience": rng.choice(["1-2yr", "Above 10yr", None], n_rows),
            "Type_of_vehicle": rng.choice(
                ["Automobile"] * 90 + ["Public (> 45 seats)"] * 5 + ["rare_type_a", "rare_type_b"], n_rows
            ),
            "Owner_of_vehicle": rng.choice(["Owner", "Governmental"], n_rows),
            "Service_year_of_vehicle": rng.choice(["Above 10yr", "5-10yrs", None], n_rows),
            "Defect_of_vehicle": rng.choice(["No defect", None], n_rows),
            "Area_accident_occured": rng.choice(
                ["Residential areas"] * 90 + ["Office areas"] * 5 + ["rare_area_a", "rare_area_b"], n_rows
            ),
            "Lanes_or_Medians": rng.choice(["Undivided Two way", None], n_rows),
            "Road_allignment": rng.choice(["Tangent road with flat terrain"], n_rows),
            "Types_of_Junction": rng.choice(["No junction", "Y Shape", None], n_rows),
            "Road_surface_type": rng.choice(["Asphalt roads"], n_rows),
            "Road_surface_conditions": rng.choice(["Dry", "Wet or damp"], n_rows),
            "Light_conditions": rng.choice(["Daylight", "Darkness - lights lit"], n_rows),
            "Weather_conditions": rng.choice(["Normal", "Raining"], n_rows),
            "Type_of_collision": rng.choice(["Collision with roadside-parked vehicles", "Vehicle with vehicle collision"], n_rows),
            "Number_of_vehicles_involved": rng.integers(1, 5, n_rows),
            "Vehicle_movement": rng.choice(["Going straight", "Overtaking"], n_rows),
            "Pedestrian_movement": rng.choice(["Not a Pedestrian", "Crossing from driver's nearside"], n_rows),
            "Cause_of_accident": rng.choice(
                ["Moving Backward"] * 90 + ["Overtaking"] * 5 + ["rare_cause_a", "rare_cause_b"], n_rows
            ),
            # Casualty-outcome columns — must never be used as features.
            "Number_of_casualties": rng.integers(1, 3, n_rows),
            "Casualty_class": rng.choice(["Driver or rider", "Pedestrian"], n_rows),
            "Sex_of_casualty": rng.choice(["Male", "Female"], n_rows),
            "Age_band_of_casualty": rng.choice(["18-30", "31-50"], n_rows),
            "Casualty_severity": rng.choice(["1", "2", "3"], n_rows),
            "Work_of_casuality": rng.choice(["Driver", None], n_rows),
            "Fitness_of_casuality": rng.choice(["Normal", None], n_rows),
            config.TARGET_COL: rng.choice(
                config.RAW_SEVERITY_CLASSES, n_rows, p=[0.846, 0.142, 0.012]
            ),
        }
    )
    return df


def test_feature_engineer_parses_hour_and_encodes_cyclically():
    panel = make_synthetic_panel(n_rows=50)
    engineered = FeatureEngineer().fit_transform(panel)

    assert "hour" in engineered.columns
    assert engineered["hour"].between(0, 23).all()
    assert engineered["hour_sin"].between(-1, 1).all()
    assert config.TIME_COL not in engineered.columns


def test_feature_engineer_hour_23_and_hour_0_are_close_on_the_circle():
    panel = make_synthetic_panel(n_rows=2)
    panel[config.TIME_COL] = ["23:30:00", "00:30:00"]
    engineered = FeatureEngineer().fit_transform(panel)

    row_23, row_0 = engineered.iloc[0], engineered.iloc[1]
    dist = np.hypot(row_23["hour_sin"] - row_0["hour_sin"], row_23["hour_cos"] - row_0["hour_cos"])
    # Adjacent hours must be much closer than opposite ends of the day.
    row_noon = FeatureEngineer().fit_transform(pd.DataFrame({config.TIME_COL: ["12:00:00"]})).iloc[0]
    dist_far = np.hypot(row_23["hour_sin"] - row_noon["hour_sin"], row_23["hour_cos"] - row_noon["hour_cos"])
    assert dist < dist_far


def test_rare_category_grouper_collapses_low_frequency_categories():
    panel = make_synthetic_panel(n_rows=200)
    grouper = RareCategoryGrouper(columns=["Type_of_vehicle"], min_frequency=0.05).fit(panel)
    transformed = grouper.transform(panel)

    assert (transformed.loc[panel["Type_of_vehicle"] == "Automobile", "Type_of_vehicle"] == "Automobile").all()
    rare_mask = panel["Type_of_vehicle"].isin(["rare_type_a", "rare_type_b"])
    assert (transformed.loc[rare_mask, "Type_of_vehicle"] == "rare").all()


def test_casualty_outcome_columns_are_never_in_raw_feature_cols():
    for col in config.CASUALTY_OUTCOME_COLS:
        assert col not in config.RAW_FEATURE_COLS


def test_full_feature_pipeline_runs_end_to_end():
    panel = make_synthetic_panel(n_rows=300)
    X = panel[config.RAW_FEATURE_COLS]
    y = panel[config.TARGET_COL]

    pipeline = build_feature_pipeline()
    transformed = pipeline.fit_transform(X, y)

    assert transformed.shape[0] == len(X)
