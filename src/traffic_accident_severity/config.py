"""Paths, constants, and column schema for the Addis Ababa Road Traffic
Accident (RTA) dataset.

Dataset: Road Traffic Accident severity records from Addis Ababa
sub-city police, 2017-2020 (mirrored on Kaggle, originally published on
Mendeley Data), 12,316 rows, 32 columns. Chosen specifically for its
*contextual* relevance to Vietnam over larger, better-known Western
datasets (US-Accidents, UK STATS19): a developing-country, mixed-vehicle
(motorcycles, pedestrians, buses, private cars) urban traffic
environment, police-recorded like Vietnam's own accident data would be
— not a claim that Ethiopia and Vietnam's traffic are the same, just
that the *kind* of problem (mixed-vehicle, less-regulated, developing
urban road network) is far closer than a US freeway dataset's.

Verified directly against the real downloaded CSV (not assumed):
`Accident_severity` is heavily imbalanced (Slight Injury 84.6%, Serious
Injury 14.2%, Fatal injury only 1.3%). Several columns are missing
20-36% of values (`Defect_of_vehicle`, `Service_year_of_vehicle`,
`Work_of_casuality`, `Fitness_of_casuality`).

The model's actual target is binary, not the raw 3-class column: a
3-class Random Forest (even with `class_weight="balanced"`) only caught
Fatal injuries about 3% of the time in cross-validation — there just
aren't enough Fatal rows (~158) for a model to reliably tell it apart
from *two* other classes at once. Collapsing Serious+Fatal into one
"Severe" class raised that class's cross-validated recall to ~35% and
macro-F1 from ~0.39 to ~0.60 (measured directly, not assumed) — a much
more useful question for a real road-safety system to answer anyway:
"is this likely to be a serious/fatal accident" is closer to an
actionable prevention signal than "is this specifically fatal."
"""

from pathlib import Path

# --- Paths -------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

RAW_CSV = DATA_RAW_DIR / "RTA Dataset.csv"
SAMPLE_CSV = DATA_SAMPLE_DIR / "accidents_sample.csv.gz"
MODEL_PATH = MODELS_DIR / "model.joblib"

# --- Kaggle source ---------------------------------------------------------
# A dataset mirror, not a competition — several identical re-uploads exist
# on Kaggle of the same underlying Mendeley Data release.

KAGGLE_DATASET = "samikshakolhe/rta-dataset-addis-ababa-subcity"

# --- Target --------------------------------------------------------------

TARGET_COL = "Accident_severity"

# The raw CSV's 3-class severity labels, as they appear in the data
# (and in the synthetic test panel, which mirrors the real schema).
RAW_SEVERITY_CLASSES = ["Slight Injury", "Serious Injury", "Fatal injury"]

# Which raw severities count as "Severe" — the model's actual target
# collapses to this vs. everything else. See the module docstring for
# why (3-class Fatal-injury detection was too weak to be useful).
SEVERE_CLASSES = ["Serious Injury", "Fatal injury"]

# The model's real output classes, after `data.split_features_target`
# collapses `Accident_severity` to binary.
TARGET_CLASSES = ["Not Severe", "Severe"]

RANDOM_SEED = 42

# --- Column schema -----------------------------------------------------
# THE CENTRAL DESIGN DECISION OF THIS PROJECT: the raw data mixes
# pre-accident risk factors (who was driving, what vehicle, what road,
# what weather — all knowable *before* a crash, the only kind of feature
# a real prevention/policy system could act on) with post-accident
# casualty-outcome columns (who got hurt, how badly, what they were
# doing — recorded *after* the crash was already happening). Using the
# latter as model features would be circular: `Casualty_severity` in
# particular is close to a restatement of the target, verified directly
# via a crosstab against `Accident_severity` (not a clean 1:1 mapping,
# but clearly the same underlying event, recorded twice). Excluded below,
# the same reasoning the Spotify project uses to exclude `popularity`/
# `track_genre` from its recommender's own feature space.

TIME_COL = "Time"

DRIVER_COLS = [
    "Age_band_of_driver",
    "Sex_of_driver",
    "Educational_level",
    "Vehicle_driver_relation",
    "Driving_experience",
]
VEHICLE_COLS = [
    "Type_of_vehicle",
    "Owner_of_vehicle",
    "Service_year_of_vehicle",
    "Defect_of_vehicle",
]
ROAD_COLS = [
    "Area_accident_occured",
    "Lanes_or_Medians",
    "Road_allignment",
    "Types_of_Junction",
    "Road_surface_type",
    "Road_surface_conditions",
]
ENVIRONMENT_COLS = ["Light_conditions", "Weather_conditions"]
COLLISION_DYNAMICS_COLS = [
    "Type_of_collision",
    "Number_of_vehicles_involved",
    "Vehicle_movement",
    "Pedestrian_movement",
    "Cause_of_accident",
]

# Post-accident casualty-outcome columns — never used as model features.
# Number_of_casualties is included here (not a "risk factor" — it's a
# direct component of how severity itself gets recorded).
CASUALTY_OUTCOME_COLS = [
    "Number_of_casualties",
    "Casualty_class",
    "Sex_of_casualty",
    "Age_band_of_casualty",
    "Casualty_severity",
    "Work_of_casuality",
    "Fitness_of_casuality",
]

RAW_FEATURE_COLS = (
    [TIME_COL, "Day_of_week"]
    + DRIVER_COLS
    + VEHICLE_COLS
    + ROAD_COLS
    + ENVIRONMENT_COLS
    + COLLISION_DYNAMICS_COLS
)

NUMERIC_COLS = ["Number_of_vehicles_involved", "hour", "hour_sin", "hour_cos"]
CATEGORICAL_COLS = [
    c for c in RAW_FEATURE_COLS if c not in NUMERIC_COLS and c != TIME_COL
]

# High-cardinality columns needing RareCategoryGrouper before one-hot
# encoding (same technique as academic_success): Cause_of_accident (20
# distinct values), Type_of_vehicle (17), Area_accident_occured (14).
RARE_GROUPED_COLS = ["Cause_of_accident", "Type_of_vehicle", "Area_accident_occured"]
