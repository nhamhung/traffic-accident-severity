# Traffic Accident Severity

A worked, end-to-end data science project predicting **traffic accident
severity** — specifically, whether an accident is **Severe** (a serious
or fatal injury) or **Not Severe** (a slight injury) — motivated by
traffic congestion and road safety being a major, visible problem in
Vietnam.

## Why this dataset

No genuinely open, granular Vietnamese accident dataset exists — checked
directly: Hanoi's records are paper-based police forms, not open data.
Among real, well-documented public alternatives, this project uses
accident records from **Addis Ababa sub-city police (2017-2020,
12,316 rows)** — chosen specifically for its *contextual* relevance:
a developing-country, mixed-vehicle urban traffic environment
(motorcycles, pedestrians, buses, private cars sharing the same roads,
less-regulated driving conditions), far closer to Vietnam's own traffic
reality than larger, better-known Western datasets (US-Accidents, UK
STATS19), which are purely car-centric. Not a claim that Ethiopian and
Vietnamese traffic are the same — just that the *kind* of problem is
far more comparable.

## What's here

| Deliverable | Where |
|---|---|
| A well-documented notebook building the model, applying good practices | `notebooks/01_eda_and_modeling.ipynb` |
| A multi-page Streamlit app to explore the data and try the model | `app/streamlit_app.py` + `app/pages_src/` (+ `app/Dockerfile`) |
| A research-style writeup, including a discussion of what a congestion-forecasting system would need next | `report/report.qmd` |
| A script that trains and saves the production model | `scripts/train.py` |

All four share one feature-engineering/model source of truth in
`src/traffic_accident_severity/`, so the notebook, the app, and the
script can never quietly drift apart — they all load the same trained
`models/model.joblib`.

## The central data-understanding decision this project is built around

The raw data mixes **pre-accident risk factors** (driver, vehicle, road,
weather, time, collision dynamics — knowable before/at the crash) with
**post-accident casualty-outcome columns** (`Casualty_severity`,
`Casualty_class`, `Number_of_casualties`, ...) recorded *after* the
crash. Using the latter as features would be circular — a system that
needs to already know how badly someone was hurt to predict how badly
someone was hurt isn't useful for real prevention or policy work. See
`config.py`'s module docstring and `data.split_features_target` for
exactly what's excluded and why.

## Why the model's target is binary, not the raw 3-class column

Verified directly: the raw `Accident_severity` column is severely
imbalanced — Slight Injury 84.6%, Serious Injury 14.2%, **Fatal injury
only 1.3%** (~158 of 12,316 rows). A 3-class Random Forest, even with
`class_weight="balanced"`, only caught Fatal injuries about 3% of the
time in cross-validation — there just aren't enough Fatal rows for a
model to reliably tell it apart from *two* other classes at once.

Collapsing Serious + Fatal into one **Severe** class (vs. **Not
Severe**) raised that recall to ~30-35% and macro-F1 from ~0.39 to
~0.60 — measured directly, not assumed. `data.split_features_target`
does this collapse; every model, the saved `models/model.joblib`, and
the app all operate on this binary target, not the raw 3-class column.
See `config.py`'s module docstring and `report/report.qmd`'s "Choosing
the modeling target" section for the full comparison.

Accuracy is still actively misleading here (a model that never
predicts "Severe" would still score ~84.6%); every comparison in this
project uses **macro-F1** instead. Class-weighting vs. ADASYN
oversampling are compared directly rather than one assumed to help —
class-weighting won clearly.

## What this project deliberately doesn't do

This is accident-**severity classification**, not traffic-**congestion
forecasting** — a genuinely different problem (continuous sensor/GPS
time series on a road-network graph, not individual tabular records).
`report/report.qmd`'s Discussion section covers what a real
congestion-forecasting system would need, and specifically what's
missing for a Vietnam-focused version of one (no fixed sensor network,
motorbike-dominated flow dynamics unlike the car-only benchmarks the
standard models are built on).

## Project layout

```
data/raw/               # downloaded dataset CSV (gitignored — see below)
data/processed/          # any cached intermediate data (gitignored)
notebooks/               # the main EDA + modeling notebook
src/traffic_accident_severity/  # shared config, data loading, feature engineering, model code, SHAP interpretability
models/                  # trained pipeline artifact (model.joblib)
app/                     # Streamlit app (multi-page, app/pages_src/) + Dockerfile
scripts/                 # train.py
report/                  # Quarto research writeup
tests/                   # pytest tests for feature engineering and model code (synthetic data — no download needed)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Get the data

This project doesn't commit the dataset. Download it with the
[Kaggle API](https://www.kaggle.com/docs/api) (`pip install kaggle`,
then put your `kaggle.json`/access token in `~/.kaggle/`):

```bash
kaggle datasets download -d samikshakolhe/rta-dataset-addis-ababa-subcity -p data/raw
unzip -o data/raw/rta-dataset-addis-ababa-subcity.zip -d data/raw
```

This produces `data/raw/RTA Dataset.csv`.

## Run the notebook

```bash
jupyter notebook notebooks/01_eda_and_modeling.ipynb
```

## Train from the command line

```bash
python scripts/train.py                              # default: Random Forest (balanced) — the honest winner
python scripts/train.py --model LightGBM              # any model in MODEL_FACTORIES
python scripts/train.py --resample                    # ADASYN instead of class-weighting, for comparison
python scripts/train.py --help
```

## Run the app

A 4-page app: **Predict** (load a real accident record or start from
dataset averages, tweak the fields that matter most, see the predicted
severity), **Dataset Overview**, **Feature Engineering**, and **Model
Insights** (model comparison + class-imbalance comparison + live SHAP).

```bash
streamlit run app/streamlit_app.py
```

Or in Docker:

```bash
docker build -t traffic-accident-severity-app -f app/Dockerfile .
docker run -p 8501:8501 traffic-accident-severity-app
```

Then open http://localhost:8501.

## Render the research writeup

```bash
quarto render report/report.qmd
```

## Run the tests

```bash
pytest tests/
```

These test the feature engineering and model logic directly with
synthetic data — no download needed, and they already pass without any
real data.

## Deploy

The Streamlit app fetches its source data through the Kaggle API at runtime. Configure either KAGGLE_API_TOKEN or a [kaggle] secrets section containing username and key.

- Repository: <https://github.com/nhamhhung/traffic-accident-severity>
- Report: <https://nhamhung.github.io/traffic-accident-severity/>
- Streamlit: <https://traffic-accident-severity.streamlit.app>
- Fork setup: [docs/SETUP_AND_DEPLOYMENT.md](docs/SETUP_AND_DEPLOYMENT.md)
