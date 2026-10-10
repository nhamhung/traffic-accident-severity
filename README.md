# Traffic Accident Severity

A worked, end-to-end data science project predicting **traffic accident
severity** — specifically, whether an accident is **Severe** (a serious
or fatal injury) or **Not Severe** (a slight injury) — motivated by
traffic congestion and road safety being a major, visible problem in
Vietnam.

**The app and the report are in Vietnamese**, written for secondary-school
students: no code in the report, every technical term explained with an
everyday example, and numbers shown as "out of 1,000 accidents". The code,
docstrings and this README stay in English for developers. Display labels
live in one place, `src/traffic_accident_severity/vi.py`; the data and the
model keep the dataset's original English values.

## Results

5-fold cross-validation **grouped by accident** (see below), repeated 3
times, decision threshold chosen inside the training folds only
(`scripts/evaluate.py`, full 12,316-row dataset):

| Model | Macro-F1 | PR-AUC | ROC-AUC |
|---|---|---|---|
| Logistic regression (one-hot) | 0.556 | 0.230 | 0.595 |
| Random Forest, `class_weight="balanced"` (previous model) | 0.586 | 0.275 | 0.650 |
| Random Forest, tuned (deeper trees, 30% of features per split) | 0.606 | 0.309 | 0.687 |
| CatBoost (native categoricals) | 0.608 | 0.314 | 0.686 |
| **CatBoost + tuned Random Forest, averaged (production)** | **0.611** | **0.323** | **0.698** |

PR-AUC baseline (random guessing) is 0.154. At the chosen threshold (0.24)
the production model catches 37% of severe accidents, and 33% of its alerts
are right, about 2.2x random.

**Many accidents appear as several rows.** One row is written per vehicle:
rows sharing time, day, area, light, weather, road surface, collision type,
vehicle and casualty counts are the same accident (3,003 rows; severity
agrees within them 94% of the time vs. 74% by chance). Random K-fold puts
parts of one accident on both sides of the split; grouping them
(`severity_model.accident_groups`) lowers the previous model's PR-AUC from
0.289 to 0.275. Every number above uses grouped folds.

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

## Prerequisites

Install once, before Setup below:

| Dependency | Why | Install |
|---|---|---|
| **Python 3.12** | This project's `.venv` is built against 3.12 — a different version may resolve incompatible package versions from `requirements.txt`. | [python.org/downloads](https://www.python.org/downloads/) or a version manager (e.g. `pyenv install 3.12`) |
| **Quarto** | Renders `report/report.qmd` — a standalone binary, not a Python package, so `pip install` never gets it. | [quarto.org/docs/get-started](https://quarto.org/docs/get-started/) |
| **Kaggle API token** | Needed only for the complete dataset; the default Streamlit app uses a bundled sample, and `pytest` uses synthetic data. | Kaggle account → **Account → Create New API Token** → save the downloaded file as `~/.kaggle/kaggle.json` (`%USERPROFILE%\.kaggle\kaggle.json` on Windows). See the [Kaggle API docs](https://www.kaggle.com/docs/api). |
| **Docker** (optional) | Only if you want to run the app in its pre-baked container instead of `streamlit run`. | [docker.com/get-started](https://www.docker.com/get-started/) |

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

## Making changes

`src/traffic_accident_severity/` is the single source of truth —
`config.py` (paths, schema, the Severe/Not-Severe target collapse),
`data.py` (loading/splitting), `features.py` (feature engineering),
`model.py` (pipelines, training, evaluation), `interpretability.py`
(SHAP). The notebook, the app, and `scripts/train.py` all import from
here; nothing re-derives logic locally, so a change here propagates
everywhere automatically.

The edit loop:

```bash
# 1. Edit src/traffic_accident_severity/*.py

# 2. Check it against the test suite (fast, synthetic data, no download needed)
PYTHONPATH=src:. pytest tests/

# 3. Retrain, so models/model.joblib reflects your change (~3 min, full dataset required)
python scripts/train.py

# 4. Recompute every number the app and the report show (~45 min; --quick for 1 repeat)
python scripts/evaluate.py
```

`models/model.joblib` is the production model the app loads;
`results/` holds every figure the app and the report show (the app never
re-runs cross-validation itself). `train.py` refuses to run on the bundled
1,200-row sample.

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
A real congestion-forecasting system would need different data and
models: road-segment speed time series (GPS probes, since Vietnam's urban
roads lack fixed sensors), graph-based spatio-temporal models, and
validation on motorbike-dominated traffic rather than car-only freeway
benchmarks. (The Vietnamese report, written for secondary-school students,
leaves this discussion out.)

## Project layout

```
data/raw/               # downloaded dataset CSV (gitignored — see below)
data/processed/          # any cached intermediate data (gitignored)
notebooks/               # the main EDA + modeling notebook
src/traffic_accident_severity/  # shared config, data loading, feature engineering, model code, SHAP interpretability
models/                  # production model (model.joblib, ~21 MB)
results/                 # every number the app and the report show (scripts/evaluate.py)
app/                     # Streamlit app (multi-page, app/pages_src/) + Dockerfile
scripts/                 # train.py, evaluate.py, smoke_app.py
report/                  # Quarto research writeup
tests/                   # pytest tests for feature engineering and model code (synthetic data — no download needed)
```

## Setup

```bash
python3.12 -m venv .venv          # use the 3.12 interpreter specifically
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The notebook and `report/report.qmd` both pin a named Jupyter kernel
(`traffic-accident-severity`), so register one from this venv before
running either of them:

```bash
python -m ipykernel install --user --name traffic-accident-severity \
    --display-name "traffic-accident-severity"
```

## Get the data

The complete dataset is not committed. A bundled 1,200-row stratified sample
keeps the Streamlit app responsive. Download the full dataset with the
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

## Train and evaluate from the command line

```bash
python scripts/train.py      # production model: CatBoost + tuned Random Forest, threshold from grouped out-of-fold predictions
python scripts/evaluate.py   # model comparison, leak check, threshold trade-off, SHAP importance, data facts -> results/
```

The original sklearn pipelines (`model.py`: Logistic Regression, Random
Forest, LightGBM, XGBoost, ADASYN) are still used by the notebook and as
comparison models in `evaluate.py`.

## Run the app

A 4-page app in Vietnamese: **Dự đoán** (load a real accident or start
from a typical one, change the conditions, see the probability, the alert
and which factors pushed it up or down), **Dữ liệu nói gì?** (findings
straight from the data), **Chuẩn bị dữ liệu** (no peeking at outcomes, one
accident = many rows, missing values) and **Mô hình giỏi đến đâu?** (model
comparison, a threshold slider showing caught / missed / false alarms per
1,000 accidents, SHAP importance).

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

This regenerates both `report/report.html` and `report/report.pdf` (PDF
needs a LaTeX distribution — if you don't have one, run
`quarto install tinytex` once). Render just one format when you don't need
both:

```bash
quarto render report/report.qmd --to html
quarto render report/report.qmd --to pdf
```

Live-preview while editing (auto-rerenders on save):

```bash
quarto preview report/report.qmd
```

A `.qmd` file is Markdown prose plus fenced Python code chunks
(` ```{python} `/` ``` `), executed top to bottom by the kernel registered
above, same as a notebook cell. Common per-chunk options (a `#|` comment,
first line of the chunk): `#| echo: false` (hide this chunk's source
code), `#| output: false` (suppress its output, e.g. a setup/import cell),
`#| label: fig-foo` + `#| fig-cap: "..."` (name and caption a figure for
cross-referencing). The [Quarto VS Code
extension](https://marketplace.visualstudio.com/items?itemName=quarto.quarto)
adds syntax highlighting and a one-click Render button if you're doing more
than a one-line edit.

Troubleshooting:

| Symptom | Likely cause |
|---|---|
| `Jupyter engine failed ... kernel not found` | The `ipykernel install --name traffic-accident-severity` step above (under Setup) hasn't been run yet. |
| `ModuleNotFoundError` inside a code chunk | `quarto render` runs with its working directory set to `report/`, not the project root — check the chunk's `sys.path.insert(0, "../src")` points at the right relative path. |
| Output looks stale after editing | Force a clean re-run: `quarto render report/report.qmd --execute-daemon-restart`. |
| PDF render fails, HTML succeeds | Missing LaTeX — run `quarto install tinytex` once, then retry. |

## Run the tests

```bash
PYTHONPATH=src:. pytest tests/
```

These test the feature engineering and model logic directly with
synthetic data — no download needed, and they already pass without any
real data.

## Deploy

The Streamlit app starts immediately from a bundled 1,200-row stratified sample sourced from Kaggle. Set `USE_FULL_KAGGLE_DATA=true` to fetch and use the complete dataset through the Kaggle API; configure either `KAGGLE_API_TOKEN` or a `[kaggle]` secrets section containing username and key.

- Repository: <https://github.com/nhamhhung/traffic-accident-severity>
- Report: <https://nhamhung.github.io/traffic-accident-severity/>
- Streamlit: <https://traffic-accident-severity.streamlit.app>
- Fork setup: [docs/SETUP_AND_DEPLOYMENT.md](docs/SETUP_AND_DEPLOYMENT.md)
