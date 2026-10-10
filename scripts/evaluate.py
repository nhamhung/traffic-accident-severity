"""Write every number the app and the report show to results/ (after train.py).

Usage:
    python scripts/evaluate.py            # ~30 min (CatBoost, 3 x 5 folds + inner threshold search)
    python scripts/evaluate.py --quick    # 1 repeat, for a fast check

Outputs (all from the full 12,316-row dataset):
  data_facts.json        sizes, class shares, missingness, the same-accident check, each column's values
  rates_by.csv           severe-accident rate by value of the main columns
  leak_check.csv         the old model scored with random vs. grouped-by-accident folds
  model_comparison.csv   every candidate model under the same grouped, repeated cross-validation
  threshold_tradeoff.csv recall and precision of the final model at each cut-off
  feature_importance.csv mean |SHAP| per column for the final model
  summary.json           the final model's threshold and cross-validated scores
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train import load_full_data  # noqa: E402
from traffic_accident_severity import config, model  # noqa: E402
from traffic_accident_severity.severity_model import (  # noqa: E402
    SeverityModel, accident_groups, cross_validate,
)

RESULTS = config.PROJECT_ROOT / "results"
RATE_COLUMNS = ["Light_conditions", "Number_of_vehicles_involved", "Age_band_of_driver", "Type_of_collision",
                "Types_of_Junction", "Weather_conditions", "Cause_of_accident", "Day_of_week", "Driving_experience",
                "Type_of_vehicle", "Road_surface_conditions"]


def data_facts(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, groups: np.ndarray) -> dict:
    severe = (y == "Severe").to_numpy()
    sizes = pd.Series(groups).map(pd.Series(groups).value_counts()).to_numpy()
    multi = sizes > 1
    same = pd.DataFrame({"g": groups[multi], "s": severe[multi]}).groupby("g")["s"].nunique().eq(1).mean()
    p = severe.mean()
    values = {}
    for col in config.RAW_FEATURE_COLS:
        if col in (config.TIME_COL, "Number_of_vehicles_involved"):
            continue
        v = sorted(X[col].dropna().astype(str).unique())
        values[col] = v + ([None] if X[col].isna().any() else [])
    return {
        "n_rows": int(len(df)), "n_accidents": int(len(np.unique(groups))), "n_features": len(config.RAW_FEATURE_COLS),
        "rows_in_multi_row_accidents": int(multi.sum()), "multi_row_same_severity": float(same),
        "multi_row_same_severity_by_chance": float(p**2 + (1 - p) ** 2),
        "severe_share": float(p),
        "raw_shares": {k: float(v) for k, v in df[config.TARGET_COL].value_counts(normalize=True)
                       .reindex(config.RAW_SEVERITY_CLASSES).items()},
        "missing_share": {c: float(X[c].isna().mean()) for c in config.RAW_FEATURE_COLS},
        "excluded_columns": config.CASUALTY_OUTCOME_COLS, "values": values,
    }


def rates_by(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    severe = (y == "Severe").astype(int)
    frame = X.assign(hour=pd.to_datetime(X[config.TIME_COL], format="%H:%M:%S", errors="coerce").dt.hour)
    rows = []
    for col in RATE_COLUMNS + ["hour"]:
        g = severe.groupby(frame[col]).agg(["mean", "size"])
        rows += [{"column": col, "value": v, "severe_rate": r["mean"], "n": int(r["size"])} for v, r in g.iterrows()]
    return pd.DataFrame(rows)


def pipeline_model(estimator_factory):
    return lambda: model.build_pipeline(estimator_factory())


CANDIDATES = {
    # key: (Vietnamese name, factory)
    "logistic": ("Hồi quy logistic (đơn giản nhất)", pipeline_model(model.logistic_estimator)),
    "old_random_forest": ("Rừng ngẫu nhiên (mô hình cũ)", pipeline_model(model.balanced_random_forest_estimator)),
    "tuned_random_forest": ("Rừng ngẫu nhiên đã tinh chỉnh", pipeline_model(lambda: RandomForestClassifier(
        n_estimators=500, min_samples_leaf=2, max_features=0.3, class_weight="balanced_subsample",
        n_jobs=-1, random_state=config.RANDOM_SEED))),
    "catboost": ("CatBoost", lambda: SeverityModel(forest=False)),
    "blend": ("CatBoost + rừng ngẫu nhiên (mô hình mới)", SeverityModel),
}
FINAL = "blend"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="1 repeat instead of 3")
    args = parser.parse_args()
    repeats = 1 if args.quick else 3
    RESULTS.mkdir(exist_ok=True)

    df, X, y = load_full_data()
    groups = accident_groups(df)
    (RESULTS / "data_facts.json").write_text(json.dumps(data_facts(df, X, y, groups), ensure_ascii=False, indent=1),
                                             encoding="utf-8")
    rates_by(X, y).to_csv(RESULTS / "rates_by.csv", index=False)

    print("Leak check (old model, random vs grouped folds) ...", flush=True)
    old = CANDIDATES["old_random_forest"][1]
    leak = []
    for name, g in [("random", np.arange(len(y))), ("grouped", groups)]:
        result = cross_validate(X, y, g, old, repeats=repeats, log=lambda _: None)
        leak.append({"folds": name, **result.summary()})
    pd.DataFrame(leak).to_csv(RESULTS / "leak_check.csv", index=False)

    rows, final_oof = [], None
    for key, (name_vi, factory) in CANDIDATES.items():
        print(f"{name_vi} ...", flush=True)
        result = cross_validate(X, y, groups, factory, repeats=repeats, log=lambda _: None)
        s = result.summary()
        rows.append({"model": key, "name_vi": name_vi, "is_final": key == FINAL, **s})
        print(f"  macro-F1 {s['f1_tuned']:.4f} ± {s['f1_tuned_sd']:.3f}, PR-AUC {s['pr_auc']:.4f}, "
              f"ROC-AUC {s['roc_auc']:.4f}", flush=True)
        if key == FINAL:
            final_oof = result.oof
    comparison = pd.DataFrame(rows)
    comparison.to_csv(RESULTS / "model_comparison.csv", index=False)

    final = joblib.load(config.MODEL_PATH)
    y_bin = (y == "Severe").to_numpy().astype(int)
    tradeoff = []
    for t in np.round(np.arange(0.05, 0.61, 0.01), 2):
        flagged = final_oof >= t
        tp = int((flagged & (y_bin == 1)).sum())
        tradeoff.append({"threshold": t, "recall": tp / y_bin.sum(), "precision": tp / max(1, flagged.sum()),
                         "flagged_share": float(flagged.mean())})
    pd.DataFrame(tradeoff).to_csv(RESULTS / "threshold_tradeoff.csv", index=False)

    shap = final.explain(X).abs().mean()
    pd.DataFrame({"feature": shap.index, "importance": shap.values, "share": shap.values / shap.sum()}) \
        .sort_values("importance", ascending=False).to_csv(RESULTS / "feature_importance.csv", index=False)

    best = comparison.set_index("model").loc[FINAL]
    flagged = final_oof >= final.threshold
    summary = {"threshold": final.threshold, "f1_macro": best["f1_tuned"], "pr_auc": best["pr_auc"],
               "roc_auc": best["roc_auc"], "recall_severe": best["recall_severe"],
               "precision_severe": best["precision_severe"],
               "accuracy": float((flagged == (y_bin == 1)).mean()), "repeats": repeats}
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
