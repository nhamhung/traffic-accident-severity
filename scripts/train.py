"""Train the production model on the full dataset and save it.

Usage:
    python scripts/train.py            # ~2 min

The model is `severity_model.SeverityModel` (CatBoost on the raw categories).
Its decision threshold is the one that maximises macro-F1 on out-of-fold
predictions from 5-fold cross-validation grouped by accident, so it is
chosen without ever scoring the rows it was fitted on. Then
`scripts/evaluate.py` writes the results the app and the report show.
"""

import os
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from traffic_accident_severity import config, data  # noqa: E402
from traffic_accident_severity.severity_model import SeverityModel, accident_groups, best_threshold  # noqa: E402

FULL_ROWS = 12_316


def load_full_data():
    """The full dataset - never the bundled 1,200-row app sample."""
    os.environ["USE_FULL_KAGGLE_DATA"] = "true"
    df = data.load_accidents()
    if len(df) != FULL_ROWS:
        raise RuntimeError(f"Expected the full dataset ({FULL_ROWS:,} rows), got {len(df):,}. "
                           "Download it first - see the README's 'Get the data' section.")
    X, y = data.split_features_target(df)
    return df, X, y


def main() -> None:
    df, X, y = load_full_data()
    groups = accident_groups(df)
    y_bin = (y == "Severe").to_numpy().astype(int)

    oof = np.zeros(len(y))
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
    for k, (tr, te) in enumerate(folds.split(X, y_bin, groups)):
        oof[te] = SeverityModel().fit(X.iloc[tr], y.iloc[tr]).severe_probability(X.iloc[te])
        print(f"  threshold search: fold {k + 1}/5", flush=True)
    threshold = best_threshold(y_bin, oof)

    final = SeverityModel(threshold=threshold).fit(X, y)
    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, config.MODEL_PATH, compress=3)
    np.save(config.DATA_PROCESSED_DIR / "train_oof.npy", oof) if config.DATA_PROCESSED_DIR.exists() else None
    print(f"Trained on {len(X):,} rows; threshold {threshold:.3f}. Saved to {config.MODEL_PATH} "
          f"({config.MODEL_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
