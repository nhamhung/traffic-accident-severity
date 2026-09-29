"""Tests for the model pipeline and evaluation helpers, using a small
synthetic panel instead of the real Addis Ababa RTA data.
"""

import numpy as np
import pandas as pd

from traffic_accident_severity import config, data, model
from tests.test_features import make_synthetic_panel


def test_build_pipeline_predicts_proba_shape():
    panel = make_synthetic_panel(n_rows=300)
    X, y = data.split_features_target(panel)

    pipeline = model.build_pipeline(model.logistic_estimator())
    pipeline.fit(X, y)
    proba = pipeline.predict_proba(X)

    assert proba.shape == (len(X), len(config.TARGET_CLASSES))


def test_cross_validate_pipeline_returns_valid_f1():
    panel = make_synthetic_panel(n_rows=300)
    X, y = data.split_features_target(panel)

    result = model.cross_validate_pipeline(X, y, estimator=model.logistic_estimator(), cv=3)
    assert 0.0 <= result["f1_macro_mean"] <= 1.0


def test_resampled_pipeline_runs_end_to_end():
    # ADASYN needs enough minority-class ("Severe") rows for its
    # neighborhood computation — n_rows=1000 reliably yields enough
    # Severe rows (~15.4% rate) even with the default n_neighbors=5.
    panel = make_synthetic_panel(n_rows=1000)
    X, y = data.split_features_target(panel)

    pipeline = model.build_pipeline(model.logistic_estimator(), resample=True)
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    assert len(preds) == len(X)


def test_split_features_target_collapses_to_binary_severe():
    panel = make_synthetic_panel(n_rows=300)
    X, y = data.split_features_target(panel)

    assert set(y.unique()) <= set(config.TARGET_CLASSES)
    expected_severe = panel[config.TARGET_COL].isin(config.SEVERE_CLASSES)
    assert list(y == "Severe") == list(expected_severe)


def test_cross_val_severe_probabilities_returns_one_score_per_row():
    panel = make_synthetic_panel(n_rows=1000)
    X, y = data.split_features_target(panel)

    proba = model.cross_val_severe_probabilities(X, y, estimator=model.logistic_estimator(), cv=3)

    assert proba.shape == (len(X),)
    assert ((proba >= 0.0) & (proba <= 1.0)).all()


def test_severity_threshold_sweep_recall_increases_as_threshold_drops():
    rng = np.random.default_rng(0)
    y = pd.Series(["Not Severe"] * 90 + ["Severe"] * 10)
    proba = np.concatenate([rng.uniform(0, 0.6, 90), rng.uniform(0.4, 1.0, 10)])

    sweep = model.severity_threshold_sweep(y, proba, target_recalls=[0.3, 0.9])

    low_recall_row = sweep.iloc[0]
    high_recall_row = sweep.iloc[1]
    assert low_recall_row["threshold"] >= high_recall_row["threshold"]


def test_save_and_load_pipeline_roundtrip(tmp_path):
    panel = make_synthetic_panel(n_rows=300)
    X, y = data.split_features_target(panel)
    pipeline = model.train_pipeline(X, y, estimator=model.logistic_estimator())

    path = tmp_path / "model.joblib"
    model.save_pipeline(pipeline, path=path)
    loaded = model.load_pipeline(path=path)

    np.testing.assert_array_equal(pipeline.predict(X), loaded.predict(X))
