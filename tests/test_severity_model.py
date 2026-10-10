"""Tests for the production CatBoost model, the accident grouping and the
Vietnamese labels, on a small synthetic panel (never the real data)."""

import numpy as np
import pandas as pd
import pytest

from traffic_accident_severity import config, data, vi
from traffic_accident_severity.severity_model import (
    MISSING, MODEL_FEATURES, SeverityModel, accident_groups, best_threshold, to_model_frame,
)
from tests.test_features import make_synthetic_panel


@pytest.fixture(scope="module")
def fitted():
    panel = make_synthetic_panel(n_rows=400)
    X, y = data.split_features_target(panel)
    return SeverityModel(threshold=0.3, params={"iterations": 30, "depth": 4}).fit(X, y), X


def test_model_frame_keeps_missing_as_its_own_category():
    panel = make_synthetic_panel(n_rows=50)
    frame = to_model_frame(panel[config.RAW_FEATURE_COLS])
    assert list(frame.columns) == MODEL_FEATURES
    assert (frame["Educational_level"] == MISSING).any() and frame["hour"].between(0, 24).all()


def test_probabilities_threshold_and_classes(fitted):
    model, X = fitted
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2) and np.allclose(proba.sum(axis=1), 1)
    assert list(model.classes_) == config.TARGET_CLASSES
    expected = np.where(proba[:, 1] >= 0.3, "Severe", "Not Severe")
    assert (model.predict(X) == expected).all()


def test_explanations_cover_every_raw_column(fitted):
    model, X = fitted
    shap = model.explain(X.head(5))
    assert shap.shape == (5, len(MODEL_FEATURES)) and np.isfinite(shap.to_numpy()).all()


def test_rows_of_one_accident_share_a_group():
    panel = make_synthetic_panel(n_rows=20).assign(Number_of_casualties=1)
    twin = pd.concat([panel.iloc[[0]], panel.iloc[[0]].assign(Driving_experience="5-10yr")], ignore_index=True)
    groups = accident_groups(pd.concat([twin, panel.iloc[1:]], ignore_index=True))
    assert groups[0] == groups[1]                      # same accident, a different vehicle's driver
    assert groups[0] not in groups[2:] or (panel.iloc[0][["Time", "Area_accident_occured"]].tolist()
                                            in panel.iloc[1:][["Time", "Area_accident_occured"]].values.tolist())


def test_best_threshold_finds_the_separating_cut():
    y = np.array([0] * 80 + [1] * 20)
    p = np.concatenate([np.full(80, 0.1), np.full(20, 0.4)])
    assert 0.1 < best_threshold(y, p) <= 0.4


def test_every_label_shown_in_the_app_is_vietnamese():
    for col, values in vi.VALUE.items():
        assert col in vi.COLUMN
        assert all(v != k for k, v in values.items() if k not in ("Taxi",))
    assert vi.cls("Severe") == "Nghiêm trọng" and vi.val("Light_conditions", None) == "Không ghi nhận"
