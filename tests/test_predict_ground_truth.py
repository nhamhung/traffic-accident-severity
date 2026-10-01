"""Ground truth is valid only for an unchanged historical input."""

import numpy as np

from pages_src.predict import _matches_loaded_accident
from traffic_accident_severity import config


def _row():
    return {col: np.nan for col in config.RAW_FEATURE_COLS}


def test_accident_ground_truth_is_invalidated_by_edit():
    loaded = _row()
    loaded["Light_conditions"] = "Daylight"

    assert _matches_loaded_accident(loaded.copy(), loaded)
    edited = loaded.copy()
    edited["Light_conditions"] = "Darkness - no lighting"
    assert not _matches_loaded_accident(edited, loaded)
    assert not _matches_loaded_accident(loaded, {})
