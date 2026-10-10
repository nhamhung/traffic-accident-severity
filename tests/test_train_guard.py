"""train.py must never fit the production model on the bundled sample."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_training_refuses_the_sample(monkeypatch):
    import train
    from tests.test_features import make_synthetic_panel

    monkeypatch.setattr(train.data, "load_accidents", lambda: make_synthetic_panel(n_rows=1200))
    with pytest.raises(RuntimeError, match="full dataset"):
        train.load_full_data()
