"""Deployment smoke check for the traffic-accident app.

Usage:
    python scripts/smoke_app.py

1. The packaged model loads, predicts and explains one accident.
2. Every results file the app reads exists.
3. Every page renders headlessly without an exception (Streamlit AppTest).
4. `streamlit run` starts and answers its health check.
"""

from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import joblib
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from tests.test_features import make_synthetic_panel  # noqa: E402
from traffic_accident_severity import config  # noqa: E402

PAGES = ["predict", "overview", "feature_engineering", "model_insights"]
RESULT_FILES = ["data_facts.json", "rates_by.csv", "leak_check.csv", "model_comparison.csv",
                "threshold_tradeoff.csv", "feature_importance.csv", "summary.json"]


def check_model() -> None:
    model = joblib.load(config.MODEL_PATH)
    row = make_synthetic_panel(n_rows=4)[config.RAW_FEATURE_COLS].head(1)
    p = model.severe_probability(row)
    if len(p) != 1 or not 0 < float(p[0]) < 1 or model.explain(row).shape[0] != 1:
        raise RuntimeError("The packaged model did not predict and explain one accident.")


def check_results() -> None:
    missing = [f for f in RESULT_FILES if not (PROJECT_ROOT / "results" / f).exists()]
    if missing:
        raise RuntimeError(f"Missing results (run scripts/evaluate.py): {missing}")


def _render_page(app_dir: str, page: str) -> None:
    import importlib
    import sys

    sys.path.insert(0, app_dir)
    importlib.import_module(f"pages_src.{page}").render()


def check_pages() -> None:
    for page in PAGES:
        at = AppTest.from_function(_render_page, args=(str(PROJECT_ROOT / "app"), page), default_timeout=180)
        at.run()
        if at.exception:
            raise RuntimeError(f"Page {page} raised: {at.exception[0].value}")
        print(f"  page {page}: ok")


def check_streamlit() -> None:
    process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py", "--server.headless=true",
         "--server.port=8503"], cwd=PROJECT_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("Streamlit exited before becoming healthy.")
            try:
                with urllib.request.urlopen("http://127.0.0.1:8503/_stcore/health", timeout=2) as response:
                    if response.status == 200:
                        return
            except OSError:
                time.sleep(1)
        raise RuntimeError("Streamlit health endpoint did not become ready within 90 seconds.")
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


if __name__ == "__main__":
    check_model()
    check_results()
    check_pages()
    check_streamlit()
    print("traffic-accident-severity deployment smoke check passed")
