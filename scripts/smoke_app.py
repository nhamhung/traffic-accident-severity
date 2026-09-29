"""Asset-aware deployment smoke check for the traffic Streamlit app."""

from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from tests.test_features import make_synthetic_panel  # noqa: E402
from traffic_accident_severity import config, model  # noqa: E402


def check_model() -> None:
    pipeline = model.load_pipeline()
    panel = make_synthetic_panel(n_rows=4)
    prediction = pipeline.predict(panel[config.RAW_FEATURE_COLS].head(1))
    if len(prediction) != 1:
        raise RuntimeError("The packaged model did not return one prediction.")


def check_streamlit() -> None:
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app/streamlit_app.py",
        "--server.headless=true",
        "--server.port=8501",
    ]
    process = subprocess.Popen(command, cwd=PROJECT_ROOT)
    try:
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("Streamlit exited before becoming healthy.")
            try:
                with urllib.request.urlopen("http://127.0.0.1:8501/_stcore/health", timeout=2) as response:
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
    check_streamlit()
    print("traffic deployment smoke check passed")
