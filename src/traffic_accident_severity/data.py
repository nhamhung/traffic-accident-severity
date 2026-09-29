"""Data loading helpers.

The raw CSV is not committed to the repo (best fetched fresh rather than
duplicated here). Download it first — see the project README.
"""

import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def download_accidents(api_token: str | None = None) -> Path:
    """Download the configured Kaggle dataset and return the validated CSV path."""
    token = api_token or os.getenv("KAGGLE_API_TOKEN")
    if not token:
        raise RuntimeError(
            "KAGGLE_API_TOKEN is required after accepting the dataset terms on Kaggle."
        )

    config.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    prior_token = os.environ.get("KAGGLE_API_TOKEN")
    os.environ["KAGGLE_API_TOKEN"] = token
    try:
        import kagglehub

        downloaded = Path(
            kagglehub.dataset_download(
                config.KAGGLE_DATASET,
                path=config.RAW_CSV.name,
                output_dir=str(config.DATA_RAW_DIR),
            )
        )
    except Exception as exc:
        raise RuntimeError("Kaggle dataset download failed; verify the token and dataset access.") from exc
    finally:
        if prior_token is None:
            os.environ.pop("KAGGLE_API_TOKEN", None)
        else:
            os.environ["KAGGLE_API_TOKEN"] = prior_token

    if not config.RAW_CSV.exists():
        candidates = [downloaded, downloaded / config.RAW_CSV.name, *config.DATA_RAW_DIR.glob("*.csv")]
        source = next((candidate for candidate in candidates if candidate.is_file()), None)
        if source is not None and source != config.RAW_CSV:
            shutil.move(str(source), config.RAW_CSV)
    return _require_file(config.RAW_CSV)


def _require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the dataset first — see the "
            "README's 'Get the data' section, e.g.:\n"
            f"  kaggle datasets download -d {config.KAGGLE_DATASET} -p {config.DATA_RAW_DIR}\n"
            f"  unzip -o {config.DATA_RAW_DIR / (config.KAGGLE_DATASET.split('/')[-1] + '.zip')} "
            f"-d {config.DATA_RAW_DIR}"
        )
    return path


def load_accidents() -> pd.DataFrame:
    """Load the full accident-record table."""
    return pd.read_csv(_require_file(config.RAW_CSV))


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split into (X, y), restricted to pre-accident risk-factor columns
    only — see `config.py`'s module docstring for why the casualty-
    outcome columns are never included here.

    `y` is collapsed from the raw 3-class `Accident_severity` column to
    binary (`config.TARGET_CLASSES`): "Severe" for Serious/Fatal injury,
    "Not Severe" otherwise — see `config.py`'s module docstring for why.
    """
    missing = set(config.RAW_FEATURE_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Input frame is missing expected columns: {sorted(missing)}")
    X = df[config.RAW_FEATURE_COLS].copy()
    is_severe = df[config.TARGET_COL].isin(config.SEVERE_CLASSES)
    y = pd.Series(
        np.where(is_severe, "Severe", "Not Severe"), index=df.index, name=config.TARGET_COL
    )
    return X, y
