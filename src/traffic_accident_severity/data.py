"""Data loading helpers.

The raw CSV is not committed to the repo (best fetched fresh rather than
duplicated here). Download it first — see the project README — or let
`_require_file` fetch it automatically via the Kaggle API (used when
deploying without a Docker image that already bakes the file in; see
`app/pages_src/shared.py` for how deployed credentials get wired in).
"""

from pathlib import Path
from io import BytesIO
import gzip
import os
import zipfile

import numpy as np
import pandas as pd

from . import config


def _download_from_kaggle() -> bool:
    """Best-effort automatic fetch via the Kaggle API. Returns whether
    the target file exists afterward. Silently does nothing (returns
    False) if the `kaggle` package isn't installed or no credentials
    are configured — callers fall back to the manual-download error
    message either way, so this never needs to be trusted to succeed.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        config.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
        api.dataset_download_files(config.KAGGLE_DATASET, path=str(config.DATA_RAW_DIR), unzip=True, quiet=True)
    except Exception:
        return False
    return config.RAW_CSV.exists()


def _require_file(path: Path) -> Path:
    if not path.exists():
        _download_from_kaggle()
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found, and automatic download via the Kaggle API "
            "didn't produce it either (no credentials configured, or the "
            "`kaggle` package isn't installed). Download the dataset "
            "manually instead — see the README's 'Get the data' section, e.g.:\n"
            f"  kaggle datasets download -d {config.KAGGLE_DATASET} -p {config.DATA_RAW_DIR}\n"
            f"  unzip -o {config.DATA_RAW_DIR / (config.KAGGLE_DATASET.split('/')[-1] + '.zip')} "
            f"-d {config.DATA_RAW_DIR}"
        )
    return path


def using_sample_data() -> bool:
    """Whether the bundled Kaggle-derived sample is the active data source."""
    return config.SAMPLE_CSV.exists() and os.getenv("USE_FULL_KAGGLE_DATA", "").lower() not in {"1", "true", "yes"}


def load_accidents() -> pd.DataFrame:
    """Load records from CSV, including ZIP bytes saved under a CSV suffix."""
    path = config.SAMPLE_CSV if using_sample_data() else _require_file(config.RAW_CSV)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if not members:
                raise ValueError(f"{path} is a ZIP archive without a CSV file.")
            payload = archive.read(members[0])
    else:
        payload = path.read_bytes()

    # The fast hosted sample is deliberately stored as ``.csv.gz`` to
    # keep the repository small.  Once bytes are read manually, pandas
    # can no longer infer compression from the filename, so decompress
    # the gzip payload before applying the encoding fallbacks below.
    if payload.startswith(b"\x1f\x8b"):
        payload = gzip.decompress(payload)

    if payload.startswith((b"\xff\xfe", b"\xfe\xff")):
        return pd.read_csv(BytesIO(payload), encoding="utf-16")
    try:
        return pd.read_csv(BytesIO(payload), encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(BytesIO(payload), encoding="cp1252")


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
