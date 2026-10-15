"""Phase 2 共用載入、輸出與特徵定義；不修改 Phase 1 或原始 CSV。"""

import json
import os
from pathlib import Path
import platform
from importlib.metadata import version

import numpy as np
import pandas as pd

from .validate_data import EXPECTED_COLUMNS, sha256_file


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data/raw/KAG_energydata_complete.csv"
REPORT_DIR = ROOT / "reports/phase2"
FIGURE_DIR = REPORT_DIR / "figures"
MODEL_DIR = REPORT_DIR / "models"
TARGET = "Appliances"
EXCLUDED_COLUMNS = (TARGET, "date", "rv1", "rv2")
BASELINE_FEATURES = [name for name in EXPECTED_COLUMNS if name not in EXCLUDED_COLUMNS]
EXPECTED_SHA256 = "2820bf712ad0275cb18b85a05250926100d8e65ebb9f4d2d016ca91ea152a25d"


def load_dataset() -> pd.DataFrame:
    """確認原始資料身分，僅在記憶體中解析 date；保留原始資料列索引。"""
    if sha256_file(CSV_PATH) != EXPECTED_SHA256:
        raise ValueError("原始 CSV SHA-256 不符 Phase 1，停止 Phase 2。")
    frame = pd.read_csv(CSV_PATH)
    if frame.shape != (19735, 29) or list(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError("資料結構不符 Phase 1。")
    if frame.isna().any().any() or frame.duplicated().any():
        raise ValueError("原始資料存在缺失值或重複列。")
    frame["date"] = pd.to_datetime(frame["date"], format="%Y-%m-%d %H:%M:%S", errors="raise")
    if not frame["date"].is_monotonic_increasing or frame["date"].duplicated().any():
        raise ValueError("時間欄位必須依序且唯一；不自行排序或修正。")
    numeric = frame.drop(columns="date")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in numeric.dtypes):
        raise ValueError("date 以外必須都是數值欄位。")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("數值欄位含 NaN 或無窮值。")
    return frame


def dataset_information(frame: pd.DataFrame) -> dict:
    return {
        "name": "Appliances Energy Prediction",
        "csv_path": CSV_PATH.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(CSV_PATH),
        "shape": {"rows": int(len(frame)), "columns": int(len(frame.columns))},
        "target": TARGET,
        "target_unit": "Wh (energy observation; not power in W)",
        "sampling_interval_seconds": 600,
        "date_range": {"start": frame["date"].min().isoformat(),
                       "end": frame["date"].max().isoformat()},
        "timezone": "原始資料未標示時區；保留原始時間，不推定或轉換時區。",
        "original_input_feature_count": 28,
    }


def prepare_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """預先定義的 25 欄 Baseline；不依 EDA 或模型分數選擇特徵。"""
    features = frame.loc[:, BASELINE_FEATURES].copy()
    target = frame[TARGET].copy()
    if features.shape[1] != 25 or TARGET in features or "date" in features:
        raise ValueError("Baseline 特徵定義不正確。")
    return features, target


def environment_versions() -> dict:
    packages = ("pandas", "numpy", "scikit-learn", "matplotlib", "joblib",
                "nbformat", "nbclient", "ipykernel")
    return {"python": platform.python_version(),
            **{package: version(package) for package in packages}}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
                    + "\n", encoding="utf-8")


def configure_matplotlib() -> None:
    # 設定快取僅寫入本專案的忽略目錄，適用 CLI 與 Notebook。
    os.environ["MPLCONFIGDIR"] = str(ROOT / ".venv/matplotlib-config")
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                               "figure.dpi": 110, "savefig.dpi": 160})


def save_figure(figure, name: str) -> str:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as plt
    plt.close(figure)
    return path.relative_to(ROOT).as_posix()
