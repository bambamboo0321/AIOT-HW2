"""與前端無關的 Wizard v1 設定驗證、建議與 JSON 儲存。"""

from dataclasses import asdict, dataclass, field, fields
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .profiling import COLUMN_TYPES, validate_frame


@dataclass(frozen=True)
class PreprocessingConfig:
    target_column: str
    feature_columns: list[str]
    column_types: dict[str, str] = field(default_factory=dict)
    missing_numeric: str = "median"
    missing_categorical: str = "most_frequent"
    missing_target: str = "error"
    categorical_constant: str = "__UNKNOWN__"
    categorical_encoding: str = "onehot"
    numerical_scaling: str = "standard"
    outlier_strategy: str = "keep"
    iqr_multiplier: float = 1.5
    datetime_strategy: str = "exclude"
    datetime_formats: dict[str, str] = field(default_factory=dict)
    split_strategy: str = "random"
    split_column: str | None = None
    test_size: float = 0.2
    random_state: int = 42
    cv_splits: int = 5
    cv_gap: int = 0
    schema_version: int = 1

    def __post_init__(self):
        choices = {
            "missing_numeric": {"mean", "median", "drop", "none"},
            "missing_categorical": {"most_frequent", "constant", "drop"},
            "missing_target": {"error", "drop"}, "categorical_encoding": {"onehot", "drop"},
            "numerical_scaling": {"standard", "none"},
            "outlier_strategy": {"keep", "flag", "clip", "remove"},
            "datetime_strategy": {"exclude", "calendar", "cyclical"},
            "split_strategy": {"random", "chronological"},
        }
        for key, allowed in choices.items():
            value = getattr(self, key)
            if not isinstance(value, str) or value not in allowed:
                raise ValueError(f"{key} 必須為 {sorted(allowed)}。")
        if not isinstance(self.target_column, str) or not self.target_column:
            raise ValueError("target_column 必須為非空字串。")
        if (not isinstance(self.feature_columns, list) or not self.feature_columns
                or not all(isinstance(c, str) and c for c in self.feature_columns)
                or len(set(self.feature_columns)) != len(self.feature_columns)):
            raise ValueError("feature_columns 必須為非空且無重複的欄名字串陣列。")
        if self.target_column in self.feature_columns:
            raise ValueError("Target 不得同時為 Feature。")
        if (not isinstance(self.column_types, dict)
                or any(not isinstance(k, str) or not isinstance(v, str) or v not in COLUMN_TYPES
                       for k, v in self.column_types.items())):
            raise ValueError("column_types 含無效欄名或型態。")
        if self.target_column in self.column_types and self.column_types[self.target_column] != "numeric":
            raise ValueError("Target 型態覆寫僅允許 numeric。")
        if (not isinstance(self.datetime_formats, dict)
                or any(not isinstance(k, str) or not isinstance(v, str) or not v
                       for k, v in self.datetime_formats.items())):
            raise ValueError("datetime_formats 必須為欄名與日期 format 的對照表。")
        for key, minimum in (("random_state", 0), ("cv_splits", 2), ("cv_gap", 0)):
            value = getattr(self, key)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{key} 必須為 >= {minimum} 的整數。")
        if self.random_state >= 2**32:
            raise ValueError("random_state 必須小於 2**32。")
        if type(self.test_size) not in (float, int) or not np.isfinite(self.test_size) or not 0 < self.test_size < 1:
            raise ValueError("test_size 必須介於 0 與 1 之間。")
        if type(self.iqr_multiplier) not in (float, int) or not np.isfinite(self.iqr_multiplier) or self.iqr_multiplier <= 0:
            raise ValueError("iqr_multiplier 必須為有限正數。")
        if not isinstance(self.categorical_constant, str) or not self.categorical_constant:
            raise ValueError("categorical_constant 必須為非空字串。")
        if self.split_column is not None and (not isinstance(self.split_column, str) or not self.split_column):
            raise ValueError("split_column 必須為非空字串或 null。")
        if self.split_strategy == "chronological" and not self.split_column:
            raise ValueError("Chronological split 必須指定 split_column。")
        if self.split_column == self.target_column:
            raise ValueError("split_column 不得為 Target。")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("僅支援 schema_version=1。")

    def validate_columns(self, frame):
        validate_frame(frame)
        requested = ({self.target_column} | set(self.feature_columns) | set(self.column_types)
                     | set(self.datetime_formats) | ({self.split_column} if self.split_column else set()))
        if requested - set(frame.columns):
            raise ValueError(f"設定含不存在欄位：{sorted(requested - set(frame.columns))}")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, payload):
        if not isinstance(payload, dict):
            raise ValueError("Wizard 設定必須為 JSON object。")
        if set(payload) - {f.name for f in fields(cls)}:
            raise ValueError(f"未知設定欄位：{sorted(set(payload) - {f.name for f in fields(cls)})}")
        try:
            return cls(**payload)
        except TypeError as error:
            raise ValueError(f"設定缺少必要欄位：{error}") from error

    def save(self, path):
        path = Path(path)
        if path.suffix != ".json":
            raise ValueError("設定只能儲存為 .json，不可覆寫 CSV。")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def numeric_target(values):
    if (pd.api.types.is_datetime64_any_dtype(values) or pd.api.types.is_bool_dtype(values)
            or pd.api.types.is_complex_dtype(values)):
        raise ValueError("Target 必須為實數能源／連續量，不接受 datetime、boolean 或 complex。")
    try:
        result = pd.to_numeric(values, errors="raise").astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Target {values.name} 無法轉換為數值。") from error
    if np.isinf(result.to_numpy()).any():
        raise ValueError("Target 含無窮值。")
    if result.nunique() < 2:
        raise ValueError("Target 全缺失或為常數，無法建立有意義的迴歸實驗。")
    return result


def recommend_config(profile, target_column):
    lookup = {c["name"]: c for c in profile["columns"]}
    if target_column not in lookup or not lookup[target_column]["target_candidate"]:
        raise ValueError("請選擇有效的數值型 Target；可先覆寫型態再重新 profiling。")
    selected = [c["name"] for c in profile["columns"] if c["name"] != target_column
                and c["effective_type"] in {"numeric", "categorical"}
                and not c["possible_constant"] and not c["possible_id"]]
    dates = profile["candidates"]["datetime"]
    config = PreprocessingConfig(target_column=target_column, feature_columns=selected,
                                 column_types={c: lookup[c]["effective_type"] for c in selected},
                                 split_strategy="chronological" if dates else "random",
                                 split_column=dates[0] if dates else None)
    return {"config": config.to_dict(), "reasons": {
        "feature_columns": "排除 Target、常數、可能 ID、自由文字；日期預設僅供排序，均可覆寫。",
        "missing_numeric": "Median 對偏態較穩健；每個訓練 fold 單獨估計。",
        "numerical_scaling": "StandardScaler 支援未來對尺度敏感的 Lasso；不改變正式模型類型。",
        "outlier_strategy": "Keep 保留真實高耗能；IQR 候選不等於錯誤值。",
        "split_strategy": "日期可能代表時間依賴，優先提議時間切分；日期的業務意義仍需使用者確認。",
    }, "requires_user_review": True}
