"""M6-A: validate custom CSV/config and materialize split indices, without fitting."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, TimeSeriesSplit

from .data_splitting import chronological_dates, split_dataset
from .preprocessing_config import PreprocessingConfig, numeric_target

MAX_BYTES = 25 * 1024 * 1024
MAX_ROWS, MAX_COLUMNS, MAX_CELLS = 50_000, 200, 1_000_000
NA = {"", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN", "<NA>", "N/A", "NA", "NULL", "NaN", "None", "n/a", "nan", "null"}
ISO = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d{1,3})?)?)?$")
NUMERIC = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def _bytes(path, limit, suffix):
    path = Path(path)
    if path.suffix.lower() != suffix:
        raise ValueError(f"請提供 {suffix} 檔案。")
    with path.open("rb") as stream:
        value = stream.read(limit + 1)
    if not value or len(value) > limit:
        raise ValueError(f"檔案必須非空且不超過 {limit} bytes。")
    return value


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("JSON 有重複 key。")
        result[key] = value
    return result


def _document(path, limit):
    def reject_constant(_):
        raise ValueError("JSON 不接受 NaN 或 Infinity。")
    return json.loads(_bytes(path, limit, ".json").decode("utf-8-sig"),
                      object_pairs_hook=_pairs, parse_constant=reject_constant)


def _frame(raw):
    text = raw.decode("utf-8-sig", errors="strict")
    if "\0" in text:
        raise ValueError("CSV 含 NUL 字元。")
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    headers, rows = None, []
    for row in reader:
        if not row:  # Only physical blank records, not quoted empty cells.
            continue
        if len(row) > MAX_COLUMNS or any(len(v.encode("utf-16-le")) // 2 > 65536 for v in row):
            raise ValueError("CSV 超過 200 欄或單格 65,536 UTF-16 code units。")
        if headers is None:
            headers = row
            if any(not h.strip() or h != h.strip() or len(h.encode("utf-16-le")) // 2 > 128 for h in row) or len(set(row)) != len(row):
                raise ValueError("CSV 欄名須唯一、非空、無首尾空白且不超過 128 code units。")
        else:
            if len(row) != len(headers):
                raise ValueError("CSV 資料欄數與標頭不一致。")
            rows.append([np.nan if v in NA else v for v in row])
            if len(rows) > MAX_ROWS or len(rows) * len(headers) > MAX_CELLS:
                raise ValueError("CSV 超過 50,000 筆或 100 萬個儲存格。")
    if not headers or not rows:
        raise ValueError("CSV 沒有資料列。")
    return pd.DataFrame(rows, columns=headers)


def _numeric(series):
    values = series.dropna()
    if not values.map(lambda v: bool(NUMERIC.fullmatch(str(v).strip()))).all():
        raise ValueError(f"欄位 {series.name} 含非數值或非有限數值。")
    try:
        result = pd.to_numeric(series, errors="raise").astype(float)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(f"欄位 {series.name} 無法安全轉換為數值。") from error
    finite = result.dropna().to_numpy()
    if not np.isfinite(finite).all() or ((np.abs(finite) > 2**53 - 1) & (finite == np.floor(finite))).any():
        raise ValueError(f"欄位 {series.name} 含無窮值或超出瀏覽器安全整數範圍的值。")
    return result


def validate_custom_dataset(csv_path, config_path, review_path=None):
    """Return JSON-safe validation only; never fit/transform/load/save a model."""
    stage = "config"
    try:
        config = PreprocessingConfig.from_dict(_document(config_path, 128 * 1024))
        if (config.missing_numeric not in {"mean", "median", "none"}
                or config.outlier_strategy not in {"keep", "clip"}
                or config.datetime_strategy != "exclude" or config.datetime_formats
                or config.missing_categorical != "most_frequent"
                or config.categorical_encoding != "onehot"
                or config.cv_splits > 10 or config.cv_gap > MAX_ROWS
                or (config.split_strategy == "random" and config.cv_gap != 0)):
            raise ValueError("設定超出 M5／M6-A 的數值型檢查範圍。")
        active = set(config.feature_columns) | {config.target_column}
        if any(k not in active or v != "numeric" for k, v in config.column_types.items()):
            raise ValueError("本階段僅支援目標與所選數值特徵的 numeric 型態設定。")
        stage = "csv"
        raw = _bytes(csv_path, MAX_BYTES, ".csv")
        digest = hashlib.sha256(raw).hexdigest()
        frame = _frame(raw)
        config.validate_columns(frame)
        stage = "data"
        for name in sorted(active):
            frame[name] = _numeric(frame[name])
            if frame[name].nunique() < 2:
                raise ValueError(f"欄位 {name} 為常數或全缺失。")
        numeric_target(frame[config.target_column])
        if config.missing_numeric == "none" and frame[config.feature_columns].isna().any().any():
            raise ValueError("所選特徵有缺失，missing_numeric=none 不可執行。")
        if config.split_strategy == "chronological":
            values = frame[config.split_column]
            if values.isna().any() or not values.map(lambda v: isinstance(v, str) and bool(ISO.fullmatch(v))).all():
                raise ValueError("排序欄需完整、無時區的 ISO 日期。")
            chronological_dates(frame, config)  # Existing pandas parser + uniqueness checks.
        stage = "review"
        if review_path is not None:
            review = _document(review_path, 1024 * 1024)
            if (not isinstance(review, dict) or review.get("schema_version") != 1
                    or review.get("artifact_type") != "custom_dataset_review"):
                raise ValueError("不是 M5 的檢查摘要格式。")
            state = review.get("review", {})
            if not isinstance(state, dict) or state.get("confirmed") is not True or state.get("training_started") is not False or state.get("scope") != "configuration_only":
                raise ValueError("檢查摘要未標示已確認的設定預覽。")
            metadata = review.get("dataset", {})
            if not isinstance(metadata, dict) or metadata.get("sha256") != digest:
                raise ValueError("CSV 與檢查摘要的 SHA-256 不一致，請重新檢查與確認。")
            if PreprocessingConfig.from_dict(review.get("preprocessing")).to_dict() != config.to_dict():
                raise ValueError("設定與已確認摘要不一致，請重新確認。")
        stage = "split"
        split = split_dataset(frame, config)
        warnings = []
        if len(frame.columns) - 1 < 13:
            warnings.append("原始候選輸入少於 13 欄；此為作業資料規範，與選取後 k 分開。")
        if config.split_strategy == "random" and any(frame[c].dropna().astype(str).str.match(ISO).all() for c in frame if c not in active and frame[c].notna().any()):
            warnings.append("未選欄位含 ISO 日期；請確認隨機切分適合此資料。")
        splitter = (TimeSeriesSplit(n_splits=config.cv_splits, gap=config.cv_gap)
                    if config.split_strategy == "chronological" else
                    KFold(n_splits=config.cv_splits, shuffle=True, random_state=config.random_state))
        partitions = [("outer_training", split.X_train, split.y_train), ("holdout", split.X_test, split.y_test)]
        folds = []
        for i, (train, valid) in enumerate(splitter.split(split.X_train), 1):
            if len(train) < 2 or not len(valid):
                raise ValueError("CV 訓練至少需兩列、驗證至少需一列；請調整 folds／gap。")
            folds.append({"fold": i, "training_rows": len(train), "validation_rows": len(valid)})
            partitions.extend([(f"fold_{i}_training", split.X_train.iloc[train], split.y_train.iloc[train]),
                               (f"fold_{i}_validation", split.X_train.iloc[valid], split.y_train.iloc[valid])])
        for label, X, y in partitions:
            if len(X) < (2 if label.endswith("training") else 1):
                raise ValueError(f"{label} 筆數不足。")
            if y.nunique() < 2:
                if label.endswith("training"):
                    raise ValueError(f"{label} 的目標為常數，請調整資料或切分。")
                warnings.append(f"{label} 的目標為常數，R² 將無法定義。")
            constant_count = sum(X[c].nunique() <= 1 for c in config.feature_columns)
            missing_count = sum(X[c].notna().sum() == 0 for c in config.feature_columns)
            if constant_count:
                warnings.append(f"{label} 有 {constant_count} 個常數／全缺失特徵，其中 {missing_count} 個全缺失；未執行補值。")
        normalized = json.dumps(config.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return {"schema_version": 1, "status": "valid", "scope": "pretraining_validation_only", "training_started": False,
                "dataset": {"sha256": digest, "bytes": len(raw), "rows": len(frame), "columns": len(frame.columns)},
                "config_sha256": hashlib.sha256(normalized.encode()).hexdigest(), "review_checked": review_path is not None,
                "split": {"strategy": config.split_strategy, "training_rows": len(split.X_train), "holdout_rows": len(split.X_test),
                          "target_missing_dropped": len(split.dropped_target_indices), "cv_folds": folds},
                "warnings": warnings, "limits": ["未 fit 或 transform，未估計補值、縮放、IQR 門檻或選特徵。", "驗證通過不代表模型效果、PI 適用性或後續訓練成功。"]}
    except (ValueError, TypeError, KeyError, OverflowError, OSError, csv.Error) as error:
        message = "檔案無法讀取，請檢查路徑與權限。" if isinstance(error, OSError) else str(error)
        return {"schema_version": 1, "status": "invalid", "scope": "pretraining_validation_only",
                "training_started": False, "errors": [{"stage": stage, "message": message}]}


def main(argv=None):
    parser = argparse.ArgumentParser(description="只驗證 CSV 與 M5 前處理設定，不訓練、不改寫檔案。")
    parser.add_argument("--csv", required=True, help="本機 CSV 路徑")
    parser.add_argument("--config", required=True, help="M5 下載的 custom-dataset-config.json")
    parser.add_argument("--review", help="可選：M5 下載的 custom-dataset-review.json，核對資料與設定版本")
    args = parser.parse_args(argv)
    result = validate_custom_dataset(args.csv, args.config, args.review)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result["status"] == "valid" else 2


if __name__ == "__main__":
    sys.exit(main())
