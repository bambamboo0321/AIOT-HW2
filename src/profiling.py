"""唯讀、JSON-compatible 的一般 CSV 剖析；型態均為可覆寫建議。"""

import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd

COLUMN_TYPES = {"numeric", "categorical", "datetime", "text", "exclude"}
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[T ].*)?$")


def validate_frame(frame):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("資料必須是非空的 pandas DataFrame。")
    if not frame.columns.is_unique or not all(isinstance(c, str) and c for c in frame.columns):
        raise ValueError("欄位名稱必須是非空、唯一的字串。")


def read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    """不讓 pandas 自動改名掩蓋重複 CSV 標頭；不寫入任何原始資料。"""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), [])
    if not header or any(not c for c in header) or len(header) != len(set(header)):
        raise ValueError("CSV 標頭不得空白或重複。")
    frame = pd.read_csv(path, **kwargs)
    validate_frame(frame)
    return frame


def parse_datetime(values, date_format=None):
    """只接受 datetime 或 ISO 日期字串；其他格式必須明確提供 format。"""
    if pd.api.types.is_datetime64_any_dtype(values):
        return pd.to_datetime(values, errors="raise")
    nonnull = values.dropna()
    if pd.api.types.is_numeric_dtype(values) or any(not isinstance(v, str) for v in nonnull):
        raise ValueError(f"{values.name}: 不接受數值／Unix Timestamp 作日期；請明確轉換資料。")
    if date_format is None and not nonnull.map(lambda v: bool(ISO_DATE.fullmatch(v))).all():
        raise ValueError(f"{values.name}: 日期須為 ISO YYYY-MM-DD 格式，或設定 datetime_formats。")
    try:
        result = pd.to_datetime(values, format=date_format or "ISO8601", errors="raise")
        if not pd.api.types.is_datetime64_any_dtype(result):
            raise ValueError("混合時區不支援；請先統一時區。")
        return result
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(f"{values.name}: 無法解析日期：{error}") from error


def infer_type(values):
    if pd.api.types.is_datetime64_any_dtype(values):
        return "datetime"
    if pd.api.types.is_bool_dtype(values) or isinstance(values.dtype, pd.CategoricalDtype):
        return "categorical"
    if pd.api.types.is_numeric_dtype(values):
        return "numeric"
    nonnull = values.dropna()
    if nonnull.empty:
        return "text"
    strings = nonnull.astype(str)
    if strings.map(lambda v: bool(ISO_DATE.fullmatch(v))).all():
        try:
            parse_datetime(values)
            return "datetime"
        except ValueError:
            pass
    # 前置零或很長的純數字字串可能是識別碼，保留其文字語意。
    numeric = pd.to_numeric(nonnull, errors="coerce")
    identifier_like = strings.str.match(r"^(0\d+|\d{16,})$").any()
    if numeric.notna().all() and not identifier_like:
        return "numeric"
    unique = nonnull.nunique()
    if strings.str.len().mean() <= 80 and (unique <= 20 or (unique <= 50 and unique / len(nonnull) <= 0.2)):
        return "categorical"
    return "text"


def finite_number(value):
    return float(value) if pd.notna(value) and np.isfinite(value) else None


def profile_dataframe(frame: pd.DataFrame, column_types=None) -> dict:
    validate_frame(frame)
    overrides = {} if column_types is None else column_types
    if not isinstance(overrides, dict) or set(overrides) - set(frame.columns):
        raise ValueError("column_types 必須為已存在欄名的對照表。")
    if any(not isinstance(kind, str) or kind not in COLUMN_TYPES for kind in overrides.values()):
        raise ValueError(f"column_types 僅接受 {sorted(COLUMN_TYPES)}。")
    rows = len(frame)
    columns = []
    for name in frame:
        values = frame[name]
        inferred = infer_type(values)
        kind = overrides.get(name, inferred)
        count = int(values.count())
        unique = int(values.nunique())
        ratio = unique / count if count else 0.0
        warnings = []
        constant = unique <= 1
        id_name = bool(re.search(r"(^id$|_id$|^id_|uuid|identifier|^index$)", name, re.I))
        sequence = (pd.api.types.is_integer_dtype(values) and count == rows and unique == rows
                    and rows >= 3 and values.diff().dropna().eq(1).all())
        possible_id = id_name or bool(sequence)
        high_cardinality = kind in {"categorical", "text"} and unique > 20 and ratio >= 0.5
        if constant:
            warnings.append("常數或全缺失欄位，建議排除；不適合作為 Target。")
        if possible_id:
            warnings.append("可能為 ID，建議確認語意後排除；唯一數值本身不足以判定 ID。")
        if high_cardinality:
            warnings.append("高基數文字／類別，OneHot 可能增加記憶體需求。")
        if kind == "text":
            warnings.append("自由文字或無法可靠分類；不可直接作 LinearRegression Feature。")
        summary = None
        outliers = None
        invalid = 0
        infinite = 0
        if kind == "numeric":
            converted = pd.to_numeric(values, errors="coerce")
            invalid = int((values.notna() & converted.isna()).sum())
            array = converted.to_numpy(dtype=float, na_value=np.nan)
            infinite = int(np.isinf(array).sum())
            finite = pd.Series(array[np.isfinite(array)], dtype=float)
            summary = {str(k): finite_number(v) for k, v in finite.describe().items()}
            summary["skewness"] = finite_number(finite.skew())
            q1, q3 = finite.quantile([0.25, 0.75])
            low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
            outliers = {"method": "1.5_IQR_descriptive_only", "lower": finite_number(low),
                        "upper": finite_number(high), "count": int(((finite < low) | (finite > high)).sum())}
            if invalid or infinite:
                warnings.append(f"無法轉換的數值 {invalid} 筆；無窮值 {infinite} 筆，前處理會拒絕。")
        elif kind == "datetime":
            try:
                parse_datetime(values)
            except ValueError:
                warnings.append("日期覆寫後仍無法解析；需提供明確日期格式。")
        target_candidate = kind == "numeric" and not constant and not possible_id and not invalid and not infinite
        if kind != "numeric":
            warnings.append("本專案 Target 必須為數值型；此欄目前不建議作 Target。")
        columns.append({
            "name": name, "dtype": str(values.dtype), "inferred_type": inferred,
            "effective_type": kind, "overridden": name in overrides,
            "missing_count": rows - count, "missing_fraction": (rows - count) / rows,
            "unique_count": unique, "unique_fraction_nonmissing": ratio,
            "category_count": unique if kind == "categorical" else None,
            "numerical_summary": summary, "outliers": outliers,
            "possible_constant": constant, "possible_id": possible_id,
            "high_cardinality": high_cardinality, "target_candidate": target_candidate,
            "warnings": warnings,
        })
    return {"schema_version": 1, "rows": rows, "column_count": len(frame.columns),
            "column_names": list(frame.columns), "duplicate_rows": int(frame.duplicated().sum()),
            "columns": columns,
            "candidates": {kind: [c["name"] for c in columns if c["effective_type"] == kind]
                           for kind in ("numeric", "categorical", "datetime", "text")},
            "notes": ["唯讀建議；未輸出資料列或類別值清單。", "型態可覆寫，統計不得作全資料 fit 規則。"]}


def profile_csv(path, column_types=None):
    return profile_dataframe(read_csv(path), column_types)
