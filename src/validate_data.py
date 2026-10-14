"""唯讀驗證原始 CSV；產生嚴格 JSON，必要條件失敗時回傳 exit code 1。"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PROJECT_ROOT / "data/raw/KAG_energydata_complete.csv"
DEFAULT_REPORT = PROJECT_ROOT / "reports/phase1/data_validation.json"
TARGET = "Appliances"
EXPECTED_COLUMNS = [
    "date", "Appliances", "lights", "T1", "RH_1", "T2", "RH_2",
    "T3", "RH_3", "T4", "RH_4", "T5", "RH_5", "T6", "RH_6",
    "T7", "RH_7", "T8", "RH_8", "T9", "RH_9", "T_out",
    "Press_mm_hg", "RH_out", "Windspeed", "Visibility", "Tdewpoint",
    "rv1", "rv2",
]


def display_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_float(value) -> float | None:
    return float(value) if pd.notna(value) and np.isfinite(value) else None


def numeric_details(series: pd.Series) -> dict:
    return {
        "dtype": str(series.dtype),
        "nan_count": int(series.isna().sum()),
        "positive_infinity_count": int(np.isposinf(series).sum()),
        "negative_infinity_count": int(np.isneginf(series).sum()),
    }


def validate(csv_path: Path) -> dict:
    report = {
        "dataset_name": "Appliances Energy Prediction",
        "csv_path": display_path(csv_path),
        "sha256": None,
        "shape": None,
        "columns": [],
        "missing_values": None,
        "duplicates": None,
        "target": {"name": TARGET, "exists": False},
        "feature_counts": None,
        "date_validation": {"exists": False},
        "rv_validation": {},
        "numeric_validation": None,
        "environment": {"python": platform.python_version(),
                        "pandas": pd.__version__, "numpy": np.__version__},
        "checks": [],
        "overall_status": "FAIL",
    }

    def check(name, passed, message, failure_status="FAIL"):
        report["checks"].append({
            "name": name,
            "status": "PASS" if passed else failure_status,
            "message": message,
        })

    def finish():
        statuses = {item["status"] for item in report["checks"]}
        report["overall_status"] = (
            "FAIL" if "FAIL" in statuses else "WARN" if "WARN" in statuses else "PASS"
        )
        return report

    try:
        report["sha256"] = sha256_file(csv_path)
        report["csv_size_bytes"] = csv_path.stat().st_size
        with csv_path.open(encoding="utf-8-sig", newline="") as stream:
            raw_headers = next(csv.reader(stream))
        frame = pd.read_csv(csv_path)
    except (OSError, UnicodeError, ValueError, StopIteration, pd.errors.ParserError) as error:
        check("csv_readable", False, f"CSV 讀取失敗：{error}")
        return finish()

    check("csv_readable", True, "CSV 成功讀取；原始內容未修改。")
    check("unique_column_names", len(set(raw_headers)) == len(raw_headers),
          "原始 CSV 標頭不得有重複欄名。")
    check("expected_columns", list(frame.columns) == EXPECTED_COLUMNS,
          "驗證 29 個預期欄名與順序。")
    rows, columns = frame.shape
    report["shape"] = {"rows": int(rows), "columns": int(columns)}
    report["columns"] = [{"name": str(name), "dtype": str(dtype)}
                         for name, dtype in frame.dtypes.items()]
    check("expected_rows", rows == 19735, f"預期 19735 rows；實際 {rows}。")
    check("expected_column_count", columns == 29, f"預期 29 columns；實際 {columns}。")

    missing = {str(name): int(count) for name, count in frame.isna().sum().items()}
    missing_total = sum(missing.values())
    report["missing_values"] = {"total": missing_total, "by_column": missing}
    duplicate_rows = int(frame.duplicated().sum())
    report["duplicates"] = {"complete_duplicate_rows": duplicate_rows}
    check("no_missing_values", missing_total == 0, f"總缺失值：{missing_total}。")
    check("no_duplicate_rows", duplicate_rows == 0, f"完全重複資料列：{duplicate_rows}。")

    target_exists = TARGET in frame.columns
    check("target_exists", target_exists, "必要 Target 欄位為 Appliances。")
    if target_exists:
        series = frame[TARGET]
        is_numeric = bool(pd.api.types.is_numeric_dtype(series))
        target = {"name": TARGET, "exists": True, "dtype": str(series.dtype),
                  "is_numeric": is_numeric, "missing_count": int(series.isna().sum())}
        if is_numeric:
            target.update(numeric_details(series))
            target["all_finite"] = bool(np.isfinite(series).all())
        else:
            target.update({"nan_count": int(series.isna().sum()),
                           "positive_infinity_count": None,
                           "negative_infinity_count": None, "all_finite": False})
        report["target"] = target
        check("target_numeric", is_numeric, f"Target dtype：{series.dtype}。")
        check("target_finite", target["all_finite"], "Target 必須沒有 NaN 或正負無窮值。")

    # 僅統計欄名；不刪除、選擇或衍生任何特徵。
    original_count = columns - 1 if target_exists else None
    report["feature_counts"] = {
        "original_input_feature_count": original_count,
        "candidate_feature_count_exclude_date": sum(
            name not in {TARGET, "date"} for name in frame.columns),
        "candidate_feature_count_exclude_date_rv1_rv2": sum(
            name not in {TARGET, "date", "rv1", "rv2"} for name in frame.columns),
        "teacher_constraint": "13 < original_input_feature_count < 40",
        "meets_teacher_constraint": bool(original_count is not None and 13 < original_count < 40),
        "feature_selection_performed": False,
    }
    check("expected_original_features", original_count == 28,
          f"預期 28 original input features；實際 {original_count}。")
    check("teacher_feature_constraint", report["feature_counts"]["meets_teacher_constraint"],
          "原始輸入 Features 必須 >13 且 <40；包含 date，排除 Target。")
    check("expected_candidate_features_exclude_date",
          report["feature_counts"]["candidate_feature_count_exclude_date"] == 27,
          "排除 Target 與 date 後，預期 27 個候選欄位；僅統計。")
    check("expected_candidate_features_exclude_date_rv1_rv2",
          report["feature_counts"]["candidate_feature_count_exclude_date_rv1_rv2"] == 25,
          "排除 Target、date、rv1、rv2 後，預期 25 個候選欄位；僅統計。")

    date_exists = "date" in frame.columns
    check("date_exists", date_exists, "必要時間欄位為 date。")
    if date_exists:
        # 解析結果只供驗證；不寫回 dataframe 或 CSV。
        dates = pd.to_datetime(frame["date"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
        invalid = int(dates.isna().sum())
        valid_dates = dates.dropna()
        differences = dates.diff().dropna().dt.total_seconds()
        interval_counts = differences.value_counts().sort_index()
        duplicate_dates = int(valid_dates.duplicated().sum())
        ordered = bool(dates.is_monotonic_increasing) if invalid == 0 else None
        date_report = {
            "exists": True, "format": "%Y-%m-%d %H:%M:%S",
            "all_parseable": invalid == 0, "invalid_or_missing_count": invalid,
            "invalid_non_missing_count": int((dates.isna() & frame["date"].notna()).sum()),
            "missing_count": int(frame["date"].isna().sum()),
            "earliest": valid_dates.min().isoformat() if not valid_dates.empty else None,
            "latest": valid_dates.max().isoformat() if not valid_dates.empty else None,
            "duplicate_timestamps": duplicate_dates, "is_monotonic_increasing": ordered,
            "interval_basis": "原始資料列順序中，相鄰且皆可解析的時間戳記差值；不重新排序。",
            "interval_distribution": [
                {"seconds": float(seconds), "count": int(count)}
                for seconds, count in interval_counts.items()
            ],
            "interval_observation_count": int(len(differences)),
        }
        report["date_validation"] = date_report
        check("date_parseable", invalid == 0, f"不可解析或缺失的時間戳記：{invalid}。")
        check("date_unique", duplicate_dates == 0, f"重複時間戳記：{duplicate_dates}。", "WARN")
        check("date_ordered", ordered is True, "時間應依原始資料列順序遞增。", "WARN")
        regular_intervals = invalid == 0 and len(differences) == rows - 1 and bool(
            (differences == 600).all())
        check("date_ten_minute_intervals", regular_intervals,
              "預期所有相鄰時間間隔皆為 600 秒；異常只回報，不修正。", "WARN")

    for name in ("rv1", "rv2"):
        exists = name in frame.columns
        details = {"exists": exists, "dtype": None, "count": None,
                   "mean": None, "standard_deviation": None, "min": None, "max": None}
        check(f"{name}_exists", exists, f"驗證隨機變數 {name} 存在。")
        if exists:
            series = frame[name]
            details.update({"dtype": str(series.dtype), "count": int(series.count()),
                            "is_numeric": bool(pd.api.types.is_numeric_dtype(series))})
            if details["is_numeric"]:
                details.update({"mean": finite_float(series.mean()),
                                "standard_deviation": finite_float(series.std(ddof=1)),
                                "standard_deviation_ddof": 1,
                                "min": finite_float(series.min()), "max": finite_float(series.max())})
        report["rv_validation"][name] = details
    if "rv1" in frame and "rv2" in frame:
        identical = bool(frame["rv1"].equals(frame["rv2"]))
        report["rv_validation"]["rv1_rv2_identical"] = identical
        check("rv1_rv2_distinct", not identical,
              "rv1、rv2 完全相同；記錄為 WARN，是否保留由後續實驗決定。"
              if identical else "rv1、rv2 並非完全相同。", "WARN")

    numeric = {str(name): numeric_details(frame[name]) for name in frame.columns
               if pd.api.types.is_numeric_dtype(frame[name])}
    non_numeric = [str(name) for name in frame.columns
                   if name != "date" and not pd.api.types.is_numeric_dtype(frame[name])]
    all_finite = all(item["nan_count"] == 0 and item["positive_infinity_count"] == 0
                     and item["negative_infinity_count"] == 0 for item in numeric.values())
    report["numeric_validation"] = {
        "non_numeric_columns_excluding_date": non_numeric,
        "columns": numeric, "all_numeric_values_finite": all_finite,
    }
    check("numeric_columns_except_date", not non_numeric,
          f"date 以外的非數值欄位：{non_numeric}。")
    check("numeric_values_finite", all_finite, "逐欄檢查數值欄位 NaN 與正負無窮值。")
    try:
        unchanged = sha256_file(csv_path) == report["sha256"]
        check("csv_unchanged_during_validation", unchanged, "驗證前後 CSV SHA-256 必須一致。")
    except OSError as error:
        check("csv_unchanged_during_validation", False, f"無法重新讀取 CSV：{error}")
    return finish()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="待驗證 CSV 路徑")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="輸出 JSON 路徑")
    args = parser.parse_args()
    csv_path = args.csv.resolve()
    report_path = args.report.resolve()
    if report_path == csv_path:
        parser.error("報告路徑不可與原始 CSV 相同。")
    report = validate(csv_path)
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
                               + "\n", encoding="utf-8")
    except (OSError, ValueError, TypeError) as error:
        print(f"FAIL：無法寫入驗證報告：{error}")
        return 1
    print(f"{report['overall_status']}：{display_path(report_path)}")
    for item in report["checks"]:
        if item["status"] != "PASS":
            print(f"{item['status']} [{item['name']}] {item['message']}")
    return 1 if report["overall_status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
