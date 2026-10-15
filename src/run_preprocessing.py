"""Phase 2 設定與轉換驗收；不訓練模型、不計算 holdout Performance。"""

from dataclasses import replace
import hashlib

from .data_splitting import prepare_fold, split_dataset
from .phase2_common import CSV_PATH, EXPECTED_SHA256, REPORT_DIR, environment_versions, write_json
from .preprocessing_config import PreprocessingConfig, recommend_config
from .profiling import profile_csv, read_csv
from .validate_data import sha256_file


def index_digest(index):
    return hashlib.sha256(",".join(map(str, index)).encode()).hexdigest()


def run():
    before = sha256_file(CSV_PATH)
    if before != EXPECTED_SHA256:
        raise ValueError("原始 CSV SHA-256 與 Phase 1 不符。")
    frame = read_csv(CSV_PATH)
    profile = profile_csv(CSV_PATH)
    suggestions = recommend_config(profile, "Appliances")
    generic = PreprocessingConfig.from_dict(suggestions["config"])
    # 來源語意建議，不依 Target correlation 或 Test 分數；使用者可覆寫。
    config = replace(generic, feature_columns=[c for c in generic.feature_columns if c not in {"rv1", "rv2"}],
                     column_types={c: t for c, t in generic.column_types.items() if c not in {"rv1", "rv2"}},
                     cv_gap=6)
    suggestions["energy_config"] = config.to_dict()
    suggestions["reasons"]["rv1_rv2"] = "UCI 定義的隨機負對照且兩欄相同；建議排除，允許覆寫。"
    suggestions["reasons"]["cv_gap"] = "6 筆為 1 小時示例緩衝，未用 Test 調整；需依未來預測視窗預先決定。"
    split = split_dataset(frame, config)
    prepared = prepare_fold(split.X_train, split.y_train, split.X_test, split.y_test, config)
    with_dates = replace(config, feature_columns=config.feature_columns + ["date"], datetime_strategy="calendar")
    date_example = prepare_fold(split.X_train, split.y_train, split.X_test, split.y_test, with_dates)
    report = {"dataset_sha256": before, "environment": environment_versions(),
              "config_path": "reports/phase2/energy_config.json", "config": config.to_dict(),
              "target": config.target_column, "original_candidate_count": 28,
              "training_rows": len(split.X_train), "test_rows": len(split.X_test),
              "training_index_sha256": index_digest(split.X_train.index),
              "test_index_sha256": index_digest(split.X_test.index),
              "training_time_range": [split.X_train["date"].min(), split.X_train["date"].max()],
              "test_time_range": [split.X_test["date"].min(), split.X_test["date"].max()],
              "feature_summary": prepared.summary, "row_audit": prepared.row_audit,
              "calendar_example": date_example.summary,
              "test_used_for_fitting": False, "model_training_performed": False,
              "formal_feature_selection_performed": False,
              "raw_csv_unchanged": sha256_file(CSV_PATH) == before}
    if not report["raw_csv_unchanged"]:
        raise RuntimeError("CSV 在處理過程中被修改。")
    config.save(REPORT_DIR / "energy_config.json")
    write_json(REPORT_DIR / "data_profile.json", profile)
    write_json(REPORT_DIR / "wizard_recommendations.json", suggestions)
    write_json(REPORT_DIR / "preprocessing_summary.json", report)
    print(f"PASS: 原始候選 28；預設轉換後 {prepared.X_train.shape[1]}；calendar 示例 {date_example.X_train.shape[1]}。未訓練模型。")
    return report


if __name__ == "__main__":
    run()
