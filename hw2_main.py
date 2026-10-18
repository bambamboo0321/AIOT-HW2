"""HW2 暫定主程式；完整學號確認後再命名，不提供訓練命令。"""

import argparse
import json
import sys

# 成果介接命令不在 src/ 建立 Python bytecode cache。
sys.dont_write_bytecode = True


COMMANDS = {
    "dataset-summary": ("dataset_summary", "顯示 Dataset、欄位、缺失與異常值概況"),
    "crisp-dm": ("crisp_dm", "顯示 CRISP-DM 六階段摘要與證據路徑"),
    "feature-selection": ("feature_selection", "分開顯示六種方法的 Training CV／Nested CV／Training refit"),
    "model-comparison": ("model_comparison", "顯示 Baseline／MI 固定模型比較；Random Split 另列"),
    "prediction-intervals": ("prediction_intervals", "顯示既有 95% Prediction Interval 統計與限制"),
    "figures": ("figures_manifest", "列出全部既有圖表相對路徑及來源"),
    "verify": ("verify", "唯讀檢查必要檔案、來源雜湊及 Phase 1 Dataset"),
    "export-dashboard": ("export_dashboard", "只寫入 dashboard_data/ 的五份 JSON，不訓練模型"),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description="AIOT HW2：讀取既有實驗成果；不重新訓練模型。")
    subparsers = parser.add_subparsers(dest="command", title="操作")
    for name, (_, help_text) in COMMANDS.items():
        subparsers.add_parser(name, help=help_text.replace("%", "%%"), description=help_text)
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    from src.dashboard_data import ResultError, ResultStore
    store = ResultStore()
    builder = COMMANDS[args.command][0]
    try:
        payload = (getattr(store, builder)() if builder in {"verify", "export_dashboard"}
                   else store.document(builder))
        print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ResultError, OSError) as error:
        print(f"錯誤：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
