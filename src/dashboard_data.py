"""既有成果的唯讀介接；只允許將衍生 JSON 寫入 dashboard_data/。"""

import json
import math
import os
from pathlib import Path
import tempfile

from .feature_selection import METHODS
from .phase2_common import ROOT
from .validate_data import sha256_file, validate


BASELINE = "reports/phase2/baseline_metrics.json"
BEST = "reports/phase3/results/best_features.json"
RESULTS = "reports/phase3/results/feature_selection_results.json"
PLOTS = "reports/phase3/results/plot_data.json"
METRICS = "reports/phase3/phase3c/prediction_metrics.json"
INTERVALS = "reports/phase3/phase3c/prediction_interval_metrics.json"
FIGURES = "reports/phase3/phase3c/figure_manifest.json"
PROFILE = "reports/phase2/data_profile.json"
EDA = "reports/phase2/eda_statistics.json"
EXPORT_NAMES = ("dataset_summary", "feature_selection", "model_comparison",
                "prediction_intervals", "figures_manifest")
METRIC_UNITS = {"mae": "Wh", "mse": "Wh^2", "rmse": "Wh", "r2": "dimensionless"}


class ResultError(ValueError):
    """缺少、損壞或互相矛盾的結果；不建立替代成績。"""


def required(obj, key, kind):
    if (not isinstance(obj, dict) or key not in obj or not isinstance(obj[key], kind)
            or (kind is int and type(obj[key]) is not int)):
        raise ResultError(f"結果格式錯誤：必要欄位 {key} 缺少或型態不符。")
    return obj[key]


def finite_metrics(obj, keys=("mae", "rmse", "r2")):
    for key in keys:
        value = required(obj, key, (int, float))
        if isinstance(value, bool) or not math.isfinite(value):
            raise ResultError(f"結果格式錯誤：{key} 必須是有限數值。")
    return obj


def validate_document(document):
    """不依賴前端套件的 v1 envelope 基本契約檢查。"""
    if type(document.get("schema_version")) is not int or document["schema_version"] != 1:
        raise ResultError("不支援的 dashboard schema_version。")
    required(document, "artifact_type", str)
    required(document, "data", dict)
    required(document, "units", dict)
    required(document, "missing_fields", list)
    for source in required(document, "sources", list):
        path = Path(required(source, "path", str))
        if path.is_absolute() or ".." in path.parts:
            raise ResultError("來源必須使用 Repository 相對路徑。")
        digest = required(source, "sha256", str)
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ResultError("來源 SHA-256 格式錯誤。")
    json.dumps(document, ensure_ascii=False, allow_nan=False)


class ResultStore:
    """root 可供測試 fixture 使用；CLI 固定使用程式所在的 Repository。"""

    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()
        self.sources = {}

    def path(self, relative):
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ResultError(f"不允許的結果路徑：{relative}")
        resolved = (self.root / path).resolve()
        if not resolved.is_relative_to(self.root):
            raise ResultError(f"結果路徑超出 Repository：{relative}")
        if not resolved.is_file():
            raise ResultError(f"缺少必要結果檔案：{relative}")
        return resolved

    def check_hash(self, relative, expected=None):
        digest = sha256_file(self.path(relative))
        if expected is not None and digest != expected:
            raise ResultError(f"SHA-256 不一致：{relative}")
        self.sources[relative] = {"path": relative, "sha256": digest}
        return digest

    def read(self, relative):
        try:
            content = self.path(relative).read_text(encoding="utf-8")
            def reject_constant(value):
                raise ValueError(f"非 JSON 有限值：{value}")
            obj = json.loads(content, parse_constant=reject_constant)
            if not isinstance(obj, dict):
                raise ValueError("頂層必須是 object")
        except (OSError, UnicodeError, ValueError) as error:
            raise ResultError(f"無法讀取結果 {relative}：{error}") from error
        self.check_hash(relative)
        return obj

    def dataset_summary(self):
        baseline, profile = self.read(BASELINE), self.read(PROFILE)
        info = required(baseline, "dataset_information", dict)
        self.check_hash(required(info, "csv_path", str), required(info, "sha256", str))
        shape = required(info, "shape", dict)
        if (required(shape, "rows", int) != required(profile, "rows", int)
                or required(shape, "columns", int) != required(profile, "column_count", int)):
            raise ResultError("Dataset Summary 與既有 profiling 的資料大小不一致。")
        return {"dataset": info,
                "baseline_feature_count": required(baseline, "baseline_feature_count", int),
                "baseline_features": required(baseline, "feature_names", list),
                "excluded_columns": required(baseline, "excluded_columns", list),
                "profile": profile}

    def feature_selection(self):
        best, plots = self.read(BEST), self.read(PLOTS)
        results = self.read(RESULTS)
        if required(results, "status", str) != "complete":
            raise ResultError("正式 Feature Selection 結果尚未完成。")
        if results.get("winner") != best.get("winner") or results.get("finalists") != best.get("by_method"):
            raise ResultError("Feature Selection 結果與最佳特徵摘要不一致。")
        if (required(best, "holdout_used", bool)
                or required(required(results, "config", dict), "holdout_target_read", bool)):
            raise ResultError("Feature Selection 紀錄顯示使用 Holdout，停止匯出。")
        self.check_hash("reports/phase3/results/feature_selection_results.csv",
                        required(plots, "csv_sha256", str))
        by_method = {}
        for key in ("by_method", "nested_outer_assessment", "full_training_refits"):
            records = required(best, key, list)
            indexed = {required(row, "method", str): row for row in records}
            if len(records) != len(METHODS) or set(indexed) != set(METHODS):
                raise ResultError(f"{key} 必須包含六種方法各一次。")
            for row in records:
                finite_metrics(row)
            by_method[key] = indexed
        methods = []
        for name in METHODS:
            refit = by_method["full_training_refits"][name]
            selection = required(refit, "selection", dict)
            names = required(selection, "selected_feature_names", list)
            if len(names) != required(refit, "selected_k", int):
                raise ResultError(f"{name} 選取欄位數量不一致。")
            methods.append({"method": name,
                            "training_cv": by_method["by_method"][name],
                            "nested_cv": by_method["nested_outer_assessment"][name],
                            "full_training_refit": refit})
        return {"selection_level": required(best, "selection_level", str),
                "winner": required(best, "winner", dict), "methods": methods,
                "holdout_used_for_selection": required(best, "holdout_used", bool),
                "warning": required(best, "warning", str),
                "curves": required(plots, "figures", dict),
                "experiment": {"status": results["status"],
                               "case_count": len(required(results, "cases", list)),
                               "record_count": len(required(results, "records", list)),
                               "config": required(results, "config", dict)}}

    def model_comparison(self):
        metrics, baseline = self.read(METRICS), self.read(BASELINE)
        models = required(metrics, "models", dict)
        for name in ("baseline", "selected"):
            model = required(models, name, dict)
            for key in ("training_metrics", "holdout_metrics"):
                finite_metrics(required(model, key, dict), METRIC_UNITS)
            if len(required(model, "selected_features", list)) != required(model, "feature_count", int):
                raise ResultError(f"{name} 的特徵數與名稱不一致。")
            self.check_hash(required(model, "model_file", str), required(model, "model_sha256", str))
        chronological = required(required(baseline, "baselines", dict), "chronological", dict)
        if models["baseline"]["holdout_metrics"] != chronological.get("test_metrics"):
            raise ResultError("Phase 2 與 Phase 3C 的 Baseline 指標不一致。")
        if models["baseline"]["training_metrics"] != chronological.get("train_metrics"):
            raise ResultError("Phase 2 與 Phase 3C 的 Baseline Training 指標不一致。")
        output_models = {}
        for name, model in models.items():
            output_models[name] = {key: model[key] for key in (
                "model_name", "feature_count", "selected_features", "model_file", "model_sha256",
                "training_metrics", "holdout_metrics", "training_period", "holdout_period")}
        random = required(baseline["baselines"], "random", dict)
        return {"evaluation": required(metrics, "evaluation", str),
                "plan": required(metrics, "plan", dict), "models": output_models,
                "transformations": required(metrics, "transformation_checks", dict),
                "random_baseline_comparison_only": {
                    "split": required(random, "split_metadata", dict),
                    "training_metrics": finite_metrics(required(random, "train_metrics", dict), METRIC_UNITS),
                    "test_metrics": finite_metrics(required(random, "test_metrics", dict), METRIC_UNITS)},
                "cv_results": "feature_selection.json；不可與 Holdout 混成同一排名。"}

    def prediction_intervals(self):
        intervals = self.read(INTERVALS)
        required(intervals, "plan", dict)
        required(intervals, "formula", str)
        required(intervals, "sigma_source", str)
        for name in ("baseline", "selected"):
            model = required(required(intervals, "models", dict), name, dict)
            finite_metrics(model, ("nominal_coverage", "empirical_coverage", "average_width",
                                   "median_width", "width_standard_deviation"))
            if not 0 <= model["empirical_coverage"] <= 1:
                raise ResultError(f"{name} Coverage 必須為 0 到 1 的比例。")
        return {"interval_type": "individual_observation_prediction_interval", **intervals,
                "limitations": ["名目 Coverage 不是實測保證；區間未依 Holdout 校準。",
                                "OLS 假設、時間相依及選特徵不確定性限制見 Phase 3C 報告。"]}

    def figures_manifest(self):
        baseline, eda = self.read(BASELINE), self.read(EDA)
        phase3b, phase3c = self.read(PLOTS), self.read(FIGURES)
        self.check_hash("reports/phase3/results/feature_selection_results.csv",
                        required(phase3b, "csv_sha256", str))
        for name, digest in required(phase3c, "sources", dict).items():
            self.check_hash(f"reports/phase3/phase3c/{name}", digest)
        figures = []
        groups = (("eda", required(eda, "figures", list)),
                  ("baseline_evaluation", required(baseline, "figures", list)),
                  ("feature_selection", [f"reports/phase3/figures/{n}" for n in required(phase3b, "figures", dict)]),
                  ("prediction_interval", [f"reports/phase3/figures/phase3c/{n}" for n in required(phase3c, "figures", dict)]))
        for group, paths in groups:
            for path in paths:
                figures.append({"category": group, "path": path,
                                "sha256": self.check_hash(path)})
        return {"path_base": "repository_root", "figures": figures,
                "phase3b_provenance": phase3b, "phase3c_provenance": phase3c}

    def crisp_dm(self):
        stages = [
            ("Business Understanding", "家庭家電耗能估計；評估精簡 Features 的效益。", "reports/phase2/business_understanding.md"),
            ("Data Understanding", "資料來源、EDA、偏態、相關性及時間相依。", "reports/phase2/eda_summary.md"),
            ("Data Preparation", "設定驗證、補值、編碼、縮放與 fold-local 前處理。", "docs/preprocessing_wizard.md"),
            ("Modeling", "六種 Feature Selection，下游固定 LinearRegression。", "reports/phase3/phase3b_report.md"),
            ("Evaluation", "Training CV、Nested CV、固定 Holdout 及單筆觀測 PI 分別評估。", "reports/phase3/phase3c_report.md"),
            ("Deployment", "已有本機 Python 預測流程與成果 CLI；網站及 API 尚未建立，未正式部署。", "src/phase3c_evaluation.py"),
        ]
        for _, _, path in stages:
            self.check_hash(path)
        return {"stages": [{"name": name, "summary": summary, "evidence": path}
                           for name, summary, path in stages]}

    def document(self, name):
        self.sources = {}
        builder = getattr(self, name)
        try:
            data = builder()
            units = dict(METRIC_UNITS)
            units.update({"coverage": "fraction [0,1]", "interval_width": "Wh",
                          "runtime_seconds": "s", "feature_count": "count; selection k counts transformed numeric columns"})
            missing = (["six_method_holdout_comparison：既有結果僅評估 Baseline 與 Selected 的 Holdout。"]
                       if name == "feature_selection" else [])
            result = {"schema_version": 1, "artifact_type": name,
                      "sources": sorted(self.sources.values(), key=lambda x: x["path"]),
                      "units": units, "missing_fields": missing, "data": data}
            validate_document(result)
            return result
        except (KeyError, TypeError, OSError, ValueError) as error:
            if isinstance(error, ResultError):
                raise
            raise ResultError(f"{name} 結果格式或檔案存取錯誤：{error}") from error

    def verify(self):
        documents = {name: self.document(name) for name in EXPORT_NAMES}
        dataset = documents["dataset_summary"]["data"]["dataset"]
        selection = documents["feature_selection"]["data"]
        comparison = documents["model_comparison"]["data"]
        intervals = documents["prediction_intervals"]["data"]
        if selection["experiment"]["config"]["dataset_sha256"] != dataset["sha256"]:
            raise ResultError("Feature Selection 與 Dataset 版本不一致。")
        if intervals["plan"] != comparison["plan"]:
            raise ResultError("Prediction Interval 與模型評估的切分設定不一致。")
        winner = next(row for row in selection["methods"] if row["method"] == selection["winner"]["method"])
        refit = winner["full_training_refit"]
        selected = comparison["models"]["selected"]
        mapping = required(required(refit, "selection", dict), "original_feature_mapping", dict)
        if (selected["feature_count"] != refit["selected_k"]
                or selected["selected_features"] != [name for name, names in mapping.items() if names]
                or any(selected["training_metrics"][key] != refit[key] for key in ("mae", "rmse", "r2"))):
            raise ResultError("Selected 模型與 Training CV 選定的完整訓練區紀錄不一致。")
        phase1 = validate(self.path(dataset["csv_path"]))  # 現有唯讀驗證函式，不寫報告。
        if phase1["overall_status"] == "FAIL":
            raise ResultError("Phase 1 原始 Dataset 驗證失敗。")
        evidence = {}
        for path in ("reports/phase1/data_validation.json", "reports/phase2/notebook_sync_verification.json",
                     "reports/phase3/verification.json", "reports/phase3/phase3b_verification.json",
                     "reports/phase3/phase3c/phase3c_verification.json"):
            evidence[path] = self.check_hash(path)
        for path in ("reports/phase2/models/random_baseline.joblib",
                     "reports/phase2/predictions_chronological.csv", "reports/phase2/predictions_random.csv"):
            evidence[path] = self.check_hash(path)
        counts = {status: sum(c["status"] == status for c in phase1["checks"])
                  for status in ("PASS", "WARN", "FAIL")}
        return {"status": "PASS", "dataset_sha256": dataset["sha256"],
                "phase1_validation": {"overall_status": phase1["overall_status"], "checks": counts},
                "required_artifacts": "present; dashboard source schemas and recorded hashes checked",
                "figure_count": len(documents["figures_manifest"]["data"]["figures"]),
                "additional_files": evidence,
                "scope": "結果介接驗證；未重新訓練、計算實驗成績或重新執行完整測試。"}

    def export_dashboard(self):
        # 先載入並驗證所有輸入，再檢查全部輸出路徑；失敗時不產生半套新資料。
        self.verify()
        documents = {name: self.document(name) for name in EXPORT_NAMES}
        contents = {name: json.dumps(doc, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
                    for name, doc in documents.items()}
        directory = self.root / "dashboard_data"
        if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
            raise ResultError("dashboard_data 必須為 Repository 內的一般目錄。")
        for name in EXPORT_NAMES:
            destination = directory / f"{name}.json"
            if destination.is_symlink() or (destination.exists() and not destination.is_file()):
                raise ResultError(f"拒絕不安全的匯出目的地：dashboard_data/{name}.json")
        directory.mkdir(exist_ok=True)
        for name, content in contents.items():
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                                 dir=directory, prefix=".export-", delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(content)
                os.replace(temporary, directory / f"{name}.json")
            finally:
                if temporary is not None and temporary.exists():
                    temporary.unlink()
        return {"status": "PASS", "files": [f"dashboard_data/{name}.json" for name in EXPORT_NAMES]}
