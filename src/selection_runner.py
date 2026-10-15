"""Phase 3A 的 training-only 資料契約與小規模 Nested TimeSeriesSplit runner。"""

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from .data_splitting import assert_aligned, chronological_dates
from .feature_selection import AUTO_METHODS, FIXED_METHODS, SelectionSpec
from .feature_selection import count_fits, fit_selected_regression
from .phase2_common import BASELINE_FEATURES, CSV_PATH, EXPECTED_SHA256, ROOT
from .preprocessing_config import PreprocessingConfig
from .validate_data import sha256_file

PHASE3_DIR = ROOT / "reports/phase3"


@dataclass
class TrainingPartition:
    X: pd.DataFrame
    y: pd.Series
    allowed_indices: tuple
    forbidden_indices: tuple = ()
    source: str = "synthetic_or_explicit_training"

    def validate(self, config):
        assert_aligned(self.X, self.y)
        actual = set(self.X.index)
        if not actual <= set(self.allowed_indices) or actual & set(self.forbidden_indices):
            raise ValueError("Holdout／未授權索引不得進入 Feature Selection 或 Benchmark。")
        if config.split_strategy != "chronological":
            raise ValueError("Phase 3A 必須使用 Chronological / TimeSeriesSplit。")
        dates = chronological_dates(self.X, config)
        if not dates.is_monotonic_increasing:
            raise ValueError("Training 必須依時間遞增，不能隨機重排。")
        return self

    def prefix(self, samples):
        if type(samples) is not int or not 12 <= samples <= len(self.X):
            raise ValueError("samples 必須為 12 到 Training 筆數之間的整數。")
        return TrainingPartition(self.X.iloc[:samples].copy(), self.y.iloc[:samples].copy(),
                                 self.allowed_indices, self.forbidden_indices, self.source)


def load_energy_training():
    """只解析前 15,788 列；不得呼叫會載入全部 Target 的 Phase 2 loader。"""
    if sha256_file(CSV_PATH) != EXPECTED_SHA256:
        raise ValueError("CSV SHA-256 不符 Phase 1。")
    metrics = json.loads((ROOT / "reports/phase2/baseline_metrics.json").read_text())
    meta = metrics["baselines"]["chronological"]["split_metadata"]
    manifest = json.loads((ROOT / "reports/phase2/split_indices.json").read_text())["chronological"]
    if (meta["train"]["count"] != 15788 or meta["test"]["count"] != 3947
            or meta["shuffle"] or metrics["feature_names"] != BASELINE_FEATURES
            or manifest["train"] != list(range(15788)) or manifest["test"] != list(range(15788, 19735))):
        raise ValueError("Phase 2 chronological metadata／索引不符正式契約。")
    frame = pd.read_csv(CSV_PATH, nrows=15788)
    frame["date"] = pd.to_datetime(frame["date"], format="%Y-%m-%d %H:%M:%S", errors="raise")
    if (len(frame) != 15788 or frame.date.iloc[0].isoformat() != meta["train"]["date_range"]["start"]
            or frame.date.iloc[-1].isoformat() != meta["train"]["date_range"]["end"]):
        raise ValueError("Training prefix 與 Phase 2 日期 metadata 不符。")
    config = PreprocessingConfig.load(ROOT / "reports/phase2/energy_config.json")
    if config.feature_columns != BASELINE_FEATURES or config.target_column != "Appliances":
        raise ValueError("必須沿用 Phase 2 的 25 個 Baseline Features。")
    partition = TrainingPartition(frame[BASELINE_FEATURES + ["date"]].copy(), frame.Appliances.copy(),
                                  tuple(manifest["train"]), tuple(manifest["test"]),
                                  "Phase 2 chronological training prefix only")
    return partition.validate(config), config


def regression_metrics(actual, predicted):
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape or actual.ndim != 1 or len(actual) < 2:
        raise ValueError("指標需要至少兩列且形狀一致的一維 actual/predicted。")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("指標不接受 NaN／Infinity。")
    mse = float(mean_squared_error(actual, predicted))
    return {"rmse": float(np.sqrt(mse)), "mae": float(mean_absolute_error(actual, predicted)),
            "r2": float(r2_score(actual, predicted))}


def fold_manifest(X, splits):
    return [{"fold": i, "train_indices": list(X.index[train]), "validation_indices": list(X.index[valid]),
             "train_rows": len(train), "validation_rows": len(valid)}
            for i, (train, valid) in enumerate(splits)]


def run_nested_cv(partition, config, methods, k_grid=(3, 5, 10), *, outer_splits=5,
                  inner_splits=3, gap=6, selector_splits=None, alphas=None, audit=None):
    """小規模框架驗證；沒有完整 Training 最終重訓或 Holdout 評估。

    固定 k 在 Inner CV 以完整 selector+predictor 選 k；SFS 自身的 CV
    也只接收該 Inner Training，因此必要時多一層演算法 CV。
    自動 k 的 LassoCV/RFECV 直接在 Outer Training 使用各自的 Inner CV。
    """
    partition.validate(config)
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("methods 必須非空且不重複。")
    if len(k_grid) > 3:
        raise ValueError("Phase 3A 最多測試三個 k；禁止完整 k=1..25 實驗。")
    if any(type(k) is not int or not 1 <= k <= len(config.feature_columns) for k in k_grid):
        raise ValueError("k_grid 超出轉換後的欄位範圍。")
    if not 2 <= outer_splits <= 5 or not 2 <= inner_splits <= 3:
        raise ValueError("Phase 3A Outer 只能 2..5，Inner 只能 2..3。")
    selector_splits = inner_splits if selector_splits is None else selector_splits
    if not 2 <= selector_splits <= 3:
        raise ValueError("Selector CV folds 只能 2..3。")
    X, y = partition.X, partition.y
    splits = list(TimeSeriesSplit(n_splits=outer_splits, gap=gap).split(X))
    if min(len(valid) for _, valid in splits) < 2:
        raise ValueError("Outer validation 至少需要兩列。")
    output = {"purpose": "phase3a_framework_validation", "source": partition.source,
              "outer_splits": outer_splits, "inner_splits": inner_splits, "selector_splits": selector_splits,
              "gap": gap, "holdout_used": False, "outer_fold_manifest": fold_manifest(X, splits),
              "k_curve_scope": "inner CV tuning scores, never selected using outer validation",
              "final_full_training_refit_performed": False, "methods": {}}
    for method in methods:
        # 初始化時即驗證方法；不因輸入方法不同改變 outer folds。
        base = SelectionSpec(method, k_grid[0] if method in FIXED_METHODS and k_grid else None,
                             inner_splits=selector_splits, gap=gap,
                             **({"alphas": tuple(alphas)} if alphas is not None else {}))
        records = []
        with count_fits() as counts:
            for fold, (train, valid) in enumerate(splits):
                train_X, train_y = X.iloc[train], y.iloc[train]
                valid_X, valid_y = X.iloc[valid], y.iloc[valid]
                chosen = None
                curves = []
                if method in FIXED_METHODS:
                    if not k_grid:
                        raise ValueError("固定 k 方法需要非空 k_grid。")
                    if len(k_grid) == 1:
                        chosen = k_grid[0]
                    else:
                        inner = list(TimeSeriesSplit(n_splits=inner_splits, gap=gap).split(train_X))
                        for k in sorted(set(k_grid)):
                            scores = []
                            for j, (subtrain, subvalid) in enumerate(inner):
                                model = fit_selected_regression(train_X.iloc[subtrain], train_y.iloc[subtrain],
                                                                config, replace(base, k=k), audit,
                                                                f"outer_{fold}/k_{k}/inner_{j}")
                                scores.append(regression_metrics(train_y.iloc[subvalid], model.predict(train_X.iloc[subvalid])))
                            curves.append({"k": k, "mean_inner_rmse": float(np.mean([s["rmse"] for s in scores])),
                                           "fold_metrics": scores})
                        chosen = min(curves, key=lambda item: (item["mean_inner_rmse"], item["k"]))["k"]
                spec = replace(base, k=chosen) if method in FIXED_METHODS else base
                model = fit_selected_regression(train_X, train_y, config, spec, audit, f"outer_{fold}/refit")
                clock = perf_counter()
                metrics = regression_metrics(valid_y, model.predict(valid_X)) if model.predictor is not None else None
                records.append({"outer_fold": fold, "inner_k_curve": curves, "chosen_inner_k": chosen,
                                "selection": model.selection.metadata, "outer_metrics": metrics,
                                "timing": {**model.selection.timing, "predictor_fit_seconds": model.predictor_fit_seconds,
                                           "prediction_and_metrics_seconds": perf_counter() - clock}})
        scored = [record["outer_metrics"] for record in records if record["outer_metrics"] is not None]
        output["methods"][method] = {
            "folds": records, "fit_accounting": dict(counts), "scored_outer_folds": len(scored),
            "status": "ok" if len(scored) == outer_splits else "incomplete_zero_features",
            "mean_outer_metrics": {metric: float(np.mean([score[metric] for score in scored]))
                                   for metric in ("rmse", "mae", "r2")} if len(scored) == outer_splits else None,
            "zero_feature_folds": sum(r["selection"]["selected_k"] == 0 for r in records),
        }
    return output


def training_digest(partition):
    return hashlib.sha256(",".join(map(str, partition.X.index)).encode()).hexdigest()
