"""Bounded, explicitly confirmed numeric LR experiment; optional trusted temporary model output."""
import argparse
import hashlib
import json
import sys
import warnings

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .custom_dataset_validation import (_bytes, _document, _frame, _numeric,
                                        validate_custom_dataset, MAX_BYTES)
from .data_splitting import split_dataset, iter_preprocessed_folds, prepare_fold
from .preprocessing_config import PreprocessingConfig
from .feature_selection import SelectionSpec, make_selector, fit_selection

from .custom_dataset_selectors import (AUTO_METHODS, NESTED_METHODS, CustomSelectionError, EmptySelectionError, check_nested_request, fit_custom_selection)

SCOPE = "custom_baseline_training"
MAX_RFE_WORK = 100_000_000


class RfeSettingsError(ValueError):
    pass


def metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if not np.isfinite(predicted).all():
        raise ValueError("模型產生非有限預測，請檢查數值尺度。")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
        mae = float(mean_absolute_error(actual, predicted))
        r2 = None if len(actual) < 2 or np.ptp(actual) == 0 else float(r2_score(actual, predicted, force_finite=False))
    if not np.isfinite([rmse, mae]).all() or (r2 is not None and not np.isfinite(r2)):
        raise ValueError("指標無法以有限數值計算，請檢查資料尺度。")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def summary(folds, name):
    result = {}
    for metric in ("rmse", "mae", "r2"):
        values = [f[name][metric] for f in folds if f[name][metric] is not None]
        result[metric] = {"mean": float(np.mean(values)) if values else None,
                          "std": float(np.std(values, ddof=0)) if values else None,
                          "defined_folds": len(values)}
    return result


def selection_settings(experiment, config, validation):
    payload = experiment.get("selection", {"method": "none", "k": None})
    if not isinstance(payload, dict) or set(payload) != {"method", "k"}:
        raise ValueError("selection 必須只包含 method 與 k。")
    method, k = payload["method"], payload["k"]
    if method not in {"none", "f_test", "mutual_information", "rfe", "forward_sfs", "rfecv", "lasso_cv"}:
        raise ValueError("未知特徵選擇方法。")
    if method == "none" or method in AUTO_METHODS:
        if k is not None:
            raise ValueError("不選欄或自動選 k 時，k 必須為 null。")
    elif type(k) is not int or not 1 <= k <= len(config.feature_columns):
        raise ValueError("k 必須是 1 到候選特徵數之間的整數。")
    elif k < len(config.feature_columns) and min(f["training_rows"] for f in validation["split"]["cv_folds"]) < 4:
        raise ValueError("選特徵需要每個 CV Training 至少 4 筆，請減少 folds／gap 或增加資料。")
    if method == "rfe":
        if config.numerical_scaling != "standard":
            raise RfeSettingsError("RFE 需要 StandardScaler；請調整設定、確認並重新驗證。")
        p = len(config.feature_columns)
        fits = p-k+2 if k < p else 1
        rows = validation["split"]["training_rows"] + sum(f["training_rows"] for f in validation["split"]["cv_folds"])
        if p > 25 or validation["dataset"]["rows"] > 5000 or fits*p*rows > MAX_RFE_WORK:
            raise RfeSettingsError("RFE 限 5,000 筆、25 個候選欄位及有限運算量；請減少資料／特徵／folds 或增加 k。")
    check_nested_request(method, k, config, validation)
    return {"method": method, "k": k}


def fit_models(fold, config, selection, raw_training=None):
    """Reuse this fold's fitted preprocessing; the selector sees its training only."""
    p = len(config.feature_columns)
    if fold.X_train.shape[1] != p or any(len(fold.summary["original_to_transformed"][c]) != 1 for c in config.feature_columns):
        raise ValueError("目前選特徵只接受一對一的 numeric 欄位。")
    indices = np.arange(p)
    details = None
    baseline = LinearRegression().fit(fold.X_train, fold.y_train)
    if selection["method"] in NESTED_METHODS or (selection["method"] != "none" and selection["k"] < p):
        spec = SelectionSpec(method=selection["method"], k=selection["k"], random_state=config.random_state, gap=config.cv_gap)
        if selection["method"] in NESTED_METHODS:
            if raw_training is None or not raw_training.index.equals(fold.y_train.index):
                raise ValueError("缺少對齊的原始 Training。")
            fitted = fit_custom_selection(raw_training, fold.y_train, config, spec)
            indices = fitted.indices
            details = fitted.metadata["custom_details"]
            if not len(indices):
                raise EmptySelectionError("LassoCV 在某個 Training 區選出 0 個特徵；本次不產生 LR、Holdout 或模型。請查看自動選 k 比較的逐 fold 結果，或改用其他方法。")
        elif selection["method"] == "rfe":
            # Exactly the same raw Training-only selector path as the k curves.
            # Never apply NumericFoldRegressor to an already clipped/scaled matrix.
            if raw_training is None or not raw_training.index.equals(fold.y_train.index):
                raise ValueError("RFE 缺少對齊的原始 fold Training。")
            indices = fit_selection(raw_training, fold.y_train, config, spec).indices
        else:
            selector = make_selector(spec, config).fit(fold.X_train, fold.y_train)
            indices = selector.get_support(indices=True)
        if selection["method"] not in AUTO_METHODS and len(indices) != selection["k"]:
            raise ValueError("Selector 回傳欄位數不符。")
        active = baseline if len(indices) == p else LinearRegression().fit(fold.X_train[:, indices], fold.y_train)
    else:
        active = baseline
    return active, baseline, indices, details


def train_custom_dataset(csv_path, config_path, review_path, experiment_path, model_output=None):
    started = False
    stage = "confirmation"
    common = {"schema_version": 1, "scope": SCOPE}
    try:
        experiment = _document(experiment_path, 4096)
        keys = {"schema_version", "method", "confirmed", "dataset_sha256", "config_sha256"}
        if (not isinstance(experiment, dict) or set(experiment) not in (keys, keys | {"selection"})
                or type(experiment.get("schema_version")) is not int or experiment["schema_version"] != 1
                or experiment.get("method") != "linear_regression" or experiment.get("confirmed") is not True):
            raise ValueError("請明確確認固定 Linear Regression 訓練；不接受其他方法或額外執行參數。")
        validation = validate_custom_dataset(csv_path, config_path, review_path)
        if validation["status"] != "valid":
            return {**common, "status": "rejected", "training_started": False, "errors": validation["errors"]}
        if (not validation["review_checked"] or experiment["dataset_sha256"] != validation["dataset"]["sha256"]
                or experiment["config_sha256"] != validation["config_sha256"]):
            raise ValueError("訓練確認與 CSV／設定版本不一致，請重新驗證並確認。")
        stage = "confirmation"
        config = PreprocessingConfig.from_dict(_document(config_path, 128 * 1024))
        selection = selection_settings(experiment, config, validation)
        stage = "data"
        normalized = json.dumps(config.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if hashlib.sha256(normalized.encode()).hexdigest() != validation["config_sha256"]:
            raise ValueError("設定版本已改變。")
        raw = _bytes(csv_path, MAX_BYTES, ".csv")
        if hashlib.sha256(raw).hexdigest() != validation["dataset"]["sha256"]:
            raise ValueError("讀取期間 CSV 版本改變，請重新送出。")
        frame = _frame(raw)
        for name in set(config.feature_columns) | {config.target_column}:
            frame[name] = _numeric(frame[name])
        split = split_dataset(frame, config)
        stage = "training"
        started = True
        folds, notes, fold_features = [], list(validation["warnings"]), []
        if selection["method"] != "none" and selection["k"] == len(config.feature_columns):
            notes.append("k 等於候選特徵數，本次保留全部欄位，未執行特徵排名。")
        # Each iterator step creates and fits a new preprocessor on CV training only.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for i, fold in enumerate(iter_preprocessed_folds(split.X_train, split.y_train, config), 1):
                lr, baseline, indices, details = fit_models(fold, config, selection, split.X_train.loc[fold.y_train.index])
                fold_features.append({"fold": i, "features": [config.feature_columns[int(j)] for j in indices], **({"details": details} if details else {})})
                mean = DummyRegressor(strategy="mean").fit(fold.X_train, fold.y_train)
                folds.append({"fold": i, "training_rows": len(fold.y_train), "validation_rows": len(fold.y_validation),
                              "linear_regression": metrics(fold.y_validation, lr.predict(fold.X_validation[:, indices])),
                              "all_features": metrics(fold.y_validation, baseline.predict(fold.X_validation)),
                              "training_mean": metrics(fold.y_validation, mean.predict(fold.X_validation))})
                notes.extend(fold.summary["warnings"])
            # Final fit sees the entire outer Training, never Holdout labels.
            final = prepare_fold(split.X_train, split.y_train, split.X_test, split.y_test, config)
            lr, baseline, indices, details = fit_models(final, config, selection, split.X_train.loc[final.y_train.index])
            mean = DummyRegressor(strategy="mean").fit(final.X_train, final.y_train)
            predictions = lr.predict(final.X_validation[:, indices])
            baseline_predictions = baseline.predict(final.X_validation)
            references = mean.predict(final.X_validation)
            holdout = {"rows": len(final.y_validation), "linear_regression": metrics(final.y_validation, predictions),
                       "training_mean": metrics(final.y_validation, references),
                       "all_features": metrics(final.y_validation, baseline_predictions)}
            notes.extend(final.summary["warnings"])
            if caught:
                notes.append("前處理或線性求解器發出數值警告；請檢查尺度、共線性與結果穩定性。")
        if selection["method"] == "lasso_cv" and details["alpha"] in (details["alpha_grid"][0], details["alpha_grid"][-1]):
            notes.append("LassoCV 選到固定 alpha 候選範圍端點；本次只代表此範圍內的選擇，不保證涵蓋最佳正則化強度。")
        if lr.rank_ < len(indices):
            notes.append("完整 Training 的輸入矩陣不滿秩，部分欄位可能共線或為常數；係數解讀需小心。")
        if holdout["linear_regression"]["r2"] is None:
            notes.append("Holdout 目標為常數或筆數不足，R² 無法定義；RMSE／MAE 仍按全部測試筆數計算。")
        preview = []
        for i, (index, actual) in enumerate(final.y_validation.iloc[:144].items()):
            preview.append({"position": i + 1, "source_row": int(index) + 1,
                            "timestamp": str(split.X_test.loc[index, config.split_column]) if config.split_strategy == "chronological" else None,
                            "actual": float(actual), "prediction": float(predictions[i]), "baseline_prediction": float(baseline_predictions[i]), "training_mean": float(references[i])})
        if model_output is not None:
            from .custom_dataset_prediction import save_model_bundle
            save_model_bundle(model_output, final, lr, indices, split.X_train, config, validation)
        return {**common, "status": "completed", "training_started": True, "validation": validation,
                "method": "linear_regression", "target": config.target_column, "features": list(config.feature_columns),
                "selection": {"method": selection["method"], "requested_k": selection["k"],
                              "ranking_performed": not details["search_bypassed"] if details else selection["method"] != "none" and selection["k"] < len(config.feature_columns),
                              "final_features": [config.feature_columns[int(j)] for j in indices], "fold_features": fold_features, **({"details": details} if details else {})},
                "cv": {"folds": folds, "linear_regression": summary(folds, "linear_regression"), "training_mean": summary(folds, "training_mean"), "all_features": summary(folds, "all_features")},
                "holdout": holdout, "preview": preview, "warnings": list(dict.fromkeys(notes)),
                "limits": ["方法由使用者指定；固定方法指定 k，RFECV／LassoCV 只在 Training 內部 3-fold CV 自動選 k／alpha。最終預測器仍為獨立 LR。",
                           "外層 CV 為 Training 區內的逐 fold 評估；SFS／RFECV／LassoCV 在每個外層 Training 另做內部 3-fold 選欄。平均／標準差為各 fold 等權計算，標準差 ddof=0。",
                           "Holdout 未用於 fit、前處理學習或參數選擇；查看後再修改設定會造成測試集重用。",
                           "前處理與 Selector 只在當前 Training fit；目標維持原單位，未裁切預測或極端測試資料。",
                           "這是同列目標估計；時間切分不代表已完成多步未來預報。",
                           "沒有 Prediction Interval。若由本機網站執行，最終模型可暫存供單筆預測；預覽僅前 144 筆，指標使用全部 Holdout。"]}
    except (ValueError, TypeError, KeyError, OverflowError, FloatingPointError, OSError, np.linalg.LinAlgError) as error:
        # Never expose unexpected raw cells, filesystem paths or solver traceback.
        messages = {"confirmation": "訓練確認、特徵選擇方法／k 無效或版本不一致；請檢查 k 範圍與每 fold 至少 4 筆，再重新確認。",
                    "data": "訓練資料無法讀取或與確認版本不一致。",
                    "training": "前處理或線性迴歸無法完成，請檢查資料尺度、缺失與切分設定。"}
        return {**common, "status": "failed" if started else "rejected", "training_started": started,
                "errors": [{"stage": stage, "message": str(error) if isinstance(error, (RfeSettingsError, CustomSelectionError)) else messages[stage]}]}


def main(argv=None):
    parser = argparse.ArgumentParser(description="明確確認後執行隔離的數值型 LR 實驗，預設不保存模型。")
    for name in ("csv", "config", "review", "experiment"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--model-output", help="內部可信任的暫存輸出路徑；不覆寫既有檔案。")
    args = parser.parse_args(argv)
    result = train_custom_dataset(args.csv, args.config, args.review, args.experiment, args.model_output)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    sys.exit(main())
