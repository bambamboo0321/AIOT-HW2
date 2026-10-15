"""Phase 2：固定 25 個特徵的兩種 LinearRegression Baseline。"""

import hashlib
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.phase2_common import (
    BASELINE_FEATURES, MODEL_DIR, REPORT_DIR, ROOT, TARGET,
    configure_matplotlib, dataset_information, environment_versions,
    load_dataset, prepare_features, save_figure, write_json,
)


def create_splits(frame: pd.DataFrame) -> dict:
    indices = frame.index.to_numpy()
    boundary = int(len(frame) * 0.8)
    random_train, random_test = train_test_split(indices, test_size=0.2, random_state=42, shuffle=True)
    splits = {"chronological": (indices[:boundary], indices[boundary:]),
              "random": (random_train, random_test)}
    for name, (train, test) in splits.items():
        if len(np.intersect1d(train, test)) or len(train) + len(test) != len(frame):
            raise ValueError(f"{name} Train／Test 索引必須互斥且覆蓋全部資料。")
    if frame.loc[splits["chronological"][0], "date"].max() >= frame.loc[splits["chronological"][1], "date"].min():
        raise ValueError("時間切分不可讓 Training 日期晚於 Testing 日期。")
    return splits


def split_metadata(frame: pd.DataFrame, name: str, train, test) -> dict:
    def partition(indices):
        dates = frame.loc[indices, "date"]
        target = frame.loc[indices, TARGET]
        return {
            "count": int(len(indices)),
            "date_range": {"start": dates.min().isoformat(), "end": dates.max().isoformat()},
            "row_index_sha256": hashlib.sha256(json.dumps([int(value) for value in indices]).encode()).hexdigest(),
            "target_statistics": {"mean_wh": float(target.mean()),
                                  "std_wh": float(target.std(ddof=1)), "std_ddof": 1},
        }
    return {
        "method": "chronological_first_80_percent" if name == "chronological" else "random_train_test_split",
        "role": "primary_future_holdout" if name == "chronological" else "comparison_only",
        "shuffle": name == "random", "random_state": 42 if name == "random" else None,
        "requested_train_fraction": 0.8, "requested_test_fraction": 0.2,
        "train": partition(train), "test": partition(test),
        "feature_names": BASELINE_FEATURES,
        "row_index_manifest": "reports/phase2/split_indices.json",
    }


def calculate_metrics(actual, predicted) -> dict:
    metrics = {
        "mae": float(mean_absolute_error(actual, predicted)),
        "mse": float(mean_squared_error(actual, predicted)),
        "rmse": float(root_mean_squared_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
    }
    if not np.isfinite(list(metrics.values())).all():
        raise ValueError("Metrics 必須全部為有限數值。")
    return metrics


def fit_baseline(features: pd.DataFrame, target: pd.Series, train, test):
    """只有 Training 的 Target 參與 fit；Testing 只用於 predict。"""
    model = LinearRegression()
    model.fit(features.loc[train], target.loc[train])
    return model, model.predict(features.loc[train]), model.predict(features.loc[test])


def plot_test_evaluation(name: str, actual, predicted) -> list[str]:
    configure_matplotlib()
    import matplotlib.pyplot as plt
    residual = np.asarray(actual) - np.asarray(predicted)
    title = "Chronological holdout" if name == "chronological" else "Random split holdout (comparison only)"
    offset = 8 if name == "chronological" else 11
    figures = []
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(actual, predicted, s=9, alpha=0.25, color="#257caa", label="Test observations")
    minimum = float(min(np.min(actual), np.min(predicted)))
    maximum = float(max(np.max(actual), np.max(predicted)))
    ax.plot([minimum, maximum], [minimum, maximum], linestyle="--", color="#d45738", label="Perfect prediction")
    ax.set(title=f"{title}: actual vs predicted", xlabel="Actual Appliances energy (Wh)", ylabel="Predicted Appliances energy (Wh)")
    ax.legend()
    figures.append(save_figure(fig, f"{offset:02d}_{name}_actual_vs_predicted.png"))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(predicted, residual, s=9, alpha=0.25, color="#257caa", label="Test residuals")
    ax.axhline(0, linestyle="--", color="#d45738", label="Zero residual")
    ax.set(title=f"{title}: residual vs predicted", xlabel="Predicted Appliances energy (Wh)", ylabel="Residual: actual - predicted (Wh)")
    ax.legend()
    figures.append(save_figure(fig, f"{offset+1:02d}_{name}_residual_vs_predicted.png"))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residual, bins=50, color="#257caa", edgecolor="white", label="Test residuals")
    ax.axvline(0, linestyle="--", color="#d45738", label="Zero residual")
    ax.set(title=f"{title}: residual distribution", xlabel="Residual: actual - predicted (Wh)", ylabel="Test observations (count)")
    ax.legend()
    figures.append(save_figure(fig, f"{offset+2:02d}_{name}_residual_distribution.png"))
    return figures


def plot_comparison(results: dict) -> str:
    configure_matplotlib()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    specifications = [("mae", "MAE", "Wh"), ("mse", "MSE", "Wh squared"),
                      ("rmse", "RMSE", "Wh"), ("r2", "R squared", "dimensionless")]
    for ax, (key, label, unit) in zip(axes.flat, specifications):
        positions = np.arange(2)
        for shift, partition, color in [(-0.18, "train", "#96c3d4"), (0.18, "test", "#d45738")]:
            values = [results[name][f"{partition}_metrics"][key] for name in ("chronological", "random")]
            bars = ax.bar(positions + shift, values, width=0.36, label=partition.capitalize(), color=color)
            ax.bar_label(bars, fmt="%.3f" if key == "r2" else "%.1f", padding=3, fontsize=8)
        ax.set_xticks(positions, ["Chronological\n(primary)", "Random\n(comparison)"])
        ax.set(title=label, xlabel="Split strategy", ylabel=f"{label} ({unit})")
        ax.margins(y=0.18)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Fixed LinearRegression baseline: training and testing performance", y=0.995)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return save_figure(fig, "14_split_performance_comparison.png")


def render_evaluation(payload: dict) -> str:
    runs = payload["baselines"]
    metrics_rows = []
    split_rows = []
    prediction_notes = []
    for name, label in (("chronological", "時間切分（主要）"), ("random", "隨機切分（比較）")):
        run = runs[name]
        for partition, partition_label in (("train", "Training"), ("test", "Testing")):
            values = run[f"{partition}_metrics"]
            metrics_rows.append(f"| {label} | {partition_label} | {values['mae']:.6f} | {values['mse']:.6f} | {values['rmse']:.6f} | {values['r2']:.6f} |")
            info = run["split_metadata"][partition]
            split_rows.append(f"| {label} | {partition_label} | {info['count']} | {info['date_range']['start']} | {info['date_range']['end']} |")
        test = run["test_metrics"]
        train = run["train_metrics"]
        prediction_notes.append(f"- {label}：Testing − Training RMSE = {test['rmse']-train['rmse']:.6f} Wh，R² 差 = {test['r2']-train['r2']:.6f}；測試集負預測 {run['prediction_diagnostics']['negative_test_prediction_count']} 筆，殘差平均 {run['prediction_diagnostics']['mean_test_residual_wh']:.6f} Wh。")
    metric_table = "\n".join(metrics_rows)
    split_table = "\n".join(split_rows)
    notes = "\n".join(prediction_notes)
    coefficients = "\n".join(
        f"| {feature} | {runs['chronological']['model_parameters']['coefficients'][feature]:.6f} | {runs['random']['model_parameters']['coefficients'][feature]:.6f} |"
        for feature in BASELINE_FEATURES
    )
    links = "\n".join(f"- [{Path(path).name}](figures/{Path(path).name})" for path in payload["figures"])
    chrono_train_target = runs["chronological"]["split_metadata"]["train"]["target_statistics"]
    chrono_test_target = runs["chronological"]["split_metadata"]["test"]["target_statistics"]
    return f"""# Phase 2：Baseline Model Evaluation

## 設計與資料準備

兩個 Baseline 皆使用原樣的 `LinearRegression()`、相同的 25 個預先定義數值特徵：{', '.join(BASELINE_FEATURES)}。

排除 Target `Appliances`、時間欄位 `date`、隨機且完全相同的 `rv1`／`rv2`。這是預先定義的 Baseline 資料品質處理，不是依 EDA 或模型分數進行 Feature Selection。`date` 保留供切分、樣本追溯與 EDA 使用，沒有衍生模型時間特徵。

**時間切分為主要結果**：原始順序前 80% Training、後 20% Testing，不 Shuffle。隨機切分使用 `train_test_split(test_size=0.2, random_state=42)`，僅供比較。沒有 scaler、缺失值補值器、參數調整、其他模型或依測試分數挑選最終模型。模型只在對應 Training fit；Training／Testing 指標分開計算。

| 切分 | Partition | 筆數 | 最早日期 | 最晚日期 |
| --- | --- | ---: | --- | --- |
{split_table}

完整資料列索引保存在 `split_indices.json`。JSON 記錄 date range、random_state、feature names、模型預設參數與環境版本；不儲存私人本機路徑。Random 的日期範圍重疊是方法本身的結果，不能解讀為未來外推能力。

## 指標與實測結果

MAE 為平均絕對誤差（Wh）；MSE 為平均平方誤差（Wh²）；RMSE 為 MSE 平方根（Wh），對較大的錯誤較敏感；R² 無單位，以上述 partition 的 Target 平均值作為分母基準，可能為負值。R² 不等於分類準確率或節能比例。

| 切分 | Partition | MAE（Wh） | MSE（Wh²） | RMSE（Wh） | R² |
| --- | --- | ---: | ---: | ---: | ---: |
{metric_table}

{notes}

Training 與 Testing 是不同的觀測樣本；差距可觀察擬合與時間分布變化，不能單靠差距證明過擬合。Random 分數受到同一住宅、相鄰時間樣本可能同時落入兩側的影響，不能當成真正的未來預測能力。沒有根據兩個測試集分數反覆調整設定。

時間切分 Training 的 Target 平均／樣本標準差為 {chrono_train_target['mean_wh']:.6f}／{chrono_train_target['std_wh']:.6f} Wh，Testing 為 {chrono_test_target['mean_wh']:.6f}／{chrono_test_target['std_wh']:.6f} Wh。兩個期間的 Target 分布與變異不同，R² 的分母也不同；較低 RMSE 不必然代表較高 R²，不能用不同測試集的分數直接宣稱哪一種切分具有較佳未來預測能力。

## 係數與模型保存

時間切分 Intercept = {runs['chronological']['model_parameters']['intercept']:.6f} Wh；隨機切分 Intercept = {runs['random']['model_parameters']['intercept']:.6f} Wh。係數對應未標準化的原始欄位尺度，不能直接比較絕對值來做特徵重要性排名，也不能宣稱因果效果。

| Feature | 時間切分係數 | 隨機切分係數 |
| --- | ---: | ---: |
{coefficients}

`models/chronological_baseline.joblib`、`models/random_baseline.joblib` 儲存模型及 feature names、intercept、coefficients、sklearn version、training metadata；對應的 `.json` 提供可讀資料。重新載入模型後，可用相同欄位順序預測；joblib 檔案僅供本專案可信來源使用。

測試集逐筆 actual、predicted、residual 與原始 row index／date 保存於 `predictions_chronological.csv`、`predictions_random.csv`。三類評估圖均只使用該模型的 Testing 預測，沒有混入 Training 資料。比較圖明確分開 Training／Testing。

## 發現與限制

Baseline 可能低估極端耗能尖峰；請同時觀察 Actual vs Predicted、殘差分布與殘差隨預測值的型態。所有預測保留 LinearRegression 原始輸出，包括負值，未截斷或依測試集修正。OLS 的線性、加成形式可能無法描述作息、尖峰與非線性關係，輸入間高度相關也可能使係數不穩定。

此流程以**同一時間的感測器與天氣資訊估計當下 Appliances**，不是只靠歷史資料預測未來時點的多步模型；時間 Holdout 僅模擬未見過的後期資料。實際使用仍須確認這些輸入在預測時是否可取得。資料涵蓋單棟住宅約 4.5 個月，測試期間只有後 20%，沒有跨年度、跨住宅或外部驗證；未執行交叉驗證。未建立 Confidence Interval／Prediction Interval，未開始 Phase 3–4。

## 圖表

{links}
"""


def load_verified_baselines(frame: pd.DataFrame | None = None) -> dict:
    """唯讀核對既有模型與 Metrics；供 Notebook 同步，不 fit 或改寫模型。"""
    frame = load_dataset() if frame is None else frame
    payload = json.loads((REPORT_DIR / "baseline_metrics.json").read_text(encoding="utf-8"))
    features, target = prepare_features(frame)
    splits = create_splits(frame)
    if (payload["feature_names"] != BASELINE_FEATURES or payload["baseline_feature_count"] != 25
            or payload["scaling_performed"] or payload["feature_selection_performed"]):
        raise ValueError("既有 Baseline 的特徵／前處理設定不符固定定義。")
    if payload["dataset_information"]["sha256"] != dataset_information(frame)["sha256"]:
        raise ValueError("既有 Baseline 的資料集 SHA-256 不符。")
    for name, (train, test) in splits.items():
        run = payload["baselines"][name]
        artifact = joblib.load(ROOT / run["model_file"])
        model = artifact["model"]
        if (not isinstance(model, LinearRegression) or model.get_params() != LinearRegression().get_params()
                or artifact["feature_names"] != BASELINE_FEATURES
                or list(model.feature_names_in_) != BASELINE_FEATURES):
            raise ValueError(f"{name} 已儲存模型的類型／設定／特徵順序不符。")
        for partition, indices in (("train", train), ("test", test)):
            if run["split_metadata"][partition]["count"] != len(indices):
                raise ValueError(f"{name} {partition} 筆數不一致。")
            measured = calculate_metrics(target.loc[indices], model.predict(features.loc[indices]))
            for metric, actual in measured.items():
                if not np.isclose(actual, run[f"{partition}_metrics"][metric], rtol=1e-12, atol=1e-12):
                    raise ValueError(f"{name} {partition} {metric} 與既有 Metrics 不一致。")
    if any(not (ROOT / path).is_file() for path in payload["figures"]):
        raise ValueError("既有模型評估圖不完整。")
    return payload


def run_baselines(frame: pd.DataFrame | None = None) -> dict:
    frame = load_dataset() if frame is None else frame
    features, target = prepare_features(frame)
    splits = create_splits(frame)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    versions = environment_versions()
    payload = {
        "dataset_information": dataset_information(frame),
        "feature_names": BASELINE_FEATURES, "baseline_feature_count": 25,
        "excluded_columns": list((TARGET, "date", "rv1", "rv2")),
        "feature_selection_performed": False, "scaling_performed": False,
        "environment_versions": versions,
        "metric_units": {"mae": "Wh", "mse": "Wh^2", "rmse": "Wh", "r2": "dimensionless"},
        "baselines": {}, "figures": [],
    }
    manifest = {}
    for name, (train, test) in splits.items():
        manifest[name] = {"train": [int(value) for value in train], "test": [int(value) for value in test]}
        metadata = split_metadata(frame, name, train, test)
        model, train_predicted, test_predicted = fit_baseline(features, target, train, test)
        if not np.isfinite(train_predicted).all() or not np.isfinite(test_predicted).all():
            raise ValueError("模型預測不可包含 NaN 或無窮值。")
        parameters = {
            "estimator": "sklearn.linear_model.LinearRegression",
            "settings": model.get_params(), "intercept": float(model.intercept_),
            "coefficients": dict(zip(BASELINE_FEATURES, map(float, model.coef_))),
            "matrix_rank": int(model.rank_), "singular_values": [float(value) for value in model.singular_],
        }
        artifact = {
            "feature_names": BASELINE_FEATURES,
            "intercept": parameters["intercept"], "coefficients": parameters["coefficients"],
            "sklearn_version": versions["scikit-learn"],
            "training_metadata": {
                "dataset_sha256": payload["dataset_information"]["sha256"],
                "split_method": metadata["method"], "random_state": metadata["random_state"],
                "train": metadata["train"], "feature_count": 25,
                "row_index_manifest": metadata["row_index_manifest"],
                "preprocessing": "fixed column exclusion only; no scaler or imputer",
            },
            "model_parameters": parameters,
        }
        path = MODEL_DIR / f"{name}_baseline.joblib"
        joblib.dump({"model": model, **artifact}, path, compress=3)
        write_json(MODEL_DIR / f"{name}_baseline.json", artifact)
        predictions = pd.DataFrame({
            "row_index": test, "date": frame.loc[test, "date"].to_numpy(),
            "actual_wh": target.loc[test].to_numpy(), "predicted_wh": test_predicted,
            "residual_wh": target.loc[test].to_numpy() - test_predicted,
        }).sort_values("row_index")
        predictions.to_csv(REPORT_DIR / f"predictions_{name}.csv", index=False)
        payload["baselines"][name] = {
            "split_metadata": metadata,
            "train_metrics": calculate_metrics(target.loc[train], train_predicted),
            "test_metrics": calculate_metrics(target.loc[test], test_predicted),
            "model_parameters": parameters,
            "model_file": path.relative_to(ROOT).as_posix(),
            "prediction_file": f"reports/phase2/predictions_{name}.csv",
            "prediction_diagnostics": {
                "negative_test_prediction_count": int((test_predicted < 0).sum()),
                "mean_test_residual_wh": float(predictions["residual_wh"].mean()),
                "residual_definition": "actual - predicted",
            },
        }
        payload["figures"].extend(plot_test_evaluation(name, target.loc[test], test_predicted))
    write_json(REPORT_DIR / "split_indices.json", manifest)
    payload["figures"].append(plot_comparison(payload["baselines"]))
    write_json(REPORT_DIR / "baseline_metrics.json", payload)
    (REPORT_DIR / "model_evaluation.md").write_text(render_evaluation(payload), encoding="utf-8")
    return payload


def main() -> int:
    result = run_baselines()
    for name, run in result["baselines"].items():
        print(name, json.dumps(run["test_metrics"], ensure_ascii=False))
    print("PASS: reports/phase2/baseline_metrics.json；兩個 Baseline 與七張評估圖已儲存。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
