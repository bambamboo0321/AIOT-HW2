"""固定 Phase 3C 結果輸出；圖表以輸出 CSV 為資料來源，不覆寫舊圖。"""

import csv
import hashlib
import json

import numpy as np
import pandas as pd

from .phase2_common import ROOT, configure_matplotlib
from .phase3c_evaluation import OUTPUT, FIGURES, PLAN, environment


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(result):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / "prediction_metrics.json", {"evaluation": "fixed chronological holdout, descriptive",
        "metric_implementation": "sklearn.metrics", "metric_units": {"mae": "Wh", "mse": "Wh^2", "rmse": "Wh", "r2": "dimensionless"},
        "environment": environment(), "plan": PLAN, "models": result["model_metrics"],
        "transformation_checks": result["transformation_checks"]})
    write_json(OUTPUT / "prediction_interval_metrics.json", {"plan": PLAN,
        "formula": "saved_prediction +/- t.ppf(0.975, n-rank(X))*sqrt(SSE/(n-rank(X)))*sqrt(1+h0)",
        "leverage": "sum((new_design @ V_retained.T / singular_values_retained)^2, axis=1)",
        "sigma_source": "Training residuals only", "models": result["interval_metrics"]})
    fields = ["Model", "Feature Count", "Selected Features", "MAE", "MSE", "RMSE", "R2",
              "Training Start", "Training End", "Holdout Start", "Holdout End",
              "Nominal Coverage", "Empirical Coverage", "Average Width", "Median Width",
              "Width Standard Deviation", "Below Count", "Above Count", "Holdout Samples"]
    comparison = []
    for name, item in result["model_metrics"].items():
        interval = result["interval_metrics"][name]
        comparison.append({"Model": name, "Feature Count": item["feature_count"],
            "Selected Features": json.dumps(item["selected_features"]),
            **{label: item["holdout_metrics"][key] for label, key in (("MAE", "mae"), ("MSE", "mse"), ("RMSE", "rmse"), ("R2", "r2"))},
            "Training Start": item["training_period"]["start"], "Training End": item["training_period"]["end"],
            "Holdout Start": item["holdout_period"]["start"], "Holdout End": item["holdout_period"]["end"],
            **{label: interval[key] for label, key in (("Nominal Coverage", "nominal_coverage"), ("Empirical Coverage", "empirical_coverage"),
                ("Average Width", "average_width"), ("Median Width", "median_width"), ("Width Standard Deviation", "width_standard_deviation"),
                ("Below Count", "below_lower_count"), ("Above Count", "above_upper_count"), ("Holdout Samples", "total_holdout_samples"))}})
    write_csv(OUTPUT / "prediction_comparison.csv", fields, comparison)
    predictions = []
    for i in range(3947):
        row = {"Row Index": int(result["row_indices"][i]),
               "Timestamp": pd.Timestamp(result["timestamp"][i]).isoformat(), "Actual": result["actual"][i]}
        for name, label in (("baseline", "Baseline"), ("selected", "Selected")):
            interval = result["interval_predictions"][name]
            row.update({f"{label} Prediction": interval["prediction"][i],
                        f"{label} Lower 95": interval["lower"][i], f"{label} Upper 95": interval["upper"][i],
                        f"{label} Leverage": interval["leverage"][i]})
        predictions.append(row)
    write_csv(OUTPUT / "prediction_interval_predictions.csv", list(predictions[0]), predictions)
    make_figures()
    write_report(result)


def make_figures():
    configure_matplotlib()
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    FIGURES.mkdir(parents=True, exist_ok=True)
    predicted_path = OUTPUT / "prediction_interval_predictions.csv"
    comparison_path = OUTPUT / "prediction_comparison.csv"
    predictions = pd.read_csv(predicted_path, float_precision="round_trip")
    metrics = pd.read_csv(comparison_path, float_precision="round_trip").set_index("Model")
    manifest = {"sources": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (predicted_path, comparison_path)},
                "full_evaluation_samples": 3947, "plot_window_decided_before_evaluation": PLAN["plot_window"], "figures": {}}
    labels = {"baseline": "Baseline LR (25 features)", "selected": "MI + LR (6 features)"}
    colors = {"baseline": "#277DA1", "selected": "#D47735"}

    def save(fig, name, source):
        fig.tight_layout()
        fig.savefig(FIGURES / name, dpi=160, facecolor="white", metadata={"Software": "AIOT-HW2 Phase 3C"})
        plt.close(fig)
        manifest["figures"][name] = source

    view = predictions.iloc[:144]
    time = pd.to_datetime(view.Timestamp)
    for name, prefix in (("baseline", "Baseline"), ("selected", "Selected")):
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.fill_between(time, view[f"{prefix} Lower 95"], view[f"{prefix} Upper 95"],
                        color=colors[name], alpha=.2, label="95% OLS prediction interval (observation)")
        ax.plot(time, view.Actual, color="#353535", linewidth=1.1, label="Actual")
        ax.plot(time, view[f"{prefix} Prediction"], color=colors[name], linewidth=1.4, label="Saved model prediction")
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
        ax.set(title=f"{labels[name]}: first 144 consecutive holdout observations",
               xlabel="Dataset timestamp (timezone unspecified)", ylabel="Appliances energy (Wh)")
        ax.grid(alpha=.18)
        ax.legend(fontsize=8, loc="upper right")
        save(fig, f"{name}_prediction_interval.png", {"source": predicted_path.name,
            "holdout_positions": [0, 144], "row_indices": [int(view.iloc[0]["Row Index"]), int(view.iloc[-1]["Row Index"])],
            "columns": ["Timestamp", "Actual", f"{prefix} Prediction", f"{prefix} Lower 95", f"{prefix} Upper 95"],
            "points": {col: view[col].tolist() for col in ("Actual", f"{prefix} Prediction", f"{prefix} Lower 95", f"{prefix} Upper 95")}})
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.4))
    for ax, metric, unit in zip(axes, ("MAE", "MSE", "RMSE", "R2"), ("Wh", "Wh²", "Wh", "dimensionless")):
        bars = ax.bar([labels[n] for n in labels], [metrics.loc[n, metric] for n in labels], color=list(colors.values()), label="Full holdout (n=3947)")
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
        ax.set(title=metric, ylabel=f"{metric} ({unit})", xlabel="Fixed model")
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        ax.margins(y=.2)
        ax.legend(fontsize=7)
    fig.suptitle("Fixed-model prediction performance on the same chronological holdout")
    save(fig, "model_performance_comparison.png", {"source": comparison_path.name,
        "metrics": {n: {m: float(metrics.loc[n, m]) for m in ("MAE", "MSE", "RMSE", "R2")} for n in labels}})
    fig, ax = plt.subplots(figsize=(8, 5))
    coverage = [100 * metrics.loc[n, "Empirical Coverage"] for n in labels]
    bars = ax.bar([labels[n] for n in labels], coverage, color=list(colors.values()), label="Empirical coverage: all 3947 observations")
    ax.bar_label(bars, fmt="%.2f%%", padding=3)
    ax.axhline(95, color="#353535", linestyle="--", label="Nominal 95% (not guaranteed)")
    ax.set(title="Pointwise OLS prediction interval coverage", xlabel="Fixed model", ylabel="Coverage (%)", ylim=(0, 105))
    ax.legend(loc="lower right", fontsize=8)
    save(fig, "interval_coverage_comparison.png", {"source": comparison_path.name,
        "empirical_coverage": {n: float(metrics.loc[n, "Empirical Coverage"]) for n in labels}, "nominal_coverage": .95})
    fig, ax = plt.subplots(figsize=(8, 5))
    positions = np.arange(2)
    mean = [metrics.loc[n, "Average Width"] for n in labels]
    std = [metrics.loc[n, "Width Standard Deviation"] for n in labels]
    median = [metrics.loc[n, "Median Width"] for n in labels]
    ax.bar(positions, mean, yerr=std, capsize=5, color=list(colors.values()), alpha=.85, label="Mean width ± population SD")
    ax.scatter(positions, median, color="#303030", marker="D", zorder=4, label="Median width")
    ax.set(xticks=positions, xticklabels=list(labels.values()), xlabel="Fixed model", ylabel="Interval width (Wh)",
           title="OLS prediction interval width: full chronological holdout")
    ax.legend(fontsize=8)
    save(fig, "interval_width_comparison.png", {"source": comparison_path.name,
        "widths": {n: {m: float(metrics.loc[n, m]) for m in ("Average Width", "Median Width", "Width Standard Deviation")} for n in labels}})
    write_json(OUTPUT / "figure_manifest.json", manifest)


def write_report(result):
    bm, sm = [result["model_metrics"][n]["holdout_metrics"] for n in ("baseline", "selected")]
    intervals = result["interval_metrics"]
    rmse_delta, r2_delta = sm["rmse"] - bm["rmse"], sm["r2"] - bm["r2"]
    effect = "本次 Holdout 上 RMSE 降低且 R² 提升" if rmse_delta < 0 and r2_delta > 0 else "Feature Selection 未改善本次 Holdout 的 RMSE 與 R²"
    if all(sm[m] > bm[m] for m in ("mae", "mse", "rmse")) and r2_delta < 0:
        effect = "MI 模型的 MAE、MSE、RMSE 均增加且 R² 下降；Feature Selection 未改善本次 Holdout 表現"
    nested = json.loads((ROOT / "reports/phase3/results/best_features.json").read_text())
    mi_outer = next(r for r in nested["nested_outer_assessment"] if r["method"] == "mutual_information")
    lines = ["# Phase 3C — Prediction Interval 與固定模型評估", "",
        "本階段比較兩個既有模型，並建立單筆新觀測的 95% OLS Prediction Interval（PI）。不重新選 Features、不 fit 或改寫模型、不依 Holdout 調整設定，也不決定新的最佳模型。回歸沒有一般分類的『準確率』，以下以 MAE／MSE／RMSE／R² 與區間覆蓋率分別描述。", "",
        "## 固定模型、資料與轉換空間", "",
        "Model A 為 Phase 2 chronological Baseline LinearRegression，25 個原始 numeric Features，只有固定欄位排除，沒有縮放或補值。Model B 為 Phase 3B 事先由 Training CV 選定的 MI + LinearRegression，6 個 Features 依序為 T3、T4、T5、RH_6、T8、T9；使用已保存的中位數補值與 StandardScaler，再依已保存 MI support 取欄。不是重新執行 MI，也不是重訓 LR。模型檔與既有係數皆保持不變。", "",
        "使用 Phase 2 固定 manifest，前 15,788 筆 Training、後 3,947 筆 Holdout，保留原始順序。Training：" + result["periods"]["training"]["start"] + " 至 " + result["periods"]["training"]["end"] + "；Holdout：" + result["periods"]["holdout"]["start"] + " 至 " + result["periods"]["holdout"]["end"] + "。資料原始時區不明，不轉換時區。", "",
        "預測公式為 ŷ = β₀ + Σ βⱼ zⱼ；Baseline 的 z 是原始數值欄位，Selected 的 z 是已保存 scaler 轉換後的六欄。`prediction_metrics.json` 保存相同空間的 coefficients、intercept 與特徵順序；手算矩陣乘積已核對 saved-model predict，Training Metrics 已核對 Phase 2／3B 歷史值。", "",
        "## 95% OLS Prediction Interval", "",
        "設 A 為包含首欄 Intercept=1 的 Training Design Matrix，r=rank(A)，df=n−r，SSE=Σ(yᵢ−ŷᵢ)²，σ̂²=SSE/df。單筆新觀測的 PI 為：", "",
        "```text", "ŷ₀ ± t.ppf(0.975, df) × σ̂ × sqrt(1 + h₀)", "h₀ = a₀ᵀ (AᵀA)⁺ a₀", "```", "",
        "以 A=USVᵀ 的 SVD 直接計算 h₀=Σ(a₀ᵀvⱼ/sⱼ)²，不對 AᵀA 做普通 inverse，也不先形成平方條件數的矩陣。有效秩門檻預先固定為 float64 epsilon × max(n,p) × 最大 singular value，與 NumPy 預設 matrix_rank 定義一致。Training 殘差正交性、SVD 重建誤差、leverage trace 與 df 均有驗證。Rank-deficient 的新列若不在可估計 row space，會拒絕給出誤導的區間；零殘差變異亦明確拒絕，不加任意 epsilon 偽造正寬度。", "",
        "這是**單筆觀測值**區間，含額外的 `1`；對條件平均值的 Confidence Interval 只含 sqrt(h₀)，通常較窄。本階段未把平均值 CI 當成 PI，也不是同時涵蓋整條時間軸的 simultaneous band。上下界保留原值，可能低於物理上非負的耗能；未截斷、未依 Holdout 校準。", "",
        "## 同一完整 Holdout 的預測結果", "",
        "所有指標使用完整 3,947 筆及 sklearn.metrics；MSE 單位 Wh²，MAE/RMSE 為 Wh，R² 無單位。", "",
        "| 固定模型 | Features | MAE | MSE | RMSE | R² |", "|---|---:|---:|---:|---:|---:|"]
    for name, label in (("baseline", "Baseline"), ("selected", "MI + LR")):
        item = result["model_metrics"][name]
        metrics = item["holdout_metrics"]
        lines.append(f"| {label} | {item['feature_count']} | {metrics['mae']:.6f} | {metrics['mse']:.6f} | {metrics['rmse']:.6f} | {metrics['r2']:.6f} |")
    lines += ["", f"Selected − Baseline：RMSE 差 {rmse_delta:+.6f} Wh；R² 差 {r2_delta:+.6f}。{effect}。這只是固定兩模型在同一已查看 Holdout 的描述性比較，不依此重新選模。", "",
        "## 區間 Coverage 與寬度", "",
        "Nominal Coverage 固定 95%。Empirical Coverage 以 lower ≤ Actual ≤ upper 的閉區間逐筆計算；寬度標準差採 ddof=0。以下均為完整 Holdout，未用圖表前 144 筆代替。", "",
        "| 模型 | Empirical Coverage | 平均寬度 Wh | 中位寬度 Wh | 寬度 SD Wh | 低於下界 | 高於上界 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name in ("baseline", "selected"):
        item = intervals[name]
        lines.append(f"| {name} | {100*item['empirical_coverage']:.4f}% | {item['average_width']:.6f} | {item['median_width']:.6f} | {item['width_standard_deviation']:.6f} | {item['below_lower_count']} | {item['above_upper_count']} |")
    lines += ["", "95% 是模型假設下的名目機率，並不保證本資料 Empirical Coverage 等於或高於 95%；未依實測 Coverage 更動任何半徑。", "",
        "## 數學與數值驗證", "", "| 模型 | Design 欄數含 Intercept | Rank | df | σ̂ | 有效條件數 |", "|---|---:|---:|---:|---:|---:|"]
    for name in ("baseline", "selected"):
        d = intervals[name]["training_diagnostics"]
        lines.append(f"| {name} | {d['design_columns_including_intercept']} | {d['rank']} | {d['degrees_of_freedom']} | {d['sigma']:.6f} | {d['effective_condition_number']:.6f} |")
    lines += ["", "合成資料以 statsmodels OLS.get_prediction(...).summary_frame() 的 obs_ci_lower／obs_ci_upper 比對，包含完整秩及重複欄 rank-deficient 範例。參考誤差與固定容差記錄於 `phase3c/phase3c_verification.json`；statsmodels 只用於合成測試，正式預測仍由載入的 sklearn LR 提供。", "",
        "## 圖表與可追溯輸出", "",
        "兩張預測曲線在評估前固定顯示 Holdout 前 144 筆連續資料；圖中 Actual、saved prediction 與 95% PI 均來自逐筆 CSV。其餘預測指標、Coverage、寬度比較圖採完整 Holdout。五張圖位於 `figures/phase3c/`，來源 CSV SHA-256 與繪圖資料記錄於 `phase3c/figure_manifest.json`；不覆寫 Phase 3B 圖表。所有新 CSV 為 UTF-8、LF。", ""]
    for name in ("baseline_prediction_interval.png", "selected_prediction_interval.png", "model_performance_comparison.png", "interval_coverage_comparison.png", "interval_width_comparison.png"):
        lines.append(f"- [{name}](figures/phase3c/{name})")
    lines += ["", "## Nested CV、OLS 假設與限制", "",
        f"Phase 3B MI 的 Training 內部 3-fold 選模 RMSE={nested['winner']['rmse']:.6f}；5-fold Nested Outer MI RMSE={mi_outer['rmse']:.6f}、MAE={mi_outer['mae']:.6f}、R²={mi_outer['r2']:.6f}。Nested 各 Fold 的 k 與訓練期間會改變；本階段是固定 k=6、完整 Training 模型，在另一個後期期間評估。切分、Target 變異及程序不同，不能把數值差異解讀為同一無偏估計或任意挑較漂亮的數字。", "",
        "經典 OLS t 區間依賴適當的線性平均結構、誤差獨立、同方差，且常態誤差下有精確的有限樣本解釋。此資料具時間相依、右偏尖峰、可能異方差、分布漂移與多重共線性；高 Coverage 也可能來自過寬的對稱區間，不能單靠 Coverage 認定校準良好。MI support 是由同一 Training Target 選出，這份 plug-in OLS PI 未涵蓋特徵選擇的不確定性；不可宣稱 post-selection 的精確 95% 保證。", "",
        "模型使用當時的感測器資訊估計同時間能耗，不是僅用過去資料做多步預報。Phase 2 已查看這個 Holdout，因此它不是完全未查看的獨立最終測試集；本階段沒有據 Holdout 重選 Features、k、模型或調整 PI，沒有 residual calibration。", "",
        "## Phase 4 準備與停止點", "",
        "可將兩模型固定評估、數學驗證、完整指標與五張圖交由人工驗收，作為後續最終報告的材料。必須保留已查看 Holdout、時間相依、異方差與選模不確定性限制，不能將名目 95% 說成實測保證或跨住宅泛化證據。尚未開始 Phase 4、未製作 PDF、未 Commit／Push 或部署。", "",
        "參考：[statsmodels OLS observation interval 範例](https://www.statsmodels.org/v0.14.6/examples/notebooks/generated/ols.html)、[固定測試依賴 statsmodels 0.14.6](https://pypi.org/project/statsmodels/0.14.6/)。依賴增補於 `requirements-phase3c.txt`，既有 `requirements.txt` 不變。", ""]
    (ROOT / "reports/phase3/phase3c_report.md").write_text("\n".join(lines), encoding="utf-8")
