"""Phase 2 EDA：唯讀、可重現的分布／相關性／時間依賴分析。"""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.phase2_common import (
    TARGET, REPORT_DIR, configure_matplotlib, dataset_information,
    load_dataset, save_figure, write_json,
)


def analyze(frame: pd.DataFrame) -> dict:
    target = frame[TARGET]
    numeric = frame.select_dtypes(include="number")
    correlations = numeric.corr(method="pearson")
    ranking = correlations[TARGET].drop(TARGET).sort_values(key=lambda values: values.abs(), ascending=False)
    feature_correlations = correlations.drop(index=TARGET, columns=TARGET)
    pairs = [
        {"feature_1": name, "feature_2": other, "pearson_r": float(feature_correlations.loc[name, other])}
        for position, name in enumerate(feature_correlations.columns)
        for other in feature_correlations.columns[position + 1:]
        if abs(feature_correlations.loc[name, other]) >= 0.8
    ]
    pairs.sort(key=lambda pair: abs(pair["pearson_r"]), reverse=True)
    q1, q3 = target.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    hourly = frame.groupby(frame["date"].dt.hour)[TARGET].agg(["count", "mean", "median", "std"]).reindex(range(24))
    weekday = frame.groupby(frame["date"].dt.dayofweek)[TARGET].agg(["count", "mean", "median", "std"]).reindex(range(7))
    monthly = frame.groupby(frame["date"].dt.strftime("%Y-%m"))[TARGET].agg(["count", "mean", "median", "std"])
    inputs = numeric.drop(columns=TARGET)
    standardized = (inputs - inputs.mean()) / inputs.std(ddof=0)
    rank = int(np.linalg.matrix_rank(standardized.to_numpy()))
    singular = np.linalg.svd(standardized.to_numpy(), compute_uv=False)
    numerical_summary = {}
    for name in numeric:
        values = numeric[name]
        lo, hi = values.quantile([0.25, 0.75])
        lower_bound, upper_bound = lo - 1.5 * (hi - lo), hi + 1.5 * (hi - lo)
        numerical_summary[name] = {
            **{str(k): float(v) for k, v in values.describe().items()},
            "skewness": float(values.skew()),
            "iqr_flag_count": int(((values < lower_bound) | (values > upper_bound)).sum()),
        }

    def grouped_records(table, key):
        return [{key: int(index), "count": int(row["count"]),
                 "mean_wh": float(row["mean"]), "median_wh": float(row["median"]),
                 "std_wh": float(row["std"])} for index, row in table.iterrows()]

    return {
        "dataset_information": dataset_information(frame),
        "target_statistics": {**{str(name): float(value) for name, value in target.describe().items()},
                              "skewness": float(target.skew()), "excess_kurtosis": float(target.kurt())},
        "outlier_screening": {
            "method": "Tukey 1.5 * IQR descriptive flag; no rows removed",
            "lower_fence_wh": float(lower), "upper_fence_wh": float(upper),
            "below_fence_count": int((target < lower).sum()),
            "above_fence_count": int((target > upper).sum()),
            "flagged_fraction": float(((target < lower) | (target > upper)).mean()),
        },
        "feature_target_correlation_ranking": [
            {"feature": name, "pearson_r": float(value)} for name, value in ranking.items()
        ],
        "highly_correlated_feature_pairs": pairs,
        "high_correlation_threshold_absolute_r": 0.8,
        "rv1_rv2_identical": bool(frame["rv1"].equals(frame["rv2"])),
        "columns": [{"name": str(name), "analysis_dtype": str(frame[name].dtype),
                     "missing_count": int(frame[name].isna().sum()),
                     "unique_count": int(frame[name].nunique())} for name in frame],
        "numerical_summary": numerical_summary,
        "correlation_matrix": correlations.to_dict(),
        "multicollinearity": {
            "standardized_numeric_input_count": len(inputs.columns), "matrix_rank": rank,
            "rank_deficient": rank < len(inputs.columns),
            "condition_number": float(singular[0] / singular[-1]),
            "interpretation": "全資料描述性診斷；rv1/rv2 重複造成秩不足；不是特徵選擇或 VIF。",
        },
        "hourly_pattern": grouped_records(hourly, "hour"),
        "weekday_pattern": grouped_records(weekday, "weekday_monday_0"),
        "monthly_pattern": [{"month": str(index), "count": int(row["count"]),
                             "mean_wh": float(row["mean"]), "median_wh": float(row["median"]),
                             "std_wh": float(row["std"])} for index, row in monthly.iterrows()],
        "autocorrelation": [{"lag_steps": lag, "lag_minutes": lag * 10,
                             **{name: float(frame[name].autocorr(lag))
                                for name in (TARGET, "T1", "T_out", "RH_1")}}
                            for lag in (1, 6, 144, 1008)],
        "analysis_scope": "全資料描述性 EDA；未做正式 Feature Selection，不能用來宣稱泛化能力或因果關係。",
    }


def plot_eda(frame: pd.DataFrame, summary: dict) -> list[str]:
    configure_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    figures = []
    target = frame[TARGET]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(target, bins=50, color="#257caa", edgecolor="white")
    ax.axvline(target.mean(), color="#d45738", linestyle="--", label=f"Mean: {target.mean():.1f} Wh")
    ax.axvline(target.median(), color="#264653", linestyle=":", label=f"Median: {target.median():.1f} Wh")
    ax.set(title="Appliances energy distribution", xlabel="Appliances energy (Wh)", ylabel="Observations (count)")
    ax.legend()
    figures.append(save_figure(fig, "01_appliances_histogram.png"))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.boxplot(target, orientation="horizontal", tick_labels=["Appliances"],
               flierprops={"markersize": 2, "alpha": 0.3}, patch_artist=True,
               boxprops={"facecolor": "#96c3d4"})
    ax.set(title="Appliances energy boxplot (1.5 IQR whiskers)", xlabel="Appliances energy (Wh)", ylabel="Target")
    figures.append(save_figure(fig, "02_appliances_boxplot.png"))

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(frame["date"], target, linewidth=0.45, alpha=0.65, color="#257caa", label="Observed every 10 minutes")
    ax.set(title="Appliances energy over time", xlabel="Observation date (source timezone unspecified)", ylabel="Appliances energy (Wh)")
    ax.legend(loc="upper right")
    fig.autofmt_xdate()
    figures.append(save_figure(fig, "03_appliances_time_series.png"))

    corr = frame.select_dtypes(include="number").corr()
    fig, ax = plt.subplots(figsize=(13, 11))
    image = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(corr)), corr.index, fontsize=8)
    for row in range(len(corr)):
        for column in range(len(corr)):
            value = corr.iloc[row, column]
            if row != column and abs(value) >= 0.8:
                ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(value) > 0.65 else "black")
    ax.set(title="Numerical columns: Pearson correlation (full-data EDA)", xlabel="Numerical column", ylabel="Numerical column")
    fig.colorbar(image, ax=ax, label="Pearson r (dimensionless)", shrink=0.8)
    figures.append(save_figure(fig, "04_numerical_correlation_heatmap.png"))

    ranking = summary["feature_target_correlation_ranking"][::-1]
    names = [entry["feature"] for entry in ranking]
    values = [entry["pearson_r"] for entry in ranking]
    colors = ["#92989d" if name in ("rv1", "rv2") else "#257caa" if value >= 0 else "#d45738"
              for name, value in zip(names, values)]
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.barh(names, values, color=colors)
    ax.axvline(0, color="#333333", linewidth=0.7)
    ax.set(title="Feature vs target correlation, ranked by |r| (EDA only)",
           xlabel="Pearson r with Appliances (dimensionless)", ylabel="Numerical input column")
    ax.legend(handles=[Patch(color="#257caa", label="Positive r"), Patch(color="#d45738", label="Negative r"),
                       Patch(color="#92989d", label="rv1 / rv2 (random variables)")], loc="lower right")
    figures.append(save_figure(fig, "05_feature_target_correlation_ranking.png"))

    hourly = summary["hourly_pattern"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot([entry["hour"] for entry in hourly], [entry["mean_wh"] for entry in hourly],
            marker="o", label="Mean per observation", color="#257caa")
    ax.plot([entry["hour"] for entry in hourly], [entry["median_wh"] for entry in hourly],
            marker="s", linestyle="--", label="Median per observation", color="#d45738")
    ax.set_xticks(range(24))
    ax.set(title="Appliances energy by hour of day (full-data EDA)", xlabel="Hour of day (0-23)", ylabel="Appliances energy per observation (Wh)")
    ax.legend()
    figures.append(save_figure(fig, "06_hourly_appliances_pattern.png"))

    weekday = summary["weekday_pattern"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(7), [entry["mean_wh"] for entry in weekday], label="Mean per observation", color="#257caa")
    ax.set_xticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set(title="Appliances energy by weekday (full-data EDA)", xlabel="Weekday (Monday = 0)", ylabel="Mean Appliances energy per observation (Wh)")
    ax.legend()
    figures.append(save_figure(fig, "07_weekday_appliances_pattern.png"))

    numeric = frame.select_dtypes(include="number")
    fig, axes = plt.subplots(7, 4, figsize=(16, 22))
    for ax, name in zip(axes.flat, numeric.columns):
        ax.hist(numeric[name], bins=35, color="#257caa", edgecolor="white")
        ax.set(title=name, ylabel="Count", xlabel="Source units")
    fig.tight_layout()
    figures.append(save_figure(fig, "15_numerical_distributions.png"))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    monthly = summary["monthly_pattern"]
    axes[0].plot([r["month"] for r in monthly], [r["mean_wh"] for r in monthly], marker="o", label="Mean")
    axes[0].plot([r["month"] for r in monthly], [r["median_wh"] for r in monthly], marker="s", label="Median")
    axes[0].set(title="Monthly observations (partial Jan / May)", xlabel="Month", ylabel="Energy per observation (Wh)")
    axes[0].legend()
    lags = range(1, 289)
    axes[1].plot(list(lags), [target.autocorr(lag) for lag in lags], color="#257caa")
    axes[1].set(title="Appliances sample autocorrelation", xlabel="Lag (10-minute steps)", ylabel="Pearson autocorrelation")
    axes[1].axhline(0, color="gray", linewidth=0.7)
    fig.tight_layout()
    figures.append(save_figure(fig, "16_monthly_and_autocorrelation.png"))
    return figures


def render_summary(summary: dict) -> str:
    stats = summary["target_statistics"]
    flags = summary["outlier_screening"]
    hourly = summary["hourly_pattern"]
    weekday = summary["weekday_pattern"]
    peak_hour = max(hourly, key=lambda row: row["mean_wh"])
    low_hour = min(hourly, key=lambda row: row["mean_wh"])
    peak_day = max(weekday, key=lambda row: row["mean_wh"])
    low_day = min(weekday, key=lambda row: row["mean_wh"])
    day_names = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    ranking_rows = "\n".join(f"| {row['feature']} | {row['pearson_r']:.6f} |" for row in summary["feature_target_correlation_ranking"])
    pair_rows = "\n".join(f"| {row['feature_1']} | {row['feature_2']} | {row['pearson_r']:.6f} |" for row in summary["highly_correlated_feature_pairs"])
    links = "\n".join(f"- [{Path(path).name}](figures/{Path(path).name})" for path in summary["figures"])
    return f"""# Phase 2：Exploratory Data Analysis

本分析描述全部 19,735 筆資料；沒有依整份資料的 Target 相關係數進行正式 Feature Selection。未來 holdout 也包含在描述性 EDA 中，因此這些觀察不能作為調參或泛化能力證據。所有原始資料完整保留；本次沒有訓練或評估正式模型。

## Target 分布與異常值候選

`Appliances` 為能源讀值，單位 Wh。平均 {stats['mean']:.3f} Wh、中位數 {stats['50%']:.3f} Wh、樣本標準差 {stats['std']:.3f} Wh，最小 {stats['min']:.3f} Wh、最大 {stats['max']:.3f} Wh。偏態 {stats['skewness']:.6f}、超額峰度 {stats['excess_kurtosis']:.6f}；平均高於中位數且右側長尾，少數高讀值可能對平方誤差造成較大影響。

以 1.5 × IQR 描述性規則標記：下界 {flags['lower_fence_wh']:.3f} Wh、上界 {flags['upper_fence_wh']:.3f} Wh；下界外 {flags['below_fence_count']} 筆、上界外 {flags['above_fence_count']} 筆，合計占 {flags['flagged_fraction']:.2%}。這些是需要理解的高低值，不能直接認定為測量錯誤；未刪除、截尾或補值。

## 數值相關係數

Heatmap 包含 Target 與所有數值輸入欄位。下表依與 Target 的 Pearson |r| 排序，保留 r 的正負號；只描述線性關聯，不代表因果關係或選模結果。

| 欄位 | Pearson r 與 Appliances |
| --- | ---: |
{ranking_rows}

輸入欄位間 |r| ≥ 0.8 的配對共 {len(summary['highly_correlated_feature_pairs'])} 組，可能使迴歸係數不穩定，不能把單一係數解讀成因果效果。

| 特徵 1 | 特徵 2 | Pearson r |
| --- | --- | ---: |
{pair_rows}

`rv1`、`rv2` 的全部值完全相同；[UCI 原始欄位定義](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction) 說明它們是加入資料供回歸與非預測屬性檢查的隨機變數。可作負對照，但不能假設有限樣本相關性恰為零。建議可覆寫地排除兩欄，避免重複欄位與無物理意義的輸入；不是演算法優劣結論。

標準化的 27 個數值輸入矩陣秩為 {summary['multicollinearity']['matrix_rank']}，condition number 為 {summary['multicollinearity']['condition_number']:.3e}。此診斷未計算 VIF；秩不足與高度相關可能導致係數不唯一或不穩定。LinearRegression 可用最小平方求解，但單一係數解讀需保守。

## 各欄分布與資料品質

各欄型態、缺失值、唯一值數量，以及 28 個數值欄位的 count、mean、std、min、四分位數、max、skewness、IQR 候選數都存於 `eda_statistics.json`。原始 CSV 的 `date` 為字串，此處只在記憶體解析成 datetime。逐欄直方圖見第 15 張圖；不同量綱與偏態支持將縮放設為可設定選項，沒有預先修改原始值。

## 時間與消耗模式

全資料按小時分組後，平均每筆讀值最高為 {peak_hour['hour']:02d}:00（{peak_hour['mean_wh']:.3f} Wh），最低為 {low_hour['hour']:02d}:00（{low_hour['mean_wh']:.3f} Wh）。按星期分組後，最高為 {day_names[peak_day['weekday_monday_0']]}（{peak_day['mean_wh']:.3f} Wh），最低為 {day_names[low_day['weekday_monday_0']]}（{low_day['mean_wh']:.3f} Wh）。分組的 mean、median、std 與樣本數保存在 `eda_statistics.json`。

小時與星期圖表示**該時段觀測值的平均／中位數**，不是每小時或每日總耗能，也不是功率 W。未進行時區推定或季節調整。資料只有一棟住宅約 4.5 個月，時段差異可能同時受到作息、天氣與季節影響；不能推定因果或一般住宅的節能效果。

月份 count／mean／median／std 及 Target、T1、T_out、RH_1 在 10 分鐘、1 小時、1 天、1 週的自相關列於 JSON；1 月與 5 月不完整，不宜直接解讀成完整月份總耗能。Target lag-1 自相關為 {summary['autocorrelation'][0]['Appliances']:.6f}，顯示相鄰觀測非獨立。建議 chronological holdout，加上訓練區內 TimeSeriesSplit；gap 為待按預測視窗決定的設定，不由 holdout 分數選取。天氣資料的插值與感測器平滑也可能加強鄰近依賴。

## 圖表

{links}
"""


def run_eda(frame: pd.DataFrame | None = None) -> dict:
    frame = load_dataset() if frame is None else frame
    summary = analyze(frame)
    summary["figures"] = plot_eda(frame, summary)
    write_json(REPORT_DIR / "eda_statistics.json", summary)
    (REPORT_DIR / "eda_summary.md").write_text(render_summary(summary), encoding="utf-8")
    return summary


def main() -> int:
    summary = run_eda()
    print(f"PASS: EDA 完成；{len(summary['figures'])} 張圖，reports/phase2/eda_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
