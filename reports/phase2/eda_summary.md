# Phase 2：Exploratory Data Analysis

本分析描述全部 19,735 筆資料；沒有依整份資料的 Target 相關係數進行正式 Feature Selection。未來 holdout 也包含在描述性 EDA 中，因此這些觀察不能作為調參或泛化能力證據。所有原始資料完整保留；本次沒有訓練或評估正式模型。

## Target 分布與異常值候選

`Appliances` 為能源讀值，單位 Wh。平均 97.695 Wh、中位數 60.000 Wh、樣本標準差 102.525 Wh，最小 10.000 Wh、最大 1080.000 Wh。偏態 3.386367、超額峰度 13.667863；平均高於中位數且右側長尾，少數高讀值可能對平方誤差造成較大影響。

以 1.5 × IQR 描述性規則標記：下界 -25.000 Wh、上界 175.000 Wh；下界外 0 筆、上界外 2138 筆，合計占 10.83%。這些是需要理解的高低值，不能直接認定為測量錯誤；未刪除、截尾或補值。

## 數值相關係數

Heatmap 包含 Target 與所有數值輸入欄位。下表依與 Target 的 Pearson |r| 排序，保留 r 的正負號；只描述線性關聯，不代表因果關係或選模結果。

| 欄位 | Pearson r 與 Appliances |
| --- | ---: |
| lights | 0.197278 |
| RH_out | -0.152282 |
| T2 | 0.120073 |
| T6 | 0.117638 |
| T_out | 0.099155 |
| RH_8 | -0.094039 |
| Windspeed | 0.087122 |
| RH_1 | 0.086031 |
| T3 | 0.085060 |
| RH_6 | -0.083178 |
| RH_2 | -0.060465 |
| RH_7 | -0.055642 |
| T1 | 0.055447 |
| RH_9 | -0.051462 |
| T4 | 0.040281 |
| T8 | 0.039572 |
| RH_3 | 0.036292 |
| Press_mm_hg | -0.034885 |
| T7 | 0.025801 |
| T5 | 0.019760 |
| RH_4 | 0.016965 |
| Tdewpoint | 0.015353 |
| rv1 | -0.011145 |
| rv2 | -0.011145 |
| T9 | 0.010010 |
| RH_5 | 0.006955 |
| Visibility | 0.000230 |

輸入欄位間 |r| ≥ 0.8 的配對共 36 組，可能使迴歸係數不穩定，不能把單一係數解讀成因果效果。

| 特徵 1 | 特徵 2 | Pearson r |
| --- | --- | ---: |
| rv1 | rv2 | 1.000000 |
| T6 | T_out | 0.974787 |
| T7 | T9 | 0.944776 |
| T5 | T9 | 0.911055 |
| T3 | T9 | 0.901324 |
| RH_3 | RH_4 | 0.898978 |
| RH_4 | RH_7 | 0.894301 |
| T1 | T3 | 0.892402 |
| T4 | T9 | 0.889439 |
| T3 | T5 | 0.888169 |
| T1 | T5 | 0.885247 |
| RH_7 | RH_8 | 0.883984 |
| T7 | T8 | 0.882123 |
| RH_1 | RH_4 | 0.880359 |
| T4 | T7 | 0.877763 |
| T1 | T4 | 0.877001 |
| T4 | T5 | 0.871813 |
| T5 | T7 | 0.870624 |
| T8 | T9 | 0.869338 |
| RH_7 | RH_9 | 0.858686 |
| RH_4 | RH_9 | 0.856591 |
| RH_8 | RH_9 | 0.855812 |
| T3 | T4 | 0.852778 |
| T3 | T7 | 0.847374 |
| RH_4 | RH_8 | 0.847259 |
| T1 | T9 | 0.844777 |
| RH_1 | RH_3 | 0.844677 |
| T1 | T7 | 0.838705 |
| T1 | T2 | 0.836834 |
| RH_3 | RH_9 | 0.833538 |
| RH_3 | RH_7 | 0.832685 |
| RH_3 | RH_8 | 0.828822 |
| T1 | T8 | 0.825413 |
| T5 | T8 | 0.824981 |
| T2 | T6 | 0.801186 |
| RH_1 | RH_7 | 0.801122 |

`rv1`、`rv2` 的全部值完全相同；[UCI 原始欄位定義](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction) 說明它們是加入資料供回歸與非預測屬性檢查的隨機變數。可作負對照，但不能假設有限樣本相關性恰為零。建議可覆寫地排除兩欄，避免重複欄位與無物理意義的輸入；不是演算法優劣結論。

標準化的 27 個數值輸入矩陣秩為 26，condition number 為 5.463e+15。此診斷未計算 VIF；秩不足與高度相關可能導致係數不唯一或不穩定。LinearRegression 可用最小平方求解，但單一係數解讀需保守。

## 各欄分布與資料品質

各欄型態、缺失值、唯一值數量，以及 28 個數值欄位的 count、mean、std、min、四分位數、max、skewness、IQR 候選數都存於 `eda_statistics.json`。原始 CSV 的 `date` 為字串，此處只在記憶體解析成 datetime。逐欄直方圖見第 15 張圖；不同量綱與偏態支持將縮放設為可設定選項，沒有預先修改原始值。

## 時間與消耗模式

全資料按小時分組後，平均每筆讀值最高為 18:00（190.365 Wh），最低為 03:00（48.236 Wh）。按星期分組後，最高為 星期一（111.451 Wh），最低為 星期二（87.125 Wh）。分組的 mean、median、std 與樣本數保存在 `eda_statistics.json`。

小時與星期圖表示**該時段觀測值的平均／中位數**，不是每小時或每日總耗能，也不是功率 W。未進行時區推定或季節調整。資料只有一棟住宅約 4.5 個月，時段差異可能同時受到作息、天氣與季節影響；不能推定因果或一般住宅的節能效果。

月份 count／mean／median／std 及 Target、T1、T_out、RH_1 在 10 分鐘、1 小時、1 天、1 週的自相關列於 JSON；1 月與 5 月不完整，不宜直接解讀成完整月份總耗能。Target lag-1 自相關為 0.753166，顯示相鄰觀測非獨立。建議 chronological holdout，加上訓練區內 TimeSeriesSplit；gap 為待按預測視窗決定的設定，不由 holdout 分數選取。天氣資料的插值與感測器平滑也可能加強鄰近依賴。

## 圖表

- [01_appliances_histogram.png](figures/01_appliances_histogram.png)
- [02_appliances_boxplot.png](figures/02_appliances_boxplot.png)
- [03_appliances_time_series.png](figures/03_appliances_time_series.png)
- [04_numerical_correlation_heatmap.png](figures/04_numerical_correlation_heatmap.png)
- [05_feature_target_correlation_ranking.png](figures/05_feature_target_correlation_ranking.png)
- [06_hourly_appliances_pattern.png](figures/06_hourly_appliances_pattern.png)
- [07_weekday_appliances_pattern.png](figures/07_weekday_appliances_pattern.png)
- [15_numerical_distributions.png](figures/15_numerical_distributions.png)
- [16_monthly_and_autocorrelation.png](figures/16_monthly_and_autocorrelation.png)
