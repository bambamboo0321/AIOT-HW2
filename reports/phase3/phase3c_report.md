# Phase 3C — Prediction Interval 與固定模型評估

本階段比較兩個既有模型，並建立單筆新觀測的 95% OLS Prediction Interval（PI）。不重新選 Features、不 fit 或改寫模型、不依 Holdout 調整設定，也不決定新的最佳模型。回歸沒有一般分類的『準確率』，以下以 MAE／MSE／RMSE／R² 與區間覆蓋率分別描述。

## 固定模型、資料與轉換空間

Model A 為 Phase 2 chronological Baseline LinearRegression，25 個原始 numeric Features，只有固定欄位排除，沒有縮放或補值。Model B 為 Phase 3B 事先由 Training CV 選定的 MI + LinearRegression，6 個 Features 依序為 T3、T4、T5、RH_6、T8、T9；使用已保存的中位數補值與 StandardScaler，再依已保存 MI support 取欄。不是重新執行 MI，也不是重訓 LR。模型檔與既有係數皆保持不變。

使用 Phase 2 固定 manifest，前 15,788 筆 Training、後 3,947 筆 Holdout，保留原始順序。Training：2016-01-11T17:00:00 至 2016-04-30T08:10:00；Holdout：2016-04-30T08:20:00 至 2016-05-27T18:00:00。資料原始時區不明，不轉換時區。

預測公式為 ŷ = β₀ + Σ βⱼ zⱼ；Baseline 的 z 是原始數值欄位，Selected 的 z 是已保存 scaler 轉換後的六欄。`prediction_metrics.json` 保存相同空間的 coefficients、intercept 與特徵順序；手算矩陣乘積已核對 saved-model predict，Training Metrics 已核對 Phase 2／3B 歷史值。

## 95% OLS Prediction Interval

設 A 為包含首欄 Intercept=1 的 Training Design Matrix，r=rank(A)，df=n−r，SSE=Σ(yᵢ−ŷᵢ)²，σ̂²=SSE/df。單筆新觀測的 PI 為：

```text
ŷ₀ ± t.ppf(0.975, df) × σ̂ × sqrt(1 + h₀)
h₀ = a₀ᵀ (AᵀA)⁺ a₀
```

以 A=USVᵀ 的 SVD 直接計算 h₀=Σ(a₀ᵀvⱼ/sⱼ)²，不對 AᵀA 做普通 inverse，也不先形成平方條件數的矩陣。有效秩門檻預先固定為 float64 epsilon × max(n,p) × 最大 singular value，與 NumPy 預設 matrix_rank 定義一致。Training 殘差正交性、SVD 重建誤差、leverage trace 與 df 均有驗證。Rank-deficient 的新列若不在可估計 row space，會拒絕給出誤導的區間；零殘差變異亦明確拒絕，不加任意 epsilon 偽造正寬度。

這是**單筆觀測值**區間，含額外的 `1`；對條件平均值的 Confidence Interval 只含 sqrt(h₀)，通常較窄。本階段未把平均值 CI 當成 PI，也不是同時涵蓋整條時間軸的 simultaneous band。上下界保留原值，可能低於物理上非負的耗能；未截斷、未依 Holdout 校準。

## 同一完整 Holdout 的預測結果

所有指標使用完整 3,947 筆及 sklearn.metrics；MSE 單位 Wh²，MAE/RMSE 為 Wh，R² 無單位。

| 固定模型 | Features | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|---:|
| Baseline | 25 | 52.014105 | 7522.602624 | 86.732939 | 0.092352 |
| MI + LR | 6 | 53.605861 | 8179.714353 | 90.441773 | 0.013067 |

Selected − Baseline：RMSE 差 +3.708835 Wh；R² 差 -0.079285。MI 模型的 MAE、MSE、RMSE 均增加且 R² 下降；Feature Selection 未改善本次 Holdout 表現。這只是固定兩模型在同一已查看 Holdout 的描述性比較，不依此重新選模。

## 區間 Coverage 與寬度

Nominal Coverage 固定 95%。Empirical Coverage 以 lower ≤ Actual ≤ upper 的閉區間逐筆計算；寬度標準差採 ddof=0。以下均為完整 Holdout，未用圖表前 144 筆代替。

| 模型 | Empirical Coverage | 平均寬度 Wh | 中位寬度 Wh | 寬度 SD Wh | 低於下界 | 高於上界 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 96.0983% | 376.104533 | 375.915197 | 0.781429 | 0 | 154 |
| selected | 95.8196% | 401.442761 | 401.436931 | 0.092189 | 0 | 165 |

95% 是模型假設下的名目機率，並不保證本資料 Empirical Coverage 等於或高於 95%；未依實測 Coverage 更動任何半徑。

## 數學與數值驗證

| 模型 | Design 欄數含 Intercept | Rank | df | σ̂ | 有效條件數 |
|---|---:|---:|---:|---:|---:|
| baseline | 26 | 26 | 15762 | 95.748551 | 115294.412396 |
| selected | 7 | 7 | 15781 | 102.349464 | 6.536106 |

合成資料以 statsmodels OLS.get_prediction(...).summary_frame() 的 obs_ci_lower／obs_ci_upper 比對，包含完整秩及重複欄 rank-deficient 範例。參考誤差與固定容差記錄於 `phase3c/phase3c_verification.json`；statsmodels 只用於合成測試，正式預測仍由載入的 sklearn LR 提供。

## 圖表與可追溯輸出

兩張預測曲線在評估前固定顯示 Holdout 前 144 筆連續資料；圖中 Actual、saved prediction 與 95% PI 均來自逐筆 CSV。其餘預測指標、Coverage、寬度比較圖採完整 Holdout。五張圖位於 `figures/phase3c/`，來源 CSV SHA-256 與繪圖資料記錄於 `phase3c/figure_manifest.json`；不覆寫 Phase 3B 圖表。所有新 CSV 為 UTF-8、LF。

- [baseline_prediction_interval.png](figures/phase3c/baseline_prediction_interval.png)
- [selected_prediction_interval.png](figures/phase3c/selected_prediction_interval.png)
- [model_performance_comparison.png](figures/phase3c/model_performance_comparison.png)
- [interval_coverage_comparison.png](figures/phase3c/interval_coverage_comparison.png)
- [interval_width_comparison.png](figures/phase3c/interval_width_comparison.png)

## Nested CV、OLS 假設與限制

Phase 3B MI 的 Training 內部 3-fold 選模 RMSE=98.461796；5-fold Nested Outer MI RMSE=103.328278、MAE=66.586098、R²=-0.032966。Nested 各 Fold 的 k 與訓練期間會改變；本階段是固定 k=6、完整 Training 模型，在另一個後期期間評估。切分、Target 變異及程序不同，不能把數值差異解讀為同一無偏估計或任意挑較漂亮的數字。

經典 OLS t 區間依賴適當的線性平均結構、誤差獨立、同方差，且常態誤差下有精確的有限樣本解釋。此資料具時間相依、右偏尖峰、可能異方差、分布漂移與多重共線性；高 Coverage 也可能來自過寬的對稱區間，不能單靠 Coverage 認定校準良好。MI support 是由同一 Training Target 選出，這份 plug-in OLS PI 未涵蓋特徵選擇的不確定性；不可宣稱 post-selection 的精確 95% 保證。

模型使用當時的感測器資訊估計同時間能耗，不是僅用過去資料做多步預報。Phase 2 已查看這個 Holdout，因此它不是完全未查看的獨立最終測試集；本階段沒有據 Holdout 重選 Features、k、模型或調整 PI，沒有 residual calibration。

## Phase 4 準備與停止點

可將兩模型固定評估、數學驗證、完整指標與五張圖交由人工驗收，作為後續最終報告的材料。必須保留已查看 Holdout、時間相依、異方差與選模不確定性限制，不能將名目 95% 說成實測保證或跨住宅泛化證據。尚未開始 Phase 4、未製作 PDF、未 Commit／Push 或部署。

參考：[statsmodels OLS observation interval 範例](https://www.statsmodels.org/v0.14.6/examples/notebooks/generated/ols.html)、[固定測試依賴 statsmodels 0.14.6](https://pypi.org/project/statsmodels/0.14.6/)。依賴增補於 `requirements-phase3c.txt`，既有 `requirements.txt` 不變。
