# Phase 4B-1：公開解法比較

查核日期：2026-10-16。來源代號、作者、版本、網址及證據位置見 [來源清單](external_sources.md)。此文件是 GPT 輔助研究整理，不是 NotebookLM 摘要。外部成績皆為來源保存／發表的結果，本次沒有重新執行實驗。

## 本專案比較基準

本專案沿用同一 chronological 80/20 manifest：training 15,788 筆、holdout 3,947 筆，不 shuffle。原始 28 個輸入，Baseline 排除 `date`、`rv1`、`rv2` 後使用 25 個數值特徵。Target 為原始 `Appliances` Wh。

| 本專案模型 | 輸入數 | Holdout MAE（Wh） | Holdout RMSE（Wh） | Holdout R² |
| --- | ---: | ---: | ---: | ---: |
| Baseline：sklearn LinearRegression | 25 | 52.014105 | 86.732939 | 0.092352 |
| Selected：Mutual Information + LinearRegression | 6 | 53.605861 | 90.441773 | 0.013067 |

本地依據：[Phase 2 baseline_metrics.json](../reports/phase2/baseline_metrics.json)、[Phase 3C prediction_metrics.json](../reports/phase3/phase3c/prediction_metrics.json)。本表讀取既有結果，未重新 fit。

Selected 使用 `T3`、`T4`、`T5`、`RH_6`、`T8`、`T9`；保存的 training-only median imputer／StandardScaler 後接 MI support 與 LR。Baseline 使用原始數值，無 scaling。**Feature Selection 減少特徵數，但沒有改善這份 holdout 的預測表現。** 這個觀察不能推廣為 Feature Selection 對所有模型無用。

本 holdout 在 Phase 2 已查看，屬固定描述性評估，不能宣稱全新未查看的最終測試集。Training CV、nested outer CV 與此 holdout 是不同估計對象；本文件不把它們混成同一個 test 分數。

## 切分與評估條件對照

| 代號／來源 | Target／樣本處理 | 切分與訓練驗證 | 特徵或前處理 | 與本專案的比較界線 |
| --- | --- | --- | --- | --- |
| 本專案 | 原始 Wh；不因 target 偏高刪除資料 | chronological 80/20；訓練內時間序列／nested CV | Baseline 25；MI 6 | 可在同一固定 holdout 內比較這兩個模型 |
| [P1／C1：Candanedo 等，2017](https://doi.org/10.1016/j.enbuild.2017.01.083) | 原始 Wh | target 分層隨機約 75/25；10-fold CV × 3 repeats | 時間衍生、類別 dummy；排除隨機輸入；SVR center/scale | 非相同切分及特徵，僅作研究對照 |
| [P2：Assadian 等，2023](https://doi.org/10.3390/en16227536) | 原始 Wh | 論文說明 10-fold CV、random search；Table 2 為 testing；holdout 比例／seed／時間順序 UNVERIFIED | 時間及週期編碼等特徵工程 | protocol 資訊不足，不作同條件排名 |
| [K1：Lakmali Gamage，2022](https://www.kaggle.com/code/lakmaligamage/appliances-energy-consumption-prediction-lr-svr) | `ln(Appliances)`；刪除 >800／<0，剩 19,721 筆 | random 80/20，seed 42；另有存在問題的 CV | 22 features；StandardScaler | target 尺度、樣本、特徵、切分皆不同 |
| [C2：mhadeli 公開實作，2024 版本](https://github.com/mhadeli/Appliances-Energy-Prediction/tree/5bdb684b7516fc93899f70559e2d98480516ae51) | 主要 regression 使用原始 `Appliances`；另有 log 分析 | random 60/20/20，seed 23 | 自訂優化、RFE、scaling、PCA 等 | R² 計算錯誤；不採用作效能證據 |

隨機切分會讓相近時間樣本分散到 training／test，與用早期預測晚期的情境不同。這是可比性及時間相關風險，不能僅憑 random split 就判定來源一定有直接洩漏。前處理／選特徵若看到 validation 或 test，則另有資訊洩漏風險，需閱讀實際程式判斷。

## 已公開結果：原作者程式 C1

以下均為 **testing set**，來自 [固定版本 R 程式的評估呼叫與相鄰註解](https://github.com/LuisM78/Appliances-energy-prediction-data/blob/e3e4c27a4ae2b41b21f84b4c9ac2d822d3d0ace1/Data%20driven%20models.R#L880)，不是 CV 平均；註解精度照錄。不將程式未註記的值自行補齊。

| 模型 | RMSE（Wh） | R²（來源註記） | 證據狀態 |
| --- | ---: | ---: | --- |
| Multiple Linear Regression | 93.17608 | 0.16 | 已確認作者程式註解 |
| Radial SVM | 70.74 | 0.5152 | 已確認作者程式註解 |
| Random Forest | 68.48478 | 0.54 | 已確認作者程式註解 |
| Gradient Boosting Machine | 66.65093 | 0.57 | 已確認作者程式註解 |

這些為 C1 全模型評估區段，不混入後段移除 lights／weather 等子集實驗。P1 摘要的 GBM testing R² 約 0.57 與此一致；本次未完整讀取 P1 全文，表中細值的直接證據是 C1，不假稱全部抄自論文表格。

## 已發表結果：P2 的 Table 2

依據 [論文全文第 9 頁 Table 2](https://pdfs.semanticscholar.org/d9da/0ea38146d389ce5bb6d75e4f08cd63cdb54a.pdf#page=9)，表題為 testing set。保留原有精度；不能改稱 10-fold CV 平均或本專案的 holdout。

| 模型 | Testing RMSE（Wh） | Testing MAE（Wh） | Testing R² |
| --- | ---: | ---: | ---: |
| Multiple Linear Regression | 91.52 | 51.61 | 0.20 |
| Support Vector Regression | 68.31 | 30.61 | 0.55 |
| Gradient Boosting | 64.77 | 31.29 | 0.60 |
| Random Forest | 62.96 | 29.09 | 0.62 |
| XGBoost | 63.86 | 30.24 | 0.61 |
| Extra Trees | 59.61 | 26.62 | 0.66 |

只能說 Extra Trees 在該文所報告的實驗中表現較佳。測試集定義、特徵工程及調參與本專案不同，不可用 0.66 與 0.092352 的差異計算本專案可保證得到的改善率，也不能據此宣稱非線性模型必然較好。

## 公開 Notebook 的可用與不可用數值

| 來源／模型 | 實際看到的結果 | 是否納入 Wh 效能比較 |
| --- | --- | --- |
| K1 LinearRegression | holdout MAE 0.3854；R² 27.92%（即 0.2792），均在 log target 上 | 否；MAE 不是 Wh，R² 也是不同 target 尺度 |
| K1 SVR | holdout MAE 0.2962；R² 47.52%（即 0.4752），均在 log target 上 | 否；不自行反轉成原尺度成績 |
| C2 自訂 Ridge／Lasso | 確認有模型呼叫及保存輸出；有效標準 R² 為 UNVERIFIED | 否；`exp(r2_score(...))` 不等於 R²，不搬用手動彙表分數 |
| C2 LR／Neural Network／PCA 實驗 | 確認存在；各成績的評估樣本及彙表對應未充分核實 | 否；僅作方法盤點 |

K1 的 holdout「Accuracy」是 `100-MAPE`；CV「Accuracy」是 `100 + neg_MAE`，兩者不能混為分類正確率。其 CV 還包含已打亂的時間順序、fold 外 scaling，以及 SVR 區段誤用 LR 計算 R²，詳見 [來源清單 K1](external_sources.md#k1--可閱讀程式與輸出的-kaggle-notebook)。

## 方法比較與可研究方向

| 方法 | 已查證的實際來源 | 與本專案的研究重點 |
| --- | --- | --- |
| Linear Regression | P1／C1、P2、K1、C2 | 本作業指定模型；用作可解釋基準，檢視非線性／交互作用及殘差限制 |
| Ridge／Lasso | C2；有實際自訂實作，分數品質不足 | 研究正則化與共線性／稀疏性的關係；不宣稱已驗證效能改善 |
| Random Forest | P1／C1、P2 | 研究樹模型如何處理非線性及輸入交互作用 |
| Gradient Boosting | P1／C1、P2 | 研究逐步修正殘差、調參及過度擬合風險 |
| SVR | P1／C1、P2、K1 | 研究 kernel 與 scaling；K1 與原尺度結果分開 |
| XGBoost／Extra Trees | P2 | 研究另一類集成方法；不能取代相同資料切分下的驗證 |

以上是文獻啟發，**不是本專案新增的實驗結果或模型變更**。本專案的 Lasso ranking 是選特徵步驟，下游仍為 LinearRegression，不能寫成已訓練 Lasso 作最終預測模型。

報告可討論的研究假設：以 training-only 時間衍生／週期特徵、非線性模型及時間序列調參為未來工作；若未來另獲實驗授權，應固定資料版本、時間切分、target 尺度及 fold 內前處理，並同時報告 MAE、RMSE、R²。Prediction Interval 應另比較 coverage 與寬度，不能從點預測分數推定區間品質。本次沒有執行這些工作。
