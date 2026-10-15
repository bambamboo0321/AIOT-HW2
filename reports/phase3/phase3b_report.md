# Phase 3B — 正式特徵選擇實驗

本階段所有建模、選模與 Benchmark 只使用既有 chronological Training 的 15,788 筆，不讀取 3,947 筆 Holdout Target。Phase 1 回歸驗證另唯讀檢查全檔資料品質，其結果不參與選模。最終 Predictor 維持 sklearn LinearRegression。

## 方法與實驗設計

原始 28 候選輸入排除 date、rv1、rv2，保留 25 個 numeric Features。F-test／MI／RFE 使用 k=1..25；SFS 使用預先固定的 [1,3,5,8,10,13,16,20,25]；LassoCV／RFECV 自動決定 k。k=25 明確全選，不冒充 SFS 搜尋。

Outer TimeSeriesSplit=5；各 Outer Training 內以 Inner TimeSeriesSplit=3 選 k。gap=6。SFS 與自動方法另在當前可見的 Training 內使用 3-fold selector CV。所有方法共用相同切分；中位數補值、StandardScaler 與 Selector 只在當前 Training fit。沿用 Phase 2 前處理及 Phase 3A fold-local estimator。

各 k 的 Outer Curve 是描述性比較；正式 Outer 評估只取事先由 Inner CV 決定的 k。另在完整 Training 上進行 3-fold final_selection_cv，決定最終方法及 k；最小平均 RMSE 為主，差距 ≤1e-9 時依較少平均 Features、較短 fit time、宣告方法順序決定。規則在正式分數產生前已寫入設定。選六種方法的程序本身未再經獨立最外層評估，不能將最佳方法的 Outer 成績宣稱為整個選模程序的無偏估計。

## Training CV 選模結果

以下是 final_selection_cv，與後面的 Nested Outer 表格層級不同。

| 方法 | 要求 k／auto | 平均實選 k | RMSE | MAE | R² |
|---|---:|---:|---:|---:|---:|
| F-test | 6 | 6.000 | 99.164490 | 60.703080 | 0.028912 |
| Mutual Information | 6 | 6.000 | 98.461796 | 59.871672 | 0.043218 |
| Forward SFS | 10 | 10.000 | 99.165656 | 59.858338 | 0.027904 |
| RFE | 6 | 6.000 | 101.275256 | 61.091159 | -0.014403 |
| LassoCV | auto | 7.333 | 100.131852 | 63.806768 | 0.008116 |
| RFECV | auto | 11.333 | 103.002275 | 65.516701 | -0.051350 |

同一 final_selection_cv 的全 25 Features 參考：RMSE=103.122735、MAE=65.366001、R2=-0.056273。

## Nested Outer 評估

| 方法 | 各 Fold 實選 k | RMSE | MAE | R² |
|---|---|---:|---:|---:|
| F-test | [1.0, 10.0, 9.0, 5.0, 7.0] | 101.184255 | 63.092611 | 0.012190 |
| Mutual Information | [1.0, 3.0, 24.0, 3.0, 10.0] | 103.328278 | 66.586098 | -0.032966 |
| Forward SFS | [1.0, 13.0, 5.0, 3.0, 5.0] | 100.851387 | 61.248952 | 0.018761 |
| RFE | [11.0, 9.0, 11.0, 10.0, 9.0] | 100.756367 | 62.932507 | 0.019744 |
| LassoCV | [2.0, 16.0, 3.0, 14.0, 16.0] | 99.075928 | 60.438543 | 0.053379 |
| RFECV | [11.0, 9.0, 11.0, 10.0, 9.0] | 100.756367 | 62.932507 | 0.019744 |

## 最終模型與 Selected Features

依 Training CV 選定 **Mutual Information**；要求 k=6。各方法均在完整 Training 重新 fit Selector 及獨立 LinearRegression；最終模型位於 `results/models/final_linear_regression.joblib`，包含可重用前處理與 Selector。保存後已做載入／預測一致性檢查。

- F-test：實選 k=6；lights, T2, T3, T6, RH_8, RH_out。
- Mutual Information：實選 k=6；T3, T4, T5, RH_6, T8, T9。
- Forward SFS：實選 k=10；RH_1, T2, T3, T5, RH_5, RH_7, T8, RH_8, T9, RH_out。
- RFE：實選 k=6；RH_1, T2, RH_2, T3, RH_3, RH_8。
- LassoCV：實選 k=16；lights, T1, RH_1, RH_2, T3, RH_3, T4, T5, T6, RH_7, T8, RH_8, T9, RH_9, Windspeed, Visibility。
- RFECV：實選 k=6；RH_1, T2, RH_2, T3, RH_3, RH_8。

完整 Training 的 refit 分數標記為 `final_refit_training`，僅為訓練內分數，未用來選模。

## 計算時間、SFS 與續跑

SFS 先在完整 15,788 筆上執行一次完整預定 grid 的成本 pilot，不保存或採用 pilot 的選擇／分數。正式 gate 估算涵蓋 24 個 partition，使用 full-n 實測成本乘 2 安全係數。詳見 `results/sfs_budget.json`。只在同一 Training Partition 共用確定性的 filter scores、RFE elimination path 與 sklearn SFS greedy prefixes；測試對照逐 k 原始 Selector。未依分數刪除任何 k，也不跨 Fold 分享規則。

每個 method/partition 由獨立 subprocess 執行，120 秒硬期限，以 SIGKILL 終止 process group 並 reap；SFS 總啟動預算 3,600 秒。結果即時原子寫入 `.venv/phase3b_checkpoints/`，含 code/config/environment/data/range fingerprint；再次執行跳過所有 terminal cases，含 timeout，不自動重試。checkpoint 是本機暫存，不納入版本追蹤；正式 JSON 保存各 case 識別與時序。

CSV 每列 runtime_seconds 將同一 grid 的共享 fit 耗時平均分攤至各 k，再加上各自 LR fit 耗時；不代表逐 k 獨立搜尋的成本。Runtime 圖由 CSV 去除重複的 Outer 評估列後加總，Ledger 另保存含評分與程序啟動的實測耗時。各方法 grid 工作量不同，圖中呈現的是這份正式實驗的總計算成本。

- F-test：Worker 6.740 秒，含程序啟動 28.036 秒；Cases 24，Timeout 0。
- Mutual Information：Worker 13.661 秒，含程序啟動 34.989 秒；Cases 24，Timeout 0。
- Forward SFS：Worker 269.311 秒，含程序啟動 290.514 秒；Cases 24，Timeout 0。
- RFE：Worker 16.645 秒，含程序啟動 38.004 秒；Cases 24，Timeout 0。
- LassoCV：Worker 2.173 秒，含程序啟動 10.156 秒；Cases 9，Timeout 0。
- RFECV：Worker 17.414 秒，含程序啟動 25.357 秒；Cases 9，Timeout 0。

## LassoCV 收斂與失敗紀錄

StandardScaler 必用；alpha grid=[100,10,1,0.1,0.01,0.001]，tol=1e-5，support 門檻=1e-8。針對 Phase 3A 的 7 個 ConvergenceWarning，正式執行前將 max_iter 由 10,000 提升至 100,000，alpha/tol 未變，僅以 Training 診斷；正式共 0 個 ConvergenceWarning、0 個帶警告 Case。警告全文保存於 JSON。LassoCV 內部遵循 sklearn 的平均 MSE 決定 alpha；SFS／RFECV 內部使用 neg-RMSE。各方法之間及固定 k 的正式比較一律使用相同的 RMSE、MAE、R²。

k=0 的 0 筆記錄保留 selected_k=0、status=invalid_for_LR 與 null LR 指標；未補成 1，未計算不存在的 LR 分數。任何失敗 Fold 都使該候選不可參與最佳值比較。LassoCV 在各內層 Fold 分別 fit scaler 的 blocks 上決定 alpha；其混合尺度的內建 refit 不用於選擇或預測，support 另以當前完整 Training 的 scaler+Lasso 重算，最終預測仍用 LR。

## Leakage、Baseline 與 Holdout 限制

資料 loader 僅 read_csv(nrows=15788)；整檔 SHA-256 僅檢查位元組完整性。Case range 必須位於 [0,15788)，Validation 與 Training 保持 gap=6。所有補值、縮放、特徵選擇都在目前 Training Partition 學習，Inner 選 k 不讀取 Outer 分數。固定同一 random_state=42、單一 BLAS thread。

圖中的 Baseline 是相同 Training Outer CV 的 k=25 LR，不是 Phase 2 Holdout 分數。Phase 2 chronological Holdout RMSE 86.732939、MAE 52.014105、R² 0.092352 僅作既有歷史背景，無法與不同切分的本階段 CV 數值直接排名；原始 Baseline Metrics 與模型保持位元組不變。Phase 2 已查看 Holdout，不能將其稱為未查看的最終測試集。

## 輸出、驗證與 Phase 3C 準備

`results/feature_selection_results.csv/.json` 分開保存 inner_k_cv、outer_k_curve、outer_evaluation、final_selection_cv、final_refit_training；另存 Inner 選擇結果、實選 Features、Runtime 與設定／環境。七張圖位於 `figures/`；各圖資料與 CSV SHA-256 在 `results/plot_data.json`。SFS 僅畫實測 k 的點，缺失與無效結果不插值。

本次驗證詳見 `phase3b_verification.json`，與 Phase 3A 的 `verification.json` 分開。Phase 1 既有 rv1/rv2 重複警告須保留，不能誤報為全項 PASS。

完成驗收後可規劃 Phase 3C 的時間序列校準與區間覆蓋率評估；目前未製作 Prediction Interval。Holdout 已查看、時間相依性與模型選擇後評估偏差仍須處理，不能直接宣稱區間有效。尚未 Commit／Push，等待人工驗收。
