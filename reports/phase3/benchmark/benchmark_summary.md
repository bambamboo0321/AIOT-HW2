# Phase 3A：有限時間 Benchmark

僅計算成本，不比較模型預測分數；Holdout 未參與。資料為 Phase 2 Training 的前 240／480 列，保持原始時間順序與全部 25 個 Features。random_state=42，單執行緒。每 Case 上限 120 秒，整體上限 600 秒，subprocess 到時直接終止並回收。

每 Case 使用相同的 2 個 Outer folds；SFS／RFECV／LassoCV 內部使用 2 個時間 folds（gap=6）。固定 k 只驗證指定 k，不在此次 benchmark 尋找最佳 k。

| Method | k | Samples | Status | 核心實測秒 | 含啟動實測秒 |
| --- | ---: | ---: | --- | ---: | ---: |
| forward_sfs | 3 | 240 | ok | 1.2456 | 2.4804 |
| f_test | 3 | 480 | ok | 0.1113 | 1.3386 |
| f_test | 5 | 480 | ok | 0.1013 | 1.1106 |
| f_test | 10 | 480 | ok | 0.1025 | 1.0675 |
| mutual_information | 3 | 480 | ok | 0.1324 | 1.0645 |
| mutual_information | 5 | 480 | ok | 0.1275 | 1.0433 |
| mutual_information | 10 | 480 | ok | 0.1274 | 1.1469 |
| rfe | 3 | 480 | ok | 0.6978 | 1.6265 |
| rfe | 5 | 480 | ok | 0.6729 | 1.5889 |
| rfe | 10 | 480 | ok | 0.6105 | 1.5223 |
| forward_sfs | 3 | 480 | ok | 1.1584 | 2.0679 |
| forward_sfs | 5 | 480 | ok | 2.2769 | 3.2130 |
| lasso_cv | auto | 480 | ok | 0.2891 | 1.2086 |
| rfecv | auto | 480 | ok | 2.4434 | 3.3781 |

總實測牆鐘時間：23.875 秒。完整環境、警告、選中特徵 mapping、fit/path 計數與分項計時見 benchmark_results.json。零特徵 fold 不 fit Predictor；不補成 k=1。

## 完整設計成本外推

以下是基於小樣本的計算投影，**不是完整實驗實測**。固定方法為 k=1..25 各 k 獨立曲線、5 Outer／3 Selector CV folds；自動方法只跑一次 5 Outer／3 Inner nested 流程。表中總秒數另含外層前處理，不含 I/O、process 啟動、最終完整 Training 重訓或內層 k 調參。Downstream fit 包含訓練資料 transform 與 LR.fit，evaluation 包含 validation transform／predict／metrics。

| Method | 名目總秒數 | Selector 秒數 | Downstream LR fit 秒數 | Evaluation 秒數 |
| --- | ---: | ---: | ---: | ---: |
| f_test | 159.06 | 9.00 | 34.54 | 16.71 |
| mutual_information | 212.31 | 67.75 | 34.05 | 16.60 |
| forward_sfs | 11194.80 | 11039.52 | 32.87 | 15.91 |
| rfe | 923.90 | 784.86 | 32.61 | 16.00 |
| lasso_cv | 32.23 | 26.35 | 1.33 | 0.67 |
| rfecv | 239.11 | 233.58 | 1.28 | 0.64 |

計數／假設／敏感度區間見 cost_estimates.json。SFS 曲線需要 78,000 次候選 LR fits；若把 k 選擇也完整巢狀化，候選 fits 上界增至 238,860（另加 Downstream fits），不可只計算 125 次最終回歸。小資料下 Python 前處理開銷占比高，不能將此線性外推當成實際完工時間。

建議 Phase 3B 人工決定是否採較疏的 SFS k=[1,3,5,8,10,13,16,20,25]，或核准計算預算再執行全曲線。本階段不改變正式 k=1..25 設計，也不執行正式曲線。

LassoCV 的小樣本 Outer folds 實際選出 [0, 5] 個特徵；記錄了 7 個警告。 零特徵 fold 跳過 Predictor，沒有強迫補成 k=1。警告均保留原訊息供核對，沒有因此擴大重試、挑選方法或調整 alpha grid。若存在收斂警告，其時間外推只是目前迭代上限下的成本，不能視為收斂解的成本保證。
