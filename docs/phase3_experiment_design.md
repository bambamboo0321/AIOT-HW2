# Phase 3A：Feature Selection Framework & Benchmark

## 1. 研究目標與目前範圍

本階段建立六種 Feature Selection 的正確訓練邊界、可測試 runner、有限時間成本 Benchmark。目的是驗證架構及估計計算量，沒有尋找最佳方法、執行完整 k 曲線、使用 Holdout 評分、儲存新正式模型或開始 Phase 3B／3C。

既有 Phase 1／2 程式、設定、報告、Notebook、模型與原始 CSV 均保持不變。正式 Predictor 固定為 `sklearn.linear_model.LinearRegression()`，Selector 與 Predictor 為獨立物件。新程式不依賴 Notebook 執行狀態，也沒有建立另一套補值／縮放系統。

## 2. 六種方法

| 方法 | sklearn 元件 | k 與內部工作 |
| --- | --- | --- |
| F-test | SelectKBest(f_regression) | 指定 k，依 training score 選欄；時間依賴下不將 p-value 當獨立樣本推論 |
| Mutual Information | SelectKBest(mutual_info_regression) | 指定 k；random_state=42，dense numeric 採連續特徵估計，非因果解讀 |
| Forward SFS | SequentialFeatureSelector | direction=forward，內部 estimator 每次重建 Phase 2 前處理與 LinearRegression，以時間 CV 的負 RMSE 選候選 |
| RFE | RFE | step=1，以標準化後的 LinearRegression 係數逐次刪欄 |
| LassoCV | LassoCV + 同 alpha 的 Lasso 支援重訓 | 固定 alpha grid、時間 CV 自動選 alpha，再取得非零支援；不將 LassoCV 預測作最終結果 |
| RFECV | RFECV | step=1、min_features_to_select=1，時間 CV 自動選 k；每次 RFE estimator fit 重新前處理 |

F-test／MI 不強制 Scaling；目前能源設定沿用 Phase 2 的 standard，使用者可明確設定 none。RFE／RFECV／LassoCV 強制 standard，避免係數排序受不同量綱主導。SFS 使用傳入的數值設定，此次能源設定為 standard。所有方法都另以選中欄位 fit LinearRegression。

`make_selector` 建立上述真實 sklearn 類別；`fit_selection` 管理 fit 範圍與 mapping；`fit_selected_regression` 再建立獨立 Predictor。Predict 只重用已 fit 的前處理和 support indices。

## 3. k 與 Feature Mapping

k = 經過 Preprocessing 後，實際送入最終 LinearRegression 的數值欄位數。不是原始資料集規範的 28，也不是演算法輸入矩陣尚未選欄的 p。

每次 fit 記錄 selected_feature_names、selected_feature_indices、selected_k、original_feature_mapping、available_feature_names、scaling 與 fit_indices。索引保持遞增，名稱取自 Phase 2 `feature_summary`，選中矩陣為同一 fitted Preprocessor 的對應欄。未選中的原始欄在 mapping 中為空陣列。

Phase 3A 僅支援一對一 numeric 轉換，因此原始候選欄與轉換欄的索引恰好相同；程式會檢查，而非改寫 k 定義。可重用 mean／median／none、StandardScaler／none、keep／clip。拒絕 OneHot、datetime 展開、IQR flag、刪列與其他改變欄位數／資料列的情境。一般上傳資料的展開欄位選擇留待後續另行設計；本階段沒有宣稱支援。

固定 k 方法支援 1..p。k=p 時直接全選，記錄 search_bypassed，不冒充已執行 SFS 搜尋；這也處理 sklearn SFS 不接受 n_features_to_select=p 的限制。自動方法不強迫遍歷每個 k。

Lasso 支援門檻預先固定 `abs(coef)>1e-8`。若得到 0 欄，回傳 status=no_features，跳過 Predictor；predict 會明確拋出 ZeroFeaturesError。若任一 Outer fold 因此無法評分，方法狀態為 incomplete_zero_features，完整 Outer 平均值為 null，不拿部分 folds 平均與完整方法比較。沒有偷偷補成 k=1 或使用另一模型兜底。

## 4. Chronological Split 與 Holdout 隔離

沿用 Phase 2 metadata：原始 19,735 列，前 15,788 為 Training，後 3,947 為 Holdout。正式 Target 為 Appliances；Features 固定為 Phase 2 BASELINE_FEATURES 的 25 欄，排除 date、rv1、rv2。date 僅作排序／檢查。

`load_energy_training()` 檢查完整檔案 SHA-256 身分、既有切分 metadata 與索引，但 CSV parser 明確使用 `nrows=15788`，不解析 Holdout Target 供實驗，也不呼叫會載入完整資料的 `load_dataset()`。完整檔案的雜湊只用於不變性核對。metadata 僅取 count／日期邊界／欄名，不拿既有 Testing 指標或 Target 摘要選模。

TrainingPartition 帶有 allowed_indices／forbidden_indices；runner 拒絕任何 Holdout 或未授權列、重複索引、X/y 不對齊與時間重排。所有 Benchmark 都使用 Training 的 chronological prefix，沒有重新隨機切分或抽取 Holdout。正式 split 未改動。

## 5. Nested Cross Validation

正式預定 Outer=5、Inner=3，使用 TimeSeriesSplit，gap=6（10 分鐘資料為一小時緩衝）；這是沿用的預先設定，不保證已消除自相關。所有方法共用同一份 Outer fold manifest，時間遞增、Train／Validation 索引互斥。

固定 k 方法有多個候選 k 時：

1. 在每個 Outer Training 建立 Inner folds。
2. 對每個候選 k，在各 Inner Training 重 fit 完整 Selector＋Preprocessor＋Predictor；Inner Validation 只 transform／predict。
3. 依 mean Inner RMSE 選 k；相同值時取較小 k。
4. 在完整該 Outer Training 重 fit 選中的流程，只評估一次 Outer Validation。

SFS 本身也有候選特徵的 CV：當它被放入上述 Inner k tuning 時，會在該 Inner Training 再建立 3-fold 演算法 CV。這多一層計算是為了不把已見過 Inner Validation 的選欄結果當成獨立驗證。RFECV／LassoCV 自動 k 不另加固定 k grid；它們在 Outer Training 內使用自己的 3-fold CV 做選擇。

單一預先指定 k 的測試／Benchmark 不需要再用 Inner CV 挑 k；SFS／RFECV／LassoCV 仍會執行其內部 CV。Phase 3A runner 明確限制最多三個 k，拒絕完整 1..25 grid；正式執行須等待 Phase 3B 授權。

必須區分以下四種結果：

| 結果 | 用途 | 解讀邊界 |
| --- | --- | --- |
| A. 各 k 的 CV 曲線 | 在 Training 內描述預先固定 k 的表現 | 查看曲線後挑 k，不能再把該最小值當作無偏的最終分數 |
| B. 內層選出的 k | 每個 Outer fold 自己的模型選擇 | runner 保存 inner_k_curve 與 chosen_inner_k；不使用 Outer Validation |
| C. Outer 評估 | 評估完整「選欄＋選 k＋LR」程序 | 固定同一 Outer folds，報告平均與變異；本階段只有小合成資料實際驗證 |
| D. 最終完整 Training 重訓 | 在程序與設定確定後重做內層選擇，建立最終模型 | 本階段不執行；不能拿某個 Outer fold 的最佳組合直接當最終答案 |

## 6. Preprocessing 與 Scaling 的實作

所有數值轉換都呼叫既有 `src.preprocessing.build_preprocessor`，由 Phase 2 學習補值、IQR clip、scaler，沒有先 fit 全部 Training 再 CV。

SFS／RFE／RFECV 使用可 clone 的 NumericFoldRegressor。sklearn 內部以 ndarray 刪欄，因此 adapter 只為當次候選建立臨時欄名，保持原有順序；每次 fit 都在收到的 training rows 建立全新 Phase 2 Preprocessor 和 LinearRegression。標準化後的 coef_ 供消除排序，最後的正式 mapping 則由 Outer Training 的原始欄位 schema 建立。所有 n_jobs=1。

LassoCV 的內部 coordinate-descent CV 不接受在每個 split 呼叫外部 Pipeline，因此不能直接使用 `Pipeline(StandardScaler, LassoCV)` 宣稱內層隔離。採用以下等效資料隔離：

1. 預先固定 alpha grid = [100,10,1,0.1,0.01,0.001]；不使用完整 Training 的自動 alpha_max。
2. 各 Inner Training 各自 fit Phase 2 Preprocessor，轉換該 Training 和對應 Validation。
3. 將各 fold 的 train／valid block 拼接，提供明確索引給真正的 LassoCV；每個 CV split 只索引自己的 block，絕不跨 block 共用前處理規則。
4. LassoCV 的 mse_path_ 僅由這些隔離 folds 計算。依 sklearn 慣例，alpha 最小化平均 Inner MSE；這不一定等同最小化 mean Inner RMSE，與 SFS／RFECV 的負 RMSE 搜尋準則差異會明示。
5. LassoCV 的內建最終 refit 發生在混合尺度的拼接資料上，其 coef_／intercept_／predict **全部棄用**。計時與 fit 數仍納入這次額外工作。
6. 在完整當前 Outer Training 重新 fit Preprocessor，以選中 alpha fit Lasso 取得支援；最後另 fit LinearRegression。Validation 永遠不參與這次 refit。

真實整合測試逐 alpha、逐 fold，比對 LassoCV 的 mse_path_ 與獨立 fold-local Lasso 計算，確認數值一致。此 adapter 已對本機 sklearn 1.7.2 驗證，未承諾其他版本的內部資料處理相同。拼接會增加記憶體與一次無用 refit；這是保留真實 LassoCV 並維持 fold-local preprocessing 的明確代價。

## 7. Data Leakage 防護與稽核

Outer Validation 不進入 Selector／Preprocessor／Predictor 的 fit；Inner k 評估也只在相應 Inner Training fit。IQR clip 的門檻同樣逐 fit 學習。沒有 Target transformation、Target outlier removal 或全資料相關排序。

可選 audit 記錄每次外部 selector／predictor 的 fit_indices 和 scope。SFS／RFECV 的內部 CV manifest 記錄原始列索引；sklearn 刪欄後的 estimator 只收到 ndarray，因此不假裝其臨時 RangeIndex 是原始索引。測試直接觀察真正進入 estimator 的 y 列識別與 X 值，確認只可能為合法時間 training prefix，並檢查實際 scaler 樣本數與均值。Lasso 額外保留 inner_preprocessors_ 供逐 fold 核對。

繼承的 Holdout 已在 Phase 2 被查看，不是從未見過的最終測試集。Phase 3A 完全不利用它；後續若要宣稱全新外部泛化，仍需新增獨立時間資料。

## 8. 評估指標

主要 RMSE，次要 MAE、R²。`regression_metrics` 使用實際 y／prediction 計算，拒絕形狀不符、少於兩列及非有限值。Outer summary 為各 fold 指標的等權平均，不等同所有列混合重算，也不等同 pooled RMSE。固定 k tuning 使用 mean Inner RMSE；自動方法沿用各自明示的內層準則。所有方法的 Outer 評估指標與 folds 相同。

沒有用 Benchmark 分數排名方法；Benchmark JSON 刻意不發布 predictive scores。合成資料的分數只用於驗證 runner，不代表正式能源資料表現。

## 9. Benchmark 方法與重跑

```bash
.venv/bin/python -B -m src.selection_benchmark
.venv/bin/python -B -m unittest discover -s tests -v
```

Benchmark 預先限定 14 個 cases：先 SFS k=3／n=240 pilot；F-test／MI／RFE 在 n=480 測 k=3,5,10；SFS 在 n=480 測 k=3,5；LassoCV／RFECV 在 n=480 測自動 k。每個 case 都使用 2 Outer／2 Inner、gap=6。資料只在既有 Training prefix 內，所有 25 欄均保留供 Selector 搜尋。

父程序以 subprocess 獨立 process group 執行各 case，單 case 最多 120 秒，整體最多 600 秒。到期 kill 整個 process group 並回收；方法首次 timeout 後不再嘗試其他該方法 case，也不擴大重試。可執行的候選數上限 14、樣本數上限 1024。單一 worker 還保留小樣本限制，沒有正式全曲線 CLI。測試以真正會 sleep 的子程序驗證能在期限內終止，並驗證整體預算／不重試策略。

限制 BLAS／OpenMP／threadpool 為 1，固定 random_state=42。每 case 記錄核心時間與含啟動的 wall time、樣本／Features／fold 數、選中特徵、timeout、警告、環境與 fit accounting。filter score、LR fit、Preprocessor fit 是實際計數；LassoCV 的 alpha path points 是數量估計，warm-start path 不能說成完全獨立的 Lasso fits。

實際輸出：[benchmark_results.json](../reports/phase3/benchmark/benchmark_results.json)、[benchmark_summary.md](../reports/phase3/benchmark/benchmark_summary.md)、[cost_estimates.json](../reports/phase3/benchmark/cost_estimates.json)。

## 10. 實際結果與計算成本

本次 14 cases 全數在期限內完成，含各 worker 啟動的總牆鐘約 23.875 秒。480 rows 的核心秒數：F-test 約 0.101..0.111、MI 約 0.127..0.132、RFE 約 0.610..0.698；SFS k=3 為 1.158、k=5 為 2.277；LassoCV 0.289、RFECV 2.443。240 rows 的 SFS pilot 為 1.246 秒。這些是兩個 Outer folds 合計，不是單次 fit。

LassoCV 的實際選中數為 [0,5]，零特徵 fold 跳過 LR；此次有 7 個 ConvergenceWarning，訊息保留在 JSON。沒有為讓結果好看而改 grid、增加迭代、強迫非零或重新選參數。其目前耗時不能保證代表所有 alpha 已收斂的完整成本。預測方法的優劣不由這個結果決定。

完整成本以實測分項秒數、每個 fit／path 數、正式 Outer folds 平均樣本數，進行名目的線性 n 外推；不是執行完整 Training Benchmark。計算包括 Selector 內部前處理／搜尋／CV、外層 Preprocessor、Downstream LR fit（含 training transform）以及 validation transform／predict／metrics。LassoCV 包含其棄用 refit 與支援重訓。啟動、I/O 與未執行的完整 Training 最終模型重訓不在其中。

固定 k 的曲線成本與完整 nested k 選擇要分開：

| 流程 | Selector 工作量 | Downstream LR fits |
| --- | ---: | ---: |
| F-test／MI，25 個獨立 k ×5 Outer | 各 120 次 score（k=25 不搜尋） | 各 125 |
| RFE，25 個獨立 k ×5 Outer | 1,620 次候選 LR fits | 125 |
| SFS，25 個獨立 k ×5 Outer、3 個演算法 CV folds | 78,000 次候選 LR fits | 125 |
| LassoCV，5 Outer／3 Inner | 15 paths、90 alpha points、5 棄用 refits、5 支援 refits | 最多 5 |
| RFECV，5 Outer／3 Inner | 約 385..505 次候選／重訓 LR fits，視所選 k 而定 | 5 |
| 固定方法完整 Inner k tuning | 各 5×(3×25+1) 次完整候選／外層程序；F／MI score 最多各 365 次 | 各 380 |
| RFE 完整 Inner k tuning | 候選 LR fits 上界 4,985 | 380 |
| SFS 完整 Inner k tuning，加演算法自身 3-fold CV | 候選 LR fits 上界 238,860 | 380 |

獨立曲線的名目秒數見生成摘要；SFS 約 3.11 小時，RFE 約 15.4 分鐘，F-test 約 2.65 分鐘、MI 約 3.54 分鐘。LassoCV nested 約 32 秒、RFECV nested 約 239 秒。這些預測對 Python 固定開銷、樣本數、所選維度、solver convergence、快取與 thread 數非常敏感；0.5x..4x 只表示假設敏感度，並非信賴區間。小樣本 SFS pilot 比較大的 k=3 case 稍慢，也顯示噪音／固定成本；未據此擬合可靠的樣本次方成長模型。

## 11. Phase 3B 建議與待人工決策

正式 default 仍是 k=1..25；Phase 3A 沒有修改。建議人工確認：

- 是否對 SFS 先採 [1,3,5,8,10,13,16,20,25] 的稀疏 grid，或核准全曲線與 nested tuning 的時間預算。
- 區分探索各 k 曲線與評估完整選 k 程序，不能用 Outer Validation 挑 k 後宣稱同一分數無偏。
- Lasso 0-support 是否繼續採「不可評估」，或另訂明確政策；本版不自動 fallback。是否需要只在訓練區驗證更合理的 alpha grid／迭代上限及收斂診斷。
- gap=6 的應用預測視窗意義、Lasso adapter 版本約束，以及是否需要延後一般 OneHot／日期展開支援。

資料與架構可作為 Phase 3B 起點，但正式執行須先決定這些研究設定與資源預算。本階段不執行 Phase 3B／3C、完整曲線、最終模型重訓或 Prediction Interval。

## 12. 已知限制與參考

本版為 numeric-only、dense、一對一轉換，不能泛稱為所有上傳 CSV 的 Selector。非 iid 時間資料的效能與穩定性仍需後續評估；高相關會影響 RFE／Lasso 支援穩定性，小 prefix 的高值分布也未必代表全年。沒有新增 Target lag 或未來標籤，若日後有重疊視窗需另外 purge／embargo。

Benchmark 時間會隨機器與負載變動，不能拿一次 480 rows 投影作精確承諾。跨 k path／rank 的重用有節省潛力，但本版未偷用跨 fold cache，也沒有依此降低已報 fit 計數。LassoCV block 拼接會增加記憶體；若改用其他 sklearn 版本，須先重跑逐 fold path 等價測試。

依本機 sklearn 1.7.2 原始碼與官方文件核對：[SFS](https://scikit-learn.org/1.7/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html)、[RFECV](https://scikit-learn.org/1.7/modules/generated/sklearn.feature_selection.RFECV.html)、[LassoCV](https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.LassoCV.html)。本機程式與實際測試為此次版本行為的驗證依據。
