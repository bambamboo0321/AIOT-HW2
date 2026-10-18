# Phase 3D：原有 Feature Selection 唯讀查核

查核日期：2026-10-16。Branch `main`；HEAD `4619bd493a337f32d34593f12686035f2c70a630`。本報告於 Random Combination Search 前完成；未改寫既有模型或實驗結果。開始時預期變更為 Phase 4B-2 README 修改及 Phase 4B-1／4B-2 的 12 個未追蹤檔案；staging 空。完整起始 125 檔 SHA-256 見 [integrity_baseline.json](integrity_baseline.json)。

## A. 已驗證正確

| 查核項目 | 證據與結果 |
| --- | --- |
| Dataset／切分 | `selection_runner.load_energy_training` 核對 Phase 1 SHA-256、Phase 2 manifest；只以 `pd.read_csv(..., nrows=15788)` 解析前 15,788 列。後 3,947 列 Holdout 不送入搜尋；讀取全檔二進位 SHA-256 不等於使用 Holdout Target |
| 25 個候選 | Phase 3B `experiment_config.json#preprocessing` 與既有 `energy_config.json` 完全一致；`Appliances`、`date`、`rv1`、`rv2` 不在候選；date 僅維持時間排序 |
| MI | `SelectKBest(mutual_info_regression, k=6)`；dense continuous 輸入，`discrete_features=auto`、`n_neighbors=3`，seed 42、n_jobs=1。每欄對 Target 的單變量 MI 排名，不是六欄聯合 MI，也不直接最小化 LR RMSE |
| Training-only | `fit_selection` 僅接當前 training X/y；`fit_group` 不收 validation；full-training 最終 refit 與 fold 內 fit 有不同 level，不把前者支援集合用於宣稱無偏 selector CV |
| 時間 CV | 原正式 nested outer=5、inner=3、selector=3，皆 TimeSeriesSplit、gap=6。1,974 筆紀錄的 train／validation 範圍、gap、候選與有效狀態查核通過 |
| 前處理 | 每個原始數值欄獨立分支：training-only median imputer、StandardScaler，IQR keep；每個 CV training fold 重新 fit。SFS／RFECV 的內層 `NumericFoldRegressor` 亦重新 fit 前處理 |
| 路徑重用 | Filter 分數、RFE elimination、SFS prefix 僅在同一 training partition 重用；既有測試核對直接 sklearn 選擇與支援集合等價 |
| Linear Regression | 下游固定 sklearn LinearRegression（含截距、普通最小平方法）；RFE／SFS 的評分 estimator 與下游 LR 各自獨立；未新增正則化最終模型 |
| 指標 | `regression_metrics` 使用 sklearn MSE、MAE、R²，RMSE=sqrt(MSE)。各 fold RMSE 算術平均作比較，不是先合併誤差再開根號；SFS 內層 scoring 為 neg_root_mean_squared_error |
| Phase 3B／3C | 以禁止 fit 的 guard 載入可信既有模型，只對 Training 預測；MI training RMSE=102.32677150116135、MAE=60.09601269710763、R²=0.053787486889167124，與 Phase 3B refit／Phase 3C 完全相符。Phase 2／3C Baseline Holdout 指標亦一致 |
| 既有測試 | 搜尋前執行原有 132／132 PASS；含 fold-local、時間範圍、saved-model、CSV／JSON、數學指標與不 refit 測試 |

直接依據：`src/feature_selection.py`、`src/phase3b_selection.py`、`src/phase3b_runner.py`、`src/selection_runner.py`、`src/preprocessing.py`、`src/phase3c_evaluation.py`、`src/verify_phase3b.py` 及 reports/phase3/results。機器查核結果見 [audit_details.json](audit_details.json)。

## B. 發現的明確問題與已知限制

- **未發現足以中止補充搜尋的原流程計算錯誤或直接資料洩漏。** 這是程式、保存紀錄及測試所支持的結論，不是對所有歷史執行的形式化證明。
- 原 SFS grid 為 1、3、5、8、10、13、16、20、25，沒有 k=6；不能插值或把 k=5／8 當成六欄結果。本階段必須新增相同時間 CV 的 fold-local SFS k=6 評估。
- 原 MI 最終六欄來自完整 Training，與 final_selection_cv 各 fold 的六欄並不相同；其固定組合的回顧性 CV 不能冒充包含選擇程序的無偏 CV。
- MI 是單變量關聯評分，沒有保證被選欄位彼此不冗餘，也不保證對 LR 最佳。這是方法定義，不能單憑某次較差成績認定實作有錯。
- Holdout 已在 Phase 2 查看。Phase 3D 不以其成績篩選組合，也不重新評估新的 Holdout 結果。

## C. 尚無足夠證據判定

- 既有資料／紀錄不能證明跨季節、跨住宅或真正前瞻泛化能力；「最佳六方法」擇優後的分數也不是該擇優程序的無偏估計。
- 尚不能由觀察性資料判定單一特徵對耗能的因果效果；高相關與較低 CV RMSE 均不足以證明因果。
- 1,000 組隨機組合只覆蓋全部組合的一部分；搜尋前不能保證抽到全域最佳，搜尋後也不能把最小 CV 誤差當成真實泛化誤差。

## 本階段固定比較契約

選用 **Phase 3B 的 final_selection_cv**：3 folds，gap=6（單位為觀測列）、seed=42。選擇理由是它正是原 MI k=6 的訓練內最終選模層級，不與 5-fold nested outer 或 Holdout 混用。其三組半開區間：

| Fold | Train [start,end) | Validation [start,end) |
| --- | --- | --- |
| 1 | [0,3941) | [3947,7894) |
| 2 | [0,7888) | [7894,11841) |
| 3 | [0,11835) | [11841,15788) |

所有固定六欄組合重用各 fold 已 fit 的逐欄 numeric median／scaler，再各自 fit 獨立 LR；不跨 fold 共用統計量。由於既有轉換逐欄獨立且不擴張、不刪列，先轉換 25 欄再取六欄與只轉換該六欄等價；新增測試將核對此最佳化。

兩類比較必須分開：

- **A 固定組合 CV**：讀取既有完整 Training 選定的 MI／F-test／RFE 六欄，與隨機六欄組合在上述相同 fold 評估。這些方法產生的固定組合曾使用完整 Training Target，包含回顧性 CV validation 的資訊，故只是有資料重用／選取偏差的探索性參考；不是 selector 泛化評估。原 SFS 沒有完整 Training 的 k=6 組合，本階段不額外先用全 Training 選它再宣稱無偏。
- **B 包含選擇流程的 CV**：MI／F-test／RFE 讀取原 final_selection_cv 的 k=6 fold-local 紀錄；SFS k=6 只在各 CV training fold 內重新選擇，validation 只用於最後評分。其分數不與 A 混排；MI／其他方法或 k 的事後擇優仍有 selection bias。

查核結論：**PASS，可依本契約進行最多 1,000 組探索性搜尋；不得執行 Exhaustive Search、不得使用 Holdout 選模。**
