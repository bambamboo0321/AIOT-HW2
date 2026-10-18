# M7：本機自訂資料的第一個訓練實驗

以下為 M7 原始契約；M8 選欄與 M9 暫存推論擴充見文末。M6-A／M6-B 驗證保持不 fit。M7 新增獨立的明確確認入口，訓練數值型 Linear Regression，並與只預測 Training 平均值的基準比較。沒有選模、調參、六方法自動選欄、Prediction Interval、模型保存或新樣本推論。

網站在 Custom Dataset 驗證通過後顯示第 06 區；勾選本次訓練確認後才可執行。CSV／設定有更動，須重新驗證並確認。此階段結果留在頁面記憶體，重新整理或離頁會清除。

## Python 介面

```python
from src.custom_dataset_training import train_custom_dataset
report = train_custom_dataset(csv_path, config_path, review_path, experiment_path)
```

```bash
.venv/bin/python -B -m src.custom_dataset_training \
  --csv /path/to/data.csv --config /path/to/config.json \
  --review /path/to/review.json --experiment /path/to/experiment.json
```

四個輸入皆必須提供，輸入不改寫；M7／M8 預設不保存模型。M9 新增內部 --model-output 供受信任服務新建暫存模型，不覆寫既有檔案。CLI 是本機受信任操作，HTTP 另有 loopback／Origin／串流大小／取消／逾時等限制。CLI 未自行提供時間或 RAM hard limit。

`review` 沿用 M5 的 configuration_only 確認格式。`experiment` 是另外的明確訓練確認，僅接受：

```json
{
  "schema_version": 1,
  "method": "linear_regression",
  "confirmed": true,
  "dataset_sha256": "由 Python 驗證結果取得的 CSV digest",
  "config_sha256": "由 Python 驗證結果取得的正規化設定 digest"
}
```

這是本機使用者的版本確認，不是簽章或身分認證。訓練前重跑既有 M6-A 驗證，核對 review 與 experiment、CSV／設定版本。讀取器重用 M6-A 內部 bounded parser，因此共享 UTF-8／缺失 tokens／有限數值與容量契約；沒有修改既有驗證模組。

## 評估界線

固定 outer split 後，CV 只接收 Training。沿用 `iter_preprocessed_folds`，每 fold 重新建構並 fit `prepare_fold` 前處理；LR 與 DummyRegressor(mean) 各只 fit 當前 fold training。完整 Training 另外 fit 一份流程，再於全部 Holdout 評估；Holdout 不能改動模型或前處理。

RMSE／MAE 是原目標單位，R² 沒有準確率百分比意義。常數或不足兩筆的評估集 R² 為 null；CV 統計只使用有定義的 folds 並回傳分母。CV 平均／標準差為 fold 等權、ddof=0，不是 pooled score、Nested CV、標準誤或 CI。Training 平均基準不使用 Holdout 均值；R² 本身依評估集的事後均值定義。

預覽只有前 144 筆實際目標／預測／基準／原資料列序號及可選來源時間；指標使用全 Holdout。隨機切分的預覽順序不代表時間；時間切分也只是同列目標估計，不代表多步未來預報。沒有裁切負預測或丟棄測試極端值。

成功：scope=custom_baseline_training、status=completed、training_started=true，exit 0。確認或驗證失敗：status=rejected、training_started=false；開始 fit 後無法完整完成：status=failed、training_started=true；兩者 exit 2 且無部分 metrics。錯誤不回傳原始儲存格、私人路徑或 traceback。數值警告與秩不足會明示，不代表模型品質保證。

查看 Holdout 後再改設定，會造成測試集重用；網站在確認與結果均提醒。正式研究的固定模型與成果不受此本機實驗影響。

## 驗收

```bash
.venv/bin/python -B -m unittest discover -s tests -p test_custom_dataset_training.py -v
```

涵蓋獨立 LR／mean score 核對、Holdout 目標修改不影響 CV／預測、每 fold 的前處理 fit 索引、訓練內補值／縮放／IQR、缺失目標 drop、random／clip／缺失輸入、常數 Holdout 的 null R²、預覽 144 筆但完整評分、失敗不回傳部分成績、確認／digest／未知參數拒絕且不 fit，以及輸入檔案不變。


## M8 擴充：事先指定 F-test／MI 與 k

M8 不改最終 LR 預測器，也不啟動六方法 runner。experiment 新增可選的 `selection`：`{"method":"none","k":null}`、`{"method":"f_test","k":2}`、`{"method":"mutual_information","k":2}`。省略時保留 M7 不選欄行為。只有這兩個 key，固定方法需整數 k 在 1～候選輸入數之內；排名時每個 CV Training 至少四列。其他方法、布林／字串／小數 k 或額外執行參數在任何 fit 前拒絕。

流程重用 `prepare_fold`／`iter_preprocessed_folds` 與現有 `make_selector`：每 fold 的前處理及 Selector 均只看自己的 Training；再訓練獨立 LR。另用相同前處理矩陣 fit 全部候選欄位 LR 及 Training 平均值。完整 Training 另 fit 一次選欄與模型，再評估全部 Holdout。沒有把最終欄位清單套回各 fold，也沒有依 Holdout 選方法或 k。

k=p 時共用全部欄位 LR，未排名；所有輸出包含 ranking_performed 說明。結果新增 selection（含完整 Training 及逐 fold 實際欄名）、CV／Holdout 的 all_features 比較，以及預覽 baseline_prediction。沒有輸出虛構的未測 k 曲線；前端改方法或 k 清除結果並要求重新確認。既有 CSV、配置、模型與研究成果不改寫。

新增驗收：`python -B -m unittest discover -s tests -p test_custom_dataset_selection.py -v`。涵蓋兩方法獨立選欄／預測對照、逐 fold fit 範圍、Holdout 目標變更不影響 CV／選欄／預測、全部欄位對照不隨方法改變、重跑一致、none／k=p 不冒充排名、未知參數及錯誤 k 在 fit 前拒絕、小樣本與 selector 失敗不回傳部分分數。


## M9 擴充：最終模型暫存與單筆推論

`train_custom_dataset(..., model_output=None)` 預設行為維持不保存。網站 train route 明確開啟 retainModels，只在完整評估成功後保存最終 active LR 與所選欄位已 fit 的 branch Pipeline，並回傳短期 prediction_model。Bundle 不包含原始 Training／Holdout 列、CV 模型或比較基準模型。輸出以 O_EXCL／0600 新建，既有檔案會拒絕。

`src.custom_dataset_prediction` 只讀可信任的伺服器模型檔。HTTP 不接受檔案或路徑；registry 把 256-bit 隨機 handle 解析為私有暫存路徑，最長 30 分鐘、最多 3 個，每檔最多 16 MiB。刪除／淘汰立即使 handle 失效，正在推論的檔案在 lease 釋放後移除。程序重啟失去記憶體索引；異常停止孤立檔案的復原管理尚未實作。

推論只呼叫選中 branch 的 transform 及最終 LR.predict，不重新 fit／選欄。支援 numeric 一對一欄位，沿用原 IQR、imputer、scaler；未選候選欄即使原設定 none 也不用提供。mean／median 缺值使用原 Training 統計，none 必填；至少一個實際輸入。補值統計在原 branch 的 IQR 轉換之後學習。回傳實際補值、超出原 Training 有限值 min/max 的欄位與原單位預測；負值不裁切，不提供 PI 或單筆誤差分數。

predict API 使用既有 local-origin／bounded stream／固定無 shell 子程序／逾時取消限制；只接受 exact JSON schema。Python 路徑只由伺服器設定，前端不得指定；JSON 最多 64 KiB、最長 10 秒，stdout／stderr 受限，輸入暫存於程序結束後刪除。前端清除或變更輸入會取消等待並以版本 guard 忽略晚到回應；StrictMode unmount 不自動刪除伺服器模型，以免初次掛載清掉有效模型，改用明確刪除及 TTL。


## M11：批次推論契約

固定 `POST /api/custom-dataset/predict-batch` JSON 只有 `{model_id, features, rows}`；features 必須與 registry 的最終選欄順序完全一致，rows 是 1～500 筆、總共最多 10,000 個 number／null 輸入值的矩陣，每筆至少一個數字。none 模式不接受 null；未知欄位、額外參數、非有限／超過安全範圍數字、錯誤行列形狀一律拒絕。只有模型所需數值傳到伺服器；CSV 欄名重排、缺欄與多欄檢查先於瀏覽器進行。

沿用單筆服務的 local-origin、Host／Sec-Fetch-Site、串流上限與固定無 shell 子程序；batch body 最多 2 MiB、運算 30 秒、stdout 最多 8 MiB（長 Unicode 欄名的逐筆標記仍受有限 cell 數約束），stderr 64 KiB。同一 handler instance 一次一個批次，不接受 DELETE；模型刪除仍走原 predict DELETE。模型 lease 確保執行中檔案在程序停止前不移除。

`src.custom_dataset_prediction --batch --model <trusted> --input <fixed-temp>` 讀取 exact `{features,rows}`，重新檢查筆數、欄序、有限值與缺失政策。只載入一次 bundle，對選中分支各 transform 一次；輸出 batch_predicted／training_started=false，含 model provenance、row_count、連續 position、prediction、imputed_features、out_of_range_features、共用 imputation_values 與 warnings。任一筆失敗不回傳部分結果。CLI exit0 只用於成功，失敗 exit2；不保存輸出或重新訓練。

Node 與瀏覽器都重建／檢查回應，核對模型雜湊、目標、欄名、筆數、筆序、有限預測、每筆缺失欄與共用補值列表。瀏覽器以不可變的 prepared matrix 綁定整批結果與下載，檔案版本 SHA 留於本機。忽略欄位不會傳送或匯出；完全空白行遵循既有 CSV parser 規則跳過，有欄位分隔符但模型輸入全缺失的紀錄會拒絕。

## M12：Training-only 多方法與固定 k 比較

新增獨立 `src.custom_dataset_comparison` CLI 與 `POST /api/custom-dataset/compare`。固定檔案參數為 csv、config、review、experiment；experiment exact schema 為 `schema_version=1, confirmed=true, dataset_sha256, config_sha256, methods, k_values`。methods 僅 f_test／mutual_information／rfe 且不重複，k_values 為遞增且唯一的整數陣列；服務重排方法為固定順序。不接受任何執行路徑或 selector 額外參數。

重新驗證既有 config/review、綁定 CSV 與設定 SHA 後，先建立外層 Training／Holdout 切分，只使用 Training 建立一次 CV indices 清單，所有 method/k 共用。每點每 fold 呼叫既有 `fit_selected_regression`，重用研究 selector 實作；不更動研究 runner、研究成果或現有單方法訓練契約。F-test 名稱維持 f_regression，沒有冒充獨立 Pearson Correlation 演算法。

比較需要顯式 standard 縮放（與 RFE 的標準化係數排名一致），最多 5,000 列、25 欄、25 個 k，CV Training 各至少 4 列。保守成本上限 100,000,000：各方法/k 的 fit 估計數（RFE k<p 用 p-k+2，其餘 1）總和 × p × 各 CV Training 筆數總和。這是拒絕過大工作量的估計，不是進度百分比或時間保證。

成功 scope=custom_feature_comparison、status=compared、comparison_started=true（exit0）；確認／驗證失敗為 rejected/false，運算失敗為 failed/true（exit2），不回傳部分點。結果只有 Training CV、每 fold 原始欄名及分數、等權平均/std ddof0、版本與限制。無 Holdout 預測或成績、無最終模型、無模型 bundle/registry 寫入。模型選擇本身仍屬探索，沒有宣稱 Nested CV 或無偏最佳分數。

API 沿用 loopback/same-origin、exact multipart、bounded streaming、無 shell 固定 Python module、私有暫存檔及 close 後 cleanup；每 handler instance 同時一份比較。120 秒逾時／取消會終止子程序；比較 stdout 限 8 MiB，stderr 64 KiB。前端以 request generation 忽略晚到結果；比較與正式訓練互斥，改方法或 k 清除比較結果並重置確認。RFE 暫不提供正式 Holdout／推論；RFECV／LassoCV 留待另一階段獨立設計。

## M13：RFE 正式訓練與套用比較設定

單方法訓練 selection.method allowlist 新增 rfe，其餘 schema 不變；RFECV／LassoCV／SFS 及額外執行參數仍拒絕。RFE 要求 config.numerical_scaling=standard、資料最多 5,000 筆、候選最多 25 欄，且 `(p-k+2 if k<p else 1) * p * (sum(cv_training_rows)+full_training_rows) <= 100000000`，均在任何 fit 前檢查。

`fit_models` 為 RFE 接收與 fold.y_train 對齊的原始 Training。排名呼叫既有 `fit_selection`，與比較 runner 同一條路徑；避免將已補值／裁切／縮放的矩陣再交給 NumericFoldRegressor 重做前處理。其輸出 indices 對應同 fold 的最終前處理矩陣，另 fit 獨立 LR。完整 Training 另選一次最終欄位再測 Holdout，不依 Holdout 目標選 k，也不回填最終欄位到 CV。k=p 保留全部，無 RFE 搜尋。

最終 bundle 使用原有 one-to-one numeric branch 與 active LR，未修改 bundle 格式或推論服務。新增測試覆蓋 chronological/random、缺失補值、IQR clip、與 M12 逐 fold 分數／欄位相符、Holdout 改目標不影響 CV／選欄／預測、原始 fit 索引、k=p、成本/縮放拒絕、失敗不回傳部分成績，以及 frozen RFE 單筆／批次推論。

前端套用按鈕只傳 method/k，更新下方設定、清除舊結果與 confirmation 並捲到訓練區；不發送訓練請求。正式訓練期間套用停用，Holdout 已看過的提醒不因套用消失。比較結果仍是探索性 CV，不自動推薦最高 R² 或最低 RMSE 的點為無偏最終成績。


## M14：新的自訂 CSV 內層選模契約

新增 `custom_dataset_selectors.py`；不改 `feature_selection.py` 的研究版 TimeSeriesSplit 契約或正式成果。重用 NumericFoldRegressor、fit_preprocessor、FittedSelection，只為自訂 CSV 建立遵循 config 的 3-fold 內層 KFold(shuffle/seed)／TimeSeriesSplit(gap)。SFS 選子集、RFECV 選 k 依平均 RMSE；LassoCV 依固定 alpha grid 的平均 MSE，再在當層 Training 重新 fit 支援集合，最終另 fit LR。

訓練 selection `{method,k}` 保持 exact schema，新增 forward_sfs（整數 k）、rfecv／lasso_cv（k=null）。結果 selection.details 與 fold_features[].details 保存 criterion、alpha、固定 alpha_grid、inner_folds 筆數與 search_bypassed。只有成功完成所有 folds 及最終模型才回傳 completed／保存 bundle；自動方法選 0 欄則 failed，安全錯誤訊息、不回傳部分分數／模型。

比較 methods allowlist 擴充到六種，k_values 只適用固定方法；只有自動方法時必須空陣列。points 仍只包含固定方法的各 k；automatic 是 RFECV／LassoCV 各一份程序結果，包含外層各 fold 的實際特徵、nullable scores 與內層設定。零特徵只可出現在 LassoCV 比較，scores=null；summary 按實際有定義的 folds 等權計算，不補成 0，不推算平均 k。

預先估算 fits：SFS k<p 為 `3*sum(p-j,j=0..k-1)+1`（k=p 則1）；RFECV 為 `4*p+2`（p=1 則1）；LassoCV 為30。Work=fits*p*各外層 Training 筆數總和，正式訓練另加完整 Training；比較再對方法及 k 加總，超過 1e8 拒絕。SFS 2000列/15欄，其他新方法5000列/25欄；每內層 Training 至少4列。預檢在任何 estimator/preprocessor fit 前執行。

前後端重建與驗證新狀態：自動方法不可指定 k、所選特徵數可隨 fold 改變但必須來自候選欄、alpha 必須在固定 grid、內層必須三份合法切分或合法 bypass、零特徵不能冒充分數。API 延用同來源、本機入口、固定無 shell Python、大小／取消／逾時／cleanup；train stdout 上限調為8 MiB。最終 bundle／推論安全邊界不變。
