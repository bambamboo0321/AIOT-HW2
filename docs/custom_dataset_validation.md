# M6-A：自訂資料的 Python 訓練前驗證

這是 M5 CSV／設定頁後的獨立 CLI 與可呼叫函式，尚未接入網站 HTTP API。只讀取使用者明確提供的本機 CSV、設定與可選檢查摘要；不補值、縮放、裁切、選特徵、訓練、預測或保存模型，也不改寫輸入檔案。

## 操作

在 Custom Dataset 頁選擇 CSV、設定目標／特徵並確認，下載 `custom-dataset-config.json`；可另外下載 `custom-dataset-review.json`。從專案根目錄執行：

```bash
.venv/bin/python -B -m src.custom_dataset_validation \
  --csv /path/to/data.csv \
  --config /path/to/custom-dataset-config.json \
  --review /path/to/custom-dataset-review.json
```

`--review` 可省略。成功輸出 JSON 並以 exit code 0 結束；資料或設定無效輸出 `status: invalid`、錯誤階段與原因，exit code 2。參數用法錯誤由 argparse 顯示說明並回傳 2。沒有任意輸出路徑或模型載入選項；若使用 shell 重導向保存 JSON，請自行選擇新檔名，避免覆蓋既有成果。

程式介面：

```python
from src.custom_dataset_validation import validate_custom_dataset
result = validate_custom_dataset(csv_path, config_path, review_path=None)
```

## 檢查內容

- 重用 `PreprocessingConfig.from_dict`、`numeric_target`、`chronological_dates`、`split_dataset`；不重製前處理系統。
- bounded UTF-8／BOM CSV 讀取；上限 25 MiB、50,000 列、200 欄、100 萬格、每格 65,536／欄名 128 UTF-16 code units。JSON 設定上限 128 KiB、檢查摘要 1 MiB；拒絕 JSON 重複 key、NaN／Infinity。
- 保留字串直到明確指定所選數值欄，依 M5 的缺失 tokens 處理；拒絕選中欄位的非法數值、無窮值、超出瀏覽器安全整數範圍的整數及全缺失／常數。錯誤不回傳無法轉換的原始儲存格內容。
- Target 不可當 Feature；Target 缺失只接受 error 或在切分前 drop，不能補值。數值 none 遇所選欄缺失拒絕。
- chronological 先檢查完整原始時間欄的有效性、唯一性與無時區 ISO 格式，再以既有 pandas／時間切分工具排序；不自動合併重複時間。時間 gap 單位為觀測列。
- 實際以既有 outer split 及 sklearn KFold／TimeSeriesSplit 產生索引，回傳外層與逐 CV fold 筆數；不用前端公式代替 Python 檢查。Training／每個 CV Training 至少兩列且 Target 有變化。驗證／Holdout 的常數 Target 提醒 R² 無法定義，不偽造指標。
- 所選特徵在某 partition 變成常數或全缺失會提醒，未擅自補值或刪欄；真正的 fold-local imputer／scaler／IQR／Selector 仍須在未來執行階段另外驗證。
- 原始候選輸入不足 13 欄提供作業規範提醒，與 selection k 分開。隨機切分時若未選欄位含 ISO 日期，也會提醒確認時間依賴。

目前限制為數值輸入、mean／median／none、standard／none、keep／clip、2～10 CV folds，與 M5 可選設定對齊；尚未提供類別／日期衍生、刪列前處理、一般 Selector 或訓練 job。型態合法不代表每一 fold 的變換、選模、PI 或模型效能都已通過。

## 版本核對與輸出界線

回傳原始 CSV 的 SHA-256，以及正規化 v1 設定（UTF-8、JSON sorted keys、無空白分隔）的 SHA-256。若提供 M5 review，核對 artifact type、confirmed 狀態、CSV SHA-256 及正規化前處理設定。改過 CSV 或設定後，舊摘要不可沿用。

摘要是可編輯的本機 JSON；此檢查只核對版本一致性，不是使用者身分驗證、簽章或服務授權。未提供 review 時 `review_checked` 為 false，不冒充已核對使用者確認。

輸出不包含原始資料列、原始儲存格、分割索引、模型係數或機器私人路徑。只有筆數、版本、警告與是否通過訓練前驗證；不將任何研究模型指標套到新資料。

## 驗收

```bash
.venv/bin/python -B -m unittest discover -s tests -p test_custom_dataset_validation.py -v
```

測試使用隔離合成 CSV，涵蓋實際 fold 筆數、缺失／非法數值／日期、過大 gap、常數 Training Target、review 版本不符、JSON／CSV 錯誤、CLI exit code、輸入檔案不變。以禁止 LR／scaler／imputer 的 fit 及 `prepare_fold` 執行，驗證此入口沒有開始模型或前處理學習。

下一階段才接 HTTP API 與前端錯誤顯示；訓練入口、執行進度、取消及六方法 runner 仍待後續實作。
