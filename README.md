# AIOT HW2：Python Machine Learning

本專案研究 Appliances Energy Prediction 資料集，目標為了解室內溫濕度、照明與外部天氣等觀測資訊，能否用於預測家電能源消耗 `Appliances`。後續依作業要求使用 sklearn Multiple Linear Regression（多元線性迴歸）。目前只完成 Phase 1「環境建立與資料驗證」。

## 資料來源與原始檔

- 資料集名稱：Appliances Energy Prediction。
- Kaggle 公開資料集：[Appliances Energy Prediction（loveall）](https://www.kaggle.com/datasets/loveall/appliances-energy-prediction/data)。2026-10-14 查核頁面顯示 **Version 1**，檔案為 `KAG_energydata_complete.csv`，29 欄、11.98 MB。
- 原始資料來源：[UCI Machine Learning Repository，Dataset 374](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)，2017-02-14 收錄。
- Dataset DOI：[10.24432/C5VC8G](https://doi.org/10.24432/C5VC8G)。
- 原始作者署名：Candanedo, L. (2017). *Appliances Energy Prediction* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5VC8G。
- 資料授權：[Creative Commons Attribution 4.0 International（CC BY 4.0）](https://creativecommons.org/licenses/by/4.0/)，依 UCI 官方標示；Kaggle 頁面的 License 欄位顯示「Other (specified in description)」，本專案以 UCI 原始資料的明確授權為依據。
- 使用者提供的原檔：`KAG_energydata_complete.csv`。
- 正式資料來源與唯一 Git 追蹤的 CSV：`data/raw/KAG_energydata_complete.csv`。
- 根目錄原檔保留於本機，依精確檔名加入 `.gitignore`；只從 Git 追蹤中移除，未刪除實體檔案。兩份檔案 SHA-256 相同。

### 來源與版本核對

2026-10-14 從 [UCI 官方資料壓縮檔](https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip) 讀取 `energydata_complete.csv`，與本機 CSV 比對：**位元組完全一致**，SHA-256 均為 `2820bf712ad0275cb18b85a05250926100d8e65ebb9f4d2d016ca91ea152a25d`。兩者皆為 19,735 筆、29 欄，欄名、順序與全部數值完全一致；Kaggle 的 Version 1 依頁面檔案名稱、大小與欄位數核對，內容身分則以上述 UCI 官方檔案的完整比對確認。

再散布本資料時，須保留作者署名、資料來源及 CC BY 4.0 授權連結，並註明是否修改。本專案僅以 `KAG_energydata_complete.csv` 名稱保存原始資料，**CSV 內容未修改**；未清理、補值或刪除欄位。此授權聲明適用於資料集，沒有另加限制或宣稱原作者背書。

### 原始論文引用

Candanedo, L. M., Feldheim, V., & Deramaix, D. (2017). Data driven prediction models of energy use of appliances in a low-energy house. *Energy and Buildings, 140*, 81–97. https://doi.org/10.1016/j.enbuild.2017.01.083。

論文書目已核對 [出版商頁面](https://www.sciencedirect.com/science/article/abs/pii/S0378778816308970)。此處僅記錄資料來源引用，尚未進行 NotebookLM 文獻研究或摘要撰寫。

## CRISP-DM 六階段

| 階段 | 本專案對應工作 | 目前狀態 |
| --- | --- | --- |
| Business Understanding（商業理解） | 確認家電能源消耗研究目標與作業限制 | Phase 1 記錄目標 |
| Data Understanding（資料理解） | 檢查原始資料結構、品質、時間欄位與候選欄位數量 | Phase 1 驗證 |
| Data Preparation（資料準備） | 後續決定資料切分、時間處理與 Feature Selection | 尚未執行 |
| Modeling（建模） | 後續使用 sklearn 多元線性迴歸 | 尚未執行 |
| Evaluation（評估） | 後續 Model Evaluation 與 95% Confidence／Prediction Interval 圖 | 尚未執行 |
| Deployment（部署） | 彙整可重現的交付成果與研究報告 | 尚未執行 |

CRISP-DM 的六階段是研究方法論，與作業的 Phase 1–4 執行範圍不同。

## 原始 Features 與模型 Features

老師規定原始輸入 Features 數量必須 **>13 且 <40**。原始欄位共 29 個，扣除 Target `Appliances` 後，原始輸入 Features 為 **28** 個，包含 `date`。

| 統計口徑 | 排除的欄位 | 數量 |
| --- | --- | ---: |
| Original Input Feature Count | `Appliances` | 28 |
| Candidate Feature Count（Exclude date） | `Appliances`、`date` | 27 |
| Candidate Feature Count（Exclude date, rv1, rv2） | `Appliances`、`date`、`rv1`、`rv2` | 25 |

候選數量只依欄名統計，沒有執行 Feature Selection，也沒有從資料中刪除欄位。模型 Features 是後續實驗決定實際輸入模型的欄位；尚未選定或計算。

`date` 是時間欄位，只解析以驗證品質；未當作一般連續數值，也未建立衍生時間特徵。`rv1`、`rv2` 是隨機變數，是否保留由後續實驗決定。

## 專案結構

```text
AIOT-HW2/
├── KAG_energydata_complete.csv       # 本機原檔保留，Git 忽略
├── data/
│   └── raw/
│       └── KAG_energydata_complete.csv
├── src/
│   ├── __init__.py
│   └── validate_data.py
├── reports/
│   └── phase1/
│       └── data_validation.json
├── notebooks/
│   └── .gitkeep
├── .venv/                           # 本機環境，不納入 Git
├── .gitignore
├── requirements.txt
└── README.md
```

## Mac Python 環境

優先使用 Python 3.11 或 3.12。本機在 PATH 與常見安裝位置未找到這兩個版本，找到 `/opt/homebrew/bin/python3.10`（3.10.17）與系統 Python 3.9.6，因此使用既有 Python 3.10.17 建立專屬虛擬環境。未更動系統 Python，未全域安裝套件。

從專案父目錄進入 `AIOT-HW2`，再於專案根目錄執行；已有 `.venv` 時直接使用，不必重新建立：

```bash
cd AIOT-HW2
# 若已安裝 Python 3.12，建立新環境時可使用：
# python3.12 -m venv .venv
# 此次實際建立環境使用：
/opt/homebrew/bin/python3.10 -m venv .venv
.venv/bin/python -m pip install --no-cache-dir --timeout 60 -r requirements.txt
.venv/bin/python -m pip check
```

`requirements.txt` 僅記錄實際安裝並驗證成功的四項直接依賴版本；傳遞依賴由 pip 解析。跨平台或其他 Python 版本尚未驗證。

本機實際安裝並成功匯入的版本：

| 項目 | 版本 |
| --- | --- |
| Python | 3.10.17 |
| pandas | 2.3.3 |
| numpy | 2.2.6 |
| scikit-learn | 1.7.2 |
| matplotlib | 3.10.9 |

`pip check` 通過，沒有依賴衝突。未匯入繪圖介面或訓練模型；套件匯入驗收時以 `.venv/matplotlib-config` 作為 Matplotlib 的設定快取位置。

## 執行資料驗證

```bash
.venv/bin/python src/validate_data.py
```

預設 CSV 與報告路徑以程式位置解析，不依賴執行時所在目錄。也可使用模組形式：

```bash
.venv/bin/python -m src.validate_data
```

如需驗證另一份 CSV，請指定不同的報告路徑，保留正式報告：

```bash
.venv/bin/python src/validate_data.py --csv path/to/input.csv --report reports/phase1/alternative_validation.json
```

程式只讀取 CSV，重複執行會更新 JSON 報告。報告包含 SHA-256、檔案大小、shape、所有欄名與 dtype、缺失值、完全重複列、Target、三種特徵數量、時間驗證、隨機變數統計、逐欄數值檢查與各項驗證狀態。標準差使用樣本標準差（`ddof=1`）。時間間隔依原始列順序計算，不重新排序；不可解析的相鄰時間不列入間隔統計，會另外標記 FAIL。

- `PASS`：符合預期。
- `WARN`：需要注意，但沒有必要條件失敗，退出碼為 0。
- `FAIL`：必要條件不符合，退出碼為 1。

19,735 rows、29 columns、28 original input features 任一不符，或 Target、缺失值、數值有效性、必要欄位、日期可解析性等必要檢查失敗，皆標記 FAIL。時間戳記重複、順序或 10 分鐘間隔異常，以及 `rv1`／`rv2` 完全相同，會標記 WARN；不自動修正資料。JSON 禁止輸出非標準的 `NaN`／`Infinity`。

## Phase 1 驗證結果

驗證日期：2026-10-14。正式結果請參閱 [data_validation.json](reports/phase1/data_validation.json)。

| 驗證項目 | 實測結果 |
| --- | --- |
| Shape | 19,735 rows × 29 columns，符合預期 |
| 原始／排除 date／再排除 rv1、rv2 的欄位數量 | 28／27／25，符合預期 |
| 老師原始 Features 限制 | `13 < 28 < 40`，PASS |
| 缺失值 | 總計 0，各欄皆為 0 |
| 完全重複資料列 | 0 |
| Target `Appliances` | 存在、`int64`，無缺失值、NaN 或正負無窮值 |
| date 可解析性 | 全部可解析，無缺失或無效時間 |
| 日期範圍 | 2016-01-11 17:00:00 至 2016-05-27 18:00:00（原始資料未標示時區） |
| 時間戳記與排列 | 重複 0，依序遞增 |
| 時間間隔 | 600 秒 × 19,734 次，皆為 10 分鐘 |
| date 以外的非數值欄位 | 無 |
| 所有數值欄位 | 無 NaN 或正負無窮值 |
| 檢查狀態 | 24 PASS、1 WARN、0 FAIL |
| Overall Status／退出碼 | **WARN／0** |

唯一 WARN 為 **`rv1` 與 `rv2` 的全部數值完全相同**。兩欄仍完整保留；此驗證不構成 Feature Selection，也未決定後續模型是否使用它們。

| 隨機變數 | dtype | Count | Mean | Standard Deviation（ddof=1） | Min | Max |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| rv1 | float64 | 19,735 | 24.9880334850 | 14.4966338604 | 0.0053216820 | 49.9965296825 |
| rv2 | float64 | 19,735 | 24.9880334850 | 14.4966338604 | 0.0053216820 | 49.9965296825 |

原檔與 `data/raw/` 複本的 SHA-256 均為：

```text
2820bf712ad0275cb18b85a05250926100d8e65ebb9f4d2d016ca91ea152a25d
```

已完成 9 個驗收案例：正式資料、重複執行、錯誤筆數、NaN／正負無窮值、無效日期、缺少 Target、非數值 Target、重複時間戳記與 CSV 不存在。必要檢查失敗均回傳退出碼 1，時間重複的 WARN 回傳 0；重複執行產生一致的嚴格 JSON。測試只使用 `.venv` 內的暫存複本，測試複本已清除，正式 CSV 的 SHA-256 維持不變。

## Git 與資料公開檢查

原始 CSV 每份為 11,979,363 bytes（約 11.42 MiB）。依 [GitHub 官方檔案限制](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)（2026-10-14 查核），瀏覽器上傳單檔上限為 25 MiB，一般 Git 超過 50 MiB 會警告，超過 100 MiB 會阻擋；本檔低於上述限制。Git 僅追蹤 `data/raw/` 的正式 CSV；根目錄原檔依精確檔名忽略，但本機實體檔案保留。

`.gitignore` 排除虛擬環境、Python 快取、macOS 檔案、Notebook checkpoint、`.env` 與根目錄重複 CSV，保留 `.env.example` 的例外。本機首次 Commit 使用 `bambamboo0321` 與 GitHub noreply Email，沒有設定遠端或執行 Push。

已對兩份 CSV、程式與文件檢查常見私鑰、AWS／GitHub／OpenAI 金鑰及電子郵件樣式，未發現符合樣式的內容。資料欄位為觀測時間、能源消耗、溫濕度、天氣與隨機變數，沒有姓名、地址、聯絡方式或帳號識別欄位。資料來源已依 UCI 官方原檔完整比對確認，README 保留 CC BY 4.0 署名與授權資訊。

## 後續工作範圍

**Phase 2–4 尚未執行。** 未進行 Feature Selection、資料切分、模型訓練、Model Evaluation、95% 區間預測圖、NotebookLM 文獻研究或至少 100 字摘要，也尚未匯出 GPT 對話 PDF。本 README 僅記錄環境與原始資料驗證，不包含虛構的模型成果。後續報告仍需說明 GPT 輔助過程並附相關對話 PDF。
