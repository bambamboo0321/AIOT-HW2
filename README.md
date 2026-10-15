# AIOT HW2：Python Machine Learning

本專案研究 Appliances Energy Prediction 資料集，目標為了解室內溫濕度、照明與外部天氣等觀測資訊，能否用於預測家電能源消耗 `Appliances`。依作業要求使用 sklearn Multiple Linear Regression（多元線性迴歸）。目前完成 Phase 1「環境建立與資料驗證」，本次 Phase 2 為「Data Understanding & Configurable Preprocessing」。目標是逐步建立 Interactive Multiple Linear Regression & Feature Selection Lab；本次只交付 Python 核心、EDA、設定契約及驗收，尚未開始正式五種 Feature Selection、前端或部署。

接手時 main 的 HEAD 為 `8839430c2defa99a961976cb002cc94390a904c0`，但 working tree 已有未提交的 Baseline／Notebook／模型評估。這些檔案保留供追溯，本次未重跑其正式訓練或評估。現在的驗收入口是 [Phase 2 主報告](reports/phase2/phase2_report.md) 與 [Wizard 契約](docs/preprocessing_wizard.md)。Notebook 已另完成同步收尾，最新執行與安全檢查見 [Notebook 同步驗收](reports/phase2/notebook_sync_verification.json)；主報告中「未執行 Notebook runner」描述的是先前核心開發階段。既有 Baseline 模型與 Metrics 保持不變。

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
| Business Understanding（商業理解） | 確認家電能源消耗研究目標與作業限制 | 已有背景文件；本次範圍見 Phase 2 主報告 |
| Data Understanding（資料理解） | 原始品質驗證、分布、相關性與時間 EDA | Phase 1 驗證與 Phase 2 EDA 已完成 |
| Data Preparation（資料準備） | 可設定補值、編碼、縮放、日期、異常值與 fold 內前處理 | Phase 2 已完成；正式 Feature Selection 留待後續 |
| Modeling（建模） | 最終固定使用 sklearn LinearRegression | 接手前已有 Baseline；本次未新增正式模型 |
| Evaluation（評估） | CV 比較與最終評估 | 接手前已有評估檔；本次只驗收前處理核心 |
| Deployment（部署） | 彙整可重現的交付成果與研究報告 | 尚未執行 |

CRISP-DM 的六階段是研究方法論，與作業的 Phase 1–4 執行範圍不同。

## 原始 Features 與模型 Features

老師規定原始輸入 Features 數量必須 **>13 且 <40**。原始欄位共 29 個，扣除 Target `Appliances` 後，原始輸入 Features 為 **28** 個，包含 `date`。

| 統計口徑 | 排除的欄位 | 數量 |
| --- | --- | ---: |
| Original Input Feature Count | `Appliances` | 28 |
| Candidate Feature Count（Exclude date） | `Appliances`、`date` | 27 |
| Candidate Feature Count（Exclude date, rv1, rv2） | `Appliances`、`date`、`rv1`、`rv2` | 25 |

Phase 1 的候選數量只依欄名統計，沒有執行 Feature Selection，也沒有從資料中刪除欄位。本次預設前處理示例使用 25 個數值特徵，排除 Target、date、rv1、rv2；使用者可覆寫排除與型態。未依測試分數挑選設定。加入 date 的 calendar 示例會得到 29 個轉換欄位。Phase 3 的 k 一律指轉換後數值欄位數，原始欄位 mapping 另行保存。

`date` 保留供時間切分，另提供 calendar／cyclical 衍生選項，不能直接當作 Unix Timestamp 數值。`rv1`、`rv2` 為完全相同的隨機變數，建議排除且允許覆寫；沒有在本階段正式比較其預測表現。

## 專案結構

- `data/raw/`：唯一 Git 追蹤的原始 CSV。
- `src/validate_data.py`：原 Phase 1 驗證，保持未修改。
- `src/eda.py`：可重現 EDA；`phase2_common.py` 沿用共用 I/O。
- `src/profiling.py`：一般 CSV 剖析與保守型態推測。
- `src/preprocessing_config.py`：Wizard v1 設定、驗證、建議、保存。
- `src/preprocessing.py`：sklearn Pipeline、IQR、日期與 Feature mapping。
- `src/data_splitting.py`：random／chronological、X/y 同步過濾、fold-local 前處理。
- `src/run_preprocessing.py`、`src/phase2_report.py`：能源示例與主報告。
- `tests/test_preprocessing.py`：本次核心測試；`test_phase2.py` 是既存 Baseline 回歸測試。
- `docs/preprocessing_wizard.md`：前端無關的 JSON 契約、API、k 定義與限制。
- `reports/phase1/`：原驗證報告，保持未修改。
- `reports/phase2/`：EDA 圖表、profile、設定、mapping、主報告及驗收紀錄。
- `src/train_baseline.py`、`reports/phase2/models/`、predictions／baseline_metrics 等：既有 Baseline 成果；新增唯讀核對函式供 Notebook 重用，不改變訓練邏輯或模型。
- `src/run_phase2_notebook.py`、Notebook：全新 Kernel 執行目前 EDA／前處理，核對既有 Baseline，不重新 fit 或寫入模型。

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

`requirements.txt` 記錄實際安裝並驗證成功的分析、模型保存與 Notebook 執行直接依賴版本；傳遞依賴由 pip 解析。跨平台或其他 Python 版本尚未驗證。

本機實際安裝並成功匯入的版本：

| 項目 | 版本 |
| --- | --- |
| Python | 3.10.17 |
| pandas | 2.3.3 |
| numpy | 2.2.6 |
| scikit-learn | 1.7.2 |
| matplotlib | 3.10.9 |
| joblib | 1.6.0 |
| nbformat | 5.11.1 |
| nbclient | 0.11.0 |
| ipykernel | 7.3.0 |

`pip check` 通過，沒有依賴衝突。Phase 2 的圖表程式以 `.venv/matplotlib-config` 作為 Matplotlib 設定快取位置；Notebook 執行的 Jupyter／IPython 設定與快取也限定在 `.venv`。

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

## Phase 2 執行與驗收

沿用現有 `.venv` 與 requirements，不需新增套件。於專案根目錄執行：

```bash
.venv/bin/python -m src.eda
.venv/bin/python -m src.run_preprocessing
.venv/bin/python -m src.phase2_report
.venv/bin/python -m src.run_phase2_notebook
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m pip check
```

前兩個命令更新 EDA／前處理報告，第三個由實際 JSON 統計重建繁體中文主報告；Notebook runner 以全新 Kernel 顯示 9 張 EDA 與 7 張評估圖，並核對已儲存模型的指標。這些流程不訓練正式模型或修改原始 CSV。測試中的 LinearRegression 只作合成資料整合與既有 Baseline 回歸驗收，不輸出新模型評估檔。`train_baseline.py` 的 CLI 仍會重新訓練並寫入模型，Notebook 同步驗收不使用該入口。Notebook runner 已改為透過共用函式載入既有模型，重算並核對預測指標但不改寫 Baseline；以 SHA-256 檢查 CSV、Phase 1、模型、Metrics、預測檔與切分索引未變。

若驗證 Phase 1，請將輸出另外存到忽略目錄，以保留原報告：

```bash
.venv/bin/python -m src.validate_data --report .venv/phase2_checks/phase1_validation.json
```

目前核心預設為 25 個轉換欄位，calendar 示例為 29。訓練 15,788／測試 3,947；只在 training fit，test 只 transform。詳細統計、時間依賴、範圍差異與限制見 [Phase 2 主報告](reports/phase2/phase2_report.md)。[verification.json](reports/phase2/verification.json) 保留核心開發階段的驗收；[notebook_sync_verification.json](reports/phase2/notebook_sync_verification.json) 記錄本次 Run All、重跑測試與 Git 安全檢查；[repository_review.json](reports/phase2/repository_review.json) 記錄本次與接手前檔案差異。

設定示例見 [energy_config.json](reports/phase2/energy_config.json)，使用者可更換 Target／Features、覆寫型態、選擇補值／編碼／縮放／異常值／日期策略，以及 random／chronological split。k 定義、原始欄位 mapping、刪列的 X/y 對齊與 CV 限制均在 [Wizard Contract](docs/preprocessing_wizard.md)。

接手前已有 holdout 評估報告，該資料不能再宣稱是完全未查看的最終測試集。後續正式比較只使用 training CV；若需全新最終泛化證據，應使用另取得的獨立時間資料。舊評估檔保留，但本次不以其分數調參或宣稱演算法優劣。

## 後續工作範圍

Phase 3 的五種 Feature Selection 比較、巢狀 CV、Performance 曲線與模型挑選尚未實作。Dashboard、上傳介面、Vercel、Prediction Playground、95% 區間與 PDF 均留待後續。Phase 2 完成後停止，等待人工驗收；不 Commit、Push 或部署。
