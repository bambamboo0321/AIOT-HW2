# Dashboard Backend 規劃（Phase 4B-2）

本文件是未來實作建議，**尚未建立 Python API、CSV 上傳網站、Next.js 專案或正式部署**。本階段只有既有成果的 Python CLI 與 `dashboard_data/` 匯出。以下 API 名稱及配額是草案，不代表目前可呼叫的服務或已完成的效能測試。

## A. 已有且可重用的功能

| 模組／契約 | 可重用內容 | 適用界線 |
| --- | --- | --- |
| `src/profiling.py` | `read_csv`、`profile_dataframe`、`profile_csv`；欄位型態、缺失、唯一值、IQR 候選、常數／ID／高基數旗標 | 全資料描述性剖析；不把統計量直接用於正式前處理；不得自動決定 Target |
| `src/preprocessing_config.py` | `PreprocessingConfig.from_dict`、`validate_columns`、`numeric_target`、`recommend_config` | 數值迴歸；拒絕未知設定、非法 Target；建議需使用者審閱 |
| `src/preprocessing.py` | `resolve_types`、`build_preprocessor`、`feature_summary`；補值、OneHot、縮放、IQR keep／flag／clip、日期衍生 | 純 Transformer 不得刪列；轉換後欄位到原始欄位有 mapping |
| `src/data_splitting.py` | `split_dataset`、`prepare_fold`、`iter_preprocessed_folds` | X/y 同步過濾；IQR removal 僅刪 training；重複時間戳的群組切分未支援 |
| `src/feature_selection.py`、`src/selection_runner.py` | 六種 Selector、fold-local 前處理、獨立 LR、nested CV | 目前 Phase 3 Selector 契約限一對一 numeric；含 OneHot／日期衍生的一般上傳資料不能宣稱已完整支援 |
| `src/phase3b_runner.py` | timeout、checkpoint、Training 區域隔離的參考流程 | 正式 runner 綁定本能源資料與固定實驗；需抽出一般 job 介面，不能直接把任意上傳資料交給正式 runner |
| `src/phase3c_evaluation.py`、`src/prediction_interval.py` | 可信的保存模型預測、數值穩定的 OLS 單筆 PI | 評估入口綁定既有模型／切分；一般資料仍需設計模型契約、假設檢查及拒絕條件 |
| `src/dashboard_data.py`、`hw2_main.py` | 既有結果摘要、來源 SHA-256、相對圖表路徑、固定五份 JSON 匯出 | 不接受上傳、不訓練、不載入 joblib；只展示已保存成果 |

完整前處理契約見 [preprocessing_wizard.md](preprocessing_wizard.md)。該文件保留 Phase 2 當時的狀態與舊作業門檻；目前六種 Selector 已於 Phase 3 完成，老師更新後的 Dataset 原始輸入下限是 **至少 13 個**。原始輸入數與轉換後 selection `k` 必須分開。

## B. 尚缺少的功能

- HTTP API、上傳儲存、使用者／工作隔離、授權、刪除與保存期限管理。
- schema／設定審閱介面、明確訓練確認、排程、取消、進度、資源配額及失敗重試政策。
- 可接收一般 Dataset 的實驗 runner；OneHot、日期衍生、高基數輸入與各 Selector 相容性檢查。
- 成果版本、已 fit Preprocessor／Selector／LR 一起保存與輸入 schema 驗證；不得只保存未 fit 的設定 JSON。
- 一般資料的新樣本預測 API、刪列接受／拒絕遮罩，以及 PI 適用性檢查。
- Next.js／TypeScript／Tailwind／Recharts 的讀取、顯示及錯誤狀態；靜態 JSON 並不是已建好的訓練 Backend。

## C. 未來 API 建議（未實作）

| 方法與路徑草案 | 輸入／輸出重點 | 是否可啟動訓練 |
| --- | --- | --- |
| `GET /api/reference-results/{artifact}` | allowlist 對應五份 dashboard JSON；schema_version、units、sources、data | 否 |
| `POST /api/datasets` | CSV streaming 上傳；回傳隨機 dataset_id、digest、大小與處理狀態 | 否 |
| `GET /api/datasets/{id}/profile` | 欄位／型態／缺失／IQR 候選；不回傳完整原始資料 | 否 |
| `POST /api/datasets/{id}/config-preview` | Target、Features、型態覆寫、前處理與切分；回傳驗證錯誤、警告與設定摘要 | 否 |
| `POST /api/experiments` | dataset_id、已審閱設定 digest、方法與預算、明確 `confirmed=true`；回傳 job_id | **只有此處可排入訓練** |
| `GET /api/jobs/{id}` | queued／running／completed／failed／cancelled、進度及有限的錯誤資訊 | 否 |
| `POST /api/jobs/{id}/cancel` | 中止 worker；留下已完成／未完成的明確狀態 | 否 |
| `GET /api/jobs/{id}/results` | Training CV、Nested CV、Training fit、Holdout 分開；無效成績使用 null＋status | 否 |
| `POST /api/models/{id}/predict` | 已驗證新資料；只 transform／predict，回傳接受／拒絕列與預測 | 否，不允許 refit |

確認以 dataset digest＋config digest＋使用者範圍綁定；設定變更後舊確認失效。排程採 idempotency key 防止重複送出，不允許 GET／profiling／預覽或上傳完成事件自動啟動訓練。API 不允許客戶端指定伺服器檔案路徑、模型檔或任意 Python 程式。

## D. 使用者操作流程

1. 上傳 CSV，檢查檔案限制及欄名；顯示驗證失敗原因。
2. 系統只做描述性 profiling，顯示型態建議、缺失、重複與 IQR 候選；使用者可覆寫型態。
3. 使用者選 Target 與原始 Features，檢查數值迴歸合法性及原始特徵數要求。
4. 使用者選前處理、切分、CV、Feature Selection 方法與預算；預覽欄位擴張、刪列政策及限制。
5. 顯示最終設定與資料版本；**使用者明確確認後才建立 training job**。
6. 系統依隔離的 Training 執行 fold-local 前處理、選特徵與 LR；Holdout 不參與選模。
7. 顯示 CV／Nested CV／Holdout 指標、選取欄位、預測與殘差；若條件成立再展示 PI、Coverage 及寬度。
8. 使用已保存的完整轉換流程預測新資料；未接受列明確回報，不悄悄減少輸出筆數。

## E. 資料驗證與防止 Data Leakage

- profiling 只描述；不得重用全資料平均數、中位數、IQR 界限、縮放或類別字典作為訓練規則。型態覆寫由使用者明確固定，尚未覆寫的型態由當前 training fold 決定。
- Target 不得出現在 Features 或 split_column；缺失 Target 只能拒絕或依明確規則刪列，永不補 Target。拒絕常數、全缺失、無窮值、日期或非數值 Target。
- 固定 outer split 後才 fit；每層 CV 的 imputer／scaler／encoder／IQR／Selector 都只看當前 training fold。內層 SFS／Lasso／RFECV 也不能使用 outer training 全量預先 fit 的轉換矩陣。
- 會刪列的設定走 `prepare_fold`，維護 X/y 索引與刪列 audit；IQR remove 僅作用 training，評估保留極端值。若因 Feature missing drop 刪評估列，須顯示覆蓋率及分數母體。
- 時間資料預設提議 chronological，保留 split_column，驗證順序、唯一時間戳及 gap。重複時間戳在群組切分實作前拒絕；不將打亂後的列直接視為時間序列。
- 先宣告評估規則及預算；只用 training CV 選方法與 k。Nested outer 分數是各方法程序評估，不宣稱「六方法擇優」的無偏估計。Holdout 被查看後，需記錄其已查看狀態。
- `k` 指轉換數值欄位數；OneHot、日期及 flag 會擴張欄位。一般資料在 Selector 相容性未完成前，不能自動丟棄類別欄或強制套用本專案 25 欄設定。
- OLS PI 假設、可估計性、正殘差變異及 post-selection 不確定性均需說明；不依 Holdout 校準、不把高 Coverage 當成準確率，也不以平均值 CI 冒充單筆 PI。

## F. CSV 上傳安全性與資源限制（待壓測的初始建議）

以下是未來 API 的提案，**目前尚未強制執行，並非實測容量**：

| 項目 | 初始提案與處理方式 |
| --- | --- |
| 上傳大小 | 每檔最多 25 MiB；streaming 計數，超限立即拒絕；不只信任 Content-Length |
| 資料規模 | 最多 100,000 列、200 原始欄；解析過程即檢查，避免先無限制載入 |
| 格式 | 僅 CSV、UTF-8／UTF-8 BOM；空檔、重複／空白欄名、非法編碼、超長欄位或不一致欄數明確拒絕；不收 ZIP、pickle、joblib 或可執行內容 |
| 欄位擴張 | 初始上限 2,000 個轉換欄；類別數預檢，高基數先要求排除／重新設定；不能無限制 sparse → dense |
| 記憶體 | 先估 dense 矩陣與副本成本，預估超過 512 MiB 拒絕；worker 初始硬上限 1 GiB，實際值需依部署環境修訂 |
| 計算 | 每使用者最多 1 個執行中 job，初始全服務 1 worker；job 最長 10 分鐘，昂貴 selector 另設較小的 k grid／候選 fit 預算；提供取消與 timeout |
| 儲存 | 使用隨機 ID，原始檔名只作顯示；私有目錄、避免路徑穿越及符號連結，不存入版本庫或公開靜態目錄；建議完成後 24 小時到期，可由使用者提前刪除 |

檔案副檔名與 MIME 只作初步檢查，仍須實際解析。CSV 欄位是資料而非命令，不執行巨集、公式或任何程式；若將使用者文字再輸出給試算表，另處理 formula injection，不回寫原始上傳內容。前端顯示欄名需 HTML escaping；日誌不包含逐列資料、Secrets 或私人絕對路徑。

未來 worker 應限制檔案／網路權限、CPU／RAM／時間、回傳受控錯誤，不把 HTTP request 長時間綁在 SFS 運算上。提供每工作成果隔離及授權檢查；使用者上傳的任意序列化模型永不反序列化。正式安全與效能驗收屬未來 API 階段，本階段不啟動該服務。
