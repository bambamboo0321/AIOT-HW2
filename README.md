# AIOT HW2：家庭家電耗能預測與特徵選擇

本專案研究 **Appliances Energy Prediction Using Multiple Linear Regression and Feature Selection**，依作業要求使用 `sklearn.linear_model.LinearRegression`。已完成 Phase 1～3C 的資料驗證、EDA、可設定前處理、六種 Feature Selection、正式實驗與固定模型的 95% Prediction Interval，以及 M1～M14 的五頁互動網站、自訂數值 CSV 六方法比較、訓練與新資料推論。既有 Phase 4B-2 CLI 與 Dashboard JSON 仍保留。

目前可讀取、核對與匯出既有成果，也可透過 Next.js 本機 API 呼叫 Python 進行自訂 CSV 實驗；使用者已確認本機操作驗收可接受。**獨立雲端後端、Vercel 正式部署、最終 PDF 與 ZIP 尚未完成**。使用者已選定 Vercel 前端＋獨立 Python 後端，部署計畫見下方文件；本機成功不等於雲端已上線。GitHub 是 Optional，不是必要交付條件。

## Dataset 與作業規範

- [Kaggle Dataset：Appliances Energy Prediction](https://www.kaggle.com/datasets/loveall/appliances-energy-prediction/data)
- [UCI Dataset 374](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)；[Dataset DOI](https://doi.org/10.24432/C5VC8G)
- Candanedo, L. (2017). *Appliances Energy Prediction*. 原始研究見 [DOI](https://doi.org/10.1016/j.enbuild.2017.01.083)。UCI 資料頁標示 CC BY 4.0；Kaggle 頁的授權欄另依該頁說明。
- 正式資料：`data/raw/KAG_energydata_complete.csv`，19,735 筆、29 欄；28 個原始輸入、1 個 Target `Appliances`。目標單位 **Wh（能量），不是 W（功率）**。
- 約四個半月、10 分鐘取樣；原始時間未標時區，不自行推定。
- 老師更新後要求：**原始輸入至少 13 個**，本資料的 28 個符合。舊文件的不同門檻屬歷史紀錄；這個要求不適用於選取後的 `k`。
- Baseline 排除 `date`、`rv1`、`rv2` 後使用 25 個 numeric features；保留高耗能真實觀測，不依 Target 刪除異常值。
- 原始 CSV SHA-256：`2820bf712ad0275cb18b85a05250926100d8e65ebb9f4d2d016ca91ea152a25d`。

來源研究、作者與版本見 [external_sources.md](references/external_sources.md)，不同公開實驗的切分及尺度限制見 [solution_comparison.md](references/solution_comparison.md)。

## CRISP-DM 六階段

| 階段 | 已有內容與證據 |
| --- | --- |
| Business Understanding | 研究目的、使用情境與限制：[背景說明](reports/phase2/business_understanding.md) |
| Data Understanding | 一般 profiling、9 張 EDA 圖、偏態／相關性／時間相依：[EDA](reports/phase2/eda_summary.md) |
| Data Preparation | 設定驗證、補值、OneHot、縮放、IQR、日期衍生及 fold-local fit：[Wizard 契約](docs/preprocessing_wizard.md) |
| Modeling | 六種 Feature Selection，下游固定 LinearRegression：[Phase 3B](reports/phase3/phase3b_report.md) |
| Evaluation | 分開報告 Training CV、Nested CV、固定 Holdout、殘差及 OLS PI：[Phase 3C](reports/phase3/phase3c_report.md) |
| Deployment | 已有 CLI、五頁網站、本機 API 與自訂資料訓練／推論；Vercel 前端＋獨立後端待實作雲端接線與部署 |

## Feature Selection 與驗證層級

六種方法為 F-test（`f_test`）、Mutual Information（`mutual_information`）、Forward SFS（`forward_sfs`）、RFE（`rfe`）、LassoCV（`lasso_cv`）、RFECV（`rfecv`）。LassoCV 用於選特徵，最終預測器仍是 LinearRegression。

Phase 3B 保存 114 個正式案例、1,974 筆結果。固定 k 方法的訓練內最終 CV 最佳 k：F-test 6、MI 6、Forward SFS 10、RFE 6。LassoCV／RFECV 自動決定 k，不能把跨 fold 平均 k 當成完整訓練區的選取數；完整訓練區分別選 16／6 欄。詳細分數及 mapping 由 CLI 讀取既有 JSON。

- **Training CV**：訓練資料內 3 folds 選參數／方法；MI 選為最終候選，k=6。
- **Nested CV**：5 個 outer folds 評估各方法，k 可因 fold 改變；不是固定 k=6 模型的同一個分數，也不是從六方法擇優程序的無偏評估。
- **Training fit**：完整訓練區重 fit 後的樣本內指標，不能當成 CV。
- **Holdout**：固定後期 3,947 筆的描述性比較；Training 為前 15,788 筆。此 Holdout 在 Phase 2 已查看，不能宣稱未查看的最終測試集。

所有資料相依前處理及 Selector 只在當前 training fold 學習；不要先對完整資料 fit_transform 再做 CV。模型以同時點感測資訊估計耗能，不是只用過去值進行多步未來預報。

## 固定模型比較與 95% Prediction Interval

下表為同一 chronological Holdout；MAE／RMSE 單位 Wh、R² 無單位。

| 模型 | Features | MAE | RMSE | R² | PI 實測 Coverage | PI 平均寬度 Wh |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline LinearRegression | 25 | 52.014105 | 86.732939 | 0.092352 | 96.0983% | 376.104533 |
| MI + LinearRegression | 6 | 53.605861 | 90.441773 | 0.013067 | 95.8196% | 401.442761 |

MI 選取 `T3`、`T4`、`T5`、`RH_6`、`T8`、`T9`；使用保存的中位數補值與 StandardScaler。Baseline 使用原始數值，沒有 scaling。**特徵數減少，但本次 Holdout 預測表現沒有改善。** Random Split Baseline 僅作另一種切分的對照，CLI 會另外顯示，不混入上述排名。

PI 為單筆新觀測的 OLS Prediction Interval，依 training 殘差與 SVD leverage 計算，未依 Holdout 校準、未裁切負下界；不是平均值 Confidence Interval，也不是整條曲線的 simultaneous band。名目 95% 不保證泛化 Coverage；時間相依、異方差、分布漂移及選特徵的不確定性均須保留說明。

直接依據：[Baseline Metrics](reports/phase2/baseline_metrics.json)、[最佳特徵](reports/phase3/results/best_features.json)、[固定模型指標](reports/phase3/phase3c/prediction_metrics.json)、[PI 指標](reports/phase3/phase3c/prediction_interval_metrics.json)。

## 安裝與執行

既有驗收環境為 Python 3.10.17，套件版本固定在 requirements。從 Repository Root 執行；若已有 `.venv` 可直接啟用，不必重建環境。

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-phase3c.txt
python -m pip check
python hw2_main.py --help
```

`requirements-phase3c.txt` 包含基礎 requirements，以及 PI 合成測試所需的 scipy／statsmodels／patsy；本階段沒有新增依賴。尚未提供完整學號，暫用 `hw2_main.py`；確認後須改成老師指定的 `7114056XXX_hw2.py` 正式檔名。

## Python 主程式

```bash
python hw2_main.py dataset-summary
python hw2_main.py crisp-dm
python hw2_main.py feature-selection
python hw2_main.py model-comparison
python hw2_main.py prediction-intervals
python hw2_main.py figures
python hw2_main.py verify
python hw2_main.py export-dashboard
```

每個摘要命令輸出可解析 JSON；`verify` 會檢查必要檔案、來源 digest、兩個固定模型 digest、資料版本及現有 Phase 1 唯讀驗證，缺少／損壞時以錯誤訊息及非零 exit code 結束。**不重新訓練、不反序列化模型、不補造缺少的成績**。驗證結果不代表已重新執行全部測試。

主程式以自身位置定位 Repository，無私人絕對路徑。它重用現有 src 的特徵方法定義、路徑與 SHA-256／資料驗證函式，未複製模型邏輯。除 `export-dashboard` 外不輸出檔案。

## Dashboard JSON 匯出契約

`export-dashboard` 只寫入 Repository 的 `dashboard_data/`，無任意輸出路徑參數；先驗證輸入及全部輸出目的地，拒絕符號連結，再逐檔原子取代。本功能會更新這五份衍生資料，不修改 reports。沒有時間戳，同一來源重複匯出位元組一致；沒有承諾五檔跨檔案交易原子性，未來服務發布需另設版本快照。

| 檔案 | `data` 主要內容 |
| --- | --- |
| `dataset_summary.json` | dataset metadata、Baseline 輸入、排除欄位及完整既有 profiling |
| `feature_selection.json` | 六方法分開的 `training_cv`、`nested_cv`、`full_training_refit`、winner、曲線及案例／紀錄數 |
| `model_comparison.json` | 兩固定模型 `training_metrics`／`holdout_metrics`、切分期間、轉換方式；Random Baseline 另列 |
| `prediction_intervals.json` | 單筆 PI 公式、training-only 來源、名目／實測 Coverage、寬度及限制 |
| `figures_manifest.json` | 28 張既有圖的分類、Repository 相對路徑、SHA-256 及既有繪圖來源紀錄 |

共用 envelope：`schema_version`（目前 1）、`artifact_type`、`sources[]`（相對路徑與 SHA-256）、`units`、`missing_fields`、`data`。匯出保留 JSON 原始浮點精度；Coverage 使用 0～1 比例，前端顯示百分比時乘 100；MAE／RMSE／寬度 Wh，MSE Wh²，runtime 秒。`k` 為轉換數值欄位數。

缺少必要來源即報錯，不自動重訓或填值。六方法全部的 Holdout 比較並不存在，於 `missing_fields` 明示；不得將 Training CV／Nested CV／Holdout 混排，也不得替未評估方法產生成績。圖表路徑基準是 Repository Root，將來網站需明確映射資產，不能假定路徑已公開可訪問。

歷史 Backend 與上傳規劃見 [dashboard_backend_plan.md](docs/dashboard_backend_plan.md)；其中未實作狀態屬先前階段。現已完成本機數值 CSV 的「上傳 → profiling → 選 Target／前處理 → 確認 → 驗證 → 比較 → 再確認訓練 → 評估／推論」，現況以 [web/README.md](web/README.md) 的 M14 與 [操作指南](docs/user_guide.md) 為準。

## 專案目錄

```text
hw2_main.py                    暫定主程式
src/                           既有 ML 模組與 dashboard_data.py 成果介接
notebooks/                     Phase 2 已執行 Notebook
data/raw/                      正式原始 CSV，不回寫
reports/phase1/                資料驗證
reports/phase2/                EDA、Baseline、模型、預測與驗收
reports/phase3/                Benchmark、正式選特徵及 Phase 3C PI 成果
references/                    公開來源、比較與 NotebookLM 研究計畫
docs/                          前處理、實驗契約與 Backend 規劃
dashboard_data/                五份可重現的前端資料 JSON
tests/                         Phase 2～4B-2 測試
requirements.txt               固定基礎依賴
requirements-phase3c.txt        PI 與完整測試依賴
```

## 驗收與目前完成狀態

```bash
python -B -m unittest discover -s tests -v
python -B -m unittest discover -s tests -p test_hw2_main.py -v
python hw2_main.py verify
git diff --check
```

新增測試涵蓋 CLI、各摘要、驗證層級、JSON 契約、決定性匯出、缺檔／格式／雜湊錯誤、寫入範圍、symlink 防護與原始成果完整性。既有測試會在合成／隔離測試情境呼叫 fit；它們不是重跑能源正式 114 案例，也不應重寫既有模型或 reports。

歷史驗收：Phase 3C 記錄既有 115/115 tests PASS、Phase 1 為 24 PASS／1 WARN／0 FAIL；WARN 為已知 `rv1`／`rv2` 重複。本階段的實測結果另以本次命令輸出及人工驗收回報為準，不覆寫歷史 verification。

已完成：Phase 1～3C、公開研究準備、成果 CLI、Dashboard JSON 匯出、M1～M14 本機網站與自訂 CSV 流程，以及操作指南／成果報告 Markdown 草稿。9 張 EDA、7 張 Baseline 評估、7 張選特徵與 5 張 Phase 3C 圖皆沿用既有成果，不重新產圖。M14 測試紀錄為網站 127／127、Python 211／211，詳見 web/verification/m14-result.json；本次文件整理未重跑該批測試。

待完成：完整學號命名、NotebookLM 實際摘要的正式保存與來源留存、GPT 輔助範圍與全部對話 PDF、最終報告 PDF、學號 ZIP，以及獨立雲端後端與 Vercel 部署；網站與本機 API 已完成。使用者已實際取得 NotebookLM 摘要，但尚未正式保存至 Repository；`references/notebooklm_research_plan.md` 為先前準備階段紀錄，**不得把 GPT 回答當成 NotebookLM 成果，也不得自行補寫摘要**。

## 操作、成果與交付文件

- [操作指南](docs/user_guide.md)：頁面導覽、CSV 到預測的操作、k／CV／Holdout／三指標與圖表解讀。
- [成果報告草稿](docs/final_report_draft.md)：引用既有能源結果，保留限制、個人資料及附件待補項目；尚非正式 PDF。
- [交付核對清單](docs/submission_checklist.md)：依既有紀錄暫定核對，仍需老師原始完整作業單、學號與實際 NotebookLM 產物。
- [Vercel 部署計畫](docs/vercel_deployment_plan.md)：使用者選定完整 CSV 雲端功能，列出前端／後端接線、容量及模型管理改動；尚未部署。

本次未 commit 或 push，未修改既有研究結果與模型。

### M15：獨立運算後端封裝

新增 [backend/README.md](backend/README.md)、HTTP gateway、容器檔與最小 Python 依賴，重用既有 Python 模組；提供服務驗證、分 session 模型暫存及有界運算。原網站與研究算法不變，網站尚未接到此獨立服務，Vercel 代理與公開部署仍待後續完成。容器尚未實測，原生 HTTP 整合檢查見 web/verification/m15-backend-result.json。
