# Phase 4B-1：外部研究來源清單

查核日期：2026-10-16。用途：HW2 公開解法比較及 NotebookLM 匯入準備。本文件由 GPT 協助搜尋、閱讀與整理，**不是 NotebookLM 摘要**；本階段未操作 NotebookLM、未執行外部程式、未重新訓練或重現外部實驗。

`VERIFIED` 表示已閱讀相應的官方說明、原始程式或論文內容；不表示已獨立重現分數。`UNVERIFIED` 表示指定資訊尚無足夠依據。未提供的分數不以估計補齊。資料集同源依來源說明及欄位確認，未核對各外部 CSV 與本專案 CSV 的逐位元相同性。

## D1 — UCI 官方資料集

| 欄位 | 查核內容 |
| --- | --- |
| 標題 | Appliances Energy Prediction |
| 作者 | Luis Candanedo；發布平台 UCI Machine Learning Repository |
| 網址 | [官方資料說明](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)；[資料集 DOI](https://doi.org/10.24432/C5VC8G) |
| 日期 | 捐贈日期 2017-02-14 |
| Dataset／Target | 19,735 筆、28 個輸入特徵；`Appliances`，單位 Wh；每 10 分鐘觀測，約 4.5 個月 |
| 模型 | 不適用：資料說明，不是模型實驗 |
| 切分 | 不適用：未規定固定 train/test 切分 |
| 評估指標 | 不適用 |
| 主要結果 | 記錄住宅溫濕度、照明及天氣；兩個隨機變數用於檢查非預測性輸入；頁面連到原始論文與作者程式 |
| 比較適用性 | VERIFIED；適合確認背景、變數、單位及資料來源，不能當作模型成績 |

證據位置：官方頁面 Dataset Information、Variables Table、Introductory Paper、Creators、Citation。官方標示 CC BY 4.0。勿把 Wh 寫成 W，或把包含 target 的 29 欄寫成 29 個輸入特徵。

## D2 — 本專案使用的 Kaggle 資料頁

| 欄位 | 查核內容 |
| --- | --- |
| 標題 | Appliances Energy Prediction |
| 作者 | 上傳帳號 `loveall`，目前顯示名稱 Gokagglers；資料原作者依頁面致謝為 Luis Candanedo 等人 |
| 網址 | [Kaggle Data Card](https://www.kaggle.com/datasets/loveall/appliances-energy-prediction/data) |
| 日期 | 頁面 JSON-LD `dateModified`：2017-09-16T10:43:26.9Z；version 1 |
| Dataset／Target | Appliances Energy Prediction；家電耗能 `Appliances`，Wh |
| 模型 | 不適用：資料頁 |
| 切分 | 未提供固定實驗切分 |
| 評估指標 | 不適用 |
| 主要結果 | 說明 10 分鐘資料、約 4.5 個月、ZigBee 感測、天氣合併及隨機變數，並引用 P1 |
| 比較適用性 | VERIFIED；用於交代 Kaggle 來源，不是獨立模型研究 |

證據位置：Data Card 的 Context、Content、Acknowledgements，以及公開頁面的 Dataset JSON-LD。Kaggle 顯示的授權欄為「Other (specified in description)」，不要直接把平台欄位改稱 CC BY；UCI 的授權另見 D1。下載可能要求 Kaggle 登入，本次未登入或下載至 Repository。

## P1 — 資料集原始研究論文

| 欄位 | 查核內容 |
| --- | --- |
| 標題 | Data driven prediction models of energy use of appliances in a low-energy house |
| 作者 | Luis M. Candanedo、Véronique Feldheim、Dominique Deramaix |
| 網址 | [DOI](https://doi.org/10.1016/j.enbuild.2017.01.083)；[出版社](https://www.sciencedirect.com/science/article/pii/S0378778816308970)；[作者所屬大學典藏](https://orbi.umons.ac.be/handle/20.500.12907/23357)；[典藏全文 PDF](https://orbi.umons.ac.be/bitstream/20.500.12907/23357/1/1-s2.0-S0378778816308970-main.pdf) |
| 日期 | 2017；Energy and Buildings 140，81–97。Kaggle 致謝列期刊日期 2017-04-01；典藏欄位另列 2017-05-19，兩者有差異，引用以年份／卷頁／DOI 為準 |
| Dataset／Target | 本資料集的原始低耗能住宅觀測；家電耗能，Wh |
| 模型 | Multiple Linear Regression、radial SVM、Random Forest、Gradient Boosting Machine |
| 切分 | 作者程式 C1：以 target 分層抽樣約 75% training／25% testing；訓練內 10-fold CV、重複 3 次；不是 chronological holdout |
| 評估指標 | 作者 C1 記錄 RMSE、MAE、MAPE、R² |
| 主要結果 | 原研究摘要報告 GBM training R² 約 0.97、testing 約 0.57；C1 註記 GBM testing RMSE 66.65093 Wh |
| 比較適用性 | VERIFIED（書目、摘要與作者程式互相佐證）；適合研究背景及方法比較，切分和特徵不同，不能直接與本專案排名 |

查核界線：典藏書目可讀；全文 PDF 以公開 HTTP 取得 200、PDF 格式及 6,095,546 bytes，但自動文字擷取逾時，**本次未完整逐頁閱讀 P1**。模型設定與詳細數字另以 C1 原作者程式為依據，不假稱已核對全文所有表格。出版社直接讀取曾回傳 403，典藏可作使用者取得全文的替代途徑。

## C1 — 原作者 R 程式及實驗紀錄

| 欄位 | 查核內容 |
| --- | --- |
| 標題 | Appliances-energy-prediction-data／Data driven models.R |
| 作者 | GitHub 帳號 LuisM78；D1 官方連結指向此 Repository |
| 網址 | [Repository](https://github.com/LuisM78/Appliances-energy-prediction-data)；[本次查核的固定版本 R 程式](https://github.com/LuisM78/Appliances-energy-prediction-data/blob/e3e4c27a4ae2b41b21f84b4c9ac2d822d3d0ace1/Data%20driven%20models.R) |
| 日期 | 本次所見最新 commit `e3e4c27a4ae2b41b21f84b4c9ac2d822d3d0ace1`，2017-02-08T10:50:05Z |
| Dataset／Target | `energydata_complete.csv`；原始 `Appliances`，Wh |
| 模型 | caret 的 `lm`、`svmRadial`、`rf`、`gbm`；另有 Boruta／RFE 及不同輸入子集分析 |
| 切分 | `set.seed(1)`；`createDataPartition(..., p=0.75)`；訓練內 repeated 10-fold CV，3 repeats |
| 評估指標 | 自訂 `Rsqu` 為 `1-SSE/SST`；另計算 RMSE、MAE、MAPE |
| 主要結果 | 程式註解的 testing RMSE：LM 93.17608、SVM 70.74、RF 68.48478、GBM 66.65093 Wh；是作者保存的紀錄，本次未重跑 |
| 比較適用性 | VERIFIED（程式與保存註解）；可追溯 P1 方法。C1 與 P1 是同一研究的證據，不算兩個獨立研究 |

證據位置（上述固定版本）：約第 351–385 行切分；479–497 行去除 `rv1`／`rv2` 與 weekday／week-status dummy；562–565 行 CV；690 行起 R² 定義；約 880–997 行 training/testing 評估及分數註解。SVR 使用 center/scale。研究包含時間衍生及 dummy 特徵，不能把它稱為本專案相同的 25 個原始特徵實驗。

## P2 — 後續同資料集論文

| 欄位 | 查核內容 |
| --- | --- |
| 標題 | Data-Driven Modeling of Appliance Energy Usage |
| 作者 | Cameron Francis Assadian、Francis Assadian |
| 網址 | [DOI](https://doi.org/10.3390/en16227536)；[出版社](https://www.mdpi.com/1996-1073/16/22/7536)；[公開全文 PDF 備援](https://pdfs.semanticscholar.org/d9da/0ea38146d389ce5bb6d75e4f08cd63cdb54a.pdf) |
| 日期 | 2023-11-12；Energies 16(22)，7536 |
| Dataset／Target | UCI Appliances Energy Prediction；家電耗能，Wh |
| 模型 | LM、SVR、RF、Gradient Boosting、XGBoost、Extra Trees |
| 切分 | §3.2.1 說明 10-fold CV 及 random search；Table 2 標示 testing set。獨立測試集比例、抽樣 seed、是否依時間排序：UNVERIFIED（所讀全文未交代清楚） |
| 評估指標 | RMSE、R²、MAE、MAPE |
| 主要結果 | Table 2：Extra Trees testing RMSE 59.61 Wh、R² 0.66、MAE 26.62 Wh；其餘模型見比較文件 |
| 比較適用性 | VERIFIED（全文 §2、§3.1–3.2.1、Table 2）；可比較模型／特徵工程思路，不能認定與本專案同一測試條件 |

方法重點：時間衍生及 hour sine/cosine 等特徵工程。限制：未核實到可重現整套實驗的作者程式；Table 2 成績與 10-fold CV 平均不可混稱。出版社曾回傳 429，本次以公開的論文本體 PDF 閱讀；備援不是另一篇研究。文中提到 XGBoost 的 L1/L2，不代表另有獨立 Ridge／Lasso 實驗。

## K1 — 可閱讀程式與輸出的 Kaggle Notebook

| 欄位 | 查核內容 |
| --- | --- |
| 標題 | Appliances Energy Consumption Prediction-LR/SVR |
| 作者 | Lakmali Gamage；Kaggle 帳號 `lakmaligamage` |
| 網址 | [Notebook](https://www.kaggle.com/code/lakmaligamage/appliances-energy-consumption-prediction-lr-svr) |
| 日期 | JSON-LD 發表／更新 2022-04-27T16:52:42.9033333Z；Version 1 of 1，執行版本 ID 94168769 |
| Dataset／Target | 另一個 Kaggle 掛載路徑的 `energydata_complete.csv`；原始同源家電資料；實際建模 target 是 `np.log(Appliances)` |
| 模型 | sklearn LinearRegression、`SVR(gamma="scale")` |
| 切分 | 刪除 `Appliances>800` 或 `<0` 後共 19,721 筆、22 features；random 80/20，seed 42；15,776／3,945 筆；StandardScaler 只先 fit training |
| 評估指標 | holdout MAE、R²；另有 MAPE 衍生的「Accuracy」及 CV 輸出，均需依程式解讀 |
| 主要結果 | Cell 43：LR MAE 0.3854（log 尺度）、R² 27.92%；SVR MAE 0.2962（log 尺度）、R² 47.52%。RMSE 未在此評估輸出提供 |
| 比較適用性 | VERIFIED（公開程式／輸出）；可作 NotebookLM 方法批判案例，不可與本專案 Wh 誤差直接比較 |

證據位置：Cells 17、21、31–43。Cells 34–37 提供樣本數與切分；Cells 42–43 定義／輸出 holdout 指標。這是自然對數 target 的 R²，亦不可當成原 Wh 尺度的同等分數。

Cell 44 的問題：random split 後才做 TimeSeriesSplit，不代表依原始時間驗證；scaler 已先用整個 training fit，並未在各 CV fold 內重學；標示「SVR」的 CV R² 呼叫的是 `lin_model`；CV「Accuracy」其實為 `100 + neg_MAE`。這些結果不納入有效 CV 成績比較。Cell 42 的 holdout「Accuracy」另為 `100-MAPE`，也不是分類正確率。原輸出把誤差單位寫成 degrees，不適用這個 log 能源 target。

## C2 — 含 Ridge／Lasso 的公開實作（附品質限制）

| 欄位 | 查核內容 |
| --- | --- |
| 標題 | Appliance Energy Prediction（Repository：Appliances-Energy-Prediction） |
| 作者 | 公開維護帳號 `mhadeli`；README 另列 Credit：`Thomas-George-T`；未推定帳號的個人姓名或全部程式原創歸屬 |
| 網址 | [固定版本 README](https://github.com/mhadeli/Appliances-Energy-Prediction/blob/5bdb684b7516fc93899f70559e2d98480516ae51/README.md)；[固定版本 Notebook](https://github.com/mhadeli/Appliances-Energy-Prediction/blob/5bdb684b7516fc93899f70559e2d98480516ae51/Appliances%20Energy%20predicition%20Notebook.ipynb) |
| 日期 | 本次所見最新 commit `5bdb684b7516fc93899f70559e2d98480516ae51`，2024-05-31T04:37:28Z；最初發表日期 UNVERIFIED |
| Dataset／Target | README 明示 UCI、19,735 筆／29 欄、`Appliances` Wh；Notebook 的主要 regression 呼叫傳入 `y_train`，不能僅因另建 log 欄就假定所有成績為 log 尺度 |
| 模型 | 自訂 LR、Ridge、Lasso（含 GD／SGD）、PCA 前後比較、Keras Neural Network；另有特徵重要度／RFE 步驟 |
| 切分 | Notebook Cell 34：兩次 random split，60% training／20% testing／20% validation，seed 23；其他類別另有 split 輔助函式，不混為主要呼叫流程 |
| 評估指標 | 程式輸出 RMSE、標示為 R² 的數值，另有 SSE 及 bias/variance 分析 |
| 主要結果 | VERIFIED：有實際 Ridge／Lasso 程式及執行輸出；標示的 R² 定義有誤，未採其數值作合格模型成績 |
| 比較適用性 | 僅適合方法及品質問題討論；可用於證明找到公開 Ridge／Lasso 實作，不作效能排名；有效、可直接比較的 Ridge／Lasso 分數為 UNVERIFIED |

證據位置：原始 ipynb `cells` 陣列以 0 起算：34（切分）、44／50／55（LR／Ridge／Lasso 類別）、51–58（實際模型呼叫與輸出）、92（手動結果彙表）。`r2_squared` 回傳 `np.exp(r2_score(y,y_hat))`，不是標準 R²；部分輸出超過 1。手動彙表與單次 test 評估的對應也未充分釐清，因此不搬用數字、不自行反推「修正」分數。未執行或修改其程式。

## 來源取捨與待核實項目

- 已確認的獨立研究為 P1、P2；C1 是 P1 的補充證據。K1、C2 是公開實作，不等同同儕審查論文。
- 只在標題、搜尋摘要或 import 出現的模型不當成完成實驗；C2 已查到 Ridge／Lasso 的實際呼叫及保存輸出，但指標品質不足。
- P2 完整 holdout protocol、C2 有效 Ridge／Lasso 成績及外部資料檔逐位元相同性仍為 UNVERIFIED；不得在最終報告消除這些限制。
- 本次不採用搜尋結果中其他未讀全文／程式的 Notebook、論文或競賽分數，避免把改版資料或不同 target 的結果混入。
- 各來源的數值與比較範圍見 [公開解法比較](solution_comparison.md)；NotebookLM 操作仍須由使用者完成，見 [研究準備計畫](notebooklm_research_plan.md)。
