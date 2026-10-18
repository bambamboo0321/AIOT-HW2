# Phase 4B-1：NotebookLM 研究準備計畫

準備日期：2026-10-16。**狀態：NEEDS_USER_INPUT。** 本階段只完成公開來源與問題準備，未登入／操作 NotebookLM，也沒有產生或代寫 NotebookLM 摘要。本文件由 GPT 協助整理。

來源代號、日期、作者、版本、查核界線見 [來源清單](external_sources.md)；本專案和外部結果的可比性見 [公開解法比較](solution_comparison.md)。下述匯入方法是建議途徑，尚未在使用者帳號驗證能成功擷取全部內容。

## 建議匯入來源

| 優先順序／代號 | 可開啟網址 | 選擇原因 | 建議匯入方式 | 限制與匯入後檢查 |
| --- | --- | --- | --- | --- |
| 1／D1 | [UCI 官方資料說明](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction) | 確認研究背景、Target、Wh 單位、變數及原作者 | 先嘗試網頁 URL | 檢查 Variables Table 與 Additional Variable Information 是否完整；它沒有模型成績 |
| 2／P1 | [大學典藏書目](https://orbi.umons.ac.be/handle/20.500.12907/23357)；[公開全文 PDF](https://orbi.umons.ac.be/bitstream/20.500.12907/23357/1/1-s2.0-S0378778816308970-main.pdf)；[DOI](https://doi.org/10.1016/j.enbuild.2017.01.083) | 原始研究，連結資料來源、特徵與四種模型 | 優先由使用者開啟典藏 PDF，取得全文後匯入；保留 DOI 與出處 | 本次 PDF HTTP 可取得但全文自動擷取逾時；匯入後確認真的有全文而非只有摘要。核對表格欄位及頁碼，未讀到之處保留未知 |
| 3／C1 | [原作者固定版本 R 程式](https://github.com/LuisM78/Appliances-energy-prediction-data/blob/e3e4c27a4ae2b41b21f84b4c9ac2d822d3d0ace1/Data%20driven%20models.R) | 補足 75/25 抽樣、CV 與實際保存的評估紀錄 | URL 若無法完整擷取，可由使用者複製相關原始程式為帶來源的文字來源 | 標示版本與行號；分數註解不是本次重現；不需要執行 R 程式。P1/C1 是同一研究 |
| 4／K1 | [Kaggle LR／SVR Notebook](https://www.kaggle.com/code/lakmaligamage/appliances-energy-consumption-prediction-lr-svr) | 同源資料、完整程式與輸出；可辨認 log target、隨機切分及指標問題 | 先試公開 URL；若未抓到 iframe 的程式／輸出，由使用者以瀏覽器列印 PDF，或整理可閱讀文字後匯入 | 不要只匯入標題頁；確認 Cells 17、21、31–44。原始 ipynb 的 JSON 可能不易閱讀，且匯入格式是否支援須以帳號介面為準；不執行 Notebook |
| 5／P2 | [出版社／DOI](https://doi.org/10.3390/en16227536)；[公開論文 PDF 備援](https://pdfs.semanticscholar.org/d9da/0ea38146d389ce5bb6d75e4f08cd63cdb54a.pdf) | 研究時間特徵及六種模型，並有 testing Table 2 | 匯入可取得的全文 PDF，保留出版社 DOI；同一篇只算一個研究 | 出版社曾 429；備援為論文本體的公開副本。確認 §3.2.1 與 Table 2，不能自行補上未明示的 holdout 比例 |
| 6／D2 | [本專案 Kaggle Data Card](https://www.kaggle.com/datasets/loveall/appliances-energy-prediction/data) | 補充作業要求的 Kaggle 來源與原研究致謝 | 網頁 URL | 不需下載整個 CSV；動態頁可能擷取不全，且不能取代 UCI 變數定義 |
| 選讀／C2 | [含 Ridge／Lasso 的固定版本 Notebook](https://github.com/mhadeli/Appliances-Energy-Prediction/blob/5bdb684b7516fc93899f70559e2d98480516ae51/Appliances%20Energy%20predicition%20Notebook.ipynb) | 可研究正則化、PCA 及公開程式品質 | 僅在要討論品質問題時匯入帶出處的程式片段／可讀文件 | 提醒 `exp(r2_score(...))` 錯誤；不把它當可靠排名依據，不下載／執行模型物件 |

若要讓 NotebookLM 討論本專案，可由使用者另外匯入既有 [Phase 2 評估](../reports/phase2/model_evaluation.md)、[Phase 3B 報告](../reports/phase3/phase3b_report.md)、[Phase 3C 報告](../reports/phase3/phase3c_report.md)，以及本階段比較表。這些要標成「本專案內部成果／GPT 輔助整理」，不能冒充獨立外部研究。先查明公開來源，再用內部成果比較，避免把比較表本身當成外部成績的唯一證據。

## 共用提問要求

每次提問可附上：

> 請以繁體中文回答，限依已匯入來源，為主要主張附來源引用及可定位的頁碼、表號或程式 Cell。區分來源原文、你的推論與資訊不足；未提供的切分、分數、作者或方法請標 UNVERIFIED。不同 target 尺度、樣本、切分或 CV 的分數不可直接排名。不要捏造本專案已執行的模型、部署或實驗。

## 八個建議研究問題

1. **背景與預測任務（D1、D2、P1）**：Appliances Energy Prediction 如何收集？`Appliances` 與 `lights` 的意義、單位、取樣頻率為何？單一住宅、4.5 個月與天氣插值，對泛化及實際預測有何限制？請區分同時點耗能估計與未來預測，不自行假定特徵在預測時皆已可得。
2. **原始研究方法（P1、C1）**：原始研究實際比較哪四種模型，如何使用時間／環境特徵、選特徵、切分與 repeated CV？請將 training、CV 及 testing 成績分開，並指出論文與作者程式能互相核對的內容。
3. **Linear Regression 的限制（P1、P2、本專案報告）**：線性關係、共線性、偏態、高耗能尖峰與殘差結構會帶來哪些限制？哪些有來源證據，哪些只是待驗證假設？不要以單一 R² 推定限制的因果關係。
4. **非線性模型的研究理由（P1、P2、K1）**：RF、GBM、SVR、XGBoost、Extra Trees 如何表示非線性或交互作用？請用各來源自己的實驗條件說明結果，並討論計算成本、解釋性及過度擬合，不跨不同切分直接宣稱必然勝出。
5. **Feature Selection、正則化與 PCA（P1、C1、C2、本專案報告）**：刪除冗餘特徵、Lasso 稀疏化、Ridge 收縮與 PCA 有何差異？如何解釋本專案從 25 變 6 features，holdout RMSE 卻由 86.732939 升至 90.441773？請指出 C2 指標實作問題，不採其錯誤 R² 當成效能證明。
6. **切分、洩漏與指標尺度（C1、P2、K1、本專案報告）**：75/25 random、80/20 chronological、普通 K-fold、TimeSeriesSplit、nested CV 各在估計什麼？log target MAE 能否與 Wh MAE 比較？請檢查 K1 的 fold 外 scaling、打亂後時間切分及錯用模型的 CV R²，並標明 P2 未說清楚的 protocol。
7. **本專案與公開解法的差異（全部已驗證來源）**：請製作表格比較 target 尺度、資料保留規則、特徵、模型、切分、調參與證據完整性。區分已查看的固定 holdout、CV 及 testing；哪些可比較方法，哪些可以比較數字？本專案 95% Prediction Interval 的 coverage／width 能否由外部點預測成績推定？
8. **改善方向與報告結論（P1、P2、本專案報告）**：哪些時間／週期特徵、交互作用、非線性模型或時間驗證設計值得未來研究？哪些方向仍能維持 HW2 指定 Linear Regression？請提供有出處的研究假設及公平驗證條件，明確標示尚未實作，不安排本階段重新訓練。

## 使用者後續操作與留存

1. 由使用者自行建立／選擇 NotebookLM 筆記本並匯入上述來源；本階段不操作帳號。優先至少具備 D1、P1、K1；建議再加 C1、P2 補強方法查核。
2. 檢查每個來源是否有正文、程式與表格，名稱加上代號、作者、年份及版本；無法取得的來源保留 UNVERIFIED，不讓模型補造內容。
3. 依序提出八個問題，保存實際問題、原始回答、來源引用、操作日期及筆記本／分享連結（若方便提供）。對分數回查原論文或 Cell。
4. 最後要求 NotebookLM 依已核實來源產生**至少 100 字的繁體中文研究摘要**，涵蓋背景、模型、切分限制、本專案差異及改善方向，附來源引用。這項摘要尚未產生；本文件不能代替作業要求的 NotebookLM 成果。
5. 將實際摘要與引用交回人工驗收。後續獲授權製作最終報告時，另標「NotebookLM 摘要」，並將本階段文件及程式協助標為「GPT 輔助內容」。不要把兩者混稱，也不要加入與 HW2 無關的私人資訊或登入資訊。

## 本階段完成界線

已準備來源、比較與問題；尚待使用者完成 NotebookLM 匯入、實際問答及至少 100 字摘要。未製作網站、最終 PDF、主程式整合、ZIP；未操作 NotebookLM 或付費服務；未修改 Dataset、模型及既有實驗成果。
