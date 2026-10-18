# 家庭家電耗能預測與特徵選擇：成果報告草稿

整理日期：2026-10-18。這是可修改的報告草稿，尚未輸出正式 PDF；未填個人資訊與未取得的材料保留待補。研究數字引用既有保存成果，本次未重新訓練能源模型。

| 封面欄位 | 內容 |
| --- | --- |
| 課程／授課教師 | 待依正式作業單補入 |
| 姓名／完整學號／班級 | 待使用者提供 |
| 題目 | Appliances Energy Prediction Using Multiple Linear Regression and Feature Selection |
| 展示網址 | 待完成 Vercel 前端與雲端後端部署後填入 |

## 摘要

本專案以家庭環境感測與氣象資料估計家電耗能，最終預測器固定為多元線性迴歸，並比較六種特徵選擇方法。研究使用時間順序切分，將前 15,788 筆用於訓練、後 3,947 筆作固定 Holdout，並區分 Training CV、Nested CV 與樣本內擬合。正式 Training CV 選出的 MI 模型保留 6 個特徵，相較 25 特徵 Baseline 大幅精簡輸入，但 Holdout RMSE 從 86.732939 Wh 增加至 90.441773 Wh，R² 從 0.092352 降至 0.013067，未帶來預測改善。網站將研究成果呈現為互動圖表，另提供自訂數值 CSV 的驗證、方法比較、訓練與新資料推論。結果顯示特徵精簡與預測品質需分別評估，不能只以較少欄位或較佳訓練內 CV 宣稱泛化提升。

## 1. 研究問題與資料理解

研究目標是以感測資訊估計同時間的 `Appliances` 能耗，並了解特徵選擇如何影響輸入規模及誤差。這個設計使用同時點資料，未建立只用過去資訊的多步未來預報器。

資料為 Appliances Energy Prediction，共 19,735 筆、29 欄，包含 28 個原始輸入及一個目標。約每 10 分鐘觀測一次；目標單位為 Wh，原始時間未標時區。排除 `date`、`rv1`、`rv2` 後，Baseline 使用 25 個數值特徵。高耗能真實觀測保留，不依目標大小刪除測試列。

資料來源：[Kaggle](https://www.kaggle.com/datasets/loveall/appliances-energy-prediction/data)、[UCI / DOI](https://doi.org/10.24432/C5VC8G)。原始檔案 SHA-256 為 `2820bf712ad0275cb18b85a05250926100d8e65ebb9f4d2d016ca91ea152a25d`；本專案的單位、欄位及來源留存見 [專案 README](../README.md)、[來源查核](../references/external_sources.md)。

## 2. CRISP-DM 與實驗設計

| 階段 | 本專案內容 | 證據 |
| --- | --- | --- |
| Business Understanding | 耗能估計、特徵精簡及適用限制 | [研究背景](../reports/phase2/business_understanding.md) |
| Data Understanding | 資料驗證、EDA、偏態、相關性與時間相依 | [EDA](../reports/phase2/eda_summary.md) |
| Data Preparation | 排除欄位、fold-local 補值與縮放 | [前處理契約](preprocessing_wizard.md) |
| Modeling | 六種選欄方法，最終固定 LinearRegression | [正式選欄實驗](../reports/phase3/phase3b_report.md) |
| Evaluation | CV、Nested CV、Holdout、殘差與單筆 PI 分開 | [Phase 3C](../reports/phase3/phase3c_report.md) |
| Deployment | 五頁互動網站、本機 CSV 訓練與推論；雲端部署待完成 | [操作指南](user_guide.md)、[部署計畫](vercel_deployment_plan.md) |

正式能源研究前 80% Training、後 20% Holdout，不打亂。Phase 3B 使用 5 個 outer time folds、3 個 inner time folds，gap=6；當前 fold 的補值、縮放及 Selector 只能由該 fold 的 Training 學習。完整 Training 上另做 3-fold final-selection CV 選最終方法。基準 Baseline 使用原始數值；MI 模型使用保存的中位數補值及 StandardScaler，因此兩模型是既有完整流程的比較，不能把差異全歸因於欄位數。

六方法為 F-test、Mutual Information、Forward SFS、RFE、LassoCV、RFECV。前四種有預先指定的 k；後兩種在訓練區自動決定特徵數，LassoCV 用於選欄，最終仍另 fit LinearRegression。各 fold 的 k 可以不同，跨 fold 平均 k 不是任一實際模型的欄位數。

## 3. 正式選欄與固定模型結果

依事先設定的 Training CV 平均 RMSE 規則，最終選 MI、k=6，特徵為 `T3`、`T4`、`T5`、`RH_6`、`T8`、`T9`。其他方法與各 fold 結果保留於正式報告與網站，未補造六方法的 Holdout 排名。

下表只比較相同 chronological Holdout 的兩個既有固定模型。數值由 [prediction_metrics.json](../reports/phase3/phase3c/prediction_metrics.json) 摘錄，顯示至小數六位。

<!-- verified-holdout-table -->
| 模型 | 特徵數 | RMSE（Wh） | MAE（Wh） | R² |
| --- | ---: | ---: | ---: | ---: |
| Baseline LinearRegression | 25 | 86.732939 | 52.014105 | 0.092352 |
| MI + LinearRegression | 6 | 90.441773 | 53.605861 | 0.013067 |
<!-- /verified-holdout-table -->

MI 減少 19 欄（76%），但 Holdout RMSE 增加約 3.71 Wh，MAE 也增加。這個結果支持「能精簡輸入」，未支持「本次精簡使測試更準」。兩模型 R² 偏低，且時間圖可見對短暫高耗能尖峰的低估；不宜宣稱已適合精準尖峰預警。

可能原因包含線性結構限制、未納入的設備使用行為、特徵資訊不足與時段分布差異。這些是合理假設，本專案沒有做足以區分其因果影響的獨立消融實驗，不能把單一原因當作已證實結論。

## 4. 指標與 Prediction Interval 解讀

RMSE 與 MAE 越低越好，單位與目標相同；RMSE 對大誤差較敏感。R² 越高越好，並非分類準確率，負值表示相對該評估資料平均值的平方誤差更大。Training 平均值基準則只用訓練資料計算，故其 Holdout R² 不必為零。

95% OLS Prediction Interval 使用保存的 Training 殘差、設計矩陣及 leverage 計算；未依 Holdout 校準，負下界未裁切。依 [PI 指標](../reports/phase3/phase3c/prediction_interval_metrics.json)：

<!-- verified-pi-table -->
| 模型 | 實測 Coverage | 平均區間寬度（Wh） |
| --- | ---: | ---: |
| Baseline | 96.0983% | 376.104533 |
| MI | 95.8196% | 401.442761 |
<!-- /verified-pi-table -->

高涵蓋率仍可能伴隨寬區間及不佳的點預測，不能解讀為約 96% 準確率。單筆 PI 不是平均值 CI，也不是同時涵蓋整條時間曲線的保證；時間相依、異方差與選模不確定性限制仍存在。

## 5. 網站與自訂資料功能

網站採 Next.js App Router、TypeScript、Tailwind CSS 與 Recharts，支援深淺色、桌面與手機。Overview、Feature Selection、Model Evaluation、Prediction Explorer 讀取受驗證的研究快照，保留來源 SHA-256，沒有為了展示而重新訓練研究模型。

Custom Dataset 流程：上傳 UTF-8 CSV → 選數值目標與候選 → 前處理及切分設定 → 確認 → Python 驗證 → Training CV 比較六方法 → 套用設定 → 另行確認訓練 → 看 CV／Holdout → 單筆／批次推論及下載。

自訂資料的 SFS、RFECV、LassoCV 使用內部 3-fold 選欄。RFECV／LassoCV 顯示逐 fold 實際整數 k，不虛構固定 k 曲線；Lasso 零特徵不補造分數。自訂 CSV 的 alpha grid 與資源上限依 [M14 契約](custom_dataset_training.md) 執行，不能假設與既有能源研究的所有參數相同。訓練後推論沿用凍結的轉換與選欄，只需模型選中特徵，部分空白依訓練設定補值。

本機網站功能已由使用者確認可接受；M14 保存的自動檢查為網站 127/127、Python 211/211，lint、型別與 production build 通過，詳見 [驗收紀錄](../web/verification/m14-result.json)。這是該次實測紀錄，本次文件整理沒有重跑上述測試，也不等同 Vercel 雲端驗收。M14 瀏覽器模型流程採合成資料測試；未將合成資料分數當作真實房價表現。

## 6. 部署規劃與目前限制

使用者已選擇 Vercel 前端搭配獨立 Python 運算後端，保留訪客上傳、選欄、訓練與推論。現有程式仍使用本機 Python 路徑、localhost 來源限制及同程序暫存模型，雲端接線與發布尚未完成。架構及驗收門檻詳見 [部署計畫](vercel_deployment_plan.md)，正式網址與雲端實測證據待補，不提前寫成已上線。

研究限制：Holdout 在 Phase 2 已查看，這裡屬固定後期資料的描述性評估；不是全新、從未查看的最終測試。最佳方法從多方法選出，Nested CV 也沒有再涵蓋整個擇優程序的額外最外層。原時間未標時區、同時間特徵非未來已知資訊，結果不代表多步未來預報能力。

功能限制：Custom Dataset 目前限數值輸入及線性迴歸；無類別展開、一般資料 PI 或正式永久模型管理。上傳容量與方法計算量各有限制；大量資料及多人使用需依部署架構另行驗證。

## 7. 結論

本專案完成從資料理解、前處理、選欄、評估到互動展示與自訂資料實驗的本機流程。MI 保留較少特徵，但在既有 Holdout 沒有勝過 Baseline；因此研究結論應呈現精簡與誤差的取捨。後續改善需以新的獨立驗證支持，避免反覆使用既有 Holdout 調整模型。

## 8. 待補附件與引用

- 封面個人資訊、老師原始作業單與最新更新：待提供；目前只依 Repository 留存內容作暫定核對。
- NotebookLM 實際摘要與引用：待正式保存。使用者先前表示已取得摘要；本草稿不代寫或冒充該產物。
- GPT 協助範圍：資料整理、程式與網站開發、測試及文件草稿；請本人逐項確認，附作業要求的對話 PDF，不用此句代替全部對話紀錄。
- 展示截圖：可選用 [M14 桌面淺色](../web/artifacts/m14-auto-desktop-light.png)、[桌面深色](../web/artifacts/m14-lasso-desktop-dark.png)、[手機](../web/artifacts/m14-auto-mobile-light.png)，屬功能合成資料示範；能源正式結果請配對既有研究圖。`artifacts/` 被 Git 忽略，正式交付時需另外確認附件有包含。
- 外部研究比較引用 [solution_comparison.md](../references/solution_comparison.md) 與 [external_sources.md](../references/external_sources.md)，保留不同切分、尺度及未核實範圍，不直接把外部指標和本研究混排。
- 正式 PDF、對話 PDF、學號命名 ZIP、Vercel 網址及雲端驗收紀錄：待完成後補入。
