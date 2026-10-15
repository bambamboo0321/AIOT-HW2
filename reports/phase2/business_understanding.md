# Phase 2：Business Understanding 與 CRISP-DM

## 研究背景與問題定義

家電能源消耗可能與住宅內的溫濕度、照明及外部天氣共同變化。本作業使用 Appliances Energy Prediction 公開資料，了解固定的多元線性迴歸能否建立可重現的基準預測，以及在後期時間資料上會產生多大的估計誤差。這是對單棟低能源住宅的觀測資料分析，沒有介入實驗，也不虛構實際節能量或成本效益。

預測目標為 `Appliances`，資料來源標示單位 **Wh**，屬能源讀值，並非功率 W。本作業以同時刻可觀測的輸入估計當下家電能源消耗；前期訓練、後期測試用於觀察時間泛化能力，不代表僅依歷史資料即可預測下一時刻或長期耗能。

## 輸入特徵與 Baseline

原始輸入共 28 個（包含 `date`，排除 Target），符合老師 `13 < Features < 40` 的要求。欄位涵蓋照明能源、不同房間與建築外側的溫度／相對濕度、氣象站溫度、氣壓、外部相對濕度、風速、能見度、露點，以及時間與兩個隨機變數。

Baseline 預先排除 `Appliances`、`date`、`rv1`、`rv2`，固定使用 25 個數值欄位。`date` 保留作切分與分析，未衍生模型特徵；`rv1`、`rv2` 為隨機變數且完全相同，排除它們屬預先定義的資料品質處理，不是依 Target 相關係數或模型分數進行正式 Feature Selection。原始 CSV 的所有欄位與資料列保留。

多元線性迴歸將預測表達為截距加上各輸入乘以係數的加總。使用 sklearn `LinearRegression()` 預設設定，以最小平方方式建立容易重現的基準模型；係數單位取決於原始欄位尺度，相關輸入可能使係數不穩定，不能直接把係數當成因果效果或重要性排名。

## 評估指標

| 指標 | 定義與用途 | 單位／解讀 |
| --- | --- | --- |
| MAE | 絕對誤差的平均，反映一般誤差大小 | Wh，越低越好 |
| MSE | 平方誤差的平均，較重視大誤差 | Wh²，越低越好 |
| RMSE | MSE 的平方根，便於與 Target 比較 | Wh，越低越好 |
| R² | `1 - SSE / SST`，相對該評估資料平均值的解釋程度 | 無單位，可能為負；不是分類準確率或節能比例 |

每種切分都分別計算 Training／Testing 的四個指標。時間切分是主要結果；隨機切分只呈現比較，不依其較高分數挑選最終模型，也不使用 Testing 調整參數。

## 時間資料限制與預防資料洩漏

資料每 10 分鐘觀測一次，鄰近時間的天氣、感測器與耗能可能具有自相關。隨機切分可能讓相鄰或同一時段的觀測分散到 Training 與 Testing，因此評分不能等同未來外推能力。主要設計使用原始排序前 80% Training、後 20% Testing，不 Shuffle。

模型只在相應的 Training fit；不把 Target 當作輸入，不做全資料 fit scaler，不在 Testing 訓練或調整模型，不使用未來 Target 建立特徵。全資料 EDA 只描述觀測現象；模型與 25 欄 Baseline 定義在查看評分前固定，不依 EDA 圖表修改特徵。

只有單棟住宅約 4.5 個月，無跨年或其他住宅驗證；日期未標示時區，保留原始時間。感測器與天氣是否能即時取得仍是實際應用限制。分析不推定因果，也不估算尚未驗證的實際節能效果。

## CRISP-DM 六階段與本作業

| CRISP-DM 階段 | 本專案工作 | 範圍 |
| --- | --- | --- |
| Business Understanding | 定義家電能源估計問題、固定 Baseline 與評估目的 | Phase 2 本文件 |
| Data Understanding | Phase 1 品質驗證、Phase 2 分布／相關性／時間 EDA | 本階段完成 |
| Data Preparation | 固定欄位排除、前後時間與隨機切分 | Phase 2；無正式 Feature Selection |
| Modeling | 兩個預設 sklearn LinearRegression Baseline | Phase 2 |
| Evaluation | 四項 Train／Test 指標與殘差圖、切分比較 | Phase 2 |
| Deployment | 彙整後續作業交付與可重現流程 | 尚未執行正式部署 |

CRISP-DM 六階段是方法論，不等同作業 Phase 1–4。正式 Feature Selection、區間預測圖與後續其他交付工作均不在此次實作範圍；Phase 3–4 尚未開始。
