# Phase 2：Baseline Model Evaluation

## 設計與資料準備

兩個 Baseline 皆使用原樣的 `LinearRegression()`、相同的 25 個預先定義數值特徵：lights, T1, RH_1, T2, RH_2, T3, RH_3, T4, RH_4, T5, RH_5, T6, RH_6, T7, RH_7, T8, RH_8, T9, RH_9, T_out, Press_mm_hg, RH_out, Windspeed, Visibility, Tdewpoint。

排除 Target `Appliances`、時間欄位 `date`、隨機且完全相同的 `rv1`／`rv2`。這是預先定義的 Baseline 資料品質處理，不是依 EDA 或模型分數進行 Feature Selection。`date` 保留供切分、樣本追溯與 EDA 使用，沒有衍生模型時間特徵。

**時間切分為主要結果**：原始順序前 80% Training、後 20% Testing，不 Shuffle。隨機切分使用 `train_test_split(test_size=0.2, random_state=42)`，僅供比較。沒有 scaler、缺失值補值器、參數調整、其他模型或依測試分數挑選最終模型。模型只在對應 Training fit；Training／Testing 指標分開計算。

| 切分 | Partition | 筆數 | 最早日期 | 最晚日期 |
| --- | --- | ---: | --- | --- |
| 時間切分（主要） | Training | 15788 | 2016-01-11T17:00:00 | 2016-04-30T08:10:00 |
| 時間切分（主要） | Testing | 3947 | 2016-04-30T08:20:00 | 2016-05-27T18:00:00 |
| 隨機切分（比較） | Training | 15788 | 2016-01-11T17:10:00 | 2016-05-27T18:00:00 |
| 隨機切分（比較） | Testing | 3947 | 2016-01-11T17:00:00 | 2016-05-27T17:50:00 |

完整資料列索引保存在 `split_indices.json`。JSON 記錄 date range、random_state、feature names、模型預設參數與環境版本；不儲存私人本機路徑。Random 的日期範圍重疊是方法本身的結果，不能解讀為未來外推能力。

## 指標與實測結果

MAE 為平均絕對誤差（Wh）；MSE 為平均平方誤差（Wh²）；RMSE 為 MSE 平方根（Wh），對較大的錯誤較敏感；R² 無單位，以上述 partition 的 Target 平均值作為分母基準，可能為負值。R² 不等於分類準確率或節能比例。

| 切分 | Partition | MAE（Wh） | MSE（Wh²） | RMSE（Wh） | R² |
| --- | --- | ---: | ---: | ---: | ---: |
| 時間切分（主要） | Training | 54.289188 | 9152.687409 | 95.669679 | 0.172899 |
| 時間切分（主要） | Testing | 52.014105 | 7522.602624 | 86.732939 | 0.092352 |
| 隨機切分（比較） | Training | 53.217383 | 8897.953796 | 94.328966 | 0.163445 |
| 隨機切分（比較） | Testing | 52.547078 | 8312.271344 | 91.171659 | 0.169362 |

- 時間切分（主要）：Testing − Training RMSE = -8.936740 Wh，R² 差 = -0.080547；測試集負預測 45 筆，殘差平均 -6.354935 Wh。
- 隨機切分（比較）：Testing − Training RMSE = -3.157307 Wh，R² 差 = 0.005917；測試集負預測 2 筆，殘差平均 -2.571514 Wh。

Training 與 Testing 是不同的觀測樣本；差距可觀察擬合與時間分布變化，不能單靠差距證明過擬合。Random 分數受到同一住宅、相鄰時間樣本可能同時落入兩側的影響，不能當成真正的未來預測能力。沒有根據兩個測試集分數反覆調整設定。

時間切分 Training 的 Target 平均／樣本標準差為 98.024449／105.198291 Wh，Testing 為 96.376995／91.050065 Wh。兩個期間的 Target 分布與變異不同，R² 的分母也不同；較低 RMSE 不必然代表較高 R²，不能用不同測試集的分數直接宣稱哪一種切分具有較佳未來預測能力。

## 係數與模型保存

時間切分 Intercept = -67.294852 Wh；隨機切分 Intercept = 23.193744 Wh。係數對應未標準化的原始欄位尺度，不能直接比較絕對值來做特徵重要性排名，也不能宣稱因果效果。

| Feature | 時間切分係數 | 隨機切分係數 |
| --- | ---: | ---: |
| lights | 2.139424 | 1.930226 |
| T1 | -6.374035 | 0.073979 |
| RH_1 | 12.854339 | 15.905544 |
| T2 | -13.205355 | -17.595725 |
| RH_2 | -12.410981 | -13.618769 |
| T3 | 27.709276 | 25.870400 |
| RH_3 | 8.634363 | 4.484464 |
| T4 | -3.795283 | -3.878785 |
| RH_4 | -2.445068 | -0.407502 |
| T5 | -1.675221 | -0.867705 |
| RH_5 | 0.105714 | 0.212372 |
| T6 | 7.636710 | 7.386886 |
| RH_6 | 0.341985 | 0.341426 |
| T7 | 1.075958 | 2.241512 |
| RH_7 | -0.489215 | -1.662980 |
| T8 | 9.038519 | 7.449571 |
| RH_8 | -5.960234 | -4.560437 |
| T9 | -14.350566 | -15.165642 |
| RH_9 | -0.998667 | -1.030043 |
| T_out | -8.626430 | -10.501871 |
| Press_mm_hg | 0.223381 | 0.162218 |
| RH_out | -0.520918 | -1.072619 |
| Windspeed | 1.273456 | 1.777901 |
| Visibility | 0.189684 | 0.161676 |
| Tdewpoint | 3.699699 | 5.007594 |

`models/chronological_baseline.joblib`、`models/random_baseline.joblib` 儲存模型及 feature names、intercept、coefficients、sklearn version、training metadata；對應的 `.json` 提供可讀資料。重新載入模型後，可用相同欄位順序預測；joblib 檔案僅供本專案可信來源使用。

測試集逐筆 actual、predicted、residual 與原始 row index／date 保存於 `predictions_chronological.csv`、`predictions_random.csv`。三類評估圖均只使用該模型的 Testing 預測，沒有混入 Training 資料。比較圖明確分開 Training／Testing。

## 發現與限制

Baseline 可能低估極端耗能尖峰；請同時觀察 Actual vs Predicted、殘差分布與殘差隨預測值的型態。所有預測保留 LinearRegression 原始輸出，包括負值，未截斷或依測試集修正。OLS 的線性、加成形式可能無法描述作息、尖峰與非線性關係，輸入間高度相關也可能使係數不穩定。

此流程以**同一時間的感測器與天氣資訊估計當下 Appliances**，不是只靠歷史資料預測未來時點的多步模型；時間 Holdout 僅模擬未見過的後期資料。實際使用仍須確認這些輸入在預測時是否可取得。資料涵蓋單棟住宅約 4.5 個月，測試期間只有後 20%，沒有跨年度、跨住宅或外部驗證；未執行交叉驗證。未建立 Confidence Interval／Prediction Interval，未開始 Phase 3–4。

## 圖表

- [08_chronological_actual_vs_predicted.png](figures/08_chronological_actual_vs_predicted.png)
- [09_chronological_residual_vs_predicted.png](figures/09_chronological_residual_vs_predicted.png)
- [10_chronological_residual_distribution.png](figures/10_chronological_residual_distribution.png)
- [11_random_actual_vs_predicted.png](figures/11_random_actual_vs_predicted.png)
- [12_random_residual_vs_predicted.png](figures/12_random_residual_vs_predicted.png)
- [13_random_residual_distribution.png](figures/13_random_residual_distribution.png)
- [14_split_performance_comparison.png](figures/14_split_performance_comparison.png)
