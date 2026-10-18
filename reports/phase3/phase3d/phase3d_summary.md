# Phase 3D：六特徵組合探索與驗證

本階段只用 Training；沿用 Phase 3B final_selection_cv：3 folds、gap=6、seed=42、median／StandardScaler、LinearRegression。RMSE 是各 fold RMSE 算術平均，單位 Wh。

## B：包含選特徵流程的 CV（各 fold 內重選）

| 方法 k=6 | Mean CV RMSE Wh | Mean CV MAE Wh | Mean CV R² | 來源 |
| --- | ---: | ---: | ---: | --- |
| mutual_information | 98.461796 | 59.871672 | 0.043218 | existing_phase3b_final_selection_cv_rechecked |
| f_test | 99.164490 | 60.703080 | 0.028912 | existing_phase3b_final_selection_cv_rechecked |
| forward_sfs | 101.981611 | 60.492407 | -0.029032 | new_phase3d_fold_local_sfs_k6 |
| rfe | 101.275256 | 61.091159 | -0.014403 | existing_phase3b_final_selection_cv_rechecked |

MI／F-test／RFE 讀取原有同範圍 k=6 結果，另用該 fold 保存支援集合重算 LR，確認分數一致。SFS k=6 不在舊 grid，新增三個 training fold 內選擇及內層時間 CV；沒有先用完整 Training 選 SFS 再宣稱無偏。四方法的 validation 僅用於各 fold 最終評分；事後在方法間擇優仍有偏差。

## A：固定組合的回顧性 CV（與 B 分開）

| 固定六欄參考 | Mean CV RMSE Wh |
| --- | ---: |
| mutual_information | 100.078712 |
| f_test | 100.856652 |
| rfe | 99.513576 |

上述固定方法組合來自完整 Training 的既有支援集合，其產生曾看到回顧性 CV validation 的 Target；分數有資料重用／選取偏差，不是無偏的 selector 評估。Random 固定組合使用同一 CV，但挑最低者亦有搜尋偏差。原 SFS 沒有固定 k=6 全 Training 組合，因此 A 不捏造 SFS 結果，SFS 比較使用 B。

## 1,000 組 Random Combination Search

- 不重複六欄組合 1000 組；共 177,100 種可能，抽樣比例 0.5647%。
- 僅隨機搜尋耗時 2.754 秒；含準備與方法比較、未含繪圖 10.717 秒。
- RMSE 分布：min 95.277933、median 100.601151、mean 100.675023、max 107.385215 Wh；SD(ddof=0) 1.843895 Wh。完整分位數見 JSON。
- 固定 MI RMSE 100.078712 Wh；優於 MI 371 組，與 MI 差距 ≤1e-9 Wh 的 0 組。MI competition rank=372／1001。
- MI 是否原就在 1,000 組內：False；若不在，額外參考評估不計入 1,000 組。
- 最佳抽樣組合：T3, RH_3, T6, T8, RH_8, T9；RMSE 95.277933 Wh。
- 與 MI 共同 3 欄：T3, T8, T9。

## 相關性、可信度與結論

Training-only Pearson 高相關門檻沿用 Phase 2 |r|≥0.8；MI 六欄內符合門檻的配對如下：

- T3／T5：r=0.814393。
- T3／T9：r=0.813077。
- T5／T9：r=0.827232。
- T8／T9：r=0.817843。

MI 單變量關聯高不代表特徵彼此不冗餘，也不等於對線性模型最佳。本結果能指出探索性差異，不能證明 MI 普遍不適合此資料，更不能證明 Random 最佳組合具有更好的真正泛化。

原正式流程的唯讀查核及既有測試未發現直接 Holdout／fold 洩漏或指標計算錯誤；A 類固定方法組合存在明確的回顧性資料重用，已與 fold-local B 類分開。詳見 verification_report.md。未讀入 Holdout Target 作搜尋、未重新產生 Holdout 評估，未保存／取代任何模型。

目前不建議 Exhaustive Search：共有 177,100 組，以本次快取 CV 搜尋時間線性外推約 487.8 秒（僅成本估計，非保證），但增加搜尋會擴大選取偏差，不能解決時間相依及泛化證據不足。先釐清評估目標／預先指定探索範圍較有價值；若將來另獲授權，需用未參與搜尋的外層時間驗證評估整個搜尋程序。此次沒有執行 Exhaustive Search。

## 輸出與重現

`python -B -m src.phase3d_search --run` 僅重跑 Phase 3D，會更新本階段目錄；不重跑 Phase 3B。首次執行前必須已有通過的 audit_details.json、verification_report.md 與起始 integrity_baseline.json。執行上限固定 1,000 組。耗時欄位具有執行環境差異；固定 seed、支援集合與分數可重現。

全部樣本／Top 10／Worst 10 CSV 採 UTF-8、LF。CSV 保存 fold 指標與 feature indices；comparison_summary.json 保存分布、排名、參考組合及來源 digest；plot_data.json 保存三張圖的實際來源數據。完整性與測試結果另見 integrity_verification.json、test_verification.json。
