# Phase 3L — Release Closure

本文件整理已完成的 Phase 3K 正式發布，並記錄尚未驗收的範圍。Phase 3L 只改文件與發布紀錄，不新增功能、不重訓、不修改歷史測試或實驗。

## 固定正式版本

- 正式網址：[AIOT-HW2](https://aiot-hw-2.vercel.app/)。正式網域可公開存取，不需 Preview 的 Vercel 登入。
- 發布日期：2026-10-22。
- Production Commit：`0c935027aa236dea99ea23793ad939cd85dd2cda`。
- Source Branch：`main`；Deployment：`5c4zhkMYgrPZfGksim2Wg5Bypazm`，READY。
- [Vercel 部署紀錄](https://vercel.com/bambambooo/aiot-hw-2/5c4zhkMYgrPZfGksim2Wg5Bypazm)。使用者明確授權一般 Push main 及其自動 Production Deployment；沒有額外 Commit、Force Push 或手動建立部署。
- Phase 3L 文件尚未 Commit／Push，因此不是上述 Production Commit 的一部分。

## 已驗收的正式站範圍

| 項目 | Phase 3K 實際證據 |
| --- | --- |
| HTTPS、研究頁面與資料集內容 | 26／26 HTTP 檢查，含能源原預測 API、三資料集內容與 Custom 隔離 |
| 六個固定模型 | 正式網域 JSON SHA-256 與 manifest 一致；8,236 筆 Holdout＋274 個額外案例共 8,510／8,510 通過 |
| 數值政策 | absolute／relative tolerance 均為 1e-10；最大 absolute error 7.275957614183426e-12，未放寬誤差 |
| Research Live Predictor | 三資料集各 Baseline／MI，六個表單實際輸入與推論；欄位順序、模型切換、外插、非法小數拒絕、Reset、Dataset 隔離、跨頁／重新整理／上一頁與 Holdout 搜尋排序通過 |
| Custom Browser Mode | 80×3 含缺失值合成資料，64／16 Training／Holdout；RFE 訓練、16 筆 Holdout 對照、單筆／兩筆批次推論與模型刪除通過；Browser 運算沒有向 Server API 傳送資料 |
| Custom 操作介面 | CSV 上傳、設定、Python 驗證、RFE k=2 訓練、Live Predictor、合法外插輸入 999 保留，模式切換清除模型通過 |
| Worker／WASM／CSP | Pyodide 314.0.7 成功初始化及載入科學套件；同源 Worker 模組 200，CSP 保留 worker-src 'self'，WASM MIME 為 application/wasm；結束時 0 個殘留 Worker |
| Server Mode | 維持預設；無後端時明確顯示不可用，API 503，沒有自動上傳或回退 |
| 響應式與主題 | Chromium 154：1440×1000 淺色及 390×844 深色，三資料集四頁共 24／24；無水平溢出，檢查中無 console／hydration 錯誤或警告 |

上述正式站驗收不等於所有工作負載都已在 Production 重跑。完整六方法、多 k 比較與其他故障測試的本機／實驗／Preview 證據各自保留，不升格為 Production 實測。

## 使用限制

- Browser 訓練上限 2,000 列、15 個數值候選特徵、CSV 3 MiB；各方法原有 Python 工作量保護不變。
- 多 k 比較硬上限 100,000,000 work units、180 秒；超過 10,000,000 需明確確認。比較超限只阻止比較，不阻止符合條件的訓練。不抽樣、不減少 CV 或擅自改 k。
- 需要初次下載固定的公開 Pyodide／科學套件；網路、資料、裝置與選欄設定都會影響等待時間，不承諾任意 CSV 低於 60 秒。模型留在 Worker 暫存，不是持久化服務。
- Research Live Predictor 是固定線性模型的點估計；無新輸入真值時不顯示單筆 R²／MAE／RMSE／Residual，也不套用 Holdout PI。
- Energy 是同時點耗能估計，單位 Wh；Automobile 價格幣別未核實；Student 必須已知 G1／G2，absences 的累計截止時點未明，不是開學前預測。合法外插與線性模型超出目標範圍的預測不裁切。

## 尚未完成與已知事項

| 項目 | 優先 | 現況與收尾方式 |
| --- | --- | --- |
| Production macOS Safari | 高 | NOT TESTED。使用者回報的 Preview 真實 Safari 7／7 PASS 只屬 Preview；以正式網址另補人工驗收，不改寫舊紀錄 |
| HTTP2_PING_FAILED | 中 | 能源 Feature Selection 曾在 15 秒導覽＋30 秒額外等待後仍未完整載入；獨立 HTTP 200 約 1.447 秒。一次受控重載後 document complete、圖表正常，後續 24 組通過。重載觀測到 net::ERR_HTTP2_PING_FAILED，首次根因尚未確定，不能直接判定是程式問題 |
| 事件時間／網路證據 | 中 | 發生日期為 2026-10-22；精確牆鐘時間與完整 HAR 未保存。已保存路由、等待上限、HTTP 回應及重載結果；若重現再記錄當時時間、URL、瀏覽器／網路環境、Network 錯誤與重載結果。沒有建立背景監控或自動化 |
| Python 歷史契約 | 中 | 1 ERROR 保留。固定 main／4619bd4 的歷史 guard 不適用新 HEAD；125 檔歷史 manifest 中 README／.gitignore 為已核准非研究差異，其餘 123 個研究檔雜湊通過。未刪除、skip、mock、降低保護或修改舊 manifest；未來如需拆分歷史重現與現行完整性，另立任務 |
| 實體 iPhone Safari | 低～中 | NOT TESTED。有設備時補測；Chromium 手機尺寸不代表 iPhone PASS |
| 獨立 Server Mode 後端 | 可選 | 正式站尚未配置；Browser Mode 可獨立使用。沒有宣稱雲端 Python Gateway／多人負載通過 |

## 證據與後續發布

- [Phase 3K 精簡版本化紀錄](../web/verification/multi-dataset-phase3k.json)：由當時保存的 Production 驗收紀錄整理，附來源雜湊；Phase 3L 未重跑或冒稱新驗收。
- [Phase 3G 歷史驗收](../web/verification/multi-dataset-phase3g.json)：保留當時的 FAIL／NOT TESTED 與 Python 226 PASS／1 ERROR，不以後來結果覆寫。
- [Phase 3H 的 115 檔歷史候選清單](../web/verification/multi-dataset-phase3g-staging.json)：只用於核對那三次提交，不是本次文件的 staging 授權。
- 後續若需 Commit／Push 本次文件，必須重新列出實際文件差異並取得授權。Push main 會依現有設定自動觸發 Production Deployment，不能把文件 Push 視為不影響部署。
- 本階段不新增 Tag、不建立分支、不改 Vercel 設定、不執行部署；作業 PDF／學號 ZIP 等交付項目仍依原交付清單另行處理。
