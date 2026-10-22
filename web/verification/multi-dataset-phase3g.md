# Phase 3G 整站回歸與發布前驗收

Branch：`experiment/m17-pyodide`；HEAD：`4555ec599fffa0f6a3c8f16c68c1a5131c72a7c4`。

## 判定

**整站功能 PASS；嚴格完整驗收 FAIL／尚未全部完成。Preview Release Ready：有條件，未簽發無條件 Ready。**

可依精確清單準備 Commit；本階段未 staging、Commit、Push、Merge 或部署。Safari／實體手機未測，不能以 Chromium 模擬代替。

## 15 組研究功能

| 資料集 | Overview | Feature Selection | Model Evaluation | Prediction Explorer | Research Live Predictor |
|---|---|---|---|---|---|
| Energy | PASS | PASS | PASS | PASS | PASS（25／6 欄）|
| Automobile | PASS | PASS | PASS | PASS | PASS（14／7 欄）|
| Student | PASS | PASS | PASS | PASS | PASS（15／7 欄）|

Energy 保留 chronological Holdout、十分鐘時間軸、Wh、MI 六欄與原 PI。Automobile 僅用原始價格單位，幣別未核實；固定 k=7 與 auto 單點，沒有虛構 k 曲線。Student 明確要求已知 G1／G2，absences 時點未明；不裁切預測。Live 只顯示點估計，不套用 Holdout PI 或單筆誤差分數。

## 六模型數值

本次重新執行 TS 對固定且 SHA 驗證的原生參考／原保存預測；沒有重訓研究模型，也沒有重建或改寫歷史參考。容許值：absolute=1e-10、relative=1e-10。8,236 筆與 274 個額外案例零失敗。

| 模型 | Holdout 筆數 | 最大 absolute error（TS vs sklearn reference）|
|---|---:|---:|
| appliances-energy--baseline | 3947 | 4.54747350886e-13 |
| appliances-energy--mutual_information | 3947 | 5.68434188608e-14 |
| automobile-price--baseline | 41 | 7.27595761418e-12 |
| automobile-price--mutual_information | 41 | 3.63797880709e-12 |
| student-performance--baseline | 130 | 3.5527136788e-15 |
| student-performance--mutual_information | 130 | 3.5527136788e-15 |

原生參考對原保存結果差異均為 0。模型／前處理／metadata／manifest 完整性、Training-only 設定與原研究指標均通過。16 份 production trace 沒有 experiments/ 依賴。

## 品質檢查

- Typecheck、Lint、Production Build：PASS。
- Web 完整測試：225／225 PASS。
- Gateway 完整測試：9／9 PASS。
- Python 完整測試：227 項，226 PASS、1 ERROR，無跳過。
- HTTP：26 組路由檢查＋原 Energy API；另 7 項資源／404／CSP 檢查，PASS。
- 12 個研究路由的啟動 console／error／unhandled rejection 監測：0 error、0 warning、無 hydration warning。
- npm audit：0 已知漏洞。

### 必須保留的 Python 歷史失敗

`test_phase3d.SavedExperimentTests.test_all_original_models_data_research_and_dashboard_unchanged` 呼叫 `src/phase3d_search.py:56` 的保護，要求 main／4619bd4／空 staging；本次 branch／HEAD 不符合，因此 ERROR。沒有改 guard、改測試、改 manifest、skip 或放寬條件。

另行核對歷史 125 檔：123 個研究檔案完全相同；README.md 與 .gitignore 是早已提交的非研究差異，與目前 HEAD 一致。整份歷史 manifest 仍是 FAIL，這個獨立內容核對不替代原契約。建議另行審查歷史 gate 與現行發布 gate 的責任；如需重跑歷史契約，使用隔離的歷史 checkout，不能重設目前工作區。

## Custom Dataset 回歸

三組固定資料：500×10、80×3 含缺失、80×3 含缺失及時間切分。18 次正式六方法訓練、792 筆預測、三次多 k 比較、單筆／批次／部分輸入／模型失效及真正取消，全部通過。最大預測差異 2.842170943040401e-14。多 k 固定方法測 k=[1,3,5] 或 [1,2,3]，RFECV／LassoCV 保留 auto。沒有重新執行歷史完整 k=1..10 benchmark。

| 固定資料 | 多 k 比較實測 |
|---|---:|
| baseline | 105.178 秒 |
| missing | 45.590 秒 |
| chronological_missing | 47.823 秒 |

上述是未受控 HTTP 快取的回歸時間，包含比較工作自身的環境成本，不是 cold-start benchmark，不保證其他 CSV 180 秒內完成。仍保留 100,000,000 工作量上限、10,000,000 高負擔確認、180 秒 timeout、原 SFS／RFECV 訓練限制。SFS 逐 fold／逐 k 原生精確等價與 fallback 測試通過。

Browser 與本機 Server UI 都完成 CSV→確認→驗證→RFE k=2 訓練→Live Predictor；滑桿同步、999 外插保留原值、模式切換清除模型正常。Browser UI 運算 0 次伺服器運算 API 呼叫。Server UI 依原路徑運算；測試建立的 Server 模型已透過原 UI 清除。Vercel 無 Gateway 的 503／明確提示已由本次 Web 測試驗證。

真正 Worker 初始化失敗及注入 5ms 方法 watchdog 都正確清理；這是故障注入，不是改正式 timeout。最後 0 個測試 Worker。

## 互動、畫面與瀏覽器

Dataset Selector、跨頁 URL、refresh／history、無效 ID、兩模型切換、Reset、合法外插、不合法小數／範圍、載入失敗／重試、Holdout 搜尋／誤差排序／明細均 PASS；debounce 與過期回應另由本次完整單元測試驗證。

Chromium 在 1440×1000、834×1112、390×844 × 深淺色 × 三資料集 × 四研究頁共 72 項版面檢查 PASS；6 張截圖已實際檢視。

- macOS Safari：**NOT TESTED**。原生畫面工具遭 ScreenCaptureKit -3801／TCC 權限拒絕；沒有更動權限。
- 實體 iPhone Safari：**NOT TESTED**。無可操作裝置。
- 公開 HTTPS／Vercel Preview：本階段 **NOT TESTED**，未部署。

## Git、安全與建議 staging

開始時 344 個 tracked、28 個 tracked 修改、472 個 untracked；staging 空。815 個已存在且可雜湊的 tracked／non-ignored 檔案本次完全未變；本階段只新增三份 Phase 3G 報告／清單。

建議清單共 **115 個逐檔路徑**，見 [multi-dataset-phase3g-staging.json](multi-dataset-phase3g-staging.json) 的 `files`。112 個既有候選約 2.77 MiB，再加本次三份報告。清單保存這 112 檔的目前 SHA，不能沿用舊 53／36 檔清單，也不能 git add .／-A。清單只是建議，不等於授權提交。

- 候選掃描無 token／private key／credential assignment／私人絕對路徑命中；沒有超過 1 MiB 的候選。這是靜態掃描，不保證排除所有未知秘密。
- `experiments/` 有 385 個 untracked 檔（51,011,398 bytes），不可誤納。
- `student_performance.csv` 為 28,012,365 bytes 的額外原始上傳，與 Automobile_data.csv、student-por.csv、BostonHousing.csv 一律排除。
- BostonHousing.csv 已 ignore、未追蹤，未更動。依賴、.next、虛擬環境、私鑰、.env、cache／測試暫存均排除。
- 六模型 JSON、版本化 snapshots、來源 manifest 與 Browser generated assets 完整；SFS 原始碼／測試／生成 manifest 要一起 review。
- 固定數值測試 fixture 含版本化公開研究輸入，沒有來源列／受試者識別碼，僅測試使用、不在 public／production import。

## Release 決策

**可以安全準備精確範圍的 Commit；完整驗收尚未全綠，因此不宣稱無條件 Preview Release Ready。** 剩餘是歷史 Python 契約的正式處置／說明、真 Safari／iPhone 驗收及公開 HTTPS 驗收。若另行決定先開驗證用 Preview，應明列這些未測／失敗項，不可改稱 PASS。

附帶文件差異：backend/README.md 仍合稱比較／訓練 120 秒，實際比較已為 180 秒；建議後續只修文件。本次未修改 Gateway 或既有文件。

本機最新 Production Server 保持於 http://127.0.0.1:3018。沒有 Git add、Commit、Push、Merge 或部署。
