# Vercel 前端＋獨立 Python 後端：部署計畫

狀態：2026-10-18，M15 已新增可獨立啟動的 HTTP gateway＋Python worker 後端封裝與原生程序整合測試；見 [backend/README.md](../backend/README.md)。Vercel 代理、前端雲端模式與實際部署尚未完成，容器建置也尚未實測。本文件將已完成的後端封裝與後續雲端工作分開記錄。

## 1. 上線目標

訪客開啟 Vercel 網址，就能瀏覽研究、上傳數值 CSV、驗證、比較六方法、訓練、單筆／批次推論與下載；不依賴作者的 Mac 開著。

```text
瀏覽器
  ├─ Vercel：Next.js 頁面、研究快照、圖表、CSV 本地檢查
  └─ Vercel 同來源 API：驗證來源／工作權限，轉送受控請求
         └─ 獨立運算服務：Python 選欄、LR、前處理、暫存模型與推論
```

先採單一後端實例、有限同時運算的作業展示版本。模型過期或後端重啟時，明確要求重新訓練；不承諾永久保存。若要多實例或長期模型，再接共用儲存與工作狀態服務。

## 2. 為什麼不能直接上傳現在的 web 資料夾

| 現有實作 | 雲端需要改的地方 |
| --- | --- |
| Next API 用 child_process 啟動 Repository 的 .venv/bin/python | 改成呼叫獨立運算服務；保留本機模式，不假設 Vercel Node Function 自帶這個 Python 環境 |
| assertLocalRequest 只接受 localhost 的同來源請求 | Vercel 入口驗證實際允許的 HTTPS 網站來源；後端驗證服務身分，不能直接刪除來源限制 |
| ModelStore 是單程序 Map＋本機暫存檔，最多 3 個模型 | 模型集中到同一後端，綁定使用者／工作範圍、期限與刪除；避免後續請求落到不同 Vercel 實例找不到模型 |
| 本機 CSV 上限 25 MiB | Vercel Function 請求／回應有 4.5 MB 限制。第一版建議限制雲端訓練 CSV 為 3 MiB，並檢查整份 multipart 及回應大小；不是只檢查檔案大小 |
| build 前 sync-data 從 web 外讀取 dashboard_data 與 reports | 決定完整 repo 建置或部署專用快照模式，防止只送 web 時把快照改寫成 unavailable |
| /api/predictions 在執行時讀 data/prediction-explorer.json | Vercel Function 需正確包含快照，Preview 測試所有時間視窗，不能只確認首頁可開 |
| 介面文字說資料只在本機處理 | 雲端模式需明示傳送到遠端服務、用途及刪除期限，本機模式維持原說明 |

Vercel 本身提供 Python runtime，但目前程式不是 Vercel Python HTTP Function；也不能靠修改 runtime 就解決模型跨請求保存。官方 runtime 文件說明 Function 檔案系統為唯讀，僅提供 /tmp 暫存。這些是本案選用獨立後端的依據，不代表 Vercel 一概不能跑 Python。

官方依據（查核日 2026-10-18）：[Python runtime](https://vercel.com/docs/functions/runtimes/python)、[Function runtime／檔案系統](https://vercel.com/docs/functions/runtimes)、[Function 容量限制](https://vercel.com/docs/functions/limitations)。實際限制仍依選定方案與上線當日設定確認；本計畫不保證特定價格或免費額度。

## 3. 後端需要提供的介面

對外維持網站現有同來源路徑；Vercel 端代理到運算服務。以下為目標契約，現有程式尚未支援遠端 URL。

| 路徑 | 功能 | 模型狀態 |
| --- | --- | --- |
| POST /api/custom-dataset/validate | 資料、設定、digest、切分檢查 | 不訓練 |
| POST /api/custom-dataset/compare | Training CV 方法與 k 比較 | 不保存模型 |
| POST /api/custom-dataset/train | 另行確認後選欄、LR、CV／Holdout | 回傳暫存模型代碼 |
| POST /api/custom-dataset/predict | 凍結流程的單筆推論 | 僅使用所屬且有效模型 |
| POST /api/custom-dataset/predict-batch | 凍結流程的批次推論 | 同上 |
| DELETE /api/custom-dataset/predict | 主動清除暫存模型 | 刪除後不可推論 |
| GET 健康檢查路徑（待實作） | 確認服務可用 | 不回傳資料、路徑或 Secrets |

沿用現有 Python 函式、JSON schema、digest、fold-local 邏輯及有限 method allowlist；不對上傳者開放任意路徑、Python 參數、程式或序列化模型檔。後端封裝只改呼叫與保存方式，不重寫研究結論，也不重訓能源正式模型。

## 4. 部署實作順序與驗證

1. **後端封裝（M15 已實作）**：backend/server.ts、Dockerfile 與最小 Python 依賴已新增，重用既有處理器與 Python。服務密鑰驗證後依可信 session 分開 ModelStore；一次一項運算、最多 8 個 session、每 session 3 模型／30 分鐘，CSV 3 MiB／完整請求及回應 4 MiB。HTTP 整合測試核對原模型結果、跨 session 隔離、超量、並行、取消與逾時。網站尚未接入，Docker 容器待主機具備環境後實測。
2. **本機跨服務驗證**：前端連本機獨立後端，完成 validate → compare → apply → train → predict → batch → delete；對照舊本機流程的指標與特徵，確保遷移沒有改變模型定義。
3. **Vercel 接線**：新增 server-only 後端 URL、服務 Secret 與允許來源設定。Secret 不用 NEXT_PUBLIC 前綴、不放在瀏覽器或 Repository；模型代碼不可跨使用者任意操作。
4. **建置來源處理**：Vercel Root Directory 預計為 web，保留上游同步來源的建置存取；或新增獨立且驗證快照完整性的部署建置模式。實際選一條並測試，不以靜默跳過缺檔方式成功建置。
5. **後端部署**：選可運行 Python 容器的主機並完成登入，設定埠、健康檢查、Secret、運算限制與過期清理。後端主機尚未選定，不能預先宣稱帳單、容量或 URL。
6. **Vercel Preview**：先上 Preview，設定同一後端、允許的 Preview 網域與必要 Function duration，驗證大小限制、完整 API、快照與錯誤提示。不要無限制允許所有 vercel.app 來源。
7. **正式發布**：Preview 驗證通過後發布 Production，關閉作者本機服務再從另一台裝置完成一次全流程，最後補正式網址與實測紀錄到報告。

本機運算期限現為選欄／訓練 120 秒；服務代理需預留連線、上傳及回傳時間，不能把每一層都設成相同 120 秒。若選定主機不能承載此請求時間，就實作工作 ID＋輪詢及取消，不能只提高前端等待時間。初始版不要求泛用大規模排程，但需實測取消後 Python 程序已結束及資料已清除。

Vercel Git import 會使用遠端版本，不包含目前未提交檔案。依使用者要求，現在不 commit／push；可選 CLI 產生 Preview，或之後獲授權再用 Git 流程。CLI 可不連 Git，但仍需登入帳號、確認目標專案與部署內容。[官方部署方式](https://vercel.com/docs/deployments)

## 5. 雲端完成標準

- 所有五頁可直接開啟與重新整理；研究圖表與 /api/predictions 讀到正確版本。
- 雲端 CSV 上限與隱私說明對應實際限制，超量在運算前以可理解訊息拒絕。
- 六方法比較可用；固定 k 與自動 k 的表示保留，零特徵與失敗不偽造成成功。
- 確認後訓練、單筆／批次推論、下載、刪除與過期均通過；不依賴作者 Mac。
- 超時、忙碌、斷線、服務重啟均有明確結果；取消不留下無上限背景運算。
- 不同使用者不能取得或清除彼此模型；共用服務 Secret 不出現在前端回應、日誌與匯出檔。
- Preview 與 Production 的環境設定分別核對，測試資料及上傳檔不公開存入靜態目錄。
- 保存部署網址、建置版本、雲端測試範圍與限制，才將報告的部署狀態改成完成。

## 6. 目前還需要的資訊

實際發布需要使用者的 Vercel 登入／目標專案，以及後端主機帳號與可接受的方案。登入由使用者在平台完成，不把密碼或 Secret 貼到對話。老師原始完整作業要求、姓名學號及實際 NotebookLM 附件仍屬交付材料待補，不影響先實作部署封裝。


## M16 進度補充

Vercel／Next API 的 server-only 代理與簽署匿名 session 已實作，且保留未配置時的原本機模式。連線需要 AEP_BACKEND_URL、AEP_BACKEND_TOKEN、獨立 AEP_SESSION_SECRET 與精確 AEP_PUBLIC_ORIGIN；配置細節見 [web/README.md](../web/README.md) 的 M16。這是伺服器間接線與隔離，尚非使用者登入或完整公開配額。

目前未對原本機網站設定遠端變數。剩餘工作為雲端模式的容量／資料傳送文字、部署快照與執行時包含、後端容器實測與主機設定，以及 Vercel Preview／Production 驗收。不得只因 API 接線存在就將本文件上方的部署工作標為全部完成。
