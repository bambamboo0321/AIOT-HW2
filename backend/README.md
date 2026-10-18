# 獨立 Python 運算後端（M15）

這個目錄把既有 Custom Dataset 運算封裝為可獨立啟動的 HTTP 服務。Node 24 僅負責 HTTP、服務驗證、模型登記與 Python 子程序生命週期；前處理、六方法選欄、LinearRegression 與推論仍由原 Python 模組執行，沒有搬寫模型算法。

**目前狀態：獨立後端已可本機測試；網站尚未切換到此服務，Vercel 與雲端主機尚未部署。** 本機網站仍使用原 API，不需要更改已驗收的操作。

## 本機啟動

先備妥 Node 24 與專案 Python 環境。此 gateway 沒有額外 npm 執行依賴，也不需要 npm install。

```bash
# 在 AIOT-HW2 Repository 根目錄執行
export AEP_PYTHON="$PWD/.venv/bin/python"
# 每次本機測試產生新服務密鑰；正式服務由主機的 Secret 設定注入。
export AEP_SERVICE_TOKEN="$(node -e "process.stdout.write(require('node:crypto').randomBytes(32).toString('base64url'))")"
npm --prefix backend start
```

預設只監聽 `127.0.0.1:8000`。`GET /healthz` 回傳 `{"status":"ok"}`，僅表示 gateway 在運作，不是模型品質或全部 Python 依賴的保證。Ctrl-C／SIGTERM 會停止服務、中止運算並清除此程序持有的暫存檔。

設定說明見 [.env.example](.env.example)。範本不會自動載入，需由 shell 或主機設定環境變數。`AEP_PYTHON` 為絕對可執行路徑；Repository 位置由程式自身定位，不依賴啟動時所在目錄。`PORT` 可改，公開容器以 `AEP_BIND_HOST=0.0.0.0` 監聽，由主機提供 HTTPS。

## 介面與服務驗證

只有健康檢查不用服務密鑰。其他入口都要求：

```text
Authorization: Bearer <AEP_SERVICE_TOKEN>
X-AEP-Session: <由可信前端伺服器建立的 64 個小寫十六進位字元>
```

服務密鑰至少 32 個 base64url 字元，正式使用隨機值；不可放在 NEXT_PUBLIC 變數、瀏覽器、下載檔或公開 Repository。後續 Vercel 代理應從簽署且 HttpOnly 的 cookie／已驗證使用者建立穩定 session，不可直接相信瀏覽器自報的 session 標頭。目前尚未實作該代理或登入機制。

後端不提供 CORS。服務驗證通過後才建立內部 localhost Request 並重用原處理器，因此不必移除原本網站的 localhost 同來源保護；使用者傳入的 Origin、Host 或 forwarded headers 不作為服務授權。

| 方法／路徑 | 請求內容 | 行為 |
| --- | --- | --- |
| GET /healthz | 無 | 最小存活回應 |
| POST /api/custom-dataset/validate | multipart：csv、config、review | 沿用資料與 digest 驗證，不訓練 |
| POST /api/custom-dataset/compare | 上述內容＋experiment | 六方法 Training CV 比較，不保存模型 |
| POST /api/custom-dataset/train | 上述內容＋獨立訓練 experiment | CV／Holdout 與最終暫存模型 |
| POST /api/custom-dataset/predict | JSON：model_id、values | 凍結模型單筆推論 |
| POST /api/custom-dataset/predict-batch | JSON：model_id、features、rows | 凍結模型批次推論 |
| DELETE /api/custom-dataset/predict | JSON：model_id | 清除所屬 session 的模型；已失效可重複刪除 |

`config`、`review`、`experiment` 的內容、明確確認、SHA-256、回應 schema 與既有 [訓練契約](../docs/custom_dataset_training.md) 相同；不接受任意 Python 模組／參數／伺服器路徑或上傳模型。模型代碼屬所在 session，其他 session 即使取得同一代碼也無法推論或清除原模型。

## 展示版本的資源與保存限制

- 整個服務一次處理一項運算，忙碌時 429 並提供 Retry-After；不是排隊服務。
- 完整請求與回應各最多 4 MiB；CSV 最多 3 MiB。雲端限制比原本機 25 MiB 小，是為後續 Vercel 代理預留表單空間；瀏覽器端提示仍待前端接線時調整。
- 上傳讀取最長 20 秒；Python 驗證 30 秒、比較／訓練 120 秒、單筆 10 秒、批次 30 秒，沿用原有方法大小與運算量限制。
- 最多 8 個模型 session；每 session 最多 3 個模型，保留 30 分鐘，新增模型可能淘汰該 session 的舊模型。原本機網站的全域 ModelStore 不變。
- 上傳暫存在運算結束後清除。模型私有保存於此後端，工作階段到期／模型到期、刪除或正常停止時清除。非正常 SIGKILL／主機故障時無法保證程式執行清理，容器重建應拋棄其臨時檔案系統。
- 模型登記在單一 gateway 程序記憶體中；重啟失效，即使掛上持久磁碟也不會自動復原。第一版只能跑單一實例，不使用多 worker 或多副本。
- session 標頭只隔離模型，不等於公開網站已具備登入、防濫用或使用者配額；這些需在 Vercel 代理接線時一起完成。

## 容器部署檔

從 Repository Root 建置：

```bash
docker build -f backend/Dockerfile -t aep-compute .
docker run --rm -p 127.0.0.1:8000:8000 -e AEP_SERVICE_TOKEN aep-compute
```

容器使用 Python 3.10、Node 24 與 [最小 Python 依賴](requirements.txt)，以非 root 使用者運作。只 COPY `src/`、gateway 與所需 web/lib；不複製研究 CSV、reports、模型、帳號密鑰或整個網站建置。不要在 Dockerfile 內寫入真實 Secret。

**本機沒有 Docker 執行環境，這次尚未實際 build／run 容器。** HTTP／Python 整合是在主機原生程序中驗證；首次選定主機後仍需驗證容器套件、健康檢查、埠與 HTTPS。

## 檢查與下一階段

```bash
export AEP_PYTHON="$PWD/.venv/bin/python"
npm --prefix backend test
web/node_modules/.bin/tsc --project backend/tsconfig.json
cd backend
../web/node_modules/.bin/eslint --config ../web/eslint.config.mjs server.ts tests/service.test.ts --max-warnings=0
```

測試會在暫時 localhost 埠建立真實 HTTP 服務，對照原本機處理器的選中特徵／CV／Holdout，檢查全部六方法、單筆及批次、跨 session 隔離、刪除、過期、確認摘要、容量、並行、中斷、逾時與暫存清理。使用合成資料，不覆寫能源研究成果。

下一步是 Vercel server-only 代理與穩定 session、部署建置快照、雲端模式說明和容量提示。之後才在主機與 Vercel Preview 發布；見 [部署計畫](../docs/vercel_deployment_plan.md)。
