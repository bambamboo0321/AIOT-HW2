# M17 Browser Mode：Vercel 發布準備

狀態：2026-10-20。Browser Mode 已完成本機功能與數值驗收，尚未 Commit／Push／公開部署。M16 Gateway 與 API 保留；本文件取代較早的「必須先部署 Python 後端」計畫。歷史 M15／M16 紀錄仍保留。

## Vercel Hobby 必要設定

| 設定 | 本專案建議 |
| --- | --- |
| Git Repository | 既有 bambamboo0321/AIOT-HW2；需要先由使用者授權提交及 Push M17，不能部署目前只有 M16 的遠端當成 M17 |
| Framework Preset | Next.js |
| Root Directory | web |
| Include source files outside of the Root Directory in the Build Step | 開啟；建置需要 ../src、dashboard_data 與 reports，不可只上傳 web |
| Node.js Version | 24.x，與本機 Node 24／package.json 相容 |
| Install Command | npm ci（保留 web/package-lock.json；不要加 --omit=dev，建置需 TypeScript／Tailwind） |
| Build Command | npm run build；保留 prebuild 的 sync-data 與 build-browser-compute，不直接 next build 跳過前置步驟 |
| Output Directory | 使用 Next.js 預設；不要填 public、out 或使用靜態匯出 |
| Environment Variables | Browser-only 部署不需要任何 Secret 或 AEP_* 設定；不要填假的後端值；保留 Vercel 自動提供的 VERCEL 系統變數 |
| Branch | 經確認後的 M17 commit；先用 Preview，確認結果後才決定 Production Branch，不能只靠 main 現有 M16 發布 M17 |

以上設定依 [Vercel 建置說明](https://vercel.com/docs/builds/configure-a-build)、[上層來源存取](https://vercel.com/docs/monorepos/monorepo-faq)、[Node 版本](https://vercel.com/docs/functions/runtimes/node-js/node-js-versions) 查核。Hobby 適合個人非商業作業展示，仍受用量限制，不保證無限免費資源；見 [Hobby 官方說明](https://vercel.com/docs/plans/hobby)。不需另租 Python 伺服器即可使用 Browser Mode。

## Browser／Server 的實際行為

預設 Server Mode 保留。Custom Dataset 頁面動態解析安全的能力提示，只傳「設定是否齊備／本機 Python 是否存在」及固定文字，不傳環境值、密鑰或本機路徑。有設定不代表服務健康；M16 API 仍負責原有驗證及錯誤回應。

在 VERCEL 環境未設定完整後端時，畫面明示 Server 不可運算，Provider 回報 503 而不傳送 CSV；直接呼叫 M16 API 仍為原有 503。請手動切換 Browser Mode。Browser 請求完全由獨立 Client／Worker 處理，不接觸 Gateway、不自動回退。模式、CSV 或資料設定改變會清除不相容的結果與模型。

如日後也要啟用 Server Mode，另依 [backend/README.md](../backend/README.md) 部署 Python Gateway，再設定 server-only 的 AEP_BACKEND_URL、AEP_BACKEND_TOKEN、AEP_SESSION_SECRET、AEP_PUBLIC_ORIGIN；不能加 NEXT_PUBLIC_，不能提交 .env 或把 Token 放到文件。Preview／Production 的來源與 Session 必須分開驗證。Browser-only 驗收不代表 Server 的雲端容器、配額或多人負載已通過。

## 靜態資源與安全政策

- Worker 為同來源 /browser-compute/worker.mjs；正式 Python 原始碼位於 /browser-compute/python-manifest.json，以每模組 SHA-256 及整份 manifest lock 驗證。建置從當前 src/ 取得原樣內容，不使用舊實驗快照。
- 保留 public/browser-compute 的 Worker、manifest 與產生的 modules；它們是必要的公開運算程式，不是快取。測試 fixture、使用者 CSV、私有模型、.venv、node_modules、.next、實驗原生環境均不得放進 public。
- Pyodide 固定 https://cdn.jsdelivr.net/pyodide/v314.0.7/full/；Python 3.14.2、NumPy 2.4.6、pandas 3.0.2、SciPy 1.18.0、sklearn 1.8.0、joblib 1.5.3，Worker 會核對實際版本。無 pip、無 WASM 轉送層。
- 僅 Worker 回應有專用 CSP，允許同來源程式、固定 CDN 路徑及 wasm-unsafe-eval；未放寬全站 CSP、未加 unsafe-eval／unsafe-inline、未開放 Gateway CORS。主文件目前沒有 CSP；未來若額外新增，需明確容許同來源 Worker，不能直接套用互相衝突的全站政策。
- 官方 CDN 的 JS 必須為 JavaScript MIME，WASM 必須 application/wasm，且 CORS 允許載入；不得改為錯誤 MIME 的轉送。Worker 不需要 SharedArrayBuffer／COOP／COEP。若主機額外啟用 COEP，需重新驗收外部資源。
- 不使用 basePath／CDN assetPrefix 或全站重寫；目前 Worker 路徑固定網站根目錄。public 資源保留重新驗證策略，Worker no-store；不能把無內容 hash 的 manifest／Worker 改成 immutable。混合版本應失敗並要求重新載入，不能跳過 hash。
- 正式服務使用 HTTPS；本機 loopback 為測試例外。Web Crypto、module Worker 與 WASM 必須在實際瀏覽器可用。Vercel 網域憑證與自訂網域 DNS 的關係見 [SSL 官方文件](https://vercel.com/docs/domains/working-with-ssl)。

## 部署後 HTTPS 驗收清單（尚未執行）

- [ ] 確認部署 commit SHA 為核准 M17；五頁與 /api/predictions 所需研究快照正常，沒有 unavailable 或缺檔。
- [ ] HTTPS 憑證有效、HTTP 轉址、無 mixed content；自訂網域按 Vercel 要求設 DNS。訪客可開啟頁面，Preview 存取保護按展示需求設定。
- [ ] Server 預設且無 Gateway 時明確提示不可用；直接 API 回應 503；頁面不誤稱 Server 可用。
- [ ] Worker、manifest、9 個模組均 200／正確 MIME；Worker CSP／nosniff／快取標頭符合設定，無 CSP／CORS／WASM console 錯誤。
- [ ] 官方固定 CDN runtime、WASM、科學套件均成功；版本與 source hash 正確。記錄新快取分區、停用快取與暖快取各自條件，不混稱冷啟動。
- [ ] Browser 輸入 CSV 後驗證、六方法、多 k、LR、Holdout 指標、單筆與批次推論、圖表／下載均可用；Network 中不得出現 Browser 資料的 /api/custom-dataset 請求或任何 CSV 上傳。
- [ ] 2,000 列／15 候選特徵／3 MiB 訓練資料邊界與超限提示、比較超限不阻止合法訓練、缺失值及時間切分正常，Server 限制不變。
- [ ] 取消／逾時／重試、改設定、換 CSV、切模式、離頁及模型過期清理均正確，沒有殘留 Worker 或舊結果。
- [ ] 桌面與手機、深淺色與圖表正常，量測實際裝置等待時間；不宣稱任意資料皆低於 60 秒。
- [ ] CDN 被阻擋時顯示載入失敗，可手動重試，不能自動上傳到 Server；若另啟用 Server，再驗證來源限制、Session 隔離、取消、超量及完整 Gateway 流程。

## Python 歷史保護與新發布驗收

既有 tests/test_phase3d.py 的 test_all_original_models_data_research_and_dashboard_unchanged 呼叫 src/phase3d_search.py:check_protected，要求 main／4619bd493a337f32d34593f12686035f2c70a630／空 staging。現行 experiment/m17-pyodide／61478fc 不符合，因此全套 211 中有 1 項歷史 Git 契約錯誤（先前 210 通過），不是 Browser 演算法錯誤，不能標示全套 PASS。

reports/phase3/phase3d/integrity_baseline.json 的 125 檔仍有獨立歷史意義；README.md 與 .gitignore 在 M16 公開整理已變更，其餘研究保護檔須另外比對。不要改舊 manifest、刪測試、skip、mock Git 或切回舊 HEAD 偽造驗收。

正確處理是保留原 guard 供原 Phase 3D 流程使用，另建立本次 release attestation：以 M16 Stable SHA 比對研究／src／Gateway 完整性，明列核准網站與文件差異，再記錄 Browser／M16 回歸結果。這份新發布紀錄是補充，不能把歷史失敗改寫成通過。若學校／CI 要求整套無錯，仍須另行核准拆分「歷史重現契約」與「現行發布完整性」的測試設計；本階段不變更研究保護。

## 提交範圍

完整明確清單見 web/verification/m17-release-candidates.json；建議訊息：

    feat: add verified Pyodide browser compute mode

只提交 M17 web 程式、可重現公開資源、小型合成 fixture、驗收紀錄與必要 README／部署文件／ignore 規則。排除 BostonHousing.csv、Secrets、experiments/ 歷史及原生環境、下載的 WASM／wheel、快取、截圖與大型測試產物。現有 M16 已追蹤研究檔不需要重新加入。本階段未執行任何 staging、Commit、Push 或部署。
