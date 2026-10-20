# Appliances Energy Prediction Lab — M17 Browser Mode

研究 Dashboard 與自訂 CSV 線性迴歸工具。M17 Browser Mode 已完成：CSV 正式驗證、六種特徵選擇、多 k 比較、CV／Holdout 評估、單筆與批次推論可在瀏覽器 Worker 執行，不需要 Python Gateway。M16 Server Mode 仍為預設；無後端時明確提示無法運算，請使用者自行切換 Browser Mode，不會自動上傳或回退。

**尚未公開部署。** 已完成本機功能驗收與發布準備；公開 HTTPS 驗收仍待部署後執行。部署設定、資源與 HTTPS 清單見 [部署說明](../docs/vercel_deployment_plan.md)，本次提交候選與驗收見 [發布紀錄](verification/m17-release-result.json)。下方 M1～M16 段落保留各階段歷史，最新狀態以本節與部署說明為準。

Browser 限制：500 列、10 個數值候選特徵、每份 CSV ≤1 MiB。Pyodide 314.0.7 與套件版本固定；載入官方 HTTPS CDN 後，運算和暫存 joblib 模型留在 Worker 記憶體。模型最多 3 個、30 分鐘，離頁／取消／切換模式或資料設定會清除。不同網路與裝置不保證 60 秒完成。詳見 [核心說明](browser-compute/README.md)。

## 安裝與啟動

在 Repository Root：

```bash
cd web
# 若使用 nvm，可先執行 nvm use；.nvmrc 記錄本次驗收版本。
npm ci
npm run dev
```

使用 Node 24（本次 24.13.1）。`npm install` 也可安裝；版本鎖定於 package-lock.json，`npm ci` 適合重現。npm 快取固定在 web/.npm-cache，node_modules、建置檔與截圖亦留在 web/。開發伺服器只綁定 `127.0.0.1`，預設網址 `http://127.0.0.1:3000`；不會正式部署。若埠已使用，依 Next 啟動輸出確認實際埠，或 `npm run dev -- --port 3001`。

```bash
npm run sync-data
npm run test
npm run typecheck
npm run lint
npm run build
# 可選：只在本機檢視已建置版本
npm start
```

dev／build 的前置腳本會執行 sync-data 與 build-browser-compute（從目前 src/ 原樣封裝 11 個 Python 模組、manifest／hash 與瀏覽器模組）。建置需要完整 Repository 的上層來源，不能只上傳 web/。使用 Next 官方支援的 Webpack；此環境的預設 Turbopack 在 CSS 處理時受 sandbox 埠限制。Next 遙測已關閉，不使用第三方 Analytics、外部字型、API Keys 或付費服務。

ESLint 使用 ESLint 10、typescript-eslint 與 React Hooks recommended 規則，沒有使用 eslint-config-next，也不宣稱包含 Next 特有的全部 lint 規則。歷史選型不代表目前套件漏洞狀態；請以發布時 npm audit 實測為準。

## 路由與元件

| 路由 | 目前狀態 |
| --- | --- |
| `/` | 完整 Overview |
| `/feature-selection` | M2 互動特徵選擇成果 |
| `/model-evaluation` | M3 固定模型評估與殘差／PI 比較 |
| `/prediction-explorer` | M4 時間視窗與單筆預測／區間 |
| `/custom-dataset` | CSV 檢查、本機 Python 驗證、明確確認方法與 k 後的選欄／全部欄位 LR／平均值比較、CV 與 Holdout 評估 |

- `app/layout.tsx`：共用 Layout、主題初始化、Metadata。
- `components/shell.tsx`：Sidebar、Header、手機導覽、Theme Toggle。
- `app/page.tsx`：Server Component，顯示研究摘要、Dataset、模型比較、PI、CRISP-DM、研究發現與來源。
- `components/performance-chart.tsx`：Overview 圖表 Client Component，切換 RMSE／MAE／R² 及 Tooltip；完整數值另有可讀表格。
- `lib/research-data.ts`：來源 allowlist、runtime validation、精簡 Adapter。
- `lib/load-overview.ts`：server-only 快照讀取，缺資料回傳明確狀態。

## 安全且可重現的資料同步

Overview 讀取下列四個來源：

1. `dashboard_data/dataset_summary.json`
2. `dashboard_data/model_comparison.json`
3. `dashboard_data/prediction_intervals.json`
4. `reports/phase3/phase3d/comparison_summary.json`

`sync-data` 從腳本本身的位置定位 Repository，不依賴私人絕對路徑。使用固定 allowlist、realpath 邊界檢查、schema／有限數值／比例／特徵數／切分一致性驗證，精簡成 `web/data/overview.json`；只寫入 web/data，以 temporary file＋rename 更新快照。每個來源保存相對路徑與 SHA-256，沒有時間戳，因此相同來源重複同步位元組一致。

沒有複製 `data/raw/`、模型、逐筆預測、訓練診斷或完整 profiling；研究資料沒有放進 public，也沒有任意檔案 API。M17 的 public/browser-compute 僅提供可公開的運算程式與原始碼 manifest，不含使用者 CSV、模型或研究原始資料。瀏覽器只收到顯示需要的摘要與兩模型圖表數值。M2 另外讀取 feature_selection 摘要；所有 dashboard JSON 原檔均保留原樣。Phase 3D 透過獨立 Adapter 讀取，未修改原始結果。

Dataset／模型／PI 任一必要資料缺少或不符，頁面顯示「研究資料尚未就緒」，不填替代數值；只有 Phase 3D 缺少時仍顯示核心 Overview，另呈現探索性資料缺失訊息。來源修正後執行 sync-data 並重新啟動 dev；production build 須重新 build，因為頁面是靜態產生的快照，不提供即時訓練／檔案監看服務。

## 評估解讀

- 模型表格與互動圖只使用固定 chronological Holdout；不混入 Random Split 或 Training CV。
- 保留 Phase 2 已查看 Holdout 的限制。
- 95% 標示從 nominal_coverage 讀取，是單筆 Prediction Interval，不是平均值 CI。
- Coverage 原值為 0～1，顯示百分比時乘 100；寬度、RMSE／MAE 單位從 JSON 讀取。
- Phase 3D 顯示的是探索性固定組合 CV 與相同條件的固定 MI 參考，不以 fold-local MI 程序分數或 Holdout 分數替代。
- 較少 Features 不保證較準；較低的抽樣 CV 最小值不是獨立泛化驗證，未替換正式模型。
- CRISP-DM Deployment 明示仍在開發，僅本機流程及介面，未正式部署。

## 主題與人工驗收

首次載入依系統偏好選深／淺色；右上按鈕可切換，偏好只存本機 `localStorage` 的 `aep-theme`。主題初始化在畫面繪製前執行；storage 不可用時仍可切換。尊重 reduced-motion，提供 Skip Link、語意表格、鍵盤焦點及進度條。

建議人工驗收：

1. 在 1440px 桌面與 390px 手機寬度檢查 Overview、文字與圖表。
2. 切換深／淺色並重新整理，確認偏好保存。
3. 點 RMSE／MAE／R²，確認只有 Holdout 圖表變動，表格保留完整指標。
4. 手機開啟導覽、切換四個路由，確認 Feature Selection、Custom Dataset 可操作、目前頁面標示及返回 Overview。
5. 展開 Phase 3D 的組合／限制與資料來源，確認來源相對路徑。
6. 核對 RMSE、MAE、R²、Coverage、寬度與原 JSON；不要把探索性結果當成最終模型。

本次自動化與完整性結果見 `verification/`；截圖若可取得存放忽略的 `artifacts/`。原有 Python 測試從 Repository Root 執行：`.venv/bin/python -B -m unittest discover -s tests -q`。未修改任何前階段 Python 模組或成果。

參考：[Next.js 安裝／Webpack](https://nextjs.org/docs/app/getting-started/installation)、[Tailwind CSS](https://tailwindcss.com/docs/installation/framework-guides/nextjs)、[Recharts](https://recharts.github.io/en-US/guide/installation/)。


## M2：Feature Selection

開啟 `http://127.0.0.1:3000/feature-selection`。沿用既有深／淺色與導覽，不執行訓練。

- 六方法切換：F-test、MI、Forward SFS、RFE、LassoCV、RFECV。
- RMSE／MAE／R² 與實測 k 選單。固定 k 曲線只使用 Phase 3B 的 outer_k_curve；SFS 保留 9 個實測 k，連線只是閱讀輔助，未測 k 不補點。
- LassoCV／RFECV 顯示五個 Outer fold 實際分數的折線，橫軸為時間順序 Fold 1～5，每點包含該 fold 的整數 k；RMSE／MAE／R² 可切換。卡片、Tooltip 與表格皆呈現真實數值，平均值為虛線參考。每個 fold 的時段不同，這不是固定 k 的性能曲線；不補造 k=1～25 的資料。完整 Training 最終 k 獨立呈現。
- 特徵清單對應完整 Training 的最終 refit，與曲線選單分開；不把某個 fold 或其他 k 的清單冒充最終選欄。
- 六方法比較可切換 Nested Outer CV 與最終選模 CV。正式 Winner 沿用 Training CV 的 RMSE 選模規則，不隨瀏覽指標改變。
- Holdout 僅顯示 Phase 3C 的 Baseline／MI；不補造六方法 Holdout。明示 Phase 2 已看過 Holdout 的限制。
- Phase 3D 的 fold-local selector CV（B）與固定組合／1000 抽樣最小值（A）分表呈現。均屬 Training 探索；不與 Nested CV 或 Holdout 當成同層級排名。沒有 SFS k=6 完整 Training 固定組合。
- Phase 3A 為框架與成本 Benchmark，沒有拿 Benchmark 分數排名。

M2 的固定讀取來源：

1. `dashboard_data/feature_selection.json`
2. `reports/phase3/results/experiment_config.json`
3. `reports/phase3/results/best_features.json`
4. `reports/phase3/results/plot_data.json`
5. `reports/phase3/results/feature_selection_results.json`（僅提取 outer_evaluation 的 30 筆 fold 摘要）
6. `reports/phase3/phase3d/comparison_summary.json`（可選）

`lib/feature-selection-data.ts` 核對 schema、完整 folds、k grid、欄位 mapping、評估層級、數值與 Training-only 條件，也核對 Dashboard 所記錄的 best_features／plot_data／feature_selection_results 雜湊及正式內容。不同或過期即顯示資料不可用。Phase 3D 缺少時，正式 Phase 3B 內容仍可使用。`sync-data` 新增寫入 `web/data/feature-selection.json`，保留每個來源的 SHA-256；原 Overview 快照保持位元組一致。載入器為 server-only，只有顯示所需摘要送往瀏覽器。

新增檔案：`app/feature-selection/page.tsx`、`components/feature-selection-explorer.tsx`、`lib/feature-selection-types.ts`、`lib/feature-selection-data.ts`、`lib/load-feature-selection.ts`、`tests/feature-selection.test.ts`；導覽將 Feature Selection 移除 Coming Soon，里程碑標示改為 M2。

M2 人工驗收：

1. `npm run dev`，開啟 `/feature-selection`，以桌面 1440px 與手機 390px 檢查。大表格可在自己的區塊內橫向捲動；頁面本身不應水平溢出。
2. 逐一切換六方法、三指標。固定方法改變 k，檢查精確輸出；SFS 只能選實測 k，不能選 k=6。
3. LassoCV／RFECV 應有五個實測點與平均虛線、沒有固定 k 選單。逐一 hover 對照各 fold 的 k 與三指標，注意同 k 在不同 fold 的分數仍不同。
4. 更改曲線 k 時，確認最終特徵清單仍標示自己的最終 k；切換方法時才更新該清單。
5. 切換 Nested CV／最終選模 CV，核對六方法與來源 JSON；Holdout 區仍只有兩個模型。
6. 檢查 Phase 3D 的 A／B 分表、限制說明、來源與雜湊。切換深／淺色、重新整理確認保存，手機導覽可開關並返回 Overview。

M2 驗證報告見 `verification/m2-result.json`；瀏覽器驗收見 `verification/m2-browser-checks.json`。未 commit 或 push。

M2 顯示修正：LassoCV／RFECV 的每個 Outer fold 都以獨立卡片顯示實際整數 k；圖表摘要表也列出全部 fold k。驗收時確認 LassoCV 為 2、16、3、14、16（以實際 JSON 為準），最終 k 與 CV 平均 k 分開。

逐 fold 資料直接讀取正式 `feature_selection_results.json`，僅取 `outer_evaluation`；核對來源雜湊、fold 完整性、時間 Training 邊界、k 與摘要平均一致，不讀模型或重新計算分數。未傳送完整研究紀錄到瀏覽器。


## M3：Model Evaluation

開啟 `http://127.0.0.1:3000/model-evaluation`。沿用共用深／淺色與導航，里程碑標示更新為 M3。

- 兩個既有固定模型：Baseline LinearRegression（25 欄）與 MI + LinearRegression（6 欄）；沒有補造六方法 Holdout 排名。
- 預設完整 chronological Holdout，可切換完整 Training 的樣本內擬合；不是 CV。RMSE／MAE／MSE／R² 均保留正式 JSON 精度，另列 MI − Baseline 差值。
- 讀取保存的 3,947 筆預測，彙整殘差 = Actual − Prediction。殘差為正表示低估，為負表示高估。兩模型使用相同的 20 個等寬分箱，範圍對稱涵蓋全部極值，沒有抽樣或裁切。可選兩模型／Baseline／MI，並顯示平均、線性插值分位數、最大絕對殘差與方向計數。
- PI 可切換實測覆蓋率、平均寬度、區間外筆數；固定名目 95%，另列完整涵蓋／低於下界／高於上界、寬度中位數與 SD（ddof=0）。名目覆蓋率不是分類準確率或實測保證；不截斷負下界，不依 Holdout 校準。
- 明示已查看 Holdout、Training 擬合、同時間估計、時間相依與選模不確定性等限制；CV 詳細資料仍連回 Feature Selection。

M3 固定只讀來源：

1. `dashboard_data/model_comparison.json`
2. `dashboard_data/prediction_intervals.json`
3. `reports/phase3/phase3c/prediction_metrics.json`
4. `reports/phase3/phase3c/prediction_interval_metrics.json`
5. `reports/phase3/phase3c/evaluation_plan.json`
6. `reports/phase3/phase3c/figure_manifest.json`
7. `reports/phase3/phase3c/prediction_interval_predictions.csv`（既有預測匯出，不是原始 Dataset）

Adapter 核對 Dashboard 的上游 SHA-256、Figure Manifest 的預測 CSV 雜湊、各文件的計畫、單位、完整 Holdout 筆數、連續列索引及十分鐘間距。由保存預測核對 RMSE／MAE／MSE／R²、PI 計數／Coverage、平均／中位寬度與 SD；不符合即顯示明確資料錯誤，不發佈部分指標。讀取既有 CSV 不會呼叫模型、重新 fit 或重新產生研究成果。CSV parser 僅接受此固定、無引號的數值匯出格式，不提供一般 CSV 上傳。

新快照 `web/data/model-evaluation.json` 只包含兩模型摘要、20 個分箱與描述統計，不複製逐筆預測、係數、SVD 診斷、原始 Dataset、模型檔或使用者絕對路徑。原 `overview.json`、`feature-selection.json` 同步後位元組不變。日期依來源的無時區字串顯示，不轉換時區。

M3 人工驗收：

1. 開啟 Model Evaluation，確認只有兩個固定模型，頁首切分日期與原成果一致。
2. 切換 Holdout／Training，逐一點 RMSE、MAE、R²、MSE；圖表、完整表格與差值應同步更新。Training 應明示樣本內擬合。
3. 切換殘差圖模型。每模型 20 個分箱總數須為 3,947；正殘差為低估。展開分箱表，確認極端尾端仍被納入。
4. 切換 PI 的覆蓋率、平均寬度、區間外筆數，核對單位及計數。只有覆蓋率圖應出現名目 95% 虛線。
5. 在 1440px 桌面與 390px 手機檢查深／淺色、保存偏好、Tooltip、導覽與表格區塊橫向捲動；頁面本身不應水平溢出。
6. 返回 Overview 與 Feature Selection，確認 M2 的五 fold 實際分數折線與固定 k 曲線保留。

網站測試共 40 項。M3 報告及瀏覽器驗收保存於 `verification/m3-result.json`；未 commit、push、重新訓練或正式部署。


## M4：Prediction Explorer

開啟 `http://127.0.0.1:3000/prediction-explorer`，沿用共用深／淺色與手機導覽，里程碑標示 M4。

- 在完整 chronological Holdout 的 3,947 筆保存預測中，選擇 72／144／288 筆（12／24／48 小時）視窗，使用最前／上一段／下一段／最後，或指定十分鐘時間點跳轉。
- 預設沿用 Phase 3C 評估計畫預先固定的前 144 筆；其他時段明示為描述性探索。最後一段保持所選長度，因此可能與前段重疊。
- 圖表呈現全部視窗觀測，不抽樣或平滑；切換兩模型／Baseline／MI、預測圖／殘差時間圖。PI 陰影可選其中一模型或關閉；保留負下界，不重新校準。殘差圖有零參考線。
- 點圖表、表格時間或觀測選單查看單筆的實際值、預測、signed residual、絕對誤差、PI 下／上界與閉區間涵蓋判定。逐筆表格每頁 24 筆；切頁不影響圖表資料。
- 視窗 RMSE／MAE／R²／Coverage／平均寬度／區間外計數與完整 Holdout 固定參考分開；常數或不足兩筆時 R² 顯示「無法定義」。日期保留來源時鐘，不轉換時區。模型估計同時間能耗，不宣稱多步預報。

資料沿用 M3 的七個固定來源。`buildPredictionSnapshot` 先沿用完整評估驗證，再比對重新讀取的保存預測與評估計畫雜湊，避免讀取期間版本不一致。新 `web/data/prediction-explorer.json` 是 server-only 衍生快照，沒有複製原始輸入 Features、原始 CSV、leverage、係數或模型。

新增僅讀端點 `GET /api/predictions?start=0&size=144`：start 為零起算，size 僅允許 72／144／288，最多回傳 288 筆；拒絕負數、小數、超範圍、重複與未知參數。靠近結尾的合法 start 會向前移，保持完整視窗；回應包含實際 start。沒有檔案路徑、Target 或訓練參數，POST 不支援。回應標示 private/no-store。初始頁面也只序列化前 144 筆，而不是把完整系列送入前端。

客戶端會核對回應筆數、列序、時間間距、PI 及視窗統計，並確認來源版本沒有改變。切換視窗取消舊請求，舊回應不得覆寫新視窗；載入失敗保留上一個成功視窗並明示錯誤。缺少來源／損壞快照不補造資料，端點回傳 503。

M4 人工驗收：

1. 開啟 Prediction Explorer，預設應為 2016-04-30 08:20 起的前 144 筆。切換各視窗長度，檢查完整圖點與表格筆數。
2. 切換模型、PI 陰影與殘差。Hover 對照保存數值，點圖表／選單／表格選取單筆，檢查明細及涵蓋狀態。
3. 前後移動與最後一段，確認包含第 3,947 筆、沒有空白補點。指定最後時間跳轉，應選取最後一筆；時區不應偏移。
4. 檢查視窗統計會隨時段改變，頁首完整 Holdout RMSE 不變。表格分頁應保留圖表的整個視窗。
5. 在 1440px 桌面與 390px 手機切換深／淺色，確認圖表、表格區塊水平捲動與導覽。回查 M1／M2／M3。

M4 含既有測試共 53 項；結果與截圖記錄於 `verification/m4-result.json`。不重新訓練、不呼叫模型推論，不改寫研究成果；未 commit、push 或正式部署。此端點僅為保存成果的讀取介面，尚未建立 Custom Dataset 或訓練服務。

## M4 收尾：結果解讀與研究結論

以下為 M4 歷史紀錄；Custom Dataset 已於 M5／M6-B 加入檢查與驗證。

四個研究頁面已可瀏覽。Custom Dataset 暫緩，尚無上傳或訓練入口；目前網站定位為保存研究成果的互動展示。

- Overview 顯示完整 Holdout 的研究結論、特徵精簡與誤差取捨、PI 覆蓋率及平均寬度；數字直接使用既有快照。
- Feature Selection 保留六方法與真實逐 fold 評分，補上 CV 與固定 Holdout 的解讀連結。
- Model Evaluation 的「研究結論與使用限制」說明 R² 的均值基準、同時間估計與未來預報的差異，並將實測結果與可能原因分開。
- Prediction Explorer 在「目前視窗怎麼看？」顯示當前實際最大值當下的兩模型預測、低估／高估量、PI 判定、視窗 R² 解讀、覆蓋筆數、區間寬度與上下界漏失。隨時間或視窗大小更新；該最大值是當前資料的描述，不是預先定義的尖峰門檻。並列兩模型，不受圖表隱藏選項影響。
- R² 負值表示平方誤差高於使用這批實際平均值的事後比較基準；不是負準確率。常數／不足樣本保持無法定義。名目 95% PI 不保證每個短視窗都有 95% 覆蓋率，也不代表 95% 準確。
- 未新增模型實驗、重新訓練、改動預測、校準區間或改選特徵。可能的成因均標示為待驗證；無法由目前兩模型比較斷定單一原因。

人工驗收：

1. 開啟 `/`，確認「研究結論與使用限制」的 R²、特徵數、RMSE 差與 PI 數值對應原有表格；連結可前往完整解讀。
2. 開啟 `/feature-selection`，確認 LassoCV 的五個 outer fold 數量仍為 2、16、3、14、16，逐 fold 分數圖正常。
3. 開啟 `/model-evaluation#research-conclusion`，閱讀 R² 解釋、待驗證原因及使用限制，切換下方評估指標與 Training／Holdout。
4. 開啟 `/prediction-explorer`，選 12 小時、跳到 `2016-05-26 08:20`：目前視窗 72 筆，實際最大值 850 Wh；兩模型 R² 為負，PI 各涵蓋 58／72 筆。確認解讀中的模型預測對應實際最大值那一筆，而非各模型自己的最大值。
5. 切換時段與 12／24／48 小時，確認視窗解讀同步更新，完整 Holdout 參考值維持不變；以深／淺色與手機寬度檢查四個頁面。

## M5：Custom Dataset（CSV 檢查與設定）

以下為 M5 的瀏覽器檢查範圍；M6-B 已新增明確按鈕觸發的本機 Python 驗證。

開啟 `/custom-dataset`。本階段提供本機 CSV 選檔／拖放、資料品質摘要、最多五筆內容預覽、數值 Target／Features 指定、前處理與切分設定預覽、使用者確認後下載設定。里程碑標示 M5。

檔案透過 Web Worker 在瀏覽器記憶體內解析，沒有上傳端點、伺服器儲存、LocalStorage／IndexedDB 資料保存或自動訓練。選新檔會清空舊資料與確認；取消／清除／離開頁面會終止 Worker；處理超過 30 秒會拒絕並提示縮小檔案。更動任一設定後，先前確認立即失效。

### 格式與容量

僅支援逗號分隔 CSV、UTF-8／UTF-8 BOM、首列唯一非空欄名，支援 CRLF、引號跳脫、引號內逗號／換行。拒絕錯誤編碼、NUL、未閉合引號、欄數不符、欄名首尾空白等格式歧義。每檔上限 25 MiB、50,000 筆、200 欄、100 萬個資料儲存格、每格 65,536 個 UTF-16 code units、欄名 128 code units；所有上限都是目前強制條件，尚未宣稱為所有裝置的容量保證。預覽每格最多 200 code units；原始資料不改寫。

描述性檢查涵蓋完整上傳檔案，顯示缺失、不同值、常數／疑似 ID、有限數值範圍、中位數、1.5 IQR 候選、完全重複紀錄；IQR 不等於量測錯誤。型態推測為瀏覽器實作的有限範圍，不宣稱等同 pandas 的所有 dtype 推測或 Python 的完整 profiler。標準 NA tokens 與精確 SHA-256 一併記錄；純空白字串保留。前置零／長整數／超出 JavaScript 安全整數範圍的欄位保留文字，不允許默默四捨五入後當數值。

### 設定範圍與 Python 契約

- 使用者自行指定 Target，無自動 Target；拒絕常數、全缺失、非數值與無窮值。Target 缺失只允許 error 或在 split 前 drop，永不補值。
- Features 目前只支援有變化的有限數值欄位，排除 Target；類別／日期／自由文字仍在欄位清單中明確顯示不適用原因。尚未提供 OneHot、日期衍生、型態覆寫或一般資料的 Selector 執行。
- 數值缺失 median／mean／none、standard／none scaling、keep／clip IQR；只是待執行規則，沒有將全檔 profiling 統計作為 fit 參數。
- random 或 chronological split、20%／25%／30% holdout、2～10 folds、時間 CV gap。Chronological 需完整且唯一、無時區的 ISO 日期欄；不猜日月格式、Unix time 或混合時區。重複時間不自動分組；未排序資料會提示未來 Python 需排序。
- 切分筆數是依 Target 缺失政策的預計值，對齊 Python floor／ceil；CV feasibility 只作筆數預檢，正式每 fold 仍須檢查資料變異、日期可表示範圍與前處理合法性。
- 少於 13 個原始候選輸入會提示作業規範；不把這個門檻當成 Features 選取後 k。

`custom-dataset-config.json` 為可由既有 `PreprocessingConfig.from_dict` 讀取的 v1 前處理設定。`custom-dataset-review.json` 為獨立 envelope，包含 dataset digest／欄位描述統計／設定／確認狀態，不包含原始資料列或預覽；此 envelope 不直接傳入 `PreprocessingConfig.from_dict`。

下載 JSON 僅供後續人工／程式流程使用，沒有選方法、訓練、評分、PI 或模型保存；不允許把前四頁研究成績套在上傳資料上。後續實作一般訓練 Backend 時，仍需重新驗證 CSV／設定／分割、fold-local 前處理與 Selector 相容性。

M5 人工驗收：

1. 選擇 UTF-8 CSV，確認筆數、欄位、缺失、重複與預覽；沒有自動指定目標或特徵。
2. 選數值目標與特徵；選目標缺失 error／drop、數值 none 時確認錯誤提示。類別欄與 Target 不能誤選為數值 Feature。
3. 有日期時改 chronological、選時間欄，嘗試重複時間或過大的 gap，確認不能確認無效設定。
4. 有效設定經確認後下載兩份 JSON；更改比例、Features 或其他設定，確認下載再次停用。
5. 清除資料、上傳新 CSV、同名檔重選與錯誤 CSV；確認不保留前一檔的確認、統計或可下載設定。
6. 檢查手機／桌面深淺色與五頁導航；資料表可在區塊內橫向捲動，頁面不水平溢出。

M5 驗證：網站測試 71／71 通過，lint、typecheck、build 通過；實際下載設定經 Python v1 驗證器接受。瀏覽器驗證錯誤檔、重選、缺失、日期、取消、確認失效、下載、離頁清除與正式建置 Worker；六張代表性截圖已檢查。詳見 `verification/m5-result.json`。未重新訓練、commit 或 push。


## M6-B：網站串接本機 Python 驗證

以下記錄 M6-B 驗證入口；M7 另加獨立的明確訓練入口。M6-B 驗證本身仍不會訓練。

M5 的「僅瀏覽器解析、沒有上傳端點」是當時的歷史階段；目前仍先在瀏覽器檢查，只有使用者確認設定並按下「傳送至本機 Python 驗證」才送出 CSV。頁面與 HTTP API 不會啟動訓練、前處理 fit 或模型推論。

### 啟動與操作

沿用 `npm run dev`（只綁定 127.0.0.1）。預設使用 Repository 的 `.venv/bin/python` 與既有 Python 依賴；若虛擬環境在其他位置，啟動網站時設定可信的伺服器環境變數 `AEP_PYTHON`，必須為 Python 執行檔的絕對路徑。前端不能指定執行檔、模組、伺服器檔案路徑或輸出位置。正式 Next build 也支援此本機流程，尚未部署為多人服務。

選 CSV → 指定目標／Features／切分 → 確認設定 → 按「傳送至本機 Python 驗證」。結果顯示通過／未通過、錯誤階段、警告、Python 的 Training／Holdout 與逐 CV fold 筆數。這些是實際 split 索引的筆數，沒有模型評分。更改任何設定、換檔、清除或離頁會清除舊結果並取消尚在等待的請求；也可按「取消驗證」。

### API 與暫存生命週期

`POST /api/custom-dataset/validate` 使用 multipart，只有 `csv`（一份 CSV）、`config`（JSON 字串）、`review`（確認版本 JSON 字串）三個唯一欄位。只接受 loopback hostname 與相同 Origin；未提供／不符 Origin、cross-site 或其他 hostname 拒絕。沒有 GET 執行入口或對外 CORS。

伺服器在解析 multipart 前逐塊計算實際 body 大小，不依賴 Content-Length；上限為 25 MiB CSV＋128 KiB config＋1 MiB review＋64 KiB 表單餘量。接收後個別檢查檔案／設定上限、欄位與格式。每個服務 instance 同時僅執行一個驗證（其他回 429），不提供佇列。

輸入以固定檔名暫存到 OS 的隨機私有 `aep-validate-*` 目錄（檔案 0600），以無 shell 的固定參數呼叫 `python -B -m src.custom_dataset_validation`，結束或取消／逾時後等待程序終止再刪除該次暫存。沒有原檔名到伺服器路徑的映射、不存入 `public`／Repository／資料集目錄，也不在回應或日誌輸出 CSV 內容。這是暫存驗證，不提供上傳資料保存或模型服務；程序崩潰／系統強制停止等異常的復原管理仍屬正式服務的後續工作。

Python 上限 30 秒、stdout 1 MiB、stderr 64 KiB；異常／取消／逾時會終止子程序。環境找不到或回傳異常時提供受控訊息，不向網頁顯示 stderr、traceback 或私人路徑。Python 執行既有 M6-A 的數值型驗證，沒有放寬訓練範圍。

HTTP 200 表示訓練前驗證通過；422 是 CSV／設定／版本／切分的實際錯誤；400／403／413／415 是請求錯誤；429 忙碌；503 Python 環境無法使用；504 驗證逾時；499 取消。所有回應 private/no-store。型態檢查會核對 `training_started=false`、scope、版本、筆數與 fold 結構，並僅傳回允許的摘要欄位；成功時核對原始 CSV SHA-256 與 bytes。網頁另核對瀏覽器看到的檔案版本、行欄數與確認狀態，避免舊資料結果套到新設定。

### 人工驗收

1. 上傳 CSV 並設定；未確認前 Python 按鈕不可用。
2. 確認後執行，檢查成功訊息、Training／Holdout 與逐 fold 筆數；仍不顯示模型指標。
3. 使用前期目標為常數、後期才變動的資料：瀏覽器全檔檢查可通過，但 Python 應回報 Training／fold 目標常數。
4. 改設定或取消驗證，確認舊結果清除；無效資料／忙碌／Python 無法使用都有明確訊息。
5. 檢查深淺色與手機版，回查前四頁；正常執行結束後不留下該次上傳暫存。

M6-B 驗收通過：網站測試 83／83、Python 驗證器測試 10／10、lint、typecheck、production build 全數通過；開發版及正式建置版均完成真實上傳與 Python 驗證。錯誤提示、取消重試、設定失效、桌面深色／手機淺色與原有圖表已檢查。詳見 `verification/m6b-result.json`。未訓練、commit 或 push。


## M7：Custom Dataset 本機訓練與評估

以下記錄 M7 的固定 LR 階段；M8 已在相同入口加入明確指定的 F-test／MI 與 k，見後文。

流程：上傳 CSV → 選數值目標／輸入與前處理／切分 → 確認設定 → Python 驗證通過 → 勾選本次訓練確認 → 「開始本機訓練與評估」。選檔、確認設定、驗證、瀏覽或切換指標均不會自動訓練。

- 固定 Linear Regression 與 DummyRegressor(strategy="mean")；各 fold 及完整 Training 分別學習平均基準。沒有方法／k 搜尋、調參、六方法 Selector、Nested CV 或 PI。
- 重用現有 `iter_preprocessed_folds`／`prepare_fold`。每 fold 分別 fit 補值、IQR clip 規則、縮放與 LR；最後只用完整 Training refit，再於全部 Holdout 計算 RMSE／MAE／R²。Target 維持原單位，不截斷負預測、不刪測試極端值。
- CV 表列每 fold 實測成績，以及等權平均／標準差（ddof=0）；R² 常數或不足兩筆時為 null，顯示有定義的 fold 分母，不補 0 或 1。CV 與 Holdout 分表，未做模型擇優。
- Holdout 圖表預覽最多前 144 筆，表格每頁 24 筆；指標使用全部測試資料。時間切分是時間順序折線，隨機切分是實際／預測散佈圖，避免把隨機筆序當時間。預覽包含實際目標及預測；這是使用者主動訓練後才回傳的本次資料。
- 按鈕另需明確勾選確認；CSV／設定 digest 與 Python 驗證結果綁定，伺服器重新驗證而非相信前端。查看 Holdout 後再調設定有測試集重用限制，結果與確認前均明示。
- 訓練結果僅留頁面記憶體；改設定、換檔、清除、離頁或重新驗證會卸載訓練面板，取消舊請求，晚到結果不得套入新資料。無模型檔、持久 job、佇列、新樣本推論或公開部署。

`POST /api/custom-dataset/train` 仍是 localhost 同來源 multipart；沿用 M6-B 的 CSV／串流 body 上限、安全固定參數子程序、受控錯誤與暫存清除。只多接受 `experiment` JSON（上限 4 KiB）：schema_version=1、method=linear_regression、confirmed=true、dataset_sha256、config_sha256 五個固定 key。拒絕任意模型、程式、執行檔、輸出位置及額外參數。HTTP 200 為 completed，422 為 rejected（未 fit）或 failed（已開始但無完整結果）；失敗不回傳部分成績。驗證／訓練各 handler instance 分別只有一個執行名額，並非跨程序的全域排程器。

固定呼叫 `python -B -m src.custom_dataset_training`，CPU 數值函式庫限制一執行緒，子程序最多 120 秒、輸出最多 1 MiB、stderr 64 KiB；前端等待上限 130 秒。取消／逾時會終止程序並清除該次私有暫存，結束不寫入研究目錄或模型。沒有作業系統 RAM hard limit、持久進度、重啟復原或多使用者授權，不宣稱已具備正式服務架構。

Python 介面與驗收見 `docs/custom_dataset_training.md`；結果記錄於 `web/verification/m7-result.json`。此階段只以隔離合成資料執行訓練驗收，不覆訓既有能源研究模型。

M7 驗收通過：網站測試 88／88、原 Repository 完整 Python 測試 167／167、lint、typecheck、production build 全數通過。原本 localhost:3000 與獨立正式建置版均完成合成資料的上傳、驗證及明確確認訓練。深淺色、手機、指標切換、預覽分頁、設定失效、null R² 與取消／晚到回應均驗收；伺服器取消／逾時／暫存清理另由實際 Python API 整合測試覆蓋。詳見 `verification/m7-result.json`。未 commit 或 push。


## M8：自訂資料的 F-test／Mutual Information

Python 驗證通過後，第 06 區可選「不自動選欄／F-test／Mutual Information」，後兩者需指定整數 k（1～上方手動勾選的候選數）。方法與 k 的任何變更都會取消舊請求、清除結果與訓練確認；不會自動訓練或保留舊成績當作新設定的成績。

- 沿用現有 `SelectionSpec`／`make_selector` 的 sklearn SelectKBest；F-test 用 f_regression，MI 使用固定設定 random_state 與 n_jobs=1。每 fold 先只在自己的 Training fit 原有前處理，再只用該 Training 的轉換矩陣／y fit Selector，最後 fit 獨立 LR。Validation／Holdout 只 transform／predict。
- 同次實驗另 fit 全部候選欄位 LR 及 Training 平均值基準，使用相同資料列、分割與 fold-local 前處理，並列 CV／完整 Holdout。選欄後較少欄位不保證誤差更小，沒有依分數自動選贏家、掃 k 或改方法。
- 明確顯示完整 Training 的最終欄名與每 fold 的實際欄名；各 fold 可以不同。候選欄位只指使用者上方勾選的輸入，並非 CSV 所有欄。
- k=p 時保留全部欄位、共用全部欄位模型，不執行排名，回傳 ranking_performed=false 並提示。不自動選欄時 k=null；原 M7 五 key experiment 仍視為此模式，維持數值相容。
- 實際排名需每 CV Training 至少 4 筆；小樣本會在 fit 前拒絕，避免 MI 的近鄰估計無法執行。只支援一對一數值輸入；未新增 OneHot、日期衍生或刪列前處理。
- 結果卡片、表格、預測圖與逐筆預覽同時辨識選欄後／全部候選欄位／平均值。圖表範圍涵蓋全部顯示模型，不截斷負預測；維持上限 144 筆預覽與完整 Holdout 評分。

`experiment` 在原五 key 之外，可加 `selection: {"method":"f_test","k":6}`（或 mutual_information、none）；拒絕其他方法、額外選欄參數、非整數／超範圍 k。前端確認包含資料、前處理、方法與 k；Python 重驗資料及設定 digest、選欄設定合法性。變更方法不需重新做資料驗證，但必須重新確認訓練。HTTP 安全、120 秒運算上限、取消、單 instance 執行名額、受控錯誤與暫存清除維持 M7。

回傳的 selection 包含 method、requested_k、ranking_performed、final_features 與逐 fold 的 fold_features；all_features 提供獨立對照成績與預覽 baseline_prediction。runtime parser 核對實際欄名子集、k、原候選順序、fold 完整性及三模型指標／摘要，前端再核對方法與 k。維持 v1 作為既有固定 LR 回應的擴充，不提供任意模型執行。

看過 Holdout 後改方法／k 會重複使用測試集，介面於確認與結果明示，且同一面板內保留已查看提示；沒有跨頁持久追蹤。Training CV 是事先指定程序的評估，不是 Nested CV，也不是「最佳方法／最佳 k」的無偏成績。SFS、RFE、RFECV、LassoCV、自動調參、k 曲線、PI、模型保存及新資料推論仍未加入本頁。

人工驗收：上傳數值型 CSV → 勾選候選欄位／設定切分並確認 → Python 驗證 → 選 F-test 或 MI、指定 k → 勾選訓練確認並執行。檢查三模型比較、最終與逐 fold 欄名、三指標切換、預覽分頁；更改 k 應清除結果與確認。試 k=0、超過候選數、k=p、不自動選欄；檢查手機深淺色與取消。驗收結果見 `verification/m8-result.json`。

M8 驗收通過：網站 91／91、完整 Python 175／175、lint、typecheck、production build 全數通過。正式建置版驗證 F-test／MI、k 無效阻擋、k=p 不排名、none、實際最終／逐 fold 欄名、三組比較、時間與隨機圖、深淺色及手機；原 localhost:3000 也通過 MI 訓練。取消與晚到回應以延遲真實回應驗證，伺服器中止／逾時／清理由既有實際 Python API 測試涵蓋。驗收僅使用隔離合成資料；既有能源研究未重訓、未改寫，未 commit 或 push。


## M9：自訂輸入單筆預測

訓練成功後，結果中的「自訂輸入 · 單筆預測」使用完整 Training 的最終選欄與 LR。只顯示最終使用的原始 numeric 特徵；不需填未選中的欄位，也不需手動標準化。至少填一欄；mean／median 模式下空白沿用已學到的補值，不補值模式須填完整。非法文字不會當成空白補值。結果顯示原單位估計、本次補值與超出 Training 範圍的欄位；不重訓、不修改成績、不宣稱單筆 R²／RMSE／MAE 或 PI。

本機 Node 服務保留最終模型和選中欄位的 fitted 前處理（不含 CSV／Training 原列或 Holdout 答案），放在 OS 私有隨機目錄，模型檔 0600、上限 16 MiB。固定 30 分鐘到期，最多 3 個，以不可猜測的隨機代碼查找；可手動「清除此模型」。換檔、改設定或離頁會清除頁面結果，但伺服器模型依期限或容量清理。服務重啟後代碼失效；強制終止／系統崩潰留下的孤立暫存不屬本階段自動復原範圍。原始上傳暫存仍在當次程序結束後刪除。

`POST /api/custom-dataset/predict` 接受 JSON `{model_id, values}`，只允許本機同來源、已存在的模型代碼及選中特徵的有限 number／null；最多 64 KiB，單次 10 秒，不接受使用者模型檔、路徑或執行參數。`DELETE` 同一路徑接收 `{model_id}` 使模型失效。執行中的推論結束前不移除其模型檔。純 Python 訓練函式／CLI 預設仍不保存模型，內部 `--model-output` 只允許新建檔案。

人工驗收：

1. 開啟 `/custom-dataset`，上傳數值迴歸 CSV，選目標、候選欄位與前處理，確認設定並通過 Python 驗證。
2. 選 none／F-test／MI 及 k，明確確認後訓練。結果上方應出現單筆表單，欄名與完整 Training 的最終清單一致。
3. 填完整原始數值並預測；再清空部分欄位，確認補值說明與結果。不補值模式應阻擋不完整輸入。
4. 輸入文字、Infinity 或過大數字，確認預測按鈕停用。超出觀測範圍的有限輸入則顯示提醒，負預測不裁切。
5. 清空輸入會清除舊預測。清除此模型後無法再推論，但 CV／Holdout 成績保留。切換深淺色，並在手機寬度檢查表單。

本階段仍未加入類別輸入、批次新檔推論、永久模型保存、模型下載、SFS／RFE／自動調參或正式部署。

M9 驗收通過：網站測試 97／97、原 Git 專案完整 Python 測試 182／182、新增推論測試 7／7、lint、typecheck、production build 通過。隔離正式建置版與原 localhost:3000 均完成真實單筆推論；none／F-test／MI、mean／median／none、部分輸入、無效值、超出範圍、清空、模型清除、取消晚到回應、桌面／手機深淺色均驗收。晚到回應使用受控延遲的真實回應；伺服器逾時／中止、到期／容量與 lease 清理由實際 API 測試覆蓋。詳見 `verification/m9-result.json`。既有研究成果與未提交變更保留，未 commit 或 push。


## M9 補充：Holdout 圖表讀圖提示

Custom Dataset 的散佈圖於圖旁標示 X 軸實際值、Y 軸預測值，以及使用者指定的目標欄名與原始單位；附高估／低估、斜虛線與 Training 平均值基準的解讀。時間切分折線圖改為標示「依時間排序的 Holdout 筆次」，說明筆次不代表等距未來時間。文字標示放在 SVG 外，手機可換行，並提供圖表標題與輔助描述。

本次只調整畫面文字與樣式；不改評估數值、資料、訓練或推論流程。網站 97／97、lint、typecheck、production build 通過；隔離合成資料驗收隨機／時間切分與桌面淺色／手機深色，無頁面水平溢出。詳見 `verification/chart-guide-result.json`。


## M10：保存評估與單筆預測

完成訓練後，結果底部新增三個瀏覽器端下載，不重新訓練、不呼叫額外 API、不保存檔案到伺服器：

- `custom-evaluation-report.json`：完整 CV／Holdout 指標、逐 fold 與最終選欄、前處理設定、切分、資料／設定 SHA-256、警告及限制。只匯出明確欄位，不含原始 CSV、逐筆預覽、模型代碼／到期時間或模型。
- `custom-evaluation-metrics.csv`：Holdout、Training CV 摘要與逐 fold 分列。CV 是等權平均／標準差（ddof=0），與 Holdout 全筆計分分開。未定義 R² 保持空白，並標示 undefined／有效 fold 數，沒有補成 0。
- `custom-holdout-preview.csv`：只包含結果已保存的最多前 144 筆；每列標記預覽範圍、完整 Holdout 筆數、預覽筆數、原始資料列／時間、實際值與各組預測及來源雜湊。完整 Holdout 超過 144 筆時，這不是全量逐筆匯出。

單筆推論成功後可下載 `custom-single-prediction.json`，包含該次送出當時的原始輸入（空白為 null）、實際 Training 補值、預測、超出範圍欄位與限制。結果與輸入快照一併保存；改輸入、清空、重新預測或清除模型時舊下載隨結果移除，不能混用新輸入與舊答案。匯出沒有模型代碼，也不提供重新匯入或模型還原。

JSON 保留數字精度與 null；CSV 為 UTF-8 BOM、CRLF，正確處理引號／逗號／換行，方便試算表閱讀。文字欄位如以公式起始記號（含前置空白／控制字元）開頭，會加單引號作為純文字；負數型數值不加引號前綴。CSV 欄位 `experiment_selection_method`／`experiment_requested_k` 是整次實驗設定，`model` 區分 active LR、全候選 LR 與訓練平均值比較組。

人工驗收：完成一次實驗後下載三份檔案，比對畫面、逐 fold 與全部 Holdout 指標；確認預覽 CSV 筆數與按鈕相符。完成單筆預測後下載 JSON，檢查輸入、補值與答案；修改輸入應移除舊下載。切換深淺色及手機寬度，確認按鈕正常換行。

M10 驗收通過：網站 103／103（新增 6 項匯出測試）、lint、typecheck、production build 通過。已從隔離正式建置版實際下載四種檔案並核對真實計算結果、設定 SHA、完整 Holdout／144 筆預覽、未定義 R² 與單筆輸入／補值；下載不增加 API 呼叫。手機深色與桌面淺色、修改輸入／方法的舊下載失效、模型清除後仍可下載評估均驗收。原 localhost:3000 也成功下載。Python 與研究流程未修改，本次未重跑完整 Python 測試；M9 的 182／182 紀錄保留。詳見 `verification/m10-result.json`；未 commit 或 push。


## M11：新 CSV 批次預測

完成訓練後，在單筆預測下方的「新 CSV · 批次預測」選擇另一份 CSV。沿用同一個 30 分鐘暫存模型，不重新 fit、選欄、調參或更新原 CV／Holdout 成績。

檔案上限 2 MiB、500 筆、200 個原始欄位，且所選模型欄位 × 筆數不超過 10,000；畫面會計算此模型可接受的最大筆數。CSV 需包含全部最終使用欄名，大小寫相同，欄位順序可不同；多餘欄位（含目標）會列出並忽略。新檔不需要目標，也不以它重新評分。空白／常見 NA 記號使用原 Training 的 mean／median；none 模式須完整。每筆至少一個非缺失數值；任何不合法紀錄會使整批停止，不靜默跳列。

選檔後先在瀏覽器檢查，按預測才傳送所需欄位的 number／null 矩陣與模型代碼到同來源 `POST /api/custom-dataset/predict-batch`。不傳原始 CSV、多餘欄位內容或使用者檔名。伺服器依固定模型順序重新驗證欄名、行列限制與數值，最多 30 秒；使用現有模型 lease，取消／逾時等待 Python 結束才清除暫存並釋放 lease。只允許已登記的本機模型代碼，不接受模型檔、檔案路徑或執行參數。

Python 一次載入可信任 bundle，對每個選中 numeric branch 做已 fit 的 transform，再向量化 predict；不需要未選中的原候選欄位。回傳全部筆次的預測、實際補值欄位、超出 Training 範圍欄位與共用補值統計。沒有實際目標計分或 PI，負預測不裁切。

畫面只預覽前 10 筆；`custom-batch-predictions.csv` 下載本批全部結果（最高 500 筆，與 Holdout 144 筆預覽不同），包含模型使用的原始數值、空白標記、實際補值、來源 CSV SHA-256 與訓練資料／設定 SHA-256。忽略欄位與模型代碼不匯出。CSV 資料列不含標頭與略過的完全空白行，並非實體行號；JSON 補值欄保留精度，沿用 UTF-8 BOM、CRLF、引號與公式文字保護。

換檔、取消、清除或模型失效會取消／忽略晚到回應並清除舊結果；父層模型被手動清除時，批次表單一起失效。評估摘要下載仍保留。程序異常停止的暫存復原限制沿用 M9。

人工驗收：訓練後上傳沒有目標的新 CSV，檢查欄名與忽略清單，按批次預測並下載全部結果。再試少欄、非法文字、全空白模型輸入或超過筆數，應顯示原因並阻擋；none 模式遇空白應拒絕。換檔與清除後舊下載應消失。桌面／手機檢查深淺色及表格內捲動。

M11 驗收通過：網站 110／110（新增 7 項批次測試）、原 Git 專案完整 Python 187／187（新增 5 項批次測試）、lint、typecheck、production build 通過。正式建置版完成 155 筆推論，下載全 155 筆並逐筆核對原始輸入、預測、補值、範圍與來源 SHA；畫面只預覽 10 筆。缺欄、非法數字、全缺失、超量、none 補值限制、取消晚到回應與清除失效均驗收。原 localhost:3000 另通過無目標新檔 3 筆推論與下載，修整後選檔按鈕及深淺色／手機已檢視。晚到回應採受控延遲真實回應；伺服器中止、逾時及模型 lease 刪除另由實際 API 測試驗證。詳見 `verification/m11-result.json`；保留既有成果，未 commit 或 push。

## M12：Custom Dataset 的方法 × k 比較

Python 驗證通過後，新增「06 · 特徵選擇 × k 比較」，位於正式訓練／Holdout 測試之前。先選 F-test、Mutual Information、RFE（可複選）與連續整數 k 範圍，另外確認後才開始本機比較。

- 各方法／k 使用相同 Training CV 切分；每 fold 只在自己的 Training 學習前處理、選欄與獨立 LR。Random 使用固定 seed 的 KFold；chronological 使用原 gap 的 TimeSeriesSplit。
- 比較需要 `numerical_scaling=standard`；若原設定為 none，請回上方調整、確認並重新驗證，服務不會偷偷變更設定。IQR clip、缺失補值仍依原設定，只在 fold Training fit。
- 切換 RMSE／MAE／R²，曲線只連接實際計算點。表格保留平均 ± 標準差（ddof=0）與有定義的 fold 數；undefined R² 保持 null，曲線不跨空值補線。
- 用 k 選單查看各方法、各 fold 的實際特徵與三項分數。k=p 保留全部且不聲稱做過排名。沒有完整 Training 最終選欄、Holdout 分數、PI 或暫存模型。
- 下載 JSON 含設定、資料版本、完整比較分數與各 fold 特徵；CSV 含全部 k／方法／fold／metric，文字採公式防護，undefined 值留白。下載不含原始資料列、預測模型或模型代碼。
- 上限 5,000 筆、25 個候選欄、25 個 k，另依 RFE 反覆 fit 次數、矩陣大小與 CV Training 筆數估算成本，超過 1 億 work units 會在 fit 前拒絕。Python 最長 120 秒、stdout 8 MiB；可取消，不顯示部分成績。
- 這是探索性調參 CV；從多個點挑出的最佳分數可能偏樂觀。固定方法／k 後才評估一次獨立 Holdout。已查看過 Holdout 後再調整，會顯示測試集重用提醒。
- M12 當時正式訓練／推論僅支援 none／F-test／MI；M13 已接入 RFE（見下節）。RFECV／LassoCV 自動 k 尚未完成，未虛構它們的曲線。

人工驗收：上傳數值迴歸 CSV → 指定目標與候選特徵 → StandardScaler → 確認 → Python 驗證 → 比較方法與 k → 切換三項指標 → 查看 k 的各 fold 特徵 → 下載 CSV／JSON。改方法或範圍會清除舊結果並要求重新確認；取消後晚到回應不會覆蓋新狀態。請抽查桌面／手機及深淺色。完整紀錄見 `verification/m12-result.json`。

## M13：從比較套用設定，完成 RFE 訓練與新資料推論

比較結果新增每種方法的「套用到正式訓練」按鈕，使用目前查看的整數 k。這只帶入方法與 k，會清除舊評估及訓練確認，不會自動訓練，不會把某個 fold 的特徵清單固定套到其他 fold／最終模型。套用後再次明確確認才執行正式訓練與 Holdout。

正式訓練下拉新增 RFE，最終仍是獨立 Linear Regression。每個 CV fold 與完整 Training 分別使用原始 Training 資料，沿用 M12 `fit_selection` 的 RFE 選欄路徑；與比較曲線使用相同切分、前處理及排名。保留全部候選 LR 和 Training 平均值作為對照。k=p 不執行 RFE 排名。

RFE 要求明確選擇 StandardScaler（不會自動改原設定），限 5,000 筆、25 個候選欄與 1 億 work units。成本為 `(p-k+2 if k<p else 1) × p × (各 CV Training 筆數總和 + 完整 Training 筆數)`；120 秒逾時、取消及結果大小上限沿用既有訓練 API。其他既有方法的允許範圍不變。

RFE 最終模型沿用 30 分鐘私有暫存、單筆與 CSV 批次推論、刪除及下載機制；推論只 transform 原先選中的凍結前處理分支，不重新選欄或 fit。CV、Holdout 與下載的選欄方法均明確標為 RFE。RFECV／LassoCV 自動 k 及 SFS 仍待後續接入。

人工驗收：比較完成 → 查看 k → 套用 RFE → 確認下方方法／k 正確且訓練尚未開始 → 另勾確認並訓練 → 查看最終與各 fold 特徵、Holdout 三項指標 → 輸入選中特徵做單筆／CSV 批次預測 → 下載並清除暫存模型。若已看過 Holdout 再套用，重用提醒會保留。完整紀錄見 `verification/m13-result.json`。

## M13 介面補充：k 說明與圖例顯示切換

比較區新增 k 的數值例子、目前候選欄位數、預設比較範圍與「查看 → 套用 → 確認訓練」的說明。查看 k 不代表第 k 個欄位；例如 k=3 代表讓方法從候選欄選 3 欄，實際欄名需看各 fold 清單。1～10 是預設比較範圍，可以在支援的候選數內調整。

Holdout 散佈／時間折線圖下方的模型名稱現在是可用滑鼠或鍵盤操作的切換按鈕。點一下隱藏該組點／線，再點恢復；隱藏時保留按鈕，顯示「已隱藏」及刪除線，全部隱藏時會說明如何恢復。切換只影響繪圖與 tooltip，軸尺度保持一致，評估表格、預覽數值及下載不變，也不會發送新訓練請求。


## M14：SFS／RFECV／LassoCV 自訂 CSV

自訂 CSV 現在支援六種方法：F-test、MI、RFE、Forward SFS（指定 k），以及 RFECV、LassoCV（k 必須為 null，由 Training 自動選）。正式預測器一律為獨立 Linear Regression；Lasso 只用來選欄，不把 Lasso 係數直接當成網站最終預測器。

- SFS 每步用內部 3-fold CV 的平均 RMSE 決定加入哪一欄。RFECV 用內部 3-fold CV 平均 RMSE 選 k。內部隨機 KFold／TimeSeriesSplit 與 gap 跟隨使用者設定；每一層只在自己的 Training 學習前處理。
- LassoCV 用固定 alpha grid `[10000,1000,100,10,1,0.1,0.01,0.001,0.0001]` 的平均 MSE 選 alpha。各內層 split 有獨立前處理 blocks，LassoCV 的混合 blocks 內建 refit 係數完全不用；再於該外層 Training 的尺度 refit Lasso，以絕對係數 >1e-8 選欄，再 fit 獨立 LR。此 grid 不由 Holdout 或完整資料推算，選到端點會提醒。
- 比較區可勾選六方法（預設維持較快的 F-test／MI／RFE）。固定方法使用實際 k 曲線；自動方法另外顯示外層 CV 摘要、各 fold 的整數 k／特徵／alpha／成績。只有自動方法時，起訖 k 停用且送出空 k_values，不虛構自動方法的完整 k 曲線。
- 比較的「套用」只帶入方法；RFECV／LassoCV 不會沿用某 fold 的 k。正式訓練在完整 Training 再做內部 CV，獨立 Holdout 仍未參與選欄。訓練結果列出最終與各 fold 不同的 k 及 alpha。
- 自動比較中 Lasso 若某 fold 選 0 欄，該 fold 不 fit LR／不計分，分數為 null，摘要顯示有效 folds 並提醒不能直接排名不完整結果。正式訓練任一 fold／最終選出 0 欄會失敗，無部分 Holdout、無模型、不強制補欄或回退其他預測器。
- 新三方法都要求明確 StandardScaler。SFS 限 2,000 筆／15 候選欄；RFECV／LassoCV 限 5,000 筆／25 欄；另有成本 1 億 work units、120 秒、結果 8 MiB 上限。每個內層 Training 至少 4 筆，無法滿足時在 fit 前拒絕並提示減少 folds/gap 或增加資料。SFS k=p 與 RFECV p=1 不做不必要的搜尋，明確標示 bypass。
- JSON 保存內部選模設定、外層實際結果及最終選欄。評估 CSV 新增實際 final k／fold k／alpha／特徵；比較 CSV 區分固定與自動程序，摘要不填平均 k。單筆／批次推論沿用凍結的最終選欄與前處理，不重新選欄或 fit。

人工驗收：上傳數值 CSV → StandardScaler → 確認／驗證 → 勾選 SFS／RFECV／LassoCV 比較（SFS 請先縮小 k 範圍）→ 查看固定與自動區分 → 套用 → 再確認正式訓練 → 查看最終 k／逐 fold k／alpha → 單筆與批次預測、下載。留意小樣本／較大時間 gap、零特徵與已看過 Holdout 的限制。驗收紀錄見 `verification/m14-result.json`。


## 操作指南、報告草稿與雲端部署

2026-10-18：使用者已確認本機人工驗收可接受。新增 [完整操作指南](../docs/user_guide.md)、[成果報告草稿](../docs/final_report_draft.md) 與 [交付核對清單](../docs/submission_checklist.md)。報告數字沿用正式保存成果，M14 測試紀錄保留，這次文件整理未重跑模型／測試或修改網站功能。

使用者選定「Vercel 前端＋獨立 Python 後端」，讓訪客完整使用 CSV 訓練與預測。目前尚未接入遠端服務；不要直接把 web 上傳就宣稱完成。現有 API 僅允許 localhost、Python 依賴本機 .venv、模型是程序內暫存；sync-data 還需要 web 外的上游來源，/api/predictions 執行時需要快照。具體改動與發布驗收見 [部署計畫](../docs/vercel_deployment_plan.md)。


## M15：獨立後端封裝（前端接線前）

Repository 新增 [獨立運算服務](../backend/README.md)，使用無額外 npm 執行依賴的 Node 24 HTTP gateway 呼叫既有 Python workers。服務驗證與每 session ModelStore 在後端處理；原 web/lib 處理器與 Python 模型邏輯保持位元組不變。網站目前仍呼叫原本機 API，尚未啟用遠端模式或更改使用者介面。

新服務的 CSV 3 MiB、請求／回應 4 MiB、8 sessions／每 session 3 模型等限制僅適用於獨立服務；不要誤稱目前網頁的 25 MiB 已改為雲端限制。容器檔已準備但未實際 Docker build。Vercel 接線、session cookie、雲端文案、快照建置與 Preview／Production 驗收仍依部署計畫進行。完整紀錄見 verification/m15-backend-result.json。


## M16：遠端運算連線層（尚未啟用公開部署）

五個 Custom Dataset API 已接上 server-only 連線包裝。未設定下列變數且不在 Vercel 時，保持既有本機處理器；只設定部分變數或在 Vercel 缺少設定時回傳 503，不偷偷退回嘗試本機 Python。此階段沒有在使用者環境設定密鑰，也未公開發布。

需在前端伺服器設定四個變數（不得使用 NEXT_PUBLIC 前綴）：

| 變數 | 用途 |
| --- | --- |
| AEP_BACKEND_URL | 後端的 HTTPS 根網址，不能含帳密、子路徑、query 或 hash |
| AEP_BACKEND_TOKEN | 對應後端的 AEP_SERVICE_TOKEN，至少 32 個 base64url 字元 |
| AEP_SESSION_SECRET | 獨立隨機簽章密鑰，至少 32 字元，必須與服務密鑰不同 |
| AEP_PUBLIC_ORIGIN | 此環境前端的精確 HTTPS origin，例如 https://project.example；Preview 與 Production 分別設定，不使用 wildcard |

只允許精確 Origin／Host 與 same-origin 操作，不信任 forwarded host、瀏覽器 Authorization 或自報的 X-AEP-Session。代理建立 30 分鐘、HMAC 簽署、HttpOnly／SameSite=Strict cookie，HTTPS 加上 Secure；既有 cookie 驗證通過才沿用工作 ID。新驗證／訓練可建立工作階段，推論／刪除缺少有效 cookie 時回傳 410。此為匿名瀏覽工作階段隔離，尚非登入系統或完整防濫用配額。

代理只轉送固定的五種運算路徑與必要標頭；請求／回應上限 4 MiB，CSV 3 MiB。HTTPS 網站只連 HTTPS 後端；兩端都在 loopback 的本機整合測試可使用 HTTP。簽章綁定 origin，因此 Preview／Production cookie 不能混用。後端密鑰、診斷及上游 Set-Cookie 不傳給瀏覽器。

已測試本機代理到獨立後端的真實 RFECV 訓練、單筆／批次推論與刪除。網站預設仍是本機模式；雲端上傳提示、資料傳送說明、快照建置／Function 檔案包含與容器部署驗證尚待完成，**現在仍不能宣稱可直接發布正式版**。紀錄見 verification/m16-proxy-result.json。
