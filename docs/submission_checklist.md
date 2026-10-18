# 作業交付核對清單

日期：2026-10-18。這份清單依現有 README、研究文件與使用者驗收整理；Repository 未找到老師原始完整作業單，因此不是逐條核對正式評分規準後的合格保證。收到最新作業單後再修訂門檻及命名。

## 已有證據與待補內容

| 項目 | 目前狀態 | 依據／下一步 |
| --- | --- | --- |
| Kaggle 迴歸資料與來源 | 已有 | [README](../README.md)、[來源](../references/external_sources.md) |
| 原始輸入數量 | 28 個；符合 README 記錄的至少 13 個 | 這是原始輸入，不是選取後 k；舊 Phase 2 文件保留歷史不同門檻，最終仍以老師最新原文為準 |
| Multiple Linear Regression | 已有 | 正式預測器為 sklearn LinearRegression |
| 六種特徵選擇 | 已有 | [Phase 3B](../reports/phase3/phase3b_report.md)；自訂 CSV 已接入六方法 |
| CRISP-DM、EDA、前處理 | 已有 | [成果草稿](final_report_draft.md) 的證據連結 |
| CV／Holdout／PI 解讀 | 已有 | [Phase 3C](../reports/phase3/phase3c_report.md)；限制需保留 |
| 網站本機操作 | 使用者已確認接受 | 五頁、深淺色、自訂 CSV、推論與下載 |
| 自動測試證據 | 已保存 | M14：網站 127、Python 211 通過；不是本次重新執行 |
| 操作指南 | 草稿完成 | [user_guide.md](user_guide.md) |
| 成果報告 | Markdown 草稿完成 | [final_report_draft.md](final_report_draft.md)，尚非正式 PDF |
| NotebookLM 摘要及引用 | 待正式保存使用者的實際產物 | [準備紀錄](../references/notebooklm_research_plan.md)；不得以 GPT 文字代替 |
| GPT 輔助說明及對話 PDF | 待整理 | 依老師原文確認範圍；移除不屬本作業的私人內容 |
| 完整學號、姓名與正式主程式名 | 待提供及修改 | 暫用 hw2_main.py，README 記錄命名為 7114056XXX_hw2.py；不可猜測學號 |
| 報告／ZIP 命名與繳交管道 | 待核對正式作業單 | 目前不自行假設正式壓縮檔名或截止時間 |
| Vercel 前端＋獨立後端 | 使用者已選定架構；尚未部署 | [部署計畫](vercel_deployment_plan.md)；此為使用者新增需求，未查證為教師必繳項 |
| GitHub | README 記錄 Optional | Git import 路線才需要推送；Vercel CLI 可不依賴 Git。未 commit／push |

## 正式交付前順序

1. 取得老師原始作業要求與更新、完整個人資訊，逐條修訂本表。
2. 保存實際 NotebookLM 摘要、來源及 GPT 協助／對話紀錄。
3. 完成雲端接線、Preview 驗證，再發布正式網址；報告補入實際部署狀態。
4. 檢閱成果草稿，加入研究圖表、封面與引用，輸出並檢查報告 PDF。
5. 依最新命名規範整理主程式、必要 src、requirements、資料來源／必要資料、既有成果、README 與附件。
6. 對最終交付副本檢查連結、檔名、缺檔及啟動方式，再製作 ZIP；正式檔名確認前不改名入口。

ZIP 不包含 `.venv/`、`node_modules/`、`.next/`、快取、登入資訊或 Secret。研究成果與必要資料是否隨檔繳交以老師原文及授權為準；原 Repository 的既有未提交成果一律保留，不因打包而刪除。

## 驗收與版本界線

使用者的本機操作驗收已接受；本次不要求再次手動重驗。新增雲端架構後仍需針對公開網址做獨立連線驗證，不能拿 localhost 的成功當作雲端通過。文件草稿不等於已繳交、已發布或已完成全部教師要求。
