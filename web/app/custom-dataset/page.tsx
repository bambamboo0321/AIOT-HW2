import { access } from "node:fs/promises";
import { constants } from "node:fs";
import { resolve } from "node:path";
import { serverComputeStatus } from "@/lib/custom-compute-availability";
import { Icon } from "@/components/icon";
import { CustomDatasetWizard } from "@/components/custom-dataset-wizard";
import { CustomComputeProvider } from "@/components/custom-compute-provider";
export const metadata = { title: "Custom Dataset" };
// Resolve capability per request, including deployment changes after build.
export const dynamic = "force-dynamic";
export default async function CustomDatasetPage() {
  let executable = false;
  if (!process.env.VERCEL) {
    try { await access(process.env.AEP_PYTHON ?? resolve(process.cwd(), "../.venv/bin/python"), constants.X_OK); executable = true; } catch { /* No local interpreter; Browser Mode remains independent. */ }
  }
  const serverStatus = serverComputeStatus(process.env, executable);
  return <div className="page-container cd-page"><section className="research-header"><div><div className="eyebrow"><span className="eyebrow-line"/>YOUR DATA, YOUR QUESTION</div><h1>Custom dataset<span className="heading-dot">.</span></h1><p className="page-description">檢查 CSV、指定預測目標，驗證設定、比較特徵選擇與 k，訓練評估後用單筆輸入或新 CSV 做預測。</p></div><div className="source-badge"><span className="status-dot"/>Server / Browser compute</div></section><div className="research-context"><span className="context-icon"><Icon name="dataset" size={18}/></span><p><strong>從檔案到可審閱設定</strong><span className="context-separator">/</span>先在瀏覽器檢查；確認運算模式後由 Python 驗證，先比較六種特徵選擇方法的 Training CV 結果，套用所選方法與 k 到正式訓練，確認後執行線性迴歸與基準比較，並用最終模型預測新輸入。</p></div><CustomComputeProvider serverStatus={serverStatus}><CustomDatasetWizard/></CustomComputeProvider></div>;
}
