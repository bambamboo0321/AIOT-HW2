import { Icon } from "@/components/icon";
import { CustomDatasetWizard } from "@/components/custom-dataset-wizard";
export const metadata = { title: "Custom Dataset" };
export default function CustomDatasetPage() {
  return <div className="page-container cd-page"><section className="research-header"><div><div className="eyebrow"><span className="eyebrow-line"/>YOUR DATA, YOUR QUESTION</div><h1>Custom dataset<span className="heading-dot">.</span></h1><p className="page-description">檢查 CSV、指定預測目標，驗證設定、比較特徵選擇與 k，訓練評估後用單筆輸入或新 CSV 做預測。</p></div><div className="source-badge"><span className="status-dot"/>Local training & evaluation</div></section><div className="research-context"><span className="context-icon"><Icon name="dataset" size={18}/></span><p><strong>從檔案到可審閱設定</strong><span className="context-separator">/</span>先在瀏覽器檢查；確認後送至本機 Python 驗證，先比較六種特徵選擇方法的 Training CV 結果，套用所選方法與 k 到正式訓練，確認後執行線性迴歸與基準比較，並用最終模型預測新輸入。</p></div><CustomDatasetWizard/></div>;
}
