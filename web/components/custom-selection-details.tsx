import type { SelectionDetails } from "@/lib/custom-selection-policy";
export function CustomSelectionDetails({details}:{details:SelectionDetails}){
  if(details.search_bypassed)return <p className="cd-confirmation">已保留全部可用欄位，本次不需執行搜尋或內部 CV。</p>;
  return <p className="cd-confirmation">內部 {details.inner_folds.length}-fold CV · 依平均 {details.criterion?.toUpperCase()} 選擇{details.alpha!==null?` alpha＝${details.alpha.toLocaleString("en-US",{maximumFractionDigits:8})}，再以非零係數決定 k` : "特徵子集"}。{details.alpha!==null&&<>固定 alpha 候選：{details.alpha_grid.join("、")}；係數絕對值大於 10⁻⁸ 才保留。</>}最終預測器為另行訓練的 Linear Regression。</p>;
}
