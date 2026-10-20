"use client";
import {createContext,useCallback,useContext,useEffect,useRef,useState,type ReactNode} from "react";
import {BrowserComputeClient,BrowserComputeError} from "@/lib/browser-compute/client";
import {type ComputeMetadata,type Progress,type ComputeResult} from "@/lib/browser-compute/protocol";
import type {BatchRows} from "@/lib/custom-batch-types";
import {downloadArtifact} from "@/lib/browser-download";
import type {ServerComputeStatus} from "@/lib/custom-compute-availability";
type Mode="server"|"browser";
type ComputeContext={mode:Mode;request:(url:string,init:RequestInit)=>Promise<Response>;reset:()=>void;clearModels:()=>void;hasModel:(id:string)=>boolean;modelEpoch:number};
const Context=createContext<ComputeContext>({mode:"server",request:(url,init)=>fetch(url,init),reset:()=>{},clearModels:()=>{},hasModel:()=>true,modelEpoch:0});
export const useCustomCompute=()=>useContext(Context);
const stages:Record<string,string>={runtime:"初始化 Python",packages:"載入科學套件",modules:"核對並載入原始 Python 模組",ready:"Python 環境已就緒",validate:"正式 Python 驗證",train:"訓練與評估",compare:"Training CV 比較",predict:"單筆推論","predict-batch":"批次推論","delete-model":"清除模型"};
const methods:Record<string,string>={none:"全部候選欄位",f_test:"F-test",mutual_information:"Mutual Information",forward_sfs:"Forward SFS",rfe:"RFE",rfecv:"RFECV",lasso_cv:"LassoCV"};
const stageText=(p:Progress)=>p.method?`${methods[p.method]??p.method}：${p.phase==="completed"?"完成":"計算中"}`:stages[p.stage]??(p.stage.startsWith("package:")?`載入 ${p.stage.slice(8)}`:p.stage);
export function CustomComputeProvider({children,serverStatus}:{children:ReactNode;serverStatus:ServerComputeStatus}){
 const [mode,setMode]=useState<Mode>("server"),[progress,setProgress]=useState<Progress|null>(null),[busy,setBusy]=useState(false),[message,setMessage]=useState(""),[metadata,setMetadata]=useState<ComputeMetadata|null>(null),[modelEpoch,setModelEpoch]=useState(0),[completed,setCompleted]=useState<string[]>([]);
 const client=useRef<BrowserComputeClient|null>(null),generation=useRef(0);
 const reset=useCallback(()=>{generation.current++;client.current?.invalidate();setBusy(false);setProgress(null);setMetadata(null);setCompleted([]);},[]);
 const getClient=useCallback(()=>{if(!client.current)client.current=new BrowserComputeClient({onProgress:p=>{setProgress(p);if(p.method&&p.phase==="completed")setCompleted(v=>v.includes(p.method!)?v:[...v,p.method!]);},onReset:()=>setModelEpoch(n=>n+1)});return client.current;},[]);
 useEffect(()=>()=>{generation.current++;client.current?.dispose();client.current=null;},[]);
 const request=useCallback(async(url:string,init:RequestInit)=>{
  if(mode==="server")return serverStatus.available?fetch(url,init):Response.json({error:serverStatus.message},{status:503});
  const own=getClient();if(own.busy)throw new BrowserComputeError("BUSY","已有 Browser 工作，請先等待或取消。");
  const token=++generation.current;setBusy(true);setMessage("");setCompleted([]);
  try{
   const options={signal:init.signal??undefined};let result:ComputeResult<unknown>;
   if(["/api/custom-dataset/validate","/api/custom-dataset/train","/api/custom-dataset/compare"].includes(url)){
    if(!(init.body instanceof FormData))throw new Error("Browser 輸入格式錯誤。");const csv=init.body.get("csv");if(!(csv instanceof Blob))throw new Error("缺少 CSV。");
    const input={csv,config:JSON.parse(String(init.body.get("config"))),review:JSON.parse(String(init.body.get("review"))),...(init.body.has("experiment")?{experiment:JSON.parse(String(init.body.get("experiment")))}:{})};
    result=url.endsWith("/validate")?await own.validate(input,options):url.endsWith("/train")?await own.train(input,options):await own.compare(input,options);
   }else if(url==="/api/custom-dataset/predict"||url==="/api/custom-dataset/predict-batch"){
    const body=JSON.parse(String(init.body));result=init.method==="DELETE"?await own.deleteModel(body.model_id,options):url.endsWith("predict-batch")?await own.predictBatch(body.model_id,body.rows as BatchRows,options):await own.predict(body.model_id,body.values,options);
   }else throw new Error("Browser 不支援此操作，沒有轉送至伺服器。");
   if(token!==generation.current)throw new BrowserComputeError("STALE","模式或設定已變更。");
   setMetadata(result.metadata);setModelEpoch(n=>n+1);setMessage(`完成 · ${(result.metadata.elapsed_ms/1000).toFixed(2)} 秒`);
   const report=result.report as {status?:string};return Response.json(result.report,{status:["invalid","rejected","failed"].includes(report.status??"")?422:200});
  }catch(error){if(token===generation.current)setMessage(error instanceof Error?error.message:"Browser 運算失敗，請手動重試。");if(error instanceof BrowserComputeError&&error.code==="MODEL_EXPIRED")return Response.json({error:error.message},{status:410});throw error;}
  finally{if(token===generation.current)setBusy(false);}
 },[mode,getClient,serverStatus]);
 function switchMode(next:Mode){if(next===mode)return;generation.current++;client.current?.dispose();client.current=null;setBusy(false);setProgress(null);setMetadata(null);setCompleted([]);setMessage("已切換模式，CSV、確認、舊結果與 Browser 模型已清除。請重新選檔。");setMode(next);}
 return <Context.Provider value={{mode,request,reset,clearModels:()=>client.current?.releaseModels(),hasModel:id=>mode==="server"||!!client.current?.hasModel(id),modelEpoch}}><section className="panel cd-compute-mode" aria-labelledby="compute-mode-title"><div className="panel-header"><div><div className="section-kicker">CHOOSE WHERE TO COMPUTE</div><h2 id="compute-mode-title">運算模式</h2></div></div><div className="cd-body"><div className="segmented-control" role="group" aria-label="運算模式"><button id="compute-mode-server" aria-pressed={mode==="server"} className={mode==="server"?"selected":""} onClick={()=>switchMode("server")}>Server Mode · 預設</button><button id="compute-mode-browser" aria-pressed={mode==="browser"} className={mode==="browser"?"selected":""} onClick={()=>switchMode("browser")}>Browser Mode</button></div>
 {mode==="browser"?<><p>CSV、選欄、訓練與推論都在這個瀏覽器執行，不會自動上傳伺服器。首次需從官方 CDN 下載約 32 MB Python／科學套件；訓練資料上限 2,000 列、15 個數值候選特徵、3 MiB。訓練與多 k 比較分別檢查工作量；比較超限只阻止該次比較，仍可執行符合限制的訓練。</p><p className="cd-confirmation">每方法上限 60 秒、每項工作總上限 90 秒；不同資料、裝置與網路可能超過 60 秒。RFECV／LassoCV 保留自動選 k；不自動抽樣或降低 CV。取消、逾時、換檔、改資料設定、離頁或切換模式會清除記憶體模型。</p></>:<p role={serverStatus.available?"status":"alert"} id="server-compute-status">{serverStatus.message} Browser Mode 失敗時不會自動切回或上傳。</p>}
 {mode==="browser"&&<div aria-live="polite" role="status">{busy&&<><p>{progress?stageText(progress):"準備 Browser 工作…"}</p><button className="cd-button" id="browser-compute-cancel" onClick={()=>client.current?.cancel()}>取消 Browser 工作</button></>}{completed.length>0&&<p>已完成：{completed.map(m=>methods[m]??m).join("、")}</p>}{message&&<p>{message}</p>}{metadata&&<details><summary>Browser 執行來源與版本</summary><p>Pyodide {metadata.runtime} · sklearn {metadata.versions.sklearn} · Python {metadata.versions.python}</p><p>原始碼 manifest：<code>{metadata.source_sha256}</code></p><button className="cd-button" onClick={()=>downloadArtifact({filename:"browser-compute-metadata.json",mime:"application/json;charset=utf-8",content:JSON.stringify(metadata,null,2)+"\n"})}>下載 Browser 執行資訊</button></details>}</div>}</div></section><div key={mode}>{children}</div></Context.Provider>;
}
