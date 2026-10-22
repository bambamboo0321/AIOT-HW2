import { parseValidationReport, type ValidationReport } from "../custom-validation-result.ts";
import { parseTrainingReport, type TrainingReport } from "../custom-training-result.ts";
import { parseComparisonReport, type ComparisonReport } from "../custom-comparison-result.ts";
import { parsePredictionResult, type PredictionModel, type PredictionResult } from "../custom-prediction-types.ts";
import { parseBatchResult, type BatchRows, type BatchResult } from "../custom-batch-types.ts";
import { COMPARISON_LIMITS } from "../custom-comparison-work.ts";
import { BROWSER_LIMITS, BrowserComputeError, type Operation, type DatasetInput, type ComputeResult, type Progress, type WorkerReply, type WorkerRequest } from "./protocol.ts";
export { BROWSER_LIMITS, BrowserComputeError } from "./protocol.ts";
export type WorkerPort = { postMessage(message:WorkerRequest, transfer:Transferable[]):void; terminate():void; onmessage:((event:MessageEvent<WorkerReply>)=>void)|null; onerror:((event:ErrorEvent)=>void)|null; onmessageerror:((event:MessageEvent)=>void)|null };
type Options = { workerFactory?:()=>WorkerPort; onProgress?:(progress:Progress)=>void; onReset?:()=>void; initializationTimeoutMs?:number; jobTimeoutMs?:number; methodTimeoutMs?:number; comparisonTimeoutMs?:number };
type Pending = { id:string; operation:Operation; resolve:(v:ComputeResult<unknown>)=>void; reject:(e:Error)=>void; timer:ReturnType<typeof setTimeout>; methodTimer?:ReturnType<typeof setTimeout>; cleanup:()=>void };
type JobOptions = { signal?:AbortSignal };
function stable(value:unknown):string {
  if(value===null||typeof value!=="object")return JSON.stringify(value);
  if(Array.isArray(value))return '['+value.map(stable).join(',')+']';
  return '{'+Object.entries(value).sort(([a],[b])=>a<b?-1:a>b?1:0).map(([k,v])=>JSON.stringify(k)+':'+stable(v)).join(',')+'}';
}
async function sha(data:ArrayBuffer):Promise<string>{return [...new Uint8Array(await crypto.subtle.digest("SHA-256",data))].map(n=>n.toString(16).padStart(2,"0")).join("");}
/** One browser session, one serial Worker. No fetch to any M16 API and no automatic server fallback. */
export class BrowserComputeClient {
  private worker:WorkerPort|null=null;
  private pending:Pending|null=null;
  private closed=false;
  private comparison:BrowserComputeClient|null=null;
  private epoch=0;
  private ready:ComputeResult<unknown>|null=null;
  private binding:{key:string;dataset:string;config:string}|null=null;
  private models=new Map<string,PredictionModel>();
  private initialization:Promise<ComputeResult<unknown>>|null=null;
  private readonly options:Options;
  constructor(options:Options={}){this.options=options;}
  get busy(){return !!this.pending||!!this.comparison;}
  get initialized(){return !!this.ready;}
  hasModel(id:string){const m=this.models.get(id);return !!m&&m.expires_at>Date.now();}
  releaseModels(){if(this.models.size){const binding=this.binding;this.reset(new BrowserComputeError("STALE","舊模型已清除。"));this.binding=binding;}}
  private ensureWorker():WorkerPort {
    if(this.closed)throw new BrowserComputeError("DISPOSED","Browser 運算已關閉。");
    if(this.worker)return this.worker;
    const worker=this.options.workerFactory?.()??new Worker("/browser-compute/worker.mjs",{type:"module"});
    this.worker=worker;
    worker.onmessage=event=>{
      if(this.worker!==worker)return;
      const message=event.data,p=this.pending;
      if(!p||!message||message.jobId!==p.id)return;
      if(message.kind==="progress"){
        if(message.progress.phase==="started"){
          clearTimeout(p.methodTimer);const ms=p.operation==="compare"?(this.options.comparisonTimeoutMs??COMPARISON_LIMITS.timeoutMs):(this.options.methodTimeoutMs??60_000);p.methodTimer=setTimeout(()=>this.reset(new BrowserComputeError("TIMEOUT",p.operation==="compare"?"比較超過 180 秒，已終止比較 Worker。":"此方法超過 60 秒上限，Worker 與模型已清除。")),ms);
        }else if(message.progress.phase==="completed")clearTimeout(p.methodTimer);
        this.options.onProgress?.(message.progress);return;
      }
      if(message.kind==="error"){
        this.reset(new BrowserComputeError(message.code,message.message));return;
      }
      if(message.kind!=="result"){this.reset(new BrowserComputeError("PROTOCOL","Browser 回應格式不符。"));return;}
      clearTimeout(p.timer);clearTimeout(p.methodTimer);p.cleanup();this.pending=null;
      this.ready={report:{ready:true},metadata:message.metadata};
      p.resolve({report:message.report,metadata:message.metadata});
    };
    worker.onerror=()=>{if(this.worker===worker)this.reset(new BrowserComputeError("WORKER_FAILED","Browser Worker 載入或執行失敗；請重試。"));};
    worker.onmessageerror=()=>{if(this.worker===worker)this.reset(new BrowserComputeError("PROTOCOL","Browser 結果無法傳回。"));};
    return worker;
  }
  private reset(error:BrowserComputeError){
    this.epoch++;this.comparison?.dispose();this.comparison=null;const p=this.pending;this.pending=null;
    this.worker?.terminate();this.worker=null;this.ready=null;this.binding=null;this.models.clear();this.initialization=null;
    if(p){clearTimeout(p.timer);clearTimeout(p.methodTimer);p.cleanup();p.reject(error);}
    this.options.onReset?.();
  }
  cancel(){if(this.comparison){this.comparison.cancel();return;}this.reset(new BrowserComputeError("ABORTED","已取消 Browser 運算；記憶體模型已清除。"));}
  invalidate(){this.reset(new BrowserComputeError("STALE","資料、設定或模式已變更；請重新驗證與確認。"));}
  dispose(){this.closed=true;this.reset(new BrowserComputeError("DISPOSED","Browser 運算已關閉，模型與資料已清除。"));}
  private job(operation:Operation,prepare:()=>Promise<{payload:Record<string,unknown>;csv?:ArrayBuffer}>,options:JobOptions={}):Promise<ComputeResult<unknown>> {
    if(this.closed)return Promise.reject(new BrowserComputeError("DISPOSED","Browser 運算已關閉。"));
    if(this.busy)return Promise.reject(new BrowserComputeError("BUSY","已有 Browser 工作執行中，請先取消或等待完成。"));
    if(options.signal?.aborted)return Promise.reject(new BrowserComputeError("ABORTED","工作已取消。"));
    return new Promise((resolve,reject)=>{
      const id=crypto.randomUUID();
      const abort=()=>{if(this.pending?.id===id)this.cancel();};
      const timer=setTimeout(()=>{if(this.pending?.id===id)this.reset(new BrowserComputeError("TIMEOUT","Browser 等待逾時，Worker 與模型已清除；可手動重試。"));},operation==="initialize"?(this.options.initializationTimeoutMs??45_000):operation==="compare"?(this.options.comparisonTimeoutMs??COMPARISON_LIMITS.timeoutMs):(this.options.jobTimeoutMs??90_000));
      this.pending={id,operation,resolve,reject,timer,cleanup:()=>options.signal?.removeEventListener("abort",abort)};
      options.signal?.addEventListener("abort",abort,{once:true});
      void (async()=>{
        try {
          const prepared=await prepare();if(this.pending?.id!==id)return;
          const worker=this.ensureWorker();
          worker.postMessage({jobId:id,operation,...prepared},prepared.csv?[prepared.csv]:[]);
        }catch(error){if(this.pending?.id===id){const p=this.pending;this.pending=null;clearTimeout(p.timer);p.cleanup();p.reject(error instanceof Error?error:new Error("Browser 輸入錯誤。"));}}
      })();
    });
  }
  initialize(options:JobOptions={}):Promise<ComputeResult<unknown>> {
    if(this.closed)return Promise.reject(new BrowserComputeError("DISPOSED","Browser 運算已關閉。"));
    if(this.ready)return Promise.resolve(this.ready);
    if(this.initialization)return this.initialization;
    const promise=this.job("initialize",async()=>({payload:{}}),options).then(result=>{this.ready=result;return result;});
    this.initialization=promise;
    void promise.finally(()=>{if(this.initialization===promise)this.initialization=null;}).catch(()=>{});
    return promise;
  }
  private dataset(operation:"validate"|"train"|"compare",input:DatasetInput,options:JobOptions):Promise<ComputeResult<unknown>> {
    // Snapshot caller objects before the first await; Blob bytes are immutable.
    const configText=JSON.stringify(input.config),reviewText=JSON.stringify(input.review),experimentText=input.experiment===undefined?undefined:JSON.stringify(input.experiment);
    return this.job(operation,async()=>{
      if(!input.csv.size||input.csv.size>BROWSER_LIMITS.bytes)throw new BrowserComputeError("LIMIT","Browser 訓練 CSV 限非空檔案、3 MiB；不會自動上傳或抽樣。");
      const config=JSON.parse(configText),review=JSON.parse(reviewText);
      if(!Array.isArray(config.feature_columns)||config.feature_columns.length>BROWSER_LIMITS.features)throw new BrowserComputeError("LIMIT","Browser 訓練最多 15 個數值候選特徵。");
      const csv=await input.csv.arrayBuffer(),digest=await sha(csv);
      const key=digest+stable(config)+stable(review);
      if(operation!=="validate"&&(!this.binding||this.binding.key!==key))throw new BrowserComputeError("STALE","CSV 或設定與已驗證版本不同；請重新確認並驗證。");
      const experiment=experimentText===undefined?undefined:JSON.parse(experimentText);
      if(operation!=="validate"&&(!experiment?.confirmed||experiment.dataset_sha256!==digest||experiment.config_sha256!==this.binding?.config))throw new BrowserComputeError("STALE","訓練／比較確認與已驗證的資料設定版本不符。");
      if(operation==="validate"){this.binding=null;this.models.clear();}
      return {csv,payload:{config,review,...(experiment?{experiment}:{}),bindingKey:key}};
    },options);
  }
  async validate(input:DatasetInput,options:JobOptions={}):Promise<ComputeResult<ValidationReport>> {
    if(this.busy)throw new BrowserComputeError("BUSY","已有 Browser 工作，請先等待或取消。");
    if(this.binding||this.models.size)this.invalidate();
    const snapshot={...input,config:structuredClone(input.config),review:structuredClone(input.review)};
    const epoch=this.epoch;
    const result=await this.dataset("validate",snapshot,options),report=parseValidationReport(result.report);
    if(epoch!==this.epoch)throw new BrowserComputeError("STALE","驗證已過期。");
    if(report.status==="valid"){
      const review=snapshot.review as {dataset?:{sha256?:string}};
      if(report.dataset.sha256!==review.dataset?.sha256||!report.review_checked)throw new BrowserComputeError("PROTOCOL","Browser 驗證來源不一致。");
      this.binding={key:report.dataset.sha256+stable(snapshot.config)+stable(snapshot.review),dataset:report.dataset.sha256,config:report.config_sha256};
    }
    return {report,metadata:result.metadata};
  }
  async train(input:DatasetInput,options:JobOptions={}):Promise<ComputeResult<TrainingReport>> {
    const result=await this.dataset("train",input,options),report=parseTrainingReport(result.report);
    if(report.status==="completed"){
      if(report.validation.dataset.sha256!==this.binding?.dataset||report.validation.config_sha256!==this.binding.config)throw new BrowserComputeError("PROTOCOL","Browser 訓練來源不一致。");
      if(report.prediction_model){this.models.set(report.prediction_model.id,report.prediction_model);while(this.models.size>3)this.models.delete(this.models.keys().next().value!);}
    }
    return {report,metadata:result.metadata};
  }
  async compare(input:DatasetInput,options:JobOptions={}):Promise<ComputeResult<ComparisonReport>> {
    if(this.closed)throw new BrowserComputeError("DISPOSED","Browser 運算已關閉。");
    if(this.busy)throw new BrowserComputeError("BUSY","已有 Browser 工作，請先等待或取消。");
    // Comparison has no saved model. A disposable Worker protects the primary
    // Worker/MEMFS models from compare timeout, cancellation and memory failures.
    const epoch=this.epoch,isolated=new BrowserComputeClient({...this.options,onReset:undefined});
    isolated.binding=this.binding?{...this.binding}:null;this.comparison=isolated;
    try{
      const result=await isolated.dataset("compare",input,options),report=parseComparisonReport(result.report);
      if(epoch!==this.epoch)throw new BrowserComputeError("STALE","資料或模式已變更。");
      if(report.status==="compared"&&(report.validation.dataset.sha256!==this.binding?.dataset||report.validation.config_sha256!==this.binding.config))throw new BrowserComputeError("PROTOCOL","Browser 比較來源不一致。");
      return {report,metadata:result.metadata};
    }catch(error){
      if(epoch!==this.epoch)throw new BrowserComputeError("STALE","資料或模式已變更。");
      const code=error instanceof BrowserComputeError?error.code:"EXECUTION_FAILED";
      const message=code==="TIMEOUT"?"比較達到 180 秒上限":code==="ABORTED"?"比較已取消":code==="WORKER_FAILED"||code==="RESOURCE_EXHAUSTED"?"比較 Worker 執行失敗，可能為記憶體／資源不足":"比較未完成，請檢查輸入與載入狀態";
      throw new BrowserComputeError(code,message+"；比較 Worker 已停止，未刪除既有訓練模型。可調整比較設定後手動重試。");
    }finally{isolated.dispose();if(this.comparison===isolated)this.comparison=null;}
  }
  private model(id:string):PredictionModel {
    const model=this.models.get(id);if(!model||model.expires_at<=Date.now())throw new BrowserComputeError("MODEL_EXPIRED","Browser 模型已清除或到期，請重新訓練。");return model;
  }
  async predict(id:string,values:Record<string,number|null>,options:JobOptions={}):Promise<ComputeResult<PredictionResult>> {
    const model=this.model(id),snapshot=JSON.parse(JSON.stringify(values));
    const result=await this.job("predict",async()=>({payload:{model_id:id,values:snapshot}}),options);
    this.checkModelError(result.report);return {report:parsePredictionResult(result.report,model),metadata:result.metadata};
  }
  async predictBatch(id:string,rows:BatchRows,options:JobOptions={}):Promise<ComputeResult<BatchResult>> {
    const model=this.model(id),snapshot=JSON.parse(JSON.stringify(rows)) as BatchRows;
    const result=await this.job("predict-batch",async()=>({payload:{model_id:id,features:model.features,rows:snapshot}}),options);
    this.checkModelError(result.report);return {report:parseBatchResult(result.report,model,snapshot),metadata:result.metadata};
  }
  async deleteModel(id:string,options:JobOptions={}):Promise<ComputeResult<unknown>> {
    this.model(id);const result=await this.job("delete-model",async()=>({payload:{model_id:id}}),options);
    this.models.delete(id);return result;
  }
  private checkModelError(report:unknown){if(report&&typeof report==="object"&&"http_status" in report&&report.http_status===410)throw new BrowserComputeError("MODEL_EXPIRED","Browser 模型已清除或到期，請重新訓練。");}
}
