import { BATCH_LIMITS, parseBatchResult, validateBatchRows, type BatchRows } from "./custom-batch-types.ts";
import { access, mkdtemp, rm, writeFile } from "node:fs/promises";
import { constants } from "node:fs";
import { isAbsolute, join, resolve } from "node:path";
import { tmpdir } from "node:os";
import { modelStore, type ModelStore } from "./custom-model-store.ts";
import { parsePredictionResult } from "./custom-prediction-types.ts";
import { assertLocalRequest, headers, readBounded, RequestError, runPython } from "./custom-validation-service.ts";
type Options={batch?:boolean;store?:ModelStore;repoRoot?:string;pythonPath?:string;temporaryRoot?:string;timeoutMs?:number;bodyLimit?:number};
export function createPredictionHandler(options:Options={}) {
  let occupied=false;
  return async function handle(request:Request):Promise<Response> {
    let directory:string|undefined;let lease:Awaited<ReturnType<ModelStore["acquire"]>>=null;let ownsSlot=false;
    try {
      const allowed=options.batch?["POST"]:["POST","DELETE"];
      if(!allowed.includes(request.method))return Response.json({error:`只接受 ${allowed.join(" 或 ")}。`},{status:405,headers:{...headers,Allow:allowed.join(", ")}});
      assertLocalRequest(request);
      if(request.headers.get("content-type")?.split(";")[0].trim().toLowerCase()!=="application/json")throw new RequestError(415,"請使用網站的單筆預測表單。");
      const limit=options.bodyLimit??(options.batch?BATCH_LIMITS.bytes:64*1024);const length=request.headers.get("content-length");
      if(length!==null&&(!/^\d+$/.test(length)||Number(length)>limit))throw new RequestError(413,"預測輸入過大或長度不合法。");
      let payload:unknown;try{payload=JSON.parse(new TextDecoder("utf-8",{fatal:true}).decode(await readBounded(request,limit)));}catch(e){if(e instanceof RequestError)throw e;throw new RequestError(400,"預測輸入不是有效 JSON。");}
      if(!payload||typeof payload!=="object"||Array.isArray(payload))throw new RequestError(400,"預測輸入格式不符。");
      const p=payload as Record<string,unknown>;const expected=request.method==="POST"?(options.batch?["model_id","features","rows"]:["model_id","values"]):["model_id"];
      if(Object.keys(p).length!==expected.length||Object.keys(p).some(k=>!expected.includes(k))||typeof p.model_id!=="string"||!/^[a-f0-9]{64}$/.test(p.model_id))throw new RequestError(400,"模型代碼或欄位格式不符。");
      const store=options.store??modelStore();
      if(request.method==="DELETE"){await store.delete(p.model_id);return new Response(null,{status:204,headers});}
      if(occupied)throw new RequestError(429,"本機正在處理另一筆預測，請稍後再試。");
      occupied=true;ownsSlot=true;lease=await store.acquire(p.model_id);
      if(!lease)throw new RequestError(410,"模型已到期、清除或服務已重啟；請重新訓練後再預測。");
      let batchRows:BatchRows|undefined;
      const names=lease.info.features;
      if(options.batch){
        if(JSON.stringify(p.features)!==JSON.stringify(lease.info.features))throw new RequestError(422,"批次欄名與模型使用順序不符。");
        try{batchRows=validateBatchRows(p.rows,lease.info);}catch(e){throw new RequestError(422,e instanceof Error?e.message:"批次資料無效。");}
      }
      const values=p.values;
      if(!options.batch){
      if(!values||typeof values!=="object"||Array.isArray(values))throw new RequestError(400,"特徵輸入格式不符。");
      const entries=Object.entries(values);
      if(entries.some(([k,v])=>!names.includes(k)||(v!==null&&(typeof v!=="number"||!Number.isFinite(v)||Math.abs(v)>Number.MAX_SAFE_INTEGER)))||!entries.some(([,v])=>v!==null))throw new RequestError(422,"至少填一個有限數值；只接受模型使用的特徵。");
      if(lease.info.missing_numeric==="none"&&names.some(n=>!Object.hasOwn(values as object,n)||(values as Record<string,unknown>)[n]===null))throw new RequestError(422,"訓練時選擇不補值，請填完整模型使用的特徵。");
      }
      const root=options.repoRoot??resolve(process.cwd(),"..");const python=options.pythonPath??process.env.AEP_PYTHON??join(root,".venv/bin/python");
      if(!isAbsolute(python))throw new RequestError(503,"本機 Python 路徑無效。");
      try{await access(python,constants.X_OK);}catch{throw new RequestError(503,"找不到本機 Python 環境。");}
      directory=await mkdtemp(join(options.temporaryRoot??tmpdir(),"aep-predict-"));
      await writeFile(join(directory,"input.json"),JSON.stringify(options.batch?{features:lease.info.features,rows:batchRows}:values),{mode:0o600});
      const output=await runPython(python,root,directory,request.signal,options.timeoutMs??(options.batch?30_000:10_000),options.batch?"batch_predict":"predict",lease.path);
      if(output&&typeof output==="object"&&"status" in output&&output.status==="rejected")throw new RequestError(422,"無法預測，請檢查數值尺度與訓練時的補值設定。");
      if(options.batch)return Response.json(parseBatchResult(output,lease.info,batchRows!),{headers});
      const result=parsePredictionResult(output,lease.info);
      const missing=names.filter(n=>!Object.hasOwn(values as object,n)||(values as Record<string,unknown>)[n]===null);
      if(JSON.stringify(result.imputed.map(r=>r.feature))!==JSON.stringify(missing))throw new RequestError(500,"預測補值與輸入不一致。");
      return Response.json(result,{headers});
    }catch(error){const e=error instanceof RequestError?error:new RequestError(500,"本機預測暫時無法完成，請重新檢查或訓練。");return Response.json({error:e.message},{status:e.status,headers});}
    finally{try{if(directory)await rm(directory,{recursive:true,force:true});}finally{try{await lease?.release();}finally{if(ownsSlot)occupied=false;}}}
  };
}
