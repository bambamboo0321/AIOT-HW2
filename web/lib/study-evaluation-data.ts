import { parseStudyOverview, type StudyDatasetId, type StudyOverview, type StudyModel, type Metrics } from "./study-overview-data.ts";
export type EvaluationModelId = "baseline" | "mutual_information";
type Unavailable = { status:"unavailable"; reason:string };
export type StudyPredictions = { status:"ready"; actual:number[]; predicted:number[] } | Unavailable;
export type StudyPI = { status:"ready"; lower:number[]; upper:number[];
  stats:{nominal:0.95;coverage:number;averageWidth:number;medianWidth:number;widthSD:number;covered:number;below:number;above:number;total:number};
  trainingRows:number;holdoutResidualsUsed:false;boundsClipped:false;method:"ols-pointwise-svd" } | Unavailable;
export type StudyEvaluationModel = { id:EvaluationModelId; predictions:StudyPredictions; interval:StudyPI; summary:StudyModel };
export type StudyEvaluationData = {schemaVersion:1;artifactType:"study-model-evaluation";datasetId:StudyDatasetId;overviewSha256:string;sources:StudyOverview["sources"];models:[StudyEvaluationModel,StudyEvaluationModel];study:Pick<StudyOverview,"dataset"|"protocol">};
export type StudyEvaluationSnapshot = {status:"ready";data:StudyEvaluationData} | {status:"unavailable";issues:string[]};
const obj=(v:unknown):v is Record<string,unknown>=>!!v&&typeof v==="object"&&!Array.isArray(v);
function required(v:unknown,field:string):asserts v{if(!v)throw new Error(`模型評估快照缺失或不相容：${field}`);}
const finite=(v:unknown):v is number=>typeof v==="number"&&Number.isFinite(v);
const text=(v:unknown):v is string=>typeof v==="string"&&v.trim().length>0;
const numbers=(v:unknown):v is number[]=>Array.isArray(v)&&v.every(finite);
const count=(v:unknown):v is number=>finite(v)&&Number.isSafeInteger(v)&&v>=0;
export const evaluationClose=(a:number,b:number)=>Math.abs(a-b)<=1e-10*Math.max(1,Math.abs(a),Math.abs(b));
function close(a:number,b:unknown,field:string){required(finite(b)&&evaluationClose(a,b),field);}
export function savedEvaluationMetrics(actual:number[],predicted:number[]):Metrics{
  const n=actual.length,mean=actual.reduce((a,b)=>a+b,0)/n;
  const error=actual.reduce((s,a,i)=>s+(a-predicted[i])**2,0);
  const total=actual.reduce((s,a)=>s+(a-mean)**2,0);
  return {mae:actual.reduce((s,a,i)=>s+Math.abs(a-predicted[i]),0)/n,rmse:Math.sqrt(error/n),r2:1-error/total};
}
export function quantile(values:number[],q:number){const sorted=[...values].sort((a,b)=>a-b),p=(sorted.length-1)*q,i=Math.floor(p);return sorted[i]+(sorted[Math.ceil(p)]-sorted[i])*(p-i);}
export function parseStudyEvaluation(value:unknown,id:StudyDatasetId,overviewHash:string,overview:unknown):StudyEvaluationSnapshot{
  try{
    const base=parseStudyOverview(overview,id);required(base.status==="ready","linked Overview unavailable");
    required(obj(value)&&value.schemaVersion===1&&value.artifactType==="study-model-evaluation"&&value.datasetId===id&&/^[a-f0-9]{64}$/.test(overviewHash)&&value.overviewSha256===overviewHash,"version/dataset/Overview hash");
    required(Array.isArray(value.sources)&&value.sources.length>=base.data.sources.length&&JSON.stringify(value.sources.slice(0,base.data.sources.length))===JSON.stringify(base.data.sources),"source linkage");
    const prefix=id==="automobile-price"?"experiments/multi-dataset-phase2/automobile/":"experiments/student-performance-phase2/";
    required(value.sources.slice(base.data.sources.length).every(s=>obj(s)&&text(s.path)&&s.path.startsWith(prefix)&&!s.path.includes("..")&&typeof s.sha256==="string"&&/^[a-f0-9]{64}$/.test(s.sha256)),"prediction source hashes");
    required(new Set(value.sources.map(s=>(s as {path:string}).path)).size===value.sources.length,"unique sources");
    required(Array.isArray(value.models)&&value.models.length===2,"two models");
    let actualReference:number[]|undefined;
    const models:StudyEvaluationModel[]=[];
    for(const [i,m] of value.models.entries()){
      const summary=base.data.models[i];required(obj(m)&&m.id===summary.id,"model identity");
      const p=m.predictions,pi=m.interval;required(obj(p)&&obj(pi),"predictions/PI state");
      if(p.status==="unavailable"){
        required(text(p.reason)&&pi.status==="unavailable"&&text(pi.reason),"missing predictions/PI reason");
        models.push({...(m as Omit<StudyEvaluationModel,"summary">),summary});continue;
      }
      required(summary.status==="ready"&&p.status==="ready"&&numbers(p.actual)&&numbers(p.predicted)&&p.actual.length===base.data.protocol.holdoutRows&&p.predicted.length===p.actual.length,"complete saved predictions");
      required(value.sources.some(s=>obj(s)&&s.path===`${prefix}${m.id}-holdout.csv`),"saved prediction CSV provenance");
      const actual=p.actual,predicted=p.predicted;
      required(actual.every(v=>v>=0&&(id!=="student-performance"||v<=20)),"target values");
      if(actualReference)required(JSON.stringify(actual)===JSON.stringify(actualReference),"two models share Holdout");actualReference=actual;
      const recomputed=savedEvaluationMetrics(actual,predicted);
      for(const k of ["rmse","mae","r2"] as const)close(recomputed[k],summary.holdout[k],`Holdout ${k}`);
      if(pi.status==="unavailable")required(text(pi.reason),"missing PI reason");
      else{
        required(summary.interval.status==="ready"&&pi.status==="ready"&&numbers(pi.lower)&&numbers(pi.upper)&&pi.lower.length===actual.length&&pi.upper.length===actual.length&&pi.trainingRows===base.data.protocol.trainRows&&pi.holdoutResidualsUsed===false&&pi.boundsClipped===false&&pi.method==="ols-pointwise-svd"&&obj(pi.stats),"saved PI definition");
        const lower=pi.lower,upper=pi.upper,stats=pi.stats;
        const widths:number[]=[];let below=0,above=0;
        for(let j=0;j<actual.length;j++){
          required(lower[j]<=predicted[j]&&predicted[j]<=upper[j],"PI bounds");close((lower[j]+upper[j])/2,predicted[j],"PI center");widths.push(upper[j]-lower[j]);if(actual[j]<lower[j])below++;else if(actual[j]>upper[j])above++;
        }
        const n=actual.length,covered=n-below-above,average=widths.reduce((a,b)=>a+b,0)/n;
        required(stats.nominal===.95&&count(stats.covered)&&stats.covered===covered&&stats.below===below&&stats.above===above&&stats.total===n,"PI counts");
        close(covered/n,stats.coverage,"PI coverage");close(average,stats.averageWidth,"PI width");close(quantile(widths,.5),stats.medianWidth,"PI median width");close(Math.sqrt(widths.reduce((s,w)=>s+(w-average)**2,0)/n),stats.widthSD,"PI width SD");
        close(stats.coverage as number,summary.interval.coverage,"Overview PI coverage");close(stats.averageWidth as number,summary.interval.averageWidth,"Overview PI width");required(covered===summary.interval.covered,"Overview PI count");
      }
      models.push({...(m as Omit<StudyEvaluationModel,"summary">),summary});
    }
    return {status:"ready",data:{...(value as Omit<StudyEvaluationData,"study"|"models">),models:models as StudyEvaluationData["models"],study:{dataset:base.data.dataset,protocol:base.data.protocol}}};
  }catch(error){return {status:"unavailable",issues:[error instanceof Error?error.message:"評估快照無法讀取。"]};}
}
export function predictionPoints(model:StudyEvaluationModel){
  if(model.predictions.status!=="ready")return [];
  const saved=model.predictions;
  return saved.actual.map((actual,i)=>({position:i+1,actual,predicted:saved.predicted[i],residual:actual-saved.predicted[i]}));
}
export function intervalPoints(model:StudyEvaluationModel){
  if(model.predictions.status!=="ready"||model.interval.status!=="ready")return [];
  const pi=model.interval;
  return predictionPoints(model).map((p,i)=>({...p,bounds:[pi.lower[i],pi.upper[i]] as [number,number]})).sort((a,b)=>a.predicted-b.predicted||a.position-b.position).map((p,i)=>({...p,rank:i+1}));
}
export function residualBins(models:StudyEvaluationModel[],binCount=20){
  const values=models.flatMap(m=>predictionPoints(m).map(p=>p.residual));if(!values.length)return [];
  const limit=Math.max(...values.map(Math.abs))||.5,step=2*limit/binCount;
  const bins=Array.from({length:binCount},(_,i)=>({lower:-limit+i*step,upper:-limit+(i+1)*step,baseline:0,mutual_information:0}));
  for(const m of models)for(const p of predictionPoints(m))bins[Math.min(binCount-1,Math.max(0,Math.floor((p.residual+limit)/step)))][m.id]++;
  return bins;
}
