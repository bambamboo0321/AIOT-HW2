import type {FixedFeature,FixedModel} from './core.ts';
export type ResearchDraft=Record<string,string>;
export function requiresObservedInput(f:FixedFeature){return f.description.startsWith('已知的');}
function legal(f:FixedFeature,n:number){return Number.isFinite(n)&&Math.abs(n)<=Number.MAX_SAFE_INTEGER&&(f.kind!=='integer'||Number.isSafeInteger(n))&&(!f.allowedValues||f.allowedValues.includes(n))&&(f.domainMin===null||n>=f.domainMin)&&(f.domainMax===null||n<=f.domainMax);}
/** Defaults only use frozen Training metadata. Known-observation fields and categories without defaults remain blank. */
export function researchDefaults(model:FixedModel):ResearchDraft{
 return Object.fromEntries(model.features.map(f=>{
  if(f.encoding||requiresObservedInput(f))return [f.name,''];
  const n=f.imputer?.value??(f.trainingMin/2+f.trainingMax/2);
  return [f.name,legal(f,n)?String(n):''];
 }));
}
export function defaultExplanation(f:FixedFeature){
 if(requiresObservedInput(f))return '請輸入本次已知值，不以示範中位數代替。';
 if(f.encoding)return '模型未保存類別預設值，請自行選擇。';
 return f.imputer?'預設：保存的 Training median。':'預設：Training 觀測範圍中點（示範值）。';
}
export function researchDraftValues(model:FixedModel,draft:ResearchDraft):Record<string,number|string|null>{
 if(Object.keys(draft).some(n=>!model.featureOrder.includes(n)))throw new Error('輸入欄位與目前模型不一致。');
 return Object.fromEntries(model.features.map(f=>{
  const text=(draft[f.name]??'').trim();
  if(!text){if(f.imputer&&!requiresObservedInput(f))return [f.name,null];throw new Error(`${f.name}：請輸入本次數值或選擇合法選項。`);}
  if(f.encoding){if(!Object.hasOwn(f.encoding,text))throw new Error(`${f.name}：未知類別。`);return [f.name,text];}
  if(!/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i.test(text))throw new Error(`${f.name}：請輸入有效的有限數字。`);
  const n=Number(text);if(!legal(f,n))throw new Error(`${f.name}：數值不符合合法範圍、整數或序數規則。`);
  return [f.name,n];
 }));
}
/** Clamp the visual thumb only. The stored draft and inference value are never rewritten. */
export function researchSlider(f:FixedFeature,text:string){
 const n=text.trim()?Number(text):NaN,valid=Number.isFinite(n);
 return {min:f.trainingMin,max:f.trainingMax,step:f.kind==='integer'?1:'any',value:valid?Math.max(f.trainingMin,Math.min(f.trainingMax,n)):f.trainingMin,disabled:f.trainingMin===f.trainingMax,outside:valid&&(n<f.trainingMin||n>f.trainingMax)};
}
