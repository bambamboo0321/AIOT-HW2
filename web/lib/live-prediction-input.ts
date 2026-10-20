import {parseFeatureInput,type TrainingInputRanges} from './custom-prediction-types.ts';
export function initialPredictionInputs(ranges?:TrainingInputRanges):Record<string,string>{
 return Object.fromEntries((ranges?.features??[]).filter(r=>r.min!==null&&r.max!==null).map(r=>[r.feature,String(r.min!/2+r.max!/2)]));
}
export function sliderPresentation(text:string,min:number|null,max:number|null){
 const available=min!==null&&max!==null;let n:number|null=null;
 try{n=parseFeatureInput(text);}catch{/* Invalid number input never becomes a prediction value. */}
 const outside=available&&n!==null&&(n<min!||n>max!);
 return {disabled:!available||min===max,min:min??0,max:max??1,value:available?Math.max(min!,Math.min(max!,n??(min!/2+max!/2))):0,step:available&&max!>min!?(max!-min!)/1000:1,outside};
}
