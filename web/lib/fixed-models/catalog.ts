import manifest from "../../data/fixed-models/v1/manifest.json" with {type:"json"};
import {FIXED_DATASETS,FIXED_MODELS,verifyFixedModelJson,type FixedDatasetId,type FixedModelId,type ModelBinding} from "./core.ts";
/** Trusted bundled lock. Assets are public same-origin JSON, never model uploads or an API. */
export function fixedModelDescriptor(datasetId:FixedDatasetId,modelId:FixedModelId){
 if(!FIXED_DATASETS.includes(datasetId)||!FIXED_MODELS.includes(modelId))throw new Error("未知研究資料集或固定模型。");
 const key=`${datasetId}--${modelId}` as keyof typeof manifest.models,descriptor=manifest.models[key];
 if(manifest.modelVersion!=="1.0.0"||descriptor.path!==`/research-models/v1/${key}.json`)throw new Error("固定模型 manifest 不相容。");
 return {...descriptor,binding:{datasetId,modelId,modelVersion:"1.0.0"} satisfies ModelBinding};
}
export async function loadFixedModel(datasetId:FixedDatasetId,modelId:FixedModelId,signal?:AbortSignal){
 const {path,sha256,binding}=fixedModelDescriptor(datasetId,modelId);
 const response=await fetch(path,{signal,cache:"no-cache"});
 if(!response.ok)throw new Error("固定模型暫不可用，未使用其他模型替代。");
 return verifyFixedModelJson(await response.text(),binding,sha256);
}
