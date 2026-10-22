import {predictFixedModel,type FixedModel,type FixedPrediction,type ModelBinding} from './core.ts';
import {researchDefaults,researchDraftValues,type ResearchDraft} from './live-input.ts';
export type ResearchLiveState={phase:'loading'|'waiting'|'ready'|'invalid'|'error';model:FixedModel|null;draft:ResearchDraft;result:FixedPrediction|null;error:string|null};
export const initialResearchState=():ResearchLiveState=>({phase:'loading',model:null,draft:{},result:null,error:null});
type Options={binding:ModelBinding;load:(signal:AbortSignal)=>Promise<FixedModel>;onState:(state:ResearchLiveState)=>void;delayMs?:number;loadTimeoutMs?:number;predict?:(model:FixedModel,values:Record<string,number|string|null>)=>FixedPrediction|Promise<FixedPrediction>};
export const researchModelKey=(b:ModelBinding)=>`${b.datasetId}:${b.modelId}:${b.modelVersion}`;
/** Research-owned lifecycle. It never touches Custom Compute, Session or a Worker. */
export class ResearchLiveController{
 private state=initialResearchState();private revision=0;private disposed=false;
 private timer:ReturnType<typeof setTimeout>|null=null;private loadTimer:ReturnType<typeof setTimeout>|null=null;private abort:AbortController|null=null;
 private options:Options;
 constructor(options:Options){this.options=options;}
 private emit(){if(!this.disposed)this.options.onState({...this.state,draft:{...this.state.draft}});}
 private stopTimer(){if(this.timer)clearTimeout(this.timer);this.timer=null;}
 start(){
  if(this.disposed)return;const token=++this.revision;this.stopTimer();this.abort?.abort();if(this.loadTimer)clearTimeout(this.loadTimer);
  const abort=new AbortController();this.abort=abort;this.state=initialResearchState();this.emit();
  this.loadTimer=setTimeout(()=>{if(this.disposed||token!==this.revision)return;this.revision++;abort.abort();this.state={...initialResearchState(),phase:'error',error:'固定模型載入逾時，請手動重試。'};this.emit();},this.options.loadTimeoutMs??15000);
  void Promise.resolve().then(()=>this.options.load(abort.signal)).then(model=>{
   if(this.disposed||token!==this.revision)return;
   if(researchModelKey(model)!==researchModelKey(this.options.binding))throw new Error('載入模型與目前資料集／模型版本不一致。');
   if(this.loadTimer)clearTimeout(this.loadTimer);this.loadTimer=null;this.state={phase:'waiting',model,draft:researchDefaults(model),result:null,error:null};this.schedule();
  }).catch(error=>{if(this.disposed||token!==this.revision)return;if(this.loadTimer)clearTimeout(this.loadTimer);this.loadTimer=null;this.state={...initialResearchState(),phase:'error',error:error instanceof Error?error.message:'固定模型無法載入，請重試。'};this.emit();});
 }
 edit(name:string,text:string){if(!this.state.model||!this.state.model.featureOrder.includes(name)||this.disposed)return;this.state={...this.state,draft:{...this.state.draft,[name]:text}};this.schedule();}
 reset(){if(!this.state.model||this.disposed)return;this.state={...this.state,draft:researchDefaults(this.state.model)};this.schedule();}
 private schedule(){
  const token=++this.revision;this.stopTimer();const model=this.state.model;if(!model)return;
  this.state={...this.state,result:null,error:null};let values:Record<string,number|string|null>;
  try{values=researchDraftValues(model,this.state.draft);}catch(error){this.state={...this.state,phase:'invalid',error:error instanceof Error?error.message:'輸入無效。'};this.emit();return;}
  this.state={...this.state,phase:'waiting'};this.emit();
  this.timer=setTimeout(()=>{
   this.timer=null;if(this.disposed||token!==this.revision)return;
   void Promise.resolve().then(()=>(this.options.predict??predictFixedModel)(model,values)).then(result=>{
    if(this.disposed||token!==this.revision)return;
    if(researchModelKey(result)!==researchModelKey(this.options.binding))throw new Error('預測結果與目前模型不一致。');
    this.state={...this.state,phase:'ready',result,error:null};this.emit();
   }).catch(error=>{if(this.disposed||token!==this.revision)return;this.state={...this.state,phase:'invalid',result:null,error:error instanceof Error?error.message:'無法完成推論。'};this.emit();});
  },this.options.delayMs??350);
 }
 dispose(){this.disposed=true;this.revision++;this.stopTimer();if(this.loadTimer)clearTimeout(this.loadTimer);this.loadTimer=null;this.abort?.abort();this.abort=null;this.state=initialResearchState();}
}
