export type PredictionValues=Record<string,number|null>;
type Options<T>={request:(values:PredictionValues,signal:AbortSignal)=>Promise<T>;onStart:()=>void;onResult:(result:T,values:PredictionValues)=>void;onError:(error:unknown)=>void;onIdle:()=>void;delayMs?:number;timeoutMs?:number};
// Serial requests preserve Browser MEMFS models during edits. Only explicit cancel,
// expiry or disposal aborts an active inference; stale results are discarded.
export class LivePredictionController<T>{
 private revision=0;private disposed=false;private timer:ReturnType<typeof setTimeout>|null=null;
 private queued:{revision:number;values:PredictionValues}|null=null;
 private active:{abort:AbortController;timer:ReturnType<typeof setTimeout>}|null=null;
 private options:Options<T>;
 constructor(options:Options<T>){this.options=options;}
 schedule(values:PredictionValues|null,immediate=false){
  this.revision++;if(this.timer)clearTimeout(this.timer);this.timer=null;this.queued=null;
  if(!values||this.disposed)return;
  const job={revision:this.revision,values:structuredClone(values)};
  if(immediate){this.queued=job;this.flush();}else this.timer=setTimeout(()=>{this.timer=null;if(!this.disposed&&job.revision===this.revision){this.queued=job;this.flush();}},this.options.delayMs??500);
 }
 private flush(){
  if(this.disposed||this.active||!this.queued)return;
  const job=this.queued;this.queued=null;const abort=new AbortController();
  const active={abort,timer:setTimeout(()=>abort.abort(),this.options.timeoutMs??15000)};this.active=active;this.options.onStart();
  void(async()=>{
   try{const result=await this.options.request(job.values,abort.signal);if(!abort.signal.aborted&&!this.disposed&&job.revision===this.revision)this.options.onResult(result,job.values);}
   catch(error){if(!this.disposed&&job.revision===this.revision)this.options.onError(error);}
   finally{clearTimeout(active.timer);if(this.active===active)this.active=null;if(!this.disposed){this.options.onIdle();this.flush();}}
  })();
 }
 cancel(){this.schedule(null);this.active?.abort.abort();}
 dispose(){this.disposed=true;this.cancel();}
}
