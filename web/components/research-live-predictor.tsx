"use client";
import {useEffect,useRef,useState} from 'react';
import {loadFixedModel,fixedModelDescriptor} from '@/lib/fixed-models/catalog';
import {type FixedDatasetId,type FixedModelId} from '@/lib/fixed-models/core';
import {ResearchLiveController,initialResearchState,researchModelKey} from '@/lib/fixed-models/live-controller';
import {defaultExplanation,requiresObservedInput,researchSlider} from '@/lib/fixed-models/live-input';
const fmt=(n:number)=>n.toLocaleString('en-US',{maximumFractionDigits:6});
export function ResearchLivePredictor({datasetId}:{datasetId:FixedDatasetId}){return <ResearchDatasetLive key={datasetId} datasetId={datasetId}/>;}
function ResearchDatasetLive({datasetId}:{datasetId:FixedDatasetId}){
 const [modelId,setModelId]=useState<FixedModelId>('baseline');
 const binding=fixedModelDescriptor(datasetId,modelId).binding;
 return <section className="panel rlp-panel" data-research-live={datasetId} aria-labelledby="research-live-title">
  <div className="panel-header"><div><div className="section-kicker">LIVE PREDICTION · NEW INPUT</div><h2 id="research-live-title">Research Live Predictor</h2></div><div className="segmented-control" role="group" aria-label="即時預測模型">{(['baseline','mutual_information'] as const).map(id=><button key={id} className={id===modelId?'selected':''} aria-pressed={id===modelId} onClick={()=>setModelId(id)}>{id==='baseline'?'Baseline LR':'MI + LR'}</button>)}</div></div>
  <div className="rlp-body"><p className="rlp-intro">輸入你自己的新資料，使用已驗證的固定研究模型即時估計。全部推論在瀏覽器執行，不重新訓練，也不會上傳輸入。</p><ResearchModelLive key={researchModelKey(binding)} datasetId={datasetId} modelId={modelId}/></div>
 </section>;
}
function ResearchModelLive({datasetId,modelId}:{datasetId:FixedDatasetId;modelId:FixedModelId}){
 const [state,setState]=useState(initialResearchState);const controller=useRef<ResearchLiveController|null>(null);
 const binding=fixedModelDescriptor(datasetId,modelId).binding;
 useEffect(()=>{
  const own=new ResearchLiveController({binding:fixedModelDescriptor(datasetId,modelId).binding,load:signal=>loadFixedModel(datasetId,modelId,signal),onState:setState});
  controller.current=own;own.start();return()=>{own.dispose();if(controller.current===own)controller.current=null;};
 },[datasetId,modelId]);
 const {model,draft,result}=state;
 return <div data-research-live-key={researchModelKey(binding)} data-research-live-phase={state.phase} aria-busy={state.phase==='loading'}>
  {state.phase==='loading'&&<p role="status" className="rlp-status">載入並核對固定模型…</p>}
  {state.phase==='error'&&<div className="data-error" role="alert"><p>{state.error}</p><button className="rlp-button" onClick={()=>controller.current?.start()}>重新載入固定模型</button></div>}
  {model&&<>
   <div className="rlp-model-meta"><strong>{modelId==='baseline'?'Baseline LR':'MI + LR'} · {model.features.length} 個輸入</strong><span>Target：{model.target} · {model.unit} · v{model.modelVersion}</span><span>模型來源已核對 · 僅使用固定 Training 前處理</span></div>
   <div className="rlp-limitations">{model.limitations.map(note=><p key={note}>{note}</p>)}</div>
   <p className="rlp-note">預設值只來自已驗證的 Training metadata，僅供示範，請替換為本次實際資料。Training Min／Max 是觀測範圍；合法範圍外的輸入會被拒絕，合法外插只顯示警告。</p>
   <form noValidate onSubmit={e=>e.preventDefault()} aria-label="研究即時預測輸入">
    <div className="rlp-fields">{model.features.map((f,i)=>{
     const value=draft[f.name]??'',slider=researchSlider(f,value),isSelect=!!f.encoding||!!f.allowedValues;
     const legal=f.encoding?`合法選項：${Object.keys(f.encoding).join('／')}`:f.allowedValues?`合法編碼：${f.allowedValues.join('、')}`:`${f.kind==='integer'?'整數；':'連續數值；'}${f.domainMin===null?'無已核實下界':`合法下界 ${f.domainMin}`}；${f.domainMax===null?'無已核實上界':`合法上界 ${f.domainMax}`}`;
     return <div className="rlp-feature" key={f.name} data-research-feature={f.name}>
      <label htmlFor={`research-input-${i}`}><span>{f.name}{requiresObservedInput(f)?' · 本次已知值必填':''}</span><small>{f.unit}</small></label>
      {isSelect?<select id={`research-input-${i}`} data-research-input={f.name} value={value} aria-describedby={`research-help-${i}`} onChange={e=>controller.current?.edit(f.name,e.target.value)}><option value="">請選擇{f.imputer?'（空白使用 Training median）':''}</option>{(f.encoding?Object.keys(f.encoding):f.allowedValues!.map(String)).map(option=><option key={option} value={option}>{option}</option>)}</select>:<>
       <input id={`research-input-${i}`} data-research-input={f.name} type="number" step={f.kind==='integer'?1:'any'} inputMode={f.kind==='integer'?'numeric':'decimal'} value={value} autoComplete="off" placeholder={f.imputer&&!requiresObservedInput(f)?'空白使用 Training median':'請輸入本次數值'} aria-describedby={`research-help-${i}`} onChange={e=>controller.current?.edit(f.name,e.target.validity.badInput?'invalid-number':e.target.value)}/>
       <input type="range" data-research-slider={f.name} aria-label={`${f.name} 研究滑桿`} min={slider.min} max={slider.max} step={slider.step} value={slider.value} disabled={slider.disabled} onChange={e=>controller.current?.edit(f.name,e.target.value)} aria-describedby={`research-help-${i}`}/>
      </>}
      <div id={`research-help-${i}`} className="rlp-field-help"><p>{f.description}</p><p>{legal}</p><p>Training Min {fmt(f.trainingMin)} · Max {fmt(f.trainingMax)}</p><p>{defaultExplanation(f)}</p></div>
      {!isSelect&&slider.outside&&<p className="rlp-warning" role="status">外插警告：超出 Training 觀測範圍。滑桿只顯示邊界，數值與預測保留你的原始輸入。</p>}
     </div>;
    })}</div>
    <div className="rlp-actions"><button type="button" className="rlp-button" onClick={()=>controller.current?.reset()}>Reset to Defaults</button><span>停止輸入 350 毫秒後更新</span></div>
   </form>
   {state.phase==='waiting'&&<p role="status" className="rlp-status">等待輸入停止後更新點預測…</p>}
   {state.phase==='invalid'&&<p role="alert" className="rlp-input-error">{state.error}</p>}
   {result&&<div className="rlp-result" role="status" aria-live="polite" data-research-result data-research-prediction={result.prediction}>
    <span>Live Prediction · 本次輸入的單筆點估計</span><strong>{fmt(result.prediction)}</strong><p>{result.target} · {result.unit}</p><p>{modelId==='baseline'?'Baseline LR':'MI + LR'} · v{result.modelVersion}</p>
    {result.imputed.length>0&&<p>本次使用 Training median：{result.imputed.map(x=>`${x.feature} = ${fmt(x.value)}`).join('；')}</p>}
    {result.warnings.filter(w=>!model.limitations.includes(w)).map(w=><p className="rlp-warning" key={w}>{w}</p>)}
   </div>}
   <p className="rlp-note">此處只有新輸入的點預測，沒有實際答案可供計算誤差或評分。下方 Holdout Explorer 的區間只屬於已保存的研究樣本，不套用到本次輸入。</p>
  </>}
 </div>;
}
