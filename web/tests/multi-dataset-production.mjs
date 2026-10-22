// Production HTTP acceptance. Uses only served HTML/API, never reads experiment datasets.
import assert from 'node:assert/strict';
const origin=process.argv[2]??'http://127.0.0.1:3018';
const paths=['/','/feature-selection','/model-evaluation','/prediction-explorer'];
const records=[];
async function page(path,query=''){
 const response=await fetch(origin+path+query);
 assert.equal(response.status,200);
 const html=await response.text();
 const main=html.match(/<main\b[^>]*id="main-content"[^>]*>([\s\S]*?)<\/main>/)?.[1];
 assert.ok(main,'main content missing');
 return {html,main};
}
for(const path of paths){
 for(const id of ['automobile-price','student-performance']){
  const {html,main}=await page(path,'?dataset='+id);
  assert.ok(main.includes(`data-dataset-id="${id}"`));
  assert.ok(!main.includes('15,788')&&!main.includes('3,947')&&!main.includes('APPLIANCES ENERGY PREDICTION LAB'));
  if(path==='/'){
   assert.ok(main.includes('data-overview-state="ready"'));
   assert.ok(!main.includes('研究結果尚未接入'));
   assert.ok(main.includes('Holdout performance')&&main.includes('95% Prediction Interval'));
   assert.ok(!main.includes('Chronological split')&&!main.includes('耗能')&&!main.includes('Wh'));
   for(const value of id==='automobile-price'?['3,657.66','5,215.62','0.7777','0.7621','201','14','82.93%','fuel-type','gas=0','diesel=1','幣別尚未核實']:['0.7348','1.1664','0.8605','0.8630','649','15','99.23%','G1、G2','非學期開始前']) assert.ok(main.includes(value),`Missing ${id} ${value}`);
  } else if(path==='/feature-selection'){
   assert.ok(main.includes('data-feature-state="ready"'));
   assert.ok(!main.includes('研究結果尚未接入')&&!main.includes('Wh')&&!main.includes('時間 CV 曲線'));
   for(const method of ['F-test','Mutual Information','Forward SFS','RFE','LassoCV','RFECV'])assert.ok(main.includes(method));
   for(const value of ['六方法比較','完整 k 曲線 unavailable','Training CV','固定 k=7','auto','非時間 CV'])assert.ok(main.includes(value));
   assert.ok(!main.includes('id="measured-k"'));
   if(id==='student-performance')for(const value of ['1.2871','0.2687','最終 k=5','最終 k=2','Fedu','studytime','Dalc','Walc','G1、G2','非學期開始前'])assert.ok(main.includes(value),`Missing Student ${value}`);
   else for(const value of ['3,013.1005','350.5562','最終 k=9','最終 k=3','201','14','fuel-type'])assert.ok(main.includes(value),`Missing Automobile ${value}`);
  } else if(path==='/model-evaluation'){
   const content=main.replace(/<[^>]*>/g,'');
   assert.ok(main.includes('data-evaluation-state="ready"'));
   assert.ok(!main.includes('研究結果尚未接入'));
   assert.ok(!content.includes(' Wh')&&!content.includes('chronological')&&!content.includes('Timestamp'));
   for(const value of ['Actual vs Predicted','Residual Plot','Residual Distribution','95% Prediction Interval','Training CV','Holdout','非時間'])assert.ok(content.includes(value),`Missing ${value}`);
   const n=id==='automobile-price'?41:130;
   assert.ok(main.includes(`data-study-scatter-count="${n*2}"`)&&main.includes(`data-study-pi-count="${n}"`));
   for(const value of id==='automobile-price'?['3,657.66','5,215.62','0.7777','0.7621','82.93%','34/41','幣別未核實']:['0.7348','1.1664','0.8605','0.7360','1.1558','0.8630','99.23%','129/130','G1、G2','非學期開始前'])assert.ok(content.includes(value),`Missing evaluation ${id} ${value}`);
  } else {
   const n=id==='automobile-price'?41:130;
   assert.ok(main.includes('data-prediction-state="ready"'));
   assert.ok(main.includes(`data-study-sample-count="${n}"`)&&main.includes(`data-study-visible-count="${n}"`));
   assert.ok(!main.includes('研究結果尚未接入'));
   for(const value of ['樣本編號','Actual','Predicted','Residual','Absolute Error','Lower 95% PI','Upper 95% PI','Baseline LR','MI + LR','實際 − 預測','非時間'])assert.ok(main.includes(value),`Missing explorer ${id} ${value}`);
   const content=main.replace(/<[^>]*>/g,'');
   assert.ok(!content.includes(' Wh')&&!content.includes('Timestamp')&&!content.includes('USD')&&!content.includes('TWD'));
   if(id==='student-performance')for(const value of ['G1、G2','非學期開始前','0–20','不裁切'])assert.ok(content.includes(value));
   else assert.ok(content.includes('幣別未核實'));
  }
  if(path==='/prediction-explorer'){assert.ok(main.includes(`data-research-live="${id}"`));assert.ok(main.indexOf('Research Live Predictor')<main.indexOf('Holdout Explorer'));}
  assert.ok(html.includes(`<title>`)&&html.includes(id==='automobile-price'?'Automobile Price':'Student Performance'));
  records.push({path,dataset:id,state:path==='/'?'overview-ready':path==='/feature-selection'?'feature-selection-ready':path==='/model-evaluation'?'model-evaluation-ready':'prediction-explorer-ready',energyResultsAbsent:true});
 }
 for(const query of ['','?dataset=appliances-energy','?dataset=invalid','?dataset=student-performance&dataset=automobile-price']){
  const {main}=await page(path,query);
  assert.ok(!main.includes('data-research-state="not-connected"'));
  assert.ok(main.includes('Holdout')||main.includes('holdout'));
  if(path==='/prediction-explorer')assert.ok(main.includes('data-research-live="appliances-energy"'));
  records.push({path,query,state:'energy'});
 }
}
for(const id of ['automobile-price','student-performance']){
 const {html,main}=await page('/custom-dataset','?dataset='+id);
 assert.ok(!html.includes('id="research-dataset"'));assert.ok(!main.includes('data-research-live='));
 assert.ok(main.includes('id="dataset-file"'));
 assert.ok(main.includes('Server Mode · 預設'));
 assert.ok(!main.includes('data-research-state="not-connected"'));
 records.push({path:'/custom-dataset',queryDataset:id,state:'independent',serverDefault:true});
}
const response=await fetch(origin+'/api/predictions?start=0&size=72');assert.equal(response.status,200);
const window=await response.json();assert.equal(window.rows.length,72);assert.equal(window.rows[0].timestamp,'2016-04-30T08:20:00');
console.log(JSON.stringify({status:'PASS',checks:records.length,records,energyPredictionApiUnchanged:true},null,2));
