import {test} from 'node:test';
import assert from 'node:assert/strict';
import {LivePredictionController,type PredictionValues} from '../lib/live-prediction-controller.ts';
import {sliderPresentation,initialPredictionInputs} from '../lib/live-prediction-input.ts';
import {parseTrainingInputRanges} from '../lib/custom-prediction-types.ts';
const wait=(ms=20)=>new Promise(r=>setTimeout(r,ms));
function setup(request:(v:PredictionValues,s:AbortSignal)=>Promise<number>,timeoutMs=1000){const results:{n:number;v:PredictionValues}[]=[],errors:unknown[]=[];let starts=0;const c=new LivePredictionController({request,onStart:()=>starts++,onResult:(n,v)=>results.push({n,v}),onError:e=>errors.push(e),onIdle:()=>{},delayMs:5,timeoutMs});return {c,results,errors,starts:()=>starts};}
test('Training ranges validate source, rows and selected feature order; null and constants are explicit',()=>{
 const raw={source:'full_training_raw',training_rows:64,features:[{feature:'x',min:1,max:3},{feature:'constant',min:4,max:4},{feature:'empty',min:null,max:null}]};
 const r=parseTrainingInputRanges(raw,['x','constant','empty'],64);assert.deepEqual(initialPredictionInputs(r),{x:'2',constant:'4'});
 for(const value of [{...raw,source:'holdout'},{...raw,training_rows:80},{...raw,features:[{feature:'x',min:5,max:3}]}])assert.throws(()=>parseTrainingInputRanges(value,['x','constant','empty'],64));
 assert.throws(()=>parseTrainingInputRanges(raw,['empty','constant','x'],64));
 assert.equal(sliderPresentation('4',4,4).disabled,true);assert.equal(sliderPresentation('',null,null).disabled,true);
});
test('slider clamps only its displayed thumb, preserving extrapolation number and missing inputs',()=>{
 const raw='120.5';assert.equal(sliderPresentation(raw,0,100).value,100);assert.equal(sliderPresentation(raw,0,100).outside,true);assert.equal(raw,'120.5');
 assert.equal(sliderPresentation('-20',0,100).value,0);assert.equal(sliderPresentation('',0,100).outside,false);assert.equal(sliderPresentation('garbage',0,100).outside,false);
});
test('debounce coalesces bursts and snapshots inputs without retraining',async()=>{const s=setup(async v=>v.x!);const value={x:1};s.c.schedule(value);value.x=99;s.c.schedule({x:2});s.c.schedule({x:3});await wait();assert.equal(s.starts(),1);assert.deepEqual(s.results,[{n:3,v:{x:3}}]);s.c.dispose();});
test('slow inference is serial, never aborted by editing, and cannot overwrite latest input',async()=>{
 let release!:(n:number)=>void;const signals:AbortSignal[]=[];const s=setup(async(v,signal)=>{signals.push(signal);return v.x===1?new Promise<number>(r=>release=r):v.x!;});
 s.c.schedule({x:1},true);s.c.schedule({x:2});await wait();assert.equal(s.starts(),1);assert.equal(signals[0].aborted,false);release(1);await wait();assert.equal(s.starts(),2);assert.deepEqual(s.results,[{n:2,v:{x:2}}]);s.c.dispose();
});
test('clear suppresses active response; disabling live leaves model request intact',async()=>{let release!:(n:number)=>void;let signal!:AbortSignal;const s=setup(async(_,a)=>{signal=a;return new Promise(r=>release=r);});s.c.schedule({x:1},true);s.c.schedule(null);assert.equal(signal.aborted,false);release(9);await wait();assert.deepEqual(s.results,[]);s.c.dispose();});
test('cancel and dispose abort active work and discard queued responses',async()=>{
 for(const action of ['cancel','dispose'] as const){let aborted=false;const s=setup(async(_,signal)=>new Promise((_,reject)=>signal.addEventListener('abort',()=>{aborted=true;reject(new DOMException('cancelled','AbortError'));})));s.c.schedule({x:1},true);s.c.schedule({x:2});s.c[action]();await wait();assert.equal(aborted,true);assert.equal(s.starts(),1);assert.equal(s.results.length,0);s.c.dispose();}
});
test('timeout aborts request; errors require manual retry and never loop automatically',async()=>{
 const s=setup(async(_,signal)=>new Promise((_,reject)=>signal.addEventListener('abort',()=>reject(new Error('timeout')))),5);s.c.schedule({x:1},true);await wait();assert.equal(s.errors.length,1);assert.equal(s.starts(),1);await wait();assert.equal(s.starts(),1);s.c.dispose();
 const retry=setup(async()=>{throw new Error('server unavailable');});retry.c.schedule({x:1},true);await wait();assert.equal(retry.errors.length,1);retry.c.schedule({x:2},true);await wait();assert.equal(retry.errors.length,2);retry.c.dispose();
});
