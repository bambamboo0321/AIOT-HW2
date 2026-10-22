import { RUNTIME, BROWSER_LIMITS } from "./modules/lib/browser-compute/protocol.mjs";
import { SOURCE_SHA256 } from "./modules/lib/browser-compute/source-lock.mjs";
let runtime, initialization, busy=false, currentJob, stage="idle";
const send=value=>self.postMessage(value);
const progress=value=>send({jobId:currentJob,kind:"progress",progress:value});
const hash=async bytes=>[...new Uint8Array(await crypto.subtle.digest("SHA-256",bytes))].map(n=>n.toString(16).padStart(2,"0")).join("");
const encode=text=>new TextEncoder().encode(text);
const versions={};
const timings={};
async function initialize(){
 if(initialization)return initialization;
 initialization=(async()=>{
  let start=performance.now();stage="runtime";progress({stage});
  const {loadPyodide}=await import(RUNTIME.base+"pyodide.mjs");
  runtime=await loadPyodide({indexURL:RUNTIME.base,packages:[],stderr:()=>{}});
  if(runtime.version!==RUNTIME.version)throw new Error("Runtime version mismatch");
  timings.runtime_ms=performance.now()-start;
  stage="packages";progress({stage});start=performance.now();
  for(const name of Object.keys(RUNTIME.packages)){
   progress({stage:"package:"+name});await runtime.loadPackage(name,{checkIntegrity:true});
   if(!Object.hasOwn(runtime.loadedPackages,name))throw new Error("Missing required package");
  }
  timings.packages_ms=performance.now()-start;
  stage="modules";progress({stage});start=performance.now();
  const response=await fetch(new URL("./python-manifest.json",import.meta.url),{cache:"no-store"});
  if(!response.ok)throw new Error("Python manifest unavailable");
  const bytes=new Uint8Array(await response.arrayBuffer());
  if(await hash(bytes)!==SOURCE_SHA256)throw new Error("Python manifest checksum mismatch");
  const manifest=JSON.parse(new TextDecoder().decode(bytes));
  runtime.FS.mkdirTree("/browser/src");runtime.FS.mkdirTree("/browser/jobs");runtime.FS.mkdirTree("/browser/models");
  for(const row of [...manifest.modules,manifest.adapter]){
   if(!/^(?:src\/[a-z_]+\.py|bridge\.py)$/.test(row.path)||await hash(encode(row.source))!==row.sha256)throw new Error("Python source integrity failure");
   runtime.FS.writeFile("/browser/"+row.path,encode(row.source));
  }
  globalThis.browser_progress=text=>progress(JSON.parse(text));
  await runtime.runPythonAsync("import sys, json\nsys.dont_write_bytecode = True\nsys.path.insert(0, '/browser')\nimport bridge");
  Object.assign(versions,JSON.parse(await runtime.runPythonAsync("json.dumps(bridge.versions(), allow_nan=False)")));
  for(const [name,version] of Object.entries(RUNTIME.packages))if(versions[name==="scikit-learn"?"sklearn":name]!==version)throw new Error("Scientific package version mismatch");
  if(!versions.python.startsWith("3.14.2"))throw new Error("Python version mismatch");
  timings.modules_ms=performance.now()-start;stage="ready";progress({stage});
 })();
 return initialization;
}
self.onmessage=async({data})=>{
 if(!data||typeof data.jobId!=="string"||!/^[a-f0-9-]{36}$/.test(data.jobId))return;
 if(busy){send({jobId:data.jobId,kind:"error",code:"BUSY",message:"Browser Worker 已有工作。",stage});return;}
 busy=true;currentJob=data.jobId;const started=performance.now();
 try{
  await initialize();let report;
  if(data.operation==="initialize")report={ready:true};
  else{
   const payload=data.payload,root="/browser/jobs/"+data.jobId;
   if(["validate","train","compare"].includes(data.operation)){
    if(!(data.csv instanceof ArrayBuffer)||data.csv.byteLength>BROWSER_LIMITS.bytes)throw new Error("Browser Mode 限 CSV 3 MiB");
    runtime.FS.mkdirTree(root);runtime.FS.writeFile(root+"/input.csv",new Uint8Array(data.csv));
    for(const name of ["config","review","experiment"])if(payload[name]!==undefined)runtime.FS.writeFile(root+"/"+name+".json",encode(JSON.stringify(payload[name])));
    if(data.operation==="validate")await runtime.runPythonAsync("[__import__('pathlib').Path(v['path']).unlink(missing_ok=True) for v in bridge.MODELS.values()]\nbridge.MODELS.clear()");
   }
   stage=data.operation;progress({stage});
   const modelId=[...crypto.getRandomValues(new Uint8Array(32))].map(n=>n.toString(16).padStart(2,"0")).join("");
   runtime.globals.set("browser_request",JSON.stringify({operation:data.operation,payload,jobId:data.jobId,modelId}));
   try {report=JSON.parse(await runtime.runPythonAsync("json.dumps(bridge.execute(browser_request), ensure_ascii=False, allow_nan=False)"));}
   finally{runtime.globals.delete("browser_request");}
  }
  // JSON serialization rejects Python NaN/Infinity above; no PyProxy crosses this boundary.
  send({jobId:data.jobId,kind:"result",report,metadata:{engine:"browser",runtime:RUNTIME.version,versions,source_sha256:SOURCE_SHA256,elapsed_ms:performance.now()-started,timings}});
 }catch(error){
  // Do not forward Python tracebacks or raw CSV cells into the UI.
  const limit=String(error).includes("Browser Mode 限");
  const memory=/out of memory|memory access|allocation failed|Cannot enlarge memory/i.test(String(error));
  send({jobId:data.jobId,kind:"error",code:memory?"RESOURCE_EXHAUSTED":limit?"LIMIT":"EXECUTION_FAILED",message:memory?"比較或運算記憶體不足，已停止此 Worker；請降低同時開啟的工作或調整比較範圍。":limit?"Browser 訓練 CSV 限 2,000 列、15 個數值候選特徵、3 MiB；不會自動抽樣或上傳。":"Browser 在「"+stage+"」階段失敗；請檢查網路、輸入與套件載入後手動重試。",stage});
 }finally{busy=false;}
};
