// Deterministic packaging from current production sources; never consumes experiment source snapshots.
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { createHash } from "node:crypto";
import { resolve, dirname, relative } from "node:path";
import { fileURLToPath } from "node:url";
import ts from "typescript";
const web = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const repo = resolve(web, ".."), output = resolve(web, "public/browser-compute");
const hash = data => createHash("sha256").update(data).digest("hex");
const modules = new Map(), queue = ["__init__", "custom_dataset_validation", "custom_dataset_selectors", "custom_dataset_training", "custom_dataset_comparison", "custom_dataset_prediction"];
while (queue.length) {
  const name = queue.shift(); if (modules.has(name)) continue;
  const path = "src/"+name+".py", source = await readFile(resolve(repo,path), "utf8");
  modules.set(name, {path, sha256: hash(source), source});
  for (const match of source.matchAll(/^\s*from \.(\w+) import /gm)) if (!modules.has(match[1])) queue.push(match[1]);
}
const adapter = await readFile(resolve(web,"browser-compute/bridge.py"),"utf8");
const manifest = { schema_version:1, source:"current repository src/", modules:[...modules.values()].sort((a,b)=>a.path.localeCompare(b.path)), adapter:{path:"bridge.py",sha256:hash(adapter),source:adapter} };
const bytes=JSON.stringify(manifest,null,2)+"\n";
await mkdir(output,{recursive:true});
await writeFile(resolve(output,"python-manifest.json"),bytes);
await writeFile(resolve(web,"lib/browser-compute/source-lock.ts"),`// Generated from current src/ by build-browser-compute.mjs.\nexport const SOURCE_SHA256 = ${JSON.stringify(hash(bytes))};\n`);
// The same TS client is importable by Next and by the production browser smoke tests.
const emitted=new Set();
async function emit(file) {
  if(emitted.has(file))return; emitted.add(file);
  const source=await readFile(file,"utf8");
  for(const match of source.matchAll(/(?:from\s+|import\s*)["'](\.[^"']+\.ts)["']/g)) await emit(resolve(dirname(file),match[1]));
  const js=ts.transpileModule(source,{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.ESNext,verbatimModuleSyntax:false}}).outputText.replace(/(["'])(\.[^"']+)\.ts\1/g,'$1$2.mjs$1');
  const dest=resolve(output,"modules",relative(web,file).replace(/\.ts$/,".mjs"));
  await mkdir(dirname(dest),{recursive:true}); await writeFile(dest,js);
}
await emit(resolve(web,"lib/browser-compute/client.ts"));
await emit(resolve(web,"lib/browser-compute/source-lock.ts"));
console.log(JSON.stringify({modules:manifest.modules.length,manifest_sha256:hash(bytes),client_modules:emitted.size}));
