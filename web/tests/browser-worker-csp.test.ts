import {test} from "node:test";
import assert from "node:assert/strict";
import config from "../next.config.ts";
test("Worker response allows Safari same-origin module imports without broadening document/CDN policy",async()=>{
 const rules=await config.headers!();
 assert.equal(rules.length,1);assert.equal(rules[0].source,"/browser-compute/worker.mjs");
 const headers=Object.fromEntries(rules[0].headers.map(h=>[h.key.toLowerCase(),h.value]));
 const directives=Object.fromEntries(headers["content-security-policy"].split(";").map(s=>s.trim().split(/\s+/)).filter(p=>p[0]).map(([k,...v])=>[k,v]));
 assert.deepEqual(directives["worker-src"],["'self'"]);
 assert.deepEqual(directives["default-src"],["'none'"]);
 assert.deepEqual(directives["script-src"],["'self'","https://cdn.jsdelivr.net/pyodide/v314.0.7/full/","'wasm-unsafe-eval'"]);
 assert.deepEqual(directives["connect-src"],["'self'","https://cdn.jsdelivr.net/pyodide/v314.0.7/full/"]);
 assert.deepEqual(directives["object-src"],["'none'"]);
 assert.equal(headers["x-content-type-options"],"nosniff");assert.equal(headers["cache-control"],"no-store");
 assert.ok(!headers["content-security-policy"].includes("'unsafe-eval'"));assert.ok(!headers["content-security-policy"].includes("*"));
});
