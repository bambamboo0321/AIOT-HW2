import {test} from "node:test";
import assert from "node:assert/strict";
import {serverComputeStatus} from "../lib/custom-compute-availability.ts";
test("Vercel without Gateway never advertises local Python, even if executable exists",()=>{
  const result=serverComputeStatus({VERCEL:"1"},true);
  assert.equal(result.available,false);assert.match(result.message,/Browser Mode/);
});
test("partial configuration does not fall back to local Python",()=>{
  assert.equal(serverComputeStatus({AEP_BACKEND_URL:"placeholder"},true).available,false);
});
test("local Python availability is advisory; configured service is not claimed healthy",()=>{
  assert.equal(serverComputeStatus({},false).available,false);
  assert.equal(serverComputeStatus({},true).available,true);
  const result=serverComputeStatus({AEP_BACKEND_URL:"placeholder",AEP_BACKEND_TOKEN:"placeholder",AEP_SESSION_SECRET:"placeholder",AEP_PUBLIC_ORIGIN:"placeholder"},false);
  assert.equal(result.available,true);assert.match(result.message,/尚未確認服務健康/);
  assert.ok(!result.message.includes("placeholder"));
});
