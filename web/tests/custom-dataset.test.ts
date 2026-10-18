import { test } from "node:test";
import assert from "node:assert/strict";
import { CSV_LIMITS, decodeCsv, emptyDraft, isoTime, parseCsv, profileCsv, validatePlan, type WizardDraft } from "../lib/custom-dataset.ts";
const csv = "date,x,target,category,code\n" + Array.from({length:30}, (_, i) => `2016-05-${String(i+1).padStart(2,"0")},${i*2},${i%7},${i%2?"a":"b"},00${i}`).join("\n");
const profile = profileCsv(csv);
const draft = (): WizardDraft => ({...emptyDraft(), target:"target", features:["x"]});

test("CSV parsing preserves BOM, escaped quotes, commas, multiline fields and final empty fields", () => {
  const p = parseCsv('\uFEFFname,value,tail\r\n"a,b","say ""hi""",\r\n"two\nlines",2,3');
  assert.deepEqual(p.headers,["name","value","tail"]); assert.deepEqual(p.rows,[["a,b",'say "hi"',""],["two\nlines","2","3"]]);
  assert.deepEqual(parseCsv('a\nb\n\n""\n').rows,[["b"],[""]]);
});
test("malformed structure and ambiguous headers are rejected without silently repairing the source", () => {
  for (const value of ["", "a,b\n", "a,a\n1,2", "a, \n1,2", " a,b\n1,2", "a,b\n1", 'a,b\n"x" junk,2', 'a,b\n"open,2', 'a,b\nx"y,2', "a,b\n1,\0", `${"a".repeat(129)},b\n1,2`]) assert.throws(()=>parseCsv(value));
});
test("UTF-8 decoding rejects invalid bytes and empty or oversized buffers", () => {
  assert.equal(decodeCsv(new TextEncoder().encode("a\n中").buffer),"a\n中");
  assert.throws(()=>decodeCsv(new Uint8Array([0xff,0xfe]).buffer));
  assert.throws(()=>decodeCsv(new ArrayBuffer(0))); assert.throws(()=>decodeCsv(new ArrayBuffer(CSV_LIMITS.bytes+1)));
});
test("record, column, cell and field limits are enforced during parsing", () => {
  assert.throws(()=>parseCsv(Array.from({length:201},(_,i)=>`c${i}`).join(",")+"\n1"),/200/);
  assert.throws(()=>parseCsv("a\n"+"x\n".repeat(50001)),/50,000/);
  assert.throws(()=>parseCsv("a\n"+"x".repeat(65537)),/65,536/);
  const head=Array.from({length:200},(_,i)=>`c${i}`).join(","); const row=Array(200).fill("1").join(",");
  assert.throws(()=>parseCsv(head+"\n"+(row+"\n").repeat(5001)),/100 萬/);
});
test("profiles are descriptive and preserve leading zeros, long integers, NA rules and real extreme values", () => {
  const p=profileCsv("code,x,y\n001,1,NA\n002,2,2\n003,3,3\n004,4,4\n005,1000,5");
  assert.notEqual(p.columns[0].kind,"numeric"); assert.equal(p.preview[0][0],"001");
  assert.equal(p.columns[1].numeric!.max,1000); assert.equal(p.columns[1].numeric!.median,3); assert.equal(p.columns[1].numeric!.outliers,1);
  assert.equal(p.columns[2].missing,1); assert.equal(p.rows,5);
  const unsafe=profileCsv("a,b\n-9007199254740993,1\n-9007199254740994,2");assert.notEqual(unsafe.columns[0].kind,"numeric");
  assert.equal(profileCsv("a\n \nNA\nnull\n1").columns[0].missing,2);
});
test("constant, all-missing, infinite and mixed targets are not offered as valid regression targets", () => {
  for (const values of ["1\n1", "NA\nnull", "1\nInfinity", "1\n1e999", "1\nword", "true\nfalse"]) assert.equal(profileCsv("target\n"+values).columns[0].targetEligible,false);
});
test("ISO date detection rejects impossible dates, timestamps and ambiguous locale formats", () => {
  assert.notEqual(isoTime("2024-02-29 12:30:00.250"),null);
  for (const value of ["2023-02-29", "2024-02-30", "2024-13-01", "2024-01-01T24:00", "2024-01-01T10:61", "01/02/2024", "1704067200", "2024-01-01T00:00Z"]) assert.equal(isoTime(value),null);
  assert.equal(profile.columns[0].kind,"datetime"); assert.equal(profile.columns[0].dates!.unique,30);
});
test("duplicate rows are counted without dropping them and preview values stay bounded", () => {
  const p=profileCsv('a,b\n1,2\n1,2\n3,"'+"x".repeat(250)+'"');assert.equal(p.rows,3);assert.equal(p.duplicateRows,1);assert.equal(p.preview[2][1].length,201);
});
test("no target or features are chosen without user action, and configured plans follow Python v1 keys", () => {
  const empty=emptyDraft();assert.equal(empty.target,"");assert.deepEqual(empty.features,[]);assert.ok(validatePlan(profile,empty).errors.length);
  const result=validatePlan(profile,draft());assert.deepEqual(result.errors,[]);assert.equal(result.config.schema_version,1);
  assert.equal(result.config.target_column,"target");assert.deepEqual(result.config.feature_columns,["x"]);assert.deepEqual(result.config.column_types,{x:"numeric",target:"numeric"});assert.equal(result.config.cv_gap,0);
  assert.equal(result.usableRows,30);assert.equal(result.trainRows,24);assert.equal(result.testRows,6);
  assert.ok(result.warnings.some(w=>w.includes("13")));assert.ok(result.warnings.some(w=>w.includes("時間依賴")));
});
test("target leakage, missing columns, duplicate features, categories and constant features are rejected", () => {
  for (const features of [["target"],["absent"],["x","x"],["category"],["code"]]) assert.ok(validatePlan(profile,{...draft(),features}).errors.length);
  assert.ok(validatePlan(profileCsv("x,target\n1,1\n1,2"),draft()).errors.some(e=>e.includes("特徵")));
});
test("missing targets can only error or be excluded before split; selected feature missingness is checked", () => {
  const p=profileCsv(csv.replace("2016-05-01,0,0", "2016-05-01,,NA"));
  assert.ok(validatePlan(p,draft()).errors.some(e=>e.includes("不能補目標")));
  const d={...draft(),missingTarget:"drop" as const};const result=validatePlan(p,d);assert.equal(result.usableRows,29);assert.deepEqual(result.errors,[]);
  assert.ok(validatePlan(p,{...d,missingNumeric:"none"}).errors.some(e=>e.includes("不能選擇不補值")));
});
test("time split requires complete unique ISO dates; duplicates are not automatically grouped", () => {
  const d={...draft(),split:"chronological" as const,splitColumn:"date",gap:1};assert.deepEqual(validatePlan(profile,d).errors,[]);
  const duplicate=profileCsv(csv.replace("2016-05-02","2016-05-01"));assert.ok(validatePlan(duplicate,d).errors.length);
  const missing=profileCsv(csv.replace("2016-05-01","NA"));assert.ok(validatePlan(missing,d).errors.length);
  assert.ok(validatePlan(profile,{...d,splitColumn:"target"}).errors.length);
});
test("CV feasibility rejects fractional, empty, oversized and excessive gap configurations", () => {
  for(const cvSplits of [NaN,1,2.5,11])assert.ok(validatePlan(profile,{...draft(),cvSplits}).errors.length);
  for(const gap of [NaN,-1,1.5,50001])assert.ok(validatePlan(profile,{...draft(),gap}).errors.length);
  assert.ok(validatePlan(profile,{...draft(),split:"chronological",splitColumn:"date",gap:30}).errors.length);
  assert.ok(validatePlan(profile,{...draft(),gap:1}).errors.length);
  assert.ok(validatePlan(profileCsv("x,target\n1,2\n2,3\n3,4"),draft()).errors.length);
});
test("chronological and random split counts match Python rounding, including floating-point boundaries", () => {
  const p={...profile,rows:90};assert.equal(validatePlan(p,{...draft(),split:"chronological",splitColumn:"date",testSize:0.3}).trainRows,Math.floor(90*(1-0.3)));
  assert.equal(validatePlan(p,{...draft(),testSize:0.3}).testRows,Math.ceil(90*0.3));
});
test("prototype-like names and formula/HTML-looking cells stay plain data", () => {
  const p=profileCsv('__proto__,target,note\n1,2,"<img src=x onerror=alert(1)>"\n3,4,=1+1\n5,6,text');
  const cfg=validatePlan(p,{...draft(),features:["__proto__"],cvSplits:2}).config;
  assert.ok(Object.hasOwn(cfg.column_types,"__proto__"));assert.equal(Object.getPrototypeOf(cfg.column_types),Object.prototype);
  assert.equal(p.preview[0][2],"<img src=x onerror=alert(1)>");assert.equal(p.preview[1][2],"=1+1");
});
