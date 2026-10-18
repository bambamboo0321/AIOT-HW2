import "server-only";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { parseEvaluationSnapshot } from "./evaluation-data.ts";
import type { EvaluationSnapshot } from "./evaluation-types.ts";
export async function loadEvaluation(): Promise<EvaluationSnapshot> {
  try { return parseEvaluationSnapshot(JSON.parse(await readFile(resolve(process.cwd(), "data/model-evaluation.json"), "utf8"))); }
  catch { return { schemaVersion: 1, status: "unavailable", sources: [], issues: ["模型評估快照缺少或損壞。請執行 npm run sync-data 後重新啟動。"] }; }
}
