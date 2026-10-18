import "server-only";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { parsePredictionSnapshot } from "./prediction-data.ts";
import type { PredictionSnapshot } from "./prediction-types.ts";
export async function loadPredictions(): Promise<PredictionSnapshot> {
  try { return parsePredictionSnapshot(JSON.parse(await readFile(resolve(process.cwd(), "data/prediction-explorer.json"), "utf8"))); }
  catch { return { schemaVersion: 1, status: "unavailable", sources: [], issues: ["逐筆預測快照缺少或格式不符，請執行 npm run sync-data 後重新啟動。"] }; }
}
