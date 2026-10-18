import "server-only";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { parseFeatureSnapshot } from "./feature-selection-data.ts";
import type { FeatureSnapshot } from "./feature-selection-types.ts";

export async function loadFeatureSelection(): Promise<FeatureSnapshot> {
  try { return parseFeatureSnapshot(JSON.parse(await readFile(resolve(process.cwd(), "data/feature-selection.json"), "utf8"))); }
  catch { return { schemaVersion: 1, status: "unavailable", sources: [], issues: ["特徵選擇快照缺少或格式錯誤。請執行 npm run sync-data 後重新啟動。"] }; }
}
