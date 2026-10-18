import "server-only";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { parseSnapshot, type Snapshot } from "./research-data.ts";

export async function loadOverview(): Promise<Snapshot> {
  try {
    return parseSnapshot(JSON.parse(await readFile(resolve(process.cwd(), "data/overview.json"), "utf8")));
  } catch {
    return { schemaVersion: 1, status: "unavailable", sources: [],
      issues: ["研究資料快照缺少或格式錯誤。請於 web/ 執行 npm run sync-data 後重新啟動。"] };
  }
}
