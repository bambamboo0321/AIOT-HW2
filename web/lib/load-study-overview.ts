import "server-only";
import { readFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { join } from "node:path";
import manifest from "../data/research-overviews/v1/manifest.json";
import { parseStudyOverview, type StudyDatasetId, type StudySnapshot } from "./study-overview-data.ts";

export async function loadStudyOverview(id: StudyDatasetId): Promise<StudySnapshot> {
  try {
    // Literal allowlisted paths keep production tracing independent of experiments/.
    const bytes = id === "automobile-price"
      ? await readFile(join(process.cwd(), "data/research-overviews/v1/automobile-price.json"))
      : await readFile(join(process.cwd(), "data/research-overviews/v1/student-performance.json"));
    if (manifest.schemaVersion !== 1 || createHash("sha256").update(bytes).digest("hex") !== manifest.snapshots[id].sha256) throw new Error("研究快照 SHA-256 完整性檢查失敗。");
    return parseStudyOverview(JSON.parse(bytes.toString("utf8")), id);
  } catch (error) {
    return { status: "unavailable", issues: [error instanceof Error && error.message.includes("SHA-256") ? error.message : "研究快照缺少或格式錯誤；沒有填入替代資料。"] };
  }
}
