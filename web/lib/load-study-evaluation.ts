import "server-only";
import { readFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { join } from "node:path";
import manifest from "../data/research-evaluation/v1/manifest.json";
import overviewManifest from "../data/research-overviews/v1/manifest.json";
import { loadStudyOverview } from "./load-study-overview.ts";
import { parseStudyEvaluation,type StudyEvaluationSnapshot } from "./study-evaluation-data.ts";
import type { StudyDatasetId } from "./study-overview-data.ts";
export async function loadStudyEvaluation(id:StudyDatasetId):Promise<StudyEvaluationSnapshot>{
  try{
    const overview=await loadStudyOverview(id);if(overview.status!=="ready")return overview;
    const bytes=id==="automobile-price"?await readFile(join(process.cwd(),"data/research-evaluation/v1/automobile-price.json")):await readFile(join(process.cwd(),"data/research-evaluation/v1/student-performance.json"));
    if(manifest.schemaVersion!==1||createHash("sha256").update(bytes).digest("hex")!==manifest.snapshots[id].sha256)throw new Error("評估快照 SHA-256 完整性檢查失敗。");
    return parseStudyEvaluation(JSON.parse(bytes.toString("utf8")),id,overviewManifest.snapshots[id].sha256,overview.data);
  }catch(error){return {status:"unavailable",issues:[error instanceof Error&&error.message.includes("SHA-256")?error.message:"評估快照缺少或格式錯誤；沒有填入替代結果。"]};}
}
