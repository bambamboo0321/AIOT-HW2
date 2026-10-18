import { lstat, mkdir, rename, unlink, writeFile } from "node:fs/promises";
import { randomUUID } from "node:crypto";
import { resolve } from "node:path";
import type { PredictionSnapshot } from "./prediction-types.ts";
import type { Snapshot } from "./research-data.ts";
import type { EvaluationSnapshot } from "./evaluation-types.ts";
import type { FeatureSnapshot } from "./feature-selection-types.ts";

async function checkDestination(path: string, directory: boolean) {
  try {
    const stat = await lstat(path);
    if (stat.isSymbolicLink() || (directory ? !stat.isDirectory() : !stat.isFile())) throw new Error("同步目的地必須為 web/ 內的一般目錄或檔案。");
  } catch (error) {
    if (!(error instanceof Error && "code" in error && error.code === "ENOENT")) throw error;
  }
}

export async function writeSnapshot(webRoot: string, snapshot: Snapshot | FeatureSnapshot | EvaluationSnapshot | PredictionSnapshot, name: "overview" | "feature-selection" | "model-evaluation" | "prediction-explorer" = "overview") {
  const destination = resolve(webRoot, "data");
  const file = resolve(destination, `${name}.json`);
  await checkDestination(destination, true);
  await mkdir(destination, { recursive: true });
  await checkDestination(file, false);
  const temporary = resolve(destination, `.${name}-${randomUUID()}.tmp`);
  try {
    await writeFile(temporary, JSON.stringify(snapshot, null, 2) + "\n", { encoding: "utf8", flag: "wx" });
    await rename(temporary, file);
  } finally {
    await unlink(temporary).catch((error: NodeJS.ErrnoException) => { if (error.code !== "ENOENT") throw error; });
  }
}
