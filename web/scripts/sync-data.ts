import { buildPredictionSnapshot } from "../lib/prediction-data.ts";
import { buildEvaluationSnapshot } from "../lib/evaluation-data.ts";
import { buildFeatureSnapshot } from "../lib/feature-selection-data.ts";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { buildSnapshot } from "../lib/research-data.ts";
import { writeSnapshot } from "../lib/write-snapshot.ts";

const webRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const root = resolve(webRoot, "..");
const snapshot = await buildSnapshot(root);
await writeSnapshot(webRoot, snapshot);
console.log(`Research data: ${snapshot.status}; ${snapshot.sources.length} allowlisted sources; no original data copied.`);
if (snapshot.status === "unavailable") console.warn(snapshot.issues.join("\n"));

const featureSnapshot = await buildFeatureSnapshot(root);
await writeSnapshot(webRoot, featureSnapshot, "feature-selection");
console.log(`Feature Selection: ${featureSnapshot.status}; ${featureSnapshot.sources.length} allowlisted sources.`);
if (featureSnapshot.status === "unavailable") console.warn(featureSnapshot.issues.join("\n"));

const evaluationSnapshot = await buildEvaluationSnapshot(root);
await writeSnapshot(webRoot, evaluationSnapshot, "model-evaluation");
console.log(`Model Evaluation: ${evaluationSnapshot.status}; ${evaluationSnapshot.sources.length} allowlisted sources.`);
if (evaluationSnapshot.status === "unavailable") console.warn(evaluationSnapshot.issues.join("\n"));

const predictionSnapshot = await buildPredictionSnapshot(root);
await writeSnapshot(webRoot, predictionSnapshot, "prediction-explorer");
console.log(`Prediction Explorer: ${predictionSnapshot.status}; ${predictionSnapshot.sources.length} allowlisted sources.`);
if (predictionSnapshot.status === "unavailable") console.warn(predictionSnapshot.issues.join("\n"));
