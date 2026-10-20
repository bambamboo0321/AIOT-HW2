export const BROWSER_LIMITS = { bytes: 3 * 1024 * 1024, rows: 2000, features: 15 } as const;
export const RUNTIME = { version: "314.0.7", base: "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/",
  packages: { numpy: "2.4.6", pandas: "3.0.2", scipy: "1.18.0", "scikit-learn": "1.8.0", joblib: "1.5.3" } } as const;
export type Operation = "initialize" | "validate" | "train" | "compare" | "predict" | "predict-batch" | "delete-model";
export type Progress = { stage: string; method?: string; phase?: "started" | "completed"; elapsed_ms?: number };
export type ComputeMetadata = { engine: "browser"; runtime: string; versions: Record<string,string>; source_sha256: string; elapsed_ms: number };
export type ComputeResult<T> = { report: T; metadata: ComputeMetadata };
export type DatasetInput = { csv: Blob; config: unknown; review: unknown; experiment?: unknown };
export type WorkerRequest = { jobId: string; operation: Operation; payload: Record<string,unknown>; csv?: ArrayBuffer };
export type WorkerReply = { jobId: string; kind: "progress"; progress: Progress } | { jobId: string; kind: "result"; report: unknown; metadata: ComputeMetadata } | { jobId: string; kind: "error"; code: string; message: string; stage: string };
export class BrowserComputeError extends Error {
  readonly code: string;
  constructor(code: string, message: string) { super(message); this.code=code; this.name = "BrowserComputeError"; }
}
