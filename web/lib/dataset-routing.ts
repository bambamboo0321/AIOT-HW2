import { getDataset, resolveDatasetId, type DatasetDefinition, type DatasetId } from "./dataset-registry.ts";

export const RESEARCH_PATHS = ["/", "/feature-selection", "/model-evaluation", "/prediction-explorer"] as const;
export type ResearchPath = typeof RESEARCH_PATHS[number];
export type WorkspacePath = ResearchPath | "/custom-dataset";
export type ResearchSearchParams = { dataset?: string | string[] };
export type ResearchPageProps = { searchParams: Promise<ResearchSearchParams> };

export function isResearchPath(path: string): path is ResearchPath {
  return (RESEARCH_PATHS as readonly string[]).includes(path);
}
export function datasetIdFromQuery(params: Pick<URLSearchParams, "getAll">): DatasetId {
  const ids = params.getAll("dataset");
  return resolveDatasetId(ids.length === 1 ? ids[0] : undefined);
}
export function datasetFromPageParams(params: ResearchSearchParams): DatasetDefinition {
  return getDataset(params.dataset);
}
/** Custom Dataset never consumes the research selection, even with a manually added query. */
export function datasetForPath(path: string, params: Pick<URLSearchParams, "getAll">): DatasetDefinition | null {
  return isResearchPath(path) ? getDataset(datasetIdFromQuery(params)) : null;
}
export function researchHref(path: ResearchPath, id: DatasetId, anchor?: string): string {
  const query = new URLSearchParams({ dataset: resolveDatasetId(id) });
  return `${path}?${query}${anchor ? `#${encodeURIComponent(anchor)}` : ""}`;
}
export function workspaceHref(path: WorkspacePath, id: DatasetId): string {
  return isResearchPath(path) ? researchHref(path, id) : path;
}
export function datasetSwitchHref(path: string, query: string, next: unknown, hash = ""): string | null {
  if (!isResearchPath(path)) return null;
  const params = new URLSearchParams(query);
  params.set("dataset", resolveDatasetId(next));
  return `${path}?${params}${hash.startsWith("#") ? hash : ""}`;
}
