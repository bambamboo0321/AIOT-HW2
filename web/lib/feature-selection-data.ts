import { readSource } from "./research-data.ts";
import { METHOD_NAMES, type MethodId, type Scores, type FeatureData, type FeatureSnapshot } from "./feature-selection-types.ts";

type Obj = Record<string, unknown>;
export const FEATURE_SOURCES = {
  selection: "dashboard_data/feature_selection.json",
  config: "reports/phase3/results/experiment_config.json",
  best: "reports/phase3/results/best_features.json",
  plots: "reports/phase3/results/plot_data.json",
  records: "reports/phase3/results/feature_selection_results.json",
  exploration: "reports/phase3/phase3d/comparison_summary.json",
} as const;
function obj(v: unknown): Obj {
  if (!v || typeof v !== "object" || Array.isArray(v)) throw new Error("資料必須為 object");
  return v as Obj;
}
function arr(v: unknown): unknown[] { if (!Array.isArray(v)) throw new Error("缺少陣列"); return v; }
function num(v: unknown): number { if (typeof v !== "number" || !Number.isFinite(v)) throw new Error("缺少有限數值"); return v; }
function count(v: unknown): number { const n = num(v); if (!Number.isInteger(n) || n < 1) throw new Error("欄數／folds 必須為正整數"); return n; }
function str(v: unknown): string { if (typeof v !== "string" || !v.trim()) throw new Error("缺少文字"); return v; }
function names(v: unknown): string[] { return arr(v).map(str); }
function methodId(v: unknown): MethodId {
  const id = str(v); if (!Object.hasOwn(METHOD_NAMES, id)) throw new Error("未知的方法"); return id as MethodId;
}
function scores(v: unknown, wh = false): Scores {
  const o = obj(v); const rmse = num(o[wh ? "rmse_wh" : "rmse"]); const mae = num(o[wh ? "mae_wh" : "mae"]); const r2 = num(o.r2);
  if (rmse < 0 || mae < 0 || r2 > 1) throw new Error("指標超出合法範圍");
  return { rmse, mae, r2 };
}
function same(a: unknown, b: unknown, label: string) { if (JSON.stringify(a) !== JSON.stringify(b)) throw new Error(`${label} 不一致`); }

export function adaptFeatureSelection(value: unknown, configValue: unknown): FeatureData {
  const root = obj(value); const config = obj(configValue);
  if (root.schema_version !== 1 || root.artifact_type !== "feature_selection" || config.phase !== "3B" || config.schema_version !== 1) throw new Error("Feature Selection schema／Phase 錯誤");
  const units = obj(root.units); if (units.rmse !== "Wh" || units.mae !== "Wh" || units.r2 !== "dimensionless") throw new Error("評估指標單位錯誤");
  const data = obj(root.data); const experiment = obj(data.experiment);
  if (experiment.status !== "complete" || config.holdout_target_read !== false || data.holdout_used_for_selection !== false || data.selection_level !== "final_selection_cv (3 folds on Training only)") throw new Error("非完整 Training-only 正式結果");
  for (const [key, v] of Object.entries(obj(experiment.config))) same(v, config[key], `Dashboard 與正式實驗設定 ${key}`);
  for (const key of ["phase", "training_rows", "holdout_rows", "outer_splits", "inner_splits", "gap", "grids"]) same(obj(experiment.config)[key], config[key], `必要設定 ${key}`);
  const outerFolds = count(config.outer_splits); const innerFolds = count(config.inner_splits);
  const featureNames = names(obj(config.preprocessing).feature_columns); const candidateFeatures = featureNames.length;
  const gap = num(config.gap); if (!Number.isInteger(gap) || gap < 0) throw new Error("gap 格式錯誤");
  const curves = obj(data.curves);
  const points = arr(obj(curves["rmse_vs_feature_count.png"]).points).map(obj);
  for (const key of ["mae", "r2"]) same(obj(curves[`${key}_vs_feature_count.png`]).points, points, "三指標曲線來源");
  const methods = arr(data.methods).map(v => {
    const m = obj(v); const id = methodId(m.method); const automatic = id === "rfecv" || id === "lasso_cv";
    const cv = obj(m.training_cv); const nested = obj(m.nested_cv); const refit = obj(m.full_training_refit); const selection = obj(refit.selection);
    if (cv.method !== id || nested.method !== id || refit.method !== id || cv.status !== "ok" || refit.status !== "ok" || nested.level !== "outer_evaluation" || refit.level !== "final_refit_training") throw new Error("方法評估層級／狀態不一致");
    if (count(cv.fold_count) !== innerFolds || count(nested.folds) !== outerFolds || nested.valid_folds !== outerFolds) throw new Error("CV folds 不完整");
    const finalK = count(refit.selected_k); const transformedFeatures = names(selection.selected_feature_names);
    const mapping = obj(selection.original_feature_mapping);
    const features = Object.entries(mapping).filter(([, v]) => names(v).length > 0).map(([key]) => key);
    if (finalK > candidateFeatures || finalK !== selection.selected_k || transformedFeatures.length !== finalK || new Set(transformedFeatures).size !== finalK || features.length !== finalK || features.some(f => !featureNames.includes(f))) throw new Error("最終 Features 與 k 不符");
    same(features.flatMap(f => names(mapping[f])), transformedFeatures, "原始／轉換欄位 mapping");
    const outerK = arr(nested.selected_k).map(count); if (outerK.length !== outerFolds || outerK.some(k => k > candidateFeatures)) throw new Error("Outer k 格式錯誤");
    const grid = arr(obj(config.grids)[id]);
    if (automatic ? cv.requested_k !== null || refit.requested_k !== null || grid.length !== 1 || grid[0] !== null : cv.requested_k !== finalK || refit.requested_k !== finalK) throw new Error("自動／固定 k 定義不一致");
    const meanFinalK = num(cv.mean_selected_k); if (meanFinalK < 1 || meanFinalK > candidateFeatures) throw new Error("平均 k 錯誤");
    const own = points.filter(p => p.method === id);
    if (automatic) {
      if (own.length !== 1 || own[0].requested_k !== null || own[0].level !== "outer_evaluation") throw new Error("自動選 k 不得有虛構曲線");
      same(own[0], nested, "自動 k 單點與 Nested CV");
    } else {
      same(own.map(p => p.requested_k).sort((a, b) => Number(a) - Number(b)), grid, "實測 k 與 grid");
    }
    const curve = automatic ? [] : own.map(p => {
      const k = count(p.requested_k); if (p.level !== "outer_k_curve" || p.folds !== outerFolds) throw new Error("k 曲線層級錯誤");
      const validFolds = num(p.valid_folds); if (!Number.isInteger(validFolds) || validFolds < 0 || validFolds > outerFolds) throw new Error("有效 folds 錯誤");
      if (arr(p.selected_k).some(v => v !== k)) throw new Error("固定 k 與 fold k 不符");
      return { k, scores: validFolds === outerFolds ? scores(p) : null, validFolds };
    }).sort((a, b) => a.k - b.k);
    return { id, automatic, finalK, features, transformedFeatures, finalCV: scores(cv), meanFinalK,
      nestedCV: scores(nested), outerK, outerResults: [], curve };
  });
  same([...methods.map(m => m.id)].sort(), Object.keys(METHOD_NAMES).sort(), "六種方法");
  if (points.some(p => !methods.some(m => m.id === p.method))) throw new Error("曲線包含未知方法");
  const winner = obj(data.winner); const winnerId = methodId(winner.method);
  const winnerMethod = methods.find(m => m.id === winnerId)!;
  same(scores(winner), winnerMethod.finalCV, "Winner CV");
  return { trainingRows: count(config.training_rows), holdoutRows: count(config.holdout_rows), outerFolds, innerFolds,
    gap, candidateFeatures, winner: winnerId, selectionRule: str(config.selection_rule), methods,
    exploration: { available: false, reason: "Phase 3D 尚未載入。" } };
}

// Only measured outer-evaluation rows are exposed, never inner search scores or model objects.
export function attachOuterResults(data: FeatureData, value: unknown): void {
  const root = obj(value); const config = obj(root.config);
  if (root.status !== "complete" || config.phase !== "3B" || config.holdout_target_read !== false || config.training_rows !== data.trainingRows || config.holdout_rows !== data.holdoutRows || config.outer_splits !== data.outerFolds || config.gap !== data.gap) throw new Error("逐 fold 成果條件不一致");
  const rows = arr(root.records).map(obj).filter(r => r.level === "outer_evaluation");
  if (rows.length !== data.methods.length * data.outerFolds) throw new Error("逐 fold 評估筆數不完整");
  for (const method of data.methods) {
    const own = rows.filter(r => r.method === method.id).sort((a, b) => num(a.outer_fold) - num(b.outer_fold));
    if (own.length !== data.outerFolds) throw new Error("方法缺少完整 Outer folds");
    const results = own.map((r, index) => {
      const fold = count(r.outer_fold); const k = count(r.selected_k);
      if (fold !== index + 1 || r.status !== "ok" || r.inner_fold !== null || k !== method.outerK[index] || k > data.candidateFeatures || (method.automatic && r.requested_k !== null)) throw new Error("逐 fold 編號／狀態／選欄數不一致");
      const train = arr(r.train_range).map(num); const validation = arr(r.validation_range).map(num);
      if (train.length !== 2 || validation.length !== 2 || [...train, ...validation].some(n => !Number.isInteger(n) || n < 0) || train[0] !== 0 || train[1] <= train[0] || validation[0] < train[1] + data.gap || validation[1] <= validation[0] || validation[1] > data.trainingRows) throw new Error("逐 fold 評估超出 Training 時間邊界");
      return { fold, k, scores: scores(r) };
    });
    checkOuterMeans(results.map(r => r.scores), method.nestedCV);
    method.outerResults = results;
  }
}

function checkOuterMeans(results: Scores[], summary: Scores): void {
  for (const metric of ["rmse", "mae", "r2"] as const) {
    const mean = results.reduce((sum, r) => sum + r[metric], 0) / results.length;
    if (Math.abs(mean - summary[metric]) > 1e-10) throw new Error(`逐 fold ${metric} 平均與 Nested CV 摘要不符`);
  }
}

export function adaptFeatureExploration(value: unknown, core: FeatureData): FeatureData["exploration"] {
  const root = obj(value); const c = obj(root.conditions);
  if (root.phase !== "3D" || root.status !== "complete" || c.holdout_used !== false || c.training_rows !== core.trainingRows || c.gap !== core.gap || c.cv_splits !== core.innerFolds || c.comparison_level !== "final_selection_cv") throw new Error("Phase 3D 評估條件不符");
  const k = count(c.k); const random = obj(root.random_search);
  const sampleCount = count(random.sample_count); if (sampleCount !== random.unique_count || sampleCount !== c.sample_count) throw new Error("抽樣數量不一致");
  const fixedRows = Object.entries(obj(root.fixed_combinations)).map(([id, v]) => {
    const row = obj(v); const features = names(row.features); if (features.length !== k) throw new Error("Phase 3D 固定組合 k 不符");
    return { label: METHOD_NAMES[methodId(id)], scores: scores(row, true), features };
  });
  const best = obj(random.best); const bestFeatures = names(best.features); if (bestFeatures.length !== k) throw new Error("抽樣最佳 k 不符");
  const selectorRows = arr(root.selector_cv).map(v => {
    const row = obj(v); if (row.evaluation !== "B_selector_inside_each_training_fold" || row.k !== k || arr(row.folds).length !== c.cv_splits) throw new Error("Phase 3D selector CV 層級不符");
    return { id: methodId(row.method), scores: scores(row) };
  });
  return { available: true, k, folds: count(c.cv_splits), sampleCount, selectorRows,
    fixedRows: [...fixedRows, { label: "Random sample minimum", scores: scores(best, true), features: bestFeatures }], warning: str(root.selection_bias) };
}

export async function buildFeatureSnapshot(repo: string): Promise<FeatureSnapshot> {
  const sources: FeatureSnapshot["sources"] = [];
  try {
    const selection = await readSource(repo, FEATURE_SOURCES.selection); sources.push(selection.source);
    const config = await readSource(repo, FEATURE_SOURCES.config); sources.push(config.source);
    const best = await readSource(repo, FEATURE_SOURCES.best); sources.push(best.source);
    const plots = await readSource(repo, FEATURE_SOURCES.plots); sources.push(plots.source);
    const records = await readSource(repo, FEATURE_SOURCES.records); sources.push(records.source);
    const raw = obj(selection.value); const rawData = obj(raw.data); const bestData = obj(best.value);
    for (const source of [best.source, plots.source, records.source]) {
      const recorded = arr(raw.sources).map(obj).find(s => s.path === source.path);
      if (!recorded || recorded.sha256 !== source.sha256) throw new Error(`Dashboard source hash mismatch: ${source.path}`);
    }
    same(rawData.curves, obj(plots.value).figures, "Formal curves / Dashboard");
    same(rawData.winner, bestData.winner, "Formal winner / Dashboard");
    for (const value of arr(rawData.methods)) {
      const m = obj(value);
      for (const [field, origin] of [["training_cv", "by_method"], ["nested_cv", "nested_outer_assessment"], ["full_training_refit", "full_training_refits"]]) {
        same(m[field], arr(bestData[origin]).map(obj).find(v => v.method === m.method), `Formal ${field} / Dashboard`);
      }
    }
    for (const [key, value] of Object.entries(obj(obj(records.value).config))) same(value, obj(config.value)[key], `Fold results config ${key}`);
    const data = adaptFeatureSelection(selection.value, config.value);
    attachOuterResults(data, records.value);
    try {
      const exploration = await readSource(repo, FEATURE_SOURCES.exploration); sources.push(exploration.source);
      data.exploration = adaptFeatureExploration(exploration.value, data);
    } catch { data.exploration = { available: false, reason: "Phase 3D 成果缺少或格式／評估條件不符，無法呈現探索比較。" }; }
    return { schemaVersion: 1, status: "ready", sources, data };
  } catch (error) {
    return { schemaVersion: 1, status: "unavailable", sources, issues: ["Feature Selection 正式資料缺少或格式不符。", error instanceof Error ? error.message : "未知資料錯誤"] };
  }
}

export function parseFeatureSnapshot(value: unknown): FeatureSnapshot {
  const root = obj(value);
  if (root.schemaVersion !== 1) throw new Error("Snapshot 版本錯誤");
  for (const v of arr(root.sources)) { const s = obj(v); if (!Object.values(FEATURE_SOURCES).includes(str(s.path) as typeof FEATURE_SOURCES[keyof typeof FEATURE_SOURCES]) || !/^[a-f0-9]{64}$/.test(str(s.sha256))) throw new Error("來源格式錯誤"); }
  if (root.status === "unavailable") { if (!names(root.issues).length) throw new Error("缺少錯誤原因"); return value as FeatureSnapshot; }
  if (root.status !== "ready") throw new Error("Snapshot 狀態錯誤");
  const d = obj(root.data);
  for (const key of ["trainingRows", "holdoutRows", "outerFolds", "innerFolds", "candidateFeatures"]) count(d[key]);
  if (!Number.isInteger(d.gap) || num(d.gap) < 0) throw new Error("gap 錯誤");
  methodId(d.winner); str(d.selectionRule);
  const methods = arr(d.methods).map(obj);
  same(methods.map(m => methodId(m.id)).sort(), Object.keys(METHOD_NAMES).sort(), "六種方法");
  for (const m of methods) {
    const auto = m.id === "rfecv" || m.id === "lasso_cv";
    if (m.automatic !== auto || names(m.features).length !== count(m.finalK) || names(m.transformedFeatures).length !== m.finalK || count(m.finalK) > num(d.candidateFeatures)) throw new Error("Features 錯誤");
    scores(m.finalCV); scores(m.nestedCV); num(m.meanFinalK);
    if (arr(m.outerK).map(count).length !== d.outerFolds) throw new Error("Outer k 錯誤");
    const outerResults = arr(m.outerResults).map(obj);
    if (outerResults.length !== d.outerFolds) throw new Error("逐 fold 評分不完整");
    const foldScores = outerResults.map((r, index) => {
      if (count(r.fold) !== index + 1 || count(r.k) !== arr(m.outerK)[index] || count(r.k) > num(d.candidateFeatures)) throw new Error("逐 fold 編號／k 不一致");
      return scores(r.scores);
    });
    checkOuterMeans(foldScores, scores(m.nestedCV));
    const curve = arr(m.curve); if (auto && curve.length) throw new Error("自動 k 不得有曲線");
    if (!auto && !curve.length) throw new Error("固定 k 缺少曲線");
    for (const v of curve) { const p = obj(v); count(p.k); num(p.validFolds); if (p.scores !== null) scores(p.scores); }
  }
  const e = obj(d.exploration);
  if (e.available === false) str(e.reason);
  else if (e.available === true) {
    count(e.k); count(e.folds); count(e.sampleCount); str(e.warning);
    for (const v of arr(e.selectorRows)) { const r = obj(v); methodId(r.id); scores(r.scores); }
    for (const v of arr(e.fixedRows)) { const r = obj(v); str(r.label); scores(r.scores); if (names(r.features).length !== e.k) throw new Error("探索特徵數錯誤"); }
  } else throw new Error("缺少探索狀態");
  return value as FeatureSnapshot;
}
