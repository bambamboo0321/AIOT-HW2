export const AUTO_METHODS = ["rfecv", "lasso_cv"];
export const NESTED_METHODS = ["forward_sfs", "rfecv", "lasso_cv"];
export const ALPHA_GRID = [10000, 1000, 100, 10, 1, .1, .01, .001, .0001];
export function isAutoMethod(method) { return AUTO_METHODS.includes(method); }
export function isNestedMethod(method) { return NESTED_METHODS.includes(method); }
export function estimatedFits(method, k, p) {
    if (method === "forward_sfs")
        return k === p ? 1 : 3 * Array.from({ length: k }, (_, i) => p - i).reduce((s, n) => s + n, 0) + 1;
    if (method === "rfecv")
        return p === 1 ? 1 : 4 * p + 2;
    if (method === "lasso_cv")
        return 30;
    return method === "rfe" && k < p ? p - k + 2 : 1;
}
export function nestedSelectionError(method, k, p, rows, trainingRows, strategy, gap) {
    if (!isNestedMethod(method))
        return null;
    if (p > 25 || rows > 5000)
        return "RFECV／LassoCV 限 5,000 筆、25 個候選欄；請縮小資料或候選欄位。";
    if (method === "forward_sfs" && (p > 15 || rows > 2000))
        return "Forward SFS 較慢，限 2,000 筆、15 個候選欄；請縮小資料或候選欄位。";
    const bypass = (method === "forward_sfs" && k === p) || (method === "rfecv" && p === 1);
    if (!bypass && trainingRows.some(n => (strategy === "chronological" ? n - 3 * Math.floor(n / 4) - gap : n - Math.ceil(n / 3)) < 4))
        return "每個內部 CV Training 至少需要 4 筆；請增加資料、減少外層 folds 或 gap。";
    if (estimatedFits(method, k, p) * p * trainingRows.reduce((s, n) => s + n, 0) > 100_000_000)
        return "選欄運算量過大；請減少資料／候選特徵／folds 或縮小比較範圍。";
    return null;
}
export function parseSelectionDetails(value, method, rows, p, k) {
    if (!value || typeof value !== "object" || Array.isArray(value))
        throw new Error("缺少內部選模紀錄。");
    const v = value;
    const bypass = (method === "forward_sfs" && k === p) || (method === "rfecv" && p === 1);
    const criterion = bypass ? null : method === "lasso_cv" ? "mse" : "rmse";
    if (!isNestedMethod(method) || v.search_bypassed !== bypass || v.criterion !== criterion || JSON.stringify(v.alpha_grid) !== JSON.stringify(method === "lasso_cv" ? ALPHA_GRID : []))
        throw new Error("內部選模規則不符。");
    if (method === "lasso_cv" ? (typeof v.alpha !== "number" || !ALPHA_GRID.includes(v.alpha)) : v.alpha !== null)
        throw new Error("選中的 alpha 不符固定候選範圍。");
    if (!Array.isArray(v.inner_folds) || v.inner_folds.length !== (bypass ? 0 : 3))
        throw new Error("內部 CV folds 數量不符。");
    const inner = v.inner_folds.map((row, i) => { if (!row || typeof row !== "object" || Array.isArray(row))
        throw new Error("內部 CV 格式不符。"); const f = row; if (f.fold !== i + 1 || typeof f.training_rows !== "number" || !Number.isInteger(f.training_rows) || f.training_rows < 4 || typeof f.validation_rows !== "number" || !Number.isInteger(f.validation_rows) || f.validation_rows < 1 || f.training_rows + f.validation_rows > rows)
        throw new Error("內部 CV 切分筆數不符。"); return { fold: i + 1, training_rows: f.training_rows, validation_rows: f.validation_rows }; });
    return { criterion, alpha: v.alpha, alpha_grid: method === "lasso_cv" ? [...ALPHA_GRID] : [], inner_folds: inner, search_bypassed: bypass };
}
