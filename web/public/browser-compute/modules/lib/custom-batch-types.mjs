import { parsePredictionModel } from "./custom-prediction-types.mjs";
export const BATCH_LIMITS = { bytes: 2 * 1024 * 1024, rows: 500, cells: 10000 };
export function validateBatchRows(value, model) {
    if (!Array.isArray(value) || !value.length || value.length > BATCH_LIMITS.rows || value.length * model.features.length > BATCH_LIMITS.cells)
        throw new Error("批次超過 500 筆或 10,000 個模型輸入值，或沒有資料。");
    for (const [i, row] of value.entries()) {
        if (!Array.isArray(row) || row.length !== model.features.length)
            throw new Error(`第 ${i + 1} 筆欄數與模型不符。`);
        if (row.some(n => n !== null && (typeof n !== "number" || !Number.isFinite(n) || Math.abs(n) > Number.MAX_SAFE_INTEGER)))
            throw new Error(`第 ${i + 1} 筆包含不合法數值。`);
        if (!row.some(n => n !== null))
            throw new Error(`第 ${i + 1} 筆的模型輸入全空白，請至少填一欄。`);
        if (model.missing_numeric === "none" && row.some(n => n === null))
            throw new Error(`第 ${i + 1} 筆有空白；此模型設定不補值。`);
    }
    return value;
}
function object(v) { if (!v || typeof v !== "object" || Array.isArray(v))
    throw new Error("批次結果格式不符。"); return v; }
export function parseBatchResult(value, modelValue, input) {
    const model = parsePredictionModel(modelValue);
    validateBatchRows(input, model);
    const v = object(value);
    const names = model.features;
    if (v.schema_version !== 1 || v.status !== "batch_predicted" || v.training_started !== false || v.target !== model.target || v.dataset_sha256 !== model.dataset_sha256 || v.config_sha256 !== model.config_sha256 || JSON.stringify(v.features) !== JSON.stringify(names) || v.row_count !== input.length || !Array.isArray(v.rows) || v.rows.length !== input.length)
        throw new Error("批次結果與目前模型或輸入版本不一致。");
    const subset = (value) => { if (!Array.isArray(value) || value.some(n => typeof n !== "string" || !names.includes(n)) || JSON.stringify(value) !== JSON.stringify(names.filter(n => value.includes(n))))
        throw new Error("批次欄位標記格式不符。"); return value; };
    const rows = v.rows.map((raw, i) => { const r = object(raw); if (r.position !== i + 1 || typeof r.prediction !== "number" || !Number.isFinite(r.prediction))
        throw new Error("批次筆序或預測值不符。"); const imputed = subset(r.imputed_features), outside = subset(r.out_of_range_features); const expected = names.filter((_, j) => input[i][j] === null); if (JSON.stringify(imputed) !== JSON.stringify(expected) || outside.some(n => imputed.includes(n)))
        throw new Error("批次補值與送出資料不一致。"); return { position: i + 1, prediction: r.prediction, imputed_features: imputed, out_of_range_features: outside }; });
    const filled = names.filter((_, j) => input.some(row => row[j] === null));
    if (!Array.isArray(v.imputation_values) || v.imputation_values.length !== filled.length)
        throw new Error("批次缺少補值統計。");
    const fills = v.imputation_values.map((raw, i) => { const r = object(raw); if (r.feature !== filled[i] || typeof r.value !== "number" || !Number.isFinite(r.value))
        throw new Error("批次補值統計不符。"); return { feature: filled[i], value: r.value }; });
    if (!Array.isArray(v.warnings) || v.warnings.length > 20 || v.warnings.some(n => typeof n !== "string" || n.length > 4000))
        throw new Error("批次提醒格式不符。");
    return { schema_version: 1, status: "batch_predicted", training_started: false, target: model.target, features: [...names], dataset_sha256: model.dataset_sha256, config_sha256: model.config_sha256, row_count: input.length, rows, imputation_values: fills, warnings: v.warnings };
}
