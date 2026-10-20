const object = (v) => { if (!v || typeof v !== "object" || Array.isArray(v))
    throw new Error("驗證結果格式錯誤。"); return v; };
const string = (v) => { if (typeof v !== "string" || v.length > 4000)
    throw new Error("驗證訊息格式錯誤。"); return v; };
const integer = (v, min, max) => { if (typeof v !== "number" || !Number.isInteger(v) || v < min || v > max)
    throw new Error("驗證筆數格式錯誤。"); return v; };
const digest = (v) => { if (typeof v !== "string" || !/^[a-f0-9]{64}$/.test(v))
    throw new Error("驗證版本格式錯誤。"); return v; };
const strings = (v) => { if (!Array.isArray(v) || v.length > 200)
    throw new Error("驗證訊息格式錯誤。"); return v.map(string); };
export function parseValidationReport(value) {
    const v = object(value);
    if (v.schema_version !== 1 || v.scope !== "pretraining_validation_only" || v.training_started !== false)
        throw new Error("驗證結果的範圍或版本不符。");
    const common = { schema_version: 1, scope: "pretraining_validation_only", training_started: false };
    if (v.status === "invalid") {
        if (!Array.isArray(v.errors) || !v.errors.length || v.errors.length > 20)
            throw new Error("缺少驗證錯誤。");
        const errors = v.errors.map(e => { const row = object(e); const stage = string(row.stage); if (!["config", "csv", "data", "review", "split"].includes(stage))
            throw new Error("驗證階段不符。"); return { stage, message: string(row.message) }; });
        return { ...common, status: "invalid", errors };
    }
    if (v.status !== "valid")
        throw new Error("未知驗證結果。");
    const d = object(v.dataset);
    const s = object(v.split);
    const dataset = { sha256: digest(d.sha256), bytes: integer(d.bytes, 1, 25 * 1024 * 1024), rows: integer(d.rows, 1, 50_000), columns: integer(d.columns, 1, 200) };
    if (typeof v.review_checked !== "boolean" || !["random", "chronological"].includes(String(s.strategy)) || !Array.isArray(s.cv_folds) || s.cv_folds.length < 2 || s.cv_folds.length > 10)
        throw new Error("驗證切分格式錯誤。");
    const split = { strategy: s.strategy, training_rows: integer(s.training_rows, 2, dataset.rows), holdout_rows: integer(s.holdout_rows, 1, dataset.rows), target_missing_dropped: integer(s.target_missing_dropped, 0, dataset.rows), cv_folds: s.cv_folds.map((row, i) => { const f = object(row); if (f.fold !== i + 1)
            throw new Error("Fold 順序錯誤。"); return { fold: i + 1, training_rows: integer(f.training_rows, 2, dataset.rows), validation_rows: integer(f.validation_rows, 1, dataset.rows) }; }) };
    if (split.training_rows + split.holdout_rows + split.target_missing_dropped !== dataset.rows || split.cv_folds.some(f => f.training_rows + f.validation_rows > split.training_rows))
        throw new Error("驗證切分筆數不一致。");
    return { ...common, status: "valid", dataset, config_sha256: digest(v.config_sha256), review_checked: v.review_checked, split, warnings: strings(v.warnings), limits: strings(v.limits) };
}
