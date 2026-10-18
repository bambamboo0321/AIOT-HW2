export const CSV_LIMITS = { bytes: 25 * 1024 * 1024, rows: 50_000, columns: 200, cells: 1_000_000, fieldLength: 65_536 } as const;
export const MISSING_TOKENS = ["", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN", "<NA>", "N/A", "NA", "NULL", "NaN", "None", "n/a", "nan", "null"];
const missingTokens = new Set(MISSING_TOKENS);
export type ColumnKind = "numeric" | "categorical" | "datetime" | "text";
export type ColumnProfile = {
  name: string; kind: ColumnKind; missing: number; unique: number; possibleId: boolean;
  invalidNumeric: number; infinite: number; targetEligible: boolean;
  numeric: { count: number; min: number; max: number; mean: number | null; median: number; outliers: number } | null;
  dates: { count: number; unique: number; ordered: boolean } | null; warnings: string[];
};
export type CsvProfile = { rows: number; columns: ColumnProfile[]; duplicateRows: number; preview: string[][] };
export type DatasetUpload = CsvProfile & { fileName: string; bytes: number; sha256: string };
export type WizardDraft = { target: string; features: string[]; missingTarget: "error" | "drop"; missingNumeric: "median" | "mean" | "none"; scaling: "standard" | "none"; outliers: "keep" | "clip"; split: "random" | "chronological"; splitColumn: string; testSize: number; cvSplits: number; gap: number };
export type PreprocessingPlan = {
  schema_version: 1; target_column: string; feature_columns: string[]; column_types: Record<string, "numeric">;
  missing_numeric: WizardDraft["missingNumeric"]; missing_categorical: "most_frequent"; missing_target: WizardDraft["missingTarget"];
  categorical_constant: "__UNKNOWN__"; categorical_encoding: "onehot"; numerical_scaling: WizardDraft["scaling"];
  outlier_strategy: WizardDraft["outliers"]; iqr_multiplier: 1.5; datetime_strategy: "exclude"; datetime_formats: Record<string, string>;
  split_strategy: WizardDraft["split"]; split_column: string | null; test_size: number; random_state: 42; cv_splits: number; cv_gap: number;
};
export function emptyDraft(): WizardDraft { return { target: "", features: [], missingTarget: "error", missingNumeric: "median", scaling: "standard", outliers: "keep", split: "random", splitColumn: "", testSize: 0.2, cvSplits: 5, gap: 0 }; }
export function decodeCsv(buffer: ArrayBuffer): string {
  if (!buffer.byteLength || buffer.byteLength > CSV_LIMITS.bytes) throw new Error("CSV 必須為非空檔案，且不超過 25 MiB。");
  try { return new TextDecoder("utf-8", { fatal: true }).decode(buffer); } catch { throw new Error("CSV 不是有效的 UTF-8；請以 UTF-8 或 UTF-8 BOM 重新匯出。"); }
}
export function parseCsv(input: string): { headers: string[]; rows: string[][] } {
  if (new TextEncoder().encode(input).byteLength > CSV_LIMITS.bytes) throw new Error("CSV 超過 25 MiB。");
  const text = input.replace(/^\uFEFF/, "");
  if (text.includes("\0")) throw new Error("CSV 含不支援的 NUL 字元。");
  const records: string[][] = []; let row: string[] = []; let field = ""; let quoted = false; let closed = false; let touched = false; let cells = 0;
  function pushField() { row.push(field); field = ""; closed = false; if (row.length > CSV_LIMITS.columns) throw new Error("CSV 超過 200 欄。"); }
  function pushRow() {
    if (!touched && row.length === 0 && !field) return;
    pushField();
    if (records.length && row.length !== records[0].length) throw new Error(`第 ${records.length + 1} 筆 CSV 紀錄的欄數與標頭不同。`);
    if (records.length) { cells += row.length; if (cells > CSV_LIMITS.cells) throw new Error("資料超過 100 萬個儲存格，請縮小檔案。"); }
    records.push(row); if (records.length > CSV_LIMITS.rows + 1) throw new Error("CSV 超過 50,000 筆資料。");
    row = []; touched = false;
  }
  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    if (quoted) {
      if (char === '"') { if (text[i + 1] === '"') { field += '"'; i++; } else { quoted = false; closed = true; } }
      else field += char;
    } else if (char === ",") { touched = true; pushField(); }
    else if (char === "\n" || char === "\r") { pushRow(); if (char === "\r" && text[i + 1] === "\n") i++; }
    else if (closed) throw new Error("CSV 引號結束後只能接逗號或換行。");
    else if (char === '"') { if (field.length) throw new Error("CSV 引號只能出現在欄位開頭。"); quoted = true; touched = true; }
    else { field += char; touched = true; }
    if (field.length > CSV_LIMITS.fieldLength) throw new Error("單一欄位文字過長（上限 65,536 字元）。");
  }
  if (quoted) throw new Error("CSV 有未閉合的引號。");
  pushRow();
  const [headers, ...rows] = records;
  if (!headers || headers.some(h => !h.trim() || h !== h.trim())) throw new Error("欄名不得空白或含首尾空白；請先修正 CSV 標頭。");
  if (headers.some(h => h.length > 128)) throw new Error("欄名最多 128 個字元。");
  if (new Set(headers).size !== headers.length) throw new Error("CSV 欄名重複，請先修正標頭。");
  if (!rows.length) throw new Error("CSV 只有標頭或沒有資料列。");
  return { headers, rows };
}
export function isoTime(value: string): number | null {
  const m = /^(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2})(?::(\d{2})(?:\.(\d{1,3}))?)?)?$/.exec(value);
  if (!m) return null;
  const [y, month, day, h, min, s] = [Number(m[1]), Number(m[2]), Number(m[3]), Number(m[4] ?? 0), Number(m[5] ?? 0), Number(m[6] ?? 0)];
  const d = new Date(0); d.setUTCFullYear(y, month - 1, day); d.setUTCHours(h, min, s, Number((m[7] ?? "").padEnd(3, "0")));
  return d.getUTCFullYear() === y && d.getUTCMonth() === month - 1 && d.getUTCDate() === day && d.getUTCHours() === h && d.getUTCMinutes() === min && d.getUTCSeconds() === s ? d.getTime() : null;
}
function numericValue(value: string): number | null {
  const s = value.trim();
  if (/^[+-]?(?:inf|infinity)$/i.test(s)) return s.startsWith("-") ? -Infinity : Infinity;
  return /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/.test(s) ? Number(s) : null;
}
function quantile(values: number[], q: number): number {
  const pos = (values.length - 1) * q; const lo = Math.floor(pos); const fraction = pos - lo;
  return fraction === 0 ? values[lo] : values[lo] * (1 - fraction) + values[lo + 1] * fraction;
}
export function profileCsv(input: string): CsvProfile {
  const { headers, rows } = parseCsv(input);
  const columns = headers.map((name, index): ColumnProfile => {
    const values = rows.map(r => r[index]).filter(v => !missingTokens.has(v));
    const numbers = values.map(numericValue); const invalidNumeric = numbers.filter(v => v === null).length;
    const infinite = numbers.filter(v => v !== null && !Number.isFinite(v)).length;
    const finite = numbers.filter((v): v is number => v !== null && Number.isFinite(v));
    const dates = values.map(isoTime); const allDates = values.length > 0 && dates.every(v => v !== null);
    const unsafeInteger = finite.some(v => Number.isInteger(v) && !Number.isSafeInteger(v));
    const leadingId = values.some(v => /^(?:0\d+|\d{16,})$/.test(v));
    const uniqueStrings = new Set(values).size; const averageLength = values.length ? values.reduce((s, v) => s + v.length, 0) / values.length : 0;
    const kind: ColumnKind = allDates ? "datetime" : values.length && invalidNumeric === 0 && !leadingId && !unsafeInteger ? "numeric" : averageLength <= 80 && values.length > 0 && (uniqueStrings <= 20 || (uniqueStrings <= 50 && uniqueStrings / values.length <= 0.2)) ? "categorical" : "text";
    const sequence = kind === "numeric" && finite.length === rows.length && finite.length >= 3 && finite.every((v, i) => Number.isInteger(v) && (i === 0 || v - finite[i - 1] === 1));
    const possibleId = /(^id$|_id$|^id_|uuid|identifier|^index$)/i.test(name) || sequence || leadingId;
    const unique = kind === "numeric" ? new Set(numbers).size : uniqueStrings; const warnings: string[] = [];
    if (unique <= 1) warnings.push("常數或全缺失，請確認是否排除");
    if (possibleId) warnings.push("疑似 ID，請確認欄位意義");
    if (unsafeInteger) warnings.push("超出瀏覽器安全整數範圍，保留為文字並排除數值設定");
    if (leadingId) warnings.push("保留前置零／長整數字串，不自動轉數值");
    if (infinite) warnings.push(`${infinite} 筆非有限數值，不可作模型輸入`);
    if (kind === "categorical" || kind === "text") warnings.push("本階段尚不支援此型態作模型特徵");
    if (kind === "datetime" && new Set(dates).size < dates.length) warnings.push("有重複時間戳，不能使用目前的時間切分契約");
    let numeric: ColumnProfile["numeric"] = null;
    if (kind === "numeric" && finite.length) {
      const sorted = [...finite].sort((a, b) => a - b); const q1 = quantile(sorted, 0.25); const q3 = quantile(sorted, 0.75);
      const low = q1 - 1.5 * (q3 - q1); const high = q3 + 1.5 * (q3 - q1);
      const mean = finite.reduce((m, v, i) => m * (i / (i + 1)) + v / (i + 1), 0);
      numeric = { count: finite.length, min: sorted[0], max: sorted.at(-1)!, mean: Number.isFinite(mean) ? mean : null, median: quantile(sorted, 0.5), outliers: finite.filter(v => v < low || v > high).length };
    }
    return { name, kind, missing: rows.length - values.length, unique, possibleId, invalidNumeric, infinite,
      targetEligible: kind === "numeric" && unique > 1 && invalidNumeric === 0 && infinite === 0,
      numeric, dates: allDates ? { count: dates.length, unique: new Set(dates).size, ordered: dates.every((v, i) => i === 0 || v! >= dates[i - 1]!) } : null, warnings };
  });
  return { rows: rows.length, columns, duplicateRows: rows.length - new Set(rows.map(r => JSON.stringify(r))).size, preview: rows.slice(0, 5).map(row => row.map(value => value.length > 200 ? value.slice(0, 200) + "…" : value)) };
}
export function validatePlan(profile: CsvProfile, draft: WizardDraft): { errors: string[]; warnings: string[]; usableRows: number; trainRows: number; testRows: number; config: PreprocessingPlan } {
  const errors: string[] = []; const warnings: string[] = []; const target = profile.columns.find(c => c.name === draft.target);
  if (!target?.targetEligible) errors.push("請選擇具有變化、可轉為有限數值的目標欄位。");
  if (target?.missing && draft.missingTarget === "error") errors.push(`目標有 ${target.missing} 筆缺失；請選擇排除這些列或修正 CSV，不能補目標值。`);
  if (!draft.features.length || new Set(draft.features).size !== draft.features.length) errors.push("請選擇至少一個不重複的數值輸入欄位。");
  for (const name of draft.features) {
    const c = profile.columns.find(v => v.name === name);
    if (name === draft.target) errors.push("目標不得同時作為輸入特徵。");
    if (!c || c.kind !== "numeric" || c.infinite || c.invalidNumeric || c.unique <= 1) errors.push(`${name} 不是可用且具有變化的有限數值特徵。`);
    if (c?.missing && draft.missingNumeric === "none") errors.push(`${name} 有缺失，不能選擇不補值。`);
    if (c?.possibleId) warnings.push(`${name} 可能為 ID；請確認它是有意義、預測時可取得的輸入。`);
  }
  if (target?.possibleId) warnings.push("目標欄位疑似 ID，請確認要預測的是數量而非識別碼。");
  if (profile.columns.length - 1 < 13) warnings.push("原始候選輸入少於 13 欄；可檢查此資料，但不符合本作業目前的資料集欄數要求。此門檻與選取後 k 不同。");
  if (profile.duplicateRows) warnings.push(`有 ${profile.duplicateRows} 筆完全重複紀錄；目前保留，請確認是否來自重複量測。`);
  if (draft.split === "chronological") {
    const column = profile.columns.find(c => c.name === draft.splitColumn);
    if (!column?.dates || column.missing || column.dates.unique !== profile.rows || column.name === draft.target) errors.push("時間切分需要非缺失、有效且唯一的 ISO 日期欄位，且不得為目標。");
    if (draft.features.includes(draft.splitColumn)) errors.push("時間排序欄位不納入本階段的數值輸入。");
    if (column?.dates && !column.dates.ordered) warnings.push("原始時間未排序；未來 Python 流程將依時間穩定排序後切分。");
  } else if (profile.columns.some(c => c.kind === "datetime")) warnings.push("資料含日期欄位；若有時間依賴，隨機切分可能過度樂觀，請確認評估目的。");
  if (!Number.isFinite(draft.testSize) || draft.testSize <= 0 || draft.testSize >= 1) errors.push("保留測試比例必須介於 0 與 1。");
  if (!Number.isInteger(draft.cvSplits) || draft.cvSplits < 2 || draft.cvSplits > 10) errors.push("CV folds 必須為 2～10 的整數。");
  if (!Number.isInteger(draft.gap) || draft.gap < 0 || draft.gap > CSV_LIMITS.rows) errors.push("時間 gap 必須為 0～50,000 的整數。");
  if (draft.split === "random" && draft.gap !== 0) errors.push("隨機切分不適用時間 gap，請設為 0。");
  const usableRows = profile.rows - (draft.missingTarget === "drop" ? target?.missing ?? 0 : 0);
  const trainRows = draft.split === "chronological" ? Math.floor(usableRows * (1 - draft.testSize)) : usableRows - Math.ceil(usableRows * draft.testSize); const testRows = usableRows - trainRows;
  if (trainRows < 2 || testRows < 1) errors.push("切分後訓練至少需 2 筆、測試至少需 1 筆。");
  const foldSize = Math.floor(trainRows / (draft.cvSplits + 1));
  if (draft.split === "chronological" ? foldSize < 1 || trainRows - foldSize * draft.cvSplits - draft.gap < 2 : trainRows < draft.cvSplits || trainRows - Math.ceil(trainRows / draft.cvSplits) < 2) errors.push("訓練資料不足以支援目前的 CV folds／gap；請調整設定或增加資料。");
  const columnTypes = Object.fromEntries([...draft.features, draft.target].filter(Boolean).map(name => [name, "numeric" as const]));
  const config: PreprocessingPlan = { schema_version: 1, target_column: draft.target, feature_columns: draft.features, column_types: columnTypes, missing_numeric: draft.missingNumeric, missing_categorical: "most_frequent", missing_target: draft.missingTarget, categorical_constant: "__UNKNOWN__", categorical_encoding: "onehot", numerical_scaling: draft.scaling, outlier_strategy: draft.outliers, iqr_multiplier: 1.5, datetime_strategy: "exclude", datetime_formats: {}, split_strategy: draft.split, split_column: draft.split === "chronological" ? draft.splitColumn : null, test_size: draft.testSize, random_state: 42, cv_splits: draft.cvSplits, cv_gap: draft.gap };
  return { errors: [...new Set(errors)], warnings: [...new Set(warnings)], usableRows, trainRows, testRows, config };
}
