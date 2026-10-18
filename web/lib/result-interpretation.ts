import type { PredictionRow } from "./prediction-types.ts";

export function explainR2(value: number | null): string {
  if (value === null) return "實際值為常數或樣本不足，R² 無法定義。";
  if (value < 0) return "平方誤差高於以這批實際平均值作為固定預測的事後比較基準。";
  if (value === 0) return "平方誤差等於以這批實際平均值作為固定預測的事後比較基準。";
  return "平方誤差低於以這批實際平均值作為固定預測的事後比較基準；仍需搭配誤差大小與使用需求判斷。";
}

export function highestObservation(rows: readonly PredictionRow[]): PredictionRow {
  if (!rows.length) throw new Error("沒有可解讀的觀測");
  return rows.reduce((highest, row) => row.actual > highest.actual ? row : highest);
}
