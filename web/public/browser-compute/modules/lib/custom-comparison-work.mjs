import { estimatedFits, isAutoMethod } from './custom-selection-policy.mjs';
export const COMPARISON_LIMITS = { work: 100_000_000, warningWork: 10_000_000, timeoutMs: 180_000 };
// Same aggregate bound as the existing comparison UI and Python settings().
// This estimate applies only to comparison, never to dataset-wide admission.
export function comparisonWork(methods, ks, features, trainingRows) {
    const fits = methods.reduce((sum, m) => sum + (isAutoMethod(m) ? [null] : ks).reduce((n, k) => n + estimatedFits(m, k, features), 0), 0);
    return fits * features * trainingRows.reduce((sum, n) => sum + n, 0);
}
