export const BROWSER_LIMITS = { bytes: 1024 * 1024, rows: 500, features: 10 };
export const RUNTIME = { version: "314.0.7", base: "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/",
    packages: { numpy: "2.4.6", pandas: "3.0.2", scipy: "1.18.0", "scikit-learn": "1.8.0", joblib: "1.5.3" } };
export class BrowserComputeError extends Error {
    code;
    constructor(code, message) { super(message); this.code = code; this.name = "BrowserComputeError"; }
}
