# Safari module Worker CSP hotfix

Status: local Production Build acceptance PASS. The user confirmed Safari works locally on 2026-10-20; this is user-reported manual acceptance, not an automated Safari trace. Commit and Push to the experiment branch were authorized in the follow-up task. A new Preview must still be checked separately; no Production deployment is authorized.

## Cause and scope

The authenticated Preview's Worker response actually contains `worker-src 'none'`. `protocol.mjs` and `source-lock.mjs` both return 200 with JavaScript MIME. They are relative static imports in `public/browser-compute/worker.mjs`, so both resolve to the same origin; there is no missing module or rewritten CDN path.

The reported Safari violation names `worker-src` for these imports. WebKit's [WorkerModuleScriptLoader](https://github.com/WebKit/WebKit/blob/main/Source/WebCore/bindings/js/WorkerModuleScriptLoader.cpp) chooses a CSP check according to the module fetch destination: Worker destinations are checked against worker-src, while Script/worklet destinations use script-src. This is an engine-specific path that Chrome-only acceptance did not cover; do not assume every module fetch follows Chrome's path. The repository's current upstream source is supporting implementation evidence, not a claim that every Safari release behaves identically.

The main document controls creating the module Worker. Separately, the Worker's response CSP controls resources inside it. Changing only script-src would not resolve the observed worker-src denial. The correction changes only the Worker's response from `worker-src 'none'` to `worker-src 'self'`. It does not add a document-wide policy or allow arbitrary origins, blob/data Workers, wildcard sources or unsafe-eval. The existing script-src remains self plus the fixed Pyodide CDN and wasm-unsafe-eval; connect-src remains self plus that same CDN. This intentionally permits same-origin Worker destinations, including the static module graph.

## Manual Safari acceptance

The user chose manual Safari acceptance after native UI control timed out. Official safaridriver was not used, and no Safari security or automation preference was changed.

The local **Production Build** is running at http://127.0.0.1:3018/custom-dataset for this acceptance. This is not the old deployed Preview. The server has VERCEL=1 with no Gateway credentials, so Browser cannot silently depend on the local Python fallback. If the acceptance server has subsequently stopped, start the existing production build from web/ with:

```sh
env -u AEP_BACKEND_URL -u AEP_BACKEND_TOKEN -u AEP_SESSION_SECRET -u AEP_PUBLIC_ORIGIN -u AEP_PYTHON VERCEL=1 npm start -- --port 3018
```

1. Open the local production URL in Safari. Confirm Server Mode is selected by default and explicitly unavailable, then choose Browser Mode.
2. Select `web/tests/fixtures/browser-compute/missing.csv` (80 synthetic rows, three numeric candidates, with missing values). Select target `target`, and select all remaining numeric candidates `x1`, `x2`, `x3`. The dataset-size warning about the coursework's 13-column requirement is expected for this small test fixture.
3. Keep median imputation, StandardScaler, random split seed 42, 20% Holdout and five CV folds. Confirm the configuration, then press “在瀏覽器執行 Python 驗證”. Require a successful validation with Training 64 / Holdout 16; this exercises actual Pyodide initialization and official packages.
4. Under training, select RFE, set k=2, check the training consent and press “開始訓練與評估”. Require completed selected-feature, Training CV, Holdout RMSE/MAE/R² and prediction-chart results. Do not treat a successful page load or module HEAD response as a successful Python test.
5. In Safari's console/network inspector, require no worker-src/script-src/module import errors. Worker, protocol.mjs and source-lock.mjs must load as JavaScript; the Worker response must show worker-src self. If any later CDN module is denied, preserve the exact URL/directive and stop; do not broaden the policy preemptively.
6. Require no `/api/custom-dataset/*` upload during Browser work. Switch to Server Mode and confirm the previous dataset/results disappear. Record Safari version, validation/training outcome and any console errors.

The user has now confirmed local Safari Production success. Keep the steps above for regression and later public Preview acceptance: verify that the new deployed Worker response actually contains the new policy. The old immutable Preview cannot demonstrate the Hotfix. Commit/Push to the experiment branch is authorized; Production deployment remains excluded.

## Automated checks

The new `tests/browser-worker-csp.test.ts` asserts same-origin worker-src, unchanged fixed script/connect allowlists, no wildcard or unsafe-eval, Worker-only scope, nosniff and no-store. Actual test/production/Chrome results and user-confirmed local Safari status are recorded in `verification/m17-hotfix-safari-csp.json`. Historical experiments and previous release evidence are preserved.
