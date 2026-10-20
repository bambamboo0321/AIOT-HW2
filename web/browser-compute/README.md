# Browser Compute

Custom Dataset defaults to unchanged M16 Server Mode. Browser Mode is explicit and never forwards data to M16 APIs or automatically falls back to Server Mode.

`lib/browser-compute/client.ts` owns one serial module Worker. Operations: initialize, validate, compare, train, predict, predictBatch, deleteModel, cancel, dispose. Jobs have unique IDs, immutable input snapshots, dataset/config/review binding and watchdogs. Formal reports use existing M16 parsers; browser metadata is a separate envelope.

The Worker loads fixed official Pyodide 314.0.7 directly. It checks Python 3.14.2, NumPy 2.4.6, pandas 3.0.2, SciPy 1.18.0, sklearn 1.8.0 and joblib 1.5.3. No pip install, WASM forwarding or user-supplied Python is used.

`npm run build-browser-compute` deterministically packages current `src/` sources and `bridge.py`, discovers relative imports, records every source SHA-256 and pins the complete manifest hash. It also transpiles the same TS client/result parsers for production browser acceptance. Predev/prebuild run packaging; generated assets must be refreshed together. Mixed revisions fail closed. Original algorithms are not edited.

The bridge only handles admission, files, progress and lifecycle. Comparison first checks the original aggregate budget, then calls the original complete comparison entry point for each method with unchanged k/CV, combining formal reports. Progress marks actual method boundaries, never invented fold or download percentages.

Models are original trusted joblib bundles in MEMFS. They expire after 30 minutes; at most three remain in a Worker. Original single/batch functions use frozen selected branches without refitting. There is no model upload or permanent storage. Cancel, timeout, changing CSV/configuration/mode or leaving the page terminate the Worker and erase its memory. Changing the training choice releases previous Browser models. After cancel, revalidate before training again.

Browser limits: 500 CSV rows, 10 numeric candidate features, 1 MiB per CSV. Server limits stay unchanged. Watchdogs: 60 seconds per method, 90 per job, 45 for explicit initialization. These are safety bounds, not an arbitrary-CSV completion guarantee. CV, sampling and auto-k behavior are never silently altered.

Only the Worker receives the new scoped CSP: same-origin modules, fixed CDN path, WASM compilation. No global unsafe-eval/unsafe-inline, backend CORS, SharedArrayBuffer or COOP/COEP change is introduced. Main document policy stays unchanged. CDN package caching is independent of CSV/model memory lifetime.

## Acceptance

- npm test / npm run typecheck / npm run lint / npm run build.
- `tests/browser-compute/prepare.py`: additional deterministic native references, using existing isolated B4 native environment. B4.6 baseline reference is reused read-only.
- `tests/browser-compute/run-browser.mjs`: production browser acceptance against the actual packaged client. Test fixture data is passed from runner to browser, not published with Python assets.
- `verification/m17-b5-*.json`: actual results. Local production acceptance does not substitute for later independent public HTTPS deployment acceptance.

## Release

Browser Mode is implemented but not publicly deployed. See [Vercel settings and HTTPS acceptance](../../docs/vercel_deployment_plan.md). A dynamic server-rendered advisory reports missing Gateway configuration without exposing credentials; configured does not mean healthy. M16 remains the transport authority. Browser-only Vercel needs no Python or Gateway secrets.

Committed synthetic fixtures are sufficient to run normal tests; `prepare.py` is optional native evidence regeneration and additionally requires the preserved B4.6 historical reference plus a separately installed native scientific environment. It is not an install/build/CI prerequisite. No historical experiment or native environment is included in the proposed release commit.
