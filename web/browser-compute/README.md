# Browser Compute

Custom Dataset defaults to unchanged M16 Server Mode. Browser Mode is explicit and never forwards data to M16 APIs or automatically falls back to Server Mode.

`lib/browser-compute/client.ts` owns one serial model Worker and an isolated disposable comparison Worker. Operations: initialize, validate, compare, train, predict, predictBatch, deleteModel, cancel, dispose. Jobs have unique IDs, immutable input snapshots, dataset/config/review binding and watchdogs. Formal reports use existing M16 parsers; browser metadata is a separate envelope.

The Worker loads fixed official Pyodide 314.0.7 directly. It checks Python 3.14.2, NumPy 2.4.6, pandas 3.0.2, SciPy 1.18.0, sklearn 1.8.0 and joblib 1.5.3. No pip install, WASM forwarding or user-supplied Python is used.

`npm run build-browser-compute` deterministically packages current `src/` sources and `bridge.py`, discovers relative imports, records every source SHA-256 and pins the complete manifest hash. It also transpiles the same TS client/result parsers for production browser acceptance. Predev/prebuild run packaging; generated assets must be refreshed together. Mixed revisions fail closed. Original algorithms are not edited.

The bridge only handles admission, files, progress and lifecycle. Comparison first checks the original aggregate budget, then calls the original complete comparison entry point for each method with unchanged k/CV, combining formal reports. Progress marks actual method boundaries, never invented fold or download percentages.

Models are original trusted joblib bundles in MEMFS. They expire after 30 minutes; at most three remain in a Worker. Original single/batch functions use frozen selected branches without refitting. There is no model upload or permanent storage. Comparison cancel/timeout/error terminates only its disposable Worker and does not delete existing model bundles. Changing CSV/configuration/mode or leaving the page terminates both Workers; ordinary training/inference cancellation still clears its model Worker. Changing the training choice releases previous Browser models. After ordinary training/inference cancellation, revalidate before training again. Existing models remain subject to their original TTL; comparison does not extend it.

Browser training input limits: 2,000 CSV rows, 15 numeric candidate features, 3 MiB per CSV. Original Python per-method and aggregate comparison work guards remain unchanged; rejected comparisons do not prevent independently permitted training. Batch inference remains separately limited to 500 rows, 10,000 values and 2 MiB. Server limits stay unchanged. Watchdogs: comparison 180 seconds total (including its Worker loading); ordinary training 60 seconds per method / 90 per job; explicit initialization 45 seconds. Server comparison/proxy/UI timers are also 180 seconds. These are safety bounds, not an arbitrary-CSV completion guarantee. CV, sampling and auto-k behavior are never silently altered.

Only the Worker receives the new scoped CSP: same-origin modules, fixed CDN path, WASM compilation. No global unsafe-eval/unsafe-inline, backend CORS, SharedArrayBuffer or COOP/COEP change is introduced. Main document policy stays unchanged. CDN package caching is independent of CSV/model memory lifetime.

## Acceptance

- npm test / npm run typecheck / npm run lint / npm run build.
- `tests/browser-compute/prepare.py`: additional deterministic native references, using existing isolated B4 native environment. B4.6 baseline reference is reused read-only.
- `tests/browser-compute/run-browser.mjs`: production browser acceptance against the actual packaged client. Test fixture data is passed from runner to browser, not published with Python assets.
- `verification/m17-b5-*.json`: actual results. Local production acceptance does not substitute for later independent public HTTPS deployment acceptance.

## Release

Browser Mode is implemented but not publicly deployed. See [Vercel settings and HTTPS acceptance](../../docs/vercel_deployment_plan.md). A dynamic server-rendered advisory reports missing Gateway configuration without exposing credentials; configured does not mean healthy. M16 remains the transport authority. Browser-only Vercel needs no Python or Gateway secrets.

Committed synthetic fixtures are sufficient to run normal tests; `prepare.py` is optional native evidence regeneration and additionally requires the preserved B4.6 historical reference plus a separately installed native scientific environment. It is not an install/build/CI prerequisite. No historical experiment or native environment is included in the proposed release commit.

## 180-second comparison assessment

The shared hard work budget remains 100,000,000. Work is `p × sum(outer fold training rows) × sum(estimatedFits(method,k))`; it is not a time forecast. Work above 10,000,000 requires an additional explicit confirmation, validated both in the UI and Python. Confirmation cannot bypass the hard budget or original SFS 2,000-row/15-candidate and selector guards. No sampling, k substitution or CV reduction is performed.

An isolated bounded candidate-budget probe used synthetic numeric CSVs only. Native Python completed the 245,376,000-work stress case in about 83.33 s, but a Browser 500×10 six-method k=1..10 comparison reached its 180 s computation watchdog during SFS. Therefore the hard budget was not increased; the reported 238,714,560 user workload cannot be validated from its scalar alone. See `verification/m17-comparison-180.json`. Browser/OS process-wide termination can still lose all in-memory models; worker isolation protects against handled comparison cancellation/timeout/resource errors, not a killed renderer.


## SFS multi-k equivalence

Enabled by default only after native and actual Pyodide fold/k parity passed.

The comparison-only `SFSPrefixCache` reuses sklearn's own forward-search step results for the same outer Training fold. Every requested k still runs sklearn `fit()` and `get_support()`; no final unordered support list is truncated. Candidate evaluation, scoring and tie-breaking are delegated to the original sklearn method. The guarded private hook is audited for sklearn 1.7.2 / 1.8.0; unknown versions/settings fall back to the original selector. Revalidate this integration before upgrading sklearn.

Each request creates a separate cache for each outer Training fold. Fingerprints bind full numeric X, y, row order, preprocessing configuration and exact inner CV indices. Only the identical outer-fold preprocessor may be reused across k. Inner-fold preprocessing is never supplied from the outer-fold cache. Each k retains its independent downstream LR fit and original transforms. No cache survives a request/Worker, enters model persistence or changes ordinary training.

`compare_custom_dataset(..., reuse_sfs_prefixes=False)` preserves the original unoptimized path as a private Python regression reference. It is not an HTTP option. The conservative original workload estimate, 100-million hard cap, 10-million confirmation threshold and 180-second job watchdog remain unchanged.

`tests/test_custom_sfs_prefix.py` checks every fold/k, missing values, chronological gaps, exact ties, non-monotone selection order, Holdout independence, cache scope, unknown-version fallback and ordinary training. Independent benchmark inputs and raw records are in `experiments/m17-sfs-prefix/`; production acceptance is recorded separately under `verification/m17-sfs-prefix-*.json`. A split one-k original reference is explicitly labeled and is not a monolithic end-to-end timing.

Measured acceptance (synthetic 500×10, k=1..10): native SFS 24.11→5.10 s; actual Pyodide final SFS 58.61 s. The original complete-range timed attempt stopped at 180 s; its later full one-k reference sum was 272.21 s and includes repeated validation. The final production comparison, with a new Worker and initialization inside the unchanged 180-second budget, completed in 156.79 s. Cache/JIT state was uncontrolled; this is not a cold-profile or all-CSV guarantee. A 45-second initialization failure and its controlled retry are retained. See `verification/m17-sfs-prefix.json`.
