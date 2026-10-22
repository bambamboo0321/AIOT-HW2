# Fixed research Linear Regression models — Phase 3F-B

Six versioned models (Energy/Automobile/Student × Baseline/MI). This milestone adds artifacts and inference only. No Research Live Predictor UI or research-page change is included.

## Layout

- `v1/policy.json`: tolerance fixed before reconstruction: atol=1e-10, rtol=1e-10. Acceptance is `abs(actual-reference) <= atol + rtol*abs(reference)`. Relative-error reporting uses denominator `max(abs(reference), 1e-12)`; that floor does not change acceptance.
- `v1/manifest.json`: trusted bundled SHA-256 lock for all six public JSON files; includes the test fixture hash.
- `../../public/research-models/v1/*.json`: coefficients, intercept, ordered selected branches, exact saved scalers/imputers, category mappings, Training-only ranges, split fingerprints, source hashes, package versions and limitations. No row-level training or Holdout input data, model capabilities, secrets or pickled executable objects.
- `../../lib/fixed-models/core.ts`: pure synchronous inference and strict model parsing; async Web Crypto byte verification. No Python, Node, Worker, Gateway or API dependency.
- `../../lib/fixed-models/catalog.ts`: same-origin static JSON loading with a trusted bundled manifest. There is no fallback to another model.
- `../../tests/fixtures/fixed-models/v1/holdout.json.gz`: deterministic compressed test-only inputs, original saved predictions, native predictions and native extra-case predictions. Contains no subject IDs or source-row IDs. Not a public runtime asset or a production dependency.

## Reproduce

From repository root, using the original installed Python environment (no package installation):

```sh
.venv/bin/python -B web/scripts/export-fixed-models.py
node --disable-warning=ExperimentalWarning --experimental-strip-types web/scripts/verify-fixed-models.ts
npm --prefix web run typecheck
npm --prefix web run lint
npm --prefix web test
npm --prefix web run build
```

The offline exporter requires local original CSVs and historical experiment records, verifies dataset/split identity and fits **only four** LinearRegression models using frozen Baseline/MI feature names and the original preprocessing. It runs no selector, CV, k search or tuning. Selector fit guards raise on execution. Energy loads hash-locked saved models and has additional fit guards; it is never refitted. Reconstruction reproduces all historical Holdout predictions before publishing artifacts. Historical research files are never written. Generated files are deterministic for unchanged sources and environment; outputs remain FAIL on numerical mismatch, without loosening tolerance.

Production uses only the manifest, public model JSON and TypeScript modules. References to `experiments/` inside provenance strings are not file dependencies. The offline exporter and test fixture are not imported by application code.

## Inference contract

1. Select the explicit dataset/model/version via `fixedModelDescriptor` or `loadFixedModel`.
2. Verify model bytes, version, target, coefficient count/order, Training split, finite scalers/imputers and source metadata.
3. Pass only `model.featureOrder` inputs to `predictFixedModel`. The returned model object is a detached deep-frozen copy.
4. Encode known category labels (or explicit valid numeric codes), impute missing numeric inputs only when the saved model permits it, apply the saved scaler, then add the original linear intercept.

Energy Baseline has no imputation/scaling. Energy MI uses six selected frozen branches; the historical full preprocessor schema also contains an excluded date field, which is not a predictor. Automobile's full transformer historically requires 14 columns; the new artifact projects independent selected branches only. Its equivalence is tested against the original full transformed-matrix prediction for every Holdout row and extra cases.

Automobile category inputs are mandatory: gas=0/diesel=1, std=0/turbo=1. Other numeric blanks use saved Training medians. Currency and unverified per-feature units remain explicitly unverified.

Student fields use integers; ordinal allowed codes are explicit. `failures` is restricted to this supplied CSV version's documented 0–3 codes, not an assumed unrestricted count. Valid G1/G2 scores are 0–20, independently of narrower Training ranges. G1/G2 must be known for the model's stated use; a mathematical missing-value estimate does not establish that the real grades are known. Absences' accumulation cutoff is uncertain. Domain validation rejects illegal codes/fractions; valid inputs outside Training range produce warnings without changing inputs. Predictions are never clipped, including negative prices/energy and Student predictions outside 0–20.

No new-input PI, MAE, RMSE or R² is supplied. Holdout PI cannot be reused as a new-input interval. Future UI state must bind to dataset/model/version and remain separate from Custom Dataset's workers and temporary model store.
