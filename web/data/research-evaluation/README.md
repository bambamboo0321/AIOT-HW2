# Versioned Model Evaluation snapshots

Automobile and Student saved Baseline/MI predictions only. Energy retains its original unchanged evaluation snapshot, adapter and explorer.

```sh
# Explicit reproduction from verified local artifacts; Python standard library only.
python3 web/scripts/export-study-evaluation.py --check
# Intentional regeneration after review:
python3 web/scripts/export-study-evaluation.py
```

The exporter reuses the Overview source adapter and validates its byte identity. It verifies the experiment artifact manifest and both original Holdout CSV hashes, matching each original source-row order, actual value, prediction and PI bound against results.json. It checks all 342 model predictions and bound pairs, recomputes MAE/RMSE/R² and PI coverage/mean/median/population width SD/counts, and preserves original float precision. It never fits, selects a model, synthesizes predictions from summary scores or calibrates intervals. Original source-row IDs are not copied into the public snapshot; the display uses Holdout ordinal positions.

Snapshots contain full saved target/prediction/PI values, source hashes and the linked Overview SHA-256. They do not contain raw input records, identifiers or serialized models. Production/build/dev never reads `experiments/` or runs the exporter; only allowlisted versioned snapshots/manifests and linked Overview files are traced. Source paths are provenance labels.

The runtime adapter rechecks dataset/source linkage, both models' shared actual observations, all numerical metrics, PI centering, widths and empirical counts against the official saved summaries. Missing predictions/PI use explicit unavailable states and empty chart rows; no Energy fallback or reverse engineering from metrics. Prediction arrays retain random Holdout order; PI visualization sorts a new view by saved prediction and labels rank explicitly as non-time. Scatter and residual charts use every saved point. Shared 20-bin histograms are descriptive counts, with no outlier clipping.

Dataset changes remount only the keyed study evaluation client, resetting metric/model/PI/residual controls. The client has no async fetch, fitting or Worker; Custom Dataset remains isolated.

Price unit remains the supplied source unit, currency unverified. G3 uses 0–20 points; G1/G2 availability is required and absences timing is uncertain. Negative PI bounds and bounds above 20 are retained. Nominal 95% is a pointwise OLS observation interval, not mean CI, not empirical calibration, not a future coverage guarantee. Training-only SVD/residual estimates do not account for selection uncertainty or heteroscedasticity. Training CV method selection remains locked before Holdout; a winner's CV can have selection optimism.
