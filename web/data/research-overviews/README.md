# Versioned research Overview snapshots

`v1/` contains only the verified Automobile and Student Overview summaries. Energy still uses its unchanged `data/overview.json` and adapter. Other research pages are not connected for these two datasets.

Explicit reproduction from the supplied, verified local experiment artifacts (no training):

```sh
# From repository root; Python standard library only.
python3 web/scripts/export-study-overviews.py --check
# To intentionally regenerate after reviewing verified experiment artifacts:
python3 web/scripts/export-study-overviews.py
```

The exporter verifies original artifact hashes, source CSV hashes, split isolation, locked Training-only decision, per-fold CV means, recomputed Holdout metrics and PI coverage/width. It preserves full precision and never copies raw rows, models or per-observation predictions. The manifest records SHA-256 of each exported JSON file. Source paths in the snapshot are provenance labels only, not runtime filesystem dependencies.

Production reads only these allowlisted versioned JSON files and the manifest; it never imports or opens `experiments/`. Build/dev do not run this exporter. Missing, incompatible or corrupted snapshots fail closed with an explicit unavailable state; there is no fallback to Energy data. Missing model/PI artifacts use explicit unavailable states.

The snapshots describe a fixed split, not a new model fit. Random seed 42, 80/20 Holdout, outer 5-fold Training CV, inner 3-fold for applicable automatic/wrapper selectors, predeclared k=7 for fixed-k methods. MI method selection was locked by Training CV before Holdout evaluation; winner CV is still subject to selection optimism. Overview shows Baseline and the locked MI model only.

Automobile keeps 12 numeric and 2 fixed semantic binary predictors, fold-local median imputation and standard scaling. Price currency is unverified. Student uses G1/G2 and is conditional on their availability; absences timing is unspecified. PI is a pointwise OLS observation interval, not a mean confidence interval or a calibration guarantee.
