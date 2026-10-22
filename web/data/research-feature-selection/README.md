# Versioned Feature Selection research snapshots

Only Automobile and Student research evidence is stored here. Energy keeps its unchanged `data/feature-selection.json`, original curve explorer and Phase 3B/3D artifacts.

Explicit source export, Python standard library only, no fitting:

```sh
# From repository root, with the already verified supplied experiment artifacts present:
python3 web/scripts/export-study-feature-selection.py --check
# Intentionally regenerate after review:
python3 web/scripts/export-study-feature-selection.py
```

The exporter reuses the Overview export adapter and its existing source verification. It requires the linked Overview bytes to match their verified export; stores that Overview SHA-256; checks each experiment's recorded source hashes and Training-only locked method decision. For all twelve methods it checks selected features, fixed/auto configuration, five outer Training partitions, applicable three inner partitions and disjointness from Holdout, and recomputes CV mean/std from saved fold scores. No algorithms run, no original source files or experiment artifacts are changed.

Each JSON contains the six methods, full-Training final features/k, actual per-fold features/k/scores, recorded warnings, and CV mean/population std (`ddof=0`). No original data rows, models, Holdout scores, fabricated k curve or mean-k coordinate is included. Fixed methods only have the predeclared k=7 experiment. LassoCV/RFECV keep auto and different per-fold k; the final k is not the k used by every CV fold.

The comparison uses five-fold Training CV. SFS/RFECV use inner three-fold RMSE; LassoCV selects alpha by inner MSE before independent LR; F-test/MI/RFE use fixed k without inner k search. The Training CV winner was locked before Holdout evaluation, but its winning CV still has selection optimism. Holdout remains separate, with a link to the existing Overview.

Production reads allowlisted versioned JSON/manifest files and the linked Overview snapshots only. Source experiment paths are provenance labels. Neither build/dev nor runtime runs this exporter or depends on `experiments/`. Byte hashes, dataset identity, source linkage, protocol, final/fold selection counts and CV summaries are validated. Missing method/CV becomes explicit `unavailable` and null chart values, never zero, interpolation or Energy fallback. Dataset changes remount only the keyed study explorer, resetting method and metric state; Custom Dataset is untouched.
