"""Fold-local memoization of sklearn SFS's unchanged forward-search steps.

Only multi-k comparison opts in. Each k still executes sklearn.fit/get_support;
we never truncate an unordered final feature set or mix CV folds. The identical
outer-fold preprocessor can be reused; inner-fold preprocessors remain separate.
The private sklearn hook is restricted to the two audited deployed versions.
"""
import hashlib
import inspect
import json

import numpy as np
import sklearn
from sklearn.feature_selection import SequentialFeatureSelector


class _PrefixSelector(SequentialFeatureSelector):
    def _get_best_new_feature_score(self, estimator, X, y, cv, current_mask, **params):
        # Metadata routing is not used by the formal caller. Preserve sklearn's
        # original behavior if it ever supplies additional estimator parameters.
        if params:
            return super()._get_best_new_feature_score(estimator, X, y, cv, current_mask, **params)
        key = current_mask.tobytes()
        cache = self._comparison_prefix
        if key not in cache.steps:
            # All candidate ordering, scores, NaN behavior and ties are decided
            # by sklearn itself, not by a replacement greedy implementation.
            cache.steps[key] = super()._get_best_new_feature_score(estimator, X, y, cv, current_mask)
            cache.misses += 1
        else:
            cache.hits += 1
        return cache.steps[key]


class SFSPrefixCache:
    """One invocation + one outer Training fold; steps and its own preprocessor."""
    def __init__(self):
        self.steps = {}
        self.signature = None
        self.hits = 0
        self.misses = 0
        self._outer_signature = None
        self._preprocessor = None

    @staticmethod
    def supported():
        if sklearn.__version__ not in {'1.7.2', '1.8.0'}:
            return False
        hook = getattr(SequentialFeatureSelector, '_get_best_new_feature_score', None)
        if hook is None:
            return False
        try:
            parameters = inspect.signature(hook).parameters
        except (TypeError, ValueError):
            return False
        return list(parameters) == ['self', 'estimator', 'X', 'y', 'cv', 'current_mask', 'params']

    def wrap(self, selector, X, y):
        if (not self.supported() or type(selector) is not SequentialFeatureSelector
                or selector.direction != 'forward' or selector.tol is not None
                or selector.scoring != 'neg_root_mean_squared_error' or selector.n_jobs != 1
                or type(selector.n_features_to_select) is not int):
            return selector  # Unknown versions/settings retain the original path.
        digest = hashlib.sha256()
        settings = {'config': selector.estimator.config.to_dict(),
                    'estimator': type(selector.estimator).__qualname__,
                    'scoring': selector.scoring, 'direction': selector.direction,
                    'tol': selector.tol, 'n_jobs': selector.n_jobs,
                    'version': sklearn.__version__, 'index': y.index.tolist()}
        digest.update(json.dumps(settings, sort_keys=True, separators=(',', ':')).encode())
        for value in [X, np.asarray(y), *[a for pair in selector.cv for a in pair]]:
            array = np.ascontiguousarray(value)
            digest.update(str((array.dtype.str, array.shape)).encode())
            digest.update(array.tobytes())
        signature = digest.digest()
        if self.signature is not None and self.signature != signature:
            raise ValueError('SFS prefix cache must not cross data, settings or CV folds.')
        self.signature = signature
        wrapped = _PrefixSelector(**selector.get_params(deep=False))
        wrapped._comparison_prefix = self
        return wrapped

    def preprocessor(self, X, y, config, values, fit):
        # This processor is fitted on this outer Training only. It is never
        # supplied to the inner CV estimator. Independent downstream LR fits
        # and every original transform/feature-support operation still run.
        digest = hashlib.sha256()
        digest.update(json.dumps({'config': config.to_dict(), 'index': y.index.tolist()},
                                 sort_keys=True, separators=(',', ':')).encode())
        for value in (values, np.asarray(y)):
            array = np.ascontiguousarray(value)
            digest.update(str((array.dtype.str, array.shape)).encode())
            digest.update(array.tobytes())
        signature = digest.digest()
        if self._outer_signature is not None and self._outer_signature != signature:
            raise ValueError('SFS preprocessor cache must not cross data, settings or CV folds.')
        self._outer_signature = signature
        if self._preprocessor is None:
            self._preprocessor, _ = fit(X, y, config)
        return self._preprocessor
