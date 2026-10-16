"""Phase 3C 固定設定比較：只 load/transform/predict，不 fit 模型或 Selector。"""

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import hashlib
from importlib.metadata import version
import json
import platform
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, RFE, RFECV, SequentialFeatureSelector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .phase2_common import ROOT, CSV_PATH, TARGET, BASELINE_FEATURES, EXPECTED_SHA256, load_dataset
from .prediction_interval import design_matrix, estimate_interval, interval_bounds, interval_metrics

PARENT = "4093655dd3db1395b090c99e3ea0a5bc0d64fddc"
OUTPUT = ROOT / "reports/phase3/phase3c"
FIGURES = ROOT / "reports/phase3/figures/phase3c"
SELECTED_FEATURES = ["T3", "T4", "T5", "RH_6", "T8", "T9"]
MODEL_FILES = {
    "baseline": "reports/phase2/models/chronological_baseline.joblib",
    "selected": "reports/phase3/results/models/final_linear_regression.joblib",
}
MODEL_HASHES = {
    "baseline": "9f9330522d7b52d6d279bf4b05f9a5f9123e1693e760dbd540910d74cd6a33ed",
    "selected": "8be97c965b56fed6d4a665703b44faf57508fd9a759b837d48cf3a2ad418eb12",
}
PLAN = {"alpha": .05, "nominal_coverage": .95, "training_rows": 15788, "holdout_rows": 3947,
        "split": "reuse Phase 2 chronological manifest, unchanged row order",
        "plot_window": {"holdout_start_position": 0, "consecutive_samples": 144},
        "rank_policy": "eps(float64)*max(n,p)*s_max, SVD of training design",
        "width_std_ddof": 0, "intervals_clipped": False,
        "method_or_parameter_selection_using_holdout": False,
        "interval_estimation_inputs": "training design, training y, saved training predictions only",
        "holdout_role": "single fixed descriptive evaluation; already viewed in Phase 2",
        "no_fit_guard": True, "thread_limit": 1}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def no_fitting():
    def forbidden(*args, **kwargs):
        raise RuntimeError("Phase 3C fixed-model evaluation forbids fit or feature selection")
    classes = (LinearRegression, StandardScaler, SimpleImputer, ColumnTransformer, Pipeline,
               SelectKBest, RFE, RFECV, SequentialFeatureSelector, Lasso, LassoCV)
    with ExitStack() as stack:
        for cls in classes:
            stack.enter_context(patch.object(cls, "fit", forbidden))
        for function in ("fit_selection", "fit_selected_regression"):
            stack.enter_context(patch("src.feature_selection." + function, forbidden))
        yield


@dataclass
class SavedModel:
    name: str
    artifact: object
    predictor: LinearRegression
    original_features: list
    transformed_features: list
    input_schema: list

    def features(self, frame):
        if self.name == "baseline":
            return frame.loc[:, self.original_features].to_numpy(dtype=float)
        return np.asarray(self.artifact.selection.transform(frame.loc[:, self.input_schema]), dtype=float)

    def predict(self, frame):
        data = frame.loc[:, self.input_schema]
        return np.asarray(self.predictor.predict(data) if self.name == "baseline" else self.artifact.predict(data))


def load_models():
    for name, path in MODEL_FILES.items():
        if sha256(ROOT / path) != MODEL_HASHES[name]:
            raise ValueError(f"{name}: model bytes differ from the accepted artifact")
    baseline = joblib.load(ROOT / MODEL_FILES["baseline"])
    bmodel = baseline["model"]
    metadata = baseline["training_metadata"]
    readable = json.loads((ROOT / "reports/phase2/models/chronological_baseline.json").read_text())
    if (type(bmodel) is not LinearRegression or not bmodel.fit_intercept or bmodel.positive
            or baseline["feature_names"] != BASELINE_FEATURES
            or list(bmodel.feature_names_in_) != BASELINE_FEATURES
            or bmodel.n_features_in_ != 25 or metadata["train"]["count"] != 15788
            or metadata["dataset_sha256"] != EXPECTED_SHA256
            or metadata["split_method"] != "chronological_first_80_percent"
            or metadata != readable["training_metadata"]):
        raise ValueError("Baseline feature, estimator or training metadata mismatch")
    np.testing.assert_array_equal(bmodel.coef_, [readable["coefficients"][n] for n in BASELINE_FEATURES])
    if bmodel.intercept_ != readable["intercept"]:
        raise ValueError("Baseline intercept mismatch")
    selected = joblib.load(ROOT / MODEL_FILES["selected"])
    smodel, selection = selected.predictor, selected.selection
    meta = selection.metadata
    raw = [n for n, mapped in meta["original_feature_mapping"].items() if mapped]
    expected_indices = [BASELINE_FEATURES.index(n) for n in SELECTED_FEATURES]
    available = list(selection.preprocessor.get_feature_names_out())
    schema = list(selection.preprocessor.feature_names_in_)
    best = json.loads((ROOT / "reports/phase3/results/best_features.json").read_text())
    if (best["winner"]["method"] != "mutual_information" or best["winner"]["requested_k"] != 6
            or type(smodel) is not LinearRegression or not smodel.fit_intercept or smodel.positive
            or smodel.n_features_in_ != 6 or meta["method"] != "mutual_information"
            or meta["requested_k"] != 6 or meta["selected_k"] != 6 or raw != SELECTED_FEATURES
            or meta["fit_indices"] != list(range(15788)) or meta["scaling"] != "standard"
            or selection.indices.tolist() != expected_indices
            or meta["selected_feature_indices"] != expected_indices
            or meta["selected_feature_names"] != [available[i] for i in expected_indices]
            or schema != BASELINE_FEATURES + ["date"]):
        raise ValueError("Selected model metadata differs from accepted MI + 6 features; stop")
    return {
        "baseline": SavedModel("baseline", baseline, bmodel, list(BASELINE_FEATURES), list(BASELINE_FEATURES), list(BASELINE_FEATURES)),
        "selected": SavedModel("selected", selected, smodel, raw, meta["selected_feature_names"], schema),
    }


def fixed_partitions(frame):
    manifest = json.loads((ROOT / "reports/phase2/split_indices.json").read_text())["chronological"]
    if (manifest["train"] != list(range(15788)) or manifest["test"] != list(range(15788, 19735))
            or len(frame) != 19735 or list(frame.index) != list(range(19735))):
        raise ValueError("fixed chronological partition or row order differs from Phase 2")
    train, holdout = frame.iloc[manifest["train"]], frame.iloc[manifest["test"]]
    if (not frame.date.is_monotonic_increasing or frame.date.duplicated().any()
            or train.date.max() >= holdout.date.min()):
        raise ValueError("chronological ordering is invalid")
    return train, holdout


def prediction_metrics(actual, prediction):
    y, pred = np.asarray(actual), np.asarray(prediction)
    if y.ndim != 1 or y.shape != pred.shape or not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError("metrics require aligned finite vectors")
    return {"mae": float(mean_absolute_error(y, pred)), "mse": float(mean_squared_error(y, pred)),
            "rmse": float(root_mean_squared_error(y, pred)), "r2": float(r2_score(y, pred))}


def validate_transformation(model, train, transformed):
    if transformed.shape != (15788, len(model.original_features)) or not np.isfinite(transformed).all():
        raise ValueError("saved feature transformation shape/values mismatch")
    if model.name == "baseline":
        np.testing.assert_array_equal(transformed, train[BASELINE_FEATURES].to_numpy())
        return {"space": "raw numeric features, no scaler or imputer", "matches_saved_input_order": True}
    columns = model.artifact.selection.preprocessor.named_steps["columns"]
    expected = []
    for name in SELECTED_FEATURES:
        i = BASELINE_FEATURES.index(name)
        branch = columns.named_transformers_[f"c{i}"]
        scaler, imputer = branch.named_steps["scale"], branch.named_steps["impute"]
        raw = train[name].to_numpy(dtype=float)
        if int(scaler.n_samples_seen_) != 15788:
            raise ValueError("saved scaler was not fitted on the expected Training rows")
        np.testing.assert_allclose(scaler.mean_, [raw.mean()], rtol=1e-13)
        np.testing.assert_allclose(scaler.var_, [raw.var()], rtol=1e-13)
        np.testing.assert_allclose(imputer.statistics_, [np.median(raw)], rtol=1e-13)
        expected.append((raw - scaler.mean_[0]) / scaler.scale_[0])
    np.testing.assert_allclose(transformed, np.column_stack(expected), rtol=1e-12, atol=1e-12)
    return {"space": "saved Phase 2 median imputer + StandardScaler, then saved MI support",
            "training_scaler_and_imputer_statistics_match": True,
            "matches_saved_input_order": True, "selection_or_preprocessing_refit": False}


def evaluate(frame=None, models=None):
    """回傳固定評估；所有設定先固定，兩個 interval state 均在取 Holdout y 前完成。"""
    frame = load_dataset() if frame is None else frame
    with no_fitting(), threadpool_limits(limits=1):
        models = load_models() if models is None else models
        train, holdout = fixed_partitions(frame)
        predictions, interval_results, metrics, states, transformations = {}, {}, {}, {}, {}
        historical = json.loads((ROOT / "reports/phase2/baseline_metrics.json").read_text())["baselines"]["chronological"]
        best = json.loads((ROOT / "reports/phase3/results/best_features.json").read_text())
        selected_training = next(r for r in best["full_training_refits"] if r["method"] == "mutual_information")
        for name, model in models.items():
            Xtrain = model.features(train)
            transformations[name] = validate_transformation(model, train, Xtrain)
            predicted_train = model.predict(train)
            manual = Xtrain @ model.predictor.coef_ + model.predictor.intercept_
            np.testing.assert_allclose(predicted_train, manual, rtol=1e-12, atol=1e-9)
            training_metrics = prediction_metrics(train[TARGET], predicted_train)
            expected = historical["train_metrics"] if name == "baseline" else selected_training
            for key in ("mae", "rmse", "r2"):
                np.testing.assert_allclose(training_metrics[key], expected[key], rtol=1e-12, atol=1e-9)
            states[name] = estimate_interval(design_matrix(Xtrain), train[TARGET], predicted_train, alpha=PLAN["alpha"])
            predictions[name] = model.predict(holdout)
            Xtest = model.features(holdout)
            np.testing.assert_allclose(predictions[name], Xtest @ model.predictor.coef_ + model.predictor.intercept_, rtol=1e-12, atol=1e-9)
            interval_results[name] = interval_bounds(states[name], design_matrix(Xtest), predictions[name])
            metrics[name] = {"training_metrics": training_metrics,
                             "model_name": "Baseline LinearRegression" if name == "baseline" else "MI + LinearRegression",
                             "feature_count": len(model.original_features), "selected_features": model.original_features,
                             "transformed_feature_names": model.transformed_features,
                             "coefficients_in_saved_model_space": model.predictor.coef_.tolist(),
                             "intercept_in_saved_model_space": float(model.predictor.intercept_),
                             "model_file": MODEL_FILES[name], "model_sha256": MODEL_HASHES[name]}
        # 以下才使用完整 Holdout y 評分；不回饋到任何模型／區間設定。
        actual = holdout[TARGET].to_numpy()
        intervals = {}
        for name in models:
            metrics[name]["holdout_metrics"] = prediction_metrics(actual, predictions[name])
            intervals[name] = {**interval_metrics(actual, interval_results[name]["lower"], interval_results[name]["upper"]),
                               "training_diagnostics": states[name].diagnostics,
                               "holdout_leverage_min": float(interval_results[name]["leverage"].min()),
                               "holdout_leverage_max": float(interval_results[name]["leverage"].max())}
        for metric in ("mae", "mse", "rmse", "r2"):
            np.testing.assert_allclose(metrics["baseline"]["holdout_metrics"][metric], historical["test_metrics"][metric], rtol=1e-12, atol=1e-9)
    periods = {"training": {"rows": len(train), "start": train.date.iloc[0].isoformat(), "end": train.date.iloc[-1].isoformat()},
               "holdout": {"rows": len(holdout), "start": holdout.date.iloc[0].isoformat(), "end": holdout.date.iloc[-1].isoformat()}}
    for item in metrics.values():
        item.update({"training_period": periods["training"], "holdout_period": periods["holdout"]})
    return {"plan": PLAN, "periods": periods, "model_metrics": metrics, "interval_metrics": intervals,
            "transformation_checks": transformations, "timestamp": holdout.date.to_numpy(),
            "row_indices": holdout.index.to_numpy(), "actual": actual, "predictions": predictions,
            "interval_predictions": interval_results}


def environment():
    return {"python": platform.python_version(), **{p: version(p) for p in (
        "numpy", "pandas", "scikit-learn", "scipy", "statsmodels", "patsy", "joblib", "matplotlib")}}


def main():
    from .phase3c_report import write_outputs
    OUTPUT.mkdir(parents=True, exist_ok=True)
    # 宣告設定先落盤，再讀 Holdout 評估；不因結果改寫 PLAN。
    (OUTPUT / "evaluation_plan.json").write_text(json.dumps({"plan": PLAN, "parent_commit": PARENT,
          "environment": environment(), "dataset_sha256": EXPECTED_SHA256,
          "models": MODEL_FILES, "model_sha256": MODEL_HASHES}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_outputs(evaluate())


if __name__ == "__main__":
    main()
