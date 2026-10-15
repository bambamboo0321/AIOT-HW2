"""Phase 2 核心驗收：合成資料、訓練邊界、型態契約及可重現性。"""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from src.data_splitting import iter_preprocessed_folds, prepare_fold, split_dataset
from src.phase2_common import CSV_PATH, EXPECTED_SHA256
from src.preprocessing import ColumnValues, build_preprocessor, feature_summary
from src.preprocessing_config import PreprocessingConfig, recommend_config
from src.profiling import profile_csv, profile_dataframe, read_csv


def dense(values):
    return values.toarray() if hasattr(values, "toarray") else values


def fixture():
    n = 40
    return pd.DataFrame({"date": pd.date_range("2025-01-01", periods=n, freq="h"),
                         "x": np.arange(n, dtype=float), "z": np.arange(n, dtype=float) ** 2,
                         "group": ["a", "b"] * (n // 2), "y": np.arange(n, dtype=float) * 3 + 1},
                        index=np.arange(100, 100 + n))


def config(**kwargs):
    return PreprocessingConfig(target_column="y", feature_columns=["x", "group"], **kwargs)


class ProfilingTests(unittest.TestCase):
    def test_normal_csv_readonly_and_json_without_data_rows(self):
        frame = fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            frame.to_csv(path, index=False)
            before = path.read_bytes()
            profile = profile_csv(path)
            self.assertEqual(before, path.read_bytes())
        self.assertEqual((profile["rows"], profile["column_count"]), (40, 5))
        self.assertEqual(profile["duplicate_rows"], 0)
        self.assertIn("date", profile["candidates"]["datetime"])
        self.assertIn("group", profile["candidates"]["categorical"])
        self.assertNotIn("data", profile)
        json.dumps(profile, allow_nan=False)

    def test_missing_unique_outliers_constants_and_duplicates(self):
        frame = pd.DataFrame({"value": [1., 1., 1., 100., np.nan], "fixed": ["a"] * 5})
        result = profile_dataframe(frame)
        value, fixed = result["columns"]
        self.assertEqual((value["missing_count"], value["missing_fraction"], value["unique_count"]), (1, .2, 2))
        self.assertEqual(value["outliers"]["count"], 1)
        self.assertTrue(fixed["possible_constant"])
        self.assertEqual(result["duplicate_rows"], 2)
        self.assertEqual(fixed["category_count"], 1)

    def test_text_ids_and_high_cardinality_are_not_silently_categories(self):
        result = profile_dataframe(pd.DataFrame({"user_id": [f"user-{i}" for i in range(60)],
                                                 "notes": [f"Long free text description {i} " * 5 for i in range(60)],
                                                 "code": [f"00{i:04d}" for i in range(60)]}))
        self.assertEqual(result["candidates"]["datetime"], [])
        self.assertEqual(result["candidates"]["categorical"], [])
        self.assertTrue(result["columns"][0]["possible_id"])
        self.assertTrue(result["columns"][0]["high_cardinality"])
        self.assertEqual(result["columns"][1]["effective_type"], "text")

    def test_date_detection_does_not_parse_numbers_or_ambiguous_dates(self):
        data = pd.DataFrame({"epoch": [1000000000, 1000000010], "id": ["20240101", "20240102"],
                             "ambiguous": ["01/02/2024", "02/03/2024"], "iso": ["2024-01-01", "2024-01-02"]})
        self.assertEqual(profile_dataframe(data)["candidates"]["datetime"], ["iso"])

    def test_type_override_preserves_source_and_summary_is_finite_json(self):
        frame = pd.DataFrame({"n": [1., np.inf, np.nan], "s": ["1", "2", None]})
        before = frame.copy(deep=True)
        result = profile_dataframe(frame, {"s": "categorical"})
        self.assertEqual(result["columns"][1]["inferred_type"], "numeric")
        self.assertTrue(result["columns"][1]["overridden"])
        self.assertFalse(result["columns"][0]["target_candidate"])
        json.dumps(result, allow_nan=False)
        pd.testing.assert_frame_equal(before, frame)

    def test_bad_headers_and_empty_frame_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            path.write_text("x,x\n1,2\n")
            with self.assertRaisesRegex(ValueError, "標頭"):
                read_csv(path)
        for frame in [pd.DataFrame(), pd.DataFrame([[1, 2]], columns=["x", "x"])]:
            with self.assertRaises(ValueError):
                profile_dataframe(frame)

    def test_recommendations_are_reviewable_and_overridable(self):
        frame = fixture()
        suggestion = recommend_config(profile_dataframe(frame), "y")
        cfg = PreprocessingConfig.from_dict(suggestion["config"])
        self.assertEqual(cfg.split_strategy, "chronological")
        self.assertTrue(suggestion["reasons"])
        self.assertEqual(replace(cfg, split_strategy="random").split_strategy, "random")


class ConfigTests(unittest.TestCase):
    def test_json_roundtrip_and_unknown_keys(self):
        cfg = config()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            cfg.save(path)
            self.assertEqual(PreprocessingConfig.load(path), cfg)
            with self.assertRaisesRegex(ValueError, "CSV"):
                cfg.save(Path(directory) / "data.csv")
        with self.assertRaisesRegex(ValueError, "未知"):
            PreprocessingConfig.from_dict({**cfg.to_dict(), "typo": True})
        with self.assertRaisesRegex(ValueError, "必要"):
            PreprocessingConfig.from_dict({})

    def test_invalid_values_rejected(self):
        cases = [{"test_size": 1}, {"test_size": True}, {"test_size": float("nan")},
                 {"random_state": True}, {"random_state": -1}, {"cv_splits": 1}, {"cv_gap": -1},
                 {"missing_numeric": "magic"}, {"outlier_strategy": []}, {"iqr_multiplier": 0},
                 {"schema_version": True}, {"column_types": {"x": "bad"}},
                 {"split_strategy": "chronological"}, {"split_column": "y"},
                 {"column_types": {"y": "categorical"}}, {"datetime_formats": []}]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                config(**kwargs)
        for names in [[], ["x", "x"], ["y"], "x"]:
            with self.subTest(names=names), self.assertRaises(ValueError):
                PreprocessingConfig("y", names)

    def test_invalid_target_features_and_unused_override_rejected(self):
        frame = fixture()
        for cfg in [PreprocessingConfig("absent", ["x"]), PreprocessingConfig("y", ["absent"]),
                    config(column_types={"absent": "exclude"})]:
            with self.assertRaisesRegex(ValueError, "不存在"):
                split_dataset(frame, cfg)
        for values in [["bad"] * 40, [1.] * 40, [np.inf] * 40,
                       list(pd.date_range("2024-01-01", periods=40)), [True, False] * 20]:
            with self.subTest(values=str(values[:2])), self.assertRaisesRegex(ValueError, "Target"):
                split_dataset(frame.assign(y=values), config())


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.frame = fixture()
        self.cfg = config()
        self.split = split_dataset(self.frame, self.cfg)

    def fold(self, cfg=None):
        s = self.split
        return prepare_fold(s.X_train, s.y_train, s.X_test, s.y_test, cfg or self.cfg)

    def test_mean_median_and_no_imputation(self):
        train = pd.DataFrame({"x": [1., 2., 9., np.nan]}, index=[0, 1, 2, 3])
        for strategy, expected in [("mean", 4.), ("median", 2.)]:
            cfg = PreprocessingConfig("y", ["x"], missing_numeric=strategy, numerical_scaling="none")
            processor = build_preprocessor(train, cfg)
            np.testing.assert_allclose(processor.fit_transform(train)[:, 0], [1., 2., 9., expected])
        with self.assertRaisesRegex(ValueError, "缺失值"):
            build_preprocessor(train, replace(cfg, missing_numeric="none")).fit_transform(train)
        clean = train.dropna()
        np.testing.assert_allclose(build_preprocessor(clean, replace(cfg, missing_numeric="none")).fit_transform(clean)[:, 0], [1, 2, 9])

    def test_categorical_imputation_unknown_and_drop(self):
        train = pd.DataFrame({"group": ["a", "a", "b", None], "x": [1., 2., 3., 4.]})
        cfg = config(numerical_scaling="none")
        processor = build_preprocessor(train, cfg)
        values = dense(processor.fit_transform(train))
        np.testing.assert_array_equal(values[0, 1:], values[3, 1:])
        transformed = dense(processor.transform(pd.DataFrame({"group": ["new"], "x": [8.]})))
        np.testing.assert_array_equal(transformed[0, 1:], [0, 0])
        constant = build_preprocessor(train, replace(cfg, missing_categorical="constant"))
        constant.fit(train)
        self.assertTrue(any("UNKNOWN" in n for n in constant.get_feature_names_out()))
        dropped = build_preprocessor(train, replace(cfg, categorical_encoding="drop")).fit_transform(train)
        self.assertEqual(dropped.shape, (4, 1))

    def test_standard_scaler_is_fit_to_training_only(self):
        result = self.fold()
        branch = result.preprocessor.named_steps["columns"].named_transformers_["c0"]
        self.assertAlmostEqual(branch.named_steps["scale"].mean_[0], self.split.X_train.x.mean())
        self.assertEqual(branch.named_steps["scale"].n_samples_seen_, len(self.split.X_train))

    def test_calendar_cyclical_and_exclude(self):
        train = pd.DataFrame({"date": ["2024-01-01 00:00:00", "2024-01-02 06:00:00"], "x": [1., 2.]})
        cfg = PreprocessingConfig("y", ["date", "x"], datetime_strategy="calendar", numerical_scaling="none")
        processor = build_preprocessor(train, cfg)
        values = processor.fit_transform(train)
        np.testing.assert_allclose(values, [[1, 1, 0, 0, 1], [1, 2, 1, 6, 2]])
        summary = feature_summary(processor, cfg)
        self.assertEqual(len(summary["original_to_transformed"]["date"]), 4)
        cyclic = build_preprocessor(train, replace(cfg, datetime_strategy="cyclical"))
        transformed = cyclic.fit_transform(train)
        np.testing.assert_allclose(transformed[:, 0], [0., 1.], atol=1e-12)
        np.testing.assert_allclose(transformed[:, 1], [1., 0.], atol=1e-12)
        self.assertEqual(build_preprocessor(train, replace(cfg, datetime_strategy="exclude")).fit_transform(train).shape[1], 1)

    def test_explicit_date_format_missing_and_invalid(self):
        train = pd.DataFrame({"date": ["31/01/2024", None, "02/02/2024"]})
        cfg = PreprocessingConfig("y", ["date"], column_types={"date": "datetime"}, datetime_strategy="calendar",
                                  datetime_formats={"date": "%d/%m/%Y"})
        processor = build_preprocessor(train, cfg).fit(train)
        self.assertTrue(np.isfinite(processor.transform(train)).all())
        with self.assertRaisesRegex(ValueError, "日期"):
            processor.transform(pd.DataFrame({"date": ["impossible"]}))
        with self.assertRaisesRegex(ValueError, "Timestamp"):
            build_preprocessor(pd.DataFrame({"date": [1, 2]}), cfg).fit(pd.DataFrame({"date": [1, 2]}))

    def test_type_override_and_failed_numeric_conversion(self):
        train = pd.DataFrame({"x": ["001", "002", None]})
        cfg = PreprocessingConfig("y", ["x"], column_types={"x": "numeric"}, numerical_scaling="none")
        processor = build_preprocessor(train, cfg).fit(train)
        np.testing.assert_allclose(processor.transform(train)[:, 0], [1, 2, 1.5])
        for value in ["bad", np.inf]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                processor.transform(pd.DataFrame({"x": [value]}))

    def test_free_text_and_zero_active_features_rejected(self):
        train = pd.DataFrame({"notes": ["a" * 100, "b" * 100]})
        with self.assertRaisesRegex(ValueError, "自由文字"):
            build_preprocessor(train, PreprocessingConfig("y", ["notes"]))
        with self.assertRaisesRegex(ValueError, "沒有可用"):
            build_preprocessor(train, PreprocessingConfig("y", ["notes"], column_types={"notes": "exclude"}))

    def test_row_removal_cannot_be_silently_used_in_plain_pipeline(self):
        for cfg in [replace(self.cfg, outlier_strategy="remove"), replace(self.cfg, missing_numeric="drop"),
                    replace(self.cfg, missing_categorical="drop")]:
            with self.subTest(config=cfg), self.assertRaisesRegex(ValueError, "prepare_fold"):
                build_preprocessor(self.split.X_train, cfg)

    def test_all_missing_features_keep_stable_names(self):
        train = pd.DataFrame({"x": [np.nan] * 4, "group": [None] * 4})
        cfg = config(column_types={"group": "categorical"})
        processor = build_preprocessor(train, cfg).fit(train)
        transformed = processor.transform(train)
        self.assertEqual(transformed.shape[1], len(processor.get_feature_names_out()))
        self.assertTrue(np.isfinite(dense(transformed)).all())
        self.assertEqual(processor.transform(pd.DataFrame({"x": [5.], "group": ["new"]})).shape[1], transformed.shape[1])

    def test_iqr_clip_flag_and_keep_are_learned_from_training(self):
        train = pd.DataFrame({"x": [0., 1., 2., 3.]})
        valid = pd.DataFrame({"x": [1000.]})
        for strategy, count in [("keep", 1), ("clip", 1), ("flag", 2)]:
            cfg = PreprocessingConfig("y", ["x"], outlier_strategy=strategy, numerical_scaling="none")
            processor = build_preprocessor(train, cfg).fit(train)
            result = processor.transform(valid)
            self.assertEqual(result.shape[1], count)
            self.assertEqual(result[0, 0], 4.5 if strategy == "clip" else 1000.)
            if strategy == "flag":
                self.assertEqual(result[0, 1], 1.)
            iqr = processor.named_steps["columns"].named_transformers_["c0"].named_steps["outliers"]
            self.assertEqual(iqr.n_samples_seen_, 4)
            self.assertEqual(iqr.upper_[0], 4.5)

    def test_feature_names_mapping_and_count_match_actual_output(self):
        cfg = replace(self.cfg, outlier_strategy="flag", feature_columns=["x", "group", "date"], datetime_strategy="calendar")
        result = self.fold(cfg)
        names = result.summary["feature_names"]
        self.assertEqual(result.X_train.shape[1], len(names))
        self.assertEqual(len(names), 8)  # x + flag + 2 OHE + 4 calendar
        self.assertEqual(len(set(names)), len(names))
        self.assertEqual(sum(map(len, result.summary["original_to_transformed"].values())), len(names))

    def test_test_perturbations_cannot_change_training_state(self):
        original = self.fold()
        valid = self.split.X_test.copy()
        valid["x"] = 1000000.
        valid["group"] = "unseen"
        changed = prepare_fold(self.split.X_train, self.split.y_train, valid, self.split.y_test + 10000, self.cfg)
        np.testing.assert_array_equal(dense(original.X_train), dense(changed.X_train))
        self.assertEqual(original.summary, changed.summary)
        for step in ("impute", "scale"):
            first = original.preprocessor.named_steps["columns"].named_transformers_["c0"].named_steps[step]
            second = changed.preprocessor.named_steps["columns"].named_transformers_["c0"].named_steps[step]
            key = "statistics_" if step == "impute" else "mean_"
            np.testing.assert_array_equal(getattr(first, key), getattr(second, key))

    def test_sklearn_pipeline_clone_serialization_and_predict_reuse(self):
        processor = build_preprocessor(self.split.X_train, self.cfg)
        model = Pipeline([("preprocess", processor), ("regressor", LinearRegression())])
        model = clone(model).fit(self.split.X_train, self.split.y_train)
        expected = model.predict(self.split.X_test)
        original_fit = ColumnValues.fit
        with patch.object(ColumnValues, "fit", side_effect=AssertionError("predict must not fit")):
            np.testing.assert_array_equal(expected, model.predict(self.split.X_test))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pipeline.joblib"
            joblib.dump(model, path)
            np.testing.assert_array_equal(expected, joblib.load(path).predict(self.split.X_test))
        self.assertIs(ColumnValues.fit, original_fit)

    def test_reproducible_transforms(self):
        first, second = self.fold(), self.fold()
        np.testing.assert_array_equal(dense(first.X_train), dense(second.X_train))
        np.testing.assert_array_equal(dense(first.X_validation), dense(second.X_validation))
        self.assertEqual(first.summary, second.summary)


class SplittingTests(unittest.TestCase):
    def test_random_split_reproducible_no_overlap_and_alignment(self):
        frame, cfg = fixture(), config()
        original = frame.copy(deep=True)
        first, second = split_dataset(frame, cfg), split_dataset(frame, cfg)
        pd.testing.assert_index_equal(first.X_train.index, first.y_train.index)
        pd.testing.assert_index_equal(first.X_test.index, first.y_test.index)
        self.assertEqual(first.X_train.index.intersection(first.X_test.index).size, 0)
        self.assertEqual(set(first.X_train.index) | set(first.X_test.index), set(frame.index))
        pd.testing.assert_frame_equal(first.X_train, second.X_train)
        pd.testing.assert_frame_equal(frame, original)

    def test_chronological_sort_and_time_series_gap(self):
        frame = fixture().sample(frac=1, random_state=8)
        cfg = config(split_strategy="chronological", split_column="date", cv_splits=3, cv_gap=2)
        split = split_dataset(frame, cfg)
        self.assertLess(split.X_train.date.max(), split.X_test.date.min())
        for fold in iter_preprocessed_folds(split.X_train, split.y_train, cfg):
            last = frame.loc[fold.y_train.index, "date"].max()
            first = frame.loc[fold.y_validation.index, "date"].min()
            self.assertGreaterEqual(first - last, pd.Timedelta(hours=3))
            self.assertTrue(set(fold.y_validation.index).isdisjoint(split.X_test.index))

    def test_duplicate_or_missing_times_and_duplicate_index_rejected(self):
        cfg = config(split_strategy="chronological", split_column="date")
        for value in [pd.NaT, fixture().date.iloc[0]]:
            frame = fixture()
            frame.loc[frame.index[-1], "date"] = value
            with self.assertRaisesRegex(ValueError, "時間戳"):
                split_dataset(frame, cfg)
        frame = fixture()
        frame.index = [1] * len(frame)
        with self.assertRaisesRegex(ValueError, "索引"):
            split_dataset(frame, config())

    def test_target_missing_is_separate_and_never_imputed(self):
        frame = fixture()
        frame.loc[101, "y"] = np.nan
        with self.assertRaisesRegex(ValueError, "Target 含缺失值"):
            split_dataset(frame, config())
        split = split_dataset(frame, config(missing_target="drop"))
        self.assertEqual(split.dropped_target_indices, [101])
        self.assertNotIn(101, set(split.y_train.index) | set(split.y_test.index))
        self.assertFalse(split.y_train.isna().any())

    def test_drop_rows_retains_alignment_and_reports_validation_coverage(self):
        frame = fixture()
        frame.loc[100, "x"] = np.nan
        frame.loc[138, "group"] = None
        cfg = config(missing_numeric="drop", missing_categorical="drop", split_strategy="chronological", split_column="date")
        split = split_dataset(frame, cfg)
        fold = prepare_fold(split.X_train, split.y_train, split.X_test, split.y_test, cfg)
        self.assertEqual(fold.row_audit["train_missing_dropped_indices"], [100])
        self.assertEqual(fold.row_audit["validation_missing_dropped_indices"], [138])
        self.assertEqual(fold.X_train.shape[0], len(fold.y_train))
        self.assertEqual(fold.X_validation.shape[0], len(fold.y_validation))
        for cv_fold in iter_preprocessed_folds(split.X_train, split.y_train, replace(cfg, cv_splits=3)):
            self.assertEqual(cv_fold.X_train.shape[0], len(cv_fold.y_train))

    def test_iqr_removal_training_only_and_fold_local(self):
        frame = fixture()
        frame.loc[110, "x"] = 10000
        frame.loc[139, "x"] = 20000
        cfg = config(outlier_strategy="remove", split_strategy="chronological", split_column="date", numerical_scaling="none")
        split = split_dataset(frame, cfg)
        fold = prepare_fold(split.X_train, split.y_train, split.X_test, split.y_test, cfg)
        self.assertEqual(fold.row_audit["train_outlier_dropped_indices"], [110])
        self.assertIn(139, fold.y_validation.index)
        self.assertLess(fold.removal_bounds["x"]["upper"], 10000)
        np.testing.assert_array_equal(fold.y_train.to_numpy(), frame.loc[fold.y_train.index, "y"].to_numpy())

    def test_alignment_overlap_and_unprocessed_target_rejected(self):
        split = split_dataset(fixture(), config())
        with self.assertRaisesRegex(ValueError, "相同"):
            prepare_fold(split.X_train, split.y_train.iloc[::-1], split.X_test, split.y_test, config())
        with self.assertRaisesRegex(ValueError, "重疊"):
            prepare_fold(split.X_train, split.y_train, split.X_train, split.y_train, config())
        with self.assertRaisesRegex(ValueError, "Target"):
            prepare_fold(split.X_train, split.y_train * np.nan, split.X_test, split.y_test, config())

    def test_cv_fit_observes_only_current_training_rows(self):
        cfg = config(cv_splits=3)
        split = split_dataset(fixture(), cfg)
        original = ColumnValues.fit
        seen = []

        def spy(transformer, X, y=None):
            seen.append(set(X.index))
            return original(transformer, X, y)

        with patch.object(ColumnValues, "fit", autospec=True, side_effect=spy):
            for fold in iter_preprocessed_folds(split.X_train, split.y_train, cfg):
                for indices in seen:
                    self.assertTrue(indices <= set(fold.y_train.index))
                    self.assertFalse(indices & set(fold.y_validation.index))
                    self.assertFalse(indices & set(split.X_test.index))
                seen.clear()

    def test_no_imputation_validation_missing_fails_clearly(self):
        cfg = config(missing_numeric="none")
        split = split_dataset(fixture(), cfg)
        split.X_test.iloc[0, split.X_test.columns.get_loc("x")] = np.nan
        with self.assertRaisesRegex(ValueError, "缺失值"):
            prepare_fold(split.X_train, split.y_train, split.X_test, split.y_test, cfg)

    def test_raw_energy_csv_hash_matches_phase1(self):
        before = hashlib.sha256(CSV_PATH.read_bytes()).hexdigest()
        frame = read_csv(CSV_PATH)
        profile_dataframe(frame)
        self.assertEqual(hashlib.sha256(CSV_PATH.read_bytes()).hexdigest(), before)
        self.assertEqual(before, EXPECTED_SHA256)


if __name__ == "__main__":
    unittest.main()
