"""M6-A: real split validation, invalid input, version binding and no-fit boundary."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.custom_dataset_validation import validate_custom_dataset
from src.preprocessing_config import PreprocessingConfig


class CustomDatasetValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.csv = self.root / "data.csv"
        self.config = self.root / "config.json"
        self.csv.write_text("date,x,y\n" + "\n".join(f"2024-01-{i+1:02d}, {i*2},{i%7}" for i in range(30)), encoding="utf-8")
        self.payload = PreprocessingConfig(target_column="y", feature_columns=["x"],
                    column_types={"y": "numeric", "x": "numeric"}, split_strategy="chronological",
                    split_column="date", cv_gap=1).to_dict()
        self.write()

    def write(self):
        self.config.write_text(json.dumps(self.payload), encoding="utf-8")

    def run_validation(self, review=None):
        return validate_custom_dataset(self.csv, self.config, review)

    def test_actual_time_folds_and_no_fit_transform_or_model_write(self):
        original = {p: p.read_bytes() for p in self.root.iterdir()}
        with patch("sklearn.linear_model.LinearRegression.fit", side_effect=AssertionError("fit forbidden")), \
             patch("sklearn.preprocessing.StandardScaler.fit", side_effect=AssertionError("fit forbidden")), \
             patch("sklearn.impute.SimpleImputer.fit", side_effect=AssertionError("fit forbidden")), \
             patch("src.data_splitting.prepare_fold", side_effect=AssertionError("transform forbidden")):
            result = self.run_validation()
        self.assertEqual(result["status"], "valid", result)
        self.assertEqual(result["split"]["training_rows"], 24)
        self.assertEqual(result["split"]["holdout_rows"], 6)
        self.assertEqual(result["split"]["cv_folds"], [{"fold": i, "training_rows": 3+4*(i-1), "validation_rows": 4} for i in range(1,6)])
        self.assertFalse(result["training_started"])
        self.assertEqual(original, {p: p.read_bytes() for p in self.root.iterdir()})
        self.assertNotIn("2024-01", json.dumps(result))

    def test_unknown_config_and_unsupported_feature_expansion(self):
        for change in [{"unknown": 1}, {"datetime_strategy": "calendar"}, {"outlier_strategy": "remove"}, {"cv_splits": 11}, {"column_types": {"x": "categorical"}}]:
            with self.subTest(change=change):
                original = self.payload.copy(); self.payload.update(change); self.write()
                self.assertEqual(self.run_validation()["status"], "invalid")
                self.payload = original

    def test_missing_target_error_or_deterministic_drop(self):
        self.csv.write_text(self.csv.read_text().replace("2024-01-01, 0,0", "2024-01-01, 0,NA"))
        self.assertEqual(self.run_validation()["status"], "invalid")
        self.payload["missing_target"] = "drop"; self.write()
        result = self.run_validation(); self.assertEqual(result["status"], "valid", result)
        self.assertEqual(result["split"]["target_missing_dropped"], 1)
        self.assertEqual(result["split"]["training_rows"]+result["split"]["holdout_rows"], 29)

    def test_missing_feature_none_and_non_numeric_values(self):
        text = self.csv.read_text()
        for value in ["NA", "secret-invalid-value", "Infinity", "1e999", "9007199254740993"]:
            self.csv.write_text(text.replace("2024-01-01, 0,0", f"2024-01-01,{value},0"))
            self.payload["missing_numeric"] = "none"; self.write()
            result = self.run_validation(); self.assertEqual(result["status"], "invalid", result)
            self.assertNotIn("secret-invalid-value", json.dumps(result))
        self.csv.write_text(text.replace("2024-01-01, 0,0", "2024-01-01,NA,0"))
        self.payload["missing_numeric"] = "median"; self.write()
        self.assertEqual(self.run_validation()["status"], "valid")

    def test_duplicate_missing_invalid_or_zoned_dates(self):
        text = self.csv.read_text()
        for value in ["2024-01-02", "NA", "2024-02-30", "2024-01-01T00:00Z", "01/02/2024"]:
            self.csv.write_text(text.replace("2024-01-01", value))
            self.assertEqual(self.run_validation()["status"], "invalid")

    def test_impossible_gap_and_constant_training_target(self):
        self.payload["cv_gap"] = 100; self.write()
        self.assertEqual(self.run_validation()["status"], "invalid")
        self.payload["cv_gap"] = 1; self.write()
        lines = self.csv.read_text().splitlines()
        self.csv.write_text("\n".join(lines[:1]+[line.rsplit(",",1)[0]+",1" if i < 24 else line for i,line in enumerate(lines[1:])]))
        result = self.run_validation(); self.assertEqual(result["status"], "invalid")
        self.assertIn("常數", result["errors"][0]["message"])

    def test_review_binds_actual_csv_and_configuration(self):
        review = self.root / "review.json"
        document = {"schema_version": 1, "artifact_type": "custom_dataset_review", "dataset": {"sha256": hashlib.sha256(self.csv.read_bytes()).hexdigest()},
                    "preprocessing": self.payload.copy(), "review": {"confirmed": True, "training_started": False, "scope": "configuration_only"}}
        review.write_text(json.dumps(document)); self.assertTrue(self.run_validation(review)["review_checked"])
        self.payload["test_size"] = .25; self.write(); self.assertEqual(self.run_validation(review)["status"], "invalid")
        self.payload["test_size"] = .2; self.write(); self.csv.write_text(self.csv.read_text()+"\n")
        self.assertEqual(self.run_validation(review)["status"], "invalid")

    def test_bad_csv_and_json_do_not_crash(self):
        for raw in [b"\xff", b"a,a\n1,2", b"a,b\n1,2,3", b'a,b\n"open,2', b"a,b\n1,\0"]:
            self.csv.write_bytes(raw); self.assertEqual(self.run_validation()["status"], "invalid")
        for raw in ['{"target_column":"x","target_column":"y"}', '{"test_size":NaN}', '[1,2]', '{bad']:
            self.config.write_text(raw); self.assertEqual(self.run_validation()["status"], "invalid")

    def test_cli_returns_json_and_exit_status(self):
        args = [sys.executable, "-B", "-m", "src.custom_dataset_validation", "--csv", str(self.csv), "--config", str(self.config)]
        good = subprocess.run(args, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(good.returncode, 0, good.stderr); self.assertEqual(json.loads(good.stdout)["status"], "valid")
        self.csv.unlink()
        bad = subprocess.run(args, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(bad.returncode, 2); self.assertEqual(json.loads(bad.stdout)["status"], "invalid")

    def test_random_split_and_constant_validation_target_warning(self):
        self.payload.update(split_strategy="random", split_column=None, cv_gap=0); self.write()
        result = self.run_validation(); self.assertEqual(result["status"], "valid", result)
        self.assertEqual(len(result["split"]["cv_folds"]), 5)
        self.payload.update(split_strategy="chronological", split_column="date", cv_gap=1); self.write()
        lines=self.csv.read_text().splitlines(); self.csv.write_text("\n".join(lines[:25]+[line.rsplit(",",1)[0]+",2" for line in lines[25:]]))
        result=self.run_validation(); self.assertEqual(result["status"],"valid",result)
        self.assertTrue(any("holdout" in w and "R²" in w for w in result["warnings"]))


if __name__ == "__main__":
    unittest.main()
