"""可 clone 的 sklearn 前處理；所有有學習的轉換僅在 fit 時估計。"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from .profiling import infer_type, parse_datetime


def resolve_types(training, config):
    missing = set(config.feature_columns) - set(training.columns)
    if missing:
        raise ValueError(f"Features 不存在：{sorted(missing)}")
    result = {name: (config.column_types[name] if name in config.column_types else infer_type(training[name]))
              for name in config.feature_columns}
    for name, kind in result.items():
        if kind == "text":
            raise ValueError(f"{name}: 自由文字無法作迴歸 Feature；請排除或明確覆寫型態。")
    return result


class ColumnValues(TransformerMixin, BaseEstimator):
    """固定語意轉換，不學習中位數、類別或日期解析格式。"""

    def __init__(self, kind="numeric", datetime_strategy="calendar", date_format=None):
        self.kind = kind
        self.datetime_strategy = datetime_strategy
        self.date_format = date_format

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        self.all_missing_ = bool(X.iloc[:, 0].isna().all())
        self.transform(X)
        return self

    def transform(self, X):
        check_is_fitted(self, "feature_names_in_")
        if list(X.columns) != list(self.feature_names_in_):
            raise ValueError("輸入欄位名稱或順序與 fit 時不同。")
        series = X.iloc[:, 0]
        if self.kind == "numeric":
            if pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_complex_dtype(series):
                raise ValueError(f"{series.name}: 不接受 datetime／complex 當一般數值 Feature。")
            try:
                array = pd.to_numeric(series, errors="raise").to_numpy(dtype=float, na_value=np.nan)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{series.name}: 無法轉換數值。") from error
            if np.isinf(array).any():
                raise ValueError(f"{series.name}: 數值含無窮值。")
            return array.reshape(-1, 1)
        if self.kind == "categorical":
            return series.map(lambda v: np.nan if pd.isna(v) else str(v)).to_numpy(dtype=object).reshape(-1, 1)
        dates = parse_datetime(series, self.date_format)
        if self.datetime_strategy == "calendar":
            return np.column_stack([dates.dt.month, dates.dt.day, dates.dt.dayofweek, dates.dt.hour]).astype(float)
        hour = dates.dt.hour + dates.dt.minute / 60 + dates.dt.second / 3600
        weekday = dates.dt.dayofweek + hour / 24
        return np.column_stack([np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
                                np.sin(2 * np.pi * weekday / 7), np.cos(2 * np.pi * weekday / 7)]).astype(float)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")
        name = str(self.feature_names_in_[0])
        if self.kind != "datetime":
            return np.asarray([name], dtype=object)
        suffixes = (["month", "day", "weekday", "hour"] if self.datetime_strategy == "calendar"
                    else ["hour_sin", "hour_cos", "week_sin", "week_cos"])
        return np.asarray([f"{name}__{suffix}" for suffix in suffixes], dtype=object)


class FiniteValues(TransformerMixin, BaseEstimator):
    """none/drop 模式：不靜默讓 NaN 傳給下游模型。"""

    def fit(self, X, y=None):
        self.transform(X)
        return self

    def transform(self, X):
        if pd.isna(X).any() or (np.issubdtype(np.asarray(X).dtype, np.number) and not np.isfinite(X).all()):
            raise ValueError("未處理的缺失值／無窮值；drop 須先用 prepare_fold 同步過濾 X/y。")
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


class IQRTransformer(TransformerMixin, BaseEstimator):
    """僅處理原始 numeric Features；不碰 Target、不刪列。"""

    def __init__(self, strategy="keep", multiplier=1.5):
        self.strategy = strategy
        self.multiplier = multiplier

    def fit(self, X, y=None):
        if self.strategy not in {"keep", "flag", "clip"}:
            raise ValueError("IQRTransformer 僅接受 keep/flag/clip。")
        array = np.asarray(X, dtype=float)
        self.n_features_in_ = array.shape[1]
        self.n_samples_seen_ = len(array)
        bounds = []
        for col in array.T:
            finite = col[np.isfinite(col)]
            if len(finite):
                q1, q3 = np.quantile(finite, [0.25, 0.75])
                bounds.append((q1 - self.multiplier * (q3 - q1), q3 + self.multiplier * (q3 - q1)))
            else:
                bounds.append((-np.inf, np.inf))
        self.lower_ = np.asarray([b[0] for b in bounds])
        self.upper_ = np.asarray([b[1] for b in bounds])
        return self

    def row_mask(self, X):
        check_is_fitted(self, "lower_")
        array = np.asarray(X, dtype=float)
        return ~((array < self.lower_) | (array > self.upper_)).any(axis=1)

    def transform(self, X):
        check_is_fitted(self, "lower_")
        array = np.asarray(X, dtype=float)
        if self.strategy == "clip":
            return np.clip(array, self.lower_, self.upper_)
        if self.strategy == "flag":
            flags = ((array < self.lower_) | (array > self.upper_)).astype(float)
            return np.column_stack([array, flags])
        return array.copy()

    def get_feature_names_out(self, input_features=None):
        names = np.asarray(input_features, dtype=object)
        return np.concatenate([names, [f"{name}__iqr_flag" for name in names]]) if self.strategy == "flag" else names


def build_preprocessor(training, config, *, column_types=None, rows_prepared=False):
    """建構未 fit Pipeline；schema 僅由 training 推測，預測時不再推測。"""
    if not rows_prepared and (config.outlier_strategy == "remove" or config.missing_numeric == "drop"
                              or config.missing_categorical == "drop"):
        raise ValueError("刪列設定必須使用 prepare_fold；一般 sklearn Pipeline 不能同步刪除 X/y。")
    types = resolve_types(training, config) if column_types is None else column_types
    transformers = []
    for index, name in enumerate(config.feature_columns):
        kind = types[name]
        if (kind == "exclude" or (kind == "categorical" and config.categorical_encoding == "drop")
                or (kind == "datetime" and config.datetime_strategy == "exclude")):
            continue
        if kind not in {"numeric", "categorical", "datetime"}:
            raise ValueError(f"{name}: 不支援型態 {kind}。")
        steps = [("values", ColumnValues(kind, config.datetime_strategy, config.datetime_formats.get(name)))]
        if kind == "categorical":
            if config.missing_categorical == "drop":
                steps.append(("impute", FiniteValues()))
            else:
                strategy = config.missing_categorical
                steps.append(("impute", SimpleImputer(strategy=strategy, fill_value=config.categorical_constant,
                                                       keep_empty_features=True)))
            steps.append(("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=True)))
        else:
            if kind == "numeric":
                strategy = "keep" if config.outlier_strategy == "remove" else config.outlier_strategy
                steps.append(("outliers", IQRTransformer(strategy, config.iqr_multiplier)))
            if config.missing_numeric in {"none", "drop"}:
                steps.append(("impute", FiniteValues()))
            else:
                steps.append(("impute", SimpleImputer(strategy=config.missing_numeric, keep_empty_features=True)))
            if config.numerical_scaling == "standard":
                steps.append(("scale", StandardScaler()))
        transformers.append((f"c{index}", Pipeline(steps), [name]))
    if not transformers:
        raise ValueError("排除／Encoding 設定後沒有可用 Features。")
    columns = ColumnTransformer(transformers, remainder="drop", sparse_threshold=1.0,
                                verbose_feature_names_out=True)
    return Pipeline([("columns", columns)])


def feature_summary(preprocessor, config):
    columns = preprocessor.named_steps["columns"]
    check_is_fitted(columns, "transformers_")
    names = list(preprocessor.get_feature_names_out())
    mapping = {name: [] for name in config.feature_columns}
    warnings = []
    for branch, transformer, source in columns.transformers_:
        if branch != "remainder":
            mapping[source[0]] = names[columns.output_indices_[branch]]
            if transformer.named_steps["values"].all_missing_:
                warnings.append(f"{source[0]}: Training 全缺失；numeric 補值為 0，categorical 使用固定空欄位表示。建議確認或排除。")
    return {"selected_original_feature_count": len(config.feature_columns),
            "active_original_feature_count": sum(bool(v) for v in mapping.values()),
            "transformed_feature_count": len(names), "feature_names": names,
            "original_to_transformed": mapping,
            "warnings": warnings,
            "k_definition": "轉換後的單一數值欄位數；OneHot、日期衍生與 IQR flag 各自計數。"}
