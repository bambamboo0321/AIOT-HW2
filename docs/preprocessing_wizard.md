# Preprocessing Wizard Contract v1

本文件定義與前端框架無關的 Python／JSON 契約。本階段沒有 Dashboard、上傳介面、外部部署或正式 Feature Selection。模型固定為 `sklearn.linear_model.LinearRegression`。

## 資料剖析與使用者覆寫

`src.profiling.read_csv(path)` 以唯讀方式載入 CSV，檢查空資料、重複或空白標頭。回傳的 DataFrame 不會寫回原檔；pandas 預設 CSV 缺失標記規則仍適用。若前置零是重要識別資訊，載入時應傳入 `dtype={"code": "string"}`，避免 CSV dtype 推測先遺失前置零。

`profile_dataframe(frame, column_types={})`／`profile_csv(path, column_types={})` 回傳嚴格 JSON-compatible dict：

- `schema_version`、`rows`、`column_count`、`column_names`、`duplicate_rows`。
- `columns[]`：原始 dtype、`inferred_type`、`effective_type`、是否覆寫、缺失數／比例、唯一數／比例、類別數、數值統計、IQR 描述性候選數與門檻、常數／可能 ID／高基數旗標、Target 候選與警告。
- `candidates`：numeric／categorical／datetime／text 名單。

不輸出原始資料列、個別類別值清單或整份資料。NaN／Infinity 統計以 null 表示。IQR 只描述有限數值；無法轉換與無窮值另外警告，前處理會拒絕。ID 偵測是欄名與連續整數等啟發式，可能有誤判；高唯一率的連續數值不會只因唯一率高就當成 ID。短文字且低基數才推測為類別，其餘保留 text。原生 datetime 或能完整解析的 ISO 日期才推測為日期；不猜測普通數字、Unix Timestamp 或日／月順序。

`recommend_config(profile, target_column)` 回傳 `config`、`reasons`、`requires_user_review`，不直接執行轉換。這是未來 Wizard 的使用者審閱狀態，Python API 無需額外確認流程即可接受明確設定。型態覆寫可為 `numeric / categorical / datetime / text / exclude`，不代表已驗證可轉換；選到 text 會要求排除或覆寫。數值型 Target 也可由完整可轉換的數字字串指定；日期、bool、complex、非數值、全缺失、常數與無窮值 Target 會拒絕。Target 缺失只能報錯或排除，永不補值。

全資料 profiling 是描述與建議，不能把其平均數、分位數或類別字典直接當作轉換規則。正式實驗請先固定設定與 outer split；每個 CV training fold 重新推測尚未覆寫的型態，再 fit 所有資料相依規則。使用者也可明確覆寫 schema，讓各 fold 使用相同業務型態。

## JSON 設定

正式能源範例的完整可重現設定在 [energy_config.json](../reports/phase2/energy_config.json)。精簡範例：

```json
{
  "schema_version": 1,
  "target_column": "Appliances",
  "feature_columns": ["T1", "RH_1", "T_out"],
  "column_types": {},
  "missing_numeric": "median",
  "missing_categorical": "most_frequent",
  "missing_target": "error",
  "categorical_constant": "__UNKNOWN__",
  "categorical_encoding": "onehot",
  "numerical_scaling": "standard",
  "outlier_strategy": "keep",
  "iqr_multiplier": 1.5,
  "datetime_strategy": "exclude",
  "datetime_formats": {},
  "split_strategy": "chronological",
  "split_column": "date",
  "test_size": 0.2,
  "random_state": 42,
  "cv_splits": 5,
  "cv_gap": 6
}
```

| 欄位 | 支援值／規則 |
| --- | --- |
| target_column | 必要，數值迴歸 Target；不得同時選為 Feature 或 split_column |
| feature_columns | 必要，非空且不重複的原始欄名陣列；排除某欄可移出陣列或覆寫 exclude |
| column_types | 已存在欄名到型態的對照表；Target 覆寫僅允許 numeric |
| missing_numeric | mean / median / drop / none；也適用日期衍生值 |
| missing_categorical | most_frequent / constant / drop |
| missing_target | error / drop；在 outer split 前 deterministic 處理並記錄索引 |
| categorical_constant | 非空字串；預設 __UNKNOWN__。若原資料已有相同類別，會合併為同一類；可自行更換 |
| categorical_encoding | onehot / drop；未知類別 ignore，對該原始欄的所有 OneHot 值為 0 |
| numerical_scaling | standard / none；數值、日期衍生與 IQR flag 均可縮放，類別 OneHot 不縮放 |
| outlier_strategy | keep / flag / clip / remove；預設 keep |
| iqr_multiplier | 有限正數，預設 1.5；僅計算原始 numeric Features，不處理 Target |
| datetime_strategy | exclude / calendar / cyclical |
| datetime_formats | 非 ISO 日期須覆寫 datetime 並明確提供 `strptime` format；不推測格式或時區 |
| split_strategy | random / chronological；預設 random，一般建議在有日期時提議 chronological |
| split_column | chronological 必填；可作排序欄而不納入 Feature |
| test_size | 0 < fraction < 1，預設 0.2；兩側必須非空 |
| random_state | 0 到 2**32-1 的整數；控制 random split／KFold |
| cv_splits | >=2 的整數，預設 5；資料不足時 sklearn 會清楚拒絕 |
| cv_gap | >=0 的整數；只適用 TimeSeriesSplit，單位為觀測列，預設 0 |
| schema_version | 僅支援整數 1 |

`PreprocessingConfig.from_dict` 拒絕未知 key、缺少必要值與不合法 enum／型態／範圍，回報 ValueError。`split_dataset` 再驗證欄位存在與 Target 實際資料。非數值 Feature 轉換失敗、無窮值、日期格式錯誤、篩選後零特徵或資料不足，均不會靜默改用其他策略。

## Pipeline 與資料列生命週期

```text
原始 DataFrame（保留唯一 row index）
  → 驗證設定／Target → Target 缺失 error 或 drop（永不 impute）
  → outer random split 或按時間排序切分
  → 僅 outer Training 內產生 CV folds
  → 每個 fold：Training 推測 schema → X/y 同步缺失刪列
               → Training-only IQR removal（選用）
               → Pipeline.fit_transform(Training)
               → 同一 Pipeline.transform(Validation)
  → 未來 Phase 3：fold-local Selector → LinearRegression
```

Pipeline 使用 ColumnTransformer，為每個原始欄建立獨立分支，remainder=drop：

- Numeric：驗證轉實數 → 訓練區 IQR keep／clip／flag → mean／median imputer 或缺失檢查 → StandardScaler 或略過。
- Categorical：轉字串並保留缺失 → most_frequent／constant imputer 或缺失檢查 → OneHotEncoder(handle_unknown="ignore")。
- Datetime：保守解析 → calendar 的 month／day／weekday(0=Monday)／hour，或 cyclical 的 hour_sin／hour_cos／week_sin／week_cos → numerical 缺失處理及縮放。cyclical hour 含分鐘／秒，week phase 含日內時間；不產生 Unix Timestamp、年或絕對時間趨勢。

每個 fold 的中位數、平均數、IQR bounds、scaler 與 OneHot vocabulary 都只在該 training fold 估計。clip 不重算 validation 分位數；flag 增加欄位，缺失原值的 flag 為 0，之後才補原值。IQR 在插補前估計，只納入有限觀測；全缺失欄位沒有 IQR 門檻。`keep_empty_features=True` 保持每個已選欄的穩定表示：numeric 全缺失訓練欄補 0；categorical most_frequent 的全空欄由 sklearn 保留固定空欄位類別（目前為 0），constant 則使用指定標記。摘要會發出警告，建議確認或排除；未知的新值仍依 handle_unknown 處理。

**會刪列的設定必須使用 `prepare_fold`／`iter_preprocessed_folds`。** 普通 sklearn Transformer 不能同步刪除 y，`build_preprocessor` 因此拒絕直接使用 drop／remove。`prepare_fold` 回傳轉換陣列、對齊後 y、已 fit preprocessor、Feature 摘要、刪列 audit 與 removal bounds；陣列第 i 列對應回傳 y.index[i]。刪列順序為 schema、缺失過濾、Training-only IQR removal、Pipeline fit。

Feature missing 的 drop 依固定規則分別作用於 training／validation，並記錄兩側刪除索引與覆蓋率；若評估時刪了缺失資料列，分數只代表剩餘資料，不得與全覆蓋分數不加說明地比較。IQR removal **只刪 training**，validation／test／predict 保留真實極端值，避免只評估容易的資料。每個 CV fold 自己計算 removal 門檻；不依 Target 刪除異常值或套用全資料 IQR。過濾後 training <2 或 validation 為空會報錯。未來使用 drop 策略的 prediction endpoint 也須回報缺失資料列的拒絕／接受遮罩，不可悄悄減少預測列數。

`missing_numeric=none` 只接受完全有限的實際輸入，遇到 NaN 會報錯，包含預測時的新缺失值。Pipeline 本身預測不刪列。輸出可能是 dense ndarray 或 sparse matrix；OneHot 使用 sparse，不能對任意高基數資料直接無限制轉 dense。

## Python 使用與重現

```python
from src.profiling import read_csv
from src.preprocessing_config import PreprocessingConfig
from src.data_splitting import split_dataset, iter_preprocessed_folds, prepare_fold

frame = read_csv("data/raw/KAG_energydata_complete.csv")
config = PreprocessingConfig.load("reports/phase2/energy_config.json")
split = split_dataset(frame, config)

# 僅執行前處理；本階段不加入 Feature Selector 或訓練正式模型。
for fold in iter_preprocessed_folds(split.X_train, split.y_train, config):
    print(fold.summary["transformed_feature_count"], fold.row_audit)

prepared = prepare_fold(split.X_train, split.y_train,
                        split.X_test, split.y_test, config)
# 預測一律重用 prepared.preprocessor.transform(new_X)，不可重新 fit。
```

只有純轉換（沒有 drop／remove）時，也可將 `build_preprocessor(training, config)` 放入 sklearn Pipeline，後接固定 LinearRegression。未知 schema 由傳入的 training 決定；若要嚴格 fold-local 推測型態，應使用上方 iterator，而非先用全部資料推測後再把固定 Pipeline 交給 CV。不要把全資料 `fit_transform` 的結果交給 cross_val_score。

`config.save(path.json)`／load 保存完整 v1 設定。能源範例另保存 dataset SHA-256、套件版本、split 索引 digest、時間範圍與欄位 mapping 於 preprocessing_summary.json，不含逐筆資料。重現需相同 CSV、設定及套件版本；設定 JSON 不包含已 fit 的轉換器。未來保存可預測模型時，應將已 fit Preprocessor、Selector、LinearRegression 與 metadata 一起保存；合成測試驗證 joblib roundtrip 與 predict 不再 fit。僅載入可信來源的模型檔。

## Feature Count k 與欄位對應

**k 一律指 Feature Selection 後的轉換數值欄位數**，不是原始 CSV 欄位數。每個 OneHot dummy、日期衍生與 IQR flag 各計 1。原始 Features 的 `13 < count < 40` 是資料集規範，與 k 分開顯示。

`feature_summary` 提供 `selected_original_feature_count`、`active_original_feature_count`、`transformed_feature_count`、`feature_names`、`original_to_transformed`、`warnings`、`k_definition`。名稱用每個原始欄的分支 cN 前綴避免衍生名稱相撞。例如 x／兩類 group／date calendar 可對應：

```json
{
  "x": ["c0__x"],
  "group": ["c1__group_a", "c1__group_b"],
  "date": ["c2__date__month", "c2__date__day", "c2__date__weekday", "c2__date__hour"]
}
```

此例選了 3 個原始欄，轉換後為 7 欄；排除欄在 mapping 中為空陣列。每個 fold 的 OneHot vocabulary 可不同，p 與 k 可行範圍也可能不同；Phase 3 要在各 fold 檢查 `1 <= k <= p_fold`，超出時標記不可評估，不可用全資料 vocabulary 強行補齊。Selector support mask 應套在該 fold 的 feature_names，再由 mapping 回溯來源。單獨選到某 dummy 合法，不暗示整個原始欄全選；若未來採整組原始欄 selection，必須另立 metric，不能混用同一 k 軸。

## Split／CV 與邊界

Random 使用 train_test_split(test_size, random_state)，訓練區內 KFold(shuffle=True)。Chronological 使用穩定排序後前 floor(n*(1-test_size)) 列 training，其餘 test；TimeSeriesSplit 擴張訓練窗、可設定 gap。split_column 即使不當 Feature 也要保留。API 要求唯一資料索引；本版為確保同時刻不跨界，拒絕重複或缺失時間戳，群組時間切分尚未實作。TimeSeriesSplit 要求輸入已按時間排序；irregular time 的 gap 單位仍是列，不保證等長時間窗，使用者需先確認採樣意義。

能源資料建議 chronological，cv_gap=6 只是一小時示例，不保證足夠消除自相關；應由應用預測視窗決定，並僅用 training CV 做敏感度評估。沒有建立 lagged Target、未來視窗標籤或 Target transformation；未來有重疊時間視窗時還需 purge／embargo。

Phase 3 的 SFS／Lasso 調參若有內層 CV，內層也必須重新 fit 全部 Preprocessing，不能對已用 outer training 全部資料 fit 的矩陣直接執行內層 CV，再把分數當作無洩漏結果。這個 selector／nested-CV runner 屬 Phase 3，尚未實作。本階段只提供正確的單層 fold 前處理基礎。

參考：[sklearn 資料洩漏說明](https://scikit-learn.org/1.7/common_pitfalls.html#data-leakage)、[時間序列 CV](https://scikit-learn.org/1.7/modules/cross_validation.html#time-series-split)。
