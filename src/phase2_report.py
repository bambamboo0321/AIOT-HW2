"""以實際 EDA 與前處理摘要產生繁體中文 Phase 2 主報告。"""

import json

from .phase2_common import REPORT_DIR


def render_report(summary, eda):
    target = eda["target_statistics"]
    auto = eda["autocorrelation"]
    months = "\n".join(f"| {r['month']} | {r['count']} | {r['mean_wh']:.3f} | {r['std_wh']:.3f} |" for r in eda["monthly_pattern"])
    return f"""# Phase 2：Data Understanding & Configurable Preprocessing

## 1. Dataset Overview

正式資料為 `data/raw/KAG_energydata_complete.csv`：**19,735 rows × 29 columns**，數值型 Target 為 `Appliances`（Wh）。原始候選 Features 為 **28**（含 date），排除 date 後為 27，另排除 rv1／rv2 後為 25，符合原始特徵數規範。全部欄位無缺失值，完整重複列為 0。每欄原始 dtype、缺失與唯一數在 [data_profile.json](data_profile.json)，描述性統計在 [eda_statistics.json](eda_statistics.json)。

資料範圍為 2016-01-11 17:00:00 到 2016-05-27 18:00:00，相鄰 19,734 次間隔均為 600 秒。時區未標示，不推定時區。SHA-256：`{summary['dataset_sha256']}`，與 Phase 1 相同，兩份原始 CSV 均未修改。

接手時 HEAD 為 `8839430c2defa99a961976cb002cc94390a904c0`、分支 main，但 working tree 已存在未提交的 Baseline、Notebook、模型、指標與報告；並非僅有 Phase 1。本次保留它們，未執行 train_baseline 或 Notebook runner，也未產生新的正式模型測試結果。舊報告不屬於本次 Phase 2 驗收輸出；實際檔案差異見 [repository_review.json](repository_review.json)。

## 2. Target Distribution

Target mean **{target['mean']:.3f} Wh**、median **{target['50%']:.3f} Wh**、sample std **{target['std']:.3f} Wh**，min／max 為 {target['min']:.0f}／{target['max']:.0f} Wh，skewness **{target['skewness']:.6f}**，右側長尾。描述性 1.5 IQR 界線為 -25／175 Wh，標記高值 {eda['outlier_screening']['above_fence_count']:,} 筆（{eda['outlier_screening']['flagged_fraction']:.2%}），不能據此宣稱測量錯誤；預設 keep。

未套用 log Target、刪除高 Target 或使用 Test 分布決定轉換。未來若選 Target transformation，應另外在每個 training fold fit 並說明輸出反轉換及評估尺度。

## 3. Feature Analysis

數值欄位的 count／mean／std／quartiles／range／skewness／IQR 候選均由 CSV 計算，見 EDA JSON 的 numerical_summary 及 [逐欄直方圖](figures/15_numerical_distributions.png)。溫度、濕度、天氣、照明與隨機欄位量綱不同，數值 Feature 支援 standard／none；尚未宣稱最佳設定。

[UCI 定義](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction) 指出 rv1／rv2 為隨機測試變數。本機確認兩欄完全相同；其與 Target 的 Pearson r 均約 -0.011145。可用於負對照，預設建議排除兩欄且允許覆寫，不能因低全資料相關性宣稱其未來表現。date 保留作時間排序；可選擇轉 calendar 或 cyclical Features。預設選 25 原始欄，處理後 **{summary['feature_summary']['transformed_feature_count']}** 欄；加入 date 的 calendar 示例為 **{summary['calendar_example']['transformed_feature_count']}** 欄。

## 4. Correlation / Multicollinearity

數值輸入間 |Pearson r| ≥ 0.8 共 **{len(eda['highly_correlated_feature_pairs'])}** 組。rv1／rv2 r=1；T6／T_out r≈0.974787。標準化 27 欄輸入矩陣 rank={eda['multicollinearity']['matrix_rank']}，condition number≈{eda['multicollinearity']['condition_number']:.3e}，存在秩不足及共線性；沒有計算 VIF。LinearRegression 可求最小平方解，但係數不穩定，無法作因果解讀。與 Target 單變量 |r| 最高為 lights（約 0.197278），只描述本資料的線性關係，不是已驗證的最佳特徵。

完整係數矩陣、配對與圖在 [EDA 摘要](eda_summary.md) 和 [相關矩陣](figures/04_numerical_correlation_heatmap.png)。

## 5. Time-based Characteristics

Target 自相關：10 分鐘 **{auto[0]['Appliances']:.6f}**、1 小時 **{auto[1]['Appliances']:.6f}**、1 天 **{auto[2]['Appliances']:.6f}**、1 週 **{auto[3]['Appliances']:.6f}**；T1／T_out 的 lag-1 分別約 {auto[0]['T1']:.6f}／{auto[0]['T_out']:.6f}，相鄰觀測顯然有時間依賴。全資料每筆平均讀值於 18 時最高、03 時最低，分組不是每小時總耗能。

| 月份 | 觀測數 | 每筆平均 Wh | 樣本標準差 Wh |
| --- | ---: | ---: | ---: |
{months}

1／5 月非完整月份；只有單棟住宅約 4.5 個月，無法驗證全年季節性或跨住宅泛化。月份平均差異不能解釋為季節因果。原始來源有插值天氣資料；若應用要做真正未來預測，必須另外確定各 Feature 在預測當下可取得，不能把同時刻感測器估計任務誤稱為長期 forecast。

## 6. Preprocessing Decisions

[Wizard 規格](../../docs/preprocessing_wizard.md) 定義完整設定與理由，[energy_config.json](energy_config.json) 為此次可重現示例。預設 median、most_frequent、OneHot(ignore unknown)、StandardScaler、keep outliers、exclude datetime、chronological，並不聲稱這是最佳組合。

Numeric 分支為數值驗證 → train-only IQR 規則 → imputer → optional scaler；類別為字串正規化／保留缺失 → imputer → OneHot；日期為明確解析 → calendar／cyclical → numerical imputer／scaler。可覆寫型態與選擇／排除欄位，不直接將原始日期字串或 Unix Timestamp 當一般連續 Feature。

Target 缺失獨立 error／drop，永不補 Target。Feature drop 使用同步 X/y 遮罩並記錄 coverage；IQR remove 只作用 training。普通 sklearn Pipeline 無法同步刪 y，因此直接使用刪列設定會拒絕並要求 prepare_fold。未知類別轉為全零 OneHot；沒有「預設移除異常值」。

**k 定義為 Selector 輸出之轉換後數值欄位數**，OneHot／日期／flag 分別計數。原始欄到轉換名稱以 original_to_transformed 保存；預設原始選取 25／啟用 25／轉換 25，calendar 示例原始選取 26／啟用 26／轉換 29。名稱與完整 mapping 在 [preprocessing_summary.json](preprocessing_summary.json)。

## 7. Data Splitting Strategy

建議主要使用 chronological：training **{summary['training_rows']:,}**、test **{summary['test_rows']:,}**；training 為 {summary['training_time_range'][0]} 到 {summary['training_time_range'][1]}，test 為 {summary['test_time_range'][0]} 到 {summary['test_time_range'][1]}。API 按時間穩定排序，保留唯一原始索引；本版拒絕缺失／重複時間戳，不將同時刻資料拆至兩側。

Training 內用 TimeSeriesSplit（建議 5 folds、gap=6 筆作一小時示例）。高感測器自相關意味著 gap=6 並不保證獨立；gap 應由應用預測視窗預先訂定，敏感度只能在 training CV 評估。一般獨立資料支援 seeded random split 與 KFold；能源 random 結果不等同未來外推。未執行任何本階段正式 CV 模型評分。

## 8. Data Leakage Prevention

先切分，再在每個 fold 推測 schema、估計 imputer／IQR bounds／scaler／OneHot vocabulary；Validation／Test 只 transform。篩選資料列時回傳對齊的 y 與索引，validation 不依 outlier 被刪除。Predict 重用已 fit Pipeline。合成資料測試擾動 held-out Features／Target 不會改變 training 轉換狀態，並監看 fit 收到的索引排除當前 validation 與 outer test。

全資料 EDA 依本次範圍描述含未來 holdout 的資料，不將它用於模型調參、Target transformation 或正式排名。本地原本已有 holdout Performance 報告，因此 **不能再宣稱該 holdout 是從未查看的最終驗證資料**。後續比較只用 training CV，若需要嚴格全新最終泛化證據，應取得新時間資料或事先另外確立獨立外部驗證。沒有用本次 Test 挑設定。

未來 Selector 必須在相同訓練邊界內 fit；SFS／Lasso 若加內層 CV，該層也要重 fit preprocessing，不能對 outer-fit 矩陣直接計內層分數。這個 nested selector runner 留待 Phase 3，不偽裝為已實作。

## 9. Generic Dataset Support

提供唯讀 profile_csv／profile_dataframe、可覆寫型態、警告與 reasons、JSON config 驗證／保存、random／chronological split、fold iterator、Feature mapping／row audit。自由文字不自動編碼，ID／常數只提出建議，不永久修改原始資料。JSON 不保存資料列或整份類別字典；高基數 OneHot 可能增加成本，sparse 輸出仍需下游方法支援。

限制：不支援自由文字向量化、混合時區、自動 Unix 日期、群組時間切分、Target 變換、prediction 刪列 endpoint；尚無上傳／前端／部署。generic OneHot 的 p 可隨 fold 改變，Phase 3 要逐 fold 檢查 k，不能用全資料先固定類別字典。實際測試結果見 [verification.json](verification.json)。

## 10. Phase 3 Readiness

資料與前處理核心可作為 Phase 3 的起點；正式五種比較還需要實作 Selector、fold-local／nested-CV runner、計算預算、指標聚合與選定後的完整模型保存。本次不執行這些工作，也未宣稱任何 Feature／Algorithm 最佳。

建議候選：F-test（f_regression）、Mutual Information（mutual_info_regression）、Forward Sequential Selection（LinearRegression）、RFE（LinearRegression）、Lasso 係數排序。Lasso 只用於 selection，最終估計器維持 LinearRegression；alpha 僅由內層 training CV 設定。Lasso 的非零支援數不等於任意 k，若以 top-k 產生曲線，應保留零係數與並列的明確規則，不將人工補滿 k 說成自然稀疏解。依據 [sklearn Feature Selection 文件](https://scikit-learn.org/1.7/modules/feature_selection.html) 與 [Lasso 文件](https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.Lasso.html)。

預設 p=25，建議 k=1..25；calendar p=29 則 k=1..29，只有固定該前處理情境時才公平比較。所有通用資料的 k 都需 <= 當前 fold 的 p。

計算成本以下為公式估計，**未 Benchmark、未估算實測秒數**：25 個 k × 5 folds = 每演算法 125 次下游 LinearRegression fits，五種合計 625 次，另加 selection 與內層 CV。單條 forward path 的候選評估至多 25+24+...+1=325 次；若每步用 5-fold 內層 CV、5 個外層 folds，約 8,125 次候選模型 fits。若每個 k 都獨立重跑 forward，候選總量可達 5×5×sum(r²,r=1..25)=138,125 次，應規劃訓練區內可重用路徑；RFE／Lasso 也增加多次 fit。以上兩類計數各自描述下游評分與候選選擇成本，非已測總時間。

高相關特徵可能造成 RFE／Lasso 排名不穩，MI 也有估計隨機性；需固定 seed 與報告 fold 變異。高基數 OneHot 及 sparse 連續值的 MI 支援限制需在 Phase 3 明確處理，避免不必要 dense conversion。沒有新增測試集選模、曲線、Playground、Next.js、Vercel、PDF 或部署。**Phase 2 到此停止，等待人工驗收與後續授權。**
"""


def main():
    summary = json.loads((REPORT_DIR / "preprocessing_summary.json").read_text())
    eda = json.loads((REPORT_DIR / "eda_statistics.json").read_text())
    if summary["dataset_sha256"] != eda["dataset_information"]["sha256"]:
        raise ValueError("EDA 與前處理資料集 SHA-256 不同。")
    (REPORT_DIR / "phase2_report.md").write_text(render_report(summary, eda), encoding="utf-8")
    print("PASS: reports/phase2/phase2_report.md（僅引用實際統計，未訓練模型）。")


if __name__ == "__main__":
    main()
