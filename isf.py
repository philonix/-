import json
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
EXPORTS_DIR = BASE_DIR / "exports"

TARGET_SERVICE = "carts"
TRAIN_RUNS = ["run_normal_001", "run_normal_002", "run_normal_003"]
VALIDATION_RUNS = ["run_001", "run_003", "run_005"]
TEST_RUNS = ["run_002", "run_004", "run_006"]
CHALLENGE_RUNS = ["run_007", "run_008", "run_009", "run_010"]
ROLLING_WINDOW = 5
NEW_POD_POINTS = 4
FEATURE_COLUMNS = [
    "cpu_rate",
    "rolling_mean",
    "rolling_std",
    "rolling_median",
    "rolling_min",
    "rolling_max",
    "rolling_range",
    "rolling_q25",
    "rolling_q75",
    "rolling_iqr",
    "abs_rate_change",
    "ema_rate",
    "ema_deviation",
    "rate_zscore",
    "relative_to_mean",
    "memory_value",
    "memory_diff",
    "memory_change_rate",
    "memory_rolling_mean",
    "memory_rolling_std",
    "memory_rolling_median",
    "memory_rolling_min",
    "memory_rolling_max",
    "memory_rolling_range",
    "memory_rolling_q25",
    "memory_rolling_q75",
    "memory_rolling_iqr",
    "memory_zscore",
    "memory_relative_to_mean",
    "pod_age_points",
    "is_new_pod_window",
]
FEATURE_SETS = {
    "rate_only": ["cpu_rate"],
    "rate_plus_rolling": [
        "cpu_rate",
        "rolling_mean",
        "rolling_std",
        "abs_rate_change",
    ],
    "stat_enhanced": [
        "cpu_rate",
        "rolling_mean",
        "rolling_std",
        "rolling_median",
        "rolling_range",
        "rolling_iqr",
        "abs_rate_change",
        "rate_zscore",
    ],
    "cpu_memory": [
        "cpu_rate",
        "rolling_mean",
        "rolling_std",
        "abs_rate_change",
        "memory_value",
        "memory_diff",
        "memory_change_rate",
        "memory_rolling_mean",
        "memory_rolling_std",
        "memory_rolling_range",
        "memory_zscore",
    ],
    "full_features": FEATURE_COLUMNS,
}
PARAM_GRID = [
    {"n_estimators": 100, "contamination": 0.03},
    {"n_estimators": 100, "contamination": 0.05},
    {"n_estimators": 200, "contamination": 0.05},
    {"n_estimators": 200, "contamination": 0.08},
    {"n_estimators": 300, "contamination": 0.08},
]


def extract_service_name(pod_name):
    match = re.match(r"^(.*)-[a-z0-9]{9,10}-[a-z0-9]{5}$", pod_name)
    if match:
        return match.group(1)
    return pod_name


def read_run_export(run_id, metric_name="cpu"):
    run_dir = EXPORTS_DIR / run_id
    csv_path = run_dir / f"{metric_name}.csv"
    metadata_path = run_dir / "run_metadata.json"

    if not csv_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"缺少导出结果: {run_dir}。请先执行 runner.py 和 export_prometheus_run.py。"
        )

    df = pd.read_csv(csv_path).copy()
    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    return df, metadata


def prepare_metric_frame(df, value_column):
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["metric_dict"] = df["metric_dict"].apply(json.loads)
    df["pod"] = df["metric_dict"].apply(lambda item: item.get("pod", "unknown"))
    df["service"] = df["pod"].apply(extract_service_name)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    df = df.dropna(subset=["timestamp", "value"]).sort_values(["pod", "timestamp"])
    df = df[["timestamp", "datetime", "pod", "service", "value"]].rename(columns={"value": value_column})
    return df


def load_run_dataset(run_id):
    cpu_df, metadata = read_run_export(run_id, "cpu")
    memory_df, _ = read_run_export(run_id, "memory")
    cpu_df = prepare_metric_frame(cpu_df, "cpu_value")
    memory_df = prepare_metric_frame(memory_df, "memory_value")
    df = pd.merge(
        cpu_df,
        memory_df[["timestamp", "pod", "memory_value"]],
        on=["timestamp", "pod"],
        how="inner",
    )
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    df["service"] = df["pod"].apply(extract_service_name)
    df["run_id"] = run_id
    df = df.sort_values(["pod", "timestamp"]).reset_index(drop=True)
    return df, metadata


def load_multiple_runs(run_ids):
    datasets = []
    metadata_by_run = {}
    for run_id in run_ids:
        df, metadata = load_run_dataset(run_id)
        datasets.append(df)
        metadata_by_run[run_id] = metadata
    if not datasets:
        raise ValueError("没有可加载的数据集。")
    return pd.concat(datasets, ignore_index=True), metadata_by_run


def build_features(df, target_service):
    service_df = df[df["service"] == target_service].copy()
    if service_df.empty:
        raise ValueError(f"未在数据中找到服务 {target_service!r}")

    service_df["pod_age_points"] = service_df.groupby(["run_id", "pod"]).cumcount()
    service_df["delta_time"] = service_df.groupby(["run_id", "pod"])["timestamp"].diff()
    service_df["delta_cpu_value"] = service_df.groupby(["run_id", "pod"])["cpu_value"].diff()
    service_df["cpu_rate"] = service_df["delta_cpu_value"] / service_df["delta_time"]
    service_df["memory_diff"] = service_df.groupby(["run_id", "pod"])["memory_value"].diff()
    service_df["memory_change_rate"] = service_df["memory_diff"] / service_df["delta_time"]

    service_df = service_df[service_df["delta_time"] > 0].copy()
    service_df = service_df[service_df["delta_cpu_value"] >= 0].copy()
    service_df["memory_diff"] = service_df["memory_diff"].fillna(0.0)
    service_df["memory_change_rate"] = (
        service_df["memory_change_rate"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )

    service_df["rate_change"] = service_df.groupby(["run_id", "pod"])["cpu_rate"].diff()
    service_df["abs_rate_change"] = service_df["rate_change"].abs().fillna(0.0)
    service_df["rolling_mean"] = (
        service_df.groupby(["run_id", "pod"])["cpu_rate"]
        .transform(lambda series: series.rolling(ROLLING_WINDOW, min_periods=1).mean())
    )
    service_df["rolling_std"] = (
        service_df.groupby(["run_id", "pod"])["cpu_rate"]
        .transform(lambda series: series.rolling(ROLLING_WINDOW, min_periods=1).std())
        .fillna(0.0)
    )
    service_df["rolling_median"] = (
        service_df.groupby(["run_id", "pod"])["cpu_rate"]
        .transform(lambda series: series.rolling(ROLLING_WINDOW, min_periods=1).median())
    )
    service_df["rolling_min"] = (
        service_df.groupby(["run_id", "pod"])["cpu_rate"]
        .transform(lambda series: series.rolling(ROLLING_WINDOW, min_periods=1).min())
    )
    service_df["rolling_max"] = (
        service_df.groupby(["run_id", "pod"])["cpu_rate"]
        .transform(lambda series: series.rolling(ROLLING_WINDOW, min_periods=1).max())
    )
    service_df["rolling_q25"] = (
        service_df.groupby(["run_id", "pod"])["cpu_rate"]
        .transform(lambda series: series.rolling(ROLLING_WINDOW, min_periods=1).quantile(0.25))
    )
    service_df["rolling_q75"] = (
        service_df.groupby(["run_id", "pod"])["cpu_rate"]
        .transform(lambda series: series.rolling(ROLLING_WINDOW, min_periods=1).quantile(0.75))
    )
    service_df["rolling_range"] = service_df["rolling_max"] - service_df["rolling_min"]
    service_df["rolling_iqr"] = service_df["rolling_q75"] - service_df["rolling_q25"]
    service_df["ema_rate"] = (
        service_df.groupby(["run_id", "pod"])["cpu_rate"]
        .transform(lambda series: series.ewm(span=ROLLING_WINDOW, adjust=False).mean())
    )
    service_df["ema_deviation"] = service_df["cpu_rate"] - service_df["ema_rate"]
    std_denominator = service_df["rolling_std"].replace(0.0, np.nan)
    mean_denominator = service_df["rolling_mean"].abs().replace(0.0, np.nan)
    service_df["rate_zscore"] = (
        (service_df["cpu_rate"] - service_df["rolling_mean"]) / std_denominator
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    service_df["relative_to_mean"] = (
        service_df["cpu_rate"] / mean_denominator
    ).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    service_df["memory_rolling_mean"] = (
        service_df.groupby(["run_id", "pod"])["memory_value"]
        .transform(lambda series: series.rolling(ROLLING_WINDOW, min_periods=1).mean())
    )
    service_df["memory_rolling_std"] = (
        service_df.groupby(["run_id", "pod"])["memory_value"]
        .transform(lambda series: series.rolling(ROLLING_WINDOW, min_periods=1).std())
        .fillna(0.0)
    )
    service_df["memory_rolling_median"] = (
        service_df.groupby(["run_id", "pod"])["memory_value"]
        .transform(lambda series: series.rolling(ROLLING_WINDOW, min_periods=1).median())
    )
    service_df["memory_rolling_min"] = (
        service_df.groupby(["run_id", "pod"])["memory_value"]
        .transform(lambda series: series.rolling(ROLLING_WINDOW, min_periods=1).min())
    )
    service_df["memory_rolling_max"] = (
        service_df.groupby(["run_id", "pod"])["memory_value"]
        .transform(lambda series: series.rolling(ROLLING_WINDOW, min_periods=1).max())
    )
    service_df["memory_rolling_q25"] = (
        service_df.groupby(["run_id", "pod"])["memory_value"]
        .transform(lambda series: series.rolling(ROLLING_WINDOW, min_periods=1).quantile(0.25))
    )
    service_df["memory_rolling_q75"] = (
        service_df.groupby(["run_id", "pod"])["memory_value"]
        .transform(lambda series: series.rolling(ROLLING_WINDOW, min_periods=1).quantile(0.75))
    )
    service_df["memory_rolling_range"] = (
        service_df["memory_rolling_max"] - service_df["memory_rolling_min"]
    )
    service_df["memory_rolling_iqr"] = (
        service_df["memory_rolling_q75"] - service_df["memory_rolling_q25"]
    )
    memory_std_denominator = service_df["memory_rolling_std"].replace(0.0, np.nan)
    memory_mean_denominator = service_df["memory_rolling_mean"].abs().replace(0.0, np.nan)
    service_df["memory_zscore"] = (
        (service_df["memory_value"] - service_df["memory_rolling_mean"]) / memory_std_denominator
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    service_df["memory_relative_to_mean"] = (
        service_df["memory_value"] / memory_mean_denominator
    ).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    service_df["is_new_pod_window"] = (
        service_df["pod_age_points"] <= NEW_POD_POINTS
    ).astype(int)

    service_df = service_df.dropna(subset=FEATURE_COLUMNS).copy()
    return service_df


def print_dataset_summary(name, df):
    print(f"\n=== {name} ===")
    print(f"记录数: {len(df)}")
    print(f"Run 数: {df['run_id'].nunique()}")
    print(f"Pod 数: {df['pod'].nunique()}")
    print(f"时间范围: {df['datetime'].min()} -> {df['datetime'].max()}")
    print("Run 计数:")
    print(df["run_id"].value_counts().to_string())


def build_fault_windows(metadata_by_run, service):
    windows = {}
    for run_id, metadata in metadata_by_run.items():
        events = []
        for event in metadata.get("events", []):
            if event.get("service") != service:
                continue
            start_text = event.get("expected_fault_window_start") or event.get("start_time")
            end_text = event.get("expected_fault_window_end") or event.get("end_time")
            if not start_text or not end_text:
                continue
            events.append(
                {
                    "start": pd.to_datetime(start_text),
                    "end": pd.to_datetime(end_text),
                    "fault_type": event.get("fault_type", "unknown"),
                }
            )
        windows[run_id] = events
    return windows


def apply_weak_labels(result_df, fault_windows):
    labeled_df = result_df.copy()
    labeled_df["is_fault_window"] = 0
    labeled_df["fault_type_label"] = "normal"
    for run_id, windows in fault_windows.items():
        for window in windows:
            mask = (
                (labeled_df["run_id"] == run_id)
                & (labeled_df["datetime"] >= window["start"])
                & (labeled_df["datetime"] <= window["end"])
            )
            labeled_df.loc[mask, "is_fault_window"] = 1
            labeled_df.loc[mask, "fault_type_label"] = window["fault_type"]
    return labeled_df


def evaluate_predictions(result_df):
    tp = int(((result_df["is_anomaly"] == 1) & (result_df["is_fault_window"] == 1)).sum())
    fp = int(((result_df["is_anomaly"] == 1) & (result_df["is_fault_window"] == 0)).sum())
    fn = int(((result_df["is_anomaly"] == 0) & (result_df["is_fault_window"] == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    anomaly_ratio = result_df["is_anomaly"].mean()
    return {
        "points": int(len(result_df)),
        "fault_window_points": int(result_df["is_fault_window"].sum()),
        "anomaly_points": int(result_df["is_anomaly"].sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "anomaly_ratio": round(float(anomaly_ratio), 4),
    }


def evaluate_runs(result_df):
    rows = []
    for run_id, run_df in result_df.groupby("run_id"):
        metrics = evaluate_predictions(run_df)
        metrics["run_id"] = run_id
        rows.append(metrics)
    return pd.DataFrame(rows)


def evaluate_fault_types(result_df):
    rows = []
    fault_df = result_df[result_df["fault_type_label"] != "normal"].copy()
    for fault_type, group_df in fault_df.groupby("fault_type_label"):
        metrics = evaluate_predictions(group_df)
        metrics["fault_type"] = fault_type
        rows.append(metrics)
    return pd.DataFrame(rows)


def summarize_metric_rows(metrics_df):
    avg = metrics_df[["precision", "recall", "f1", "anomaly_ratio"]].mean().to_dict()
    return {f"avg_{key}": round(float(value), 4) for key, value in avg.items()}


def create_dataset_summary(stage_name, df):
    return {
        "stage": stage_name,
        "rows": int(len(df)),
        "run_count": int(df["run_id"].nunique()) if not df.empty else 0,
        "pod_count": int(df["pod"].nunique()) if not df.empty else 0,
        "start_time": df["datetime"].min().strftime("%Y-%m-%d %H:%M:%S") if not df.empty else "",
        "end_time": df["datetime"].max().strftime("%Y-%m-%d %H:%M:%S") if not df.empty else "",
    }


def run_iforest_experiment(x_train, eval_features, params, fault_windows, feature_columns):
    model = IsolationForest(
        n_estimators=params["n_estimators"],
        contamination=params["contamination"],
        random_state=42,
    )
    model.fit(x_train)

    x_eval = eval_features[feature_columns].to_numpy()
    raw_pred = model.predict(x_eval)
    scores = model.decision_function(x_eval)

    result_df = eval_features.copy()
    result_df["raw_prediction"] = raw_pred
    result_df["is_anomaly"] = np.where(raw_pred == -1, 1, 0)
    result_df["anomaly_score"] = scores
    result_df = apply_weak_labels(result_df, fault_windows)
    metrics_df = evaluate_runs(result_df)
    summary = summarize_metric_rows(metrics_df)
    summary.update(params)
    summary["feature_set"] = ",".join(feature_columns)
    return result_df, metrics_df, summary


def save_results_plot(result_df, output_path):
    anomaly_df = result_df[result_df["is_anomaly"] == 1]
    run_ids = result_df["run_id"].unique().tolist()
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=False)

    for run_id in run_ids:
        run_df = result_df[result_df["run_id"] == run_id]
        axes[0].plot(run_df["datetime"], run_df["cpu_rate"], linewidth=1.2, label=run_id)
    axes[0].scatter(
        anomaly_df["datetime"],
        anomaly_df["cpu_rate"],
        color="red",
        marker="x",
        s=35,
        label="Predicted anomaly",
        zorder=5,
    )
    fault_window_df = result_df[result_df["is_fault_window"] == 1]
    if not fault_window_df.empty:
        axes[0].axvspan(
            fault_window_df["datetime"].min(),
            fault_window_df["datetime"].max(),
            color="orange",
            alpha=0.12,
            label="Fault window",
        )
    axes[0].set_title(f"Isolation Forest on {TARGET_SERVICE} service across multiple runs")
    axes[0].set_ylabel("cpu_rate")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper left")

    for run_id in run_ids:
        run_df = result_df[result_df["run_id"] == run_id]
        axes[1].plot(run_df["datetime"], run_df["anomaly_score"], linewidth=1.0, label=run_id)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    axes[1].set_title("Anomaly score (lower means more abnormal)")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("decision_function")
    axes[1].grid(True, alpha=0.3)

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def split_available_runs(run_ids):
    available = [run_id for run_id in run_ids if (EXPORTS_DIR / run_id).exists()]
    missing = [run_id for run_id in run_ids if run_id not in available]
    return available, missing


def require_runs(stage_name, available_runs, missing_runs, minimum_count=1):
    print(f"\n{stage_name} 可用轮次: {available_runs if available_runs else '无'}")
    if missing_runs:
        print(f"{stage_name} 缺失轮次: {missing_runs}")
    if len(available_runs) < minimum_count:
        raise FileNotFoundError(
            f"{stage_name} 至少需要 {minimum_count} 个已导出的 run，"
            f"当前只有 {len(available_runs)} 个。"
        )


def main():
    train_runs, missing_train = split_available_runs(TRAIN_RUNS)
    validation_runs, missing_validation = split_available_runs(VALIDATION_RUNS)
    test_runs, missing_test = split_available_runs(TEST_RUNS)
    challenge_runs, missing_challenge = split_available_runs(CHALLENGE_RUNS)

    require_runs("训练集", train_runs, missing_train, minimum_count=1)
    require_runs("验证集", validation_runs, missing_validation, minimum_count=1)
    if test_runs:
        require_runs("测试集", test_runs, missing_test, minimum_count=1)
    else:
        print("\n测试集当前没有可用导出轮次，将跳过测试集评估。")
    if challenge_runs:
        require_runs("挑战集", challenge_runs, missing_challenge, minimum_count=1)
    else:
        print("\n挑战集当前没有可用导出轮次，将跳过挑战集评估。")

    print("加载 exports/run_xxx 数据 ...")
    train_df, _ = load_multiple_runs(train_runs)
    validation_df, validation_metadata = load_multiple_runs(validation_runs)
    test_df, test_metadata = load_multiple_runs(test_runs) if test_runs else (pd.DataFrame(), {})
    challenge_df, challenge_metadata = load_multiple_runs(challenge_runs) if challenge_runs else (pd.DataFrame(), {})

    train_features = build_features(train_df, TARGET_SERVICE)
    validation_features = build_features(validation_df, TARGET_SERVICE)
    test_features = build_features(test_df, TARGET_SERVICE) if not test_df.empty else pd.DataFrame()
    challenge_features = build_features(challenge_df, TARGET_SERVICE) if not challenge_df.empty else pd.DataFrame()

    print_dataset_summary("训练集 / normal runs", train_features)
    print_dataset_summary("验证集 / single fault first runs", validation_features)
    if not test_features.empty:
        print_dataset_summary("测试集 / single fault second runs", test_features)
    if not challenge_features.empty:
        print_dataset_summary("挑战集 / composite runs", challenge_features)

    validation_windows = build_fault_windows(validation_metadata, TARGET_SERVICE)
    test_windows = build_fault_windows(test_metadata, TARGET_SERVICE) if test_metadata else {}
    challenge_windows = build_fault_windows(challenge_metadata, TARGET_SERVICE) if challenge_metadata else {}

    x_train = train_features[FEATURE_COLUMNS].to_numpy()

    print("\n进行参数对比实验（基于验证集平均 F1）...")
    comparison_rows = []
    validation_results = {}
    for params in PARAM_GRID:
        result_df, metrics_df, summary = run_iforest_experiment(
            x_train, validation_features, params, validation_windows, FEATURE_COLUMNS
        )
        summary["stage"] = "validation"
        comparison_rows.append(summary)
        validation_results[(params["n_estimators"], params["contamination"])] = (result_df, metrics_df)
        print(
            "参数 "
            f"n_estimators={params['n_estimators']}, contamination={params['contamination']}: "
            f"avg_precision={summary['avg_precision']}, avg_recall={summary['avg_recall']}, avg_f1={summary['avg_f1']}"
        )

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        ["avg_f1", "avg_precision", "avg_recall", "avg_anomaly_ratio"],
        ascending=[False, False, False, True],
    )
    best_row = comparison_df.iloc[0]
    best_params = {
        "n_estimators": int(best_row["n_estimators"]),
        "contamination": float(best_row["contamination"]),
    }

    print("\n进行特征对比实验（基于验证集）...")
    feature_rows = []
    feature_results = {}
    for feature_name, feature_columns in FEATURE_SETS.items():
        feature_train = train_features[feature_columns].to_numpy()
        feature_result_df, feature_metrics_df, feature_summary = run_iforest_experiment(
            feature_train,
            validation_features,
            best_params,
            validation_windows,
            feature_columns,
        )
        feature_summary["feature_name"] = feature_name
        feature_rows.append(feature_summary)
        feature_results[feature_name] = (feature_result_df, feature_metrics_df)
        print(
            f"特征组 {feature_name}: avg_precision={feature_summary['avg_precision']}, "
            f"avg_recall={feature_summary['avg_recall']}, avg_f1={feature_summary['avg_f1']}"
        )

    feature_comparison_df = pd.DataFrame(feature_rows).sort_values(
        ["avg_f1", "avg_precision", "avg_recall", "avg_anomaly_ratio"],
        ascending=[False, False, False, True],
    )
    best_feature_row = feature_comparison_df.iloc[0]
    best_feature_name = best_feature_row["feature_name"]
    selected_feature_columns = FEATURE_SETS[best_feature_name]

    print("\n使用最优参数与特征组进行后续评估 ...")
    final_train = train_features[selected_feature_columns].to_numpy()
    test_result_df, test_metrics_df, test_summary = (pd.DataFrame(), pd.DataFrame(), {})
    challenge_result_df, challenge_metrics_df, challenge_summary = (pd.DataFrame(), pd.DataFrame(), {})
    if not test_features.empty:
        test_result_df, test_metrics_df, test_summary = run_iforest_experiment(
            final_train,
            test_features,
            best_params,
            test_windows,
            selected_feature_columns,
        )
    if not challenge_features.empty:
        challenge_result_df, challenge_metrics_df, challenge_summary = run_iforest_experiment(
            final_train,
            challenge_features,
            best_params,
            challenge_windows,
            selected_feature_columns,
        )

    output_csv = BASE_DIR / f"{TARGET_SERVICE}_iforest_test_results.csv"
    output_png = BASE_DIR / f"{TARGET_SERVICE}_iforest_test_plot.png"
    output_summary_csv = BASE_DIR / f"{TARGET_SERVICE}_iforest_param_comparison.csv"
    output_feature_summary_csv = BASE_DIR / f"{TARGET_SERVICE}_iforest_feature_comparison.csv"
    output_test_metrics_csv = BASE_DIR / f"{TARGET_SERVICE}_iforest_test_run_metrics.csv"
    output_challenge_metrics_csv = BASE_DIR / f"{TARGET_SERVICE}_iforest_challenge_run_metrics.csv"
    output_challenge_csv = BASE_DIR / f"{TARGET_SERVICE}_iforest_challenge_results.csv"
    output_challenge_png = BASE_DIR / f"{TARGET_SERVICE}_iforest_challenge_plot.png"
    output_dataset_summary_csv = BASE_DIR / f"{TARGET_SERVICE}_iforest_dataset_summary.csv"
    output_stage_summary_csv = BASE_DIR / f"{TARGET_SERVICE}_iforest_stage_summary.csv"
    output_test_fault_metrics_csv = BASE_DIR / f"{TARGET_SERVICE}_iforest_test_fault_type_metrics.csv"
    output_challenge_fault_metrics_csv = BASE_DIR / f"{TARGET_SERVICE}_iforest_challenge_fault_type_metrics.csv"

    comparison_df.to_csv(output_summary_csv, index=False)
    feature_comparison_df.to_csv(output_feature_summary_csv, index=False)
    dataset_summary_df = pd.DataFrame(
        [
            create_dataset_summary("train", train_features),
            create_dataset_summary("validation", validation_features),
            create_dataset_summary("test", test_features) if not test_features.empty else create_dataset_summary("test", pd.DataFrame()),
            create_dataset_summary("challenge", challenge_features)
            if not challenge_features.empty
            else create_dataset_summary("challenge", pd.DataFrame()),
        ]
    )
    dataset_summary_df.to_csv(output_dataset_summary_csv, index=False)

    stage_summary_rows = [
        {
            "stage": "validation_best_params",
            "best_n_estimators": best_params["n_estimators"],
            "best_contamination": best_params["contamination"],
            "selected_feature_set": best_feature_name,
            "avg_precision": best_row["avg_precision"],
            "avg_recall": best_row["avg_recall"],
            "avg_f1": best_row["avg_f1"],
            "avg_anomaly_ratio": best_row["avg_anomaly_ratio"],
        }
    ]
    if not test_result_df.empty:
        test_result_df.to_csv(output_csv, index=False)
        test_metrics_df.to_csv(output_test_metrics_csv, index=False)
        test_fault_metrics_df = evaluate_fault_types(test_result_df)
        test_fault_metrics_df.to_csv(output_test_fault_metrics_csv, index=False)
        save_results_plot(test_result_df, output_png)
        stage_summary_rows.append(
            {
                "stage": "test",
                "best_n_estimators": best_params["n_estimators"],
                "best_contamination": best_params["contamination"],
                "selected_feature_set": best_feature_name,
                "avg_precision": test_summary["avg_precision"],
                "avg_recall": test_summary["avg_recall"],
                "avg_f1": test_summary["avg_f1"],
                "avg_anomaly_ratio": test_summary["avg_anomaly_ratio"],
            }
        )
    if not challenge_result_df.empty:
        challenge_result_df.to_csv(output_challenge_csv, index=False)
        challenge_metrics_df.to_csv(output_challenge_metrics_csv, index=False)
        challenge_fault_metrics_df = evaluate_fault_types(challenge_result_df)
        challenge_fault_metrics_df.to_csv(output_challenge_fault_metrics_csv, index=False)
        save_results_plot(challenge_result_df, output_challenge_png)
        stage_summary_rows.append(
            {
                "stage": "challenge",
                "best_n_estimators": best_params["n_estimators"],
                "best_contamination": best_params["contamination"],
                "selected_feature_set": best_feature_name,
                "avg_precision": challenge_summary["avg_precision"],
                "avg_recall": challenge_summary["avg_recall"],
                "avg_f1": challenge_summary["avg_f1"],
                "avg_anomaly_ratio": challenge_summary["avg_anomaly_ratio"],
            }
        )
    pd.DataFrame(stage_summary_rows).to_csv(output_stage_summary_csv, index=False)

    print(
        "\n自动选出的最优参数: "
        f"n_estimators={best_params['n_estimators']}, "
        f"contamination={best_params['contamination']}, "
        f"avg_precision={best_row['avg_precision']}, avg_recall={best_row['avg_recall']}, avg_f1={best_row['avg_f1']}"
    )
    print(
        "最优特征组: "
        f"{best_feature_name}, avg_precision={best_feature_row['avg_precision']}, "
        f"avg_recall={best_feature_row['avg_recall']}, avg_f1={best_feature_row['avg_f1']}"
    )
    if test_summary:
        print(
            "\n测试集结果: "
            f"avg_precision={test_summary['avg_precision']}, avg_recall={test_summary['avg_recall']}, avg_f1={test_summary['avg_f1']}"
        )
        print(f"测试集结果明细已保存: {output_csv}")
        print(f"测试集逐轮指标已保存: {output_test_metrics_csv}")
        print(f"测试集按故障类型指标已保存: {output_test_fault_metrics_csv}")
        print(f"测试集结果图已保存: {output_png}")
    else:
        print("\n测试集结果: 当前无可用测试轮次，已跳过。")
    if challenge_summary:
        print(
            "挑战集结果: "
            f"avg_precision={challenge_summary['avg_precision']}, avg_recall={challenge_summary['avg_recall']}, avg_f1={challenge_summary['avg_f1']}"
        )
        print(f"挑战集结果明细已保存: {output_challenge_csv}")
        print(f"挑战集逐轮指标已保存: {output_challenge_metrics_csv}")
        print(f"挑战集按故障类型指标已保存: {output_challenge_fault_metrics_csv}")
        print(f"挑战集结果图已保存: {output_challenge_png}")
    else:
        print("挑战集结果: 当前无可用挑战轮次，已跳过。")

    print(f"参数对比结果已保存: {output_summary_csv}")
    print(f"特征对比结果已保存: {output_feature_summary_csv}")
    print(f"数据集摘要已保存: {output_dataset_summary_csv}")
    print(f"阶段汇总已保存: {output_stage_summary_csv}")


if __name__ == "__main__":
    main()
