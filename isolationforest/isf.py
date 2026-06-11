import ast
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
NORMAL_CSV = BASE_DIR / "normal.csv"
FAULT_CSV = BASE_DIR / "fault.csv"

# 先聚焦故障最明显的 carts 服务，便于复现实验。
TARGET_SERVICE = "carts"
KILLED_POD = "carts-b6c5c87f9-cz2bq"
ROLLING_WINDOW = 5
NEW_POD_POINTS = 4
FEATURE_COLUMNS = [
    "cpu_rate",
    "rolling_mean",
    "rolling_std",
    "abs_rate_change",
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
    "full_features": FEATURE_COLUMNS,
}
PARAM_GRID = [
    {"n_estimators": 100, "contamination": 0.03},
    {"n_estimators": 100, "contamination": 0.05},
    {"n_estimators": 200, "contamination": 0.05},
    {"n_estimators": 200, "contamination": 0.08},
    {"n_estimators": 300, "contamination": 0.08},
]


def parse_metric(metric_str):
    try:
        return ast.literal_eval(metric_str)
    except (SyntaxError, ValueError):
        return {}


def extract_service_name(pod_name):
    match = re.match(r"^(.*)-[a-z0-9]{9,10}-[a-z0-9]{5}$", pod_name)
    if match:
        return match.group(1)
    return pod_name


def load_dataset(csv_path):
    df = pd.read_csv(csv_path).dropna().copy()
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["metric_dict"] = df["metric"].apply(parse_metric)
    df["pod"] = df["metric_dict"].apply(lambda item: item.get("pod", "unknown"))
    df["service"] = df["pod"].apply(extract_service_name)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.dropna(subset=["timestamp", "value"]).sort_values(["pod", "timestamp"])
    return df


def build_features(df, target_service):
    service_df = df[df["service"] == target_service].copy()
    if service_df.empty:
        raise ValueError(f"未在数据中找到服务 {target_service!r}")

    service_df["pod_age_points"] = service_df.groupby("pod").cumcount()
    service_df["delta_time"] = service_df.groupby("pod")["timestamp"].diff()
    service_df["delta_value"] = service_df.groupby("pod")["value"].diff()
    service_df["cpu_rate"] = service_df["delta_value"] / service_df["delta_time"]

    service_df = service_df[service_df["delta_time"] > 0].copy()
    # Counter 理论上单调增加，负跳变通常意味着重置或坏点，这里直接过滤。
    service_df = service_df[service_df["delta_value"] >= 0].copy()

    service_df["rate_change"] = service_df.groupby("pod")["cpu_rate"].diff()
    service_df["abs_rate_change"] = service_df["rate_change"].abs().fillna(0.0)
    service_df["rolling_mean"] = (
        service_df.groupby("pod")["cpu_rate"]
        .transform(lambda series: series.rolling(ROLLING_WINDOW, min_periods=1).mean())
    )
    service_df["rolling_std"] = (
        service_df.groupby("pod")["cpu_rate"]
        .transform(lambda series: series.rolling(ROLLING_WINDOW, min_periods=1).std())
        .fillna(0.0)
    )
    service_df["is_new_pod_window"] = (
        service_df["pod_age_points"] <= NEW_POD_POINTS
    ).astype(int)

    service_df = service_df.dropna(subset=FEATURE_COLUMNS).copy()
    return service_df


def print_dataset_summary(name, df):
    print(f"\n=== {name} ===")
    print(f"记录数: {len(df)}")
    print(f"Pod 数: {df['pod'].nunique()}")
    print(f"时间范围: {df['datetime'].min()} -> {df['datetime'].max()}")
    print("Pod 计数:")
    print(df["pod"].value_counts().to_string())


def infer_fault_window(fault_features):
    killed_df = fault_features[fault_features["pod"] == KILLED_POD].sort_values("timestamp")
    replacement_df = fault_features[fault_features["pod"] != KILLED_POD].sort_values("timestamp")

    if killed_df.empty or replacement_df.empty:
        raise ValueError("无法根据被杀 pod 推断故障窗口，请检查 pod 名称是否正确。")

    killed_last_ts = killed_df["timestamp"].max()
    replacement_first_ts = replacement_df["timestamp"].min()
    replacement_first_dt = replacement_df["datetime"].min()
    killed_last_dt = killed_df["datetime"].max()

    # 给重建阶段留一点缓冲，避免窗口过窄。
    fault_window_start = replacement_first_dt
    fault_window_end = killed_last_dt + pd.Timedelta(minutes=2)
    return {
        "killed_last_ts": killed_last_ts,
        "replacement_first_ts": replacement_first_ts,
        "fault_window_start": fault_window_start,
        "fault_window_end": fault_window_end,
    }


def apply_weak_labels(result_df, fault_window):
    labeled_df = result_df.copy()
    labeled_df["is_fault_window"] = (
        (labeled_df["datetime"] >= fault_window["fault_window_start"])
        & (labeled_df["datetime"] <= fault_window["fault_window_end"])
    ).astype(int)
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


def run_iforest_experiment(x_train, fault_features, params, fault_window, feature_columns):
    model = IsolationForest(
        n_estimators=params["n_estimators"],
        contamination=params["contamination"],
        random_state=42,
    )
    model.fit(x_train)

    x_test = fault_features[feature_columns].to_numpy()
    raw_pred = model.predict(x_test)
    scores = model.decision_function(x_test)

    result_df = fault_features.copy()
    result_df["raw_prediction"] = raw_pred
    result_df["is_anomaly"] = np.where(raw_pred == -1, 1, 0)
    result_df["anomaly_score"] = scores
    result_df = apply_weak_labels(result_df, fault_window)
    metrics = evaluate_predictions(result_df)
    metrics.update(params)
    metrics["feature_set"] = ",".join(feature_columns)
    return result_df, metrics


def save_results_plot(result_df, output_path):
    anomaly_df = result_df[result_df["is_anomaly"] == 1]
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    for pod_name, group in result_df.groupby("pod"):
        axes[0].plot(group["datetime"], group["cpu_rate"], linewidth=1.4, label=pod_name)
    axes[0].scatter(
        anomaly_df["datetime"],
        anomaly_df["cpu_rate"],
        color="red",
        marker="x",
        s=50,
        label="Predicted anomaly",
        zorder=5,
    )
    fault_window_df = result_df[result_df["is_fault_window"] == 1]
    if not fault_window_df.empty:
        axes[0].axvspan(
            fault_window_df["datetime"].min(),
            fault_window_df["datetime"].max(),
            color="orange",
            alpha=0.15,
            label="Fault window",
        )
    axes[0].set_title(
        f"Isolation Forest on {TARGET_SERVICE} service (feature: cpu_rate and rolling stats)"
    )
    axes[0].set_ylabel("cpu_rate")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper left")

    for pod_name, group in result_df.groupby("pod"):
        axes[1].plot(group["datetime"], group["anomaly_score"], linewidth=1.2, label=pod_name)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    axes[1].set_title("Anomaly score (lower means more abnormal)")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("decision_function")
    axes[1].grid(True, alpha=0.3)

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    print("加载 normal.csv 和 fault.csv ...")
    normal_df = load_dataset(NORMAL_CSV)
    fault_df = load_dataset(FAULT_CSV)

    normal_features = build_features(normal_df, TARGET_SERVICE)
    fault_features = build_features(fault_df, TARGET_SERVICE)

    print_dataset_summary("训练集 normal.csv / carts", normal_features)
    print_dataset_summary("测试集 fault.csv / carts", fault_features)
    fault_window = infer_fault_window(fault_features)
    print("\n推断出的故障窗口:")
    print(
        f"被杀 pod: {KILLED_POD}\n"
        f"替代 pod 首次出现: {fault_window['replacement_first_ts']} "
        f"({fault_window['fault_window_start']})\n"
        f"被杀 pod 最后出现: {fault_window['killed_last_ts']} "
        f"({fault_features[fault_features['timestamp'] == fault_window['killed_last_ts']]['datetime'].iloc[0]})\n"
        f"弱标签故障窗口: {fault_window['fault_window_start']} -> {fault_window['fault_window_end']}"
    )

    x_train = normal_features[FEATURE_COLUMNS].to_numpy()

    print("\n进行参数对比实验 ...")
    comparison_rows = []
    experiment_results = {}
    for params in PARAM_GRID:
        result_df, metrics = run_iforest_experiment(
            x_train, fault_features, params, fault_window, FEATURE_COLUMNS
        )
        comparison_rows.append(metrics)
        experiment_results[(params["n_estimators"], params["contamination"])] = result_df
        print(
            "参数 "
            f"n_estimators={params['n_estimators']}, contamination={params['contamination']}: "
            f"anomaly={metrics['anomaly_points']}, "
            f"tp={metrics['tp']}, fp={metrics['fp']}, fn={metrics['fn']}, "
            f"precision={metrics['precision']}, recall={metrics['recall']}, f1={metrics['f1']}"
        )

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        ["f1", "precision", "recall", "anomaly_points"],
        ascending=[False, False, False, True],
    )
    best_row = comparison_df.iloc[0]
    best_key = (int(best_row["n_estimators"]), float(best_row["contamination"]))
    selected_result_df = experiment_results[best_key]
    best_params = {
        "n_estimators": int(best_row["n_estimators"]),
        "contamination": float(best_row["contamination"]),
    }

    print("\n进行特征对比实验 ...")
    feature_rows = []
    feature_results = {}
    for feature_name, feature_columns in FEATURE_SETS.items():
        feature_train = normal_features[feature_columns].to_numpy()
        feature_result_df, feature_metrics = run_iforest_experiment(
            feature_train,
            fault_features,
            best_params,
            fault_window,
            feature_columns,
        )
        feature_metrics["feature_name"] = feature_name
        feature_rows.append(feature_metrics)
        feature_results[feature_name] = feature_result_df
        print(
            f"特征组 {feature_name}: anomaly={feature_metrics['anomaly_points']}, "
            f"tp={feature_metrics['tp']}, fp={feature_metrics['fp']}, fn={feature_metrics['fn']}, "
            f"precision={feature_metrics['precision']}, recall={feature_metrics['recall']}, "
            f"f1={feature_metrics['f1']}"
        )

    feature_comparison_df = pd.DataFrame(feature_rows).sort_values(
        ["f1", "precision", "recall", "anomaly_points"],
        ascending=[False, False, False, True],
    )
    best_feature_row = feature_comparison_df.iloc[0]
    best_feature_name = best_feature_row["feature_name"]

    output_csv = BASE_DIR / f"{TARGET_SERVICE}_iforest_results.csv"
    output_png = BASE_DIR / f"{TARGET_SERVICE}_iforest_plot.png"
    output_summary_csv = BASE_DIR / f"{TARGET_SERVICE}_iforest_param_comparison.csv"
    output_feature_summary_csv = BASE_DIR / f"{TARGET_SERVICE}_iforest_feature_comparison.csv"
    selected_result_df.to_csv(output_csv, index=False)
    comparison_df.to_csv(output_summary_csv, index=False)
    feature_comparison_df.to_csv(output_feature_summary_csv, index=False)
    save_results_plot(selected_result_df, output_png)

    anomaly_df = selected_result_df[selected_result_df["is_anomaly"] == 1].sort_values(
        "anomaly_score"
    )
    print(
        "\n自动选出的最优参数: "
        f"n_estimators={int(best_row['n_estimators'])}, "
        f"contamination={float(best_row['contamination'])}, "
        f"precision={best_row['precision']}, recall={best_row['recall']}, f1={best_row['f1']}"
    )
    print(
        "最优特征组: "
        f"{best_feature_name}, precision={best_feature_row['precision']}, "
        f"recall={best_feature_row['recall']}, f1={best_feature_row['f1']}"
    )
    print(
        f"\n选定参数检测完成: 共 {len(selected_result_df)} 个测试点, "
        f"异常点 {len(anomaly_df)} 个"
    )
    if not anomaly_df.empty:
        print("\n最异常的前 10 个点:")
        print(
            anomaly_df[
                [
                    "datetime",
                    "pod",
                    "cpu_rate",
                    "anomaly_score",
                    "is_new_pod_window",
                    "is_fault_window",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )

    print(f"\n结果明细已保存: {output_csv}")
    print(f"参数对比结果已保存: {output_summary_csv}")
    print(f"特征对比结果已保存: {output_feature_summary_csv}")
    print(f"结果图已保存: {output_png}")


if __name__ == "__main__":
    main()
