from pathlib import Path

import pandas as pd

import isf as base
import isf_v2_postprocess as v2


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "optimized_analysis_v3"
EVENT_TOLERANCE_SECONDS = 60
BALANCED_MAX_ANOMALY_RATIO = 0.45
PARAM_GRID = [
    {"n_estimators": 100, "contamination": 0.02},
    {"n_estimators": 100, "contamination": 0.03},
    {"n_estimators": 100, "contamination": 0.05},
    {"n_estimators": 100, "contamination": 0.08},
    {"n_estimators": 200, "contamination": 0.03},
    {"n_estimators": 200, "contamination": 0.05},
    {"n_estimators": 200, "contamination": 0.08},
    {"n_estimators": 200, "contamination": 0.10},
    {"n_estimators": 300, "contamination": 0.05},
    {"n_estimators": 300, "contamination": 0.08},
    {"n_estimators": 300, "contamination": 0.10},
    {"n_estimators": 500, "contamination": 0.08},
    {"n_estimators": 500, "contamination": 0.10},
    {"n_estimators": 500, "contamination": 0.12},
]
POSTPROCESS_GRID = [
    {
        "postprocess_name": "streak_2",
        "min_streak": 2,
        "smooth_window": 1,
        "vote_threshold": 1,
    },
    {
        "postprocess_name": "streak_3",
        "min_streak": 3,
        "smooth_window": 1,
        "vote_threshold": 1,
    },
    {
        "postprocess_name": "streak_2_smooth_3_vote_2",
        "min_streak": 2,
        "smooth_window": 3,
        "vote_threshold": 2,
    },
    {
        "postprocess_name": "streak_3_smooth_3_vote_2",
        "min_streak": 3,
        "smooth_window": 3,
        "vote_threshold": 2,
    },
    {
        "postprocess_name": "streak_3_smooth_5_vote_3",
        "min_streak": 3,
        "smooth_window": 5,
        "vote_threshold": 3,
    },
    {
        "postprocess_name": "streak_4_smooth_5_vote_3",
        "min_streak": 4,
        "smooth_window": 5,
        "vote_threshold": 3,
    },
]


def smooth_binary_flags(raw_flags: list[int], window: int, vote_threshold: int) -> list[int]:
    if window <= 1:
        return [int(flag) for flag in raw_flags]
    series = pd.Series(raw_flags, dtype="int64")
    smoothed = (
        series.rolling(window=window, min_periods=1, center=True).sum() >= vote_threshold
    ).astype(int)
    return smoothed.tolist()


def apply_postprocess(result_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    filtered_df = v2.apply_consecutive_filter(result_df, config["min_streak"])
    processed_df = filtered_df.copy().sort_values(["run_id", "pod", "datetime"]).reset_index(drop=True)
    processed_df["is_anomaly_filtered"] = processed_df["is_anomaly"]
    processed_df["is_anomaly"] = 0

    for (_, _), group_df in processed_df.groupby(["run_id", "pod"], sort=False):
        smoothed_flags = smooth_binary_flags(
            group_df["is_anomaly_filtered"].tolist(),
            config["smooth_window"],
            config["vote_threshold"],
        )
        processed_df.loc[group_df.index, "is_anomaly"] = smoothed_flags

    return processed_df


def compute_event_metrics(result_df: pd.DataFrame, fault_windows: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    return v2.evaluate_event_level(result_df, fault_windows, EVENT_TOLERANCE_SECONDS)


def run_tuned_experiment(x_train, eval_features, params, postprocess_config, fault_windows, feature_columns):
    result_df, _, _ = base.run_iforest_experiment(x_train, eval_features, params, fault_windows, feature_columns)
    tuned_df = apply_postprocess(result_df, postprocess_config)
    run_metrics_df = base.evaluate_runs(tuned_df)
    summary = base.summarize_metric_rows(run_metrics_df)
    event_details_df, event_summary_df = compute_event_metrics(tuned_df, fault_windows)
    summary.update(params)
    summary["feature_set"] = ",".join(feature_columns)
    summary["postprocess_name"] = postprocess_config["postprocess_name"]
    summary["postprocess_min_streak"] = postprocess_config["min_streak"]
    summary["postprocess_smooth_window"] = postprocess_config["smooth_window"]
    summary["postprocess_vote_threshold"] = postprocess_config["vote_threshold"]
    if not event_summary_df.empty:
        overall_row = event_summary_df[
            (event_summary_df["group"] == "overall") & (event_summary_df["name"] == "all")
        ].iloc[0]
        summary["event_hit_rate"] = float(overall_row["hit_rate"])
    else:
        summary["event_hit_rate"] = 0.0
    summary["selection_score"] = round(
        summary["avg_f1"] * 0.6
        + summary["avg_precision"] * 0.2
        + summary["avg_recall"] * 0.1
        + summary["event_hit_rate"] * 0.1,
        4,
    )
    return tuned_df, run_metrics_df, summary, event_details_df, event_summary_df


def calculate_window_coverage_ratio(
    features_df: pd.DataFrame, fault_windows: dict, tolerance_seconds: int
) -> dict:
    tolerance = pd.Timedelta(seconds=tolerance_seconds)
    labeled_count = 0
    total_count = len(features_df)

    for run_id, windows in fault_windows.items():
        run_df = features_df[features_df["run_id"] == run_id]
        if run_df.empty:
            continue
        mask = pd.Series(False, index=run_df.index)
        for window in windows:
            start = pd.to_datetime(window["start"]) - tolerance
            end = pd.to_datetime(window["end"]) + tolerance
            mask = mask | ((run_df["datetime"] >= start) & (run_df["datetime"] <= end))
        labeled_count += int(mask.sum())

    ratio = labeled_count / total_count if total_count else 0.0
    return {
        "tolerance_seconds": tolerance_seconds,
        "covered_points": labeled_count,
        "total_points": int(total_count),
        "coverage_ratio": round(float(ratio), 4),
    }


def build_window_coverage_summary(validation_features, test_features, challenge_features, validation_windows, test_windows, challenge_windows):
    rows = []
    for stage_name, features_df, fault_windows in [
        ("validation", validation_features, validation_windows),
        ("test", test_features, test_windows),
        ("challenge", challenge_features, challenge_windows),
    ]:
        for tolerance_seconds in [0, 60, 120, 180]:
            metrics = calculate_window_coverage_ratio(features_df, fault_windows, tolerance_seconds)
            metrics["stage"] = stage_name
            rows.append(metrics)
    return pd.DataFrame(rows)


def select_balanced_candidate(comparison_df: pd.DataFrame) -> pd.Series:
    balanced_df = comparison_df[
        comparison_df["avg_anomaly_ratio"] <= BALANCED_MAX_ANOMALY_RATIO
    ].copy()
    if balanced_df.empty:
        balanced_df = comparison_df.copy()
    balanced_df = balanced_df.sort_values(
        ["avg_f1", "event_hit_rate", "avg_precision", "avg_recall", "avg_anomaly_ratio"],
        ascending=[False, False, False, False, True],
    )
    return balanced_df.iloc[0]


def write_summary_file(
    stage_summary_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    param_postprocess_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    selected_configs_df: pd.DataFrame,
):
    validation_row = stage_summary_df[stage_summary_df["stage"] == "validation"].iloc[0]
    test_row = stage_summary_df[stage_summary_df["stage"] == "test"].iloc[0]
    challenge_row = stage_summary_df[stage_summary_df["stage"] == "challenge"].iloc[0]
    coverage_best_row = selected_configs_df[selected_configs_df["selection_mode"] == "coverage_best"].iloc[0]
    balanced_best_row = selected_configs_df[selected_configs_df["selection_mode"] == "balanced_best"].iloc[0]
    validation_0 = coverage_df[
        (coverage_df["stage"] == "validation") & (coverage_df["tolerance_seconds"] == 0)
    ].iloc[0]
    validation_120 = coverage_df[
        (coverage_df["stage"] == "validation") & (coverage_df["tolerance_seconds"] == 120)
    ].iloc[0]
    top_param_rows = param_postprocess_df.head(5)
    top_feature_rows = feature_df.head(5)

    lines = [
        "Isolation Forest V3 调参与后处理优化说明",
        "",
        "1. V3 的调整目标",
        "",
        "V3 在不修改原始实验数据的前提下，同时扩大了模型参数与后处理搜索范围。",
        "主要考虑以下现实约束：",
        "- 每轮包含 15 个异常事件",
        "- 单轮总时长约 50~67 分钟",
        "- 事件之间间隔只有 30~90 秒",
        "- cooldown 时间较短，故障效应与恢复效应容易彼此粘连",
        "",
        "2. 对 contamination 合理性的重新判断",
        "",
        f"- validation 严格故障窗口覆盖率 = {validation_0['coverage_ratio']}",
        f"- validation 扩展 +/-120 秒后的覆盖率 = {validation_120['coverage_ratio']}",
        "- 这说明如果只按严格故障窗口看，异常占比并不高；",
        "- 但如果把故障邻域与恢复效应考虑进来，有效异常区域会明显变宽；",
        "- 因此 V2 只搜索到 0.08 的 contamination 偏保守，V3 扩展到 0.02~0.12 更合理。",
        "",
        "3. V3 的两类代表配置",
        "",
        (
            f"- 高覆盖配置: n_estimators={int(coverage_best_row['n_estimators'])}, "
            f"contamination={float(coverage_best_row['contamination'])}, "
            f"postprocess={coverage_best_row['postprocess_name']}, "
            f"event_hit_rate={coverage_best_row['event_hit_rate']}, "
            f"anomaly_ratio={coverage_best_row['avg_anomaly_ratio']}"
        ),
        (
            f"- 平衡配置: n_estimators={int(balanced_best_row['n_estimators'])}, "
            f"contamination={float(balanced_best_row['contamination'])}, "
            f"postprocess={balanced_best_row['postprocess_name']}, "
            f"event_hit_rate={balanced_best_row['event_hit_rate']}, "
            f"anomaly_ratio={balanced_best_row['avg_anomaly_ratio']}"
        ),
        "- V3 最终推荐使用平衡配置，避免异常比例被推得过高。",
        "",
        "4. V3 最终推荐配置",
        "",
        f"- n_estimators = {int(validation_row['best_n_estimators'])}",
        f"- contamination = {float(validation_row['best_contamination'])}",
        f"- feature_set = {validation_row['selected_feature_set']}",
        f"- postprocess = {validation_row['postprocess_name']}",
        f"- min_streak = {int(validation_row['postprocess_min_streak'])}",
        f"- smooth_window = {int(validation_row['postprocess_smooth_window'])}",
        f"- vote_threshold = {int(validation_row['postprocess_vote_threshold'])}",
        "",
        "5. 阶段级结果",
        "",
        (
            f"- validation: precision={validation_row['avg_precision']}, "
            f"recall={validation_row['avg_recall']}, f1={validation_row['avg_f1']}, "
            f"event_hit_rate={validation_row['event_hit_rate']}"
        ),
        (
            f"- test: precision={test_row['avg_precision']}, "
            f"recall={test_row['avg_recall']}, f1={test_row['avg_f1']}, "
            f"event_hit_rate={test_row['event_hit_rate']}"
        ),
        (
            f"- challenge: precision={challenge_row['avg_precision']}, "
            f"recall={challenge_row['avg_recall']}, f1={challenge_row['avg_f1']}, "
            f"event_hit_rate={challenge_row['event_hit_rate']}"
        ),
        "",
        "6. 参数合理性结论",
        "",
        "- 如果 contamination 太低，模型更容易漏掉相邻事件之间的异常拖尾；",
        "- 如果 contamination 太高，在短 cooldown 场景下又会把恢复期大量打成异常；",
        "- 因此 contamination 需要和后处理一起看，而不是单独看模型输出；",
        "- V3 的目标就是先允许模型稍微敏感一些，再用更强的时序后处理收缩误报边界。",
        f"- 当前平衡配置把 avg_anomaly_ratio 控制在 <= {BALANCED_MAX_ANOMALY_RATIO} 的范围内，更适合做主版本。",
        "",
        "7. 验证集前 5 组参数 + 后处理结果",
        "",
    ]

    for _, row in top_param_rows.iterrows():
        lines.append(
            (
                f"- n_estimators={int(row['n_estimators'])}, contamination={float(row['contamination'])}, "
                f"postprocess={row['postprocess_name']}, selection_score={row['selection_score']}, "
                f"precision={row['avg_precision']}, recall={row['avg_recall']}, "
                f"f1={row['avg_f1']}, event_hit_rate={row['event_hit_rate']}, "
                f"anomaly_ratio={row['avg_anomaly_ratio']}"
            )
        )

    lines.extend(
        [
            "",
            "8. 验证集前 5 组特征结果",
            "",
        ]
    )

    for _, row in top_feature_rows.iterrows():
        lines.append(
            (
                f"- feature={row['feature_name']}, selection_score={row['selection_score']}, "
                f"precision={row['avg_precision']}, recall={row['avg_recall']}, "
                f"f1={row['avg_f1']}, event_hit_rate={row['event_hit_rate']}, "
                f"anomaly_ratio={row['avg_anomaly_ratio']}"
            )
        )

    lines.extend(
        [
            "",
            "9. 当前建议",
            "",
            "- 如果 V3 的 challenge 提升而 test 没同步提升，说明当前优化更偏向提高覆盖率；",
            "- 如果希望继续压误报，优先收紧 postprocess，再观察 event_hit_rate 是否明显下降；",
            "- 如果 challenge 仍明显弱于 test，下一步应优先做 kill/pause 分类型优化。",
            "",
        ]
    )

    (OUTPUT_DIR / "v3_tuning_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_runs, missing_train = base.split_available_runs(base.TRAIN_RUNS)
    validation_runs, missing_validation = base.split_available_runs(base.VALIDATION_RUNS)
    test_runs, missing_test = base.split_available_runs(base.TEST_RUNS)
    challenge_runs, missing_challenge = base.split_available_runs(base.CHALLENGE_RUNS)

    base.require_runs("训练集", train_runs, missing_train, minimum_count=1)
    base.require_runs("验证集", validation_runs, missing_validation, minimum_count=1)
    base.require_runs("测试集", test_runs, missing_test, minimum_count=1)
    base.require_runs("挑战集", challenge_runs, missing_challenge, minimum_count=1)

    print("加载 exports/run_xxx 数据 ...")
    train_df, _ = base.load_multiple_runs(train_runs)
    validation_df, validation_metadata = base.load_multiple_runs(validation_runs)
    test_df, test_metadata = base.load_multiple_runs(test_runs)
    challenge_df, challenge_metadata = base.load_multiple_runs(challenge_runs)

    train_features = base.build_features(train_df, base.TARGET_SERVICE)
    validation_features = base.build_features(validation_df, base.TARGET_SERVICE)
    test_features = base.build_features(test_df, base.TARGET_SERVICE)
    challenge_features = base.build_features(challenge_df, base.TARGET_SERVICE)

    validation_windows = base.build_fault_windows(validation_metadata, base.TARGET_SERVICE)
    test_windows = base.build_fault_windows(test_metadata, base.TARGET_SERVICE)
    challenge_windows = base.build_fault_windows(challenge_metadata, base.TARGET_SERVICE)

    coverage_df = build_window_coverage_summary(
        validation_features,
        test_features,
        challenge_features,
        validation_windows,
        test_windows,
        challenge_windows,
    )
    coverage_df.to_csv(OUTPUT_DIR / "carts_iforest_v3_window_coverage_summary.csv", index=False)

    print("\n进行参数 + 后处理联合对比实验（基于验证集）...")
    comparison_rows = []
    x_train_full = train_features[base.FEATURE_COLUMNS].to_numpy()
    for params in PARAM_GRID:
        for postprocess_config in POSTPROCESS_GRID:
            _, _, summary, _, _ = run_tuned_experiment(
                x_train_full,
                validation_features,
                params,
                postprocess_config,
                validation_windows,
                base.FEATURE_COLUMNS,
            )
            summary["stage"] = "validation"
            comparison_rows.append(summary)
            print(
                "参数 "
                f"n_estimators={params['n_estimators']}, contamination={params['contamination']}, "
                f"postprocess={postprocess_config['postprocess_name']}: "
                f"score={summary['selection_score']}, f1={summary['avg_f1']}, "
                f"precision={summary['avg_precision']}, recall={summary['avg_recall']}, "
                f"event_hit_rate={summary['event_hit_rate']}"
            )

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        ["selection_score", "avg_f1", "event_hit_rate", "avg_precision", "avg_recall", "avg_anomaly_ratio"],
        ascending=[False, False, False, False, False, True],
    )
    coverage_best_row = comparison_df.iloc[0]
    balanced_best_row = select_balanced_candidate(comparison_df)
    selected_configs_df = pd.DataFrame(
        [
            {
                "selection_mode": "coverage_best",
                "n_estimators": int(coverage_best_row["n_estimators"]),
                "contamination": float(coverage_best_row["contamination"]),
                "postprocess_name": coverage_best_row["postprocess_name"],
                "postprocess_min_streak": int(coverage_best_row["postprocess_min_streak"]),
                "postprocess_smooth_window": int(coverage_best_row["postprocess_smooth_window"]),
                "postprocess_vote_threshold": int(coverage_best_row["postprocess_vote_threshold"]),
                "avg_precision": coverage_best_row["avg_precision"],
                "avg_recall": coverage_best_row["avg_recall"],
                "avg_f1": coverage_best_row["avg_f1"],
                "avg_anomaly_ratio": coverage_best_row["avg_anomaly_ratio"],
                "event_hit_rate": coverage_best_row["event_hit_rate"],
                "selection_score": coverage_best_row["selection_score"],
            },
            {
                "selection_mode": "balanced_best",
                "n_estimators": int(balanced_best_row["n_estimators"]),
                "contamination": float(balanced_best_row["contamination"]),
                "postprocess_name": balanced_best_row["postprocess_name"],
                "postprocess_min_streak": int(balanced_best_row["postprocess_min_streak"]),
                "postprocess_smooth_window": int(balanced_best_row["postprocess_smooth_window"]),
                "postprocess_vote_threshold": int(balanced_best_row["postprocess_vote_threshold"]),
                "avg_precision": balanced_best_row["avg_precision"],
                "avg_recall": balanced_best_row["avg_recall"],
                "avg_f1": balanced_best_row["avg_f1"],
                "avg_anomaly_ratio": balanced_best_row["avg_anomaly_ratio"],
                "event_hit_rate": balanced_best_row["event_hit_rate"],
                "selection_score": balanced_best_row["selection_score"],
            },
        ]
    )
    best_params = {
        "n_estimators": int(balanced_best_row["n_estimators"]),
        "contamination": float(balanced_best_row["contamination"]),
    }
    best_postprocess = {
        "postprocess_name": balanced_best_row["postprocess_name"],
        "min_streak": int(balanced_best_row["postprocess_min_streak"]),
        "smooth_window": int(balanced_best_row["postprocess_smooth_window"]),
        "vote_threshold": int(balanced_best_row["postprocess_vote_threshold"]),
    }
    comparison_df.to_csv(OUTPUT_DIR / "carts_iforest_v3_param_postprocess_comparison.csv", index=False)
    selected_configs_df.to_csv(OUTPUT_DIR / "carts_iforest_v3_selected_configs.csv", index=False)

    print("\n进行特征对比实验（基于验证集）...")
    feature_rows = []
    for feature_name, feature_columns in base.FEATURE_SETS.items():
        feature_train = train_features[feature_columns].to_numpy()
        _, _, summary, _, _ = run_tuned_experiment(
            feature_train,
            validation_features,
            best_params,
            best_postprocess,
            validation_windows,
            feature_columns,
        )
        summary["feature_name"] = feature_name
        feature_rows.append(summary)
        print(
            f"特征组 {feature_name}: score={summary['selection_score']}, "
            f"f1={summary['avg_f1']}, precision={summary['avg_precision']}, "
            f"recall={summary['avg_recall']}, event_hit_rate={summary['event_hit_rate']}"
        )

    feature_df = pd.DataFrame(feature_rows).sort_values(
        ["selection_score", "avg_f1", "event_hit_rate", "avg_precision", "avg_recall", "avg_anomaly_ratio"],
        ascending=[False, False, False, False, False, True],
    )
    best_feature_row = feature_df.iloc[0]
    best_feature_name = best_feature_row["feature_name"]
    selected_feature_columns = base.FEATURE_SETS[best_feature_name]
    feature_df.to_csv(OUTPUT_DIR / "carts_iforest_v3_feature_comparison.csv", index=False)

    print("\n使用最优参数、后处理与特征组进行测试集和挑战集评估 ...")
    final_train = train_features[selected_feature_columns].to_numpy()
    test_result_df, test_run_metrics_df, test_summary, test_event_details_df, test_event_summary_df = run_tuned_experiment(
        final_train,
        test_features,
        best_params,
        best_postprocess,
        test_windows,
        selected_feature_columns,
    )
    challenge_result_df, challenge_run_metrics_df, challenge_summary, challenge_event_details_df, challenge_event_summary_df = run_tuned_experiment(
        final_train,
        challenge_features,
        best_params,
        best_postprocess,
        challenge_windows,
        selected_feature_columns,
    )

    base.save_results_plot(test_result_df, OUTPUT_DIR / "carts_iforest_v3_test_plot.png")
    base.save_results_plot(challenge_result_df, OUTPUT_DIR / "carts_iforest_v3_challenge_plot.png")
    test_result_df.to_csv(OUTPUT_DIR / "carts_iforest_v3_test_results.csv", index=False)
    test_run_metrics_df.to_csv(OUTPUT_DIR / "carts_iforest_v3_test_run_metrics.csv", index=False)
    test_event_details_df.to_csv(OUTPUT_DIR / "carts_iforest_v3_test_event_details.csv", index=False)
    test_event_summary_df.to_csv(OUTPUT_DIR / "carts_iforest_v3_test_event_summary.csv", index=False)
    challenge_result_df.to_csv(OUTPUT_DIR / "carts_iforest_v3_challenge_results.csv", index=False)
    challenge_run_metrics_df.to_csv(OUTPUT_DIR / "carts_iforest_v3_challenge_run_metrics.csv", index=False)
    challenge_event_details_df.to_csv(OUTPUT_DIR / "carts_iforest_v3_challenge_event_details.csv", index=False)
    challenge_event_summary_df.to_csv(OUTPUT_DIR / "carts_iforest_v3_challenge_event_summary.csv", index=False)

    stage_summary_df = pd.DataFrame(
        [
            {
                "stage": "validation",
                "best_n_estimators": best_params["n_estimators"],
                "best_contamination": best_params["contamination"],
                "selected_feature_set": best_feature_name,
                "postprocess_name": best_postprocess["postprocess_name"],
                "postprocess_min_streak": best_postprocess["min_streak"],
                "postprocess_smooth_window": best_postprocess["smooth_window"],
                "postprocess_vote_threshold": best_postprocess["vote_threshold"],
                "avg_precision": best_feature_row["avg_precision"],
                "avg_recall": best_feature_row["avg_recall"],
                "avg_f1": best_feature_row["avg_f1"],
                "avg_anomaly_ratio": best_feature_row["avg_anomaly_ratio"],
                "event_hit_rate": best_feature_row["event_hit_rate"],
                "selection_score": best_feature_row["selection_score"],
            },
            {
                "stage": "test",
                "best_n_estimators": best_params["n_estimators"],
                "best_contamination": best_params["contamination"],
                "selected_feature_set": best_feature_name,
                "postprocess_name": best_postprocess["postprocess_name"],
                "postprocess_min_streak": best_postprocess["min_streak"],
                "postprocess_smooth_window": best_postprocess["smooth_window"],
                "postprocess_vote_threshold": best_postprocess["vote_threshold"],
                "avg_precision": test_summary["avg_precision"],
                "avg_recall": test_summary["avg_recall"],
                "avg_f1": test_summary["avg_f1"],
                "avg_anomaly_ratio": test_summary["avg_anomaly_ratio"],
                "event_hit_rate": test_summary["event_hit_rate"],
                "selection_score": test_summary["selection_score"],
            },
            {
                "stage": "challenge",
                "best_n_estimators": best_params["n_estimators"],
                "best_contamination": best_params["contamination"],
                "selected_feature_set": best_feature_name,
                "postprocess_name": best_postprocess["postprocess_name"],
                "postprocess_min_streak": best_postprocess["min_streak"],
                "postprocess_smooth_window": best_postprocess["smooth_window"],
                "postprocess_vote_threshold": best_postprocess["vote_threshold"],
                "avg_precision": challenge_summary["avg_precision"],
                "avg_recall": challenge_summary["avg_recall"],
                "avg_f1": challenge_summary["avg_f1"],
                "avg_anomaly_ratio": challenge_summary["avg_anomaly_ratio"],
                "event_hit_rate": challenge_summary["event_hit_rate"],
                "selection_score": challenge_summary["selection_score"],
            },
        ]
    )
    stage_summary_df.to_csv(OUTPUT_DIR / "carts_iforest_v3_stage_summary.csv", index=False)
    write_summary_file(stage_summary_df, coverage_df, comparison_df, feature_df, selected_configs_df)

    print(
        "\nV3 高覆盖配置: "
        f"n_estimators={int(coverage_best_row['n_estimators'])}, "
        f"contamination={float(coverage_best_row['contamination'])}, "
        f"postprocess={coverage_best_row['postprocess_name']}"
    )
    print(
        "V3 平衡配置参数: "
        f"n_estimators={best_params['n_estimators']}, contamination={best_params['contamination']}"
    )
    print(
        "V3 平衡配置后处理: "
        f"{best_postprocess['postprocess_name']}, min_streak={best_postprocess['min_streak']}, "
        f"smooth_window={best_postprocess['smooth_window']}, vote_threshold={best_postprocess['vote_threshold']}"
    )
    print(
        "V3 最优特征组: "
        f"{best_feature_name}, selection_score={best_feature_row['selection_score']}, "
        f"avg_f1={best_feature_row['avg_f1']}, event_hit_rate={best_feature_row['event_hit_rate']}"
    )
    print(
        f"V3 测试集: avg_precision={test_summary['avg_precision']}, avg_recall={test_summary['avg_recall']}, "
        f"avg_f1={test_summary['avg_f1']}, event_hit_rate={test_summary['event_hit_rate']}"
    )
    print(
        f"V3 挑战集: avg_precision={challenge_summary['avg_precision']}, avg_recall={challenge_summary['avg_recall']}, "
        f"avg_f1={challenge_summary['avg_f1']}, event_hit_rate={challenge_summary['event_hit_rate']}"
    )
    print(f"V3 结果目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
