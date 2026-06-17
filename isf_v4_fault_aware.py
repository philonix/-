from pathlib import Path

import pandas as pd

import isf as base
import isf_v2_postprocess as v2
import isf_v3_tuned as v3


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "optimized_analysis_v4"
EVENT_TOLERANCE_SECONDS = 60
FAULT_AWARE_MAX_ANOMALY_RATIO = 0.4
PARAM_GRID = [
    {"n_estimators": 100, "contamination": 0.02},
    {"n_estimators": 100, "contamination": 0.03},
    {"n_estimators": 100, "contamination": 0.05},
    {"n_estimators": 100, "contamination": 0.08},
    {"n_estimators": 100, "contamination": 0.10},
    {"n_estimators": 200, "contamination": 0.03},
    {"n_estimators": 200, "contamination": 0.05},
    {"n_estimators": 200, "contamination": 0.08},
    {"n_estimators": 200, "contamination": 0.10},
    {"n_estimators": 200, "contamination": 0.12},
    {"n_estimators": 300, "contamination": 0.05},
    {"n_estimators": 300, "contamination": 0.08},
    {"n_estimators": 300, "contamination": 0.10},
    {"n_estimators": 300, "contamination": 0.12},
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
        "postprocess_name": "streak_2_smooth_5_vote_2",
        "min_streak": 2,
        "smooth_window": 5,
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


def extract_fault_type_hit_rate(event_summary_df: pd.DataFrame, fault_type: str) -> float:
    if event_summary_df.empty:
        return 0.0
    row = event_summary_df[
        (event_summary_df["group"] == "fault_type") & (event_summary_df["name"] == fault_type)
    ]
    if row.empty:
        return 0.0
    return float(row.iloc[0]["hit_rate"])


def run_fault_aware_experiment(x_train, eval_features, params, postprocess_config, fault_windows, feature_columns):
    result_df, _, _ = base.run_iforest_experiment(x_train, eval_features, params, fault_windows, feature_columns)
    tuned_df = v3.apply_postprocess(result_df, postprocess_config)
    run_metrics_df = base.evaluate_runs(tuned_df)
    summary = base.summarize_metric_rows(run_metrics_df)
    event_details_df, event_summary_df = v2.evaluate_event_level(
        tuned_df, fault_windows, EVENT_TOLERANCE_SECONDS
    )
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
    summary["kill_hit_rate"] = extract_fault_type_hit_rate(event_summary_df, "kill")
    summary["pause_hit_rate"] = extract_fault_type_hit_rate(event_summary_df, "pause")
    summary["fault_aware_score"] = round(
        summary["avg_f1"] * 0.35
        + summary["avg_precision"] * 0.20
        + summary["avg_recall"] * 0.05
        + summary["event_hit_rate"] * 0.10
        + summary["kill_hit_rate"] * 0.10
        + summary["pause_hit_rate"] * 0.20,
        4,
    )
    return tuned_df, run_metrics_df, summary, event_details_df, event_summary_df


def select_fault_aware_candidate(comparison_df: pd.DataFrame) -> pd.Series:
    candidate_df = comparison_df[
        comparison_df["avg_anomaly_ratio"] <= FAULT_AWARE_MAX_ANOMALY_RATIO
    ].copy()
    if candidate_df.empty:
        candidate_df = comparison_df.copy()

    pause_positive_df = candidate_df[candidate_df["pause_hit_rate"] > 0].copy()
    if not pause_positive_df.empty:
        candidate_df = pause_positive_df

    candidate_df = candidate_df.sort_values(
        [
            "pause_hit_rate",
            "avg_f1",
            "avg_precision",
            "event_hit_rate",
            "kill_hit_rate",
            "avg_anomaly_ratio",
        ],
        ascending=[False, False, False, False, False, True],
    )
    return candidate_df.iloc[0]


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
    fault_aware_best_row = selected_configs_df[selected_configs_df["selection_mode"] == "fault_aware_best"].iloc[0]
    validation_0 = coverage_df[
        (coverage_df["stage"] == "validation") & (coverage_df["tolerance_seconds"] == 0)
    ].iloc[0]
    validation_120 = coverage_df[
        (coverage_df["stage"] == "validation") & (coverage_df["tolerance_seconds"] == 120)
    ].iloc[0]
    top_param_rows = param_postprocess_df.head(5)
    top_feature_rows = feature_df.head(5)

    lines = [
        "Isolation Forest V4 fault-aware 优化说明",
        "",
        "1. V4 的目标",
        "",
        "V4 延续 V3 的平衡思路，但不再只看 overall 指标。",
        "本版重点解决的问题是：",
        "- pause 类型故障命中率长期低于 kill",
        "- 更严格的平衡配置虽然提升 precision，但会过度牺牲 pause 命中",
        "- 因此参数选择需要显式考虑 fault_type 差异",
        "",
        "2. V4 的核心变化",
        "",
        "- 保留 V3 的基础参数与后处理联合搜索框架",
        "- 在验证集上新增 kill_hit_rate 与 pause_hit_rate",
        "- 选择主配置时优先考虑 pause_hit_rate，而不是只看总 event_hit_rate",
        f"- 同时继续约束 avg_anomaly_ratio <= {FAULT_AWARE_MAX_ANOMALY_RATIO}",
        "",
        "3. 覆盖率背景",
        "",
        f"- validation 严格故障窗口覆盖率 = {validation_0['coverage_ratio']}",
        f"- validation 扩展 +/-120 秒后的覆盖率 = {validation_120['coverage_ratio']}",
        "- 说明异常邻域仍然存在，但 V4 更关注不同故障类型之间的检测公平性。",
        "",
        "4. 两类代表配置",
        "",
        (
            f"- 高覆盖配置: n_estimators={int(coverage_best_row['n_estimators'])}, "
            f"contamination={float(coverage_best_row['contamination'])}, "
            f"postprocess={coverage_best_row['postprocess_name']}, "
            f"event_hit_rate={coverage_best_row['event_hit_rate']}, "
            f"pause_hit_rate={coverage_best_row['pause_hit_rate']}, "
            f"anomaly_ratio={coverage_best_row['avg_anomaly_ratio']}"
        ),
        (
            f"- fault-aware 主配置: n_estimators={int(fault_aware_best_row['n_estimators'])}, "
            f"contamination={float(fault_aware_best_row['contamination'])}, "
            f"postprocess={fault_aware_best_row['postprocess_name']}, "
            f"event_hit_rate={fault_aware_best_row['event_hit_rate']}, "
            f"pause_hit_rate={fault_aware_best_row['pause_hit_rate']}, "
            f"anomaly_ratio={fault_aware_best_row['avg_anomaly_ratio']}"
        ),
        "",
        "5. V4 最终推荐配置",
        "",
        f"- n_estimators = {int(validation_row['best_n_estimators'])}",
        f"- contamination = {float(validation_row['best_contamination'])}",
        f"- feature_set = {validation_row['selected_feature_set']}",
        f"- postprocess = {validation_row['postprocess_name']}",
        f"- min_streak = {int(validation_row['postprocess_min_streak'])}",
        f"- smooth_window = {int(validation_row['postprocess_smooth_window'])}",
        f"- vote_threshold = {int(validation_row['postprocess_vote_threshold'])}",
        "",
        "6. 阶段级结果",
        "",
        (
            f"- validation: precision={validation_row['avg_precision']}, recall={validation_row['avg_recall']}, "
            f"f1={validation_row['avg_f1']}, event_hit_rate={validation_row['event_hit_rate']}, "
            f"kill_hit_rate={validation_row['kill_hit_rate']}, pause_hit_rate={validation_row['pause_hit_rate']}"
        ),
        (
            f"- test: precision={test_row['avg_precision']}, recall={test_row['avg_recall']}, "
            f"f1={test_row['avg_f1']}, event_hit_rate={test_row['event_hit_rate']}, "
            f"kill_hit_rate={test_row['kill_hit_rate']}, pause_hit_rate={test_row['pause_hit_rate']}"
        ),
        (
            f"- challenge: precision={challenge_row['avg_precision']}, recall={challenge_row['avg_recall']}, "
            f"f1={challenge_row['avg_f1']}, event_hit_rate={challenge_row['event_hit_rate']}, "
            f"kill_hit_rate={challenge_row['kill_hit_rate']}, pause_hit_rate={challenge_row['pause_hit_rate']}"
        ),
        "",
        "7. 验证集前 5 组参数 + 后处理结果",
        "",
    ]

    for _, row in top_param_rows.iterrows():
        lines.append(
            (
                f"- n_estimators={int(row['n_estimators'])}, contamination={float(row['contamination'])}, "
                f"postprocess={row['postprocess_name']}, score={row['fault_aware_score']}, "
                f"precision={row['avg_precision']}, recall={row['avg_recall']}, f1={row['avg_f1']}, "
                f"event_hit_rate={row['event_hit_rate']}, kill_hit_rate={row['kill_hit_rate']}, "
                f"pause_hit_rate={row['pause_hit_rate']}, anomaly_ratio={row['avg_anomaly_ratio']}"
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
                f"- feature={row['feature_name']}, score={row['fault_aware_score']}, "
                f"precision={row['avg_precision']}, recall={row['avg_recall']}, f1={row['avg_f1']}, "
                f"event_hit_rate={row['event_hit_rate']}, kill_hit_rate={row['kill_hit_rate']}, "
                f"pause_hit_rate={row['pause_hit_rate']}, anomaly_ratio={row['avg_anomaly_ratio']}"
            )
        )

    lines.extend(
        [
            "",
            "9. 当前结论",
            "",
            "- V4 不是重新追求全局高覆盖，而是让 pause 不至于在选型阶段被系统性牺牲。",
            "- 如果 V4 的 pause_hit_rate 提升且 F1 仍稳定，就说明 fault-aware 选型是有效的。",
            "- 如果 pause 提升但 precision 明显恶化，下一步应做更细的 pause 专项特征或窗口策略。",
            "",
        ]
    )

    (OUTPUT_DIR / "v4_fault_aware_summary.txt").write_text("\n".join(lines), encoding="utf-8")


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

    coverage_df = v3.build_window_coverage_summary(
        validation_features,
        test_features,
        challenge_features,
        validation_windows,
        test_windows,
        challenge_windows,
    )
    coverage_df.to_csv(OUTPUT_DIR / "carts_iforest_v4_window_coverage_summary.csv", index=False)

    print("\n进行参数 + 后处理联合对比实验（fault-aware，基于验证集）...")
    comparison_rows = []
    x_train_full = train_features[base.FEATURE_COLUMNS].to_numpy()
    for params in PARAM_GRID:
        for postprocess_config in POSTPROCESS_GRID:
            _, _, summary, _, _ = run_fault_aware_experiment(
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
                f"score={summary['fault_aware_score']}, f1={summary['avg_f1']}, "
                f"pause_hit_rate={summary['pause_hit_rate']}, kill_hit_rate={summary['kill_hit_rate']}, "
                f"precision={summary['avg_precision']}, event_hit_rate={summary['event_hit_rate']}"
            )

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        [
            "fault_aware_score",
            "pause_hit_rate",
            "avg_f1",
            "avg_precision",
            "event_hit_rate",
            "avg_anomaly_ratio",
        ],
        ascending=[False, False, False, False, False, True],
    )
    coverage_best_row = comparison_df.iloc[0]
    fault_aware_best_row = select_fault_aware_candidate(comparison_df)
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
                "kill_hit_rate": coverage_best_row["kill_hit_rate"],
                "pause_hit_rate": coverage_best_row["pause_hit_rate"],
                "fault_aware_score": coverage_best_row["fault_aware_score"],
            },
            {
                "selection_mode": "fault_aware_best",
                "n_estimators": int(fault_aware_best_row["n_estimators"]),
                "contamination": float(fault_aware_best_row["contamination"]),
                "postprocess_name": fault_aware_best_row["postprocess_name"],
                "postprocess_min_streak": int(fault_aware_best_row["postprocess_min_streak"]),
                "postprocess_smooth_window": int(fault_aware_best_row["postprocess_smooth_window"]),
                "postprocess_vote_threshold": int(fault_aware_best_row["postprocess_vote_threshold"]),
                "avg_precision": fault_aware_best_row["avg_precision"],
                "avg_recall": fault_aware_best_row["avg_recall"],
                "avg_f1": fault_aware_best_row["avg_f1"],
                "avg_anomaly_ratio": fault_aware_best_row["avg_anomaly_ratio"],
                "event_hit_rate": fault_aware_best_row["event_hit_rate"],
                "kill_hit_rate": fault_aware_best_row["kill_hit_rate"],
                "pause_hit_rate": fault_aware_best_row["pause_hit_rate"],
                "fault_aware_score": fault_aware_best_row["fault_aware_score"],
            },
        ]
    )
    best_params = {
        "n_estimators": int(fault_aware_best_row["n_estimators"]),
        "contamination": float(fault_aware_best_row["contamination"]),
    }
    best_postprocess = {
        "postprocess_name": fault_aware_best_row["postprocess_name"],
        "min_streak": int(fault_aware_best_row["postprocess_min_streak"]),
        "smooth_window": int(fault_aware_best_row["postprocess_smooth_window"]),
        "vote_threshold": int(fault_aware_best_row["postprocess_vote_threshold"]),
    }
    comparison_df.to_csv(OUTPUT_DIR / "carts_iforest_v4_param_postprocess_comparison.csv", index=False)
    selected_configs_df.to_csv(OUTPUT_DIR / "carts_iforest_v4_selected_configs.csv", index=False)

    print("\n进行特征对比实验（fault-aware，基于验证集）...")
    feature_rows = []
    for feature_name, feature_columns in base.FEATURE_SETS.items():
        feature_train = train_features[feature_columns].to_numpy()
        _, _, summary, _, _ = run_fault_aware_experiment(
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
            f"特征组 {feature_name}: score={summary['fault_aware_score']}, "
            f"f1={summary['avg_f1']}, pause_hit_rate={summary['pause_hit_rate']}, "
            f"kill_hit_rate={summary['kill_hit_rate']}, precision={summary['avg_precision']}"
        )

    feature_df = pd.DataFrame(feature_rows).sort_values(
        [
            "fault_aware_score",
            "pause_hit_rate",
            "avg_f1",
            "avg_precision",
            "event_hit_rate",
            "avg_anomaly_ratio",
        ],
        ascending=[False, False, False, False, False, True],
    )
    best_feature_row = feature_df.iloc[0]
    best_feature_name = best_feature_row["feature_name"]
    selected_feature_columns = base.FEATURE_SETS[best_feature_name]
    feature_df.to_csv(OUTPUT_DIR / "carts_iforest_v4_feature_comparison.csv", index=False)

    print("\n使用最优参数、后处理与特征组进行测试集和挑战集评估 ...")
    final_train = train_features[selected_feature_columns].to_numpy()
    test_result_df, test_run_metrics_df, test_summary, test_event_details_df, test_event_summary_df = run_fault_aware_experiment(
        final_train,
        test_features,
        best_params,
        best_postprocess,
        test_windows,
        selected_feature_columns,
    )
    challenge_result_df, challenge_run_metrics_df, challenge_summary, challenge_event_details_df, challenge_event_summary_df = run_fault_aware_experiment(
        final_train,
        challenge_features,
        best_params,
        best_postprocess,
        challenge_windows,
        selected_feature_columns,
    )

    base.save_results_plot(test_result_df, OUTPUT_DIR / "carts_iforest_v4_test_plot.png")
    base.save_results_plot(challenge_result_df, OUTPUT_DIR / "carts_iforest_v4_challenge_plot.png")
    test_result_df.to_csv(OUTPUT_DIR / "carts_iforest_v4_test_results.csv", index=False)
    test_run_metrics_df.to_csv(OUTPUT_DIR / "carts_iforest_v4_test_run_metrics.csv", index=False)
    test_event_details_df.to_csv(OUTPUT_DIR / "carts_iforest_v4_test_event_details.csv", index=False)
    test_event_summary_df.to_csv(OUTPUT_DIR / "carts_iforest_v4_test_event_summary.csv", index=False)
    challenge_result_df.to_csv(OUTPUT_DIR / "carts_iforest_v4_challenge_results.csv", index=False)
    challenge_run_metrics_df.to_csv(OUTPUT_DIR / "carts_iforest_v4_challenge_run_metrics.csv", index=False)
    challenge_event_details_df.to_csv(OUTPUT_DIR / "carts_iforest_v4_challenge_event_details.csv", index=False)
    challenge_event_summary_df.to_csv(OUTPUT_DIR / "carts_iforest_v4_challenge_event_summary.csv", index=False)

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
                "kill_hit_rate": best_feature_row["kill_hit_rate"],
                "pause_hit_rate": best_feature_row["pause_hit_rate"],
                "fault_aware_score": best_feature_row["fault_aware_score"],
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
                "kill_hit_rate": test_summary["kill_hit_rate"],
                "pause_hit_rate": test_summary["pause_hit_rate"],
                "fault_aware_score": test_summary["fault_aware_score"],
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
                "kill_hit_rate": challenge_summary["kill_hit_rate"],
                "pause_hit_rate": challenge_summary["pause_hit_rate"],
                "fault_aware_score": challenge_summary["fault_aware_score"],
            },
        ]
    )
    stage_summary_df.to_csv(OUTPUT_DIR / "carts_iforest_v4_stage_summary.csv", index=False)
    write_summary_file(stage_summary_df, coverage_df, comparison_df, feature_df, selected_configs_df)

    print(
        "\nV4 高覆盖配置: "
        f"n_estimators={int(coverage_best_row['n_estimators'])}, "
        f"contamination={float(coverage_best_row['contamination'])}, "
        f"postprocess={coverage_best_row['postprocess_name']}"
    )
    print(
        "V4 fault-aware 配置参数: "
        f"n_estimators={best_params['n_estimators']}, contamination={best_params['contamination']}"
    )
    print(
        "V4 fault-aware 配置后处理: "
        f"{best_postprocess['postprocess_name']}, min_streak={best_postprocess['min_streak']}, "
        f"smooth_window={best_postprocess['smooth_window']}, vote_threshold={best_postprocess['vote_threshold']}"
    )
    print(
        "V4 最优特征组: "
        f"{best_feature_name}, score={best_feature_row['fault_aware_score']}, "
        f"f1={best_feature_row['avg_f1']}, pause_hit_rate={best_feature_row['pause_hit_rate']}"
    )
    print(
        f"V4 测试集: avg_precision={test_summary['avg_precision']}, avg_recall={test_summary['avg_recall']}, "
        f"avg_f1={test_summary['avg_f1']}, event_hit_rate={test_summary['event_hit_rate']}, "
        f"pause_hit_rate={test_summary['pause_hit_rate']}"
    )
    print(
        f"V4 挑战集: avg_precision={challenge_summary['avg_precision']}, avg_recall={challenge_summary['avg_recall']}, "
        f"avg_f1={challenge_summary['avg_f1']}, event_hit_rate={challenge_summary['event_hit_rate']}, "
        f"pause_hit_rate={challenge_summary['pause_hit_rate']}"
    )
    print(f"V4 结果目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
