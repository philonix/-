from pathlib import Path

import pandas as pd

import isf as base


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "optimized_analysis_v2"
POSTPROCESS_MIN_STREAK = 2
EVENT_TOLERANCE_SECONDS = 60


def apply_consecutive_filter(result_df: pd.DataFrame, min_streak: int) -> pd.DataFrame:
    filtered = result_df.copy().sort_values(["run_id", "pod", "datetime"]).reset_index(drop=True)
    filtered["is_anomaly_raw"] = filtered["is_anomaly"]
    filtered["is_anomaly"] = 0

    for (_, _), group_df in filtered.groupby(["run_id", "pod"], sort=False):
        raw_flags = group_df["is_anomaly_raw"].tolist()
        keep_flags = [0] * len(raw_flags)
        start_idx = None
        for idx, flag in enumerate(raw_flags + [0]):
            if idx < len(raw_flags) and flag == 1 and start_idx is None:
                start_idx = idx
            elif (idx == len(raw_flags) or flag == 0) and start_idx is not None:
                streak_length = idx - start_idx
                if streak_length >= min_streak:
                    for mark_idx in range(start_idx, idx):
                        keep_flags[mark_idx] = 1
                start_idx = None
        filtered.loc[group_df.index, "is_anomaly"] = keep_flags

    return filtered


def evaluate_event_level(result_df: pd.DataFrame, fault_windows: dict, tolerance_seconds: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    tolerance = pd.Timedelta(seconds=tolerance_seconds)

    for run_id, windows in fault_windows.items():
        run_df = result_df[result_df["run_id"] == run_id]
        for event_idx, window in enumerate(windows, start=1):
            start = pd.to_datetime(window["start"])
            end = pd.to_datetime(window["end"])
            expanded_start = start - tolerance
            expanded_end = end + tolerance
            hit = (
                (run_df["is_anomaly"] == 1)
                & (run_df["datetime"] >= expanded_start)
                & (run_df["datetime"] <= expanded_end)
            ).any()
            rows.append(
                {
                    "run_id": run_id,
                    "event_id": f"event_{event_idx:02d}",
                    "fault_type": window.get("fault_type", "unknown"),
                    "window_start": start.strftime("%Y-%m-%d %H:%M:%S"),
                    "window_end": end.strftime("%Y-%m-%d %H:%M:%S"),
                    "expanded_start": expanded_start.strftime("%Y-%m-%d %H:%M:%S"),
                    "expanded_end": expanded_end.strftime("%Y-%m-%d %H:%M:%S"),
                    "hit": int(bool(hit)),
                }
            )

    details_df = pd.DataFrame(rows)
    if details_df.empty:
        return details_df, pd.DataFrame()

    summary_rows = []
    overall_hit_rate = details_df["hit"].mean()
    summary_rows.append(
        {
            "group": "overall",
            "name": "all",
            "events": int(len(details_df)),
            "hits": int(details_df["hit"].sum()),
            "hit_rate": round(float(overall_hit_rate), 4),
        }
    )
    for run_id, group_df in details_df.groupby("run_id"):
        summary_rows.append(
            {
                "group": "run_id",
                "name": run_id,
                "events": int(len(group_df)),
                "hits": int(group_df["hit"].sum()),
                "hit_rate": round(float(group_df["hit"].mean()), 4),
            }
        )
    for fault_type, group_df in details_df.groupby("fault_type"):
        summary_rows.append(
            {
                "group": "fault_type",
                "name": fault_type,
                "events": int(len(group_df)),
                "hits": int(group_df["hit"].sum()),
                "hit_rate": round(float(group_df["hit"].mean()), 4),
            }
        )
    return details_df, pd.DataFrame(summary_rows)


def run_postprocessed_experiment(x_train, eval_features, params, fault_windows, feature_columns):
    result_df, _, _ = base.run_iforest_experiment(x_train, eval_features, params, fault_windows, feature_columns)
    filtered_df = apply_consecutive_filter(result_df, POSTPROCESS_MIN_STREAK)
    run_metrics_df = base.evaluate_runs(filtered_df)
    summary = base.summarize_metric_rows(run_metrics_df)
    event_details_df, event_summary_df = evaluate_event_level(filtered_df, fault_windows, EVENT_TOLERANCE_SECONDS)
    summary.update(params)
    summary["feature_set"] = ",".join(feature_columns)
    summary["postprocess_min_streak"] = POSTPROCESS_MIN_STREAK
    if not event_summary_df.empty:
        overall_row = event_summary_df[
            (event_summary_df["group"] == "overall") & (event_summary_df["name"] == "all")
        ].iloc[0]
        summary["event_hit_rate"] = overall_row["hit_rate"]
    else:
        summary["event_hit_rate"] = 0.0
    return filtered_df, run_metrics_df, summary, event_details_df, event_summary_df


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

    print("\n进行参数对比实验（后处理版，基于验证集平均 F1）...")
    param_rows = []
    for params in base.PARAM_GRID:
        result_df, run_metrics_df, summary, _, _ = run_postprocessed_experiment(
            train_features[base.FEATURE_COLUMNS].to_numpy(),
            validation_features,
            params,
            validation_windows,
            base.FEATURE_COLUMNS,
        )
        summary["stage"] = "validation"
        param_rows.append(summary)
        print(
            f"参数 n_estimators={params['n_estimators']}, contamination={params['contamination']}: "
            f"avg_precision={summary['avg_precision']}, avg_recall={summary['avg_recall']}, "
            f"avg_f1={summary['avg_f1']}, event_hit_rate={summary['event_hit_rate']}"
        )

    param_df = pd.DataFrame(param_rows).sort_values(
        ["avg_f1", "avg_precision", "avg_recall", "event_hit_rate"],
        ascending=[False, False, False, False],
    )
    best_param_row = param_df.iloc[0]
    best_params = {
        "n_estimators": int(best_param_row["n_estimators"]),
        "contamination": float(best_param_row["contamination"]),
    }

    print("\n进行特征对比实验（后处理版，基于验证集）...")
    feature_rows = []
    for feature_name, feature_columns in base.FEATURE_SETS.items():
        result_df, run_metrics_df, summary, _, _ = run_postprocessed_experiment(
            train_features[feature_columns].to_numpy(),
            validation_features,
            best_params,
            validation_windows,
            feature_columns,
        )
        summary["feature_name"] = feature_name
        feature_rows.append(summary)
        print(
            f"特征组 {feature_name}: avg_precision={summary['avg_precision']}, "
            f"avg_recall={summary['avg_recall']}, avg_f1={summary['avg_f1']}, "
            f"event_hit_rate={summary['event_hit_rate']}"
        )

    feature_df = pd.DataFrame(feature_rows).sort_values(
        ["avg_f1", "avg_precision", "avg_recall", "event_hit_rate"],
        ascending=[False, False, False, False],
    )
    best_feature_row = feature_df.iloc[0]
    best_feature_name = best_feature_row["feature_name"]
    selected_feature_columns = base.FEATURE_SETS[best_feature_name]

    print("\n使用最优参数与特征组进行测试集和挑战集评估 ...")
    final_train = train_features[selected_feature_columns].to_numpy()

    test_result_df, test_run_metrics_df, test_summary, test_event_details_df, test_event_summary_df = run_postprocessed_experiment(
        final_train, test_features, best_params, test_windows, selected_feature_columns
    )
    challenge_result_df, challenge_run_metrics_df, challenge_summary, challenge_event_details_df, challenge_event_summary_df = run_postprocessed_experiment(
        final_train, challenge_features, best_params, challenge_windows, selected_feature_columns
    )

    base.save_results_plot(test_result_df, OUTPUT_DIR / "carts_iforest_v2_test_plot.png")
    base.save_results_plot(challenge_result_df, OUTPUT_DIR / "carts_iforest_v2_challenge_plot.png")

    param_df.to_csv(OUTPUT_DIR / "carts_iforest_v2_param_comparison.csv", index=False)
    feature_df.to_csv(OUTPUT_DIR / "carts_iforest_v2_feature_comparison.csv", index=False)
    test_result_df.to_csv(OUTPUT_DIR / "carts_iforest_v2_test_results.csv", index=False)
    test_run_metrics_df.to_csv(OUTPUT_DIR / "carts_iforest_v2_test_run_metrics.csv", index=False)
    test_event_details_df.to_csv(OUTPUT_DIR / "carts_iforest_v2_test_event_details.csv", index=False)
    test_event_summary_df.to_csv(OUTPUT_DIR / "carts_iforest_v2_test_event_summary.csv", index=False)
    challenge_result_df.to_csv(OUTPUT_DIR / "carts_iforest_v2_challenge_results.csv", index=False)
    challenge_run_metrics_df.to_csv(OUTPUT_DIR / "carts_iforest_v2_challenge_run_metrics.csv", index=False)
    challenge_event_details_df.to_csv(OUTPUT_DIR / "carts_iforest_v2_challenge_event_details.csv", index=False)
    challenge_event_summary_df.to_csv(OUTPUT_DIR / "carts_iforest_v2_challenge_event_summary.csv", index=False)

    stage_summary_df = pd.DataFrame(
        [
            {
                "stage": "validation",
                "best_n_estimators": best_params["n_estimators"],
                "best_contamination": best_params["contamination"],
                "selected_feature_set": best_feature_name,
                "avg_precision": best_feature_row["avg_precision"],
                "avg_recall": best_feature_row["avg_recall"],
                "avg_f1": best_feature_row["avg_f1"],
                "event_hit_rate": best_feature_row["event_hit_rate"],
                "postprocess_min_streak": POSTPROCESS_MIN_STREAK,
            },
            {
                "stage": "test",
                "best_n_estimators": best_params["n_estimators"],
                "best_contamination": best_params["contamination"],
                "selected_feature_set": best_feature_name,
                "avg_precision": test_summary["avg_precision"],
                "avg_recall": test_summary["avg_recall"],
                "avg_f1": test_summary["avg_f1"],
                "event_hit_rate": test_summary["event_hit_rate"],
                "postprocess_min_streak": POSTPROCESS_MIN_STREAK,
            },
            {
                "stage": "challenge",
                "best_n_estimators": best_params["n_estimators"],
                "best_contamination": best_params["contamination"],
                "selected_feature_set": best_feature_name,
                "avg_precision": challenge_summary["avg_precision"],
                "avg_recall": challenge_summary["avg_recall"],
                "avg_f1": challenge_summary["avg_f1"],
                "event_hit_rate": challenge_summary["event_hit_rate"],
                "postprocess_min_streak": POSTPROCESS_MIN_STREAK,
            },
        ]
    )
    stage_summary_df.to_csv(OUTPUT_DIR / "carts_iforest_v2_stage_summary.csv", index=False)

    print(
        "\nV2 最优参数: "
        f"n_estimators={best_params['n_estimators']}, contamination={best_params['contamination']}"
    )
    print(
        "V2 最优特征组: "
        f"{best_feature_name}, avg_f1={best_feature_row['avg_f1']}, event_hit_rate={best_feature_row['event_hit_rate']}"
    )
    print(
        f"V2 测试集: avg_precision={test_summary['avg_precision']}, avg_recall={test_summary['avg_recall']}, "
        f"avg_f1={test_summary['avg_f1']}, event_hit_rate={test_summary['event_hit_rate']}"
    )
    print(
        f"V2 挑战集: avg_precision={challenge_summary['avg_precision']}, avg_recall={challenge_summary['avg_recall']}, "
        f"avg_f1={challenge_summary['avg_f1']}, event_hit_rate={challenge_summary['event_hit_rate']}"
    )
    print(f"V2 结果目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
