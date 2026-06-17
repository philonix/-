from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(__file__).resolve().parent
TABLE_DIR = OUTPUT_DIR / "tables"
FIG_DIR = OUTPUT_DIR / "figures"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def build_feature_ablation_table() -> pd.DataFrame:
    records = []

    sources = [
        (
            "V1",
            BASE_DIR / "refined_analysis" / "carts_iforest_feature_comparison.csv",
            {"event_hit_rate": None, "kill_hit_rate": None, "pause_hit_rate": None},
        ),
        (
            "V2",
            BASE_DIR / "optimized_analysis_v2" / "carts_iforest_v2_feature_comparison.csv",
            {"kill_hit_rate": None, "pause_hit_rate": None},
        ),
        (
            "V3",
            BASE_DIR / "optimized_analysis_v3" / "carts_iforest_v3_feature_comparison.csv",
            {"kill_hit_rate": None, "pause_hit_rate": None},
        ),
        (
            "V4",
            BASE_DIR / "optimized_analysis_v4" / "carts_iforest_v4_feature_comparison.csv",
            {},
        ),
    ]

    for version, path, defaults in sources:
        df = read_csv(path).copy()
        for column, value in defaults.items():
            if column not in df.columns:
                df[column] = value
        keep_cols = [
            "feature_name",
            "avg_precision",
            "avg_recall",
            "avg_f1",
            "avg_anomaly_ratio",
            "event_hit_rate",
            "kill_hit_rate",
            "pause_hit_rate",
        ]
        df = df[keep_cols].copy()
        for _, row in df.iterrows():
            record = row.to_dict()
            record["version"] = version
            records.append(record)

    result = pd.DataFrame(
        records,
        columns=[
            "version",
            "feature_name",
            "avg_precision",
            "avg_recall",
            "avg_f1",
            "avg_anomaly_ratio",
            "event_hit_rate",
            "kill_hit_rate",
            "pause_hit_rate",
        ],
    )
    result.to_csv(TABLE_DIR / "feature_ablation_comparison.csv", index=False, encoding="utf-8-sig")
    return result


def build_postprocess_tradeoff_table() -> pd.DataFrame:
    rows = [
        {
            "version": "V2",
            "selection_mode": "fixed_main",
            "contamination": 0.08,
            "postprocess_name": "streak_2",
            "postprocess_min_streak": 2,
            "postprocess_smooth_window": 1,
            "postprocess_vote_threshold": 1,
            "avg_precision": 0.0639,
            "avg_recall": 0.3703,
            "avg_f1": 0.1072,
            "avg_anomaly_ratio": 0.3530,
            "event_hit_rate": 0.7778,
            "kill_hit_rate": None,
            "pause_hit_rate": None,
        }
    ]

    v3 = read_csv(BASE_DIR / "optimized_analysis_v3" / "carts_iforest_v3_selected_configs.csv")
    for _, row in v3.iterrows():
        rows.append(
            {
                "version": "V3",
                "selection_mode": row["selection_mode"],
                "contamination": row["contamination"],
                "postprocess_name": row["postprocess_name"],
                "postprocess_min_streak": row["postprocess_min_streak"],
                "postprocess_smooth_window": row["postprocess_smooth_window"],
                "postprocess_vote_threshold": row["postprocess_vote_threshold"],
                "avg_precision": row["avg_precision"],
                "avg_recall": row["avg_recall"],
                "avg_f1": row["avg_f1"],
                "avg_anomaly_ratio": row["avg_anomaly_ratio"],
                "event_hit_rate": row["event_hit_rate"],
                "kill_hit_rate": None,
                "pause_hit_rate": None,
            }
        )

    v4 = read_csv(BASE_DIR / "optimized_analysis_v4" / "carts_iforest_v4_selected_configs.csv")
    for _, row in v4.iterrows():
        rows.append(
            {
                "version": "V4",
                "selection_mode": row["selection_mode"],
                "contamination": row["contamination"],
                "postprocess_name": row["postprocess_name"],
                "postprocess_min_streak": row["postprocess_min_streak"],
                "postprocess_smooth_window": row["postprocess_smooth_window"],
                "postprocess_vote_threshold": row["postprocess_vote_threshold"],
                "avg_precision": row["avg_precision"],
                "avg_recall": row["avg_recall"],
                "avg_f1": row["avg_f1"],
                "avg_anomaly_ratio": row["avg_anomaly_ratio"],
                "event_hit_rate": row["event_hit_rate"],
                "kill_hit_rate": row.get("kill_hit_rate"),
                "pause_hit_rate": row.get("pause_hit_rate"),
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(TABLE_DIR / "postprocess_tradeoff_summary.csv", index=False, encoding="utf-8-sig")
    return result


def plot_version_comparison() -> None:
    df = read_csv(TABLE_DIR / "v1_to_v4_stage_comparison.csv")
    plot_df = df[df["stage"].isin(["test", "challenge"])].copy()
    versions = ["V1", "V2", "V3", "V4"]
    metrics = ["avg_f1", "avg_precision", "avg_recall"]
    stages = ["test", "challenge"]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    colors = {"V1": "#4e79a7", "V2": "#f28e2b", "V3": "#59a14f", "V4": "#e15759"}

    for row_idx, stage in enumerate(stages):
        stage_df = plot_df[plot_df["stage"] == stage].set_index("version")
        for col_idx, metric in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            values = [stage_df.loc[v, metric] for v in versions]
            ax.bar(versions, values, color=[colors[v] for v in versions])
            ax.set_title(f"{stage} {metric}")
            ax.set_ylim(0, max(values) * 1.25 if max(values) > 0 else 1)
            ax.grid(axis="y", alpha=0.3)
            for x, value in enumerate(values):
                ax.text(x, value + 0.005, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("V1-V4 Stage Comparison", fontsize=14)
    fig.savefig(FIG_DIR / "v1_to_v4_stage_comparison.png", dpi=200)
    plt.close(fig)


def plot_feature_ablation() -> None:
    df = read_csv(TABLE_DIR / "feature_ablation_comparison.csv")
    versions = ["V1", "V2", "V3", "V4"]
    feature_order = ["rate_plus_rolling", "full_features", "stat_enhanced", "cpu_memory", "rate_only"]
    pivot = (
        df.pivot_table(index="feature_name", columns="version", values="avg_f1", aggfunc="first")
        .reindex(feature_order)
        .reindex(columns=versions)
    )

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    pivot.plot(kind="bar", ax=ax, width=0.8)
    ax.set_title("Feature Ablation Comparison (Validation Avg F1)")
    ax.set_xlabel("Feature Set")
    ax.set_ylabel("Avg F1")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Version")
    plt.xticks(rotation=20, ha="right")
    fig.savefig(FIG_DIR / "feature_ablation_comparison.png", dpi=200)
    plt.close(fig)


def plot_postprocess_tradeoff() -> None:
    df = read_csv(TABLE_DIR / "postprocess_tradeoff_summary.csv")
    labels = [
        f"{row.version}-{row.selection_mode}"
        for row in df[["version", "selection_mode"]].itertuples(index=False)
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    axes[0].bar(labels, df["avg_f1"], color="#4e79a7")
    axes[0].set_title("Postprocess Tradeoff: Avg F1")
    axes[0].set_ylabel("Avg F1")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(labels, df["avg_anomaly_ratio"], color="#e15759")
    axes[1].set_title("Postprocess Tradeoff: Anomaly Ratio")
    axes[1].set_ylabel("Avg Anomaly Ratio")
    axes[1].grid(axis="y", alpha=0.3)

    for ax in axes:
        ax.tick_params(axis="x", rotation=20)

    fig.savefig(FIG_DIR / "postprocess_tradeoff_comparison.png", dpi=200)
    plt.close(fig)


def plot_final_recommendation() -> None:
    df = read_csv(TABLE_DIR / "v1_to_v4_stage_comparison.csv")
    compare_df = df[df["version"].isin(["V3", "V4"]) & df["stage"].isin(["test", "challenge"])].copy()

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    colors = {"V3": "#59a14f", "V4": "#e15759"}
    metrics = [
        ("avg_f1", "F1"),
        ("avg_anomaly_ratio", "Anomaly Ratio"),
        ("event_hit_rate", "Event Hit Rate"),
        ("pause_hit_rate", "Pause Hit Rate"),
    ]

    for ax, (metric, title) in zip(axes.flatten(), metrics):
        plot_df = compare_df.pivot(index="stage", columns="version", values=metric)
        stages = ["test", "challenge"]
        x = range(len(stages))
        width = 0.35
        v3_vals = [plot_df.loc[s, "V3"] for s in stages]
        v4_vals = [plot_df.loc[s, "V4"] for s in stages]
        ax.bar([i - width / 2 for i in x], v3_vals, width=width, label="V3", color=colors["V3"])
        ax.bar([i + width / 2 for i in x], v4_vals, width=width, label="V4", color=colors["V4"])
        ax.set_xticks(list(x))
        ax.set_xticklabels(stages)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ymax = max(v3_vals + v4_vals)
        ax.set_ylim(0, ymax * 1.28 if ymax > 0 else 1)
        for idx, value in enumerate(v3_vals):
            ax.text(idx - width / 2, value + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
        for idx, value in enumerate(v4_vals):
            ax.text(idx + width / 2, value + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    axes[0, 0].legend(loc="upper right")
    fig.suptitle("Final Recommendation: Why V3 Is the Main Version", fontsize=14)
    fig.savefig(FIG_DIR / "final_recommendation_summary.png", dpi=200)
    plt.close(fig)


def main() -> None:
    TABLE_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)
    build_feature_ablation_table()
    build_postprocess_tradeoff_table()
    plot_version_comparison()
    plot_feature_ablation()
    plot_postprocess_tradeoff()
    plot_final_recommendation()


if __name__ == "__main__":
    main()
