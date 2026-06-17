from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent


def plot_stage_summary():
    df = pd.read_csv(BASE_DIR / "carts_iforest_stage_summary.csv")
    df = df[df["stage"].isin(["test", "challenge"])].copy()
    metrics = ["avg_precision", "avg_recall", "avg_f1"]
    labels = ["Precision", "Recall", "F1"]

    x = range(len(df))
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, (metric, label) in enumerate(zip(metrics, labels)):
        positions = [item + (idx - 1) * width for item in x]
        ax.bar(positions, df[metric], width=width, label=label)

    ax.set_xticks(list(x))
    ax.set_xticklabels(df["stage"].tolist())
    ax.set_ylim(0, max(df[metrics].max()) * 1.25)
    ax.set_ylabel("Score")
    ax.set_title("Isolation Forest Diagnostic Summary")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    for idx, row in df.reset_index(drop=True).iterrows():
        ax.text(idx, row["avg_f1"] + 0.01, f"F1={row['avg_f1']:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    output_path = BASE_DIR / "carts_iforest_diagnostic_overview.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_run_breakdown():
    test_df = pd.read_csv(BASE_DIR / "carts_iforest_test_run_metrics.csv")
    challenge_df = pd.read_csv(BASE_DIR / "carts_iforest_challenge_run_metrics.csv")
    test_df["dataset"] = "test"
    challenge_df["dataset"] = "challenge"
    df = pd.concat([test_df, challenge_df], ignore_index=True)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    metrics = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]
    colors = {"test": "#4C78A8", "challenge": "#F58518"}

    x_labels = df["run_id"].tolist()
    x = list(range(len(df)))

    for ax, (metric, label) in zip(axes, metrics):
        bar_colors = [colors[item] for item in df["dataset"]]
        ax.bar(x, df[metric], color=bar_colors)
        ax.set_ylabel(label)
        ax.set_ylim(0, max(df[metric].max() * 1.25, 0.15))
        ax.grid(axis="y", alpha=0.3)
        for idx, value in enumerate(df[metric]):
            ax.text(idx, value + 0.005, f"{value:.3f}", ha="center", fontsize=8, rotation=90)

    axes[0].set_title("Per-run Diagnostic Effect")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(x_labels, rotation=30)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors["test"], label="Test runs"),
        plt.Rectangle((0, 0), 1, 1, color=colors["challenge"], label="Challenge runs"),
    ]
    axes[0].legend(handles=legend_handles, loc="upper right")

    plt.tight_layout()
    output_path = BASE_DIR / "carts_iforest_run_breakdown.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    overview_path = plot_stage_summary()
    breakdown_path = plot_run_breakdown()
    print(f"Saved: {overview_path}")
    print(f"Saved: {breakdown_path}")


if __name__ == "__main__":
    main()
