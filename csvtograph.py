import argparse
import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
EXPORTS_DIR = BASE_DIR / "exports"
DEFAULT_RUN_ID = "run_001"
DEFAULT_METRIC = "cpu"
DEFAULT_GRAFANA_URL = "http://127.0.0.1:1393/"
DEFAULT_CONTEXT_MINUTES = 2


def load_metric_csv(run_id: str, metric_name: str) -> tuple[pd.DataFrame, Path]:
    csv_path = EXPORTS_DIR / run_id / f"{metric_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"未找到导出文件: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"导出文件为空: {csv_path}")
    return df, csv_path


def load_run_metadata(run_id: str) -> dict:
    metadata_path = EXPORTS_DIR / run_id / "run_metadata.json"
    if not metadata_path.exists():
        return {"events": []}
    with metadata_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def parse_metric_dict(metric_text: str) -> dict:
    try:
        return json.loads(metric_text)
    except Exception:
        return {}


def prepare_dataframe(df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    df["metric_parsed"] = df["metric_dict"].apply(parse_metric_dict)
    df["pod"] = df["metric_parsed"].apply(lambda item: item.get("pod", "unknown"))
    df["namespace"] = df["metric_parsed"].apply(lambda item: item.get("namespace", "unknown"))
    df["service"] = df["pod"].astype(str).str.split("-").str[0]
    df = df.dropna(subset=["timestamp", "value"]).sort_values(["pod", "datetime"])

    if metric_name == "cpu":
        df["delta_time"] = df.groupby("pod")["timestamp"].diff()
        df["delta_value"] = df.groupby("pod")["value"].diff()
        df["plot_value"] = df["delta_value"] / df["delta_time"]
        df["plot_value"] = df["plot_value"].where(df["delta_time"] > 0)
        df["plot_label"] = "CPU Rate (seconds/s)"
    else:
        df["plot_value"] = df["value"]
        df["plot_label"] = metric_name

    return df


def filter_dataframe(df: pd.DataFrame, service: str, pod_keyword: str) -> pd.DataFrame:
    filtered = df
    if service:
        filtered = filtered[filtered["service"] == service]
    if pod_keyword:
        filtered = filtered[filtered["pod"].str.contains(pod_keyword, case=False, na=False)]
    if filtered.empty:
        raise ValueError("筛选后没有数据，请检查 run_id、service 或 pod 关键字。")
    return filtered


def filter_events(events: list[dict], service: str) -> list[dict]:
    if not service:
        return events
    return [event for event in events if event.get("service") == service]


def get_run_start_epoch(metadata: dict, df: pd.DataFrame) -> float:
    epoch = metadata.get("window_start_epoch")
    if epoch is not None:
        return float(epoch)
    return float(df["timestamp"].min())


def compute_focus_window(events: list[dict], context_minutes: int) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    windows = []
    for event in events:
        start_text = event.get("expected_fault_window_start") or event.get("start_time")
        end_text = event.get("expected_fault_window_end") or event.get("end_time")
        if not start_text or not end_text:
            continue
        windows.append((pd.to_datetime(start_text), pd.to_datetime(end_text)))

    if not windows:
        return None

    start = min(item[0] for item in windows) - pd.Timedelta(minutes=context_minutes)
    end = max(item[1] for item in windows) + pd.Timedelta(minutes=context_minutes)
    return start, end


def limit_time_window(df: pd.DataFrame, focus_window: tuple[pd.Timestamp, pd.Timestamp] | None) -> pd.DataFrame:
    if not focus_window:
        return df
    start, end = focus_window
    limited = df[(df["datetime"] >= start) & (df["datetime"] <= end)]
    return limited if not limited.empty else df


def add_event_windows(ax, events: list[dict], service: str, relative_time: bool, run_start: pd.Timestamp) -> None:
    used_label = False
    for event in filter_events(events, service):
        start_text = event.get("expected_fault_window_start") or event.get("start_time")
        end_text = event.get("expected_fault_window_end") or event.get("end_time")
        if not start_text or not end_text:
            continue
        start = pd.to_datetime(start_text)
        end = pd.to_datetime(end_text)
        if relative_time:
            start = (start - run_start).total_seconds() / 60
            end = (end - run_start).total_seconds() / 60
        label = "fault window" if not used_label else None
        ax.axvspan(start, end, color="tab:red", alpha=0.12, label=label)
        used_label = True


def plot_series(ax, group: pd.DataFrame, pod_name: str, relative_time: bool, run_start: pd.Timestamp) -> None:
    x_data = group["datetime"]
    if relative_time:
        x_data = (group["datetime"] - run_start).dt.total_seconds() / 60
    ax.plot(
        x_data,
        group["plot_value"],
        marker=".",
        linewidth=1,
        markersize=2,
        label=pod_name,
    )


def finalize_axis(ax, ylabel: str, relative_time: bool) -> None:
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if relative_time:
        ax.set_xlabel("Relative Time (minutes)")
    else:
        ax.set_xlabel("Time")
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))


def plot_run(
    metric_name: str,
    df: pd.DataFrame,
    metadata: dict,
    run_id: str,
    service: str,
    output_file: str,
    relative_time: bool,
    split_pods: bool,
) -> Path:
    run_start = pd.to_datetime(metadata.get("window_start_text")) if metadata.get("window_start_text") else df["datetime"].min()
    run_start_epoch = get_run_start_epoch(metadata, df)
    pod_groups = [(pod_name, group) for pod_name, group in df.groupby("pod") if len(group) >= 2]
    if not pod_groups:
        raise ValueError("可绘制的数据点不足，至少需要每个 pod 有 2 个点。")

    ylabel = df["plot_label"].dropna().iloc[0] if not df["plot_label"].dropna().empty else metric_name

    if split_pods and len(pod_groups) > 1:
        fig, axes = plt.subplots(len(pod_groups), 1, figsize=(14, max(4, len(pod_groups) * 2.8)), sharex=True)
        if len(pod_groups) == 1:
            axes = [axes]
        for ax, (pod_name, group) in zip(axes, pod_groups):
            if relative_time:
                x_data = (group["timestamp"] - run_start_epoch) / 60
                ax.plot(x_data, group["plot_value"], marker=".", linewidth=1, markersize=2, label=pod_name)
            else:
                plot_series(ax, group, pod_name, relative_time, run_start)
            add_event_windows(
                ax,
                metadata.get("events", []),
                service,
                relative_time,
                pd.to_datetime(run_start_epoch, unit="s", utc=True).tz_convert("Asia/Shanghai").tz_localize(None),
            )
            ax.set_title(pod_name, fontsize=10)
            finalize_axis(ax, ylabel, relative_time)
        fig.suptitle(f"{run_id} - {metric_name} - {service or 'all services'}", y=0.995)
        plt.tight_layout(rect=(0, 0, 1, 0.98))
    else:
        fig, ax = plt.subplots(figsize=(14, 6))
        for pod_name, group in pod_groups:
            if relative_time:
                x_data = (group["timestamp"] - run_start_epoch) / 60
                ax.plot(x_data, group["plot_value"], marker=".", linewidth=1, markersize=2, label=pod_name)
            else:
                plot_series(ax, group, pod_name, relative_time, run_start)
        add_event_windows(
            ax,
            metadata.get("events", []),
            service,
            relative_time,
            pd.to_datetime(run_start_epoch, unit="s", utc=True).tz_convert("Asia/Shanghai").tz_localize(None),
        )
        ax.set_title(f"{run_id} - {metric_name} - {service or 'all services'}")
        finalize_axis(ax, ylabel, relative_time)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()

    output_path = Path(output_file) if output_file else EXPORTS_DIR / run_id / f"{metric_name}_plot.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize exported Prometheus run data.")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID, help="实验轮次，例如 run_001")
    parser.add_argument(
        "--metric",
        default=DEFAULT_METRIC,
        choices=["cpu", "memory", "net_rx", "net_tx"],
        help="要绘制的指标，默认 cpu",
    )
    parser.add_argument("--service", default="", help="可选，按服务过滤，例如 carts")
    parser.add_argument("--pod-keyword", default="", help="可选，按 pod 名关键字过滤")
    parser.add_argument("--output-file", default="", help="可选，自定义输出图片路径")
    parser.add_argument(
        "--absolute-time",
        action="store_true",
        help="使用绝对时间轴；默认使用相对分钟时间轴，便于观察故障前后。",
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="把多个 pod 叠加在一张图上；默认多 pod 分子图显示。",
    )
    parser.add_argument(
        "--full-window",
        action="store_true",
        help="显示整轮实验时间窗；默认只聚焦故障前后几分钟。",
    )
    parser.add_argument(
        "--context-minutes",
        type=int,
        default=DEFAULT_CONTEXT_MINUTES,
        help="聚焦故障窗口时前后保留的分钟数，默认 2。",
    )
    args = parser.parse_args()

    df, csv_path = load_metric_csv(args.run_id, args.metric)
    metadata = load_run_metadata(args.run_id)
    df = prepare_dataframe(df, args.metric)
    df = filter_dataframe(df, args.service, args.pod_keyword)
    focus_window = None if args.full_window else compute_focus_window(filter_events(metadata.get("events", []), args.service), args.context_minutes)
    df = limit_time_window(df, focus_window)

    output_path = plot_run(
        args.metric,
        df,
        metadata,
        args.run_id,
        args.service,
        args.output_file,
        relative_time=not args.absolute_time,
        split_pods=not args.overlay,
    )

    print(f"数据文件: {csv_path}")
    print(f"Grafana 地址: {DEFAULT_GRAFANA_URL}")
    print(f"时间范围: {df['datetime'].min()} -> {df['datetime'].max()}")
    print(f"数据点数: {len(df)}")
    print(f"Pod 数量: {df['pod'].nunique()}")
    print(f"输出图片: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
