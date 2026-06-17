import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import requests


BASE_DIR = Path(__file__).resolve().parent
RUNTIME_LOG_CSV = BASE_DIR / "event_labels_runtime.csv"
DEFAULT_NAMESPACE = "sock-shop"
DEFAULT_STEP = "15s"
DEFAULT_PROMETHEUS_URL = "http://127.0.0.1:1355/"

METRIC_QUERIES = {
    "cpu": 'container_cpu_usage_seconds_total{{namespace="{namespace}"}}',
    "memory": 'container_memory_usage_bytes{{namespace="{namespace}"}}',
    "net_rx": 'container_network_receive_bytes_total{{namespace="{namespace}"}}',
    "net_tx": 'container_network_transmit_bytes_total{{namespace="{namespace}"}}',
}


def parse_time(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%d %H:%M:%S")


def load_run_rows(runtime_log_csv: Path, run_id: str, status_filter: str) -> List[dict]:
    with runtime_log_csv.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = []
        for raw_row in reader:
            row = {(key or "").strip(): value for key, value in raw_row.items()}
            if row.get("run_id") == run_id:
                rows.append(row)

    if status_filter != "all":
        rows = [row for row in rows if row.get("status") == status_filter]

    if not rows:
        raise ValueError(f"No rows found for run_id={run_id!r} with status={status_filter!r}.")
    return rows


def compute_run_window(rows: List[dict]) -> tuple[int, int]:
    start = min(parse_time(row["baseline_start"]) for row in rows)
    end = max(parse_time(row["observe_end"]) for row in rows)
    return int(start.timestamp()), int(end.timestamp())


def fetch_query_range(prometheus_url: str, query: str, start: int, end: int, step: str) -> List[dict]:
    api_url = f"{prometheus_url.rstrip('/')}/api/v1/query_range"
    response = requests.get(
        api_url,
        params={"query": query, "start": start, "end": end, "step": step},
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()

    if data.get("status") != "success":
        raise ValueError(f"Prometheus API error: {data.get('error', 'unknown error')}")

    return data.get("data", {}).get("result", [])


def write_metric_csv(output_csv: Path, results: List[dict]) -> int:
    row_count = 0
    with output_csv.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["timestamp", "value", "metric", "metric_dict"])

        for series in results:
            metric_dict = series.get("metric", {})
            metric_text = str(metric_dict)
            metric_json = json.dumps(metric_dict, ensure_ascii=False, sort_keys=True)
            for point in series.get("values", []):
                writer.writerow([point[0], point[1], metric_text, metric_json])
                row_count += 1
    return row_count


def write_run_metadata(output_dir: Path, run_id: str, rows: List[dict], start: int, end: int, step: str) -> None:
    metadata = {
        "run_id": run_id,
        "window_start_epoch": start,
        "window_end_epoch": end,
        "window_start_text": datetime.fromtimestamp(start).strftime("%Y-%m-%d %H:%M:%S"),
        "window_end_text": datetime.fromtimestamp(end).strftime("%Y-%m-%d %H:%M:%S"),
        "step": step,
        "events": rows,
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)


def parse_metric_names(metric_text: str) -> List[str]:
    return [item.strip() for item in metric_text.split(",") if item.strip()]


def resolve_queries(metric_names: List[str], namespace: str) -> Dict[str, str]:
    queries = {}
    for metric_name in metric_names:
        if metric_name not in METRIC_QUERIES:
            raise ValueError(
                f"Unsupported metric {metric_name!r}. Supported: {', '.join(sorted(METRIC_QUERIES))}"
            )
        queries[metric_name] = METRIC_QUERIES[metric_name].format(namespace=namespace)
    return queries


def summarize_metric_series(results: List[dict]) -> Dict[str, int]:
    pod_names = set()
    point_count = 0
    for series in results:
        metric = series.get("metric", {})
        pod = metric.get("pod")
        if pod:
            pod_names.add(pod)
        point_count += len(series.get("values", []))
    return {"series_count": len(results), "pod_count": len(pod_names), "point_count": point_count}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export Prometheus query_range data for a recorded experiment run window."
    )
    parser.add_argument("--run-id", required=True, help="Run ID recorded in event_labels_runtime.csv")
    parser.add_argument(
        "--prometheus-url",
        default=DEFAULT_PROMETHEUS_URL,
        help="Prometheus base URL. Default: http://127.0.0.1:1355/",
    )
    parser.add_argument(
        "--namespace",
        default=DEFAULT_NAMESPACE,
        help="Kubernetes namespace used to fill built-in PromQL templates. Default: sock-shop.",
    )
    parser.add_argument(
        "--metrics",
        default="cpu,memory,net_rx,net_tx",
        help="Comma separated metric presets to export. Default: cpu,memory,net_rx,net_tx",
    )
    parser.add_argument("--step", default=DEFAULT_STEP, help="Prometheus query_range step. Default: 15s")
    parser.add_argument(
        "--status-filter",
        default="executed",
        choices=["executed", "dry_run", "all"],
        help="Filter run rows by runtime status. Default: executed.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output directory. Default: isolationforest/exports/<run_id>",
    )
    args = parser.parse_args()

    rows = load_run_rows(RUNTIME_LOG_CSV, args.run_id, args.status_filter)
    start, end = compute_run_window(rows)
    queries = resolve_queries(parse_metric_names(args.metrics), args.namespace)

    output_dir = Path(args.output_dir) if args.output_dir else BASE_DIR / "exports" / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    write_run_metadata(output_dir, args.run_id, rows, start, end, args.step)

    print(f"Run ID: {args.run_id}")
    print(f"Window: {datetime.fromtimestamp(start)} -> {datetime.fromtimestamp(end)}")
    print(f"Output dir: {output_dir}")

    export_summary = []
    for metric_name, query in queries.items():
        print(f"\nExporting metric: {metric_name}")
        print(f"Query: {query}")
        results = fetch_query_range(args.prometheus_url, query, start, end, args.step)
        output_csv = output_dir / f"{metric_name}.csv"
        row_count = write_metric_csv(output_csv, results)
        summary = summarize_metric_series(results)
        summary["metric_name"] = metric_name
        summary["row_count"] = row_count
        export_summary.append(summary)
        print(
            f"Saved {metric_name}.csv, series={summary['series_count']}, "
            f"pods={summary['pod_count']}, points={summary['point_count']}"
        )

    with (output_dir / "export_summary.json").open("w", encoding="utf-8") as file:
        json.dump(export_summary, file, ensure_ascii=False, indent=2)

    print("\nExport completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
