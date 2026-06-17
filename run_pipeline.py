import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List


BASE_DIR = Path(__file__).resolve().parent
RUNNER_SCRIPT = BASE_DIR / "runner.py"
EXPORT_SCRIPT = BASE_DIR / "export_prometheus_run.py"
DEFAULT_NAMESPACE = "sock-shop"
DEFAULT_PROMETHEUS_URL = "http://127.0.0.1:1355/"


def parse_run_ids(run_ids_text: str) -> List[str]:
    return [item.strip() for item in run_ids_text.split(",") if item.strip()]


def build_runner_command(python_executable: str, run_id: str, namespace: str, execute: bool, sleep_enabled: bool) -> List[str]:
    command = [python_executable, str(RUNNER_SCRIPT), "--run-id", run_id, "--namespace", namespace]
    if execute:
        command.append("--execute")
    if sleep_enabled:
        command.append("--sleep")
    return command


def build_export_command(
    python_executable: str,
    run_id: str,
    prometheus_url: str,
    namespace: str,
    metrics: str,
    step: str,
) -> List[str]:
    return [
        python_executable,
        str(EXPORT_SCRIPT),
        "--run-id",
        run_id,
        "--prometheus-url",
        prometheus_url,
        "--namespace",
        namespace,
        "--metrics",
        metrics,
        "--step",
        step,
    ]


def run_command(command: List[str], cwd: Path) -> int:
    print(f"$ {' '.join(command)}")
    completed = subprocess.run(command, cwd=str(cwd), check=False)
    return completed.returncode


def write_summary(summary_path: Path, summary: List[dict]) -> None:
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one or multiple experiment rounds and export Prometheus data automatically."
    )
    parser.add_argument(
        "--run-ids",
        required=True,
        help="Comma separated run IDs, for example: run_006,run_007,run_008,run_009,run_010",
    )
    parser.add_argument(
        "--namespace",
        default=DEFAULT_NAMESPACE,
        help="Kubernetes namespace. Default: sock-shop.",
    )
    parser.add_argument(
        "--prometheus-url",
        default=DEFAULT_PROMETHEUS_URL,
        help="Prometheus base URL. Default: http://127.0.0.1:1355/",
    )
    parser.add_argument(
        "--metrics",
        default="cpu,memory,net_rx,net_tx",
        help="Comma separated metric presets for export.",
    )
    parser.add_argument(
        "--step",
        default="15s",
        help="Prometheus query_range step. Default: 15s.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually execute fault injection. Default only does dry-run pipeline.",
    )
    parser.add_argument(
        "--sleep",
        action="store_true",
        help="Respect real timing when calling runner.py.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining runs even if one run fails.",
    )
    parser.add_argument(
        "--inter-run-delay-seconds",
        type=int,
        default=30,
        help="Delay between runs after export completes. Default: 30 seconds.",
    )
    parser.add_argument(
        "--summary-file",
        default="",
        help="Optional summary json path. Default: isolationforest/pipeline_summary_<timestamp>.json",
    )
    args = parser.parse_args()

    run_ids = parse_run_ids(args.run_ids)
    python_executable = sys.executable
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = (
        Path(args.summary_file)
        if args.summary_file
        else BASE_DIR / f"pipeline_summary_{timestamp}.json"
    )

    summary: List[dict] = []
    overall_success = True

    print(f"Run count: {len(run_ids)}")
    print(f"Namespace: {args.namespace}")
    print(f"Execute: {args.execute}")
    print(f"Sleep: {args.sleep}")
    print(f"Continue on error: {args.continue_on_error}")
    print(f"Prometheus URL: {args.prometheus_url}")
    print(f"Summary file: {summary_path}")

    for idx, run_id in enumerate(run_ids, start=1):
        print(f"\n=== [{idx}/{len(run_ids)}] {run_id} ===")
        started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result = {
            "run_id": run_id,
            "started_at": started_at,
            "runner_exit_code": None,
            "export_exit_code": None,
            "status": "started",
        }

        runner_command = build_runner_command(
            python_executable=python_executable,
            run_id=run_id,
            namespace=args.namespace,
            execute=args.execute,
            sleep_enabled=args.sleep,
        )
        runner_exit_code = run_command(runner_command, BASE_DIR.parent)
        result["runner_exit_code"] = runner_exit_code

        if runner_exit_code != 0:
            result["status"] = "runner_failed"
            result["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            summary.append(result)
            overall_success = False
            write_summary(summary_path, summary)
            if not args.continue_on_error:
                print(f"Stopped because {run_id} runner failed.")
                return 1
            continue

        export_command = build_export_command(
            python_executable=python_executable,
            run_id=run_id,
            prometheus_url=args.prometheus_url,
            namespace=args.namespace,
            metrics=args.metrics,
            step=args.step,
        )
        export_exit_code = run_command(export_command, BASE_DIR.parent)
        result["export_exit_code"] = export_exit_code

        if export_exit_code != 0:
            result["status"] = "export_failed"
            overall_success = False
            result["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            summary.append(result)
            write_summary(summary_path, summary)
            if not args.continue_on_error:
                print(f"Stopped because {run_id} export failed.")
                return 1
        else:
            result["status"] = "success"
            result["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            summary.append(result)
            write_summary(summary_path, summary)

        if idx < len(run_ids) and args.inter_run_delay_seconds > 0:
            print(f"Waiting {args.inter_run_delay_seconds} seconds before next run...")
            time.sleep(args.inter_run_delay_seconds)

    print("\nPipeline completed.")
    print(f"Summary saved to: {summary_path}")
    return 0 if overall_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
