import argparse
import csv
import json
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List


BASE_DIR = Path(__file__).resolve().parent
PLAN_CSV = BASE_DIR / "experiment_plan.csv"
RUNTIME_LOG_CSV = BASE_DIR / "event_labels_runtime.csv"
DEFAULT_NAMESPACE = "sock-shop"


@dataclass
class EventSpec:
    run_id: str
    run_type: str
    event_id: str
    service: str
    fault_type: str
    target: str
    intensity: str
    baseline_min: int
    injection_min: int
    observe_min: int
    cooldown_min: int
    notes: str


@dataclass
class RunPlan:
    run_id: str
    run_type: str
    baseline_min: int
    injection_min: int
    observe_min: int
    event_gap_seconds: int
    gap_jitter_seconds: int
    shuffle_events: bool
    random_seed: int
    cooldown_min: int
    notes: str
    events: List[EventSpec]


def parse_pipe_list(value: str) -> List[str]:
    return [item.strip() for item in value.split("|")]


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_run_plan(plan_csv: Path, run_id: str) -> RunPlan:
    with plan_csv.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row["run_id"] != run_id:
                continue

            services = parse_pipe_list(row["service_sequence"])
            fault_types = parse_pipe_list(row["fault_sequence"])
            targets = parse_pipe_list(row["target_sequence"])
            intensities = parse_pipe_list(row["intensity_sequence"])
            event_count = int(row["event_count"])

            if not all(len(items) == event_count for items in [services, fault_types, targets, intensities]):
                raise ValueError(f"Run {run_id} event sequence length mismatch.")

            events = []
            for idx in range(event_count):
                events.append(
                    EventSpec(
                        run_id=row["run_id"],
                        run_type=row["run_type"],
                        event_id=f"event_{idx + 1:02d}",
                        service=services[idx],
                        fault_type=fault_types[idx],
                        target=targets[idx],
                        intensity=intensities[idx],
                        baseline_min=int(row["baseline_min"]),
                        injection_min=int(row["injection_min_per_event"]),
                        observe_min=int(row["observe_min_per_event"]),
                        cooldown_min=int(row["cooldown_min"]),
                        notes=row["notes"],
                    )
                )
            return RunPlan(
                run_id=row["run_id"],
                run_type=row["run_type"],
                baseline_min=int(row["baseline_min"]),
                injection_min=int(row["injection_min_per_event"]),
                observe_min=int(row["observe_min_per_event"]),
                event_gap_seconds=int(row.get("event_gap_seconds", 0)),
                gap_jitter_seconds=int(row.get("gap_jitter_seconds", 0)),
                shuffle_events=parse_bool(row.get("shuffle_events", "false")),
                random_seed=int(row.get("random_seed", 0)),
                cooldown_min=int(row["cooldown_min"]),
                notes=row["notes"],
                events=events,
            )

    raise ValueError(f"Run {run_id} not found in {plan_csv}.")


def run_subprocess(command: List[str], capture_output: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        capture_output=capture_output,
        text=True,
        check=True,
    )


def list_running_pods(namespace: str) -> List[str]:
    result = subprocess.run(
        ["kubectl", "get", "pods", "-n", namespace, "-o", "json"],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    pods = []
    for item in data.get("items", []):
        phase = item.get("status", {}).get("phase")
        name = item.get("metadata", {}).get("name", "")
        if name and phase == "Running":
            pods.append(name)
    return pods


def auto_select_target(service: str, namespace: str) -> str:
    pod_names = list_running_pods(namespace)
    matches = [name for name in pod_names if service.lower() in name.lower()]
    if not matches:
        raise ValueError(f"No running pod found for service {service!r} in namespace {namespace!r}.")
    return sorted(matches)[0]


def build_fault_command(event: EventSpec, resolved_target: str, namespace: str) -> List[str]:
    if event.fault_type == "no_fault":
        return []

    if event.fault_type == "kill":
        return ["kubectl", "delete", "pod", resolved_target, "-n", namespace]

    if event.fault_type == "pause":
        return [
            "kubectl",
            "exec",
            "-n",
            namespace,
            resolved_target,
            "--",
            "sh",
            "-c",
            "kill -STOP 1",
        ]

    if event.fault_type == "network_delay":
        delay_ms = event.intensity.replace("ms", "").strip()
        return [
            "kubectl",
            "exec",
            "-n",
            namespace,
            resolved_target,
            "--",
            "sh",
            "-c",
            f"tc qdisc replace dev eth0 root netem delay {delay_ms}ms",
        ]

    raise ValueError(f"Unsupported fault type: {event.fault_type}")


def build_recovery_command(event: EventSpec, resolved_target: str, namespace: str) -> List[str]:
    if event.fault_type == "no_fault":
        return []

    if event.fault_type == "pause":
        return [
            "kubectl",
            "exec",
            "-n",
            namespace,
            resolved_target,
            "--",
            "sh",
            "-c",
            "kill -CONT 1",
        ]
    if event.fault_type == "network_delay":
        return [
            "kubectl",
            "exec",
            "-n",
            namespace,
            resolved_target,
            "--",
            "sh",
            "-c",
            "tc qdisc del dev eth0 root",
        ]
    return []


def append_runtime_log(row: dict) -> None:
    file_exists = RUNTIME_LOG_CSV.exists()
    with RUNTIME_LOG_CSV.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "run_id",
                "event_id",
                "run_type",
                "service",
                "target",
                "fault_type",
                "intensity",
                "start_time",
                "end_time",
                "baseline_start",
                "baseline_end",
                "observe_end",
                "expected_fault_window_start",
                "expected_fault_window_end",
                "status",
                "notes",
            ],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def print_command(prefix: str, command: List[str]) -> None:
    print(f"{prefix}: {' '.join(command)}")


def maybe_sleep(seconds: int, enabled: bool) -> None:
    if enabled and seconds > 0:
        time.sleep(seconds)


def compute_gap_seconds(plan: RunPlan, rng: random.Random) -> int:
    if plan.event_gap_seconds <= 0:
        return 0
    if plan.gap_jitter_seconds <= 0:
        return plan.event_gap_seconds
    lower = max(0, plan.event_gap_seconds - plan.gap_jitter_seconds)
    upper = plan.event_gap_seconds + plan.gap_jitter_seconds
    return rng.randint(lower, upper)


def dependency_hint(event: EventSpec) -> str:
    if event.fault_type == "no_fault":
        return "normal_only 轮次不会注入故障，只记录正常运行窗口。"
    if event.fault_type == "pause":
        return "pause 依赖容器内存在 `sh`，且允许向 PID 1 发送 STOP/CONT 信号。"
    if event.fault_type == "network_delay":
        return "network_delay 依赖容器内存在 `tc`，并具有 NET_ADMIN 权限。"
    return ""


def resolve_target(event: EventSpec, execute: bool, namespace: str) -> str:
    resolved_target = event.target
    if resolved_target == "auto_select":
        try:
            resolved_target = auto_select_target(event.service, namespace)
        except Exception:
            if execute:
                raise
            resolved_target = f"<auto_select:{event.service}>"
    return resolved_target


def execute_event(
    event: EventSpec,
    execute: bool,
    sleep_enabled: bool,
    namespace: str,
    baseline_start: datetime,
    baseline_end: datetime,
    fault_start: datetime,
) -> datetime:
    fault_end = fault_start + timedelta(minutes=event.injection_min)
    observe_end = fault_end + timedelta(minutes=event.observe_min)

    resolved_target = resolve_target(event, execute, namespace)
    fault_command = build_fault_command(event, resolved_target, namespace)
    recovery_command = build_recovery_command(event, resolved_target, namespace)

    print(f"\n[{event.run_id} / {event.event_id}] {event.service} -> {event.fault_type}")
    print(f"Namespace: {namespace}")
    print(f"Target: {resolved_target}")
    print(f"Baseline: {baseline_start} -> {baseline_end}")
    print(f"Fault window: {fault_start} -> {fault_end}")
    print(f"Observe until: {observe_end}")
    if fault_command:
        print_command("Fault command", fault_command)
    else:
        print("Fault command: <none>")
    if recovery_command:
        print_command("Recovery command", recovery_command)
    if dependency_hint(event):
        print(f"Dependency hint: {dependency_hint(event)}")

    status = "planned"
    actual_fault_start = ""
    actual_fault_end = ""

    if execute:
        actual_fault_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if fault_command:
            run_subprocess(fault_command, capture_output=False)
        status = "executed"

        if event.fault_type == "kill" and event.injection_min > 0:
            # For kill faults, keep a degraded window before moving into observation.
            maybe_sleep(event.injection_min * 60, sleep_enabled)
        elif event.fault_type in {"pause", "network_delay"}:
            # These faults need an explicit recovery step after the injection duration.
            time.sleep(event.injection_min * 60)
            run_subprocess(recovery_command, capture_output=False)
        actual_fault_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        maybe_sleep(event.observe_min * 60, sleep_enabled)
    else:
        status = "dry_run"

    append_runtime_log(
        {
            "run_id": event.run_id,
            "event_id": event.event_id,
            "run_type": event.run_type,
            "service": event.service,
            "target": resolved_target,
            "fault_type": event.fault_type,
            "intensity": event.intensity,
            "start_time": actual_fault_start,
            "end_time": actual_fault_end,
            "baseline_start": baseline_start.strftime("%Y-%m-%d %H:%M:%S"),
            "baseline_end": baseline_end.strftime("%Y-%m-%d %H:%M:%S"),
            "observe_end": observe_end.strftime("%Y-%m-%d %H:%M:%S"),
            "expected_fault_window_start": fault_start.strftime("%Y-%m-%d %H:%M:%S"),
            "expected_fault_window_end": fault_end.strftime("%Y-%m-%d %H:%M:%S"),
            "status": status,
            "notes": event.notes,
        }
    )
    return observe_end


def execute_run(plan: RunPlan, execute: bool, sleep_enabled: bool, namespace: str) -> None:
    now = datetime.now()
    baseline_start = now
    baseline_end = baseline_start + timedelta(minutes=plan.baseline_min)
    rng = random.Random(plan.random_seed)
    events = list(plan.events)
    if plan.shuffle_events:
        rng.shuffle(events)

    estimated_seconds = 0
    for idx, event in enumerate(events):
        estimated_seconds += (event.injection_min + event.observe_min) * 60
        if idx < len(events) - 1:
            estimated_seconds += compute_gap_seconds(plan, rng)
    cooldown_end = baseline_end + timedelta(seconds=estimated_seconds + plan.cooldown_min * 60)
    rng = random.Random(plan.random_seed)
    if plan.shuffle_events:
        rng.shuffle(events)

    print(f"Run note: {plan.notes}")
    print(f"Shared baseline: {baseline_start} -> {baseline_end}")
    print(f"Shuffle events: {plan.shuffle_events}")
    print(f"Random seed: {plan.random_seed}")
    print(f"Event gap seconds: {plan.event_gap_seconds} +/- {plan.gap_jitter_seconds}")
    print(f"Cooldown minutes: {plan.cooldown_min}")
    print(f"Estimated run end: {cooldown_end}")

    if execute:
        maybe_sleep(plan.baseline_min * 60, sleep_enabled)

    current_fault_start = baseline_end
    for idx, event in enumerate(events):
        current_fault_start = execute_event(
            event=event,
            execute=execute,
            sleep_enabled=sleep_enabled,
            namespace=namespace,
            baseline_start=baseline_start,
            baseline_end=baseline_end,
            fault_start=current_fault_start,
        )
        if idx < len(events) - 1:
            gap_seconds = compute_gap_seconds(plan, rng)
            if gap_seconds > 0:
                print(f"Inter-event gap: {gap_seconds} seconds")
                if execute:
                    maybe_sleep(gap_seconds, sleep_enabled)
                current_fault_start = current_fault_start + timedelta(seconds=gap_seconds)

    if execute:
        maybe_sleep(plan.cooldown_min * 60, sleep_enabled)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch fault injection runner for Sock Shop experiment plans."
    )
    parser.add_argument("--run-id", required=True, help="Run ID defined in experiment_plan.csv")
    parser.add_argument(
        "--namespace",
        default=DEFAULT_NAMESPACE,
        help="Kubernetes namespace that contains Sock Shop pods. Default: sock-shop.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually execute fault injection commands. Default is dry-run.",
    )
    parser.add_argument(
        "--sleep",
        action="store_true",
        help="Respect baseline/observe timing in real time. Useful together with --execute.",
    )
    args = parser.parse_args()

    try:
        plan = load_run_plan(PLAN_CSV, args.run_id)
        print(f"Loaded {len(plan.events)} events for {args.run_id}")
        print(f"Namespace: {args.namespace}")
        print(f"Mode: {'execute' if args.execute else 'dry-run'}")
        print(f"Timing: {'real-time' if args.sleep else 'no-sleep'}")
        execute_run(plan=plan, execute=args.execute, sleep_enabled=args.sleep, namespace=args.namespace)

        print(f"\nRuntime log updated: {RUNTIME_LOG_CSV}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
