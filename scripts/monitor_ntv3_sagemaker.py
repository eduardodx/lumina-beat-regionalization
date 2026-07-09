#!/usr/bin/env python3
"""Poll a SageMaker NTv3 job and optional W&B run on a fixed cadence."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import boto3
import requests

from src.sagemaker_utils import load_dotenv_if_available

FINAL_TRAINING_STATUSES = {"Completed", "Failed", "Stopped"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor a SageMaker NTv3 benchmark job and track when its W&B run appears.",
    )
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--region", default="us-east-2")
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wandb-entity", default="ai4bio-lumina")
    parser.add_argument("--wandb-project", default="lumina-ntv3")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--max-checks", type=int, default=None)
    parser.add_argument("--daemonize", action="store_true")
    parser.add_argument("--pid-file", default=None)
    parser.add_argument("--stdout-log", default=None)
    return parser.parse_args()


def _load_env_value(name: str) -> str | None:
    import os

    value = os.environ.get(name)
    if value:
        return value
    return None


def _wandb_headers() -> dict[str, str] | None:
    api_key = _load_env_value("WANDB_API_KEY")
    if not api_key:
        return None
    return {"Authorization": f"Bearer {api_key}"}


def _fetch_training_job(*, client: Any, job_name: str) -> dict[str, Any]:
    response = client.describe_training_job(TrainingJobName=job_name)
    transitions = [
        {
            "status": entry.get("Status"),
            "start_time": _isoformat(entry.get("StartTime")),
            "end_time": _isoformat(entry.get("EndTime")),
            "message": entry.get("StatusMessage"),
        }
        for entry in response.get("SecondaryStatusTransitions", [])
    ]
    return {
        "job_name": response.get("TrainingJobName"),
        "status": response.get("TrainingJobStatus"),
        "secondary_status": response.get("SecondaryStatus"),
        "failure_reason": response.get("FailureReason"),
        "creation_time": _isoformat(response.get("CreationTime")),
        "training_start_time": _isoformat(response.get("TrainingStartTime")),
        "training_end_time": _isoformat(response.get("TrainingEndTime")),
        "instance_type": response.get("ResourceConfig", {}).get("InstanceType"),
        "instance_count": response.get("ResourceConfig", {}).get("InstanceCount"),
        "image_uri": response.get("AlgorithmSpecification", {}).get("TrainingImage"),
        "secondary_status_transitions": transitions,
    }


def _fetch_log_stream(*, client: Any, job_name: str) -> dict[str, Any] | None:
    response = client.describe_log_streams(
        logGroupName="/aws/sagemaker/TrainingJobs",
        logStreamNamePrefix=job_name,
        limit=1,
    )
    streams = response.get("logStreams", [])
    if not streams:
        return None
    stream = streams[0]
    return {
        "log_stream_name": stream.get("logStreamName"),
        "last_event_timestamp": _epoch_millis_to_iso(stream.get("lastEventTimestamp")),
        "last_ingestion_time": _epoch_millis_to_iso(stream.get("lastIngestionTime")),
    }


def _fetch_wandb_run(
    *,
    entity: str,
    project: str,
    display_name: str | None,
) -> dict[str, Any] | None:
    if not display_name:
        return None
    headers = _wandb_headers()
    if headers is None:
        return None

    query = """
    query ProjectRuns($entity: String!, $project: String!) {
      project(name: $project, entityName: $entity) {
        runs(first: 100) {
          edges {
            node {
              name
              displayName
              state
              group
            }
          }
        }
      }
    }
    """
    response = requests.post(
        "https://api.wandb.ai/graphql",
        headers=headers,
        json={"query": query, "variables": {"entity": entity, "project": project}},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    edges = payload.get("data", {}).get("project", {}).get("runs", {}).get("edges", [])
    for edge in edges:
        node = edge.get("node", {})
        if node.get("displayName") != display_name:
            continue
        run_id = node.get("name")
        return {
            "id": run_id,
            "display_name": node.get("displayName"),
            "state": node.get("state"),
            "group": node.get("group"),
            "url": f"https://wandb.ai/{entity}/{project}/runs/{run_id}" if run_id else None,
        }
    return None


def _isoformat(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "astimezone"):
        return value.astimezone(UTC).isoformat()
    return str(value)


def _epoch_millis_to_iso(value: Any) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(int(value) / 1000, tz=UTC).isoformat()


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _daemonize(*, pid_file: Path | None, stdout_log: Path | None) -> None:
    if os.fork() > 0:
        raise SystemExit(0)
    os.setsid()
    if os.fork() > 0:
        raise SystemExit(0)

    os.chdir("/")
    os.umask(0o022)

    target_log = stdout_log or Path("/dev/null")
    target_log.parent.mkdir(parents=True, exist_ok=True)
    with target_log.open("a+", encoding="utf-8") as handle:
        os.dup2(handle.fileno(), 1)
        os.dup2(handle.fileno(), 2)
    with Path("/dev/null").open("r", encoding="utf-8") as handle:
        os.dup2(handle.fileno(), 0)

    if pid_file is not None:
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        pid_file.write_text(f"{os.getpid()}\n", encoding="utf-8")


def main() -> int:
    load_dotenv_if_available()
    args = parse_args()
    output_dir = Path(args.output_dir)
    history_path = output_dir / "history.jsonl"
    latest_path = output_dir / "latest.json"
    pid_file = Path(args.pid_file) if args.pid_file else None
    stdout_log = Path(args.stdout_log) if args.stdout_log else None

    if args.daemonize:
        _daemonize(pid_file=pid_file, stdout_log=stdout_log)

    sm_client = boto3.client("sagemaker", region_name=args.region)
    logs_client = boto3.client("logs", region_name=args.region)

    checks = 0
    while True:
        checks += 1
        now = datetime.now(UTC).isoformat()
        training_job = _fetch_training_job(client=sm_client, job_name=args.job_name)
        log_stream = _fetch_log_stream(client=logs_client, job_name=args.job_name)
        wandb_run = _fetch_wandb_run(
            entity=args.wandb_entity,
            project=args.wandb_project,
            display_name=args.wandb_run_name,
        )
        snapshot = {
            "checked_at": now,
            "training_job": training_job,
            "log_stream": log_stream,
            "wandb_run": wandb_run,
        }
        _append_jsonl(history_path, snapshot)
        _write_json(latest_path, snapshot)
        print(
            f"[{now}] status={training_job['status']} secondary={training_job['secondary_status']} "
            f"wandb={'present' if wandb_run else 'missing'}",
            flush=True,
        )

        if training_job["status"] in FINAL_TRAINING_STATUSES:
            break
        if args.max_checks is not None and checks >= args.max_checks:
            break
        time.sleep(args.interval_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
