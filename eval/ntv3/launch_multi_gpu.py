from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from eval.ntv3.run import _collect_rows, _write_aggregate_csv

DEFAULT_ASSIGNMENTS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    ("gpu0", (("human", "functional"),)),
    ("gpu1", (("human", "annotation"), ("arabidopsis", "functional"), ("rice", "functional"))),
    ("gpu2", (("chicken", "functional"), ("cattle", "annotation"))),
    ("gpu3", (("maize", "functional"), ("tomato", "functional"), ("tomato", "annotation"))),
)


def _worker_command(
    *,
    output_root: Path,
    run_id: str,
    seed: int,
    precision: str,
    checkpoint_dir: str,
    dataset_root: str,
    num_workers: int,
    wandb_enabled: bool,
    wandb_project: str,
    wandb_entity: str | None,
    wandb_tags: tuple[str, ...],
    species_tasks: tuple[tuple[str, str], ...],
) -> str:
    commands: list[str] = []
    for species_name, task_type in species_tasks:
        command_parts = [
            shlex_quote(sys.executable),
            "-m",
            "eval.ntv3.run",
            "evaluate-species",
            "--species",
            shlex_quote(species_name),
            "--task-type",
            shlex_quote(task_type),
            "--output-root",
            shlex_quote(str(output_root)),
            "--run-id",
            shlex_quote(run_id),
            "--seed",
            shlex_quote(str(seed)),
            "--precision",
            shlex_quote(precision),
            "--checkpoint-dir",
            shlex_quote(checkpoint_dir),
            "--dataset-root",
            shlex_quote(dataset_root),
            "--num-workers",
            shlex_quote(str(num_workers)),
            "--overwrite",
        ]
        if wandb_enabled:
            command_parts.extend(
                [
                    "--wandb-enabled",
                    "--wandb-project",
                    shlex_quote(wandb_project),
                ]
            )
            if wandb_entity:
                command_parts.extend(["--wandb-entity", shlex_quote(wandb_entity)])
            if wandb_tags:
                command_parts.extend(["--wandb-tags", *(shlex_quote(tag) for tag in wandb_tags)])
        commands.append(" ".join(command_parts))
    return " && ".join(commands)


def shlex_quote(value: str) -> str:
    import shlex

    return shlex.quote(value)


def _status_payload(
    *,
    run_id: str,
    output_root: Path,
    workers: list[dict[str, Any]],
    started_at: float,
    finished: bool,
    aggregate_csv: Path | None,
) -> dict[str, Any]:
    tasks: list[dict[str, Any]] = []
    for worker in workers:
        for species_name, task_type in worker["species_tasks"]:
            output_dir = output_root / species_name / task_type
            tasks.append(
                {
                    "worker": worker["name"],
                    "species": species_name,
                    "task_type": task_type,
                    "output_dir": str(output_dir),
                    "completed": (output_dir / "dataset_scores.csv").is_file(),
                }
            )

    return {
        "run_id": run_id,
        "output_root": str(output_root),
        "started_at_unix": started_at,
        "elapsed_seconds": time.time() - started_at,
        "finished": finished,
        "aggregate_csv": str(aggregate_csv) if aggregate_csv is not None else None,
        "workers": [
            {
                "name": worker["name"],
                "gpu_index": worker["gpu_index"],
                "pid": worker["pid"],
                "log_path": str(worker["log_path"]),
                "species_tasks": worker["species_tasks"],
                "alive": worker["process"].poll() is None,
                "return_code": worker["process"].poll(),
            }
            for worker in workers
        ],
        "tasks": tasks,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the NTv3 benchmark across the 4 local GPUs.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=("auto", "bf16", "fp32"), default="fp32")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--wandb-enabled", action="store_true")
    parser.add_argument("--wandb-project", default="lumina-ntv3")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-tags", nargs="*", default=[])
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    logs_root = output_root / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    started_at = time.time()

    workers: list[dict[str, Any]] = []
    for gpu_index, (worker_name, species_tasks) in enumerate(DEFAULT_ASSIGNMENTS):
        log_path = logs_root / f"{worker_name}.log"
        command = _worker_command(
            output_root=output_root,
            run_id=args.run_id,
            seed=args.seed,
            precision=args.precision,
            checkpoint_dir=args.checkpoint_dir,
            dataset_root=args.dataset_root,
            num_workers=args.num_workers,
            wandb_enabled=args.wandb_enabled,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_tags=tuple(args.wandb_tags or ()),
            species_tasks=species_tasks,
        )
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        env["PYTHONUNBUFFERED"] = "1"
        handle = log_path.open("ab")
        process = subprocess.Popen(
            ["bash", "-lc", command],
            cwd=Path.cwd(),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
        workers.append(
            {
                "name": worker_name,
                "gpu_index": gpu_index,
                "species_tasks": species_tasks,
                "log_path": log_path,
                "process": process,
                "pid": process.pid,
                "handle": handle,
            }
        )

    status_path = output_root / "status.json"
    status_path.write_text(
        json.dumps(
            _status_payload(
                run_id=args.run_id,
                output_root=output_root,
                workers=workers,
                started_at=started_at,
                finished=False,
                aggregate_csv=None,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    aggregate_csv: Path | None = None
    try:
        while True:
            alive = [worker for worker in workers if worker["process"].poll() is None]
            status_path.write_text(
                json.dumps(
                    _status_payload(
                        run_id=args.run_id,
                        output_root=output_root,
                        workers=workers,
                        started_at=started_at,
                        finished=not alive,
                        aggregate_csv=aggregate_csv,
                    ),
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            if not alive:
                break
            time.sleep(args.poll_seconds)

        failures = [worker for worker in workers if worker["process"].returncode not in {0, None}]
        if failures:
            raise RuntimeError(
                "NTv3 multi-GPU run failed for workers: "
                + ", ".join(f"{worker['name']} rc={worker['process'].returncode}" for worker in failures)
            )

        all_rows: list[dict[str, Any]] = []
        for _worker_name, species_tasks in DEFAULT_ASSIGNMENTS:
            for species_name, task_type in species_tasks:
                dataset_scores_path = output_root / species_name / task_type / "dataset_scores.csv"
                all_rows.extend(_collect_rows(dataset_scores_path))
        aggregate_csv = _write_aggregate_csv(output_root / "ntv3_benchmark_results.csv", all_rows)
    finally:
        for worker in workers:
            worker["handle"].close()
        status_path.write_text(
            json.dumps(
                _status_payload(
                    run_id=args.run_id,
                    output_root=output_root,
                    workers=workers,
                    started_at=started_at,
                    finished=True,
                    aggregate_csv=aggregate_csv,
                ),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
