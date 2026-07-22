from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import yaml


@dataclass(frozen=True)
class Shard:
    index: int
    start_index: int
    sample_size: int
    output_dir: Path
    stdout_path: Path
    stderr_path: Path
    command: list[str]


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_summary(shard_output_dir: Path) -> tuple[Path | None, dict[str, Any] | None]:
    run_dirs = sorted(shard_output_dir.glob("phase0_PRM800K_*"), key=lambda path: path.stat().st_mtime, reverse=True)
    for run_dir in run_dirs:
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            return summary_path, _read_json(summary_path)
    return None, None


def _reject_conflicting_forwarded_args(forward_args: list[str]) -> None:
    blocked = {
        "--sample-size",
        "--start-index",
        "--split",
        "--output-dir",
        "--config",
        "--continue-on-error",
    }
    conflicts = [arg for arg in forward_args if arg in blocked or any(arg.startswith(item + "=") for item in blocked)]
    if conflicts:
        joined = ", ".join(conflicts)
        raise ValueError(f"Do not pass shard-controlled args via forwarded args: {joined}")


def _build_shards(
    *,
    args: argparse.Namespace,
    forward_args: list[str],
    project_root: Path,
    master_dir: Path,
    total_examples: int,
) -> list[Shard]:
    shards: list[Shard] = []
    remaining = total_examples
    current_start = args.start_index
    shard_index = 0
    while remaining > 0:
        sample_size = min(args.chunk_size, remaining)
        shard_dir = master_dir / f"shard_{shard_index:04d}_start_{current_start}_n_{sample_size}"
        stdout_path = shard_dir / "stdout.log"
        stderr_path = shard_dir / "stderr.log"
        command = [
            args.python,
            "-B",
            str((project_root / args.runner).resolve()),
            "--config",
            args.config,
            "--split",
            args.split,
            "--start-index",
            str(current_start),
            "--sample-size",
            str(sample_size),
            "--output-dir",
            str(shard_dir),
        ]
        if args.continue_on_error:
            command.append("--continue-on-error")
        command.extend(forward_args)
        shards.append(
            Shard(
                index=shard_index,
                start_index=current_start,
                sample_size=sample_size,
                output_dir=shard_dir,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                command=command,
            )
        )
        current_start += sample_size
        remaining -= sample_size
        shard_index += 1
    return shards


async def _run_shard(shard: Shard, *, cwd: Path, timeout_seconds: float | None) -> dict[str, Any]:
    shard.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    with shard.stdout_path.open("w", encoding="utf-8") as stdout, shard.stderr_path.open("w", encoding="utf-8") as stderr:
        process = await asyncio.create_subprocess_exec(
            *shard.command,
            cwd=str(cwd),
            stdout=stdout,
            stderr=stderr,
            env=os.environ.copy(),
        )
        timed_out = False
        try:
            returncode = await asyncio.wait_for(process.wait(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            timed_out = True
            process.kill()
            returncode = await process.wait()

    elapsed_seconds = time.perf_counter() - start
    summary_path, summary = _latest_summary(shard.output_dir)
    return {
        "shard_index": shard.index,
        "start_index": shard.start_index,
        "sample_size": shard.sample_size,
        "returncode": returncode,
        "timed_out": timed_out,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(shard.output_dir),
        "summary_path": str(summary_path) if summary_path else None,
        "summary": summary,
        "stdout_path": str(shard.stdout_path),
        "stderr_path": str(shard.stderr_path),
        "command": shard.command,
    }


async def _run_all(shards: list[Shard], *, args: argparse.Namespace, project_root: Path) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(args.workers)
    results: list[dict[str, Any]] = []

    async def run_one(shard: Shard) -> dict[str, Any]:
        async with semaphore:
            if args.inter_launch_delay_seconds > 0:
                await asyncio.sleep(args.inter_launch_delay_seconds * shard.index)
            print(f"[launch] shard={shard.index} start={shard.start_index} n={shard.sample_size}", flush=True)
            result = await _run_shard(shard, cwd=project_root, timeout_seconds=args.shard_timeout_seconds)
            status = "ok" if result["returncode"] == 0 and not result["timed_out"] else "failed"
            print(
                f"[{status}] shard={shard.index} rc={result['returncode']} elapsed={result['elapsed_seconds']}s "
                f"summary={result['summary_path']}",
                flush=True,
            )
            return result

    tasks = [asyncio.create_task(run_one(shard)) for shard in shards]
    for task in asyncio.as_completed(tasks):
        results.append(await task)
    return sorted(results, key=lambda item: item["shard_index"])


def _combine_results(master_dir: Path, shards: list[Shard], results: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    for result in results:
        summary = result.get("summary")
        if isinstance(summary, dict):
            summaries.append(summary)

    total_num_examples = sum(int(summary.get("num_examples", 0)) for summary in summaries)
    total_planned_examples = sum(int(summary.get("planned_examples", 0)) for summary in summaries)
    total_errors = sum(len(summary.get("errors", [])) for summary in summaries)
    total_latency_ms = sum(int(summary.get("latency_ms", 0)) for summary in summaries)
    failed_shards = [result for result in results if result.get("returncode") != 0 or result.get("timed_out")]
    return {
        "master_dir": str(master_dir),
        "split": args.split,
        "total_examples_requested": args.total_examples,
        "start_index": args.start_index,
        "chunk_size": args.chunk_size,
        "workers": args.workers,
        "num_shards": len(shards),
        "num_finished_shards": len(results),
        "num_failed_shards": len(failed_shards),
        "total_planned_examples": total_planned_examples,
        "total_num_examples": total_num_examples,
        "total_errors": total_errors,
        "total_latency_ms": total_latency_ms,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "failed_shard_indices": [result["shard_index"] for result in failed_shards],
        "shards": results,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Phase0_PRM800K in parallel shards. Unknown args are forwarded to run_phase0_prm800k.py."
    )
    parser.add_argument("--config", default="config/phase0_PRM800K.yaml")
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--total-examples", type=int, help="Default: train_size or test_size from config.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", help="Master output directory. Default: runs/phase0_PRM800K_parallel_<timestamp>.")
    parser.add_argument("--runner", default="run_phase0_prm800k.py")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--shard-timeout-seconds", type=float)
    parser.add_argument("--inter-launch-delay-seconds", type=float, default=0.0)
    return parser


def main() -> int:
    parser = build_parser()
    args, forward_args = parser.parse_known_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be >= 1")
    if args.start_index < 0:
        raise ValueError("--start-index must be >= 0")
    _reject_conflicting_forwarded_args(forward_args)

    project_root = Path(__file__).resolve().parent
    config = _load_config(project_root / args.config)
    total_examples = args.total_examples
    if total_examples is None:
        total_examples = int(config.get("train_size" if args.split == "train" else "test_size", 0))
        args.total_examples = total_examples
    if total_examples < 1:
        raise ValueError("No examples requested. Set --total-examples or train_size/test_size in config.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    master_dir = Path(args.output_dir) if args.output_dir else project_root / "runs" / f"phase0_PRM800K_parallel_{timestamp}"
    if not master_dir.is_absolute():
        master_dir = project_root / master_dir
    master_dir.mkdir(parents=True, exist_ok=False)

    shards = _build_shards(
        args=args,
        forward_args=forward_args,
        project_root=project_root,
        master_dir=master_dir,
        total_examples=total_examples,
    )
    manifest = {
        "master_dir": str(master_dir),
        "project_root": str(project_root),
        "config": args.config,
        "split": args.split,
        "total_examples": total_examples,
        "start_index": args.start_index,
        "chunk_size": args.chunk_size,
        "workers": args.workers,
        "forward_args": forward_args,
        "dry_run": args.dry_run,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "shards": [asdict(shard) for shard in shards],
    }
    _write_json(master_dir / "parallel_manifest.json", manifest)

    print(f"master_dir={master_dir}", flush=True)
    print(f"shards={len(shards)} workers={args.workers} total_examples={total_examples}", flush=True)
    if args.dry_run:
        for shard in shards:
            print(" ".join(shard.command))
        _write_json(master_dir / "parallel_summary.json", {**manifest, "dry_run_only": True})
        return 0

    results = asyncio.run(_run_all(shards, args=args, project_root=project_root))
    combined = _combine_results(master_dir, shards, results, args)
    _write_json(master_dir / "parallel_summary.json", combined)
    print(f"parallel_summary={master_dir / 'parallel_summary.json'}", flush=True)
    return 1 if combined["num_failed_shards"] else 0


if __name__ == "__main__":
    raise SystemExit(main())