from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any


RUNNER_VERSION = "lomo_parallel_v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _example_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return str(row["split"]), int(row["question_id"]), int(row["source_index"])


def _plan_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _branch_name(row: dict[str, Any]) -> str:
    return (
        f"p{int(row['plan_index']):04d}_q{int(row['question_id']):04d}_"
        f"src{int(row['source_index']):04d}_{row['candidate_id']}"
    )


def prepare_plan(plan_path: Path, *, temperature: float) -> list[dict[str, Any]]:
    rows = _read_jsonl(plan_path)
    if not rows:
        raise ValueError("LOMO plan is empty")
    seen: set[tuple[str, str]] = set()
    prepared: list[dict[str, Any]] = []
    project_root = Path(__file__).resolve().parents[2]
    for index, source_row in enumerate(rows):
        row = dict(source_row)
        configured_parent = Path(str(row["parent_run"]))
        parent_run = configured_parent if configured_parent.is_absolute() else project_root / configured_parent
        parent_run = parent_run.resolve()
        key = (str(parent_run), str(row["candidate_id"]))
        if key in seen:
            raise ValueError(f"Duplicate LOMO target in plan: {key}")
        seen.add(key)
        if not (parent_run / "stage_actions.jsonl").is_file():
            raise ValueError(f"LOMO parent lacks migrated stage metadata: {parent_run}")
        row.update(
            {
                "runner_version": RUNNER_VERSION,
                "plan_index": index,
                "parent_run": str(parent_run),
                "temperature": temperature,
            }
        )
        prepared.append(row)
    return prepared


def _validate_child(row: dict[str, Any], batch_dir: Path) -> dict[str, Any]:
    batch_results = _read_jsonl(batch_dir / "lomo_results.jsonl")
    if len(batch_results) != 1 or batch_results[0].get("status") != "completed":
        raise ValueError(f"Expected one completed LOMO result in {batch_dir}")

    candidate_id = str(row["candidate_id"])
    child_run = batch_dir / candidate_id
    predictions = _read_jsonl(child_run / "predictions.jsonl")
    if len(predictions) != 1 or _example_key(predictions[0]) != _example_key(row):
        raise ValueError(f"LOMO child prediction does not match plan target {row['plan_index']}")

    parent_run = Path(str(row["parent_run"]))
    parent_candidates = _read_jsonl(parent_run / "edge_candidates.jsonl")
    parent_candidate = next(
        (candidate for candidate in parent_candidates if str(candidate["candidate_id"]) == candidate_id),
        None,
    )
    if parent_candidate is None:
        raise ValueError(f"Parent candidate disappeared: {parent_run}/{candidate_id}")
    target_stage_id = str(parent_candidate.get("stage_action_id") or "")
    if not target_stage_id:
        raise ValueError(f"Parent candidate lacks stage_action_id: {parent_run}/{candidate_id}")

    key = _example_key(row)
    parent_actions = [
        action
        for action in _read_jsonl(parent_run / "stage_actions.jsonl")
        if _example_key(action) == key
    ]
    child_actions = _read_jsonl(child_run / "stage_actions.jsonl")
    parent_target_index = next(
        (index for index, action in enumerate(parent_actions) if str(action["stage_action_id"]) == target_stage_id),
        None,
    )
    child_target_index = next(
        (index for index, action in enumerate(child_actions) if str(action["stage_action_id"]) == target_stage_id),
        None,
    )
    if parent_target_index is None or child_target_index != parent_target_index:
        raise ValueError(f"LOMO child does not share the identity prefix for {target_stage_id}")

    invariant_fields = (
        "stage_action_id",
        "split",
        "question_id",
        "source_index",
        "round",
        "stage",
        "state_message_ids",
        "state_hash",
        "candidate_ids",
        "action_candidate_ids",
        "edge_order",
    )
    for index in range(parent_target_index + 1):
        parent_action = parent_actions[index]
        child_action = child_actions[index]
        for field in invariant_fields:
            if child_action.get(field) != parent_action.get(field):
                raise ValueError(f"Identity prefix mismatch at {target_stage_id} field {field}")
        if index < parent_target_index and child_action.get("action_mask") != parent_action.get("action_mask"):
            raise ValueError(f"LOMO changed an action before target stage {target_stage_id}")

    target_action = child_actions[child_target_index]
    if str(target_action.get("action_mask", "")).count("1") != 1:
        raise ValueError(f"LOMO target is not one-hot: {target_action}")
    if target_action.get("dropped_candidate_ids") != [candidate_id]:
        raise ValueError(f"LOMO dropped the wrong candidate: {target_action}")
    if any(
        "1" in str(action.get("action_mask", ""))
        for index, action in enumerate(child_actions)
        if index != child_target_index
    ):
        raise ValueError(f"LOMO child contains a drop outside {target_stage_id}")

    child_candidates = {
        str(candidate["candidate_id"]): candidate
        for candidate in _read_jsonl(child_run / "edge_candidates.jsonl")
        if str(candidate.get("stage_action_id")) == target_stage_id
    }
    identity_stage_candidates = {
        str(candidate["candidate_id"]): candidate
        for candidate in parent_candidates
        if _example_key(candidate) == key and str(candidate.get("stage_action_id")) == target_stage_id
    }
    semantic_fields = (
        "candidate_id",
        "split",
        "question_id",
        "source_index",
        "round",
        "source_message_id",
        "created_by_activation_id",
        "sender",
        "recipient",
        "kind",
        "original_content",
        "state_message_ids",
        "state_hash",
        "stage",
        "stage_action_id",
        "stage_edge_index",
        "stage_state_hash",
        "terminal",
        "termination_reason",
    )
    if child_candidates.keys() != identity_stage_candidates.keys():
        raise ValueError(f"LOMO target stage candidates differ from identity: {target_stage_id}")
    for current_id, identity_candidate in identity_stage_candidates.items():
        child_candidate = child_candidates[current_id]
        for field in semantic_fields:
            if child_candidate.get(field) != identity_candidate.get(field):
                raise ValueError(f"LOMO candidate {current_id} differs from identity in field {field}")

    return {
        **batch_results[0],
        "child_run_dir": str(child_run),
        "stage_action_id": target_stage_id,
        "action_mask": target_action["action_mask"],
        "fork_stage_state_hash": target_action["state_hash"],
        "identity_prefix_verified": True,
    }


async def _run_branch(
    row: dict[str, Any],
    *,
    output_root: Path,
    project_root: Path,
    python: str,
    semaphore: asyncio.Semaphore,
    inter_launch_delay: float,
) -> dict[str, Any]:
    branch_dir = output_root / _branch_name(row)
    branch_dir.mkdir(parents=True, exist_ok=True)
    result_path = branch_dir / "result.json"
    if result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("status") == "completed":
            return existing

    attempt_index = len(list(branch_dir.glob("attempt_*")))
    attempt_dir = branch_dir / f"attempt_{attempt_index:02d}"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    child_root = attempt_dir / "child"
    stdout_path = attempt_dir / "stdout.log"
    stderr_path = attempt_dir / "stderr.log"
    command = [
        python,
        "-B",
        str(project_root / "run_lomo_batch.py"),
        "--replay-from-run",
        str(row["parent_run"]),
        "--lomo-candidate-id",
        str(row["candidate_id"]),
        "--temperature",
        str(row["temperature"]),
        "--output-dir",
        str(child_root),
        "--batch-continue-on-error",
    ]
    started_at = _utc_now()
    started_clock = time.perf_counter()
    async with semaphore:
        if inter_launch_delay > 0:
            await asyncio.sleep(inter_launch_delay * (int(row["plan_index"]) % 10))
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=project_root,
                stdout=stdout,
                stderr=stderr,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            returncode = await process.wait()

    result: dict[str, Any] = {
        **row,
        "status": "failed",
        "attempt": attempt_index,
        "returncode": returncode,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "elapsed_seconds": round(time.perf_counter() - started_clock, 3),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "command": command,
    }
    try:
        if returncode != 0:
            raise RuntimeError(f"LOMO child exited with code {returncode}")
        batch_dirs = sorted(child_root.glob("lomo_batch_*"))
        if len(batch_dirs) != 1:
            raise RuntimeError(f"Expected one LOMO batch under {child_root}, found {len(batch_dirs)}")
        result.update(_validate_child(row, batch_dirs[0]))
        result["status"] = "completed"
    except Exception as exc:  # noqa: BLE001 - preserve failures for targeted resume.
        result["error"] = repr(exc)
    _write_json_atomic(result_path, result)
    print(f"[{result['status']}] {int(row['plan_index']) + 1} {_branch_name(row)}", flush=True)
    return result


async def run_parallel(args: argparse.Namespace) -> dict[str, Any]:
    plan_path = Path(args.lomo_plan).resolve()
    plan = prepare_plan(plan_path, temperature=args.temperature)
    digest = _plan_digest(plan_path)
    output_root = Path(args.output_dir).resolve()
    manifest_path = output_root / "manifest.json"
    if output_root.exists():
        if not args.resume:
            raise FileExistsError(f"Output directory exists; use --resume: {output_root}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("plan_digest") != digest:
            raise ValueError("Existing output directory belongs to a different LOMO plan")
        if float(manifest.get("temperature")) != args.temperature:
            raise ValueError("Existing output directory used a different LOMO temperature")
    else:
        output_root.mkdir(parents=True)
        manifest = {
            "runner_version": RUNNER_VERSION,
            "lomo_plan": str(plan_path),
            "plan_digest": digest,
            "num_planned": len(plan),
            "workers": args.workers,
            "temperature": args.temperature,
            "created_at": _utc_now(),
        }
        _write_json_atomic(manifest_path, manifest)

    semaphore = asyncio.Semaphore(args.workers)
    project_root = Path(__file__).resolve().parents[2]
    tasks = [
        asyncio.create_task(
            _run_branch(
                row,
                output_root=output_root,
                project_root=project_root,
                python=args.python,
                semaphore=semaphore,
                inter_launch_delay=args.inter_launch_delay_seconds,
            )
        )
        for row in plan
    ]
    results = sorted(await asyncio.gather(*tasks), key=lambda row: int(row["plan_index"]))
    _write_jsonl(output_root / "lomo_results.jsonl", results)
    completed = sum(row.get("status") == "completed" for row in results)
    manifest.update(
        {
            "workers": args.workers,
            "temperature": args.temperature,
            "completed": completed,
            "failed": len(results) - completed,
            "resumed_at": _utc_now() if args.resume else None,
            "finished_at": _utc_now(),
        }
    )
    _write_json_atomic(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an exact LOMO plan with bounded concurrency.")
    parser.add_argument("--lomo-plan", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--inter-launch-delay-seconds", type=float, default=0.05)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    manifest = asyncio.run(run_parallel(args))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 1 if int(manifest.get("failed", 0)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
