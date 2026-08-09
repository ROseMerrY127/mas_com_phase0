from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any


PLAN_VERSION = "random_stage_replay_v1"


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
    temp = path.with_suffix(path.suffix + ".tmp")
    try:
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _example_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return str(row["split"]), int(row["question_id"]), int(row["source_index"])


def _run_priority(run_dir: Path) -> tuple[int, float]:
    # Fresh MATH500 reruns replace incomplete legacy identity prefixes for the same question.
    return (int(run_dir.name.startswith("phase0_MATH500_")), run_dir.stat().st_mtime)


def _select_identity_parents(master_dir: Path) -> dict[tuple[str, int, int], tuple[Path, dict[str, Any]]]:
    selected: dict[tuple[str, int, int], tuple[Path, dict[str, Any]]] = {}
    prediction_files = sorted(master_dir.glob("shard_*/phase0_*/predictions.jsonl"))
    if not prediction_files:
        raise FileNotFoundError(f"No identity predictions found under {master_dir}")
    for prediction_path in prediction_files:
        run_dir = prediction_path.parent
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        if str(summary.get("edge_policy")) != "identity":
            continue
        for prediction in _read_jsonl(prediction_path):
            key = _example_key(prediction)
            previous = selected.get(key)
            if previous is None or _run_priority(run_dir) > _run_priority(previous[0]):
                selected[key] = (run_dir, prediction)
    return selected


def _stable_stage_index(*, seed: int, key: tuple[str, int, int], stage_ids: list[str]) -> int:
    payload = {"seed": seed, "example": key, "stage_action_ids": stage_ids}
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % len(stage_ids)


def build_random_stage_plan(
    master_dir: Path,
    plan_dir: Path,
    *,
    seed: int,
    min_dropped_edges: int,
    expected_examples: int,
) -> Path:
    master_dir = master_dir.resolve()
    plan_dir = plan_dir.resolve()
    if plan_dir.exists():
        raise FileExistsError(f"Plan directory already exists: {plan_dir}")
    selected = _select_identity_parents(master_dir)
    train_rows = sorted(
        ((key, value) for key, value in selected.items() if key[0] == "train"),
        key=lambda item: item[0][1],
    )
    if len(train_rows) != expected_examples:
        raise ValueError(f"Expected {expected_examples} complete train identity examples, found {len(train_rows)}")
    question_ids = [key[1] for key, _value in train_rows]
    if question_ids != list(range(expected_examples)):
        raise ValueError("Complete train identity examples do not cover contiguous question_id 0..399")

    cache: dict[Path, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    plan: list[dict[str, Any]] = []
    for plan_index, (key, (parent_run, prediction)) in enumerate(train_rows):
        if parent_run not in cache:
            cache[parent_run] = (
                _read_jsonl(parent_run / "edge_candidates.jsonl"),
                _read_jsonl(parent_run / "stage_actions.jsonl"),
            )
        candidates, stage_actions = cache[parent_run]
        question_candidates = {
            str(row["candidate_id"]): row for row in candidates if _example_key(row) == key
        }
        eligible = [
            row
            for row in stage_actions
            if _example_key(row) == key
            and str(row.get("stage")) in {"planner", "solver", "judger"}
            and len(row.get("action_candidate_ids", [])) >= min_dropped_edges
            and set(str(row.get("action_mask", ""))) <= {"0"}
        ]
        if not eligible:
            raise ValueError(f"No eligible identity stage for {key} in {parent_run}")
        eligible.sort(key=lambda row: (int(row["round"]), str(row["stage_action_id"])))
        stage_ids = [str(row["stage_action_id"]) for row in eligible]
        selected_stage = eligible[_stable_stage_index(seed=seed, key=key, stage_ids=stage_ids)]
        checkpoint_id = str(selected_stage["action_candidate_ids"][0])
        checkpoint = question_candidates.get(checkpoint_id)
        if checkpoint is None:
            raise ValueError(f"Stage checkpoint {checkpoint_id} is missing in {parent_run}")
        plan.append(
            {
                "plan_version": PLAN_VERSION,
                "plan_index": plan_index,
                "split": key[0],
                "question_id": key[1],
                "source_index": key[2],
                "parent_run": str(parent_run.resolve()),
                "checkpoint_candidate_id": checkpoint_id,
                "stage_action_id": selected_stage["stage_action_id"],
                "stage": selected_stage["stage"],
                "round": selected_stage["round"],
                "edge_order": selected_stage["edge_order"],
                "identity_action_mask": selected_stage["action_mask"],
                "random_policy_seed": seed,
                "random_drop_min_edges": min_dropped_edges,
                "baseline_mas_output": prediction.get("mas_output"),
                "baseline_mas_approx_total_tokens": prediction.get("mas_approx_total_tokens"),
                "baseline_rounds_used": prediction.get("rounds_used"),
            }
        )

    plan_dir.mkdir(parents=True, exist_ok=False)
    plan_path = plan_dir / "random_stage_plan.jsonl"
    _write_jsonl(plan_path, plan)
    stage_counts = Counter(str(row["stage"]) for row in plan)
    round_counts = Counter(int(row["round"]) for row in plan)
    _write_json_atomic(
        plan_dir / "plan_summary.json",
        {
            "plan_version": PLAN_VERSION,
            "master_dir": str(master_dir),
            "plan_path": str(plan_path),
            "num_examples": len(plan),
            "seed": seed,
            "min_dropped_edges": min_dropped_edges,
            "stage_counts": dict(sorted(stage_counts.items())),
            "round_counts": {str(key): value for key, value in sorted(round_counts.items())},
            "created_at": _utc_now(),
        },
    )
    return plan_path


def _plan_digest(plan_path: Path) -> str:
    return hashlib.sha256(plan_path.read_bytes()).hexdigest()


def _branch_name(row: dict[str, Any]) -> str:
    return f"q{int(row['question_id']):04d}_src{int(row['source_index']):04d}_{row['stage']}_r{row['round']}"


def _read_existing_result(branch_dir: Path) -> dict[str, Any] | None:
    path = branch_dir / "result.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _validate_child(row: dict[str, Any], child_run: Path) -> dict[str, Any]:
    predictions = _read_jsonl(child_run / "predictions.jsonl")
    if len(predictions) != 1:
        raise ValueError(f"Expected one child prediction in {child_run}, found {len(predictions)}")
    prediction = predictions[0]
    if _example_key(prediction) != _example_key(row):
        raise ValueError(f"Child prediction key does not match plan row {row['plan_index']}")
    stage_actions = _read_jsonl(child_run / "stage_actions.jsonl")
    target_id = str(row["stage_action_id"])
    target = next((action for action in stage_actions if str(action["stage_action_id"]) == target_id), None)
    if target is None:
        raise ValueError(f"Child is missing target stage {target_id}")

    parent_run = Path(str(row["parent_run"]))
    key = _example_key(row)
    parent_actions = [
        action
        for action in _read_jsonl(parent_run / "stage_actions.jsonl")
        if _example_key(action) == key
    ]
    parent_target_index = next(
        (index for index, action in enumerate(parent_actions) if str(action["stage_action_id"]) == target_id),
        None,
    )
    child_target_index = next(
        (index for index, action in enumerate(stage_actions) if str(action["stage_action_id"]) == target_id),
        None,
    )
    if parent_target_index is None or child_target_index != parent_target_index:
        raise ValueError(f"Child does not reach target stage through the same identity prefix: {target_id}")

    # Before the action fork, stage topology and semantic state must exactly match identity.
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
        child_action = stage_actions[index]
        for field in invariant_fields:
            if child_action.get(field) != parent_action.get(field):
                raise ValueError(
                    f"Identity prefix mismatch at {parent_action['stage_action_id']} field {field}"
                )
        if index < parent_target_index and child_action.get("action_mask") != parent_action.get("action_mask"):
            raise ValueError(f"Identity prefix action mismatch at {parent_action['stage_action_id']}")

    parent_candidates = {
        str(candidate["candidate_id"]): candidate
        for candidate in _read_jsonl(parent_run / "edge_candidates.jsonl")
        if _example_key(candidate) == key and str(candidate.get("stage_action_id")) == target_id
    }
    child_candidates = {
        str(candidate["candidate_id"]): candidate
        for candidate in _read_jsonl(child_run / "edge_candidates.jsonl")
        if str(candidate.get("stage_action_id")) == target_id
    }
    candidate_fields = (
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
    if child_candidates.keys() != parent_candidates.keys():
        raise ValueError(f"Target stage candidate IDs differ from identity: {target_id}")
    for candidate_id, parent_candidate in parent_candidates.items():
        child_candidate = child_candidates[candidate_id]
        for field in candidate_fields:
            if child_candidate.get(field) != parent_candidate.get(field):
                raise ValueError(f"Target candidate {candidate_id} differs from identity in field {field}")

    min_edges = int(row["random_drop_min_edges"])
    if str(target.get("action_mask", "")).count("1") < min_edges:
        raise ValueError(f"Target stage did not drop at least {min_edges} edges: {target}")
    if any("1" in str(action.get("action_mask", "")) for action in stage_actions if action is not target):
        raise ValueError(f"Child contains a drop outside target stage {target_id}")
    summary = json.loads((child_run / "summary.json").read_text(encoding="utf-8"))
    pruning = summary.get("pruning", {})
    if pruning.get("random_drop_scope") != "checkpoint_stage_once_then_identity":
        raise ValueError(f"Child has unexpected random_drop scope: {pruning}")
    return {
        "random_mas_output": prediction.get("mas_output"),
        "random_mas_approx_total_tokens": prediction.get("mas_approx_total_tokens"),
        "random_rounds_used": prediction.get("rounds_used"),
        "action_mask": target["action_mask"],
        "dropped_candidate_ids": target["dropped_candidate_ids"],
        "identity_prefix_verified": True,
        "fork_stage_state_hash": target["state_hash"],
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
    existing = _read_existing_result(branch_dir)
    if existing is not None and existing.get("status") == "completed":
        return existing
    attempt_index = len(list(branch_dir.glob("attempt_*")))
    attempt_dir = branch_dir / f"attempt_{attempt_index:02d}"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    stdout_path = attempt_dir / "stdout.log"
    stderr_path = attempt_dir / "stderr.log"
    child_root = attempt_dir / "child"
    command = [
        python,
        "-B",
        str(project_root / "run_phase0_pruning.py"),
        "--replay-from-run",
        str(row["parent_run"]),
        "--checkpoint-candidate-id",
        str(row["checkpoint_candidate_id"]),
        "--replay-policy-name",
        "random_drop",
        "--random-policy-seed",
        str(row["random_policy_seed"]),
        "--random-drop-min-edges",
        str(row["random_drop_min_edges"]),
        "--temperature",
        str(row.get("temperature", 0.0)),
        "--output-dir",
        str(child_root),
    ]
    started = _utc_now()
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
        "started_at": started,
        "finished_at": _utc_now(),
        "elapsed_seconds": round(time.perf_counter() - started_clock, 3),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "command": command,
    }
    try:
        if returncode != 0:
            raise RuntimeError(f"random_drop child exited with code {returncode}")
        child_runs = sorted(child_root.glob("phase0_MATH500_*"))
        if len(child_runs) != 1:
            raise RuntimeError(f"Expected one child run under {child_root}, found {len(child_runs)}")
        child_run = child_runs[0]
        child_fields = _validate_child(row, child_run)
        baseline_tokens = int(row.get("baseline_mas_approx_total_tokens") or 0)
        random_tokens = int(child_fields.get("random_mas_approx_total_tokens") or 0)
        result.update(
            {
                "status": "completed",
                "child_run_dir": str(child_run),
                "token_delta_random_minus_baseline": random_tokens - baseline_tokens,
                **child_fields,
            }
        )
    except Exception as exc:  # noqa: BLE001 - persist branch failure for targeted resume.
        result["error"] = repr(exc)
    _write_json_atomic(branch_dir / "result.json", result)
    status = result["status"]
    print(f"[{status}] {int(row['plan_index']) + 1}/400 {_branch_name(row)}", flush=True)
    return result


async def run_random_stage_plan(args: argparse.Namespace) -> dict[str, Any]:
    plan_path = Path(args.plan).resolve()
    plan = _read_jsonl(plan_path)
    output_root = Path(args.output_dir).resolve()
    digest = _plan_digest(plan_path)
    manifest_path = output_root / "manifest.json"
    if output_root.exists():
        if not args.resume:
            raise FileExistsError(f"Output directory exists; use --resume: {output_root}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("plan_digest") != digest:
            raise ValueError("Existing output directory belongs to a different random stage plan")
    else:
        output_root.mkdir(parents=True)
        manifest = {
            "plan_version": PLAN_VERSION,
            "plan": str(plan_path),
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
    _write_jsonl(output_root / "random_results.jsonl", results)
    completed = sum(row.get("status") == "completed" for row in results)
    failed = len(results) - completed
    manifest.update(
        {
            "workers": args.workers,
            "temperature": args.temperature,
            "completed": completed,
            "failed": failed,
            "resumed_at": _utc_now() if args.resume else None,
            "finished_at": _utc_now(),
        }
    )
    _write_json_atomic(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and execute one random stage replay per train example.")
    parser.add_argument("--master-dir", required=True)
    parser.add_argument("--plan-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-drop-min-edges", type=int, default=2)
    parser.add_argument("--expected-examples", type=int, default=400)
    parser.add_argument("--workers", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--inter-launch-delay-seconds", type=float, default=0.05)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.random_drop_min_edges < 2:
        raise ValueError("--random-drop-min-edges must be at least 2")
    plan_dir = Path(args.plan_dir)
    plan_path = plan_dir / "random_stage_plan.jsonl"
    if not plan_path.exists():
        plan_path = build_random_stage_plan(
            Path(args.master_dir),
            plan_dir,
            seed=args.seed,
            min_dropped_edges=args.random_drop_min_edges,
            expected_examples=args.expected_examples,
        )
        print(f"Random stage plan written to: {plan_path}")
    plan_rows = _read_jsonl(plan_path)
    for row in plan_rows:
        row["temperature"] = args.temperature
    _write_jsonl(plan_path, plan_rows)
    plan_summary_path = plan_dir / "plan_summary.json"
    if plan_summary_path.is_file():
        plan_summary = json.loads(plan_summary_path.read_text(encoding="utf-8"))
        plan_summary["temperature"] = args.temperature
        _write_json_atomic(plan_summary_path, plan_summary)
    if args.build_only:
        return 0
    args.plan = str(plan_path)
    manifest = asyncio.run(run_random_stage_plan(args))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 1 if int(manifest.get("failed", 0)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
