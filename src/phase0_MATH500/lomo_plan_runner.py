from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any

from .lomo_batch import run_lomo_batch
from .run_pruning import build_parser as build_pruning_parser


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"LOMO plan not found: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _validate_plan(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("LOMO plan is empty")
    keys = [(str(row["parent_run"]), str(row["candidate_id"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("LOMO plan contains duplicate (parent_run, candidate_id) entries")


def _shard_number(parent_run: str) -> int:
    shard_name = Path(parent_run).parent.name
    match = re.fullmatch(r"shard_(\d+)(?:_.*)?", shard_name)
    if match is None:
        raise ValueError(f"Cannot determine shard number from parent run: {parent_run}")
    return int(match.group(1))


def _existing_results(run_dir: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    finished_shards = 0
    for shard_dir in sorted(run_dir.glob("shard_*")):
        results_path = shard_dir / "lomo_results.jsonl"
        shard_rows = _read_jsonl(results_path) if results_path.exists() else []
        rows.extend(shard_rows)
        manifest_path = shard_dir / "manifest.json"
        if manifest_path.exists():
            shard_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if len(shard_rows) >= int(shard_manifest.get("num_candidates", 0)):
                finished_shards += 1
    return rows, finished_shards


async def run_lomo_plan(args: argparse.Namespace) -> Path:
    plan_rows = _read_jsonl(Path(args.lomo_plan))
    _validate_plan(plan_rows)
    if args.plan_limit is not None:
        if args.plan_limit < 1:
            raise ValueError("plan_limit must be at least 1")
        plan_rows = plan_rows[: args.plan_limit]

    groups: dict[str, set[str]] = defaultdict(set)
    for row in plan_rows:
        groups[str(row["parent_run"])].add(str(row["candidate_id"]))

    start_shard = args.plan_start_shard
    if start_shard is not None and start_shard < 0:
        raise ValueError("plan_start_shard must be at least 0")
    ordered_groups = sorted(groups.items(), key=lambda item: _shard_number(item[0]))
    selected_groups = [
        item for item in ordered_groups if start_shard is None or _shard_number(item[0]) >= start_shard
    ]
    if not selected_groups:
        raise ValueError(f"No planned shards matched --plan-start-shard {start_shard}")

    run_dir = Path(args.plan_run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    combined_results = run_dir / "lomo_results.jsonl"
    manifest_path = run_dir / "manifest.json"
    previous_manifest: dict[str, Any] = {}
    if manifest_path.exists():
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if str(previous_manifest.get("lomo_plan")) != str(args.lomo_plan):
            raise ValueError(
                f"Existing run directory belongs to a different LOMO plan: {previous_manifest.get('lomo_plan')}"
            )
    existing_rows, finished_shards = _existing_results(run_dir)
    _write_jsonl(combined_results, existing_rows)
    manifest: dict[str, Any] = {
        "lomo_plan": str(args.lomo_plan),
        "planned_candidates": len(plan_rows),
        "num_parent_shards": len(groups),
        "completed": sum(row.get("status") == "completed" for row in existing_rows),
        "failed": sum(row.get("status") == "failed" for row in existing_rows),
        "finished_shards": finished_shards,
        "start_shard": start_shard,
        "skipped_existing_shards": 0,
        "created_at": previous_manifest.get("created_at", datetime.now(timezone.utc).isoformat()),
        "resumed_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(manifest_path, manifest)

    for index, (parent_run, candidate_ids) in enumerate(selected_groups, start=1):
        shard_name = Path(parent_run).parent.name
        shard_output_dir = run_dir / shard_name
        if shard_output_dir.exists():
            manifest["skipped_existing_shards"] += 1
            print(f"Skipping existing LOMO shard: {shard_name}", flush=True)
            _write_json(manifest_path, manifest)
            continue
        print(
            f"Planned LOMO shard {index}/{len(selected_groups)}: {shard_name}, {len(candidate_ids)} candidates",
            flush=True,
        )
        shard_args = argparse.Namespace(**vars(args))
        shard_args.replay_from_run = parent_run
        shard_args.lomo_candidate_id = None
        shard_args.lomo_candidate_ids = candidate_ids
        shard_args.batch_output_dir = str(shard_output_dir)
        shard_args.output_dir = str(run_dir)
        shard_args.batch_continue_on_error = True
        shard_args.lomo_question_id = None
        shard_args.lomo_limit = None

        try:
            shard_dir = await run_lomo_batch(shard_args)
            shard_results = _read_jsonl(shard_dir / "lomo_results.jsonl")
            _append_jsonl(combined_results, shard_results)
            manifest["completed"] += sum(row.get("status") == "completed" for row in shard_results)
            manifest["failed"] += sum(row.get("status") == "failed" for row in shard_results)
            manifest["finished_shards"] += 1
        except Exception as exc:  # noqa: BLE001 - retain shard failure and optionally continue the plan.
            manifest["failed"] += len(candidate_ids)
            manifest.setdefault("shard_errors", []).append(
                {"parent_run": parent_run, "candidate_count": len(candidate_ids), "error": repr(exc)}
            )
            _write_json(manifest_path, manifest)
            if not args.plan_continue_on_error:
                raise
            continue
        _write_json(manifest_path, manifest)

    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(manifest_path, manifest)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = build_pruning_parser()
    parser.description = "Execute an exact stratified LOMO candidate plan across parent shards."
    parser.add_argument("--lomo-plan", required=True)
    parser.add_argument("--plan-run-dir", required=True)
    parser.add_argument("--plan-limit", type=int)
    parser.add_argument(
        "--plan-start-shard",
        type=int,
        help="Start at this shard index; existing shard output directories are always skipped.",
    )
    parser.add_argument("--plan-continue-on-error", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        run_dir = asyncio.run(run_lomo_plan(args))
    except Exception as exc:  # noqa: BLE001 - CLI reports plan execution failure succinctly.
        print(f"Planned LOMO failed: {exc}", file=sys.stderr)
        return 1
    print(f"Planned LOMO outputs written to: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
