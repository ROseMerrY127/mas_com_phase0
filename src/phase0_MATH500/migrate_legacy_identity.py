from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterable


MIGRATION_VERSION = "legacy_identity_stage_v1"
PRUNABLE_MESSAGE_KINDS = frozenset({"plan", "solver_step", "judge_feedback"})
STAGE_EDGE_ORDER: dict[str, tuple[tuple[str, str], ...]] = {
    "input": (("Input", "Planner"),),
    "planner": (
        ("Planner", "SolverA"),
        ("Planner", "SolverB"),
        ("Planner", "Judger"),
    ),
    "solver": (
        ("SolverA", "Planner"),
        ("SolverA", "Judger"),
        ("SolverA", "SolverA"),
        ("SolverB", "Planner"),
        ("SolverB", "Judger"),
        ("SolverB", "SolverB"),
    ),
    "judger": (
        ("Judger", "Planner"),
        ("Judger", "SolverA"),
        ("Judger", "SolverB"),
        ("Judger", "Judger"),
        ("Judger", "Output"),
    ),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object in {path} at line {line_number}")
            rows.append(row)
    return rows


def _write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _stage_for_candidate(candidate: dict[str, Any]) -> str:
    kind = str(candidate.get("kind", ""))
    stage = {
        "input": "input",
        "plan": "planner",
        "solver_step": "solver",
        "judge_feedback": "judger",
        "final_output": "judger",
    }.get(kind)
    if stage is None:
        raise ValueError(f"Cannot infer stage for candidate {candidate.get('candidate_id')}: kind={kind!r}")
    return stage


def _stage_action_id(candidate: dict[str, Any], stage: str) -> str:
    return (
        f"{candidate['split']}:{candidate['source_index']}:{candidate['question_id']}:"
        f"r{candidate['round']}:{stage}"
    )


def _is_actionable(candidate: dict[str, Any]) -> bool:
    return (
        str(candidate.get("kind")) in PRUNABLE_MESSAGE_KINDS
        and str(candidate.get("sender")) != str(candidate.get("recipient"))
    )


def _ordered_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _group_candidates(candidates: list[dict[str, Any]]) -> list[tuple[str, str, list[dict[str, Any]]]]:
    groups: list[tuple[str, str, list[dict[str, Any]]]] = []
    closed_ids: set[str] = set()
    for candidate in candidates:
        stage = _stage_for_candidate(candidate)
        action_id = _stage_action_id(candidate, stage)
        if not groups or groups[-1][0] != action_id:
            if action_id in closed_ids:
                raise ValueError(f"Non-contiguous legacy stage group: {action_id}")
            if groups:
                closed_ids.add(groups[-1][0])
            groups.append((action_id, stage, []))
        groups[-1][2].append(candidate)
    return groups


def _validate_edge_order(stage: str, candidates: list[dict[str, Any]], action_id: str) -> None:
    pairs = [(str(row["sender"]), str(row["recipient"])) for row in candidates]
    expected_positions = {edge: index for index, edge in enumerate(STAGE_EDGE_ORDER[stage])}
    if any(pair not in expected_positions for pair in pairs):
        raise ValueError(f"Unexpected edge in {action_id}: {pairs}")
    expected = sorted(pairs, key=expected_positions.__getitem__)
    if pairs != expected or len(set(pairs)) != len(pairs):
        raise ValueError(f"Legacy edge order is not a unique current-stage order in {action_id}: {pairs}")


def _stage_hash(stage: str, candidates: list[dict[str, Any]]) -> str:
    first = candidates[0]
    payload = {
        "split": first["split"],
        "question_id": first["question_id"],
        "source_index": first["source_index"],
        "round": first["round"],
        "stage": stage,
        "edges": [
            {
                "sender": candidate["sender"],
                "recipient": candidate["recipient"],
                "kind": candidate["kind"],
                "content": candidate["original_content"],
                "local_state_hash": candidate["state_hash"],
            }
            for candidate in candidates
        ],
    }
    return _content_hash(_stable_json(payload))


def _build_stage_metadata(
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    migrated_candidates: list[dict[str, Any]] = []
    metadata_by_candidate: dict[str, dict[str, Any]] = {}
    stage_actions: list[dict[str, Any]] = []

    for action_id, stage, group in _group_candidates(candidates):
        _validate_edge_order(stage, group, action_id)
        state_hash = _stage_hash(stage, group)
        actionable = [candidate for candidate in group if _is_actionable(candidate)]
        action_indexes = {str(candidate["candidate_id"]): index for index, candidate in enumerate(actionable)}
        action_mask = "0" * len(actionable)

        for stage_edge_index, candidate in enumerate(group):
            candidate_id = str(candidate["candidate_id"])
            migrated = {
                **candidate,
                "stage": stage,
                "stage_action_id": action_id,
                "stage_edge_index": stage_edge_index,
                "stage_state_hash": state_hash,
            }
            migrated_candidates.append(migrated)
            metadata_by_candidate[candidate_id] = {
                "stage": stage,
                "stage_action_id": action_id,
                "stage_state_hash": state_hash,
                "action_mask": action_mask,
                "action_edge_index": action_indexes.get(candidate_id),
            }

        first = group[0]
        stage_actions.append(
            {
                "stage_action_id": action_id,
                "run_id": first["run_id"],
                "split": first["split"],
                "question_id": first["question_id"],
                "source_index": first["source_index"],
                "round": first["round"],
                "stage": stage,
                "policy_name": "identity",
                "state_message_ids": _ordered_unique(
                    str(message_id)
                    for candidate in group
                    for message_id in candidate.get("state_message_ids", [])
                ),
                "state_hash": state_hash,
                "candidate_ids": [str(candidate["candidate_id"]) for candidate in group],
                "action_candidate_ids": [str(candidate["candidate_id"]) for candidate in actionable],
                "edge_order": [f"{candidate['sender']}->{candidate['recipient']}" for candidate in actionable],
                "action_mask": action_mask,
                "dropped_candidate_ids": [],
                "replayed_candidate_ids": [],
                "reward": None,
            }
        )

    if len(metadata_by_candidate) != len(candidates):
        raise ValueError("Candidate IDs are not unique within the legacy run")
    return migrated_candidates, metadata_by_candidate, stage_actions


def _migrate_decisions(path: Path, metadata: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    seen: set[str] = set()
    migrated: list[dict[str, Any]] = []
    for row in rows:
        candidate_id = str(row["candidate_id"])
        if candidate_id not in metadata:
            raise ValueError(f"Decision references unknown candidate {candidate_id} in {path}")
        if row.get("action") != "keep" or bool(row.get("dropped")):
            raise ValueError(f"Legacy run is not identity at candidate {candidate_id} in {path}")
        seen.add(candidate_id)
        stage = metadata[candidate_id]
        migrated.append(
            {
                **row,
                "stage_action_id": stage["stage_action_id"],
                "action_mask": stage["action_mask"],
                "action_edge_index": stage["action_edge_index"],
            }
        )
    if seen != set(metadata):
        raise ValueError(f"Candidate/decision mismatch in {path}")
    return migrated


def _add_stage_fields(path: Path, metadata: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    migrated: list[dict[str, Any]] = []
    for row in rows:
        candidate_id = str(row["candidate_id"])
        if candidate_id not in metadata:
            raise ValueError(f"{path.name} references unknown candidate {candidate_id}")
        stage = metadata[candidate_id]
        migrated.append(
            {
                **row,
                "stage": stage["stage"],
                "stage_action_id": stage["stage_action_id"],
                "stage_state_hash": stage["stage_state_hash"],
                "action_mask": stage["action_mask"],
                "action_edge_index": stage["action_edge_index"],
            }
        )
    return migrated


def migrate_run(run_dir: Path, source_run_dir: Path) -> dict[str, Any]:
    candidate_path = run_dir / "edge_candidates.jsonl"
    decision_path = run_dir / "edge_decisions.jsonl"
    candidates = _read_jsonl(candidate_path)
    migrated_candidates, metadata, stage_actions = _build_stage_metadata(candidates)
    decisions = _migrate_decisions(decision_path, metadata)

    _write_jsonl_atomic(candidate_path, migrated_candidates)
    _write_jsonl_atomic(decision_path, decisions)
    _write_jsonl_atomic(run_dir / "stage_actions.jsonl", stage_actions)

    for name in ("rl_edge_samples.jsonl", "traces.jsonl"):
        path = run_dir / name
        if path.exists():
            _write_jsonl_atomic(path, _add_stage_fields(path, metadata))

    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("edge_policy") != "identity":
        raise ValueError(f"Legacy summary is not identity: {summary_path}")
    summary["train_split_path"] = str((run_dir / "splits" / "train.jsonl").resolve())
    summary["test_split_path"] = str((run_dir / "splits" / "test.jsonl").resolve())
    summary["num_stage_actions"] = len(stage_actions)
    summary["migration"] = {
        "version": MIGRATION_VERSION,
        "source_run_dir": str(source_run_dir.resolve()),
        "migrated_at": _utc_now(),
        "identity_action": "all_zero_stage_mask",
    }
    _write_json_atomic(summary_path, summary)

    stage_counts = Counter(str(row["stage"]) for row in stage_actions)
    return {
        "run_dir": str(run_dir.resolve()),
        "source_run_dir": str(source_run_dir.resolve()),
        "num_candidates": len(candidates),
        "num_decisions": len(decisions),
        "num_stage_actions": len(stage_actions),
        "stage_counts": dict(sorted(stage_counts.items())),
        "num_predictions": int(summary.get("num_examples", 0)),
        "num_errors": len(summary.get("errors", [])),
    }


def _discover_runs(root: Path) -> list[Path]:
    runs = sorted(path.parent for path in root.glob("shard_*/phase0_PRM800K_*/edge_candidates.jsonl"))
    if not runs:
        runs = sorted(path.parent for path in root.glob("shard_*/phase0_MATH500_*/edge_candidates.jsonl"))
    if not runs:
        raise FileNotFoundError(f"No legacy Phase0 run directories found under {root}")
    return runs


def _write_parallel_metadata(
    *,
    source_root: Path,
    output_root: Path,
    run_results: list[dict[str, Any]],
    migrated_at: str,
) -> None:
    legacy_summary_path = source_root / "parallel_summary.json"
    legacy_manifest_path = source_root / "parallel_manifest.json"
    legacy_summary = json.loads(legacy_summary_path.read_text(encoding="utf-8")) if legacy_summary_path.exists() else {}
    legacy_manifest = json.loads(legacy_manifest_path.read_text(encoding="utf-8")) if legacy_manifest_path.exists() else {}
    if legacy_summary_path.exists():
        shutil.copy2(legacy_summary_path, output_root / "legacy_parallel_summary.json")
    if legacy_manifest_path.exists():
        shutil.copy2(legacy_manifest_path, output_root / "legacy_parallel_manifest.json")

    shards: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for shard_index, result in enumerate(run_results):
        run_dir = Path(str(result["run_dir"]))
        shard_dir = run_dir.parent
        summary_path = run_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summaries.append(summary)
        shards.append(
            {
                "shard_index": shard_index,
                "start_index": int(summary.get("start_index", 0)),
                "sample_size": int(summary.get("requested_sample_size", 0)),
                "output_dir": str(shard_dir.resolve()),
                "summary_path": str(summary_path.resolve()),
                "stdout_path": str((shard_dir / "stdout.log").resolve()),
                "stderr_path": str((shard_dir / "stderr.log").resolve()),
                "summary": summary,
                "migrated": True,
            }
        )

    first = summaries[0]
    total_planned = sum(int(summary.get("planned_examples", 0)) for summary in summaries)
    total_completed = sum(int(summary.get("num_examples", 0)) for summary in summaries)
    total_errors = sum(len(summary.get("errors", [])) for summary in summaries)
    total_stage_actions = sum(int(summary.get("num_stage_actions", 0)) for summary in summaries)
    parallel_summary = {
        "master_dir": str(output_root),
        "split": first.get("split"),
        "total_examples_requested": int(legacy_summary.get("total_examples_requested", total_planned)),
        "start_index": int(legacy_summary.get("start_index", 0)),
        "chunk_size": int(legacy_summary.get("chunk_size", first.get("requested_sample_size", 0))),
        "workers": int(legacy_summary.get("workers", 0)),
        "num_shards": len(shards),
        "num_finished_shards": len(shards),
        "num_failed_shards": 0,
        "total_planned_examples": total_planned,
        "total_num_examples": total_completed,
        "total_errors": total_errors,
        "total_stage_actions": total_stage_actions,
        "migration_version": MIGRATION_VERSION,
        "source_root": str(source_root),
        "migrated_at": migrated_at,
        "failed_shard_indices": [],
        "shards": shards,
    }
    parallel_manifest = {
        "master_dir": str(output_root),
        "source_root": str(source_root),
        "migration_version": MIGRATION_VERSION,
        "migrated_at": migrated_at,
        "split": first.get("split"),
        "total_examples": int(legacy_manifest.get("total_examples", total_planned)),
        "start_index": int(legacy_manifest.get("start_index", 0)),
        "chunk_size": int(legacy_manifest.get("chunk_size", first.get("requested_sample_size", 0))),
        "workers": int(legacy_manifest.get("workers", legacy_summary.get("workers", 0))),
        "shards": [
            {
                "shard_index": shard["shard_index"],
                "start_index": shard["start_index"],
                "sample_size": shard["sample_size"],
                "output_dir": shard["output_dir"],
                "summary_path": shard["summary_path"],
            }
            for shard in shards
        ],
    }
    _write_json_atomic(output_root / "parallel_summary.json", parallel_summary)
    _write_json_atomic(output_root / "parallel_manifest.json", parallel_manifest)


def migrate_parallel_root(source_root: Path, output_root: Path) -> dict[str, Any]:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if not source_root.is_dir():
        raise NotADirectoryError(f"Source root is not a directory: {source_root}")
    if output_root.exists():
        raise FileExistsError(f"Output root already exists: {output_root}")

    source_runs = _discover_runs(source_root)
    shutil.copytree(source_root, output_root)
    run_results: list[dict[str, Any]] = []
    try:
        for source_run in source_runs:
            relative = source_run.relative_to(source_root)
            run_results.append(migrate_run(output_root / relative, source_run))

        migrated_at = _utc_now()
        totals = {
            "num_runs": len(run_results),
            "num_candidates": sum(int(row["num_candidates"]) for row in run_results),
            "num_decisions": sum(int(row["num_decisions"]) for row in run_results),
            "num_stage_actions": sum(int(row["num_stage_actions"]) for row in run_results),
            "num_predictions": sum(int(row["num_predictions"]) for row in run_results),
            "num_errors": sum(int(row["num_errors"]) for row in run_results),
        }
        manifest = {
            "migration_version": MIGRATION_VERSION,
            "source_root": str(source_root),
            "output_root": str(output_root),
            "migrated_at": migrated_at,
            **totals,
            "runs": run_results,
        }
        _write_parallel_metadata(
            source_root=source_root,
            output_root=output_root,
            run_results=run_results,
            migrated_at=migrated_at,
        )
        _write_json_atomic(output_root / "migration_manifest.json", manifest)
        return manifest
    except Exception:
        failed_marker = output_root / "MIGRATION_FAILED.txt"
        failed_marker.write_text(
            "Migration did not complete. Do not use this output as a replay parent.\n",
            encoding="utf-8",
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Copy legacy identity logs and reconstruct stage-level all-zero actions."
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = migrate_parallel_root(Path(args.source_root), Path(args.output_root))
    print(json.dumps({key: value for key, value in manifest.items() if key != "runs"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
