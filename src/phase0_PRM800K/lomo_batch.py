from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

from .edge_pruning import PRUNABLE_MESSAGE_KINDS
from . import run as phase0_run
from .run_pruning import _arg_or_config, _as_bool
from .run_pruning import build_parser as build_pruning_parser
from .run_pruning import run_phase0_pruning
from .scoped_replay import build_scoped_parent, scoped_parent_name


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Required baseline file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        handle.flush()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def flatten_candidate_run(run_dir: Path, candidate_dir: Path) -> Path:
    """Move a Phase0 run's files directly under its candidate directory."""
    if run_dir.parent != candidate_dir:
        raise ValueError(f"Run directory {run_dir} is not directly under candidate directory {candidate_dir}")
    for item in run_dir.iterdir():
        destination = candidate_dir / item.name
        if destination.exists():
            raise FileExistsError(f"Cannot flatten candidate run; destination already exists: {destination}")
        item.replace(destination)
    run_dir.rmdir()
    return candidate_dir


def select_prunable_candidates(
    rows: list[dict[str, Any]],
    *,
    include_self_edges: bool = False,
    question_id: int | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if str(row.get("kind")) in PRUNABLE_MESSAGE_KINDS
        and (include_self_edges or row.get("sender") != row.get("recipient"))
        and (question_id is None or int(row.get("question_id", -1)) == question_id)
    ]
    if limit is not None:
        if limit < 1:
            raise ValueError("lomo_limit must be at least 1")
        return selected[:limit]
    return selected


def _prediction_for_candidate(rows: list[dict[str, Any]], candidate: dict[str, Any]) -> dict[str, Any] | None:
    for row in rows:
        if (
            str(row.get("split")) == str(candidate.get("split"))
            and int(row.get("question_id", -1)) == int(candidate.get("question_id", -1))
            and int(row.get("source_index", -1)) == int(candidate.get("source_index", -1))
        ):
            return row
    return None


def _example_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return str(row.get("split")), int(row.get("question_id", -1)), int(row.get("source_index", -1))


async def run_lomo_batch(args: argparse.Namespace) -> Path:
    config = phase0_run._load_config(Path(args.config))
    replay_from_run = _arg_or_config(args, "replay_from_run", config)
    if not replay_from_run:
        raise ValueError("Batch LOMO requires --replay-from-run pointing to an identity baseline")
    include_self_edges = _as_bool(_arg_or_config(args, "pruning_include_self_edges", config, False))

    parent_run = Path(str(replay_from_run))
    candidates = _read_jsonl(parent_run / "edge_candidates.jsonl")
    baseline_predictions = _read_jsonl(parent_run / "predictions.jsonl")
    successful_examples = {_example_key(row) for row in baseline_predictions}
    candidates = [row for row in candidates if _example_key(row) in successful_examples]
    requested_candidate_id = _arg_or_config(args, "lomo_candidate_id", config)
    if requested_candidate_id is not None:
        candidates = [row for row in candidates if str(row.get("candidate_id")) == str(requested_candidate_id)]
    requested_candidate_ids = getattr(args, "lomo_candidate_ids", None)
    if requested_candidate_ids is not None:
        candidates = [row for row in candidates if str(row.get("candidate_id")) in requested_candidate_ids]
    selected = select_prunable_candidates(
        candidates,
        include_self_edges=include_self_edges,
        question_id=args.lomo_question_id,
        limit=args.lomo_limit,
    )
    if not selected:
        raise ValueError("No prunable candidates matched the batch filters")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = Path(args.output_dir or "runs")
    batch_dir_override = getattr(args, "batch_output_dir", None)
    batch_dir = Path(batch_dir_override) if batch_dir_override is not None else output_root / f"lomo_batch_{timestamp}"
    batch_dir.mkdir(parents=True, exist_ok=False)
    results_path = batch_dir / "lomo_results.jsonl"
    manifest_path = batch_dir / "manifest.json"
    manifest: dict[str, Any] = {
        "parent_run": str(parent_run),
        "num_candidates": len(selected),
        "completed": 0,
        "failed": 0,
        "include_self_edges": include_self_edges,
        "question_id_filter": args.lomo_question_id,
        "candidate_id_filter": str(requested_candidate_id) if requested_candidate_id is not None else None,
        "candidate_plan_filter_count": len(requested_candidate_ids) if requested_candidate_ids is not None else None,
        "limit": args.lomo_limit,
        "scoped_replay": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(manifest_path, manifest)
    scoped_workspace = tempfile.TemporaryDirectory(prefix="phase0-lomo-scoped-")
    scoped_root = Path(scoped_workspace.name)

    for index, candidate in enumerate(selected, start=1):
        candidate_id = str(candidate["candidate_id"])
        print(f"LOMO {index}/{len(selected)}: dropping {candidate_id}", flush=True)
        scoped_dir = scoped_root / scoped_parent_name(
            parent_run,
            split=str(candidate["split"]),
            question_id=int(candidate["question_id"]),
            source_index=int(candidate["source_index"]),
        )
        scoped_metadata = build_scoped_parent(
            parent_run,
            scoped_dir,
            split=str(candidate["split"]),
            question_id=int(candidate["question_id"]),
            source_index=int(candidate["source_index"]),
        )
        child_args = argparse.Namespace(**vars(args))
        child_args.prepare_only = False
        child_args.edge_policy = "identity"
        child_args.replay_from_run = str(scoped_dir)
        child_args.checkpoint_candidate_id = candidate_id
        child_args.replay_policy_name = "lomo"
        child_args.lomo_candidate_id = candidate_id
        child_args.pruning_include_self_edges = include_self_edges
        child_args.replay_counter_offsets = scoped_metadata["counter_offsets"]
        child_args.output_dir = str(batch_dir / candidate_id)

        baseline_prediction = _prediction_for_candidate(baseline_predictions, candidate)
        try:
            child_run = await run_phase0_pruning(child_args)
            child_run = flatten_candidate_run(child_run, batch_dir / candidate_id)
            child_predictions = _read_jsonl(child_run / "predictions.jsonl")
            if len(child_predictions) != 1:
                raise RuntimeError(f"Scoped child {child_run} produced {len(child_predictions)} predictions instead of 1")
            child_summary_path = child_run / "summary.json"
            child_summary = json.loads(child_summary_path.read_text(encoding="utf-8"))
            child_summary["replay_from_run"] = str(parent_run)
            child_summary["parent_run_id"] = parent_run.name
            child_summary["scoped_replay"] = {
                "original_parent_run": str(parent_run),
                "temporary_parent_persisted": False,
                "split": candidate.get("split"),
                "question_id": candidate.get("question_id"),
                "source_index": candidate.get("source_index"),
            }
            child_summary_path.write_text(
                json.dumps(child_summary, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            lomo_prediction = _prediction_for_candidate(child_predictions, candidate)
            baseline_tokens = int((baseline_prediction or {}).get("mas_approx_total_tokens", 0))
            lomo_tokens = int((lomo_prediction or {}).get("mas_approx_total_tokens", 0))
            result = {
                "candidate_id": candidate_id,
                "status": "completed",
                "split": candidate.get("split"),
                "question_id": candidate.get("question_id"),
                "source_index": candidate.get("source_index"),
                "round": candidate.get("round"),
                "sender": candidate.get("sender"),
                "recipient": candidate.get("recipient"),
                "kind": candidate.get("kind"),
                "original_content": candidate.get("original_content"),
                "baseline_mas_output": (baseline_prediction or {}).get("mas_output"),
                "lomo_mas_output": (lomo_prediction or {}).get("mas_output"),
                "baseline_mas_approx_total_tokens": baseline_tokens,
                "lomo_mas_approx_total_tokens": lomo_tokens,
                "token_delta_lomo_minus_baseline": lomo_tokens - baseline_tokens,
                "baseline_rounds_used": (baseline_prediction or {}).get("rounds_used"),
                "lomo_rounds_used": (lomo_prediction or {}).get("rounds_used"),
                "original_parent_run": str(parent_run),
                "child_prediction_count": len(child_predictions),
                "child_run_dir": str(child_run),
            }
            manifest["completed"] += 1
        except Exception as exc:  # noqa: BLE001 - preserve failed intervention in the batch manifest.
            result = {
                "candidate_id": candidate_id,
                "status": "failed",
                "split": candidate.get("split"),
                "question_id": candidate.get("question_id"),
                "source_index": candidate.get("source_index"),
                "round": candidate.get("round"),
                "sender": candidate.get("sender"),
                "recipient": candidate.get("recipient"),
                "kind": candidate.get("kind"),
                "error": repr(exc),
            }
            manifest["failed"] += 1
            _append_jsonl(results_path, result)
            _write_json(manifest_path, manifest)
            if not args.batch_continue_on_error:
                raise
            continue

        _append_jsonl(results_path, result)
        _write_json(manifest_path, manifest)

    scoped_workspace.cleanup()
    return batch_dir


def build_parser() -> argparse.ArgumentParser:
    parser = build_pruning_parser()
    parser.description = "Run one independent LOMO replay for every eligible edge in a baseline."
    parser.add_argument("--lomo-question-id", type=int)
    parser.add_argument("--lomo-limit", type=int)
    parser.add_argument("--batch-continue-on-error", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        batch_dir = asyncio.run(run_lomo_batch(args))
    except Exception as exc:  # noqa: BLE001 - CLI reports the batch failure succinctly.
        print(f"Batch LOMO failed: {exc}", file=sys.stderr)
        return 1
    print(f"Batch LOMO outputs written to: {batch_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
