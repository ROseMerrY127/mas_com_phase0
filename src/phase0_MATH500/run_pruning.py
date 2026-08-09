from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Callable

from . import run as phase0_run
from .edge_pruning import build_pruning_policy
from .scoped_replay import build_scoped_parent, scoped_parent_name


_RANDOM_POLICY_NAMES = frozenset({"random", "random_drop"})
_LOMO_POLICY_NAMES = frozenset({"lomo", "drop_candidate_once"})


def _arg_or_config(args: argparse.Namespace, name: str, config: dict[str, Any], default: Any = None) -> Any:
    value = getattr(args, name)
    return value if value is not None else config.get(name, default)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Replay log not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _resolve_random_drop_stage(
    *,
    replay_from_run: str | Path,
    checkpoint_candidate_id: str,
    include_self_edges: bool,
    min_dropped_edges: int,
) -> tuple[str, dict[str, Any]]:
    parent = Path(replay_from_run)
    summary_path = parent / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Identity replay summary not found: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if str(summary.get("edge_policy", "")) != "identity":
        raise ValueError("random_drop replay requires a parent run whose edge_policy is identity")

    candidates = _read_jsonl(parent / "edge_candidates.jsonl")
    checkpoint = next(
        (row for row in candidates if str(row.get("candidate_id")) == checkpoint_candidate_id),
        None,
    )
    if checkpoint is None:
        raise ValueError(f"checkpoint_candidate_id not found in identity parent: {checkpoint_candidate_id}")
    stage_action_id = str(checkpoint.get("stage_action_id") or "")
    if not stage_action_id:
        raise ValueError(
            "random_drop requires migrated/new identity logs with stage_action_id and stage_actions.jsonl"
        )

    stage_actions = _read_jsonl(parent / "stage_actions.jsonl")
    stage_action = next(
        (row for row in stage_actions if str(row.get("stage_action_id")) == stage_action_id),
        None,
    )
    if stage_action is None:
        raise ValueError(f"Identity parent is missing stage action {stage_action_id}")
    action_mask = str(stage_action.get("action_mask", ""))
    if any(bit != "0" for bit in action_mask):
        raise ValueError(f"random_drop parent stage is not identity: {stage_action_id} mask={action_mask!r}")

    stage_candidates = [row for row in candidates if str(row.get("stage_action_id")) == stage_action_id]
    actionable = [
        row
        for row in stage_candidates
        if str(row.get("kind")) in {"plan", "solver_step", "judge_feedback"}
        and (include_self_edges or row.get("sender") != row.get("recipient"))
    ]
    if len(actionable) < min_dropped_edges:
        raise ValueError(
            f"random_drop target stage {stage_action_id} has only {len(actionable)} actionable edges; "
            f"need at least {min_dropped_edges}"
        )
    return stage_action_id, checkpoint


async def run_phase0_pruning(args: argparse.Namespace) -> Path:
    config = phase0_run._load_config(Path(args.config))
    replay_from_run = _arg_or_config(args, "replay_from_run", config)
    checkpoint_candidate = _arg_or_config(args, "checkpoint_candidate_id", config)
    replay_mode = bool(replay_from_run or checkpoint_candidate)
    edge_policy_name = str(_arg_or_config(args, "edge_policy", config, "identity"))
    if args.replay_policy_name is not None:
        replay_policy_name = str(args.replay_policy_name)
    elif replay_mode and args.edge_policy is not None:
        replay_policy_name = edge_policy_name
    else:
        replay_policy_name = str(config.get("replay_policy_name", edge_policy_name))
    active_policy_name = replay_policy_name if replay_mode else edge_policy_name
    normalized_active_policy = active_policy_name.strip().lower()

    configured_candidate = _arg_or_config(args, "lomo_candidate_id", config)
    lomo_target = configured_candidate or checkpoint_candidate
    lomo_candidate_id = (
        str(lomo_target) if normalized_active_policy in _LOMO_POLICY_NAMES and lomo_target is not None else None
    )
    policy_seed = int(_arg_or_config(args, "random_policy_seed", config, 0))
    min_dropped_edges = int(_arg_or_config(args, "random_drop_min_edges", config, 2))
    include_self_edges = _as_bool(_arg_or_config(args, "pruning_include_self_edges", config, False))
    random_drop_stage_action_id: str | None = None
    checkpoint_row: dict[str, Any] | None = None
    if normalized_active_policy in _RANDOM_POLICY_NAMES:
        if not replay_from_run or not checkpoint_candidate:
            raise ValueError(
                "random_drop is replay-only and requires both --replay-from-run and --checkpoint-candidate-id"
            )
        random_drop_stage_action_id, checkpoint_row = _resolve_random_drop_stage(
            replay_from_run=str(replay_from_run),
            checkpoint_candidate_id=str(checkpoint_candidate),
            include_self_edges=include_self_edges,
            min_dropped_edges=min_dropped_edges,
        )

    effective_args = args
    original_parent_run: Path | None = None
    scoped_metadata: dict[str, Any] | None = None
    scoped_workspace: tempfile.TemporaryDirectory[str] | None = None
    if checkpoint_row is not None and getattr(args, "replay_counter_offsets", None) is None:
        original_parent_run = Path(str(replay_from_run))
        scoped_workspace = tempfile.TemporaryDirectory(prefix="phase0-random-drop-scoped-")
        scoped_dir = Path(scoped_workspace.name) / scoped_parent_name(
            original_parent_run,
            split=str(checkpoint_row["split"]),
            question_id=int(checkpoint_row["question_id"]),
            source_index=int(checkpoint_row["source_index"]),
        )
        try:
            scoped_metadata = build_scoped_parent(
                original_parent_run,
                scoped_dir,
                split=str(checkpoint_row["split"]),
                question_id=int(checkpoint_row["question_id"]),
                source_index=int(checkpoint_row["source_index"]),
            )
        except Exception:
            scoped_workspace.cleanup()
            raise
        effective_args = argparse.Namespace(**vars(args))
        effective_args.replay_from_run = str(scoped_dir)
        effective_args.replay_counter_offsets = scoped_metadata["counter_offsets"]
    if replay_mode:
        if effective_args is args:
            effective_args = argparse.Namespace(**vars(args))
        effective_args.replay_policy_name = active_policy_name

    def pruning_factory(name: str | None):
        return build_pruning_policy(
            name,
            lomo_candidate_id=lomo_candidate_id,
            random_policy_seed=policy_seed,
            random_drop_min_edges=min_dropped_edges,
            random_drop_stage_action_id=random_drop_stage_action_id,
            include_self_edges=include_self_edges,
        )

    # The original runner resolves policies through this module-level factory. Replace it only
    # for this call, then restore it so the standard Phase0 entry point remains unchanged.
    original_factory: Callable[..., Any] = phase0_run.build_edge_policy
    original_router = phase0_run.FullForwardRouter
    phase0_run.build_edge_policy = pruning_factory
    counter_offsets = getattr(effective_args, "replay_counter_offsets", None)
    if counter_offsets is not None:
        def scoped_router_factory(*factory_args: Any, **factory_kwargs: Any):
            router = original_router(*factory_args, **factory_kwargs)
            router._counters.update({key: int(value) for key, value in counter_offsets.items()})
            return router

        phase0_run.FullForwardRouter = scoped_router_factory
    try:
        run_dir = await phase0_run.run_phase0_math500(effective_args)
    finally:
        phase0_run.build_edge_policy = original_factory
        phase0_run.FullForwardRouter = original_router
        if scoped_workspace is not None:
            scoped_workspace.cleanup()

    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["pruning"] = {
        "active_policy": active_policy_name,
        "lomo_candidate_id": lomo_candidate_id,
        "random_policy_seed": policy_seed,
        "random_drop_min_edges": min_dropped_edges,
        "random_drop_action": "uniform_stage_subset",
        "random_drop_stage_action_id": random_drop_stage_action_id,
        "random_drop_scope": "checkpoint_stage_once_then_identity",
        "include_self_edges": include_self_edges,
    }
    if original_parent_run is not None and checkpoint_row is not None and scoped_metadata is not None:
        summary["replay_from_run"] = str(original_parent_run)
        summary["parent_run_id"] = original_parent_run.name
        summary["scoped_replay"] = {
            "original_parent_run": str(original_parent_run),
            "temporary_parent_persisted": False,
            "split": checkpoint_row["split"],
            "question_id": checkpoint_row["question_id"],
            "source_index": checkpoint_row["source_index"],
            "counter_offsets": scoped_metadata["counter_offsets"],
        }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = phase0_run.build_parser()
    parser.description = "Replay an identity prefix, then apply one LOMO or random stage action."
    parser.set_defaults(config="config/phase0_MATH500_pruning.yaml")
    parser.add_argument("--lomo-candidate-id")
    parser.add_argument("--random-policy-seed", type=int)
    parser.add_argument("--random-drop-min-edges", type=int)
    parser.add_argument("--pruning-include-self-edges", action=argparse.BooleanOptionalAction, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        run_dir = asyncio.run(run_phase0_pruning(args))
    except Exception as exc:  # noqa: BLE001 - CLI reports the experiment failure succinctly.
        print(f"Phase0 pruning failed: {exc}", file=sys.stderr)
        return 1
    print(f"Phase0 pruning outputs written to: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
