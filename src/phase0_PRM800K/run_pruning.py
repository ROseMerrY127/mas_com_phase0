from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
from typing import Any, Callable

from . import run as phase0_run
from .edge_pruning import build_pruning_policy


def _arg_or_config(args: argparse.Namespace, name: str, config: dict[str, Any], default: Any = None) -> Any:
    value = getattr(args, name)
    return value if value is not None else config.get(name, default)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


async def run_phase0_pruning(args: argparse.Namespace) -> Path:
    config = phase0_run._load_config(Path(args.config))
    replay_from_run = _arg_or_config(args, "replay_from_run", config)
    checkpoint_candidate = _arg_or_config(args, "checkpoint_candidate_id", config)
    replay_mode = bool(replay_from_run or checkpoint_candidate)
    edge_policy_name = str(_arg_or_config(args, "edge_policy", config, "identity"))
    replay_policy_name = str(_arg_or_config(args, "replay_policy_name", config, edge_policy_name))
    active_policy_name = replay_policy_name if replay_mode else edge_policy_name

    configured_candidate = _arg_or_config(args, "lomo_candidate_id", config)
    lomo_candidate_id = str(configured_candidate or checkpoint_candidate) if configured_candidate or checkpoint_candidate else None
    drop_probability = float(_arg_or_config(args, "random_drop_probability", config, 0.25))
    policy_seed = int(_arg_or_config(args, "random_policy_seed", config, 0))
    include_self_edges = _as_bool(_arg_or_config(args, "pruning_include_self_edges", config, False))

    def pruning_factory(name: str | None):
        return build_pruning_policy(
            name,
            lomo_candidate_id=lomo_candidate_id,
            random_drop_probability=drop_probability,
            random_policy_seed=policy_seed,
            include_self_edges=include_self_edges,
        )

    # The original runner resolves policies through this module-level factory. Replace it only
    # for this call, then restore it so the standard Phase0 entry point remains unchanged.
    original_factory: Callable[..., Any] = phase0_run.build_edge_policy
    original_router = phase0_run.FullForwardRouter
    phase0_run.build_edge_policy = pruning_factory
    counter_offsets = getattr(args, "replay_counter_offsets", None)
    if counter_offsets is not None:
        def scoped_router_factory(*factory_args: Any, **factory_kwargs: Any):
            router = original_router(*factory_args, **factory_kwargs)
            router._counters.update({key: int(value) for key, value in counter_offsets.items()})
            return router

        phase0_run.FullForwardRouter = scoped_router_factory
    try:
        run_dir = await phase0_run.run_phase0_prm800k(args)
    finally:
        phase0_run.build_edge_policy = original_factory
        phase0_run.FullForwardRouter = original_router

    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["pruning"] = {
        "active_policy": active_policy_name,
        "lomo_candidate_id": lomo_candidate_id,
        "random_drop_probability": drop_probability,
        "random_policy_seed": policy_seed,
        "include_self_edges": include_self_edges,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = phase0_run.build_parser()
    parser.description = "Run Phase0 PRM800K with standalone LOMO or random edge pruning."
    parser.set_defaults(config="config/phase0_PRM800K_pruning.yaml")
    parser.add_argument("--lomo-candidate-id")
    parser.add_argument("--random-drop-probability", type=float)
    parser.add_argument("--random-policy-seed", type=int)
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
