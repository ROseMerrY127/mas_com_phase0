from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

from dotenv import load_dotenv
import yaml
from tqdm import tqdm

from .agents import build_model, run_mas_full_path, run_single_agent
from .data import SplitName, build_splits, select_split, write_split_files
from .evaluation import approx_tokens
from .router import FullForwardRouter, ReplayController, build_edge_policy


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Phase0 MATH500 config must be a YAML mapping")
    return data


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _as_split(value: str) -> SplitName:
    if value not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    return value  # type: ignore[return-value]


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _arg_or_config(args: argparse.Namespace, name: str, config: dict[str, Any], default: Any = None) -> Any:
    value = getattr(args, name)
    return value if value is not None else config.get(name, default)


async def run_phase0_math500(args: argparse.Namespace) -> Path:
    load_dotenv()
    config = _load_config(Path(args.config))

    replay_from_run = _arg_or_config(args, "replay_from_run", config)
    checkpoint_candidate_id = _arg_or_config(args, "checkpoint_candidate_id", config)
    replay_mode = replay_from_run is not None or checkpoint_candidate_id is not None
    if replay_mode and (replay_from_run is None or checkpoint_candidate_id is None):
        raise ValueError("Replay requires both --replay-from-run and --checkpoint-candidate-id")

    model_name = str(args.model or config.get("model", "gpt-4o-mini"))
    temperature = float(config.get("temperature", 0.2) if args.temperature is None else args.temperature)
    max_tokens = int(config.get("max_tokens", 1024) if args.max_tokens is None else args.max_tokens)
    continue_on_error = bool(args.continue_on_error or config.get("continue_on_error", False))
    request_retries = int(config.get("request_retries", 3) if args.request_retries is None else args.request_retries)
    retry_backoff_seconds = float(
        config.get("retry_backoff_seconds", 2.0) if args.retry_backoff_seconds is None else args.retry_backoff_seconds
    )
    max_rounds = max(1, int(config.get("max_rounds", 4) if args.max_rounds is None else args.max_rounds))
    stall_rounds = max(0, int(config.get("stall_rounds", 2) if args.stall_rounds is None else args.stall_rounds))
    force_final_on_stop = (
        _as_bool(config.get("force_final_on_stop", True))
        if args.force_final_on_stop is None
        else bool(args.force_final_on_stop)
    )
    edge_policy_name = str(_arg_or_config(args, "edge_policy", config, "identity"))
    replay_policy_name = str(_arg_or_config(args, "replay_policy_name", config, edge_policy_name))

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = Path(args.output_dir or config.get("output_dir", "runs"))
    run_dir = output_root / f"phase0_MATH500_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    parent_predictions: list[dict[str, Any]] = []
    parent_summary: dict[str, Any] = {}
    if replay_mode:
        parent_run_dir = Path(str(replay_from_run))
        parent_predictions = _read_jsonl(parent_run_dir / "predictions.jsonl")
        parent_summary_path = parent_run_dir / "summary.json"
        if parent_summary_path.exists():
            parent_summary = json.loads(parent_summary_path.read_text(encoding="utf-8"))
        split = _as_split(str(parent_summary.get("split", parent_predictions[0].get("split", "test") if parent_predictions else "test")))
        sample_size = len(parent_predictions)
        start_index = int(parent_summary.get("start_index", 0))
        train_size = int(parent_summary.get("train_size", 0))
        test_size = int(parent_summary.get("test_size", 0))
        seed = int(parent_summary.get("seed", 0))
        data_path = Path(str(parent_summary.get("data_path", "")))
        train_path = Path(str(parent_summary.get("train_split_path", "")))
        test_path = Path(str(parent_summary.get("test_split_path", "")))
        split_result_scanned = int(parent_summary.get("scanned_records", 0))
        split_result_valid = int(parent_summary.get("valid_records", 0))
    else:
        data_path = Path(args.data_path or config.get("data_path", "MATH500/test.jsonl"))
        train_size = int(args.train_size if args.train_size is not None else config.get("train_size", 6000))
        test_size = int(args.test_size if args.test_size is not None else config.get("test_size", 2000))
        seed = int(args.seed if args.seed is not None else config.get("seed", 0))
        split = _as_split(str(args.split or config.get("split", "test")))
        config_sample_size = config.get("sample_size")
        sample_size = args.sample_size if args.sample_size is not None else None if config_sample_size is None else int(config_sample_size)
        start_index = int(args.start_index if args.start_index is not None else config.get("start_index", 0))
        split_result = build_splits(data_path, train_size=train_size, test_size=test_size, seed=seed)
        split_dir = run_dir / "splits"
        train_path, test_path = write_split_files(split_result, split_dir)
        split_result_scanned = split_result.scanned_records
        split_result_valid = split_result.valid_records

    base_summary = {
        "split": split,
        "requested_sample_size": sample_size,
        "start_index": start_index,
        "scanned_records": split_result_scanned,
        "valid_records": split_result_valid,
        "train_size": train_size,
        "test_size": test_size,
        "seed": seed,
        "model": model_name,
        "data_path": str(data_path),
        "train_split_path": str(train_path),
        "test_split_path": str(test_path),
        "max_rounds": max_rounds,
        "stall_rounds": stall_rounds,
        "force_final_on_stop": force_final_on_stop,
        "edge_policy": edge_policy_name,
        "replay_from_run": str(replay_from_run) if replay_mode else None,
        "parent_run_id": Path(str(replay_from_run)).name if replay_mode else None,
        "checkpoint_candidate_id": str(checkpoint_candidate_id) if replay_mode else None,
        "replay_policy_name": replay_policy_name if replay_mode else None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    if args.prepare_only:
        _write_json(
            run_dir / "summary.json",
            {
                **base_summary,
                "prepare_only": True,
                "num_examples": 0,
                "errors": [],
            },
        )
        return run_dir

    if replay_mode:
        examples: list[Any] = parent_predictions
        replay_controller = ReplayController(str(replay_from_run), str(checkpoint_candidate_id))
        policy = build_edge_policy(replay_policy_name)
    else:
        examples = select_split(split_result, split, sample_size=sample_size, start_index=start_index)  # type: ignore[name-defined]
        replay_controller = None
        policy = build_edge_policy(edge_policy_name)

    router = FullForwardRouter(run_id=run_dir.name, edge_policy=policy, replay_controller=replay_controller)
    predictions: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    model = build_model(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        request_retries=request_retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )

    def write_outputs() -> None:
        traces = router.as_dicts()
        mas_approx_input_tokens = sum(int(row.get("approx_input_tokens", 0)) for row in traces)
        mas_approx_output_tokens = sum(int(row.get("approx_output_tokens", 0)) for row in traces)
        mas_approx_total_tokens = mas_approx_input_tokens + mas_approx_output_tokens
        single_approx_input_tokens = sum(int(row.get("single_agent_approx_input_tokens", 0)) for row in predictions)
        single_approx_output_tokens = sum(int(row.get("single_agent_approx_output_tokens", 0)) for row in predictions)
        single_approx_total_tokens = single_approx_input_tokens + single_approx_output_tokens
        trace_latency = sum(int(row.get("latency_ms", 0)) for row in traces)
        single_latency = sum(int(row.get("single_agent_latency_ms", 0)) for row in predictions)
        summary = {
            **base_summary,
            "prepare_only": False,
            "num_examples": len(predictions),
            "planned_examples": len(examples),
            "single_agent_approx_input_tokens": single_approx_input_tokens,
            "single_agent_approx_output_tokens": single_approx_output_tokens,
            "single_agent_approx_total_tokens": single_approx_total_tokens,
            "mas_approx_input_tokens": mas_approx_input_tokens,
            "mas_approx_output_tokens": mas_approx_output_tokens,
            "mas_approx_total_tokens": mas_approx_total_tokens,
            "approx_tokens": single_approx_total_tokens + mas_approx_total_tokens,
            "latency_ms": trace_latency + single_latency,
            "num_messages": len(router.messages),
            "num_activations": len(router.activations),
            "num_edge_candidates": len(router.edge_candidates),
            "num_edge_decisions": len(router.edge_decisions),
            "num_stage_actions": len(router.stage_actions),
            "errors": errors,
        }
        _write_jsonl(run_dir / "traces.jsonl", traces)
        _write_jsonl(run_dir / "messages.jsonl", router.messages_as_dicts())
        _write_jsonl(run_dir / "activations.jsonl", router.activations_as_dicts())
        _write_jsonl(run_dir / "edge_candidates.jsonl", router.edge_candidates_as_dicts())
        _write_jsonl(run_dir / "edge_decisions.jsonl", router.edge_decisions_as_dicts())
        _write_jsonl(run_dir / "stage_actions.jsonl", router.stage_actions_as_dicts())
        _write_jsonl(run_dir / "rl_edge_samples.jsonl", router.rl_edge_samples_as_dicts())
        _write_json(run_dir / "summary.json", summary)

    try:
        for example in tqdm(examples, total=len(examples), desc=f"Phase0 MATH500 {split}", unit="example"):
            try:
                if replay_mode:
                    row = example
                    example_split = _as_split(str(row["split"]))
                    question_id = int(row["question_id"])
                    source_index = int(row["source_index"])
                    problem = str(row["problem"])
                    ground_truth_answer = str(row["ground_truth_answer"])
                    finish_reason = row.get("finish_reason")
                    total_time = row.get("total_time")
                    reference_steps = list(row.get("reference_steps", []))
                    single_output = str(row.get("single_agent_output", ""))
                    single_steps = list(row.get("single_agent_steps", []))
                    single_latency = int(row.get("single_agent_latency_ms", 0))
                    single_approx_input_tokens = int(row.get("single_agent_approx_input_tokens", 0))
                    single_approx_output_tokens = int(row.get("single_agent_approx_output_tokens", approx_tokens(single_output)))
                    single_approx_total_tokens = int(
                        row.get("single_agent_approx_total_tokens", single_approx_input_tokens + single_approx_output_tokens)
                    )
                else:
                    row = None
                    example_split = example.split
                    question_id = example.question_id
                    source_index = example.source_index
                    problem = example.problem
                    ground_truth_answer = example.ground_truth_answer
                    finish_reason = example.finish_reason
                    total_time = example.total_time
                    reference_steps = example.reference_steps
                    single = await run_single_agent(problem, model)
                    single_output = single.content
                    single_steps = single.steps
                    single_latency = single.latency_ms
                    single_approx_input_tokens = single.approx_input_tokens
                    single_approx_output_tokens = approx_tokens(single.content)
                    single_approx_total_tokens = single_approx_input_tokens + single_approx_output_tokens

                mas_start_index = len(router.records)
                mas_result = await run_mas_full_path(
                    split=example_split,
                    question_id=question_id,
                    source_index=source_index,
                    problem=problem,
                    model=model,
                    router=router,
                    max_rounds=max_rounds,
                    stall_rounds=stall_rounds,
                    force_final_on_stop=force_final_on_stop,
                )
                mas_records = router.records[mas_start_index:]
                mas_approx_input_tokens = sum(record.approx_input_tokens for record in mas_records)
                mas_approx_output_tokens = sum(record.approx_output_tokens for record in mas_records)
                mas_approx_total_tokens = mas_approx_input_tokens + mas_approx_output_tokens
                prediction = {
                    "split": example_split,
                    "question_id": question_id,
                    "source_index": source_index,
                    "problem": problem,
                    "ground_truth_answer": ground_truth_answer,
                    "finish_reason": finish_reason,
                    "total_time": total_time,
                    "reference_steps": reference_steps,
                    "single_agent_output": single_output,
                    "single_agent_steps": single_steps,
                    "single_agent_latency_ms": single_latency,
                    "single_agent_approx_input_tokens": single_approx_input_tokens,
                    "single_agent_approx_output_tokens": single_approx_output_tokens,
                    "single_agent_approx_total_tokens": single_approx_total_tokens,
                    "mas_output": mas_result.content,
                    "solver_a_steps": mas_result.solver_a_steps,
                    "solver_b_steps": mas_result.solver_b_steps,
                    "planner_messages": mas_result.planner_messages,
                    "judger_messages": mas_result.judger_messages,
                    "termination_reason": mas_result.termination_reason,
                    "rounds_used": mas_result.rounds_used,
                    "mas_approx_input_tokens": mas_approx_input_tokens,
                    "mas_approx_output_tokens": mas_approx_output_tokens,
                    "mas_approx_total_tokens": mas_approx_total_tokens,
                }
                predictions.append(prediction)
                _append_jsonl(run_dir / "predictions.jsonl", prediction)
                write_outputs()
            except Exception as exc:  # noqa: BLE001 - preserve per-example failure in experiment logs.
                if replay_mode:
                    split_value = str(example.get("split", split))
                    question_id_value = int(example.get("question_id", -1))
                    source_index_value = int(example.get("source_index", -1))
                else:
                    split_value = example.split
                    question_id_value = example.question_id
                    source_index_value = example.source_index
                error = {
                    "split": split_value,
                    "question_id": question_id_value,
                    "source_index": source_index_value,
                    "error": repr(exc),
                }
                errors.append(error)
                _append_jsonl(run_dir / "errors.jsonl", error)
                write_outputs()
                if not continue_on_error:
                    raise
    finally:
        await model.close()

    write_outputs()
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase0 MATH500 full-forward MAS experiment.")
    parser.add_argument("--config", default="config/phase0_MATH500.yaml")
    parser.add_argument("--data-path")
    parser.add_argument("--train-size", type=int)
    parser.add_argument("--test-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--split", choices=("train", "test"))
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--start-index", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--request-retries", type=int)
    parser.add_argument("--retry-backoff-seconds", type=float)
    parser.add_argument("--max-rounds", type=int)
    parser.add_argument("--stall-rounds", type=int)
    parser.add_argument("--force-final-on-stop", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--edge-policy")
    parser.add_argument("--replay-from-run")
    parser.add_argument("--checkpoint-candidate-id")
    parser.add_argument("--replay-policy-name")
    parser.add_argument("--output-dir")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        run_dir = asyncio.run(run_phase0_math500(args))
    except Exception as exc:  # noqa: BLE001 - CLI should print a concise actionable error.
        print(f"Phase0 MATH500 failed: {exc}", file=sys.stderr)
        return 1
    print(f"Phase0 MATH500 outputs written to: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
