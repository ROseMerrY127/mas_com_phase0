from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
from typing import Any

from dotenv import load_dotenv
import yaml
from tqdm import tqdm

from .agents import build_model, run_mas_identity, run_single_agent
from .data import load_gsm8k
from .evaluation import approx_tokens, extract_gold_answer, extract_last_number, is_correct
from .router import IdentityRouter


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Phase0 config must be a YAML mapping")
    return data


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        handle.flush()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


async def run_phase0(args: argparse.Namespace) -> Path:
    load_dotenv()
    config = _load_config(Path(args.config))

    data_path = Path(args.data_path or config.get("data_path", "gsm8K/test-00000-of-00001.parquet"))
    config_sample_size = config.get("sample_size", 50)
    sample_size = args.sample_size if args.sample_size is not None else None if config_sample_size is None else int(config_sample_size)
    seed = int(args.seed if args.seed is not None else config.get("seed", 0))
    model_name = str(args.model or config.get("model", "gpt-4o-mini"))
    temperature = float(config.get("temperature", 0.2) if args.temperature is None else args.temperature)
    max_tokens = int(config.get("max_tokens", 1024) if args.max_tokens is None else args.max_tokens)
    continue_on_error = bool(args.continue_on_error or config.get("continue_on_error", False))
    request_retries = int(config.get("request_retries", 3) if args.request_retries is None else args.request_retries)
    retry_backoff_seconds = float(
        config.get("retry_backoff_seconds", 2.0) if args.retry_backoff_seconds is None else args.retry_backoff_seconds
    )

    random.seed(seed)
    examples = load_gsm8k(data_path, sample_size=sample_size)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model = build_model(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        request_retries=request_retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )

    output_root = Path(args.output_dir or config.get("output_dir", "runs"))
    run_dir = output_root / f"phase0_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    router = IdentityRouter(run_id=run_dir.name)
    predictions: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    def write_outputs() -> None:
        total = len(predictions)
        single_correct = sum(1 for row in predictions if row["single_agent_correct"])
        mas_correct = sum(1 for row in predictions if row["mas_correct"])
        traces = router.as_dicts()
        approx_token_total = sum(
            row.get("approx_input_tokens", 0) + row.get("approx_output_tokens", 0) for row in traces
        )
        approx_token_total += sum(approx_tokens(row.get("single_agent_output", "")) for row in predictions)

        summary = {
            "num_examples": total,
            "requested_sample_size": sample_size,
            "single_agent_accuracy": single_correct / total if total else 0.0,
            "mas_identity_accuracy": mas_correct / total if total else 0.0,
            "mas_beats_single_agent": mas_correct > single_correct,
            "approx_tokens": approx_token_total,
            "model": model_name,
            "data_path": str(data_path),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "errors": errors,
        }

        _write_jsonl(run_dir / "traces.jsonl", traces)
        _write_json(run_dir / "summary.json", summary)

    try:
        for example in tqdm(examples, total=len(examples), desc="Phase0", unit="example"):
            gold = extract_gold_answer(example.answer)
            try:
                single = await run_single_agent(example.question, model)
                single_pred = extract_last_number(single.content)
                single_ok = is_correct(single_pred, gold)

                mas_output = await run_mas_identity(
                    question_id=example.question_id,
                    question=example.question,
                    model=model,
                    router=router,
                )
                mas_pred = extract_last_number(mas_output)
                mas_ok = is_correct(mas_pred, gold)
                router.backfill_rewards(question_id=example.question_id, final_correct=mas_ok)

                prediction = {
                    "question_id": example.question_id,
                    "question": example.question,
                    "gold_answer": str(gold) if gold is not None else None,
                    "single_agent_output": single.content,
                    "single_agent_prediction": str(single_pred) if single_pred is not None else None,
                    "single_agent_correct": single_ok,
                    "mas_output": mas_output,
                    "mas_prediction": str(mas_pred) if mas_pred is not None else None,
                    "mas_correct": mas_ok,
                }
                predictions.append(prediction)
                _append_jsonl(run_dir / "predictions.jsonl", prediction)
                write_outputs()
            except Exception as exc:  # noqa: BLE001 - preserve per-example failure in experiment logs.
                error = {"question_id": example.question_id, "error": repr(exc)}
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
    parser = argparse.ArgumentParser(description="Run Phase0 identity-router GSM8K experiment.")
    parser.add_argument("--config", default="config/phase0.yaml")
    parser.add_argument("--data-path")
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--model")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--request-retries", type=int)
    parser.add_argument("--retry-backoff-seconds", type=float)
    parser.add_argument("--output-dir")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        run_dir = asyncio.run(run_phase0(args))
    except Exception as exc:  # noqa: BLE001 - CLI should print a concise actionable error.
        print(f"Phase0 failed: {exc}", file=sys.stderr)
        return 1
    print(f"Phase0 outputs written to: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

