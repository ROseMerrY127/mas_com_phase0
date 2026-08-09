from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
from typing import Any

from .edge_pruning import PRUNABLE_MESSAGE_KINDS


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _example_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return str(row["split"]), int(row["question_id"]), int(row["source_index"])


def collect_parallel_baseline(master_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_files = sorted(
        {
            *master_dir.glob("shard_*/phase0_MATH500_*/edge_candidates.jsonl"),
            # Historical MATH500 runs used the incorrect PRM800K experiment prefix.
            *master_dir.glob("shard_*/phase0_PRM800K_*/edge_candidates.jsonl"),
        }
    )
    if not candidate_files:
        raise FileNotFoundError(f"No shard edge_candidates.jsonl files found under: {master_dir}")

    candidates: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    for candidate_path in candidate_files:
        parent_run = candidate_path.parent
        predictions = _read_jsonl(parent_run / "predictions.jsonl")
        successful_keys = {_example_key(row) for row in predictions}
        for prediction in predictions:
            examples.append(
                {
                    "parent_run": str(parent_run),
                    "split": prediction["split"],
                    "question_id": prediction["question_id"],
                    "source_index": prediction["source_index"],
                }
            )
        for row in _read_jsonl(candidate_path):
            if _example_key(row) not in successful_keys:
                continue
            if str(row.get("kind")) not in PRUNABLE_MESSAGE_KINDS:
                continue
            if row.get("sender") == row.get("recipient"):
                continue
            candidates.append({"parent_run": str(parent_run), **row})
    return candidates, examples


def _round_quotas(sample_size: int, availability: Counter[int]) -> dict[int, int]:
    weights = {1: 0.586, 2: 0.300, 3: 0.100, 4: 0.014}
    quotas = {round_id: min(availability[round_id], round(sample_size * weight)) for round_id, weight in weights.items()}
    remaining = sample_size - sum(quotas.values())
    while remaining > 0:
        progressed = False
        for round_id in (1, 2, 3, 4):
            if quotas[round_id] < availability[round_id]:
                quotas[round_id] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise ValueError("Requested more LOMO samples than available candidates")
    return quotas


def _balanced_sample_for_round(rows: list[dict[str, Any]], quota: int, rng: random.Random) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["sender"]), str(row["recipient"]))].append(row)
    for group in groups.values():
        rng.shuffle(group)

    selected: list[dict[str, Any]] = []
    keys = sorted(groups)
    while len(selected) < quota:
        progressed = False
        for key in keys:
            if groups[key]:
                selected.append(groups[key].pop())
                progressed = True
                if len(selected) == quota:
                    break
        if not progressed:
            raise ValueError("Round quota exceeds available candidates")
    return selected


def stratified_lomo_sample(
    candidates: list[dict[str, Any]], *, sample_size: int, seed: int
) -> tuple[list[dict[str, Any]], dict[int, int]]:
    if sample_size < 1 or sample_size > len(candidates):
        raise ValueError(f"lomo_sample_size must be between 1 and {len(candidates)}")
    rng = random.Random(seed)
    availability = Counter(int(row["round"]) for row in candidates)
    quotas = _round_quotas(sample_size, availability)
    selected: list[dict[str, Any]] = []
    for round_id in (1, 2, 3, 4):
        round_rows = [row for row in candidates if int(row["round"]) == round_id]
        selected.extend(_balanced_sample_for_round(round_rows, quotas[round_id], rng))
    rng.shuffle(selected)
    return selected, quotas


def _distribution(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted((str(key), value) for key, value in Counter(row[field] for row in rows).items()))


def build_experiment_plan(args: argparse.Namespace) -> Path:
    master_dir = Path(args.master_dir)
    candidates, examples = collect_parallel_baseline(master_dir)
    lomo_rows, round_quotas = stratified_lomo_sample(candidates, sample_size=args.lomo_sample_size, seed=args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_jsonl(output_dir / "lomo_candidates.jsonl", lomo_rows)
    summary = {
        "master_dir": str(master_dir),
        "seed": args.seed,
        "available_prunable_candidates": len(candidates),
        "successful_baseline_examples": len(examples),
        "lomo_sample_size": len(lomo_rows),
        "lomo_round_quotas": {str(key): value for key, value in sorted(round_quotas.items())},
        "lomo_kind_distribution": _distribution(lomo_rows, "kind"),
        "lomo_round_distribution": _distribution(lomo_rows, "round"),
        "lomo_sender_distribution": _distribution(lomo_rows, "sender"),
        "lomo_recipient_distribution": _distribution(lomo_rows, "recipient"),
        "total_planned_counterfactual_trajectories": len(lomo_rows),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(output_dir / "plan_summary.json", summary)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a stratified LOMO plan without model calls.")
    parser.add_argument("--master-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lomo-sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        output_dir = build_experiment_plan(args)
    except Exception as exc:  # noqa: BLE001 - CLI reports plan construction failures succinctly.
        print(f"LOMO plan failed: {exc}", file=sys.stderr)
        return 1
    print(f"LOMO plan written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
