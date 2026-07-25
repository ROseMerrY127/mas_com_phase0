from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any


_ID_RE = re.compile(r"^[a-z]+(\d+)$")
_QUESTION_LOG_FILES = (
    "messages.jsonl",
    "activations.jsonl",
    "edge_candidates.jsonl",
    "traces.jsonl",
)


def _read_jsonl(path: Path, *, required: bool = True) -> list[dict[str, Any]]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Scoped replay input not found: {path}")
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _matches_question(row: dict[str, Any], *, split: str, question_id: int, source_index: int) -> bool:
    return (
        str(row.get("split")) == split
        and int(row.get("question_id", -1)) == question_id
        and int(row.get("source_index", -1)) == source_index
    )


def _counter_offset(rows: list[dict[str, Any]], field: str, prefix: str) -> int:
    values: list[int] = []
    for row in rows:
        value = str(row.get(field, ""))
        match = _ID_RE.fullmatch(value)
        if match and value.startswith(prefix):
            values.append(int(match.group(1)))
    if not values:
        return 0
    return min(values) - 1


def scoped_parent_name(parent_run: Path, *, split: str, question_id: int, source_index: int) -> str:
    payload = f"{parent_run.resolve()}|{split}|{question_id}|{source_index}"
    suffix = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"q{question_id}_src{source_index}_{suffix}"


def build_scoped_parent(
    parent_run: str | Path,
    output_dir: str | Path,
    *,
    split: str,
    question_id: int,
    source_index: int,
) -> dict[str, Any]:
    """Create a one-question replay parent while preserving the original event IDs."""
    parent = Path(parent_run)
    destination = Path(output_dir)
    metadata_path = destination / "scoped_metadata.json"
    if metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    predictions = [
        row
        for row in _read_jsonl(parent / "predictions.jsonl")
        if _matches_question(row, split=split, question_id=question_id, source_index=source_index)
    ]
    if len(predictions) != 1:
        raise ValueError(
            f"Expected one baseline prediction for {split}/{question_id}/{source_index}, found {len(predictions)}"
        )

    scoped_logs: dict[str, list[dict[str, Any]]] = {}
    for filename in _QUESTION_LOG_FILES:
        scoped_logs[filename] = [
            row
            for row in _read_jsonl(parent / filename, required=False)
            if _matches_question(row, split=split, question_id=question_id, source_index=source_index)
        ]

    candidate_ids = {str(row["candidate_id"]) for row in scoped_logs["edge_candidates.jsonl"]}
    decisions = [
        row
        for row in _read_jsonl(parent / "edge_decisions.jsonl")
        if str(row.get("candidate_id")) in candidate_ids
    ]
    rl_samples = [
        row
        for row in _read_jsonl(parent / "rl_edge_samples.jsonl", required=False)
        if str(row.get("candidate_id")) in candidate_ids
    ]
    if not candidate_ids or not decisions:
        raise ValueError(f"Question {split}/{question_id}/{source_index} has no replayable edge events")

    offsets = {
        "m": _counter_offset(scoped_logs["messages.jsonl"], "message_id", "m"),
        "a": _counter_offset(scoped_logs["activations.jsonl"], "activation_id", "a"),
        "c": _counter_offset(scoped_logs["edge_candidates.jsonl"], "candidate_id", "c"),
        "d": _counter_offset(decisions, "decision_id", "d"),
        "s": _counter_offset(rl_samples, "sample_id", "s"),
        "src": 0,
    }

    summary_path = parent / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    summary.update(
        {
            "num_examples": 1,
            "planned_examples": 1,
            "requested_sample_size": 1,
            "scoped_replay_parent": True,
            "original_parent_run": str(parent),
            "scoped_question": {
                "split": split,
                "question_id": question_id,
                "source_index": source_index,
            },
        }
    )
    metadata = {
        "scoped_parent_dir": str(destination),
        "original_parent_run": str(parent),
        "split": split,
        "question_id": question_id,
        "source_index": source_index,
        "counter_offsets": offsets,
        "num_candidates": len(candidate_ids),
        "num_activations": len(scoped_logs["activations.jsonl"]),
    }

    destination.mkdir(parents=True, exist_ok=False)
    _write_jsonl(destination / "predictions.jsonl", predictions)
    for filename, rows in scoped_logs.items():
        _write_jsonl(destination / filename, rows)
    _write_jsonl(destination / "edge_decisions.jsonl", decisions)
    _write_jsonl(destination / "rl_edge_samples.jsonl", rl_samples)
    _write_json(destination / "summary.json", summary)
    _write_json(metadata_path, metadata)
    return metadata
