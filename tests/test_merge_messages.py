from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phase0_MATH500.merge_messages import discover_message_files, merge_dataset
from phase0_MATH500.message_scoring import load_examples, prepare_messages


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _example(question_id: int, source_index: int, problem: str) -> dict[str, object]:
    return {
        "question_id": question_id,
        "split": "train",
        "source_index": source_index,
        "problem": problem,
        "ground_truth_answer": "42",
        "reference_steps": ["Compute directly."],
    }


def _message(question_id: int, source_index: int, content: str) -> dict[str, object]:
    return {
        "run_id": "same_run_id",
        "message_id": "m000001",
        "split": "train",
        "question_id": question_id,
        "source_index": source_index,
        "round": 1,
        "sender": "Planner",
        "recipient": "SolverA",
        "kind": "plan",
        "content": content,
        "source_message_id": "src_a000001",
        "created_by_activation_id": "a000001",
        "control": False,
        "terminal": False,
        "termination_reason": None,
    }


def _source(
    root: Path,
    shard: str,
    message: dict[str, object],
    train_rows: list[dict[str, object]],
) -> None:
    run_dir = root / shard / "run"
    _write_jsonl(run_dir / "messages.jsonl", [message])
    _write_jsonl(run_dir / "splits" / "train.jsonl", train_rows)
    _write_jsonl(run_dir / "splits" / "test.jsonl", [])


def test_merge_dataset_namespaces_ids_and_deduplicates_splits(tmp_path: Path) -> None:
    root = tmp_path / "parallel"
    output = root / "merged"
    examples = [
        _example(0, 10, "Compute 40 + 2."),
        _example(1, 11, "Compute 41 + 1."),
    ]
    _source(root, "shard_0000", _message(0, 10, "Add 40 and 2."), examples)
    _source(root, "shard_0001", _message(1, 11, "Add 41 and 1."), examples)

    manifest = merge_dataset(root, output)

    assert manifest["source_message_file_count"] == 2
    assert manifest["message_count"] == 2
    assert manifest["merged_train_example_count"] == 2

    messages = [
        json.loads(line)
        for line in (output / "messages.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len({row["message_id"] for row in messages}) == 2
    assert {row["original_message_id"] for row in messages} == {"m000001"}
    assert all(row["message_id"].endswith("::m000001") for row in messages)
    assert all(row["source_message_id"].endswith("::src_a000001") for row in messages)
    assert all(row["source_run_key"] for row in messages)
    assert all(row["source_message_file"].endswith("messages.jsonl") for row in messages)

    prepared = prepare_messages(messages, load_examples(output / "splits"))
    assert len(prepared) == 2


def test_merge_rerun_excludes_its_previous_output(tmp_path: Path) -> None:
    root = tmp_path / "parallel"
    examples = [_example(0, 10, "Compute 40 + 2.")]
    _source(root, "shard_0000", _message(0, 10, "Add."), examples)

    first = merge_dataset(root)
    second = merge_dataset(root)

    assert first["message_count"] == 1
    assert second["message_count"] == 1
    assert len(discover_message_files(root, root / "merged")) == 1


def test_merge_rejects_conflicting_duplicate_split_examples(tmp_path: Path) -> None:
    root = tmp_path / "parallel"
    first = [_example(0, 10, "Compute 40 + 2.")]
    second = [_example(0, 10, "A conflicting problem.")]
    _source(root, "shard_0000", _message(0, 10, "Add."), first)
    _source(root, "shard_0001", _message(0, 10, "Add again."), second)

    with pytest.raises(ValueError, match="Conflicting duplicate split example"):
        merge_dataset(root)


def test_discovery_requires_source_messages(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover_message_files(tmp_path, tmp_path / "merged")
