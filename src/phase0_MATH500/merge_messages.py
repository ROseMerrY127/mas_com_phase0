from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_RECORDS_ROOT = REPOSITORY_ROOT / "通信记录"
_DEFAULT_MATH500_INPUT = _RECORDS_ROOT / "phase0_MATH500_parallel_20260709T151013Z"
_LEGACY_MATH500_INPUT = _RECORDS_ROOT / "phase0_PRM800K_parallel_20260709T151013Z"
DEFAULT_INPUT_ROOT = _DEFAULT_MATH500_INPUT if _DEFAULT_MATH500_INPUT.exists() else _LEGACY_MATH500_INPUT
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_ROOT / "merged"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}: {exc}") from exc
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


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def discover_message_files(input_root: str | Path, output_dir: str | Path) -> list[Path]:
    root = Path(input_root).resolve()
    output = Path(output_dir).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Input root is not a directory: {root}")
    files = [
        path
        for path in root.rglob("messages.jsonl")
        if path.is_file() and not _is_within(path, output)
    ]
    files.sort(key=lambda path: path.relative_to(root).as_posix())
    if not files:
        raise FileNotFoundError(f"No messages.jsonl files found below: {root}")
    return files


def _source_key(path: Path, input_root: Path) -> str:
    relative_parent = path.parent.resolve().relative_to(input_root.resolve())
    key = relative_parent.as_posix()
    if not key or key == ".":
        key = path.parent.name
    return key


def _namespaced(value: Any, source_key: str) -> str | None:
    if value is None:
        return None
    return f"{source_key}::{value}"


def _merge_splits(
    message_files: list[Path],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    merged: dict[str, dict[tuple[str, int, int], dict[str, Any]]] = {
        "train": {},
        "test": {},
    }
    source_counts = {"train": 0, "test": 0}

    for message_path in message_files:
        split_dir = message_path.parent / "splits"
        for split in ("train", "test"):
            split_path = split_dir / f"{split}.jsonl"
            if not split_path.exists():
                raise FileNotFoundError(
                    f"Missing {split}.jsonl next to source messages file: {message_path}"
                )
            source_counts[split] += 1
            for row in _read_jsonl(split_path):
                try:
                    key = (
                        str(row.get("split", split)),
                        int(row["question_id"]),
                        int(row["source_index"]),
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid split identity in {split_path}: {row}") from exc
                existing = merged[split].get(key)
                if existing is not None and existing != row:
                    raise ValueError(
                        f"Conflicting duplicate split example {key} in {split_path}"
                    )
                merged[split][key] = row

    ordered = {
        split: [rows[key] for key in sorted(rows)]
        for split, rows in merged.items()
    }
    return ordered, source_counts


def merge_dataset(
    input_root: str | Path = DEFAULT_INPUT_ROOT,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(input_root).resolve()
    output = Path(output_dir).resolve() if output_dir else root / "merged"
    message_files = discover_message_files(root, output)

    merged_messages: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    kind_counts: Counter[str] = Counter()
    question_keys: set[tuple[str, int, int]] = set()
    source_summaries: list[dict[str, Any]] = []

    for message_path in message_files:
        source_key = _source_key(message_path, root)
        relative_file = message_path.relative_to(root).as_posix()
        source_rows = _read_jsonl(message_path)
        source_summaries.append(
            {
                "source_run_key": source_key,
                "messages_file": relative_file,
                "message_count": len(source_rows),
            }
        )
        for row in source_rows:
            original_message_id = row.get("message_id")
            if not isinstance(original_message_id, str) or not original_message_id:
                raise ValueError(f"Missing message_id in {message_path}: {row}")
            global_message_id = _namespaced(original_message_id, source_key)
            if global_message_id in seen_ids:
                raise ValueError(f"Duplicate global message ID: {global_message_id}")
            seen_ids.add(global_message_id)

            merged_row = dict(row)
            merged_row["original_message_id"] = original_message_id
            merged_row["original_source_message_id"] = row.get("source_message_id")
            merged_row["original_created_by_activation_id"] = row.get(
                "created_by_activation_id"
            )
            merged_row["message_id"] = global_message_id
            merged_row["source_message_id"] = _namespaced(
                row.get("source_message_id"), source_key
            )
            merged_row["created_by_activation_id"] = _namespaced(
                row.get("created_by_activation_id"), source_key
            )
            merged_row["source_run_key"] = source_key
            merged_row["source_message_file"] = relative_file
            merged_messages.append(merged_row)

            kind_counts[str(row.get("kind"))] += 1
            question_keys.add(
                (
                    str(row.get("split")),
                    int(row.get("question_id")),
                    int(row.get("source_index")),
                )
            )

    merged_splits, split_source_counts = _merge_splits(message_files)
    output.mkdir(parents=True, exist_ok=True)
    split_output = output / "splits"
    split_output.mkdir(parents=True, exist_ok=True)

    _write_jsonl_atomic(output / "messages.jsonl", merged_messages)
    _write_jsonl_atomic(split_output / "train.jsonl", merged_splits["train"])
    _write_jsonl_atomic(split_output / "test.jsonl", merged_splits["test"])

    manifest = {
        "input_root": str(root),
        "output_dir": str(output),
        "created_at": _utc_now(),
        "source_message_file_count": len(message_files),
        "source_split_file_count": split_source_counts,
        "message_count": len(merged_messages),
        "message_kind_counts": dict(kind_counts),
        "unique_question_count": len(question_keys),
        "merged_train_example_count": len(merged_splits["train"]),
        "merged_test_example_count": len(merged_splits["test"]),
        "id_policy": (
            "message_id, source_message_id, and created_by_activation_id are "
            "prefixed with source_run_key; original values are retained."
        ),
        "sources": source_summaries,
    }
    _write_json_atomic(output / "manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively merge MATH500 messages.jsonl files and deduplicate their "
            "train/test split files into one scoring-ready directory."
        )
    )
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument(
        "--output-dir",
        help="Defaults to <input-root>/merged; existing merged outputs are atomically replaced.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        manifest = merge_dataset(args.input_root, args.output_dir)
    except Exception as exc:  # noqa: BLE001 - CLI emits one actionable error.
        print(f"Message merge failed: {exc}")
        return 1

    print(f"Merged dataset written to: {manifest['output_dir']}")
    print(
        f"Sources={manifest['source_message_file_count']} "
        f"messages={manifest['message_count']} "
        f"questions={manifest['unique_question_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
