from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import random
from typing import Any, Iterable, Literal


SplitName = Literal["train", "test"]


@dataclass(frozen=True)
class MathExample:
    question_id: int
    split: SplitName
    source_index: int
    problem: str
    ground_truth_answer: str
    finish_reason: str | None = None
    total_time: int | float | None = None
    reference_steps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SplitResult:
    train: list[MathExample]
    test: list[MathExample]
    scanned_records: int
    valid_records: int
    train_size: int
    test_size: int
    seed: int


def _extract_reference_steps(row: dict[str, Any]) -> list[str]:
    label = row.get("label")
    if not isinstance(label, dict):
        return []
    raw_steps = label.get("steps")
    if not isinstance(raw_steps, list):
        return []

    reference_steps: list[str] = []
    for raw_step in raw_steps:
        if not isinstance(raw_step, dict):
            continue
        completions = raw_step.get("completions")
        chosen_completion = raw_step.get("chosen_completion")
        if not isinstance(completions, list) or not isinstance(chosen_completion, int):
            continue
        if chosen_completion < 0 or chosen_completion >= len(completions):
            continue
        completion = completions[chosen_completion]
        if not isinstance(completion, dict):
            continue
        text = completion.get("text")
        if isinstance(text, str) and text.strip():
            reference_steps.append(text.strip())
    return reference_steps


def _record_from_prm800k(row: dict[str, Any], source_index: int) -> dict[str, Any] | None:
    question = row.get("question")
    if not isinstance(question, dict):
        return None
    problem = question.get("problem")
    answer = question.get("ground_truth_answer")
    if problem is None or answer is None:
        return None
    return {
        "source_index": source_index,
        "problem": str(problem),
        "ground_truth_answer": str(answer),
        "finish_reason": row.get("finish_reason"),
        "total_time": row.get("total_time"),
        "reference_steps": _extract_reference_steps(row),
    }


def _record_from_math500(row: dict[str, Any], source_index: int) -> dict[str, Any] | None:
    problem = row.get("problem")
    answer = row.get("answer")
    if problem is None or answer is None:
        return None
    solution = row.get("solution")
    reference_steps = [str(solution).strip()] if isinstance(solution, str) and solution.strip() else []
    return {
        "source_index": source_index,
        "problem": str(problem),
        "ground_truth_answer": str(answer),
        "finish_reason": "solution" if solution else None,
        "total_time": None,
        "reference_steps": reference_steps,
    }


def _iter_valid_records(path: str | Path) -> Iterable[dict[str, Any]]:
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if data_path.suffix.lower() != ".jsonl":
        raise ValueError(f"Unsupported data format: {data_path.suffix}. Use jsonl.")

    with data_path.open("r", encoding="utf-8") as handle:
        for source_index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            record = _record_from_prm800k(row, source_index) or _record_from_math500(row, source_index)
            if record is not None:
                yield record


def build_splits(
    path: str | Path,
    *,
    train_size: int,
    test_size: int,
    seed: int,
) -> SplitResult:
    """Randomly split valid JSONL math rows without inferring or filtering difficulty."""
    if train_size < 0 or test_size < 0:
        raise ValueError("train_size and test_size must be non-negative")

    records = list(_iter_valid_records(path))
    total_needed = train_size + test_size
    if len(records) < total_needed:
        raise ValueError(
            f"Not enough valid data records: need {total_needed}, found {len(records)}. "
            "Lower train_size/test_size or provide a larger data file."
        )

    rng = random.Random(seed)
    selected = rng.sample(records, total_needed)
    train_rows = selected[:train_size]
    test_rows = selected[train_size:]

    train = [
        MathExample(
            question_id=idx,
            split="train",
            source_index=int(row["source_index"]),
            problem=str(row["problem"]),
            ground_truth_answer=str(row["ground_truth_answer"]),
            finish_reason=row.get("finish_reason"),
            total_time=row.get("total_time"),
            reference_steps=list(row.get("reference_steps", [])),
        )
        for idx, row in enumerate(train_rows)
    ]
    test = [
        MathExample(
            question_id=idx,
            split="test",
            source_index=int(row["source_index"]),
            problem=str(row["problem"]),
            ground_truth_answer=str(row["ground_truth_answer"]),
            finish_reason=row.get("finish_reason"),
            total_time=row.get("total_time"),
            reference_steps=list(row.get("reference_steps", [])),
        )
        for idx, row in enumerate(test_rows)
    ]
    return SplitResult(
        train=train,
        test=test,
        scanned_records=len(records),
        valid_records=len(records),
        train_size=train_size,
        test_size=test_size,
        seed=seed,
    )


def write_split_files(split_result: SplitResult, output_dir: str | Path) -> tuple[Path, Path]:
    split_dir = Path(output_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    train_path = split_dir / "train.jsonl"
    test_path = split_dir / "test.jsonl"
    _write_examples(train_path, split_result.train)
    _write_examples(test_path, split_result.test)
    return train_path, test_path


def _write_examples(path: Path, examples: list[MathExample]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_dict(), ensure_ascii=False, default=str) + "\n")


def select_split(
    split_result: SplitResult,
    split: SplitName,
    sample_size: int | None = None,
    start_index: int = 0,
) -> list[MathExample]:
    examples = split_result.train if split == "train" else split_result.test
    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    if sample_size is None:
        return list(examples[start_index:])
    if sample_size < 0:
        raise ValueError("sample_size must be non-negative")
    return list(examples[start_index : start_index + sample_size])
