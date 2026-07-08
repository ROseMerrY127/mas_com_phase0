from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import json

import pandas as pd


@dataclass(frozen=True)
class GSM8KExample:
    question_id: int
    question: str
    answer: str


def load_gsm8k(path: str | Path, sample_size: int | None = None) -> list[GSM8KExample]:
    """Load local GSM8K data from parquet or JSONL."""
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"GSM8K data file not found: {data_path}")

    suffix = data_path.suffix.lower()
    if suffix == ".parquet":
        frame = pd.read_parquet(data_path)
        rows: Iterable[dict] = frame.to_dict(orient="records")
    elif suffix in {".jsonl", ".json"}:
        if suffix == ".jsonl":
            rows = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            raw = json.loads(data_path.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                raise ValueError("JSON GSM8K file must contain a list of records")
            rows = raw
    else:
        raise ValueError(f"Unsupported GSM8K format: {suffix}. Use parquet, jsonl, or json.")

    examples: list[GSM8KExample] = []
    for idx, row in enumerate(rows):
        if "question" not in row or "answer" not in row:
            raise ValueError("GSM8K records must contain 'question' and 'answer' fields")
        examples.append(GSM8KExample(question_id=idx, question=str(row["question"]), answer=str(row["answer"])))
        if sample_size is not None and len(examples) >= sample_size:
            break
    return examples
