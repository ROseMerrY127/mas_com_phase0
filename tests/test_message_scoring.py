from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phase0_PRM800K.message_scoring import (
    RUBRIC_PROMPT,
    ModelScore,
    OpenAIMessageScorer,
    ScoreAssessment,
    _token_cost_score,
    load_examples,
    prepare_messages,
    run_scoring,
)


def _message(
    message_id: str,
    *,
    sender: str,
    recipient: str,
    kind: str,
    content: str,
    control: bool = False,
    source_message_id: str | None = None,
) -> dict[str, object]:
    return {
        "run_id": "run",
        "message_id": message_id,
        "split": "train",
        "question_id": 0,
        "source_index": 10,
        "round": 1,
        "sender": sender,
        "recipient": recipient,
        "kind": kind,
        "content": content,
        "source_message_id": source_message_id or f"src_{message_id}",
        "created_by_activation_id": None,
        "control": control,
        "terminal": False,
        "termination_reason": None,
    }


def _example() -> dict[str, object]:
    return {
        "question_id": 0,
        "split": "train",
        "source_index": 10,
        "problem": "Compute 20 + 22.",
        "ground_truth_answer": "42",
        "reference_steps": ["Adding gives 42."],
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _args(messages: Path, splits_dir: Path, output_dir: Path, *, dry_run: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        messages=str(messages),
        splits_dir=str(splits_dir),
        model="gpt-5.6-sol",
        reasoning_effort="high",
        concurrency=2,
        request_retries=2,
        retry_backoff_seconds=0.0,
        output_dir=str(output_dir),
        continue_on_error=False,
        limit=None,
        dry_run=dry_run,
    )


class FakeScorer:
    def __init__(self) -> None:
        self.calls = 0

    async def score(self, prepared: object) -> ModelScore:
        self.calls += 1
        return ModelScore(
            assessment=ScoreAssessment(
                correctness_level=4,
                correctness_rationale="The arithmetic is correct.",
                downstream_value_level=3,
                downstream_value_rationale="The recipient can use the step directly.",
                critical_error=None,
            ),
            response_id=f"resp_{self.calls}",
            response_model="gpt-5.6-sol",
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
        )


class FakeResponses:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def parse(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            parsed: object = {
                "correctness_level": 1,
                "correctness_rationale": "Mostly wrong.",
                "downstream_value_level": 2,
                "downstream_value_rationale": "Invalid because it exceeds correctness.",
                "critical_error": "Wrong main claim.",
            }
        else:
            parsed = ScoreAssessment(
                correctness_level=3,
                correctness_rationale="Core content is correct.",
                downstream_value_level=2,
                downstream_value_rationale="Useful but incomplete.",
                critical_error=None,
            )
        usage = SimpleNamespace(input_tokens=7, output_tokens=4, total_tokens=11)
        return SimpleNamespace(
            output_parsed=parsed,
            usage=usage,
            id="resp_test",
            model="gpt-5.6-sol-2026-07-01",
        )


class FakeClient:
    def __init__(self) -> None:
        self.responses = FakeResponses()
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def test_score_contract_rejects_inconsistent_levels_and_normalizes_in_output() -> None:
    with pytest.raises(ValidationError):
        ScoreAssessment(
            correctness_level=1,
            correctness_rationale="Mostly wrong.",
            downstream_value_level=2,
            downstream_value_rationale="Cannot exceed correctness.",
            critical_error="Wrong method.",
        )

    for level in range(5):
        assert f"- {level}:" in RUBRIC_PROMPT


def test_token_cost_score_is_continuous_and_clipped_at_cutoff() -> None:
    assert _token_cost_score(0, 30.0) == 1.0
    assert _token_cost_score(15, 30.0) == 0.5
    assert _token_cost_score(30, 30.0) == 0.0
    assert _token_cost_score(31, 30.0) == 0.0

    with pytest.raises(ValueError, match="non-negative"):
        _token_cost_score(-1, 30.0)


def test_prepare_messages_uses_only_prior_non_control_recipient_history() -> None:
    rows = [
        _message("m1", sender="Input", recipient="Planner", kind="input", content="Compute 20 + 22."),
        _message(
            "m2",
            sender="Scheduler",
            recipient="SolverA",
            kind="control",
            content='{"stage":"SolverA"}',
            control=True,
        ),
        _message("m3", sender="Planner", recipient="SolverA", kind="plan", content="Add the values."),
        _message("m4", sender="Planner", recipient="SolverB", kind="plan", content="Cross-check."),
        _message("m5", sender="SolverA", recipient="SolverA", kind="solver_step", content="20 + 22 = 42."),
        _message("m6", sender="Judger", recipient="Output", kind="final_output", content="42"),
    ]
    examples = {("train", 0, 10): _example()}
    prepared = prepare_messages(rows, examples)

    assert [item.row["message_id"] for item in prepared] == ["m3", "m4", "m5"]
    assert [row["message_id"] for row in prepared[2].recipient_history] == ["m3"]
    assert "m2" not in prepared[2].user_prompt
    assert "m4" not in prepared[2].user_prompt
    assert "m6" not in prepared[2].user_prompt


def test_prepare_messages_keeps_broadcast_edges_separate_and_detects_duplicates() -> None:
    rows = [
        _message(
            "m1",
            sender="Planner",
            recipient="SolverA",
            kind="plan",
            content="Add directly.",
            source_message_id="src_shared",
        ),
        _message(
            "m2",
            sender="Planner",
            recipient="SolverB",
            kind="plan",
            content="Add directly.",
            source_message_id="src_shared",
        ),
    ]
    examples = {("train", 0, 10): _example()}
    prepared = prepare_messages(rows, examples)

    assert len(prepared) == 2
    assert prepared[0].row["source_message_id"] == prepared[1].row["source_message_id"]
    assert prepared[0].prompt_hash != prepared[1].prompt_hash

    with pytest.raises(ValueError, match="Duplicate message_id"):
        prepare_messages([rows[0], rows[0]], examples)


def test_load_examples_distinguishes_split_and_source_index(tmp_path: Path) -> None:
    splits = tmp_path / "splits"
    train = _example()
    test = {**_example(), "split": "test", "source_index": 11}
    _write_jsonl(splits / "train.jsonl", [train])
    _write_jsonl(splits / "test.jsonl", [test])

    examples = load_examples(splits)

    assert ("train", 0, 10) in examples
    assert ("test", 0, 11) in examples


def test_openai_scorer_retries_schema_inconsistency_and_uses_responses_api() -> None:
    client = FakeClient()
    scorer = OpenAIMessageScorer(
        model="gpt-5.6-sol",
        reasoning_effort="high",
        api_key="test",
        request_retries=1,
        retry_backoff_seconds=0,
        client=client,
    )
    row = _message("m1", sender="Planner", recipient="SolverA", kind="plan", content="Add.")
    prepared = prepare_messages([row], {("train", 0, 10): _example()})[0]

    result = asyncio.run(scorer.score(prepared))

    assert len(client.responses.calls) == 2
    request = client.responses.calls[-1]
    assert request["model"] == "gpt-5.6-sol"
    assert request["reasoning"] == {"effort": "high"}
    assert request["text_format"] is ScoreAssessment
    assert request["store"] is False
    assert result.assessment.downstream_value_level == 2
    assert result.total_tokens == 11


def test_run_scoring_writes_scores_and_resumes_exact_key(tmp_path: Path) -> None:
    messages = tmp_path / "messages.jsonl"
    splits = tmp_path / "splits"
    output = tmp_path / "scores"
    _write_jsonl(splits / "train.jsonl", [_example()])
    _write_jsonl(
        messages,
        [
            _message("m1", sender="Input", recipient="Planner", kind="input", content="Compute 20 + 22."),
            _message("m2", sender="Planner", recipient="SolverA", kind="plan", content="Add directly."),
        ],
    )
    scorer = FakeScorer()
    args = _args(messages, splits, output)

    first = asyncio.run(run_scoring(args, scorer=scorer))
    second = asyncio.run(run_scoring(args, scorer=scorer))

    assert first["completed_new"] == 1
    assert second["completed_new"] == 0
    assert second["completed_existing"] == 1
    assert scorer.calls == 1

    rows = [
        json.loads(line)
        for line in (output / "message_scores.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["correctness_level"] == 4
    assert rows[0]["correctness_score"] == 1.0
    assert rows[0]["downstream_value_level"] == 3
    assert rows[0]["downstream_value_score"] == 0.75
    assert rows[0]["message_approx_tokens"] == 3
    assert rows[0]["token_cost_score"] == pytest.approx(1 / 3)
    assert rows[0]["token_cost_reference_longest_tokens"] == 3
    assert rows[0]["token_cost_reference_multiplier"] == 1.5
    assert rows[0]["token_cost_cutoff_tokens"] == 4.5
    assert rows[0]["rubric_version"] == "v1"


def test_resume_recalibrates_token_cost_without_rescoring_existing_rows(tmp_path: Path) -> None:
    messages = tmp_path / "messages.jsonl"
    splits = tmp_path / "splits"
    output = tmp_path / "scores"
    _write_jsonl(splits / "train.jsonl", [_example()])
    short = _message(
        "m1",
        sender="Planner",
        recipient="SolverA",
        kind="plan",
        content="a" * 40,
    )
    _write_jsonl(messages, [short])
    scorer = FakeScorer()
    args = _args(messages, splits, output)

    asyncio.run(run_scoring(args, scorer=scorer))
    long = _message(
        "m2",
        sender="SolverA",
        recipient="Planner",
        kind="solver_step",
        content="b" * 80,
    )
    _write_jsonl(messages, [short, long])
    summary = asyncio.run(run_scoring(args, scorer=scorer))

    assert scorer.calls == 2
    assert summary["completed_existing"] == 1
    assert summary["completed_new"] == 1
    assert summary["token_cost_rows_updated"] == 1
    rows = [
        json.loads(line)
        for line in (output / "message_scores.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    latest_short = [row for row in rows if row["message_id"] == "m1"][-1]
    assert latest_short["message_approx_tokens"] == 10
    assert latest_short["token_cost_reference_longest_tokens"] == 20
    assert latest_short["token_cost_cutoff_tokens"] == 30.0
    assert latest_short["token_cost_score"] == pytest.approx(2 / 3)


def test_dry_run_needs_no_api_key_and_reports_dynamic_scope(tmp_path: Path) -> None:
    messages = tmp_path / "messages.jsonl"
    splits = tmp_path / "splits"
    _write_jsonl(splits / "train.jsonl", [_example()])
    _write_jsonl(
        messages,
        [
            _message("m1", sender="Planner", recipient="SolverA", kind="plan", content="Plan."),
            _message("m2", sender="SolverA", recipient="Planner", kind="solver_step", content="Step."),
            _message("m3", sender="Judger", recipient="Output", kind="final_output", content="42"),
        ],
    )

    summary = asyncio.run(run_scoring(_args(messages, splits, tmp_path / "unused", dry_run=True)))

    assert summary["total_input_messages"] == 3
    assert summary["eligible_messages"] == 2
    assert summary["eligible_by_kind"] == {"plan": 1, "solver_step": 1}
    assert summary["token_cost_reference_longest_tokens"] == 1
    assert summary["token_cost_cutoff_tokens"] == 1.5
    assert not (tmp_path / "unused").exists()
