from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phase0_MATH500.agents import LLMResponse, run_mas_full_path
from phase0_MATH500.edge_pruning import DropCandidateOncePolicy, StableRandomDropPolicy
from phase0_MATH500.router import FullForwardRouter, ReplayController
from phase0_MATH500.run_pruning import _resolve_random_drop_stage
from phase0_MATH500.scoped_replay import build_scoped_parent


class FinalMockModel:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def complete(self, *, system: str, user: str, source: str) -> LLMResponse:
        self.calls.append(source)
        if source.startswith("Planner"):
            return LLMResponse(content="Plan: solve directly.", latency_ms=1)
        if source.startswith("SolverA"):
            return LLMResponse(content="40 + 2 = 42", latency_ms=2)
        if source.startswith("SolverB"):
            return LLMResponse(content="The result is 42.", latency_ms=3)
        if source.startswith("Judger"):
            return LLMResponse(content="FINAL:\n# Answer\n\n42", latency_ms=4)
        raise AssertionError(f"Unexpected source: {source}")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


async def _build_two_question_parent(parent: Path) -> tuple[FullForwardRouter, str]:
    router = FullForwardRouter(run_id="parent")
    predictions = []
    for question_id in (0, 1):
        result = await run_mas_full_path(
            split="train",
            question_id=question_id,
            source_index=100 + question_id,
            problem="What is 40 + 2?",
            model=FinalMockModel(),
            router=router,
            max_rounds=1,
        )
        predictions.append(
            {
                "split": "train",
                "question_id": question_id,
                "source_index": 100 + question_id,
                "problem": "What is 40 + 2?",
                "ground_truth_answer": "42",
                "mas_output": result.content,
            }
        )
    parent.mkdir()
    _write_jsonl(parent / "predictions.jsonl", predictions)
    _write_jsonl(parent / "messages.jsonl", router.messages_as_dicts())
    _write_jsonl(parent / "activations.jsonl", router.activations_as_dicts())
    _write_jsonl(parent / "edge_candidates.jsonl", router.edge_candidates_as_dicts())
    _write_jsonl(parent / "edge_decisions.jsonl", router.edge_decisions_as_dicts())
    _write_jsonl(parent / "stage_actions.jsonl", router.stage_actions_as_dicts())
    _write_jsonl(parent / "rl_edge_samples.jsonl", router.rl_edge_samples_as_dicts())
    _write_jsonl(parent / "traces.jsonl", router.as_dicts())
    (parent / "summary.json").write_text(
        json.dumps({"split": "train", "num_examples": 2, "edge_policy": "identity"}),
        encoding="utf-8",
    )
    target = next(
        row.candidate_id
        for row in router.edge_candidates
        if row.question_id == 1 and row.sender == "Planner" and row.recipient == "SolverA"
    )
    return router, target


def test_scoped_replay_only_contains_and_reexecutes_target_question(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    parent_router, target = asyncio.run(_build_two_question_parent(parent))
    resolved_stage, resolved_candidate = _resolve_random_drop_stage(
        replay_from_run=parent,
        checkpoint_candidate_id=target,
        include_self_edges=False,
        min_dropped_edges=2,
    )
    assert resolved_candidate["candidate_id"] == target
    scoped = tmp_path / "scoped"
    metadata = build_scoped_parent(parent, scoped, split="train", question_id=1, source_index=101)

    predictions = [json.loads(line) for line in (scoped / "predictions.jsonl").read_text().splitlines()]
    assert len(predictions) == 1
    assert predictions[0]["question_id"] == 1
    assert metadata["counter_offsets"]["c"] > 0

    replay = ReplayController(scoped, target)
    child = FullForwardRouter(
        run_id="child",
        edge_policy=DropCandidateOncePolicy(target),
        replay_controller=replay,
    )
    child._counters.update(metadata["counter_offsets"])
    live_model = FinalMockModel()
    asyncio.run(
        run_mas_full_path(
            split="train",
            question_id=1,
            source_index=101,
            problem="What is 40 + 2?",
            model=live_model,
            router=child,
            max_rounds=1,
        )
    )

    assert "Planner_round_1" not in live_model.calls
    assert "SolverA_round_1" in live_model.calls
    assert {row.question_id for row in child.edge_candidates} == {1}
    dropped = [row for row in child.edge_decisions if row.dropped]
    assert [row.candidate_id for row in dropped] == [target]
    lomo_action = next(row for row in child.stage_actions if target in row.candidate_ids)
    assert lomo_action.action_mask.count("1") == 1
    assert lomo_action.dropped_candidate_ids == [target]

    target_stage_action_id = next(
        row.stage_action_id for row in parent_router.edge_candidates if row.candidate_id == target
    )
    assert target_stage_action_id is not None
    assert resolved_stage == target_stage_action_id
    random_replay = ReplayController(scoped, target)
    random_child = FullForwardRouter(
        run_id="random-child",
        edge_policy=StableRandomDropPolicy(
            seed=7,
            target_stage_action_id=target_stage_action_id,
        ),
        replay_controller=random_replay,
    )
    random_child._counters.update(metadata["counter_offsets"])
    asyncio.run(
        run_mas_full_path(
            split="train",
            question_id=1,
            source_index=101,
            problem="What is 40 + 2?",
            model=FinalMockModel(),
            router=random_child,
            max_rounds=1,
        )
    )

    random_actions = [row for row in random_child.stage_actions if "1" in row.action_mask]
    assert [row.stage_action_id for row in random_actions] == [target_stage_action_id]
    assert random_actions[0].action_mask.count("1") >= 2
    assert all(
        not decision.dropped or decision.stage_action_id == target_stage_action_id
        for decision in random_child.edge_decisions
    )


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        test_scoped_replay_only_contains_and_reexecutes_target_question(Path(tmp))
    print("scoped replay tests passed")
