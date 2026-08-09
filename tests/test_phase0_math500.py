from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phase0_MATH500.agents import LLMResponse, run_mas_full_path, run_single_agent
from phase0_MATH500.data import build_splits, select_split
from phase0_MATH500.edge_pruning import StableRandomDropPolicy
from phase0_MATH500.router import DropEdgePolicy, FullForwardRouter, ReplayController


class StepwiseMockModel:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def complete(self, *, system: str, user: str, source: str) -> LLMResponse:
        self.calls.append(source)
        if source == "single_agent_step_1":
            return LLMResponse(content="7.8 minutes is 7 minutes and 0.8 minutes.", latency_ms=1)
        if source == "single_agent_step_2":
            return LLMResponse(content="# Answer\n\n468", latency_ms=2)
        raise AssertionError(f"Unexpected source: {source}")


class FinalRoundMockModel:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def complete(self, *, system: str, user: str, source: str) -> LLMResponse:
        self.calls.append(source)
        if source.startswith("Planner"):
            return LLMResponse(content="Plan: compute directly from the input problem.", latency_ms=1)
        if source.startswith("SolverA"):
            return LLMResponse(content="40 + 2 = 42", latency_ms=2)
        if source.startswith("SolverB"):
            return LLMResponse(content="The sum is 42.", latency_ms=3)
        if source.startswith("Judger"):
            return LLMResponse(content="FINAL:\n# Answer\n\n42", latency_ms=4)
        raise AssertionError(f"Unexpected source: {source}")


class FeedbackThenFinalMockModel:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def complete(self, *, system: str, user: str, source: str) -> LLMResponse:
        self.calls.append(source)
        if source.startswith("Planner"):
            return LLMResponse(content=f"Plan for {source}.", latency_ms=1)
        if source.startswith("SolverA"):
            return LLMResponse(content=f"SolverA step from {source}.", latency_ms=2)
        if source.startswith("SolverB"):
            return LLMResponse(content=f"SolverB step from {source}.", latency_ms=3)
        if source == "Judger_round_1":
            return LLMResponse(content="FEEDBACK: verify the arithmetic before finalizing.", latency_ms=4)
        if source == "Judger_round_2":
            return LLMResponse(content="FINAL:\n# Answer\n\n42", latency_ms=5)
        raise AssertionError(f"Unexpected source: {source}")


class NeverFinalMockModel:
    async def complete(self, *, system: str, user: str, source: str) -> LLMResponse:
        if source.startswith("Planner"):
            return LLMResponse(content="Plan: keep solving.", latency_ms=1)
        if source.startswith("SolverA"):
            return LLMResponse(content="SolverA partial step.", latency_ms=2)
        if source.startswith("SolverB"):
            return LLMResponse(content="SolverB partial step.", latency_ms=3)
        if source.startswith("Judger"):
            return LLMResponse(content="FEEDBACK: keep going.", latency_ms=4)
        raise AssertionError(f"Unexpected source: {source}")


class ReplayLiveMockModel:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def complete(self, *, system: str, user: str, source: str) -> LLMResponse:
        self.calls.append(source)
        if source.startswith("SolverA"):
            return LLMResponse(content="Replay SolverA step.", latency_ms=20)
        if source.startswith("SolverB"):
            return LLMResponse(content="Replay SolverB step.", latency_ms=30)
        if source.startswith("Judger"):
            return LLMResponse(content="FINAL:\n# Answer\n\nreplay", latency_ms=40)
        if source.startswith("Planner"):
            return LLMResponse(content="Replay planner should only appear after checkpoint.", latency_ms=10)
        raise AssertionError(f"Unexpected source: {source}")


def _write_prm800k_jsonl(path: Path, count: int) -> None:
    rows = []
    for idx in range(count):
        rows.append(
            {
                "question": {"problem": f"Problem {idx}?", "ground_truth_answer": str(idx)},
                "level": 99,
                "finish_reason": "solution",
                "total_time": idx,
                "label": {
                    "steps": [
                        {
                            "completions": [{"text": f"Step {idx}: compute the relevant value.", "rating": 1}],
                            "chosen_completion": 0,
                        },
                        {
                            "completions": [
                                {"text": "Incorrect distractor.", "rating": -1},
                                {"text": f"# Answer\n\n{idx}", "rating": 1},
                            ],
                            "chosen_completion": 1,
                        },
                    ]
                },
            }
        )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _write_math500_jsonl(path: Path, count: int) -> None:
    rows = []
    for idx in range(count):
        rows.append(
            {
                "problem": f"MATH problem {idx}?",
                "solution": f"A reference solution ending in \\boxed{{{idx}}}.",
                "answer": str(idx),
                "subject": "Algebra",
                "level": 2,
                "unique_id": f"test/algebra/{idx}.json",
            }
        )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_replay_logs(run_dir: Path, router: FullForwardRouter) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(run_dir / "messages.jsonl", router.messages_as_dicts())
    _write_jsonl(run_dir / "activations.jsonl", router.activations_as_dicts())
    _write_jsonl(run_dir / "edge_candidates.jsonl", router.edge_candidates_as_dicts())
    _write_jsonl(run_dir / "edge_decisions.jsonl", router.edge_decisions_as_dicts())
    _write_jsonl(run_dir / "stage_actions.jsonl", router.stage_actions_as_dicts())


def test_build_splits_reads_prm800k_reference_steps_and_ignores_level(tmp_path: Path) -> None:
    data_path = tmp_path / "sample.jsonl"
    _write_prm800k_jsonl(data_path, 1)

    result = build_splits(data_path, train_size=1, test_size=0, seed=123)

    assert len(result.train) == 1
    assert result.train[0].problem == "Problem 0?"
    assert result.train[0].ground_truth_answer == "0"
    assert result.train[0].reference_steps == ["Step 0: compute the relevant value.", "# Answer\n\n0"]
    assert "level" not in result.train[0].to_dict()


def test_build_splits_reads_math500_flat_records(tmp_path: Path) -> None:
    data_path = tmp_path / "math500.jsonl"
    _write_math500_jsonl(data_path, 5)

    result = build_splits(data_path, train_size=4, test_size=1, seed=0)

    assert len(result.train) == 4
    assert len(result.test) == 1
    assert result.valid_records == 5
    assert result.train[0].problem.startswith("MATH problem")
    assert result.train[0].ground_truth_answer
    assert result.train[0].finish_reason == "solution"
    assert result.train[0].reference_steps
def test_build_splits_are_reproducible_and_disjoint(tmp_path: Path) -> None:
    data_path = tmp_path / "sample.jsonl"
    _write_prm800k_jsonl(data_path, 20)

    first = build_splits(data_path, train_size=6, test_size=2, seed=7)
    second = build_splits(data_path, train_size=6, test_size=2, seed=7)

    first_train_ids = [row.source_index for row in first.train]
    second_train_ids = [row.source_index for row in second.train]
    assert first_train_ids == second_train_ids
    train_ids = {row.source_index for row in first.train}
    test_ids = {row.source_index for row in first.test}
    assert train_ids.isdisjoint(test_ids)


def test_select_split_supports_non_overlapping_chunks(tmp_path: Path) -> None:
    data_path = tmp_path / "sample.jsonl"
    _write_prm800k_jsonl(data_path, 20)
    result = build_splits(data_path, train_size=12, test_size=2, seed=11)

    first = select_split(result, "train", sample_size=4, start_index=0)
    second = select_split(result, "train", sample_size=4, start_index=4)
    third = select_split(result, "train", sample_size=4, start_index=8)

    chunks = [{row.source_index for row in chunk} for chunk in (first, second, third)]
    assert len(first) == len(second) == len(third) == 4
    assert chunks[0].isdisjoint(chunks[1])
    assert chunks[0].isdisjoint(chunks[2])
    assert chunks[1].isdisjoint(chunks[2])


def test_build_splits_fails_when_not_enough_valid_records(tmp_path: Path) -> None:
    data_path = tmp_path / "sample.jsonl"
    _write_prm800k_jsonl(data_path, 3)

    try:
        build_splits(data_path, train_size=3, test_size=1, seed=0)
    except ValueError as exc:
        assert "Not enough valid data records" in str(exc)
    else:
        raise AssertionError("Expected ValueError for undersized data")


def test_single_agent_outputs_prm800k_style_steps() -> None:
    model = StepwiseMockModel()
    solution = asyncio.run(run_single_agent("How many seconds are in 7.8 minutes?", model))  # type: ignore[arg-type]

    assert solution.steps == ["7.8 minutes is 7 minutes and 0.8 minutes.", "# Answer\n\n468"]
    assert solution.content == "7.8 minutes is 7 minutes and 0.8 minutes.\n\n# Answer\n\n468"
    assert model.calls == ["single_agent_step_1", "single_agent_step_2"]


def test_router_records_full_forward_with_topology_fields() -> None:
    router = FullForwardRouter(run_id="test")
    router.forward(
        split="test",
        question_id=0,
        source_index=10,
        round_id=1,
        edge_from="A",
        edge_to="B",
        input_content="input",
        content="content",
        latency_ms=5,
        message_kind="plan",
        step_index=2,
    )
    row = router.as_dicts()[0]
    assert row["split"] == "test"
    assert row["action"] == "full_forward"
    assert row["message_kind"] == "plan"
    assert row["step_index"] == 2
    assert row["candidate_id"] == "c000001"
    assert row["message_id"] == "m000001"
    assert row["terminal"] is False
    assert row["termination_reason"] is None
    assert "level" not in row
    assert "step_reward" not in row
    assert "final_reward" not in row
    assert "correctness" not in row


async def _run_mock_mas(
    model: object,
    *,
    max_rounds: int = 4,
    router: FullForwardRouter | None = None,
) -> tuple[FullForwardRouter, object]:
    active_router = router or FullForwardRouter(run_id="mock")
    result = await run_mas_full_path(
        split="test",
        question_id=0,
        source_index=5,
        problem="What is 40 + 2?",
        model=model,  # type: ignore[arg-type]
        router=active_router,
        max_rounds=max_rounds,
        stall_rounds=2,
        force_final_on_stop=True,
    )
    return active_router, result


def test_mas_topology_records_cross_layer_edges_and_self_memory_until_final() -> None:
    router, result = asyncio.run(_run_mock_mas(FinalRoundMockModel()))
    rows = router.as_dicts()

    edges = [(row["edge_from"], row["edge_to"]) for row in rows]
    assert edges == [
        ("Input", "Planner"),
        ("Planner", "SolverA"),
        ("Planner", "SolverB"),
        ("Planner", "Judger"),
        ("SolverA", "Planner"),
        ("SolverA", "Judger"),
        ("SolverA", "SolverA"),
        ("SolverB", "Planner"),
        ("SolverB", "Judger"),
        ("SolverB", "SolverB"),
        ("Judger", "Output"),
    ]
    assert not any(edge in edges for edge in [("SolverA", "SolverB"), ("SolverB", "SolverA")])
    assert rows[-1]["message_kind"] == "final_output"
    assert rows[-1]["terminal"] is True
    assert rows[-1]["termination_reason"] == "judge_final"
    assert result.content.startswith("FINAL:")
    assert result.rounds_used == 1

    stage_actions = router.stage_actions_as_dicts()
    assert [row["stage"] for row in stage_actions] == ["input", "planner", "solver", "judger"]
    assert [row["action_mask"] for row in stage_actions] == ["", "000", "0000", ""]
    solver_action = stage_actions[2]
    assert solver_action["edge_order"] == [
        "SolverA->Planner",
        "SolverA->Judger",
        "SolverB->Planner",
        "SolverB->Judger",
    ]


def test_random_drop_applies_once_at_target_joint_stage() -> None:
    planner_stage = "test:5:0:r1:planner"
    policy = StableRandomDropPolicy(seed=7, target_stage_action_id=planner_stage)
    router = FullForwardRouter(run_id="joint-mask", edge_policy=policy)
    router, _result = asyncio.run(_run_mock_mas(FinalRoundMockModel(), router=router))

    actions = {row.stage: row for row in router.stage_actions}
    assert len(actions["planner"].action_mask) == 3
    assert actions["planner"].action_mask.count("1") >= 2
    assert actions["planner"].policy_name == "random_drop"
    assert len(actions["solver"].action_mask) == 4
    assert actions["solver"].action_mask == "0000"
    assert actions["solver"].policy_name == "identity"
    assert actions["solver"].edge_order == [
        "SolverA->Planner",
        "SolverA->Judger",
        "SolverB->Planner",
        "SolverB->Judger",
    ]
    assert actions["input"].action_mask == ""
    assert actions["judger"].action_mask == ""

    solver_candidates = [candidate for candidate in router.edge_candidates if candidate.stage == "solver"]
    assert {candidate.stage_action_id for candidate in solver_candidates} == {actions["solver"].stage_action_id}
    assert {candidate.stage_state_hash for candidate in solver_candidates} == {actions["solver"].state_hash}
    solver_candidates_by_id = {candidate.candidate_id: candidate for candidate in solver_candidates}
    dropped_self_edges = [
        decision
        for decision in router.edge_decisions
        if decision.dropped
        and decision.candidate_id in solver_candidates_by_id
        and solver_candidates_by_id[decision.candidate_id].sender
        == solver_candidates_by_id[decision.candidate_id].recipient
    ]
    assert dropped_self_edges == []

    self_edge_policy = StableRandomDropPolicy(
        seed=7,
        target_stage_action_id="test:5:0:r1:solver",
        include_self_edges=True,
    )
    self_edge_router = FullForwardRouter(run_id="joint-mask-self", edge_policy=self_edge_policy)
    self_edge_router, _result = asyncio.run(_run_mock_mas(FinalRoundMockModel(), router=self_edge_router))
    self_edge_solver_action = next(row for row in self_edge_router.stage_actions if row.stage == "solver")
    assert self_edge_solver_action.edge_order == [
        "SolverA->Planner",
        "SolverA->Judger",
        "SolverA->SolverA",
        "SolverB->Planner",
        "SolverB->Judger",
        "SolverB->SolverB",
    ]
    assert len(self_edge_solver_action.action_mask) == 6
    assert self_edge_solver_action.action_mask.count("1") >= 2
    self_edge_planner_action = next(row for row in self_edge_router.stage_actions if row.stage == "planner")
    assert self_edge_planner_action.action_mask == "000"

    feedback_router = FullForwardRouter(
        run_id="joint-mask-feedback",
        edge_policy=StableRandomDropPolicy(
            seed=7,
            target_stage_action_id="test:5:0:r1:judger",
        ),
    )
    feedback_router, _result = asyncio.run(
        _run_mock_mas(FeedbackThenFinalMockModel(), max_rounds=3, router=feedback_router)
    )
    feedback_action = next(
        row for row in feedback_router.stage_actions if row.stage == "judger" and row.round == 1
    )
    assert feedback_action.edge_order == [
        "Judger->Planner",
        "Judger->SolverA",
        "Judger->SolverB",
    ]
    assert len(feedback_action.action_mask) == 3
    assert feedback_action.action_mask.count("1") >= 2


def test_markov_prompts_use_only_inbox_and_control_message_ids() -> None:
    router, result = asyncio.run(_run_mock_mas(FeedbackThenFinalMockModel(), max_rounds=3))
    assert result.rounds_used == 2

    for activation in router.activations:
        prompt = activation.user_prompt
        for message_id in activation.input_message_ids + activation.control_message_ids:
            message = router.message_by_id(message_id)
            assert message.message_id in prompt
            assert message.content in prompt

    solver_a_round_2 = next(row for row in router.activations if row.agent == "SolverA" and row.round == 2)
    solver_a_inputs = [router.message_by_id(message_id) for message_id in solver_a_round_2.input_message_ids]
    assert any(message.sender == "SolverA" and message.recipient == "SolverA" for message in solver_a_inputs)

    judger_round_2 = next(row for row in router.activations if row.agent == "Judger" and row.round == 2)
    judger_inputs = [router.message_by_id(message_id) for message_id in judger_round_2.input_message_ids]
    assert any(message.sender == "Judger" and message.recipient == "Judger" for message in judger_inputs)


def test_judger_feedback_broadcasts_internally_before_final() -> None:
    router, result = asyncio.run(_run_mock_mas(FeedbackThenFinalMockModel(), max_rounds=3))
    rows = router.as_dicts()

    feedback_edges = [(row["edge_from"], row["edge_to"]) for row in rows if row["message_kind"] == "judge_feedback"]
    assert feedback_edges == [("Judger", "Planner"), ("Judger", "SolverA"), ("Judger", "SolverB"), ("Judger", "Judger")]
    output_rows = [row for row in rows if row["edge_to"] == "Output"]
    assert len(output_rows) == 1
    assert output_rows[0]["round"] == 2
    assert output_rows[0]["terminal"] is True
    assert result.termination_reason == "judge_final"
    assert result.rounds_used == 2


def test_max_rounds_forces_judger_output_and_stops_feedback_loop() -> None:
    router, result = asyncio.run(_run_mock_mas(NeverFinalMockModel(), max_rounds=1))
    rows = router.as_dicts()

    assert result.termination_reason == "max_rounds"
    assert result.rounds_used == 1
    assert result.content.startswith("FINAL:")
    assert not [row for row in rows if row["message_kind"] == "judge_feedback"]
    final_rows = [row for row in rows if row["message_kind"] == "final_output"]
    assert len(final_rows) == 1
    assert final_rows[0]["edge_from"] == "Judger"
    assert final_rows[0]["edge_to"] == "Output"
    assert final_rows[0]["termination_reason"] == "max_rounds"


def test_drop_policy_records_decision_without_delivering_to_inbox() -> None:
    router = FullForwardRouter(run_id="drop", edge_policy=DropEdgePolicy({("Planner", "SolverA")}))
    router, _result = asyncio.run(_run_mock_mas(FinalRoundMockModel(), router=router))

    dropped_decisions = [row for row in router.edge_decisions_as_dicts() if row["dropped"]]
    assert len(dropped_decisions) == 1
    dropped_candidate = next(
        row for row in router.edge_candidates_as_dicts() if row["candidate_id"] == dropped_decisions[0]["candidate_id"]
    )
    assert (dropped_candidate["sender"], dropped_candidate["recipient"]) == ("Planner", "SolverA")
    assert not [
        message
        for message in router.messages_as_dicts()
        if message["sender"] == "Planner" and message["recipient"] == "SolverA"
    ]


def test_reexecution_replay_uses_parent_prefix_then_live_model_after_checkpoint(tmp_path: Path) -> None:
    parent_router, _parent_result = asyncio.run(_run_mock_mas(FinalRoundMockModel(), max_rounds=1))
    parent_dir = tmp_path / "parent"
    _write_replay_logs(parent_dir, parent_router)

    # c000003 is the second Planner edge; joint replay must restart at the first edge of that stage.
    replay_controller = ReplayController(parent_dir, "c000003")
    child_router = FullForwardRouter(run_id="child", replay_controller=replay_controller)
    live_model = ReplayLiveMockModel()
    child_router, child_result = asyncio.run(_run_mock_mas(live_model, max_rounds=1, router=child_router))

    assert "Planner_round_1" not in live_model.calls
    assert "SolverA_round_1" in live_model.calls
    assert "SolverB_round_1" in live_model.calls
    assert "Judger_round_1" in live_model.calls
    planner_activation = next(row for row in child_router.activations if row.agent == "Planner")
    solver_activation = next(row for row in child_router.activations if row.agent == "SolverA")
    assert planner_activation.replayed is True
    assert solver_activation.replayed is False
    assert child_result.content.startswith("FINAL:")
    assert "replay" in child_result.content
    assert child_router.edge_decisions[0].replayed is True
    assert child_router.edge_decisions[1].replayed is False


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        test_build_splits_reads_prm800k_reference_steps_and_ignores_level(tmp_path)
        test_build_splits_reads_math500_flat_records(tmp_path)
        test_build_splits_are_reproducible_and_disjoint(tmp_path)
        test_select_split_supports_non_overlapping_chunks(tmp_path)
        test_build_splits_fails_when_not_enough_valid_records(tmp_path)
        test_reexecution_replay_uses_parent_prefix_then_live_model_after_checkpoint(tmp_path)
    test_single_agent_outputs_prm800k_style_steps()
    test_router_records_full_forward_with_topology_fields()
    test_mas_topology_records_cross_layer_edges_and_self_memory_until_final()
    test_random_drop_applies_once_at_target_joint_stage()
    test_markov_prompts_use_only_inbox_and_control_message_ids()
    test_judger_feedback_broadcasts_internally_before_final()
    test_max_rounds_forces_judger_output_and_stops_feedback_loop()
    test_drop_policy_records_decision_without_delivering_to_inbox()
    print("phase0_MATH500 tests passed")
