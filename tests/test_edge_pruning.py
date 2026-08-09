from __future__ import annotations

from dataclasses import replace
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phase0_MATH500.edge_pruning import DropCandidateOncePolicy, StableRandomDropPolicy, build_pruning_policy
from phase0_MATH500.lomo_batch import _example_key, flatten_candidate_run, select_prunable_candidates
from phase0_MATH500.lomo_plan_runner import _shard_number, _validate_plan
from phase0_MATH500.router import EdgeCandidateRecord


def _candidate(
    *,
    candidate_id: str = "c000002",
    sender: str = "Planner",
    recipient: str = "SolverA",
    kind: str = "plan",
    run_id: str = "run-a",
    stage: str = "planner",
    stage_action_id: str = "target-stage",
) -> EdgeCandidateRecord:
    return EdgeCandidateRecord(
        run_id=run_id,
        candidate_id=candidate_id,
        split="test",
        question_id=1,
        source_index=5,
        round=1,
        source_message_id="src_a000001",
        created_by_activation_id="a000001",
        sender=sender,
        recipient=recipient,
        kind=kind,
        original_content="message",
        state_message_ids=[],
        state_hash="state",
        stage=stage,
        stage_action_id=stage_action_id,
        stage_state_hash="stage-state",
    )


def test_lomo_drops_only_target_candidate() -> None:
    policy = DropCandidateOncePolicy("c000002")
    assert policy.decide(_candidate(candidate_id="c000001")).dropped is False
    assert policy.decide(_candidate(candidate_id="c000002")).dropped is True
    assert policy.decide(_candidate(candidate_id="c000003")).dropped is False


def test_random_policy_samples_one_stable_multi_edge_stage_subset() -> None:
    candidates = (
        _candidate(candidate_id="c1", sender="SolverA", recipient="Planner", kind="solver_step", stage="solver"),
        _candidate(candidate_id="c2", sender="SolverA", recipient="Judger", kind="solver_step", stage="solver"),
        _candidate(candidate_id="c3", sender="SolverA", recipient="SolverA", kind="solver_step", stage="solver"),
        _candidate(candidate_id="c4", sender="SolverB", recipient="Planner", kind="solver_step", stage="solver"),
        _candidate(candidate_id="c5", sender="SolverB", recipient="Judger", kind="solver_step", stage="solver"),
        _candidate(candidate_id="c6", sender="SolverB", recipient="SolverB", kind="solver_step", stage="solver"),
    )
    first_policy = StableRandomDropPolicy(seed=7, target_stage_action_id="target-stage")
    second_policy = StableRandomDropPolicy(seed=7, target_stage_action_id="target-stage")
    replay_policy = StableRandomDropPolicy(seed=7, target_stage_action_id="target-stage")
    first = first_policy.decide_stage(candidates, stage_state_hash="same-stage-state")
    second = second_policy.decide_stage(candidates, stage_state_hash="same-stage-state")
    replay = replay_policy.decide_stage(
        tuple(replace(candidate, run_id="replay") for candidate in candidates),
        stage_state_hash="same-stage-state",
    )

    assert [result.dropped for result in first] == [result.dropped for result in second]
    assert [result.dropped for result in first] == [result.dropped for result in replay]
    assert sum(result.dropped for result in first) >= 2
    assert first[2].dropped is False
    assert first[5].dropped is False

    later_stage = tuple(
        replace(candidate, candidate_id=f"later-{index}", stage_action_id="later-stage")
        for index, candidate in enumerate(candidates)
    )
    later = first_policy.decide_stage(later_stage, stage_state_hash="later-state")
    assert not any(result.dropped for result in later)
    assert first_policy.name == "identity"

    identity_with_self_edges = build_pruning_policy("identity", include_self_edges=True)
    assert identity_with_self_edges.is_actionable(candidates[2]) is True

    try:
        build_pruning_policy("random_drop", random_policy_seed=7)
    except ValueError as exc:
        assert "stage_action_id" in str(exc)
    else:
        raise AssertionError("random_drop without a replay stage should fail")


def test_batch_selects_each_cross_agent_candidate_once() -> None:
    rows = [
        {"candidate_id": "c1", "question_id": 1, "sender": "Input", "recipient": "Planner", "kind": "input"},
        {"candidate_id": "c2", "question_id": 1, "sender": "Planner", "recipient": "SolverA", "kind": "plan"},
        {"candidate_id": "c3", "question_id": 1, "sender": "SolverA", "recipient": "SolverA", "kind": "solver_step"},
        {"candidate_id": "c4", "question_id": 2, "sender": "SolverB", "recipient": "Judger", "kind": "solver_step"},
        {"candidate_id": "c5", "question_id": 2, "sender": "Judger", "recipient": "Output", "kind": "final_output"},
    ]
    assert [row["candidate_id"] for row in select_prunable_candidates(rows)] == ["c2", "c4"]
    assert [
        row["candidate_id"] for row in select_prunable_candidates(rows, include_self_edges=True, question_id=1)
    ] == ["c2", "c3"]
    successful = {_example_key({"split": "train", "question_id": 1, "source_index": 10})}
    candidate = {"split": "train", "question_id": 2, "source_index": 20}
    assert _example_key(candidate) not in successful


def test_flatten_candidate_run_removes_timestamp_directory(tmp_path: Path) -> None:
    candidate_dir = tmp_path / "c000002"
    run_dir = candidate_dir / "phase0_MATH500_timestamp"
    run_dir.mkdir(parents=True)
    (run_dir / "predictions.jsonl").write_text("{}\n", encoding="utf-8")
    (run_dir / "summary.json").write_text("{}", encoding="utf-8")

    result = flatten_candidate_run(run_dir, candidate_dir)

    assert result == candidate_dir
    assert not run_dir.exists()
    assert (candidate_dir / "predictions.jsonl").exists()
    assert (candidate_dir / "summary.json").exists()


def test_lomo_plan_rejects_duplicate_parent_candidate_pairs() -> None:
    row = {"parent_run": "parent", "candidate_id": "c000002"}
    _validate_plan([row])
    try:
        _validate_plan([row, dict(row)])
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate LOMO plan entry should fail validation")


def test_lomo_plan_extracts_numeric_shard_index() -> None:
    parent = "records/shard_0007_start_70_n_10/phase0_MATH500_20260709T151016Z"
    assert _shard_number(parent) == 7


if __name__ == "__main__":
    test_lomo_drops_only_target_candidate()
    test_random_policy_samples_one_stable_multi_edge_stage_subset()
    test_batch_selects_each_cross_agent_candidate_once()
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        test_flatten_candidate_run_removes_timestamp_directory(Path(tmp))
    test_lomo_plan_rejects_duplicate_parent_candidate_pairs()
    test_lomo_plan_extracts_numeric_shard_index()
    print("edge pruning tests passed")
