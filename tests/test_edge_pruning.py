from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phase0_PRM800K.edge_pruning import DropCandidateOncePolicy, StableRandomDropPolicy, stable_edge_uniform
from phase0_PRM800K.lomo_batch import _example_key, flatten_candidate_run, select_prunable_candidates
from phase0_PRM800K.lomo_plan_runner import _shard_number, _validate_plan
from phase0_PRM800K.router import EdgeCandidateRecord


def _candidate(
    *,
    candidate_id: str = "c000002",
    sender: str = "Planner",
    recipient: str = "SolverA",
    kind: str = "plan",
    run_id: str = "run-a",
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
    )


def test_lomo_drops_only_target_candidate() -> None:
    policy = DropCandidateOncePolicy("c000002")
    assert policy.decide(_candidate(candidate_id="c000001")).dropped is False
    assert policy.decide(_candidate(candidate_id="c000002")).dropped is True
    assert policy.decide(_candidate(candidate_id="c000003")).dropped is False


def test_stable_hash_ignores_run_directory_name() -> None:
    first = stable_edge_uniform(_candidate(run_id="baseline"), seed=7)
    second = stable_edge_uniform(_candidate(run_id="replay"), seed=7)
    assert first == second


def test_random_policy_protects_system_and_self_edges() -> None:
    policy = StableRandomDropPolicy(1.0, seed=7)
    assert policy.decide(_candidate()).dropped is True
    assert policy.decide(_candidate(sender="Input", recipient="Planner", kind="input")).dropped is False
    assert policy.decide(_candidate(sender="SolverA", recipient="SolverA", kind="solver_step")).dropped is False
    assert policy.decide(_candidate(sender="Judger", recipient="Output", kind="final_output")).dropped is False


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
    run_dir = candidate_dir / "phase0_PRM800K_timestamp"
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
    parent = "records/shard_0007_start_70_n_10/phase0_PRM800K_20260709T151016Z"
    assert _shard_number(parent) == 7


if __name__ == "__main__":
    test_lomo_drops_only_target_candidate()
    test_stable_hash_ignores_run_directory_name()
    test_random_policy_protects_system_and_self_edges()
    test_batch_selects_each_cross_agent_candidate_once()
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        test_flatten_candidate_run_removes_timestamp_directory(Path(tmp))
    test_lomo_plan_rejects_duplicate_parent_candidate_pairs()
    test_lomo_plan_extracts_numeric_shard_index()
    print("edge pruning tests passed")
