from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phase0.data import load_gsm8k
from phase0.evaluation import extract_gold_answer, extract_last_number, is_correct
from phase0.router import IdentityRouter


def test_load_local_parquet() -> None:
    examples = load_gsm8k(ROOT / "gsm8K" / "test-00000-of-00001.parquet", sample_size=1)
    assert len(examples) == 1
    assert examples[0].question
    assert extract_gold_answer(examples[0].answer) is not None


def test_number_parsing() -> None:
    assert str(extract_gold_answer("work\n#### 18")) == "18"
    assert str(extract_last_number("Final answer: $1,234")) == "1234"
    assert str(extract_last_number("try 2 then -3.5")) == "-3.5"
    assert extract_last_number("no numeric answer") is None
    assert is_correct(extract_last_number("Final answer: 18"), extract_gold_answer("#### 18"))


def test_identity_router_records_edges() -> None:
    router = IdentityRouter(run_id="test")
    router.forward(question_id=0, round_id=1, edge_from="A", edge_to="B", input_content="input", content="1 + 1 = 2")
    router.backfill_rewards(question_id=0, final_correct=True)
    row = router.as_dicts()[0]
    assert row["weight"] == 1.0
    assert row["compressed"] is False
    assert row["blocked"] is False
    assert row["final_reward"] == 1.0


if __name__ == "__main__":
    test_load_local_parquet()
    test_number_parsing()
    test_identity_router_records_edges()
    print("phase0 tests passed")
