from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Any

from .evaluation import approx_tokens, heuristic_step_reward


@dataclass
class TraceRecord:
    question_id: int
    run_id: str
    mode: str
    round: int
    edge_from: str
    edge_to: str
    content: str
    weight: float
    compressed: bool
    blocked: bool
    approx_input_tokens: int
    approx_output_tokens: int
    latency_ms: int
    step_reward: float | None
    final_reward: float | None


class IdentityRouter:
    """Phase0 router: forward every edge unchanged and log the communication."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.records: list[TraceRecord] = []

    def forward(
        self,
        *,
        question_id: int,
        round_id: int,
        edge_from: str,
        edge_to: str,
        content: str,
        input_content: str = "",
        latency_ms: int = 0,
    ) -> str:
        self.records.append(
            TraceRecord(
                question_id=question_id,
                run_id=self.run_id,
                mode="mas_identity",
                round=round_id,
                edge_from=edge_from,
                edge_to=edge_to,
                content=content,
                weight=1.0,
                compressed=False,
                blocked=False,
                approx_input_tokens=approx_tokens(input_content),
                approx_output_tokens=approx_tokens(content),
                latency_ms=latency_ms,
                step_reward=heuristic_step_reward(content),
                final_reward=None,
            )
        )
        return content

    def backfill_rewards(self, *, question_id: int, final_correct: bool) -> None:
        final_reward = 1.0 if final_correct else 0.0
        for record in self.records:
            if record.question_id == question_id:
                record.final_reward = final_reward
                record.step_reward = heuristic_step_reward(record.content, final_correct)

    def as_dicts(self) -> list[dict[str, Any]]:
        return [asdict(record) for record in self.records]


def now_ms() -> int:
    return int(time.perf_counter() * 1000)
