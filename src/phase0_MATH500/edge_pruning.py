from __future__ import annotations

import hashlib
from itertools import combinations
import json

from .router import (
    PRUNABLE_MESSAGE_KINDS,
    EdgeCandidateRecord,
    EdgePolicy,
    EdgePolicyResult,
    IdentityEdgePolicy,
    is_prunable_candidate,
)


def is_prunable_edge(candidate: EdgeCandidateRecord, *, include_self_edges: bool = False) -> bool:
    """Return whether a candidate belongs to the internal communication action space."""
    return is_prunable_candidate(candidate, include_self_edges=include_self_edges)


class DropCandidateOncePolicy:
    """LOMO policy: drop one exact message candidate and keep every other edge."""

    name = "lomo"

    def __init__(self, candidate_id: str, *, include_self_edges: bool = False) -> None:
        if not candidate_id.strip():
            raise ValueError("LOMO requires a non-empty candidate_id")
        self.candidate_id = candidate_id.strip()
        self.include_self_edges = include_self_edges

    def decide(self, candidate: EdgeCandidateRecord) -> EdgePolicyResult:
        if candidate.candidate_id != self.candidate_id:
            return EdgePolicyResult(action="keep", delivered_content=candidate.original_content)
        if not is_prunable_edge(candidate, include_self_edges=self.include_self_edges):
            raise ValueError(
                f"LOMO candidate {self.candidate_id} is protected: "
                f"{candidate.sender}->{candidate.recipient} ({candidate.kind})"
            )
        return EdgePolicyResult(action="drop", dropped=True, delivered_content="")

    def decide_stage(
        self,
        candidates: tuple[EdgeCandidateRecord, ...],
        *,
        stage_state_hash: str,
    ) -> tuple[EdgePolicyResult, ...]:
        del stage_state_hash
        return tuple(self.decide(candidate) for candidate in candidates)

    def is_actionable(self, candidate: EdgeCandidateRecord) -> bool:
        return is_prunable_edge(candidate, include_self_edges=self.include_self_edges)


class StableRandomDropPolicy:
    """Apply one replay-stable multi-edge mask at one exact stage checkpoint."""

    def __init__(
        self,
        *,
        seed: int,
        target_stage_action_id: str,
        include_self_edges: bool = False,
        min_dropped_edges: int = 2,
    ) -> None:
        if min_dropped_edges < 2:
            raise ValueError("random_drop_min_edges must be at least 2")
        if not target_stage_action_id.strip():
            raise ValueError("random_drop requires a non-empty target_stage_action_id")
        self.seed = seed
        self.target_stage_action_id = target_stage_action_id.strip()
        self.include_self_edges = include_self_edges
        self.min_dropped_edges = min_dropped_edges
        self.applied = False
        self._current_policy_name = "identity"

    @property
    def name(self) -> str:
        return self._current_policy_name

    def decide(self, candidate: EdgeCandidateRecord) -> EdgePolicyResult:
        return EdgePolicyResult(action="keep", delivered_content=candidate.original_content)

    def decide_stage(
        self,
        candidates: tuple[EdgeCandidateRecord, ...],
        *,
        stage_state_hash: str,
    ) -> tuple[EdgePolicyResult, ...]:
        stage_action_ids = {candidate.stage_action_id for candidate in candidates}
        if len(stage_action_ids) != 1:
            raise ValueError(f"random_drop received candidates from multiple stages: {stage_action_ids}")
        current_stage_action_id = next(iter(stage_action_ids))
        if self.applied or current_stage_action_id != self.target_stage_action_id:
            self._current_policy_name = "identity"
            return tuple(self.decide(candidate) for candidate in candidates)

        self.applied = True
        self._current_policy_name = "random_drop"
        eligible_indexes = [index for index, candidate in enumerate(candidates) if self.is_actionable(candidate)]
        if len(eligible_indexes) < self.min_dropped_edges:
            return tuple(self.decide(candidate) for candidate in candidates)

        subsets = [
            subset
            for size in range(self.min_dropped_edges, len(eligible_indexes) + 1)
            for subset in combinations(eligible_indexes, size)
        ]
        payload = {
            "seed": self.seed,
            "split": candidates[0].split,
            "source_index": candidates[0].source_index,
            "question_id": candidates[0].question_id,
            "round": candidates[0].round,
            "stage": candidates[0].stage,
            "stage_state_hash": stage_state_hash,
            "edges": [(candidate.sender, candidate.recipient, candidate.kind) for candidate in candidates],
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).digest()
        selected = set(subsets[int.from_bytes(digest[:8], "big") % len(subsets)])
        return tuple(
            EdgePolicyResult(action="drop", dropped=True, delivered_content="")
            if index in selected
            else self.decide(candidate)
            for index, candidate in enumerate(candidates)
        )

    def is_actionable(self, candidate: EdgeCandidateRecord) -> bool:
        return is_prunable_edge(candidate, include_self_edges=self.include_self_edges)


def build_pruning_policy(
    name: str | None,
    *,
    lomo_candidate_id: str | None = None,
    random_policy_seed: int = 0,
    random_drop_min_edges: int = 2,
    random_drop_stage_action_id: str | None = None,
    include_self_edges: bool = False,
) -> EdgePolicy:
    policy_name = (name or "identity").strip().lower()
    if policy_name == "identity":
        return IdentityEdgePolicy(include_self_edges=include_self_edges)
    if policy_name in {"lomo", "drop_candidate_once"}:
        if lomo_candidate_id is None:
            raise ValueError("The lomo edge policy requires --lomo-candidate-id or a replay checkpoint candidate")
        return DropCandidateOncePolicy(lomo_candidate_id, include_self_edges=include_self_edges)
    if policy_name in {"random", "random_drop"}:
        if random_drop_stage_action_id is None:
            raise ValueError("random_drop requires an identity replay checkpoint with a stage_action_id")
        return StableRandomDropPolicy(
            seed=random_policy_seed,
            target_stage_action_id=random_drop_stage_action_id,
            include_self_edges=include_self_edges,
            min_dropped_edges=random_drop_min_edges,
        )
    raise ValueError(f"Unsupported edge policy: {name}. Use identity, lomo, or random_drop.")
