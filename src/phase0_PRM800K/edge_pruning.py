from __future__ import annotations

import hashlib
import json

from .router import EdgeCandidateRecord, EdgePolicy, EdgePolicyResult, IdentityEdgePolicy


PRUNABLE_MESSAGE_KINDS = frozenset({"plan", "solver_step", "judge_feedback"})


def is_prunable_edge(candidate: EdgeCandidateRecord, *, include_self_edges: bool = False) -> bool:
    """Return whether a candidate belongs to the internal communication action space."""
    if candidate.kind not in PRUNABLE_MESSAGE_KINDS:
        return False
    if not include_self_edges and candidate.sender == candidate.recipient:
        return False
    return True


def stable_edge_uniform(candidate: EdgeCandidateRecord, *, seed: int) -> float:
    """Map an edge to a reproducible pseudo-random value in [0, 1)."""
    payload = {
        "seed": seed,
        "split": candidate.split,
        "source_index": candidate.source_index,
        "question_id": candidate.question_id,
        "round": candidate.round,
        "source_message_id": candidate.source_message_id,
        "sender": candidate.sender,
        "recipient": candidate.recipient,
        "kind": candidate.kind,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


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


class StableRandomDropPolicy:
    """Independently drop eligible messages using a replay-stable hash decision."""

    name = "random_drop"

    def __init__(self, drop_probability: float, *, seed: int, include_self_edges: bool = False) -> None:
        if not 0.0 <= drop_probability <= 1.0:
            raise ValueError("random_drop_probability must be between 0 and 1")
        self.drop_probability = drop_probability
        self.seed = seed
        self.include_self_edges = include_self_edges

    def decide(self, candidate: EdgeCandidateRecord) -> EdgePolicyResult:
        eligible = is_prunable_edge(candidate, include_self_edges=self.include_self_edges)
        should_drop = eligible and stable_edge_uniform(candidate, seed=self.seed) < self.drop_probability
        if should_drop:
            return EdgePolicyResult(action="drop", dropped=True, delivered_content="")
        return EdgePolicyResult(action="keep", delivered_content=candidate.original_content)


def build_pruning_policy(
    name: str | None,
    *,
    lomo_candidate_id: str | None = None,
    random_drop_probability: float = 0.25,
    random_policy_seed: int = 0,
    include_self_edges: bool = False,
) -> EdgePolicy:
    policy_name = (name or "identity").strip().lower()
    if policy_name == "identity":
        return IdentityEdgePolicy()
    if policy_name in {"lomo", "drop_candidate_once"}:
        if lomo_candidate_id is None:
            raise ValueError("The lomo edge policy requires --lomo-candidate-id or a replay checkpoint candidate")
        return DropCandidateOncePolicy(lomo_candidate_id, include_self_edges=include_self_edges)
    if policy_name in {"random", "random_drop"}:
        return StableRandomDropPolicy(
            random_drop_probability,
            seed=random_policy_seed,
            include_self_edges=include_self_edges,
        )
    raise ValueError(f"Unsupported edge policy: {name}. Use identity, lomo, or random_drop.")
