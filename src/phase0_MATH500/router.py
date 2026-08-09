from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Literal, Protocol

from .evaluation import approx_tokens


SplitName = Literal["train", "test"]
MessageKind = Literal["input", "plan", "solver_step", "judge_feedback", "final_output", "control", "message"]
PRUNABLE_MESSAGE_KINDS = frozenset({"plan", "solver_step", "judge_feedback"})

_STAGE_EDGE_ORDER: dict[str, tuple[tuple[str, str], ...]] = {
    "input": (("Input", "Planner"),),
    "planner": (
        ("Planner", "SolverA"),
        ("Planner", "SolverB"),
        ("Planner", "Judger"),
    ),
    "solver": (
        ("SolverA", "Planner"),
        ("SolverA", "Judger"),
        ("SolverA", "SolverA"),
        ("SolverB", "Planner"),
        ("SolverB", "Judger"),
        ("SolverB", "SolverB"),
    ),
    "judger": (
        ("Judger", "Planner"),
        ("Judger", "SolverA"),
        ("Judger", "SolverB"),
        ("Judger", "Judger"),
        ("Judger", "Output"),
    ),
}


@dataclass
class TraceRecord:
    run_id: str
    split: SplitName
    question_id: int
    source_index: int
    round: int
    edge_from: str
    edge_to: str
    action: str
    message_kind: MessageKind | str
    content: str
    blocked: bool
    compressed: bool
    approx_input_tokens: int
    approx_output_tokens: int
    latency_ms: int
    step_index: int | None = None
    terminal: bool = False
    termination_reason: str | None = None
    message_id: str | None = None
    candidate_id: str | None = None
    decision_id: str | None = None
    source_message_id: str | None = None
    activation_id: str | None = None
    stage: str | None = None
    stage_action_id: str | None = None
    action_mask: str | None = None
    action_edge_index: int | None = None


@dataclass
class MessageRecord:
    run_id: str
    message_id: str
    split: SplitName
    question_id: int
    source_index: int
    round: int
    sender: str
    recipient: str
    kind: MessageKind | str
    content: str
    source_message_id: str | None
    created_by_activation_id: str | None
    control: bool
    terminal: bool = False
    termination_reason: str | None = None


@dataclass
class ActivationRecord:
    run_id: str
    activation_id: str
    split: SplitName
    question_id: int
    source_index: int
    round: int
    agent: str
    stage: str
    input_message_ids: list[str]
    control_message_ids: list[str]
    system_prompt: str
    user_prompt: str
    prompt_hash: str
    model_config: dict[str, Any]
    output_source_message_id: str
    output_content: str
    latency_ms: int
    replayed: bool = False


@dataclass
class EdgeCandidateRecord:
    run_id: str
    candidate_id: str
    split: SplitName
    question_id: int
    source_index: int
    round: int
    source_message_id: str
    created_by_activation_id: str | None
    sender: str
    recipient: str
    kind: MessageKind | str
    original_content: str
    state_message_ids: list[str]
    state_hash: str
    stage: str = "legacy"
    stage_action_id: str | None = None
    stage_edge_index: int | None = None
    stage_state_hash: str | None = None
    terminal: bool = False
    termination_reason: str | None = None


@dataclass
class EdgeDecisionRecord:
    run_id: str
    decision_id: str
    candidate_id: str
    policy_name: str
    action: str
    dropped: bool
    delivered_message_id: str | None
    delivered_content: str
    content_hash: str
    replayed: bool = False
    stage_action_id: str | None = None
    action_mask: str | None = None
    action_edge_index: int | None = None


@dataclass
class RLEdgeSampleRecord:
    sample_id: str
    run_id: str
    candidate_id: str
    question_id: int
    round: int
    sender: str
    recipient: str
    message_kind: MessageKind | str
    state_message_ids: list[str]
    state_hash: str
    original_content: str
    action: str
    delivered_content: str
    dropped: bool
    downstream_activation_ids: list[str]
    terminal: bool
    termination_reason: str | None
    reward: float | None = None
    stage: str | None = None
    stage_action_id: str | None = None
    stage_state_hash: str | None = None
    action_mask: str | None = None
    action_edge_index: int | None = None


@dataclass(frozen=True)
class EdgeEmission:
    sender: str
    recipients: tuple[str, ...]
    kind: MessageKind | str
    content: str
    source_message_id: str
    created_by_activation_id: str | None
    state_message_ids: list[str]
    input_content: str = ""
    latency_ms: int = 0
    step_index: int | None = None
    terminal: bool = False
    termination_reason: str | None = None


@dataclass
class StageActionRecord:
    stage_action_id: str
    run_id: str
    split: SplitName
    question_id: int
    source_index: int
    round: int
    stage: str
    policy_name: str
    state_message_ids: list[str]
    state_hash: str
    candidate_ids: list[str]
    action_candidate_ids: list[str]
    edge_order: list[str]
    action_mask: str
    dropped_candidate_ids: list[str]
    replayed_candidate_ids: list[str]
    reward: float | None = None


@dataclass(frozen=True)
class EdgePolicyResult:
    action: str
    dropped: bool = False
    delivered_content: str | None = None


class EdgePolicy(Protocol):
    name: str
    include_self_edges: bool

    def decide(self, candidate: EdgeCandidateRecord) -> EdgePolicyResult:
        ...

    def decide_stage(
        self,
        candidates: tuple[EdgeCandidateRecord, ...],
        *,
        stage_state_hash: str,
    ) -> tuple[EdgePolicyResult, ...]:
        ...

    def is_actionable(self, candidate: EdgeCandidateRecord) -> bool:
        ...


def is_prunable_candidate(candidate: EdgeCandidateRecord, *, include_self_edges: bool = False) -> bool:
    if candidate.kind not in PRUNABLE_MESSAGE_KINDS:
        return False
    return include_self_edges or candidate.sender != candidate.recipient


class IdentityEdgePolicy:
    name = "identity"

    def __init__(self, *, include_self_edges: bool = False) -> None:
        self.include_self_edges = include_self_edges

    def decide(self, candidate: EdgeCandidateRecord) -> EdgePolicyResult:
        return EdgePolicyResult(action="keep", dropped=False, delivered_content=candidate.original_content)

    def decide_stage(
        self,
        candidates: tuple[EdgeCandidateRecord, ...],
        *,
        stage_state_hash: str,
    ) -> tuple[EdgePolicyResult, ...]:
        del stage_state_hash
        return tuple(self.decide(candidate) for candidate in candidates)

    def is_actionable(self, candidate: EdgeCandidateRecord) -> bool:
        return is_prunable_candidate(candidate, include_self_edges=self.include_self_edges)


class DropEdgePolicy:
    """Small test hook for future pruning policies; production default remains identity."""

    def __init__(self, edges: set[tuple[str, str]], *, name: str = "drop_mock") -> None:
        self.edges = edges
        self.name = name
        self.include_self_edges = any(sender == recipient for sender, recipient in edges)

    def decide(self, candidate: EdgeCandidateRecord) -> EdgePolicyResult:
        if (candidate.sender, candidate.recipient) in self.edges:
            return EdgePolicyResult(action="drop", dropped=True, delivered_content="")
        return EdgePolicyResult(action="keep", dropped=False, delivered_content=candidate.original_content)

    def decide_stage(
        self,
        candidates: tuple[EdgeCandidateRecord, ...],
        *,
        stage_state_hash: str,
    ) -> tuple[EdgePolicyResult, ...]:
        del stage_state_hash
        return tuple(self.decide(candidate) for candidate in candidates)

    def is_actionable(self, candidate: EdgeCandidateRecord) -> bool:
        return is_prunable_candidate(candidate, include_self_edges=self.include_self_edges)


def build_edge_policy(name: str | None) -> EdgePolicy:
    policy_name = (name or "identity").strip().lower()
    if policy_name == "identity":
        return IdentityEdgePolicy()
    raise ValueError(f"Unsupported edge policy: {name}. Only 'identity' is built in for Phase0.")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Replay log not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


class ReplayController:
    """Restores a parent run prefix, then switches to live LLM calls at a candidate checkpoint."""

    def __init__(self, parent_run_dir: str | Path, checkpoint_candidate_id: str) -> None:
        self.parent_run_dir = Path(parent_run_dir)
        self.checkpoint_candidate_id = checkpoint_candidate_id
        self.messages = _read_jsonl(self.parent_run_dir / "messages.jsonl")
        self.activations = _read_jsonl(self.parent_run_dir / "activations.jsonl")
        self.edge_candidates = _read_jsonl(self.parent_run_dir / "edge_candidates.jsonl")
        self.edge_decisions = _read_jsonl(self.parent_run_dir / "edge_decisions.jsonl")
        self._activation_by_id = {str(row["activation_id"]): row for row in self.activations}
        self._decision_by_candidate_id = {str(row["candidate_id"]): row for row in self.edge_decisions}
        self._candidate_by_id = {str(row["candidate_id"]): row for row in self.edge_candidates}
        self._candidate_ids = {str(row["candidate_id"]) for row in self.edge_candidates}
        if checkpoint_candidate_id not in self._candidate_ids:
            raise ValueError(f"checkpoint_candidate_id not found in parent run: {checkpoint_candidate_id}")
        checkpoint = self._candidate_by_id[checkpoint_candidate_id]
        self.checkpoint_stage_action_id = str(checkpoint.get("stage_action_id") or "") or None
        self.reexecution_started = False
        self.checkpoint_seen = False

    def replay_activation_output(self, activation_id: str) -> str | None:
        if self.reexecution_started:
            return None
        row = self._activation_by_id.get(activation_id)
        if row is None:
            return None
        return str(row.get("output_content", ""))

    def replay_decision(self, candidate_id: str) -> dict[str, Any] | None:
        if self.reexecution_started:
            return None
        candidate = self._candidate_by_id.get(candidate_id, {})
        at_stage_boundary = (
            self.checkpoint_stage_action_id is not None
            and str(candidate.get("stage_action_id") or "") == self.checkpoint_stage_action_id
        )
        if at_stage_boundary or candidate_id == self.checkpoint_candidate_id:
            self.checkpoint_seen = True
            self.reexecution_started = True
            return None
        return self._decision_by_candidate_id.get(candidate_id)


class FullForwardRouter:
    """Event-sourced Phase0 router with replay-ready edge candidates and delivered messages."""

    def __init__(
        self,
        run_id: str,
        *,
        edge_policy: EdgePolicy | None = None,
        replay_controller: ReplayController | None = None,
    ) -> None:
        self.run_id = run_id
        self.edge_policy = edge_policy or IdentityEdgePolicy()
        self.replay_controller = replay_controller
        self.records: list[TraceRecord] = []
        self.messages: list[MessageRecord] = []
        self.activations: list[ActivationRecord] = []
        self.edge_candidates: list[EdgeCandidateRecord] = []
        self.edge_decisions: list[EdgeDecisionRecord] = []
        self.rl_edge_samples: list[RLEdgeSampleRecord] = []
        self.stage_actions: list[StageActionRecord] = []
        self._counters = {"m": 0, "a": 0, "c": 0, "d": 0, "s": 0, "src": 0}

    def _next_id(self, prefix: str) -> str:
        self._counters[prefix] += 1
        return f"{prefix}{self._counters[prefix]:06d}"

    def next_activation_id(self) -> str:
        return self._next_id("a")

    def replay_activation_output(self, activation_id: str) -> str | None:
        if self.replay_controller is None:
            return None
        return self.replay_controller.replay_activation_output(activation_id)

    def source_id_for_activation(self, activation_id: str) -> str:
        return f"src_{activation_id}"

    def content_hash(self, content: str) -> str:
        return _sha256_text(content)

    def state_hash(self, message_ids: list[str]) -> str:
        messages = [self.message_by_id(message_id) for message_id in message_ids]
        payload = [
            {
                "message_id": message.message_id,
                "sender": message.sender,
                "recipient": message.recipient,
                "kind": message.kind,
                "content": message.content,
                "control": message.control,
            }
            for message in messages
        ]
        return _sha256_text(_stable_json(payload))

    def message_by_id(self, message_id: str) -> MessageRecord:
        for message in self.messages:
            if message.message_id == message_id:
                return message
        raise KeyError(f"Unknown message_id: {message_id}")

    def inbox(self, *, recipient: str, question_id: int, control: bool | None = False) -> list[MessageRecord]:
        return [
            message
            for message in self.messages
            if message.recipient == recipient
            and message.question_id == question_id
            and (control is None or message.control is control)
        ]

    def add_control_message(
        self,
        *,
        split: SplitName,
        question_id: int,
        source_index: int,
        round_id: int,
        recipient: str,
        content: str,
        kind: str = "control",
    ) -> str:
        message_id = self._next_id("m")
        self.messages.append(
            MessageRecord(
                run_id=self.run_id,
                message_id=message_id,
                split=split,
                question_id=question_id,
                source_index=source_index,
                round=round_id,
                sender="Scheduler",
                recipient=recipient,
                kind=kind,
                content=content,
                source_message_id=None,
                created_by_activation_id=None,
                control=True,
            )
        )
        return message_id

    def record_activation(
        self,
        *,
        activation_id: str,
        split: SplitName,
        question_id: int,
        source_index: int,
        round_id: int,
        agent: str,
        stage: str,
        input_message_ids: list[str],
        control_message_ids: list[str],
        system_prompt: str,
        user_prompt: str,
        model_config: dict[str, Any],
        output_content: str,
        latency_ms: int,
        replayed: bool = False,
    ) -> str:
        source_message_id = self.source_id_for_activation(activation_id)
        self.activations.append(
            ActivationRecord(
                run_id=self.run_id,
                activation_id=activation_id,
                split=split,
                question_id=question_id,
                source_index=source_index,
                round=round_id,
                agent=agent,
                stage=stage,
                input_message_ids=list(input_message_ids),
                control_message_ids=list(control_message_ids),
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                prompt_hash=_sha256_text(system_prompt + "\n---\n" + user_prompt),
                model_config=dict(model_config),
                output_source_message_id=source_message_id,
                output_content=output_content,
                latency_ms=latency_ms,
                replayed=replayed,
            )
        )
        return source_message_id

    @staticmethod
    def _stage_for_kind(kind: MessageKind | str) -> str:
        return {
            "input": "input",
            "plan": "planner",
            "solver_step": "solver",
            "judge_feedback": "judger",
            "final_output": "judger",
        }.get(str(kind), "legacy")

    @staticmethod
    def _unique_ids(emissions: tuple[EdgeEmission, ...]) -> list[str]:
        return list(dict.fromkeys(message_id for emission in emissions for message_id in emission.state_message_ids))

    @staticmethod
    def _ordered_edge_specs(stage: str, emissions: tuple[EdgeEmission, ...]) -> list[tuple[EdgeEmission, str]]:
        specs = [(emission, recipient) for emission in emissions for recipient in emission.recipients]
        positions = {edge: index for index, edge in enumerate(_STAGE_EDGE_ORDER.get(stage, ()))}
        fallback = len(positions)
        return sorted(
            specs,
            key=lambda spec: (
                positions.get((spec[0].sender, spec[1]), fallback),
                spec[0].sender,
                spec[1],
            ),
        )

    def emit_stage_edges(
        self,
        *,
        split: SplitName,
        question_id: int,
        source_index: int,
        round_id: int,
        stage: str,
        emissions: tuple[EdgeEmission, ...],
    ) -> list[str]:
        if not emissions:
            return []

        stage_action_id = f"{split}:{source_index}:{question_id}:r{round_id}:{stage}"
        edge_specs = self._ordered_edge_specs(stage, emissions)
        stage_state_ids = self._unique_ids(emissions)
        candidate_state_hashes = [
            self.state_hash(emission.state_message_ids)
            if emission.state_message_ids
            else self.content_hash("[]")
            for emission, _recipient in edge_specs
        ]
        stage_state_hash = self.content_hash(
            _stable_json(
                {
                    "split": split,
                    "question_id": question_id,
                    "source_index": source_index,
                    "round": round_id,
                    "stage": stage,
                    "edges": [
                        {
                            "sender": emission.sender,
                            "recipient": recipient,
                            "kind": emission.kind,
                            "content": emission.content,
                            "local_state_hash": local_state_hash,
                        }
                        for (emission, recipient), local_state_hash in zip(edge_specs, candidate_state_hashes)
                    ],
                }
            )
        )

        candidates: list[EdgeCandidateRecord] = []
        for edge_index, ((emission, recipient), local_state_hash) in enumerate(
            zip(edge_specs, candidate_state_hashes)
        ):
            candidate = EdgeCandidateRecord(
                run_id=self.run_id,
                candidate_id=self._next_id("c"),
                split=split,
                question_id=question_id,
                source_index=source_index,
                round=round_id,
                source_message_id=emission.source_message_id,
                created_by_activation_id=emission.created_by_activation_id,
                sender=emission.sender,
                recipient=recipient,
                kind=emission.kind,
                original_content=emission.content,
                state_message_ids=list(emission.state_message_ids),
                state_hash=local_state_hash,
                stage=stage,
                stage_action_id=stage_action_id,
                stage_edge_index=edge_index,
                stage_state_hash=stage_state_hash,
                terminal=emission.terminal,
                termination_reason=emission.termination_reason,
            )
            candidates.append(candidate)
            self.edge_candidates.append(candidate)

        candidate_tuple = tuple(candidates)
        decide_stage = getattr(self.edge_policy, "decide_stage", None)
        if callable(decide_stage):
            live_results = tuple(decide_stage(candidate_tuple, stage_state_hash=stage_state_hash))
        else:
            live_results = tuple(self.edge_policy.decide(candidate) for candidate in candidates)
        if len(live_results) != len(candidates):
            raise ValueError(
                f"Edge policy {self.edge_policy.name} returned {len(live_results)} decisions "
                f"for {len(candidates)} candidates in {stage_action_id}"
            )

        resolved: list[tuple[EdgeCandidateRecord, EdgeEmission, str, bool, str, str, bool]] = []
        for candidate, (emission, _recipient), policy_result in zip(candidates, edge_specs, live_results):
            parent_decision = (
                self.replay_controller.replay_decision(candidate.candidate_id) if self.replay_controller else None
            )
            if parent_decision is not None:
                action = str(parent_decision.get("action", "keep"))
                dropped = bool(parent_decision.get("dropped", False))
                delivered_content = str(parent_decision.get("delivered_content", ""))
                policy_name = str(parent_decision.get("policy_name", "identity"))
                replayed_decision = True
            else:
                action = policy_result.action
                dropped = policy_result.dropped
                if dropped:
                    delivered_content = ""
                elif policy_result.delivered_content is None:
                    delivered_content = emission.content
                else:
                    delivered_content = policy_result.delivered_content
                policy_name = self.edge_policy.name
                replayed_decision = False
            resolved.append(
                (candidate, emission, action, dropped, delivered_content, policy_name, replayed_decision)
            )

        is_actionable = getattr(self.edge_policy, "is_actionable", None)
        action_candidates = (
            [candidate for candidate in candidates if is_actionable(candidate)]
            if callable(is_actionable)
            else list(candidates)
        )
        action_edge_indexes = {
            candidate.candidate_id: index for index, candidate in enumerate(action_candidates)
        }
        dropped_by_id = {candidate.candidate_id: dropped for candidate, _e, _a, dropped, _c, _p, _r in resolved}
        action_mask = "".join("1" if dropped_by_id[candidate.candidate_id] else "0" for candidate in action_candidates)
        policy_names = list(dict.fromkeys(policy_name for _c, _e, _a, _d, _dc, policy_name, _r in resolved))
        self.stage_actions.append(
            StageActionRecord(
                stage_action_id=stage_action_id,
                run_id=self.run_id,
                split=split,
                question_id=question_id,
                source_index=source_index,
                round=round_id,
                stage=stage,
                policy_name="+".join(policy_names),
                state_message_ids=stage_state_ids,
                state_hash=stage_state_hash,
                candidate_ids=[candidate.candidate_id for candidate in candidates],
                action_candidate_ids=[candidate.candidate_id for candidate in action_candidates],
                edge_order=[f"{candidate.sender}->{candidate.recipient}" for candidate in action_candidates],
                action_mask=action_mask,
                dropped_candidate_ids=[
                    candidate.candidate_id for candidate, _e, _a, dropped, _dc, _p, _r in resolved if dropped
                ],
                replayed_candidate_ids=[
                    candidate.candidate_id for candidate, _e, _a, _d, _dc, _p, replayed in resolved if replayed
                ],
            )
        )

        delivered_message_ids: list[str] = []
        for candidate, emission, action, dropped, delivered_content, policy_name, replayed_decision in resolved:
            action_edge_index = action_edge_indexes.get(candidate.candidate_id)
            delivered_message_id: str | None = None
            if not dropped:
                delivered_message_id = self._next_id("m")
                delivered_message_ids.append(delivered_message_id)
                self.messages.append(
                    MessageRecord(
                        run_id=self.run_id,
                        message_id=delivered_message_id,
                        split=split,
                        question_id=question_id,
                        source_index=source_index,
                        round=round_id,
                        sender=candidate.sender,
                        recipient=candidate.recipient,
                        kind=candidate.kind,
                        content=delivered_content,
                        source_message_id=candidate.source_message_id,
                        created_by_activation_id=candidate.created_by_activation_id,
                        control=False,
                        terminal=candidate.terminal,
                        termination_reason=candidate.termination_reason,
                    )
                )

            decision_id = self._next_id("d")
            self.edge_decisions.append(
                EdgeDecisionRecord(
                    run_id=self.run_id,
                    decision_id=decision_id,
                    candidate_id=candidate.candidate_id,
                    policy_name=policy_name,
                    action=action,
                    dropped=dropped,
                    delivered_message_id=delivered_message_id,
                    delivered_content=delivered_content,
                    content_hash=self.content_hash(delivered_content),
                    replayed=replayed_decision,
                    stage_action_id=stage_action_id,
                    action_mask=action_mask,
                    action_edge_index=action_edge_index,
                )
            )
            sample_id = self._next_id("s")
            self.rl_edge_samples.append(
                RLEdgeSampleRecord(
                    sample_id=sample_id,
                    run_id=self.run_id,
                    candidate_id=candidate.candidate_id,
                    question_id=question_id,
                    round=round_id,
                    sender=candidate.sender,
                    recipient=candidate.recipient,
                    message_kind=candidate.kind,
                    state_message_ids=list(candidate.state_message_ids),
                    state_hash=candidate.state_hash,
                    original_content=candidate.original_content,
                    action=action,
                    delivered_content=delivered_content,
                    dropped=dropped,
                    downstream_activation_ids=[],
                    terminal=candidate.terminal,
                    termination_reason=candidate.termination_reason,
                    reward=None,
                    stage=stage,
                    stage_action_id=stage_action_id,
                    stage_state_hash=stage_state_hash,
                    action_mask=action_mask,
                    action_edge_index=action_edge_index,
                )
            )
            if not dropped:
                self.records.append(
                    TraceRecord(
                        run_id=self.run_id,
                        split=split,
                        question_id=question_id,
                        source_index=source_index,
                        round=round_id,
                        edge_from=candidate.sender,
                        edge_to=candidate.recipient,
                        action="full_forward" if action == "keep" else action,
                        message_kind=candidate.kind,
                        content=delivered_content,
                        blocked=False,
                        compressed=action == "compress",
                        approx_input_tokens=approx_tokens(emission.input_content),
                        approx_output_tokens=approx_tokens(delivered_content),
                        latency_ms=emission.latency_ms,
                        step_index=emission.step_index,
                        terminal=candidate.terminal,
                        termination_reason=candidate.termination_reason,
                        message_id=delivered_message_id,
                        candidate_id=candidate.candidate_id,
                        decision_id=decision_id,
                        source_message_id=candidate.source_message_id,
                        activation_id=candidate.created_by_activation_id,
                        stage=stage,
                        stage_action_id=stage_action_id,
                        action_mask=action_mask,
                        action_edge_index=action_edge_index,
                    )
                )
        return delivered_message_ids

    def emit_edges(
        self,
        *,
        split: SplitName,
        question_id: int,
        source_index: int,
        round_id: int,
        sender: str,
        recipients: tuple[str, ...],
        kind: MessageKind | str,
        content: str,
        source_message_id: str,
        created_by_activation_id: str | None,
        state_message_ids: list[str],
        input_content: str = "",
        latency_ms: int = 0,
        step_index: int | None = None,
        terminal: bool = False,
        termination_reason: str | None = None,
        stage: str | None = None,
    ) -> list[str]:
        return self.emit_stage_edges(
            split=split,
            question_id=question_id,
            source_index=source_index,
            round_id=round_id,
            stage=stage or self._stage_for_kind(kind),
            emissions=(
                EdgeEmission(
                    sender=sender,
                    recipients=recipients,
                    kind=kind,
                    content=content,
                    source_message_id=source_message_id,
                    created_by_activation_id=created_by_activation_id,
                    state_message_ids=state_message_ids,
                    input_content=input_content,
                    latency_ms=latency_ms,
                    step_index=step_index,
                    terminal=terminal,
                    termination_reason=termination_reason,
                ),
            ),
        )

    def forward(
        self,
        *,
        split: SplitName,
        question_id: int,
        source_index: int,
        round_id: int,
        edge_from: str,
        edge_to: str,
        content: str,
        input_content: str = "",
        latency_ms: int = 0,
        message_kind: MessageKind | str = "message",
        step_index: int | None = None,
        terminal: bool = False,
        termination_reason: str | None = None,
    ) -> str:
        source_message_id = f"legacy_{self._next_id('src')}"
        self.emit_edges(
            split=split,
            question_id=question_id,
            source_index=source_index,
            round_id=round_id,
            sender=edge_from,
            recipients=(edge_to,),
            kind=message_kind,
            content=content,
            source_message_id=source_message_id,
            created_by_activation_id=None,
            state_message_ids=[],
            input_content=input_content,
            latency_ms=latency_ms,
            step_index=step_index,
            terminal=terminal,
            termination_reason=termination_reason,
        )
        return content

    def broadcast(
        self,
        *,
        split: SplitName,
        question_id: int,
        source_index: int,
        round_id: int,
        edge_from: str,
        edge_to: tuple[str, ...],
        content: str,
        input_content: str = "",
        latency_ms: int = 0,
        message_kind: MessageKind | str = "message",
        step_index: int | None = None,
    ) -> str:
        source_message_id = f"legacy_{self._next_id('src')}"
        self.emit_edges(
            split=split,
            question_id=question_id,
            source_index=source_index,
            round_id=round_id,
            sender=edge_from,
            recipients=edge_to,
            kind=message_kind,
            content=content,
            source_message_id=source_message_id,
            created_by_activation_id=None,
            state_message_ids=[],
            input_content=input_content,
            latency_ms=latency_ms,
            step_index=step_index,
        )
        return content

    def as_dicts(self) -> list[dict[str, Any]]:
        return [asdict(record) for record in self.records]

    def messages_as_dicts(self) -> list[dict[str, Any]]:
        return [asdict(record) for record in self.messages]

    def activations_as_dicts(self) -> list[dict[str, Any]]:
        return [asdict(record) for record in self.activations]

    def edge_candidates_as_dicts(self) -> list[dict[str, Any]]:
        return [asdict(record) for record in self.edge_candidates]

    def edge_decisions_as_dicts(self) -> list[dict[str, Any]]:
        return [asdict(record) for record in self.edge_decisions]

    def rl_edge_samples_as_dicts(self) -> list[dict[str, Any]]:
        return [asdict(record) for record in self.rl_edge_samples]

    def stage_actions_as_dicts(self) -> list[dict[str, Any]]:
        return [asdict(record) for record in self.stage_actions]


def now_ms() -> int:
    return int(time.perf_counter() * 1000)
