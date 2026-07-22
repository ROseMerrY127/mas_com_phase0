from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Literal, Protocol

from phase0.evaluation import approx_tokens


SplitName = Literal["train", "test"]
MessageKind = Literal["input", "plan", "solver_step", "judge_feedback", "final_output", "control", "message"]


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


@dataclass(frozen=True)
class EdgePolicyResult:
    action: str
    dropped: bool = False
    delivered_content: str | None = None


class EdgePolicy(Protocol):
    name: str

    def decide(self, candidate: EdgeCandidateRecord) -> EdgePolicyResult:
        ...


class IdentityEdgePolicy:
    name = "identity"

    def decide(self, candidate: EdgeCandidateRecord) -> EdgePolicyResult:
        return EdgePolicyResult(action="keep", dropped=False, delivered_content=candidate.original_content)


class DropEdgePolicy:
    """Small test hook for future pruning policies; production default remains identity."""

    def __init__(self, edges: set[tuple[str, str]], *, name: str = "drop_mock") -> None:
        self.edges = edges
        self.name = name

    def decide(self, candidate: EdgeCandidateRecord) -> EdgePolicyResult:
        if (candidate.sender, candidate.recipient) in self.edges:
            return EdgePolicyResult(action="drop", dropped=True, delivered_content="")
        return EdgePolicyResult(action="keep", dropped=False, delivered_content=candidate.original_content)


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
        self._candidate_ids = {str(row["candidate_id"]) for row in self.edge_candidates}
        if checkpoint_candidate_id not in self._candidate_ids:
            raise ValueError(f"checkpoint_candidate_id not found in parent run: {checkpoint_candidate_id}")
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
        if candidate_id == self.checkpoint_candidate_id:
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
    ) -> list[str]:
        delivered_message_ids: list[str] = []
        state_hash = self.state_hash(state_message_ids) if state_message_ids else self.content_hash("[]")
        for recipient in recipients:
            candidate_id = self._next_id("c")
            candidate = EdgeCandidateRecord(
                run_id=self.run_id,
                candidate_id=candidate_id,
                split=split,
                question_id=question_id,
                source_index=source_index,
                round=round_id,
                source_message_id=source_message_id,
                created_by_activation_id=created_by_activation_id,
                sender=sender,
                recipient=recipient,
                kind=kind,
                original_content=content,
                state_message_ids=list(state_message_ids),
                state_hash=state_hash,
                terminal=terminal,
                termination_reason=termination_reason,
            )
            self.edge_candidates.append(candidate)

            replayed_decision = False
            parent_decision = self.replay_controller.replay_decision(candidate_id) if self.replay_controller else None
            if parent_decision is not None:
                action = str(parent_decision.get("action", "keep"))
                dropped = bool(parent_decision.get("dropped", False))
                delivered_content = str(parent_decision.get("delivered_content", ""))
                policy_name = str(parent_decision.get("policy_name", "identity"))
                replayed_decision = True
            else:
                policy_result = self.edge_policy.decide(candidate)
                action = policy_result.action
                dropped = policy_result.dropped
                delivered_content = "" if dropped else policy_result.delivered_content or content
                policy_name = self.edge_policy.name

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
                        sender=sender,
                        recipient=recipient,
                        kind=kind,
                        content=delivered_content,
                        source_message_id=source_message_id,
                        created_by_activation_id=created_by_activation_id,
                        control=False,
                        terminal=terminal,
                        termination_reason=termination_reason,
                    )
                )

            decision_id = self._next_id("d")
            self.edge_decisions.append(
                EdgeDecisionRecord(
                    run_id=self.run_id,
                    decision_id=decision_id,
                    candidate_id=candidate_id,
                    policy_name=policy_name,
                    action=action,
                    dropped=dropped,
                    delivered_message_id=delivered_message_id,
                    delivered_content=delivered_content,
                    content_hash=self.content_hash(delivered_content),
                    replayed=replayed_decision,
                )
            )
            sample_id = self._next_id("s")
            self.rl_edge_samples.append(
                RLEdgeSampleRecord(
                    sample_id=sample_id,
                    run_id=self.run_id,
                    candidate_id=candidate_id,
                    question_id=question_id,
                    round=round_id,
                    sender=sender,
                    recipient=recipient,
                    message_kind=kind,
                    state_message_ids=list(state_message_ids),
                    state_hash=state_hash,
                    original_content=content,
                    action=action,
                    delivered_content=delivered_content,
                    dropped=dropped,
                    downstream_activation_ids=[],
                    terminal=terminal,
                    termination_reason=termination_reason,
                    reward=None,
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
                        edge_from=sender,
                        edge_to=recipient,
                        action="full_forward" if action == "keep" else action,
                        message_kind=kind,
                        content=delivered_content,
                        blocked=False,
                        compressed=action == "compress",
                        approx_input_tokens=approx_tokens(input_content),
                        approx_output_tokens=approx_tokens(delivered_content),
                        latency_ms=latency_ms,
                        step_index=step_index,
                        terminal=terminal,
                        termination_reason=termination_reason,
                        message_id=delivered_message_id,
                        candidate_id=candidate_id,
                        decision_id=decision_id,
                        source_message_id=source_message_id,
                        activation_id=created_by_activation_id,
                    )
                )
        return delivered_message_ids

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


def now_ms() -> int:
    return int(time.perf_counter() * 1000)