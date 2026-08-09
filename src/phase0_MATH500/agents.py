from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, RateLimitError

from .evaluation import approx_tokens
from .router import EdgeEmission, FullForwardRouter, MessageRecord, SplitName


@dataclass(frozen=True)
class LLMResponse:
    content: str
    latency_ms: int


@dataclass(frozen=True)
class StepwiseSolution:
    content: str
    steps: list[str]
    latency_ms: int
    approx_input_tokens: int


@dataclass(frozen=True)
class MASResult:
    content: str
    solver_a_steps: list[str]
    solver_b_steps: list[str]
    planner_messages: list[str]
    judger_messages: list[str]
    termination_reason: str
    rounds_used: int


class ChatModel:
    def __init__(
        self,
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        api_key: str,
        base_url: str | None,
        request_retries: int,
        retry_backoff_seconds: float,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_retries = request_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def complete(self, *, system: str, user: str, source: str) -> LLMResponse:
        for attempt in range(self.request_retries + 1):
            try:
                start = time.perf_counter()
                stream = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=True,
                )
                chunks: list[str] = []
                async for event in stream:
                    delta = event.choices[0].delta.content if event.choices else None
                    if delta:
                        chunks.append(delta)
                latency_ms = int((time.perf_counter() - start) * 1000)
                return LLMResponse(content="".join(chunks), latency_ms=latency_ms)
            except (APIConnectionError, APITimeoutError, RateLimitError, APIStatusError) as exc:
                if not _should_retry(exc) or attempt >= self.request_retries:
                    raise
                delay = self.retry_backoff_seconds * (2**attempt)
                print(f"{source} request failed ({exc!r}); retrying in {delay:.1f}s", flush=True)
                await asyncio.sleep(delay)

        raise RuntimeError("unreachable retry state")

    async def close(self) -> None:
        await self.client.close()


def _should_retry(exc: Exception) -> bool:
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in {408, 409, 429, 500, 502, 503, 504}
    return False


def build_model(
    *,
    model: str,
    temperature: float,
    max_tokens: int,
    request_retries: int = 3,
    retry_backoff_seconds: float = 2.0,
) -> ChatModel:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for Phase0 MATH500 model calls")
    base_url = os.getenv("OPENAI_BASE_URL")
    model_name = os.getenv("OPENAI_MODEL") or model
    return ChatModel(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
        base_url=base_url,
        request_retries=max(0, request_retries),
        retry_backoff_seconds=max(0.0, retry_backoff_seconds),
    )


STEP_SYSTEM = (
    "You solve math problems in PRM800K step format. Return exactly one next step. "
    "Do not number the step. Do not repeat previous steps. If the solution is complete, "
    "return exactly '# Answer' followed by a blank line and the answer."
)


def _format_steps(steps: list[str]) -> str:
    if not steps:
        return "<none>"
    return "\n".join(f"{idx}. {step}" for idx, step in enumerate(steps, start=1))


def _latest(messages: list[str]) -> str:
    return messages[-1] if messages else "<none>"


def _is_final_text(content: str) -> bool:
    lowered = content.strip().lower()
    return "# answer" in lowered or "final answer:" in lowered


def _is_judge_final(content: str) -> bool:
    stripped = content.strip()
    upper = stripped.upper()
    return upper.startswith("FINAL:") or stripped.startswith("# Answer") or "Final answer:" in stripped


def _coerce_final(content: str) -> str:
    stripped = content.strip()
    if _is_judge_final(stripped):
        return stripped
    return f"FINAL:\n{stripped}"


def _join_steps(steps: list[str]) -> str:
    return "\n\n".join(step.strip() for step in steps if step.strip())


def _build_step_user(
    *,
    problem: str,
    previous_steps: list[str],
    guidance: str | None = None,
    feedback: str | None = None,
    force_final: bool = False,
) -> str:
    sections = [f"Problem:\n{problem}"]
    if guidance:
        sections.append(f"Planner guidance:\n{guidance}")
    if feedback:
        sections.append(f"Latest judger feedback:\n{feedback}")
    sections.append(f"Previous steps:\n{_format_steps(previous_steps)}")
    if force_final:
        sections.append(
            "This is the final allowed step. Output the final answer now using exactly:\n# Answer\n\n<answer>"
        )
    else:
        sections.append(
            "Output exactly one next PRM800K-style step. If the answer is ready, output only:\n# Answer\n\n<answer>"
        )
    return "\n\n".join(sections)


def build_single_agent_prompt(problem: str) -> tuple[str, str]:
    return STEP_SYSTEM, _build_step_user(problem=problem, previous_steps=[])


async def _agent_call(model: ChatModel, *, role: str, system: str, user: str) -> LLMResponse:
    return await model.complete(system=system, user=user, source=role)


async def run_stepwise_agent(
    problem: str,
    model: ChatModel,
    *,
    role: str,
    guidance: str | None = None,
    feedback: str | None = None,
    max_steps: int = 8,
) -> StepwiseSolution:
    steps: list[str] = []
    latency_ms = 0
    approx_input_tokens = 0
    total_steps = max(1, max_steps)
    for step_index in range(1, total_steps + 1):
        force_final = step_index == total_steps
        user = _build_step_user(
            problem=problem,
            previous_steps=steps,
            guidance=guidance,
            feedback=feedback,
            force_final=force_final,
        )
        approx_input_tokens += approx_tokens(STEP_SYSTEM) + approx_tokens(user)
        response = await _agent_call(
            model,
            role=f"{role}_step_{step_index}",
            system=STEP_SYSTEM,
            user=user,
        )
        step = response.content.strip()
        steps.append(step)
        latency_ms += response.latency_ms
        if _is_final_text(step):
            break
    return StepwiseSolution(
        content=_join_steps(steps),
        steps=steps,
        latency_ms=latency_ms,
        approx_input_tokens=approx_input_tokens,
    )


async def run_single_agent(problem: str, model: ChatModel, *, max_steps: int = 8) -> StepwiseSolution:
    return await run_stepwise_agent(problem, model, role="single_agent", max_steps=max_steps)


def _model_config(model: Any) -> dict[str, Any]:
    return {
        "model": getattr(model, "model", model.__class__.__name__),
        "temperature": getattr(model, "temperature", None),
        "max_tokens": getattr(model, "max_tokens", None),
    }


def _format_message_block(title: str, messages: list[MessageRecord]) -> str:
    if not messages:
        return f"{title}:\n<none>"
    lines = [f"{title}:"]
    for message in messages:
        lines.append(f"[{message.message_id}] {message.sender} -> {message.recipient} ({message.kind})")
        lines.append(message.content)
    return "\n".join(lines)


def _control_content(
    *,
    round_id: int,
    stage: str,
    max_rounds: int,
    stall_rounds: int,
    force_final: bool,
    forced_reason: str | None,
) -> str:
    return json.dumps(
        {
            "round": round_id,
            "stage": stage,
            "max_rounds": max_rounds,
            "stall_rounds": stall_rounds,
            "force_final": force_final,
            "forced_reason": forced_reason,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _build_markov_user(*, agent: str, task: str, inbox: list[MessageRecord], controls: list[MessageRecord]) -> str:
    return "\n\n".join(
        [
            "Use only the delivered inbox messages and scheduler control messages shown below.",
            _format_message_block("Scheduler control messages", controls),
            _format_message_block("Delivered inbox messages", inbox),
            task,
        ]
    )


def _build_planner_user(*, inbox: list[MessageRecord], controls: list[MessageRecord]) -> str:
    return _build_markov_user(
        agent="Planner",
        inbox=inbox,
        controls=controls,
        task=(
            "You are Planner. Create or revise a concise plan for SolverA and SolverB. "
            "If an Input message contains the original problem, preserve the necessary facts in your plan. "
            "Return the plan only."
        ),
    )


def _build_solver_user(*, role: str, inbox: list[MessageRecord], controls: list[MessageRecord]) -> str:
    style = "Use direct mathematical reasoning." if role == "SolverA" else "Use stepwise decomposition."
    return _build_markov_user(
        agent=role,
        inbox=inbox,
        controls=controls,
        task=(
            f"You are {role}. {style} Output exactly one next PRM800K-style step. "
            "Do not communicate with the other solver directly. If scheduler control says force_final=true, "
            "output the final answer now using exactly '# Answer' followed by a blank line and the answer. "
            "Otherwise, if the answer is ready, output only '# Answer' followed by the answer."
        ),
    )


def _build_judger_user(*, inbox: list[MessageRecord], controls: list[MessageRecord]) -> str:
    return _build_markov_user(
        agent="Judger",
        inbox=inbox,
        controls=controls,
        task=(
            "You are Judger. If the solution is complete or scheduler control says force_final=true, "
            "output 'FINAL:' followed by '# Answer' and the best answer. Otherwise output 'FEEDBACK:' "
            "followed by a brief critique or requested revision."
        ),
    )


async def _complete_activation(
    *,
    model: ChatModel,
    router: FullForwardRouter,
    activation_id: str,
    source: str,
    system: str,
    user: str,
) -> tuple[str, int, bool]:
    replayed_output = router.replay_activation_output(activation_id)
    if replayed_output is not None:
        return replayed_output.strip(), 0, True
    response = await _agent_call(model, role=source, system=system, user=user)
    return response.content.strip(), response.latency_ms, False


def _ids(messages: list[MessageRecord]) -> list[str]:
    return [message.message_id for message in messages]


def _progress_signature(solver_a_steps: list[str], solver_b_steps: list[str], judger_messages: list[str]) -> tuple[str, str, str]:
    return (
        _latest(solver_a_steps).strip().lower(),
        _latest(solver_b_steps).strip().lower(),
        _latest(judger_messages).strip().lower(),
    )


async def run_mas_full_path(
    *,
    split: SplitName,
    question_id: int,
    source_index: int,
    problem: str,
    model: ChatModel,
    router: FullForwardRouter,
    max_rounds: int = 4,
    stall_rounds: int = 2,
    force_final_on_stop: bool = True,
) -> MASResult:
    max_rounds = max(1, max_rounds)
    stall_rounds = max(0, stall_rounds)

    router.emit_edges(
        split=split,
        question_id=question_id,
        source_index=source_index,
        round_id=0,
        sender="Input",
        recipients=("Planner",),
        kind="input",
        content=problem,
        source_message_id=f"src_input_{split}_{question_id}_{source_index}",
        created_by_activation_id=None,
        state_message_ids=[],
        stage="input",
    )

    planner_messages: list[str] = []
    judger_messages: list[str] = []
    solver_a_steps: list[str] = []
    solver_b_steps: list[str] = []
    termination_reason = "max_rounds"
    final_output = ""
    rounds_used = 0
    stalled_round_count = 0
    previous_signature: tuple[str, str, str] | None = None
    model_config = _model_config(model)

    for round_id in range(1, max_rounds + 1):
        rounds_used = round_id
        forced_reason: str | None = None
        if force_final_on_stop and stall_rounds > 0 and stalled_round_count >= stall_rounds:
            forced_reason = "stall_rounds"
        elif force_final_on_stop and round_id == max_rounds:
            forced_reason = "max_rounds"
        force_final = forced_reason is not None

        planner_control_id = router.add_control_message(
            split=split,
            question_id=question_id,
            source_index=source_index,
            round_id=round_id,
            recipient="Planner",
            content=_control_content(
                round_id=round_id,
                stage="planner",
                max_rounds=max_rounds,
                stall_rounds=stall_rounds,
                force_final=force_final,
                forced_reason=forced_reason,
            ),
        )
        planner_inbox = router.inbox(recipient="Planner", question_id=question_id, control=False)
        planner_controls = [router.message_by_id(planner_control_id)]
        planner_user = _build_planner_user(inbox=planner_inbox, controls=planner_controls)
        planner_activation_id = router.next_activation_id()
        plan, planner_latency, planner_replayed = await _complete_activation(
            model=model,
            router=router,
            activation_id=planner_activation_id,
            source=f"Planner_round_{round_id}",
            system="You are a planner for math-solving agents. Produce a short plan only.",
            user=planner_user,
        )
        planner_source_id = router.record_activation(
            activation_id=planner_activation_id,
            split=split,
            question_id=question_id,
            source_index=source_index,
            round_id=round_id,
            agent="Planner",
            stage="planner",
            input_message_ids=_ids(planner_inbox),
            control_message_ids=_ids(planner_controls),
            system_prompt="You are a planner for math-solving agents. Produce a short plan only.",
            user_prompt=planner_user,
            model_config=model_config,
            output_content=plan,
            latency_ms=planner_latency,
            replayed=planner_replayed,
        )
        planner_messages.append(plan)
        planner_state_ids = _ids(planner_inbox) + _ids(planner_controls)
        router.emit_edges(
            split=split,
            question_id=question_id,
            source_index=source_index,
            round_id=round_id,
            sender="Planner",
            recipients=("SolverA", "SolverB", "Judger"),
            kind="plan",
            content=plan,
            source_message_id=planner_source_id,
            created_by_activation_id=planner_activation_id,
            state_message_ids=planner_state_ids,
            input_content=planner_user,
            latency_ms=planner_latency,
            stage="planner",
        )

        async def run_solver(role: str) -> tuple[str, str, str, int, bool, list[str], str]:
            control_id = router.add_control_message(
                split=split,
                question_id=question_id,
                source_index=source_index,
                round_id=round_id,
                recipient=role,
                content=_control_content(
                    round_id=round_id,
                    stage=role,
                    max_rounds=max_rounds,
                    stall_rounds=stall_rounds,
                    force_final=force_final,
                    forced_reason=forced_reason,
                ),
            )
            inbox = router.inbox(recipient=role, question_id=question_id, control=False)
            controls = [router.message_by_id(control_id)]
            user = _build_solver_user(role=role, inbox=inbox, controls=controls)
            activation_id = router.next_activation_id()
            content, latency_ms, replayed = await _complete_activation(
                model=model,
                router=router,
                activation_id=activation_id,
                source=f"{role}_round_{round_id}",
                system=STEP_SYSTEM,
                user=user,
            )
            source_id = router.record_activation(
                activation_id=activation_id,
                split=split,
                question_id=question_id,
                source_index=source_index,
                round_id=round_id,
                agent=role,
                stage="solver",
                input_message_ids=_ids(inbox),
                control_message_ids=_ids(controls),
                system_prompt=STEP_SYSTEM,
                user_prompt=user,
                model_config=model_config,
                output_content=content,
                latency_ms=latency_ms,
                replayed=replayed,
            )
            return role, content, source_id, latency_ms, replayed, _ids(inbox) + _ids(controls), user

        solver_a_result, solver_b_result = await asyncio.gather(run_solver("SolverA"), run_solver("SolverB"))
        solver_emissions: list[EdgeEmission] = []
        for role, step, source_id, latency_ms, _replayed, state_ids, user in (solver_a_result, solver_b_result):
            if role == "SolverA":
                solver_a_steps.append(step)
            else:
                solver_b_steps.append(step)
            solver_emissions.append(
                EdgeEmission(
                    sender=role,
                    recipients=("Planner", "Judger", role),
                    kind="solver_step",
                    content=step,
                    source_message_id=source_id,
                    created_by_activation_id=source_id.removeprefix("src_"),
                    state_message_ids=state_ids,
                    input_content=user,
                    latency_ms=latency_ms,
                    step_index=len(solver_a_steps) if role == "SolverA" else len(solver_b_steps),
                )
            )
        router.emit_stage_edges(
            split=split,
            question_id=question_id,
            source_index=source_index,
            round_id=round_id,
            stage="solver",
            emissions=tuple(solver_emissions),
        )

        judger_control_id = router.add_control_message(
            split=split,
            question_id=question_id,
            source_index=source_index,
            round_id=round_id,
            recipient="Judger",
            content=_control_content(
                round_id=round_id,
                stage="judger",
                max_rounds=max_rounds,
                stall_rounds=stall_rounds,
                force_final=force_final,
                forced_reason=forced_reason,
            ),
        )
        judger_inbox = router.inbox(recipient="Judger", question_id=question_id, control=False)
        judger_controls = [router.message_by_id(judger_control_id)]
        judger_user = _build_judger_user(inbox=judger_inbox, controls=judger_controls)
        judger_activation_id = router.next_activation_id()
        judge_message, judger_latency, judger_replayed = await _complete_activation(
            model=model,
            router=router,
            activation_id=judger_activation_id,
            source=f"Judger_round_{round_id}",
            system="You are the final judge. Return FINAL when ready, otherwise return FEEDBACK.",
            user=judger_user,
        )
        is_final = _is_judge_final(judge_message)
        if force_final and not is_final:
            judge_message = _coerce_final(judge_message)
            is_final = True
        judger_source_id = router.record_activation(
            activation_id=judger_activation_id,
            split=split,
            question_id=question_id,
            source_index=source_index,
            round_id=round_id,
            agent="Judger",
            stage="judger",
            input_message_ids=_ids(judger_inbox),
            control_message_ids=_ids(judger_controls),
            system_prompt="You are the final judge. Return FINAL when ready, otherwise return FEEDBACK.",
            user_prompt=judger_user,
            model_config=model_config,
            output_content=judge_message,
            latency_ms=judger_latency,
            replayed=judger_replayed,
        )
        judger_messages.append(judge_message)
        judger_state_ids = _ids(judger_inbox) + _ids(judger_controls)

        if is_final:
            termination_reason = forced_reason or "judge_final"
            final_output = judge_message
            router.emit_edges(
                split=split,
                question_id=question_id,
                source_index=source_index,
                round_id=round_id,
                sender="Judger",
                recipients=("Output",),
                kind="final_output",
                content=judge_message,
                source_message_id=judger_source_id,
                created_by_activation_id=judger_activation_id,
                state_message_ids=judger_state_ids,
                input_content=judger_user,
                latency_ms=judger_latency,
                terminal=True,
                termination_reason=termination_reason,
                stage="judger",
            )
            break

        router.emit_edges(
            split=split,
            question_id=question_id,
            source_index=source_index,
            round_id=round_id,
            sender="Judger",
            recipients=("Planner", "SolverA", "SolverB", "Judger"),
            kind="judge_feedback",
            content=judge_message,
            source_message_id=judger_source_id,
            created_by_activation_id=judger_activation_id,
            state_message_ids=judger_state_ids,
            input_content=judger_user,
            latency_ms=judger_latency,
            stage="judger",
        )
        current_signature = _progress_signature(solver_a_steps, solver_b_steps, judger_messages)
        stalled_round_count = stalled_round_count + 1 if current_signature == previous_signature else 0
        previous_signature = current_signature
    else:
        if force_final_on_stop and judger_messages:
            final_output = _coerce_final(judger_messages[-1])
        else:
            final_output = judger_messages[-1] if judger_messages else _join_steps(solver_a_steps + solver_b_steps)

    return MASResult(
        content=final_output,
        solver_a_steps=solver_a_steps,
        solver_b_steps=solver_b_steps,
        planner_messages=planner_messages,
        judger_messages=judger_messages,
        termination_reason=termination_reason,
        rounds_used=rounds_used,
    )
