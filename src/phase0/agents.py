from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, RateLimitError

from .router import IdentityRouter


@dataclass(frozen=True)
class LLMResponse:
    content: str
    latency_ms: int


@dataclass(frozen=True)
class MajorityAgentResult:
    role: str
    content: str
    latency_ms: int


@dataclass(frozen=True)
class MajorityVoteResult:
    agents: tuple[MajorityAgentResult, ...]


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
        raise RuntimeError("OPENAI_API_KEY is required for Phase0 model calls")
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


async def run_single_agent(question: str, model: ChatModel) -> LLMResponse:
    system = "You solve grade-school math problems. Reason step by step, then end with 'Final answer: <number>'."
    user = f"Problem:\n{question}\n\nSolve carefully and provide the final numeric answer."
    return await model.complete(system=system, user=user, source="single_agent")


async def _agent_call(model: ChatModel, *, role: str, system: str, user: str) -> LLMResponse:
    return await model.complete(system=system, user=user, source=role)


async def run_four_agent_independent(question: str, model: ChatModel) -> MajorityVoteResult:
    """Run four independent solvers; the caller extracts answers and applies majority vote."""
    system = (
        "You solve grade-school math problems independently. Reason carefully, do not rely on other agents, "
        "and end with exactly 'Final answer: <number>'."
    )
    prompts = (
        (
            "IndependentAgent1",
            "Use straightforward arithmetic reasoning.",
        ),
        (
            "IndependentAgent2",
            "Decompose the problem into explicit subproblems before calculating.",
        ),
        (
            "IndependentAgent3",
            "Work backward or verify intermediate quantities where useful.",
        ),
        (
            "IndependentAgent4",
            "Solve with a concise equation-based approach.",
        ),
    )
    tasks = [
        _agent_call(
            model,
            role=role,
            system=system,
            user=f"Problem:\n{question}\n\n{instruction}\nEnd with 'Final answer: <number>'.",
        )
        for role, instruction in prompts
    ]
    responses = await asyncio.gather(*tasks)
    agents = tuple(
        MajorityAgentResult(role=role, content=response.content, latency_ms=response.latency_ms)
        for (role, _), response in zip(prompts, responses)
    )
    return MajorityVoteResult(agents=agents)
    
async def run_mas_identity(*, question_id: int, question: str, model: ChatModel, router: IdentityRouter) -> str:
    planner_user = f"Problem:\n{question}\n\nCreate a concise solving plan for two independent solvers."
    planner = await _agent_call(
        model,
        role="Planner",
        system="You are a planner for math-solving agents. Produce a short plan only.",
        user=planner_user,
    )
    plan = router.forward(
        question_id=question_id,
        round_id=1,
        edge_from="User",
        edge_to="Planner",
        input_content=question,
        content=planner.content,
        latency_ms=planner.latency_ms,
    )

    solver_a_prompt = f"Problem:\n{question}\n\nPlanner guidance:\n{plan}\n\nSolve directly with chain-of-thought. End with 'Final answer: <number>'."
    solver_b_prompt = f"Problem:\n{question}\n\nPlanner guidance:\n{plan}\n\nSolve by decomposing into explicit subproblems. End with 'Final answer: <number>'."

    solver_a_task = _agent_call(
        model,
        role="SolverA",
        system="You are SolverA. Use direct arithmetic reasoning and produce a final numeric answer.",
        user=solver_a_prompt,
    )
    solver_b_task = _agent_call(
        model,
        role="SolverB",
        system="You are SolverB. Use stepwise decomposition and produce a final numeric answer.",
        user=solver_b_prompt,
    )
    solver_a, solver_b = await asyncio.gather(solver_a_task, solver_b_task)

    sol_a = router.forward(
        question_id=question_id,
        round_id=2,
        edge_from="Planner",
        edge_to="SolverA",
        input_content=solver_a_prompt,
        content=solver_a.content,
        latency_ms=solver_a.latency_ms,
    )
    sol_b = router.forward(
        question_id=question_id,
        round_id=2,
        edge_from="Planner",
        edge_to="SolverB",
        input_content=solver_b_prompt,
        content=solver_b.content,
        latency_ms=solver_b.latency_ms,
    )

    verifier_prompt = (
        f"Problem:\n{question}\n\nSolution A:\n{sol_a}\n\nSolution B:\n{sol_b}\n\n"
        "Check both solutions for arithmetic or reasoning errors. State which answer is more reliable."
    )
    verifier = await _agent_call(
        model,
        role="StepVerifier",
        system="You are a careful math verifier. Compare solutions and identify errors.",
        user=verifier_prompt,
    )
    verification = router.forward(
        question_id=question_id,
        round_id=3,
        edge_from="SolverA,SolverB",
        edge_to="StepVerifier",
        input_content=verifier_prompt,
        content=verifier.content,
        latency_ms=verifier.latency_ms,
    )

    judge_prompt = (
        f"Problem:\n{question}\n\nSolution A:\n{sol_a}\n\nSolution B:\n{sol_b}\n\nVerifier:\n{verification}\n\n"
        "Choose the final answer. End with exactly 'Final answer: <number>'."
    )
    judge = await _agent_call(
        model,
        role="FinalJudge",
        system="You are the final judge. Return one final numeric answer after brief reasoning.",
        user=judge_prompt,
    )
    final = router.forward(
        question_id=question_id,
        round_id=4,
        edge_from="StepVerifier",
        edge_to="FinalJudge",
        input_content=judge_prompt,
        content=judge.content,
        latency_ms=judge.latency_ms,
    )
    return final


