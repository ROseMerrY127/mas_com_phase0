from __future__ import annotations

from decimal import Decimal, InvalidOperation
import re

_NUMBER_RE = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?")
_GOLD_RE = re.compile(r"####\s*([-+]?\$?\d[\d,]*(?:\.\d+)?)")


def _normalize_number(text: str) -> Decimal | None:
    cleaned = text.strip().replace("$", "").replace(",", "")
    try:
        return Decimal(cleaned)
    except (InvalidOperation, ValueError):
        return None


def extract_gold_answer(answer: str) -> Decimal | None:
    match = _GOLD_RE.search(answer)
    if match:
        return _normalize_number(match.group(1))
    return extract_last_number(answer)


def extract_last_number(text: str) -> Decimal | None:
    matches = _NUMBER_RE.findall(text)
    if not matches:
        return None
    return _normalize_number(matches[-1])


def is_correct(prediction: Decimal | None, gold: Decimal | None) -> bool:
    return prediction is not None and gold is not None and prediction == gold


def approx_tokens(text: str) -> int:
    # Cheap tokenizer-independent estimate for accounting in Phase0.
    return max(1, round(len(text) / 4)) if text else 0


def heuristic_step_reward(content: str, final_correct: bool | None = None) -> float:
    score = 0.15
    if extract_last_number(content) is not None:
        score += 0.25
    if any(mark in content for mark in ("=", "+", "-", "*", "/", "therefore", "Thus", "So")):
        score += 0.2
    if len(content.strip()) >= 40:
        score += 0.1
    if final_correct is True:
        score += 0.25
    elif final_correct is False:
        score -= 0.1
    return max(0.0, min(1.0, score))
