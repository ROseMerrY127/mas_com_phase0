from __future__ import annotations

from decimal import Decimal, InvalidOperation
import re


_FINAL_ANSWER_RE = re.compile(r"final\s+answer\s*:", re.IGNORECASE)
_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def _strip_balanced_wrappers(text: str) -> str:
    value = text.strip()
    changed = True
    while changed:
        changed = False
        for left, right in (("$", "$"), ("\\(", "\\)"), ("\\[", "\\]")):
            if value.startswith(left) and value.endswith(right):
                value = value[len(left) : -len(right)].strip()
                changed = True
    return value


def _last_balanced_command_arg(text: str, command: str) -> str | None:
    marker = command + "{"
    start = text.rfind(marker)
    if start < 0:
        return None
    pos = start + len(marker)
    depth = 1
    chars: list[str] = []
    while pos < len(text):
        char = text[pos]
        if char == "{":
            depth += 1
            chars.append(char)
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars).strip()
            chars.append(char)
        else:
            chars.append(char)
        pos += 1
    return None


def extract_final_answer(text: str) -> str | None:
    """Extract the most likely final answer from MATH/PRM-style model output."""
    if not text or not text.strip():
        return None
    value = text.strip()

    boxed = _last_balanced_command_arg(value, "\\boxed") or _last_balanced_command_arg(value, "\\fbox")
    if boxed:
        return boxed

    lowered = value.lower()
    marker = "# answer"
    if marker in lowered:
        idx = lowered.rfind(marker)
        value = value[idx + len(marker) :]
    else:
        matches = list(_FINAL_ANSWER_RE.finditer(value))
        if matches:
            value = value[matches[-1].end() :]
        elif lowered.startswith("final:"):
            value = value.split(":", 1)[1]

    boxed = _last_balanced_command_arg(value, "\\boxed") or _last_balanced_command_arg(value, "\\fbox")
    if boxed:
        return boxed

    lines = [line.strip() for line in value.strip().splitlines() if line.strip()]
    if lines:
        value = lines[0]
    value = value.strip().strip(" .。,:;，；")
    return value or None


def normalize_answer(text: str | None) -> str:
    if text is None:
        return ""
    value = _strip_balanced_wrappers(text)
    boxed = _last_balanced_command_arg(value, "\\boxed") or _last_balanced_command_arg(value, "\\fbox")
    if boxed:
        value = boxed
    value = value.strip().strip(" .。,:;，；")
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    value = value.replace("\\,", "").replace("\\!", "")
    value = value.replace("\u2212", "-")
    value = re.sub(r"\s+", "", value)
    return value.lower()


def _to_decimal(text: str) -> Decimal | None:
    cleaned = text.replace("$", "").replace(",", "").strip()
    try:
        return Decimal(cleaned)
    except (InvalidOperation, ValueError):
        return None


def answer_correct(output: str, gold_answer: str) -> tuple[bool, str | None, str]:
    """Return approximate exact-match correctness after final-answer extraction."""
    prediction = extract_final_answer(output)
    pred_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(gold_answer)
    if not pred_norm or not gold_norm:
        return False, prediction, gold_norm
    if pred_norm == gold_norm:
        return True, prediction, gold_norm

    pred_num = _to_decimal(pred_norm)
    gold_num = _to_decimal(gold_norm)
    if pred_num is not None and gold_num is not None:
        return pred_num == gold_num, prediction, gold_norm
    return False, prediction, gold_norm