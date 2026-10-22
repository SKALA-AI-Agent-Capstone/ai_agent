"""Bound model inputs before API calls without sending another summarization request.

Text counts use the o200k tokenizer. Message framing, JSON schemas and image patches
are not represented exactly by local tokenization, so every budget keeps headroom.
"""

from copy import deepcopy
from functools import lru_cache
import json
import logging
import os

import tiktoken


logger = logging.getLogger(__name__)
MARKER = "\n[중간 내용은 입력 예산을 위해 생략됨]\n"
PARSE_INPUT_BUDGETS = {
    "TextAssessment": 12000,
    "EvidenceSummary": 6000,
    "ClaimMatches": 8000,
    "AdCopy": 12000,
    "IngredientMentions": 5000,
    "RewriteDraft": 5000,
    "RewriteVerification": 5000,
    "SeverityAssessment": 5000,
}
PARSE_OUTPUT_BUDGETS = {
    "TextAssessment": 4500,
    "EvidenceSummary": 1200,
    "ClaimMatches": 3000,
    "AdCopy": 4000,
    "IngredientMentions": 1200,
    "RewriteDraft": 1000,
    "RewriteVerification": 1000,
    "SeverityAssessment": 1000,
}
IMAGE_RESERVE = 6000
MESSAGE_OVERHEAD = 256
TOOL_INPUT_BUDGET = 6000
TOOL_OUTPUT_BUDGET = 600


class ContextBudgetExceeded(ValueError):
    """The fixed instructions or schema cannot fit; do not silently omit them."""


def tokens(value: str) -> int:
    encoding = _encoding()
    # A BPE token contains at least one UTF-8 byte. This overestimates text input
    # rather than allowing an unbounded request when the tokenizer file is offline.
    return len(encoding.encode(value, disallowed_special=())) if encoding else len(value.encode("utf-8"))


@lru_cache(maxsize=1)
def _encoding():
    try:
        return tiktoken.get_encoding("o200k_base")
    except Exception as error:
        logger.warning("Tokenizer vocabulary unavailable (%s); using UTF-8 byte upper bound", type(error).__name__)
        return None


def encoded_tokens(value) -> int:
    return tokens(json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str))


def _input_cap(default: int) -> int:
    configured = int(os.getenv("MODEL_INPUT_TOKEN_BUDGET", "12000"))
    if configured < 1000:
        raise ValueError("MODEL_INPUT_TOKEN_BUDGET must be at least 1000")
    return min(default, configured)


def shorten_middle(value: str, limit: int) -> str:
    """Keep the start (identifiers/instructions) and the latest tail, dropping the middle."""
    if tokens(value) <= limit:
        return value
    marker_cost = tokens(MARKER)
    if limit <= marker_cost + 2:
        raise ContextBudgetExceeded("MODEL_INPUT_BUDGET_TOO_SMALL")
    keep = limit - marker_cost
    front_budget = max(1, keep // 2)
    back_budget = keep - front_budget

    def fit(part, budget, from_end=False):
        low, high = 0, len(part)
        while low < high:
            mid = (low + high + 1) // 2
            candidate = part[-mid:] if from_end else part[:mid]
            if tokens(candidate) <= budget:
                low = mid
            else:
                high = mid - 1
        return part[-low:] if from_end and low else part[:low]

    shortened = fit(value, front_budget) + MARKER + fit(value, back_budget, from_end=True)
    # Boundary retokenization can cost a few extra tokens. Reduce the front until it fits.
    while tokens(shortened) > limit:
        before, after = shortened.split(MARKER, 1)
        if not before:
            raise ContextBudgetExceeded("MODEL_INPUT_BUDGET_TOO_SMALL")
        shortened = before[:-1] + MARKER + after
    return shortened


def prepare_parse(prompt: str, text: str, schema, *, image: bool = False) -> tuple[str, int, bool]:
    """Return bounded user text, output cap, and whether text was compacted."""
    name = schema.__name__
    cap = _input_cap(PARSE_INPUT_BUDGETS.get(name, 6000))
    schema_cost = encoded_tokens(schema.model_json_schema())
    fixed = tokens(prompt) + schema_cost + MESSAGE_OVERHEAD + (IMAGE_RESERVE if image else 0)
    available = cap - fixed
    if available < 64:
        raise ContextBudgetExceeded(f"MODEL_INPUT_BUDGET_TOO_SMALL:{name}")
    bounded = shorten_middle(text, available)
    return bounded, PARSE_OUTPUT_BUDGETS.get(name, 2000), bounded != text


def split_parse_text(prompt: str, text: str, schema, *, overlap_chars: int = 160) -> list[tuple[int, str]]:
    """Cover all text with bounded, overlapping parse inputs and original offsets."""
    cap = _input_cap(PARSE_INPUT_BUDGETS.get(schema.__name__, 6000))
    available = cap - tokens(prompt) - encoded_tokens(schema.model_json_schema()) - MESSAGE_OVERHEAD
    if available < 64:
        raise ContextBudgetExceeded(f"MODEL_INPUT_BUDGET_TOO_SMALL:{schema.__name__}")
    chunks = []
    start = 0
    while start < len(text):
        low, high = start + 1, len(text)
        while low < high:
            middle = (low + high + 1) // 2
            if tokens(text[start:middle]) <= available:
                low = middle
            else:
                high = middle - 1
        end = low
        if end < len(text):
            boundary = text.rfind("\n", start + (end - start) * 3 // 4, end)
            if boundary > start:
                end = boundary + 1
        chunk = text[start:end]
        if tokens(chunk) > available:
            raise ContextBudgetExceeded("TEXT_CHUNK_OVER_BUDGET")
        chunks.append((start, chunk))
        if end == len(text):
            break
        start = max(start + 1, end - min(overlap_chars, (end - start) // 4))
    return chunks


def _tool_status(content: str) -> str:
    try:
        value = json.loads(content)
    except (TypeError, ValueError):
        return "result received"
    if isinstance(value, dict):
        return str(value.get("status") or value.get("error") or "result received")[:80]
    return "result received"


def _tool_summary(group: list[dict]) -> str:
    calls = next((m.get("tool_calls", []) for m in group if m.get("role") == "assistant"), [])
    names = [c.get("function", {}).get("name", "tool") for c in calls]
    statuses = [_tool_status(m.get("content", "")) for m in group if m.get("role") == "tool"]
    return f"{', '.join(names) or 'tool'}: {', '.join(statuses) or 'no result'}"


def _cost(messages: list[dict], tools: list[dict]) -> int:
    return encoded_tokens(messages) + encoded_tokens(tools) + MESSAGE_OVERHEAD


def prepare_tool_messages(messages: list[dict], tools: list[dict]) -> tuple[list[dict], bool]:
    """Keep system/current request and recent tool result; summarize older complete turns.

    Assistant tool calls and their tool responses are removed together to preserve the
    Chat Completions message protocol. The input list is never mutated.
    """
    cap = _input_cap(TOOL_INPUT_BUDGET)
    prepared = deepcopy(messages)
    if _cost(prepared, tools) <= cap:
        return prepared, False
    if not prepared or prepared[0].get("role") != "system":
        raise ContextBudgetExceeded("TOOL_SYSTEM_MESSAGE_MISSING")

    groups = []
    index = 0
    while index < len(prepared):
        if prepared[index].get("role") == "assistant" and prepared[index].get("tool_calls"):
            end = index + 1
            while end < len(prepared) and prepared[end].get("role") == "tool":
                end += 1
            groups.append((index, end))
            index = end
        else:
            index += 1

    # A compact factual trace replaces old assistant/tool pairs. Keep the newest pair verbatim.
    summaries = []
    for start, end in reversed(groups[:-1]):
        summaries.append(_tool_summary(prepared[start:end]))
        del prepared[start:end]
    if summaries:
        summary = "이전 도구 호출 요약 (상세 응답 생략): " + "; ".join(reversed(summaries))
        prepared.insert(2, {"role": "user", "content": shorten_middle(summary, 400)})
    if _cost(prepared, tools) <= cap:
        return prepared, True

    # The newest result is retained unless it alone would prevent any request.
    for message in prepared:
        if message.get("role") == "tool" and isinstance(message.get("content"), str):
            message["content"] = shorten_middle(message["content"], 400)
    if _cost(prepared, tools) <= cap:
        return prepared, True

    users = [m for m in prepared if m.get("role") == "user" and isinstance(m.get("content"), str)]
    if not users:
        raise ContextBudgetExceeded("TOOL_USER_MESSAGE_MISSING")
    target = users[0]
    available = cap - (_cost(prepared, tools) - tokens(target["content"]))
    if available < 64:
        raise ContextBudgetExceeded("TOOL_INPUT_BUDGET_TOO_SMALL")
    target["content"] = shorten_middle(target["content"], available)
    if _cost(prepared, tools) > cap:
        raise ContextBudgetExceeded("TOOL_INPUT_BUDGET_TOO_SMALL")
    return prepared, True
