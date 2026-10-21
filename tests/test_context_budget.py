"""The Agent must bound prompt context before calling a model."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

import capstone_agent.context_budget as budget
from capstone_agent.review_engine import Provider


class SmallResult(BaseModel):
    value: str


def test_parse_budget_keeps_beginning_and_end_without_network(monkeypatch):
    monkeypatch.setattr(budget, "_encoding", lambda: None)
    monkeypatch.setenv("MODEL_INPUT_TOKEN_BUDGET", "1200")
    source = "제품명과 광고 식별자\n" + "오래된 광고 설명 " * 800 + "\n마지막 시험 조건"
    bounded, output_cap, changed = budget.prepare_parse("분석 지침", source, SmallResult)
    fixed = budget.tokens("분석 지침") + budget.encoded_tokens(SmallResult.model_json_schema()) + budget.MESSAGE_OVERHEAD
    assert changed and output_cap == 2000
    assert bounded.startswith("제품명과 광고 식별자") and bounded.endswith("마지막 시험 조건")
    assert budget.MARKER in bounded and budget.tokens(bounded) + fixed <= 1200


def test_fixed_prompt_cannot_be_silently_cut(monkeypatch):
    monkeypatch.setattr(budget, "_encoding", lambda: None)
    monkeypatch.setenv("MODEL_INPUT_TOKEN_BUDGET", "1000")
    with pytest.raises(budget.ContextBudgetExceeded, match="MODEL_INPUT_BUDGET_TOO_SMALL"):
        budget.prepare_parse("필수 지침" * 1000, "광고", SmallResult)


def test_split_parse_text_covers_middle_and_boundaries_without_truncation(monkeypatch):
    monkeypatch.setattr(budget, "_encoding", lambda: None)
    monkeypatch.setenv("MODEL_INPUT_TOKEN_BUDGET", "1200")
    source = "처음 문구\n" + "일반 광고 문구\n" * 400 + "중간 문제 문구\n" + "일반 광고 문구\n" * 400 + "마지막 문구"
    chunks = budget.split_parse_text("분석 지침", source, SmallResult)
    assert len(chunks) > 2
    assert chunks[0][0] == 0 and chunks[-1][0] + len(chunks[-1][1]) == len(source)
    assert any("중간 문제 문구" in chunk for _, chunk in chunks)
    assert all(source[start:start + len(chunk)] == chunk for start, chunk in chunks)
    assert all(start <= previous_start + len(previous_chunk) for (previous_start, previous_chunk), (start, _) in zip(chunks, chunks[1:]))
    fixed = budget.tokens("분석 지침") + budget.encoded_tokens(SmallResult.model_json_schema()) + budget.MESSAGE_OVERHEAD
    assert all(budget.tokens(chunk) + fixed <= 1200 for _, chunk in chunks)


def test_old_tool_turns_are_summarized_as_complete_pairs(monkeypatch):
    monkeypatch.setattr(budget, "_encoding", lambda: None)
    monkeypatch.setenv("MODEL_INPUT_TOKEN_BUDGET", "1200")
    messages = [{"role": "system", "content": "필수 선택 지침"},
                {"role": "user", "content": "현재 광고의 확인 대상"}]
    for index in range(3):
        messages += [
            {"role": "assistant", "content": "", "tool_calls": [{"id": f"call-{index}", "type": "function",
                "function": {"name": f"lookup_{index}", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": f"call-{index}",
             "content": '{"status":"found","detail":"' + ("자료" * 600) + '"}'},
        ]
    original = deepcopy(messages)
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]
    bounded, changed = budget.prepare_tool_messages(messages, tools)
    assert changed and messages == original
    assert bounded[0] == messages[0] and bounded[1] == messages[1]
    assert "이전 도구 호출 요약" in bounded[2]["content"]
    assert "lookup_0" in bounded[2]["content"] and "lookup_1" in bounded[2]["content"]
    assert [m.get("tool_call_id") for m in bounded if m["role"] == "tool"] == ["call-2"]
    assert budget._cost(bounded, tools) <= 1200


async def test_provider_enforces_budget_at_model_boundary(monkeypatch):
    monkeypatch.setattr(budget, "_encoding", lambda: None)
    monkeypatch.setenv("MODEL_INPUT_TOKEN_BUDGET", "1200")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    provider = Provider()
    sent = {}

    async def parse(**kwargs):
        sent.update(kwargs)
        return SimpleNamespace(output_parsed=SmallResult(value="ok"))

    provider.client = SimpleNamespace(responses=SimpleNamespace(parse=parse))
    result = await provider.parse("분석 지침", SmallResult, text="첫 정보\n" + "광고" * 3000 + "\n최신 정보")
    assert result.value == "ok" and sent["max_output_tokens"] == 2000
    bounded = sent["input"][1]["content"][0]["text"]
    assert "최신 정보" in bounded and budget.MARKER in bounded
    assert budget.tokens(bounded) + budget.tokens("분석 지침") \
        + budget.encoded_tokens(SmallResult.model_json_schema()) + budget.MESSAGE_OVERHEAD <= 1200
