"""Progress events the Agent sends to Spring for the SSE stream."""
from types import SimpleNamespace

import pytest

from capstone_agent import review_engine as engine
from capstone_agent.review_models import RunRequest


@pytest.mark.asyncio
async def test_business_tools_emit_start_and_finish_with_adapter_failures(monkeypatch):
    sent = []

    async def record(request, kind, **data):
        sent.append((kind, data))

    async def fake_call(name, arguments, request=None):
        if name == "verify_patent":
            return [{"status": "api_error"}]
        return {"status": "found"}

    monkeypatch.setattr(engine, "emit", record)
    monkeypatch.setattr(engine, "mcp_call", fake_call)
    request = RunRequest(reviewId="r", tenantId="t", versionNo=1, runToken="x" * 32, product={}, assets=[])
    state = {"product": {"name": "크림"}, "texts": [{"text": "특허 10-1234567"}], "findings": [],
             "tenantId": "t", "reviewId": "r"}
    await engine.select_business_tools(SimpleNamespace(client=None), state, request)

    assert sent == [
        ("tool_started", {"tool": "verify_functional_approval"}),
        ("tool_finished", {"tool": "verify_functional_approval", "ok": True}),
        ("tool_started", {"tool": "verify_patent"}),
        ("tool_finished", {"tool": "verify_patent", "ok": False}),
    ]


def test_tool_success_ignores_an_empty_error_field():
    # Real adapter results always carry "error"; None means the call worked.
    assert engine.tool_succeeded({"status": "not_found", "error": None, "data": None})
    assert engine.tool_succeeded({"status": "found", "error": None})
    assert engine.tool_succeeded([{"status": "registered", "error": None}, {"status": "needs_confirm", "error": None}])
    assert not engine.tool_succeeded({"status": "error", "error": "API_NOT_CONFIGURED"})
    assert not engine.tool_succeeded({"status": "not_found", "error": "PROVIDER_REQUEST_FAILED"})
    assert not engine.tool_succeeded([{"status": "registered", "error": None}, {"status": "api_error"}])


def test_a_lookup_that_came_back_without_an_answer_is_not_a_failed_call():
    # What the adapters really return: an informational code on a call that worked.
    assert engine.tool_succeeded({"status": "ambiguous", "error": "IDENTIFIER_MAPPING_OR_PAGINATION_REQUIRES_CONFIRMATION"})
    assert engine.tool_succeeded({"status": "cancelled", "error": "REPORTED_ITEM_CANCELLED"})
    assert engine.tool_succeeded([{"status": "needs_confirm", "error": "NUMBER_REQUIRES_CONFIRMATION"}])
    assert engine.tool_succeeded([{"status": "unavailable", "error": "NO_EXACT_MATCH_OR_INCOMPLETE_SEARCH"}])
    assert engine.tool_succeeded({"status": "incomplete", "error": "SCAN_LIMIT_REACHED"})
    # The call itself failing is still a failure.
    assert not engine.tool_succeeded({"status": "error", "error": "API_NOT_CONFIGURED"})
    assert not engine.tool_succeeded([{"status": "unavailable", "error": "PROVIDER_REQUEST_FAILED"}])
    assert not engine.tool_succeeded({"error": "VERIFY_PATENT_TOOL_UNAVAILABLE"})


@pytest.mark.asyncio
async def test_progress_is_a_step_done_event_and_never_raises(monkeypatch):
    sent = []

    async def record(request, kind, **data):
        sent.append((kind, data))

    monkeypatch.setattr(engine, "emit", record)
    await engine.progress(object(), 3)
    assert sent == [("step_done", {"step": 3})]


@pytest.mark.asyncio
async def test_emit_swallows_backend_errors(monkeypatch):
    monkeypatch.setenv("BACKEND_BASE_URL", "http://127.0.0.1:1")
    request = RunRequest(reviewId="r", tenantId="t", versionNo=1, runToken="x" * 32, product={}, assets=[])
    await engine.emit(request, "tool_started", tool="verify_patent")
    await engine.emit(None, "tool_started", tool="verify_patent")
