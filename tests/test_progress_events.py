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
