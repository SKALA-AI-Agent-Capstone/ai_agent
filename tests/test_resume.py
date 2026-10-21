"""Resume after the human decision must finish the thread once, even when delivered twice."""
import asyncio
import os
from contextlib import asynccontextmanager

import pytest
from fastapi import HTTPException
from langgraph.checkpoint.memory import InMemorySaver

from capstone_agent.resume import resume_thread, thread_lock
from capstone_agent.review_engine import Provider, build_graph


async def interrupted_graph(monkeypatch, thread):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    graph = build_graph([], [], Provider(), InMemorySaver())
    config = {"configurable": {"thread_id": thread}}
    output = await graph.ainvoke({"reviewId": "r", "tenantId": "t", "versionNo": 1, "product": {}}, config)
    assert output["__interrupt__"]
    return graph, config


async def test_second_resume_reports_already_resumed(monkeypatch):
    graph, config = await interrupted_graph(monkeypatch, "t:r:v1")
    assert await resume_thread(graph, config, "approved") == {"reportReady": True}
    assert await resume_thread(graph, config, "approved") == {"reportReady": True, "alreadyResumed": True}
    assert (await graph.aget_state(config)).values["decision"] == "approved"


async def test_concurrent_resumes_under_the_thread_lock_run_once(monkeypatch):
    graph, config = await interrupted_graph(monkeypatch, "t:r:v1")
    lock = asyncio.Lock()

    @asynccontextmanager
    async def held():
        async with lock:
            yield

    async def one():
        async with held():
            return await resume_thread(graph, config, "approved")

    results = await asyncio.gather(one(), one())
    assert sorted(r.get("alreadyResumed", False) for r in results) == [False, True]


async def test_missing_checkpoint_is_a_permanent_404(monkeypatch):
    graph = build_graph([], [], Provider(), InMemorySaver())
    with pytest.raises(HTTPException) as error:
        await resume_thread(graph, {"configurable": {"thread_id": "none"}}, "approved")
    assert error.value.status_code == 404 and error.value.detail == "CHECKPOINT_NOT_FOUND"


@pytest.mark.skipif(not os.getenv("RESUME_LOCK_DATABASE_URL"), reason="needs a PostgreSQL URL in RESUME_LOCK_DATABASE_URL")
async def test_postgres_advisory_lock_serializes_the_same_thread():
    url, order = os.environ["RESUME_LOCK_DATABASE_URL"], []

    async def hold(name, delay):
        async with thread_lock(url, "t:r:v1"):
            order.append(f"{name}-in")
            await asyncio.sleep(delay)
            order.append(f"{name}-out")

    first = asyncio.create_task(hold("a", 0.3))
    await asyncio.sleep(0.1)
    await hold("b", 0)
    await first
    assert order == ["a-in", "a-out", "b-in", "b-out"]
