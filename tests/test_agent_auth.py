import httpx

from capstone_agent.app import create_app


async def test_internal_execution_rejects_missing_token():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app()), base_url="http://test"
    ) as client:
        response = await client.post("/internal/runs")
        assert response.status_code == 401
        assert "NOT_IMPLEMENTED" not in response.text


async def test_authenticated_execution_requires_scoped_input():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app()), base_url="http://test"
    ) as client:
        response = await client.post(
            "/internal/runs",
            headers={"Authorization": "Bearer unit-test-only-token-not-for-production-12345"},
        )
        assert response.status_code == 422


async def test_database_failure_is_not_reported_as_ready():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app()), base_url="http://test"
    ) as client:
        response = await client.get("/health/ready")
        assert response.status_code == 503
        assert response.json()["status"] == "down"
        assert "postgresql://" not in response.text


async def test_rewrite_without_model_key_is_reported_not_invented(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app()), base_url="http://test"
    ) as client:
        response = await client.post(
            "/internal/rewrites",
            headers={"Authorization": "Bearer unit-test-only-token-not-for-production-12345"},
            json={"reviewId": "r", "tenantId": "t", "phrase": "7일 주름 개선"},
        )
        assert response.status_code == 503
        assert response.json()["detail"] == "TEXT_MODEL_NOT_CONFIGURED"


async def test_rewrite_rejects_missing_token():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app()), base_url="http://test"
    ) as client:
        response = await client.post("/internal/rewrites", json={"reviewId": "r", "tenantId": "t", "phrase": "x"})
        assert response.status_code == 401


class _Saver:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def setup(self):
        return None


def _failing_graph(monkeypatch, error):
    import capstone_agent.app as agent_app
    class Graph:
        async def ainvoke(self, state, config):
            raise error
    async def assets(request):
        return [], []
    monkeypatch.setattr(agent_app, "prepare_assets", assets)
    monkeypatch.setattr(agent_app.AsyncPostgresSaver, "from_conn_string", lambda dsn: _Saver())
    monkeypatch.setattr(agent_app, "build_graph", lambda *args: Graph())
    monkeypatch.setattr(agent_app, "Provider", lambda: None)


async def _run(body=None):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app()), base_url="http://test"
    ) as client:
        return await client.post(
            "/internal/runs",
            headers={"Authorization": "Bearer unit-test-only-token-not-for-production-12345"},
            json=body or {"reviewId": "r", "tenantId": "t", "versionNo": 1, "runToken": "x" * 32, "product": {}, "assets": []},
        )


async def test_failed_graph_run_returns_its_trace_id(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "unit-test-only")
    for error, status in ((ValueError("unreadable"), 422), (RuntimeError("model down"), 500)):
        _failing_graph(monkeypatch, error)
        response = await _run()
        assert response.status_code == status
        assert len(response.headers["X-Trace-Id"]) == 36
        assert "model down" not in response.text


async def test_failure_before_the_graph_or_without_tracing_has_no_trace_header(monkeypatch):
    import capstone_agent.app as agent_app
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "unit-test-only")
    async def unreadable(request):
        raise ValueError("asset expired")
    monkeypatch.setattr(agent_app, "prepare_assets", unreadable)
    response = await _run()
    assert response.status_code == 422
    assert "X-Trace-Id" not in response.headers

    monkeypatch.delenv("LANGSMITH_TRACING")
    _failing_graph(monkeypatch, RuntimeError("model down"))
    response = await _run()
    assert response.status_code == 500
    assert "X-Trace-Id" not in response.headers
