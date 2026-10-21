from openai import AsyncOpenAI

from capstone_agent.observability import run_config, trace_id, traced_client, tracing_enabled


def test_tracing_is_off_unless_both_flag_and_key_are_set(monkeypatch):
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    client = AsyncOpenAI(api_key="sk-test-not-real")
    assert not tracing_enabled() and traced_client(client) is client
    config = run_config("t:r:v1", run_name="review", review_id="r", tenant_id="t", version_no=1)
    assert config == {"configurable": {"thread_id": "t:r:v1"}} and trace_id(config) is None

    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    assert not tracing_enabled()  # a flag without a key never sends anything
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2-test-not-real")
    assert tracing_enabled()
    config = run_config("t:r:v1", run_name="review", review_id="r", tenant_id="t", version_no=1)
    assert config["configurable"] == {"thread_id": "t:r:v1"}
    assert config["metadata"] == {"review_id": "r", "tenant_id": "t", "version_no": 1}
    assert trace_id(config) == str(config["run_id"])
