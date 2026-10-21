import pytest


@pytest.fixture(autouse=True)
def development_environment(monkeypatch):
    monkeypatch.setenv("INTERNAL_SERVICE_TOKEN", "unit-test-only-token-not-for-production-12345")
    monkeypatch.setenv("BACKEND_BASE_URL", "http://127.0.0.1:8080")
    monkeypatch.setenv("AGENT_DATABASE_URL", "postgresql://invalid@127.0.0.1:1/invalid")
    monkeypatch.setenv("RAG_DATABASE_URL", "postgresql://invalid@127.0.0.1:1/invalid")
