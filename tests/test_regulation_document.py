import json

import httpx
from fastmcp import Client

import capstone_agent.app as agent_app
from capstone_agent.app import create_app
from capstone_agent.regulations import source_document
import capstone_rag_mcp.server as rag_server

AUTH = {"Authorization": "Bearer unit-test-only-token-not-for-production-12345"}
OLD, NEW = "0b6f1f3c-7a5e-4d0c-9a51-2b8f0d6c1e11", "5d2a9c1e-3f4b-4e8a-b6c7-1a2b3c4d5e6f"


async def test_basis_source_resolves_its_exact_version_through_resources(monkeypatch):
    async def inventory(dsn):
        return [{"document_version_id": OLD, "document_key": "lawapi:화장품법/법률", "version_label": "2025-01-01"},
                {"document_version_id": NEW, "document_key": "lawapi:화장품법/법률", "version_label": "2026-03-01"}]
    async def text(dsn, document_version_id):
        return {"document_version_id": document_version_id,
                "sections": [{"section": "제13조", "page": None, "text": f"합성 원문 {document_version_id}"}]}
    monkeypatch.setattr(rag_server, "corpus_inventory", inventory)
    monkeypatch.setattr(rag_server, "document_text", text)
    async with Client(rag_server.create_server()) as client:
        async def read(server, uri):
            assert server == "rag"
            return json.loads((await client.read_resource(uri))[0].text)
        document = await source_document(read, "lawapi:화장품법/법률", "2025-01-01")
        assert document["document_version_id"] == OLD
        assert await source_document(read, "lawapi:화장품법/법률", "1999-01-01") is None
        assert await source_document(read, "lawapi:약사법/법률", "2025-01-01") is None


async def test_document_endpoint_reports_missing_and_unavailable_sources(monkeypatch):
    async def missing(read, key, version):
        return None
    async def broken(read, key, version):
        raise RuntimeError("rag down")
    for fake, status, detail in ((missing, 404, "REGULATION_DOCUMENT_NOT_FOUND"),
                                 (broken, 502, "REGULATION_SOURCE_UNAVAILABLE")):
        monkeypatch.setattr(agent_app, "source_document", fake)
        monkeypatch.setattr(agent_app.McpTools, "__aenter__", lambda self: _noop())
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app()), base_url="http://test") as client:
            response = await client.get("/internal/regulations/document",
                                        params={"key": "lawapi:화장품법/법률", "version": "2025-01-01"}, headers=AUTH)
            assert response.status_code == status
            assert response.json()["detail"] == detail
            denied = await client.get("/internal/regulations/document", params={"key": "k", "version": "v"})
            assert denied.status_code == 401


async def _noop():
    return None
