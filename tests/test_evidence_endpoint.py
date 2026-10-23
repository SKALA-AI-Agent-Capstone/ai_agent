import httpx

import capstone_agent.app as agent_app
import capstone_agent.evidence_reading as evidence
from capstone_agent.app import create_app
from capstone_shared.reading import READER_VERSION
from test_reading import REPORT, text_pdf

AUTH = {"Authorization": "Bearer unit-test-only-token-not-for-production-12345"}


def client():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app()), base_url="http://test")


async def test_reader_version_is_served_to_spring_only_with_the_internal_token():
    async with client() as http:
        assert (await http.get("/internal/evidence/reader", headers=AUTH)).json() == {"readerVersion": READER_VERSION}
        assert (await http.get("/internal/evidence/reader")).status_code == 401
        assert (await http.post("/internal/evidence/parse", json={"evidenceId": "e1"})).status_code == 401


async def test_stored_file_is_fetched_and_read(monkeypatch):
    async def fetch(evidence_id):
        assert evidence_id == "e1"
        return text_pdf(REPORT)
    monkeypatch.setattr(evidence, "fetch_evidence", fetch)
    async with client() as http:
        response = await http.post("/internal/evidence/parse", json={"evidenceId": "e1", "productName": "아무 제품"}, headers=AUTH)
    body = response.json()
    assert response.status_code == 200
    assert body["readerVersion"] == READER_VERSION and body["pageCount"] == 1 and body["pages"][0]["method"] == "text_layer"


async def test_unreadable_missing_and_slow_files_get_their_own_errors(monkeypatch):
    async def garbage(evidence_id):
        return b"not a pdf or an image"

    async def gone(evidence_id):
        raise httpx.HTTPStatusError("404", request=httpx.Request("GET", "http://x"), response=httpx.Response(404))

    async def slow(request):
        raise TimeoutError

    for fetch, status, detail in ((garbage, 422, "PDF_OR_IMAGE_REQUIRED"), (gone, 404, "EVIDENCE_FILE_NOT_FOUND")):
        monkeypatch.setattr(evidence, "fetch_evidence", fetch)
        async with client() as http:
            response = await http.post("/internal/evidence/parse", json={"evidenceId": "e1"}, headers=AUTH)
        assert (response.status_code, response.json()["detail"]) == (status, detail)
    monkeypatch.setattr(agent_app, "parse_stored_evidence", slow)
    async with client() as http:
        response = await http.post("/internal/evidence/parse", json={"evidenceId": "e1"}, headers=AUTH)
    assert (response.status_code, response.json()["detail"]) == (504, "EVIDENCE_READ_TIMEOUT")
