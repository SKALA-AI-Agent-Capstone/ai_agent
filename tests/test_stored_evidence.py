"""structure_evidence no longer reads files: it returns what Spring stored when the evidence was registered on the product."""
import httpx

from capstone_business_mcp import review_tools


def serve(monkeypatch, handler):
    real = httpx.AsyncClient
    monkeypatch.setattr(review_tools.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))


async def call():
    return await review_tools.structure("t", "r", [], 1, "run-token")


async def test_stored_result_is_passed_on_with_issuer_code_version_and_the_products_pages(monkeypatch):
    seen = {}

    def handler(request):
        seen["path"], seen["query"], seen["run"] = request.url.path, request.url.query, request.headers["x-run-token"]
        return httpx.Response(200, json={"documents": [{
            "id": "e1", "name": "report.pdf", "docType": "clinical_test", "pages": 53, "text": "9.71% 변화율",
            "issuer": {"key": "ksrc", "name": "KSRC"}, "codes": [{"label": "연구 코드", "value": "HBSE-MGE-24032"}],
            "version": "Version 1.0 · 2024-04", "unreadPages": [], "productPages": [6, 7], "results": [{"changePct": 9.71}],
            "productNotInReport": False, "readerVersion": "r1"}]})
    serve(monkeypatch, handler)
    result = await call()
    assert seen == {"path": "/internal/reviews/r/product-evidence", "query": b"versionNo=1", "run": "run-token"}
    assert result["status"] == "extracted" and "9.71%" in result["text"] and result["failures"] == []
    document = result["documents"][0]
    assert document["codes"][0]["value"] == "HBSE-MGE-24032" and document["productPages"] == [6, 7]


async def test_unread_pages_and_a_product_missing_from_the_report_are_reported_not_hidden(monkeypatch):
    serve(monkeypatch, lambda request: httpx.Response(200, json={"documents": [
        {"id": "a", "name": "a.pdf", "text": "본문", "pages": 3, "unreadPages": [2]},
        {"id": "b", "name": "b.pdf", "text": "", "pages": 3, "unreadPages": [], "productNotInReport": True}]}))
    result = await call()
    assert result["status"] == "partial"
    assert {(f["documentId"], f["code"], f.get("page")) for f in result["failures"]} == {
        ("a", "OCR_UNAVAILABLE", 2), ("b", "PRODUCT_NOT_IN_REPORT", None)}


async def test_no_stored_evidence_and_an_unreachable_backend_are_different_statuses(monkeypatch):
    serve(monkeypatch, lambda request: httpx.Response(200, json={"documents": []}))
    assert (await call())["status"] == "not_submitted"
    serve(monkeypatch, lambda request: httpx.Response(410, json={"message": "RUN_FINISHED"}))
    failed = await call()
    assert failed["status"] == "failed" and failed["failures"] == [{"code": "PRODUCT_EVIDENCE_FETCH_FAILED"}]
