from langgraph.checkpoint.memory import InMemorySaver

from capstone_agent.review_engine import build_graph


def test_personal_data_masking_keeps_document_numbers():
    from capstone_agent.guardrails import mask_personal_data
    text = ("담당 연구원 연락처 010-1234-5678, 01098765432, 02-555-1234, 031) 777 8888, "
            "lab.manager@example.com, 주민번호 900101-1234567. "
            "특허 10-1234567, 보고번호 2019-12345, 시험일 2026-10-21, 사업자 123-45-67890, 고객센터 1588-1234.")
    masked, count = mask_personal_data(text)
    assert count == 6
    for secret in ("010-1234-5678", "01098765432", "02-555-1234", "777 8888", "lab.manager@", "900101-1234567"):
        assert secret not in masked
    for kept in ("10-1234567", "2019-12345", "2026-10-21", "123-45-67890", "1588-1234"):
        assert kept in masked


async def test_evidence_contact_data_is_masked_before_model_and_storage(monkeypatch):
    # The real mcp_call runs over a fake MCP client, so the mask sits on the actual tool-result boundary.
    from types import SimpleNamespace
    from capstone_agent.review_models import RunRequest, TextAssessment
    import capstone_agent.review_engine as engine
    monkeypatch.setenv("BUSINESS_MCP_URL", "http://business-mcp.invalid/mcp")
    evidence_text = "시험 담당 010-1234-5678 test@lab.kr 특허 10-1234567"
    class FakeMcpClient:
        def __init__(self, transport):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *exc):
            return False
        async def call_tool(self, name, arguments):
            data = {"status": "extracted", "text": evidence_text, "claimTypes": [], "requirements": [], "flags": [],
                    "documents": [{"id": "e1", "name": "report.pdf", "text": evidence_text}]}
            return SimpleNamespace(is_error=False, data=data if name == "structure_evidence" else [])
    async def ocr(native, tiles, provider):
        return [{"id":"t1","text":"주름 개선","location":{"kind":"image","fileId":"a","page":1,"bbox":None}}],[]
    async def quiet(request, step):
        return None
    seen = []
    async def severity(provider, finding, evidence_text):
        seen.append(evidence_text)
        return finding["severity"], None
    async def no_rewrites(findings, provider):
        return []
    monkeypatch.setattr(engine, "Client", FakeMcpClient)
    monkeypatch.setattr(engine, "extract_text", ocr)
    monkeypatch.setattr(engine, "progress", quiet)
    monkeypatch.setattr(engine, "assess_severity", severity)
    monkeypatch.setattr(engine, "build_violations", no_rewrites)
    summarized = []
    async def summarize(provider, documents):
        summarized.extend(documents)
        return []
    monkeypatch.setattr(engine, "summarize", summarize)
    model_messages = []
    class FakeProvider:
        client = True
        async def parse(self, prompt, schema, **kwargs):
            if schema is TextAssessment:
                return schema(findings=[{"phrase":"주름 개선","reason":"합성","claimType":"wrinkle","rewrites":[]}])
            raise AssertionError(f"unexpected schema {schema}")
        async def call_with_tools(self, messages, tools):
            model_messages.extend(m for m in messages if m["role"] == "tool")
            if any(m["role"] == "tool" for m in messages):
                return SimpleNamespace(content="done", tool_calls=[])
            call = SimpleNamespace(id="c1", function=SimpleNamespace(name="structure_evidence", arguments="{}"))
            call.model_dump = lambda: {"id": "c1", "type": "function",
                                       "function": {"name": "structure_evidence", "arguments": "{}"}}
            return SimpleNamespace(content="", tool_calls=[call])
    request = RunRequest(reviewId="r", tenantId="t", versionNo=1, runToken="x"*32, product={},
                         assets=[{"id":"e1","name":"report.pdf","role":"evidence"}])
    graph = build_graph([], [], FakeProvider(), InMemorySaver(), request)
    output = await graph.ainvoke({"reviewId":"r","tenantId":"t","versionNo":1,"product":{}},
                                 {"configurable":{"thread_id":"t:r:mask"}})
    evidence = output["result"]["evidence"]
    assert evidence["maskedPersonalData"] == 4  # combined text and the document's own text
    assert "010-1234-5678" not in evidence["text"] and "test@lab.kr" not in evidence["text"]
    assert "10-1234567" in evidence["text"]
    stored_document = evidence["documents"][0]["text"]
    assert "010-1234-5678" not in stored_document and "10-1234567" in stored_document
    assert summarized and all("010-1234-5678" not in d["text"] and "test@lab.kr" not in d["text"] for d in summarized)
    assert model_messages and all("010-1234-5678" not in m["content"] for m in model_messages)
    assert seen and all("010-1234-5678" not in text for text in seen)


def test_tool_result_masks_each_document_text():
    from capstone_agent.guardrails import mask_tool_result
    data = {"text": "담당 010-1111-2222", "documents": [{"id": "a", "text": "연락처 lab@example.com"},
                                                       {"id": "b", "text": "개인정보 없음"}, "unexpected"]}
    masked = mask_tool_result("structure_evidence", data)
    assert masked["maskedPersonalData"] == 2
    assert "010-1111-2222" not in masked["text"] and "lab@example.com" not in masked["documents"][0]["text"]
    assert masked["documents"][1] == {"id": "b", "text": "개인정보 없음"} and masked["documents"][2] == "unexpected"
    assert data["documents"][0]["text"] == "연락처 lab@example.com"  # the tool result itself is not mutated
    clean = {"text": "개인정보 없음", "documents": [{"id": "a", "text": "없음"}]}
    assert mask_tool_result("structure_evidence", clean) is clean
    assert mask_tool_result("verify_gmp_company", {"text": "010-1111-2222"}) == {"text": "010-1111-2222"}
