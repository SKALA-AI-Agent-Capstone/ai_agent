from fastmcp import Client

from capstone_business_mcp.server import create_server as create_business
from capstone_rag_mcp.server import create_server as create_rag


async def test_rag_mcp_serves_source_originals_as_resources(monkeypatch):
    import json
    import capstone_rag_mcp.server as rag_server
    version_id = "0b6f1f3c-7a5e-4d0c-9a51-2b8f0d6c1e11"
    async def inventory(dsn):
        return [{"document_version_id": version_id, "document_key": "lawapi:화장품법/법률", "title": "화장품법"}]
    async def text(dsn, document_version_id):
        if document_version_id != version_id:
            return None
        return {"document_version_id": version_id, "title": "화장품법", "version": "합성",
                "sections": [{"section": "제13조", "page": None, "text": "합성 조문 원문"}]}
    monkeypatch.setattr(rag_server, "corpus_inventory", inventory)
    monkeypatch.setattr(rag_server, "document_text", text)
    async with Client(create_rag()) as client:
        assert "regulation://corpus" in {str(r.uri) for r in await client.list_resources()}
        templates = {t.uri_template for t in await client.list_resource_templates()}
        assert "regulation://documents/{document_version_id}" in templates
        corpus = json.loads((await client.read_resource("regulation://corpus"))[0].text)
        assert corpus["documents"][0]["document_version_id"] == version_id
        document = json.loads((await client.read_resource(f"regulation://documents/{version_id}"))[0].text)
        assert document["sections"][0]["text"] == "합성 조문 원문"
        for missing in ("not-a-uuid", "11111111-1111-1111-1111-111111111111"):
            try:
                await client.read_resource(f"regulation://documents/{missing}")
            except Exception as error:
                assert "regulation://corpus" in str(error)
            else:
                raise AssertionError("unknown document must not return content")


async def test_business_mcp_has_no_approval_or_mutation_tool_in_bootstrap():
    async with Client(create_business()) as client:
        tools = await client.list_tools()
        names = {tool.name for tool in tools}
        assert {"get_environment_info", "structure_evidence", "verify_functional_approval", "verify_patent"} <= names
        assert not names.intersection({"approve", "delete_review", "charge"})
