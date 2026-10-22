import json
from pathlib import Path

from fastmcp import Client

from capstone_business_mcp.server import create_server as create_business
from capstone_rag_mcp.retrieval import keyword_rank
from capstone_rag_mcp.server import create_server as create_rag

ROOT = Path(__file__).resolve().parents[1]

ARTICLES = [
    {"chunk_id": "1", "section": "제13조 (부당한 표시ㆍ광고 행위 등의 금지)",
     "body": "① 영업자 또는 판매자는 다음 각 호의 어느 하나에 해당하는 표시 또는 광고를 하여서는 아니 된다.\n"
             "1. 의약품으로 잘못 인식할 우려가 있는 표시 또는 광고"},
    {"chunk_id": "2", "section": "제14조 (표시ㆍ광고 내용의 실증 등)",
     "body": "① 영업자 및 판매자는 자기가 행한 표시ㆍ광고 중 사실과 관련한 사항에 대하여는 이를 실증할 수 있어야 한다."},
]


def test_keyword_rank_finds_the_article_about_medicine_like_claims():
    ranked = keyword_rank("아토피 치료 크림, 의약품으로 오인", ARTICLES)
    assert ranked[0].row["section"].startswith("제13조")
    assert 0 < ranked[0].similarity <= 1


def test_every_source_has_a_single_acquisition_route():
    sources = json.loads((ROOT / "resources/reference-sources.json").read_text(encoding="utf-8"))
    routes = {s["id"]: s["acquisition"]["method"] for s in sources}
    assert {routes[i] for i in ["S01", "S02", "S04"]} == {"pdf"}
    # 화장품법 statutes come from the Law API only; legalize-kr (GitHub) was removed on 2026-10-22.
    assert {routes[i] for i in ["S05", "S06", "S07", "S08", "S09", "S10"]} == {"lawapi"}
    assert not any("paths" in s["acquisition"] or "legalize" in s["acquisition"]["method"] for s in sources)
    assert all(s["acquisition"]["lawapi"]["name"] for s in sources if "lawapi" in s["acquisition"]["method"])


async def test_regulation_search_reports_a_database_failure_instead_of_an_empty_result():
    async with Client(create_rag()) as client:
        result = await client.call_tool("search_regulations", {"query_text": "피부 주름 90% 감소"})
        content = result.structured_content
        assert content["status"] == "SEARCH_ERROR"
        assert content["hits"] == []


async def test_precedent_search_is_a_separate_tool():
    async with Client(create_rag()) as client:
        names = {tool.name for tool in await client.list_tools()}
        assert {"search_regulations", "search_precedents"} <= names
        assert not names & {"search_rules", "search_cases", "get_evidence"}
        result = await client.call_tool("search_precedents", {"query_text": "기미 완전 제거"})
        assert result.structured_content["status"] == "SEARCH_ERROR"


async def test_mfds_list_tools_say_not_configured_without_a_key(monkeypatch):
    monkeypatch.delenv("MFDS_SERVICE_KEY", raising=False)
    async with Client(create_business()) as client:
        assert "lookup_ingredient_regulation" not in {tool.name for tool in await client.list_tools()}
        for tool, args in [("verify_gmp_company", {"company_name": "한국콜마"})]:
            result = await client.call_tool(tool, args)
            assert result.structured_content["status"] == "error"
            assert result.structured_content["error"] == "API_NOT_CONFIGURED"


def test_pdf_source_keeps_pages_and_mixed_documents_need_page_ranges(tmp_path):
    from capstone_rag_mcp.ingest import from_pdf

    sources = json.loads((ROOT / "resources/reference-sources.json").read_text(encoding="utf-8"))
    s01 = next(s for s in sources if s["id"] == "S01")
    path = tmp_path / "S01.pdf"
    path.write_bytes(b"synthetic test bytes")
    doc = from_pdf(s01, path, pages=["", "1. 적용 범위\n이 지침은 화장품 표시 광고에 적용한다."])
    assert doc.document_key == "pdf:S01"
    assert doc.collection == "rules" and doc.source_type == "guideline"
    assert [c["page_from"] for c in doc.chunks] == [2]
    s04 = next(s for s in sources if s["id"] == "S04")
    assert s04["acquisition"]["page_ranges"] == {"rules": [2, 7], "cases": [8, 27]}


def test_fusion_keeps_keyword_and_vector_scores_apart():
    from capstone_rag_mcp.retrieval import Ranked, fuse
    a, b, c = ({"chunk_id": key} for key in "abc")
    keyword = [Ranked(a, 0.5, keyword=0.5), Ranked(b, 0.2, keyword=0.2)]
    vector = [Ranked(c, 0.71, vector=0.71), Ranked(a, 0.66, vector=0.66)]
    fused = fuse(keyword, vector)
    assert [r.row["chunk_id"] for r in fused][0] == "a"  # found by both signals, ranked first
    first = fused[0]
    assert (first.keyword, first.vector, first.similarity) == (0.5, 0.66, 0.66)
    only_vector = next(r for r in fused if r.row["chunk_id"] == "c")
    assert only_vector.keyword is None and only_vector.vector == 0.71


def test_agent_applies_a_threshold_per_signal():
    from capstone_agent.regulations import relevant
    assert relevant({"keyword_score": 0.5, "vector_score": 0.3}, 0.45, 0.6)
    assert relevant({"keyword_score": 0.1, "vector_score": 0.62}, 0.45, 0.6)
    assert not relevant({"keyword_score": 0.1, "vector_score": 0.55, "similarity": 0.55}, 0.45, 0.6)
    assert relevant({"similarity": 0.5}, 0.45, 0.6)  # older server without split scores


def test_confirm_order_keeps_keyword_ranking_and_adds_strong_vector_hits():
    from capstone_rag_mcp.retrieval import Ranked, confirm
    a, b, c, d = ({"chunk_id": key} for key in "abcd")
    keyword = [Ranked(a, 0.5, keyword=0.5), Ranked(b, 0.2, keyword=0.2)]
    vector = [Ranked(b, 0.62, vector=0.62), Ranked(c, 0.5, vector=0.5), Ranked(d, 0.3, vector=0.3)]
    order = confirm(keyword, vector)
    assert [r.row["chunk_id"] for r in order] == ["a", "b", "c"]  # d is below the vector threshold
    assert order[1].keyword == 0.2 and order[1].vector == 0.62
