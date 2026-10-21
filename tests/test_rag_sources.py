from datetime import date
import json
from pathlib import Path

from fastmcp import Client

from capstone_business_mcp.server import create_server as create_business
from capstone_rag_mcp.legalize import LawVersion, assign_effective_ranges, chunk_law, parse_front_matter
from capstone_rag_mcp.retrieval import keyword_rank
from capstone_rag_mcp.server import create_server as create_rag

ROOT = Path(__file__).resolve().parents[1]

# Shape copied from legalize-kr kr/화장품법/법률.md, shortened.
LAW = """---
제목: 화장품법
법령MST: 289763
법령구분: 법률
소관부처:
- 식품의약품안전처
공포일자: 2026-09-15
공포번호: '21950'
시행일자: 2027-09-16
첨부파일: []
---

# 화장품법

## 제3장 화장품의 취급

### 제2절 표시ㆍ광고ㆍ취급

##### 제13조 (부당한 표시ㆍ광고 행위 등의 금지)

**①** 영업자 또는 판매자는 다음 각 호의 어느 하나에 해당하는 표시 또는 광고를 하여서는 아니 된다.

  1\\. 의약품으로 잘못 인식할 우려가 있는 표시 또는 광고

##### 제14조 (표시ㆍ광고 내용의 실증 등)

**①** 영업자 및 판매자는 자기가 행한 표시ㆍ광고 중 사실과 관련한 사항에 대하여는 이를 실증할 수 있어야 한다.

##### 제14조의2

삭제 <2025.1.31>

## 부칙

제1조(시행일) 이 법은 공포 후 6개월이 경과한 날부터 시행한다.
"""


def test_front_matter_reads_dates_and_ministry_list():
    meta, _ = parse_front_matter(LAW)
    assert meta["시행일자"] == "2027-09-16"
    assert meta["공포번호"] == "21950"
    assert meta["소관부처"] == ["식품의약품안전처"]


def test_law_is_chunked_per_article_without_deleted_articles_or_supplementary_provisions():
    chunks = chunk_law(LAW)
    assert [c.section for c in chunks] == ["제13조 (부당한 표시ㆍ광고 행위 등의 금지)", "제14조 (표시ㆍ광고 내용의 실증 등)"]
    assert chunks[0].context == "화장품법 > 제3장 화장품의 취급 > 제2절 표시ㆍ광고ㆍ취급"
    assert "1. 의약품으로 잘못 인식할" in chunks[0].body
    assert "**" not in chunks[0].body
    assert all("시행일" not in c.body for c in chunks)


def test_older_statute_version_ends_the_day_before_the_next_takes_effect():
    def version(effective):
        return LawVersion(path="kr/화장품법/법률.md", commit=effective, meta={}, text="", effective_from=date.fromisoformat(effective))
    newest, current = assign_effective_ranges([version("2027-09-16"), version("2026-04-28")])
    assert newest.effective_until is None
    assert current.effective_until == date(2027, 9, 15)


def test_keyword_rank_finds_the_article_about_medicine_like_claims():
    rows = [{"chunk_id": str(i), "section": c.section, "body": c.body} for i, c in enumerate(chunk_law(LAW))]
    ranked = keyword_rank("아토피 치료 크림, 의약품으로 오인", rows)
    assert ranked[0].row["section"].startswith("제13조")
    assert 0 < ranked[0].similarity <= 1


def test_every_source_has_a_single_acquisition_route():
    sources = json.loads((ROOT / "resources/reference-sources.json").read_text(encoding="utf-8"))
    routes = {s["id"]: s["acquisition"]["method"] for s in sources}
    assert routes["S05"] == "legalize_kr" and routes["S06"] == "legalize_kr+lawapi"
    assert {routes[i] for i in ["S01", "S02", "S03", "S04"]} == {"pdf"}
    assert {routes[i] for i in ["S07", "S08", "S09", "S10"]} == {"lawapi"}
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
        for tool, args in [("verify_gmp_company", {"company_name": "한국콜마"}),
                           ("lookup_ingredient_regulation", {"ingredient_name": "하이드로퀴논"})]:
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
