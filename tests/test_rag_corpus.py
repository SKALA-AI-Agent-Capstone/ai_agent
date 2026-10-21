"""Law API parsing, article links, claim rules and the ingredient annex table. No network or DB."""
import xml.etree.ElementTree as ET

from capstone_rag_mcp.claims import expression_matches, parse_box_table, spelling_variants
from capstone_rag_mcp.ingest import chunk_pdf_sections
from capstone_rag_mcp.lawapi import admrul_articles, annexes, split_annex
from capstone_rag_mcp.links import extract
from capstone_rag_mcp.rules import item_line, load_rules, match_rules, section_matches

ANNEX_5 = """■ 화장품법 시행규칙 [별표 5]
화장품 표시ㆍ광고의 범위 및 준수사항(제22조 관련)
  1. 화장품 광고의 매체 또는 수단
  가. 신문ㆍ방송 또는 잡지
  2. 화장품 표시ㆍ광고 시 준수사항
  가. 의약품으로 잘못 인식할 우려가 있는 내용, 제품의 명칭 및 효능ㆍ효과 등에
  대한 표시ㆍ광고를 하지 말 것
  나. 기능성화장품이 아님에도 불구하고 기능성화장품으로 잘못 인식할 우려가 있는 표시ㆍ광고를 하지 말 것"""


def test_annex_is_split_at_numbered_items_so_citations_name_the_item():
    sections = [c.section for c in split_annex("5", "화장품 표시ㆍ광고의 범위", ANNEX_5)]
    assert sections == ["별표 5", "별표 5 제1호", "별표 5 제1호 가목", "별표 5 제2호", "별표 5 제2호 가목", "별표 5 제2호 나목"]


def test_table_annex_rows_become_one_chunk_per_expression():
    text = "[별표]\n표시광고 표현 실증자료\n○ 피부노화 완화\n▸ 인체 적용시험 자료 제출\n○ 콜라겐 증가, 감소 또는 활성화\n▸ 기능성화장품에서 해당 기능을 실증한 자료 제출"
    chunks = split_annex("0", "실증자료", text)
    assert [c.section for c in chunks[1:]] == ["별표 「피부노화 완화」", "별표 「콜라겐 증가, 감소 또는 활성화」"]


def test_admin_rule_articles_skip_chapter_headings_and_forms():
    root = ET.fromstring("""<AdmRulService><조문내용>제1장 총칙</조문내용>
      <조문내용>제3조(실증자료) ① 「화장품법 시행규칙」 제23조제2항에 따라 합리적인 근거로 인정될 수 있는 실증자료는</조문내용>
      <별표><별표단위><별표번호>0001</별표번호><별표가지번호>00</별표가지번호><별표구분>서식</별표구분><별표제목>신청서</별표제목><별표내용>x</별표내용></별표단위></별표>
      </AdmRulService>""")
    assert [c.section for c in admrul_articles(root, "실증 규정")] == ["제3조 (실증자료)"]
    assert annexes(root, None) == []


def test_links_follow_delegation_annex_and_cited_laws():
    assert extract("화장품법 시행규칙", "제22조", "법 제13조제2항에 따른 표시ㆍ광고의 범위와 ... 별표 5와 같다.") == {
        ("화장품법", "제13조", "implements"), ("화장품법 시행규칙", "별표 5", "annex")}
    assert extract("화장품 표시\u00b7광고 실증에 관한 규정", "제1조",
                   "이 고시는 「화장품법」 제14조 및 같은 법 시행규칙 제23조에 따라") == {
        ("화장품법", "제14조", "cites"), ("화장품법 시행규칙", "제23조", "cites")}
    assert ("화장품법", "제13조", "cites") in extract("화장품법", "제14조", "제13조제1항제4호에 해당하는지를")


def test_rules_match_by_claim_type_and_by_phrase_pattern():
    version, rules = load_rules()
    assert version and {r.id for r in rules} >= {"drug", "functional", "substantiation", "ingredient", "gmp"}
    matched = {m.rule.id: m.matched_by for m in match_rules("식약처 심사 완료, 주름 12.08% 개선", "clinical")}
    assert matched["substantiation"] == "claim_type"
    assert matched["official_endorsement"] == "pattern" and matched["functional"] == "pattern"
    assert match_rules("촉촉한 사용감", None) == []


def test_section_and_item_resolution():
    assert section_matches("제13조 (부당한 표시ㆍ광고 행위 등의 금지)", "제13조")
    assert not section_matches("제13조의2 (보고)", "제13조")
    assert section_matches("[별표 2] > 1. 「화장품", "[별표 2] > 1.")
    body = "① 영업자 또는 판매자는 다음 각 호의 ...\n  1. 의약품으로 잘못 인식할 우려가 있는 표시 또는 광고\n  2. 기능성화장품이 아닌 화장품을 ..."
    assert item_line(body, "의약품으로 잘못 인식할 우려") == ("제1항제1호", "1. 의약품으로 잘못 인식할 우려가 있는 표시 또는 광고")


def test_substantiation_rows_match_on_their_core_term():
    assert expression_matches("별표 「피부노화 완화」", "7종 피부 노화 징후 개선")
    assert not expression_matches("별표 「피부 혈행 개선」", "7종 피부 노화 징후 개선")
    assert expression_matches("별표 「붓기, 다크서클 완화」", "다크서클 케어")


def test_ingredient_table_rows_are_rebuilt_across_wrapped_lines():
    table = """├────┼────┤
│1,4-디하이드록시-5,8-비스[(2-하이드  │3179-90-6 │
│록시에칠)아미노]안트라퀴논 및 그 염류 │          │
├────┼────┤
│히드로퀴논                           │123-31-9  │
├────┼────┤"""
    rows = parse_box_table(table)
    assert rows[0][0] == "1,4-디하이드록시-5,8-비스[(2-하이드 록시에칠)아미노]안트라퀴논 및 그 염류"
    assert rows[1] == ["히드로퀴논", "123-31-9"]
    assert "히드로퀴논" in spelling_variants("하이드로퀴논")


def test_guideline_pdf_is_split_at_its_own_headings():
    body = "이 기준은 화장품 광고를 자문함에 있어 구체적 기준을 제시한다.\n" * 20  # long pages, not a slide deck
    pages = ["점검표 표지", f"Ⅰ. 화장품 광고자문\n1. 목적\n{body}", f"2. 적용범위\n{body}"]
    chunks = chunk_pdf_sections(pages)
    assert [c["section"] for c in chunks] == ["Ⅰ. 화장품 광고자문 > 1. 목적", "Ⅰ. 화장품 광고자문 > 2. 적용범위"]
    assert chunks[0]["page_from"] == 2


async def test_pdf_download_handles_read_only_host_mount(monkeypatch, tmp_path):
    import httpx
    from pathlib import Path
    from capstone_rag_mcp.batch import download_pdf

    blocked = tmp_path / "host-data" / "sources"
    monkeypatch.setenv("RAG_DATA_DIR", str(blocked.parent))
    original_mkdir = Path.mkdir

    def deny_host_write(path, *args, **kwargs):
        if path == blocked:
            raise PermissionError("read-only host directory")
        return original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", deny_host_write)
    payload = b"%PDF-1.7\nfixture"
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=payload))
    async with httpx.AsyncClient(transport=transport) as client:
        downloaded = await download_pdf(client, {"id": "S01", "verified_direct_download_url": "https://www.mfds.go.kr/source.pdf"})
    try:
        assert downloaded.is_file() and downloaded.read_bytes() == payload
        assert not blocked.exists()
    finally:
        downloaded.unlink()
