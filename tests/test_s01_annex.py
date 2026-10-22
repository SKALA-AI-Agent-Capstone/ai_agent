from pathlib import Path

from capstone_agent.review_engine import basis_context
from capstone_rag_mcp.claims import s01_annex_related
from capstone_rag_mcp.ingest import from_pdf
from capstone_rag_mcp.pdf_tables import s01_annex_rows


class FakePage:
    def __init__(self, heading: str, rows: list[list[str | None]]):
        self.heading = heading
        self.rows = rows

    def extract_text(self):
        return self.heading

    def extract_tables(self):
        return [self.rows]


class FakePdf:
    def __init__(self, pages):
        self.pages = pages

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


def test_s01_annex_rows_keep_exception_and_continued_evidence(monkeypatch):
    pages = [
        FakePage("[별표 1]\n화장품 표시·광고의 표현 범위", [
            ["구 분", "금 지 표 현", "비 고"],
            ["질병 관련", "· 여드름", "단, [별표2] 1.에 해당하는 표현은 제외"],
        ]),
        FakePage("[별표 2]\n화장품 표시·광고 주요 실증대상", [
            ["구 분", "실증 대상", "비 고"],
            ["1. 고시에 따른 표현", "· 여드름성 피부에 사용에 적합", "인체 적용시험 자료로 입증"],
            [None, "· 빠지는 모발을 감소시킨다", "기능성화장품으로서 심사"],
        ]),
        FakePage("구 분 실증 대상 비 고", [
            ["구 분", "실증 대상", "비 고"],
            ["", "", "받은 자료에 근거가 포함되거나 별도 실증"],
            ["2. 효능·효과", "· 수분감 개선", "인체 적용시험으로 입증"],
        ]),
    ]
    import pdfplumber
    monkeypatch.setattr(pdfplumber, "open", lambda _: FakePdf(pages))
    rows = s01_annex_rows(Path("S01.pdf"))
    assert len(rows) == 4
    assert rows[0]["section"] == "[별표 1] > 질병 관련"
    assert "[별표2] 1." in rows[0]["body"]
    assert "여드름성 피부에 사용에 적합" not in rows[0]["body"]  # separate Annex 2 row
    assert "여드름성 피부에 사용에 적합" in rows[1]["body"]
    assert rows[2]["page_until"] == 3
    assert "별도 실증" in rows[2]["body"]
    assert rows[3]["section"].startswith("[별표 2] > 2.")


def test_s01_annex_reference_expands_only_the_cited_group():
    doc = {"title": "관리 지침", "version_label": "2025-08", "source_url": "https://example.org/s01",
           "chunks": [
               {"section": "[별표 2] > 1. 고시에 따른 표현", "body": "실증 대상: 여드름성 피부\n비고: 인체 적용시험"},
               {"section": "[별표 2] > 1. 고시에 따른 표현", "body": "실증 대상: 콜라겐 증가\n비고: 별도 실증"},
               {"section": "[별표 2] > 2. 효능·효과", "body": "실증 대상: 수분감"},
           ]}
    row = {"document_key": "pdf:S01", "section": "[별표 1] > 질병 관련",
           "body": "금지 표현: 여드름\n비고: 단, [별표2] 1.에 해당하는 표현은 제외"}
    related = s01_annex_related(row, doc)
    assert len(related) == 2
    assert all(item.relation == "annex" and item.document_key == "pdf:S01" for item in related)
    assert s01_annex_related({**row, "body": "금지 표현: 여드름"}, doc) == []


def test_agent_context_includes_linked_annex_without_reindexing_its_text():
    context = basis_context([{
        "sourceName": "관리 지침", "article": "[별표 1] > 질병 관련",
        "original": "금지 표현: 여드름\n비고: 단, [별표2] 1.에 해당하는 표현은 제외",
        "related": [{"relation": "annex", "article": "[별표 2] > 1. 고시에 따른 표현",
                     "original": "실증 대상: 여드름성 피부에 사용에 적합\n비고: 인체 적용시험 자료로 입증"}],
    }])
    assert "금지 표현: 여드름" in context
    assert "여드름성 피부에 사용에 적합" in context
    assert "인체 적용시험 자료로 입증" in context
    assert "표현별 적용 여부 확인 필요" in context


def test_s01_ingest_uses_table_rows_instead_of_flattened_annex_text(tmp_path):
    path = tmp_path / "S01.pdf"
    path.write_bytes(b"synthetic")
    source = {"id": "S01", "title": "관리 지침", "provider": "mfds", "official_page": "https://example.org",
              "acquisition": {"method": "pdf", "collection": "rules", "source_type": "guideline"}}
    pages = ["[별표 1]\n" + "여드름 비고 [별표2] 1.에 해당하는 표현은 제외 " * 25]
    row = {"section": "[별표 1] > 질병 관련", "body": "금지 표현: 여드름\n비고: [별표2] 1. 제외",
           "page_from": 1, "page_until": 1}
    before = from_pdf(source, path, pages=pages)
    after = from_pdf(source, path, pages=pages, extra_chunks=[row])
    assert len(before.chunks) == 1
    assert [chunk["section"] for chunk in after.chunks] == [row["section"]]
    assert before.content_sha256 != after.content_sha256


def test_s01_annex_row_cut_at_page_break_stays_one_chunk(monkeypatch):
    pages = [
        FakePage("[별표 1]\n화장품 표시·광고의 표현 범위", [
            ["구 분", "금 지 표 현", "비 고"],
            ["인체 유래 성분\n관련 표현", "· 줄기세포 화장품\n· 특정성분이 들어 있는 것으", "안전기준 참고"],
        ]),
        FakePage("구 분 금 지 표 현 비 고", [
            ["구 분", "금 지 표 현", "비 고"],
            ["", "로 오인할 수 있는 표현", ""],
            ["", "· 엑소좀 원액", "단, 식물 엑소좀 제외"],
        ]),
    ]
    import pdfplumber
    monkeypatch.setattr(pdfplumber, "open", lambda _: FakePdf(pages))
    rows = s01_annex_rows(Path("S01.pdf"))
    assert len(rows) == 2
    assert rows[0]["body"] == ("금지 표현: · 줄기세포 화장품\n· 특정성분이 들어 있는 것으\n로 오인할 수 있는 표현"
                               "\n비고: 안전기준 참고")
    assert (rows[0]["page_from"], rows[0]["page_until"]) == (1, 2)
    assert rows[1]["section"] == "[별표 1] > 인체 유래 성분 관련 표현"  # a new '·' item on the next page is its own row
