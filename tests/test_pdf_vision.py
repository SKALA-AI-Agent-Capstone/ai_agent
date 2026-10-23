"""Image table extraction must stay selective and fail closed on uncertain text."""
import sys
from types import SimpleNamespace

import pytest

from capstone_rag_mcp.pdf_vision import PdfVisionError, needs_vision, transcribe_page


class Page:
    width = 100
    height = 100
    images = [{"x0": 0, "x1": 90, "top": 10, "bottom": 90}]

    def __init__(self, text):
        self.text = text

    def extract_text(self):
        return self.text


def test_image_heavy_short_page_is_selected_but_searchable_page_is_not():
    assert needs_vision(Page("표시광고 기준"), "표시광고 기준")
    assert not needs_vision(Page("본문 " * 130), "본문 " * 130)
    page = Page("표")
    page.images = [{"x0": 0, "x1": 20, "top": 0, "bottom": 20}]
    assert not needs_vision(page, "표")


def test_uncertain_vision_result_is_not_indexed():
    class Responses:
        def parse(self, **kwargs):
            return SimpleNamespace(output_parsed=SimpleNamespace(text="개정 후 표의 수치가 모호함", uncertain=True))
    client = SimpleNamespace(responses=Responses())
    with pytest.raises(PdfVisionError, match="REVIEW_REQUIRED"):
        transcribe_page("data:image/jpeg;base64,fixture", "제목", client=client)
    with pytest.raises(PdfVisionError, match="REVIEW_REQUIRED"):
        transcribe_page("data:image/jpeg;base64,fixture", "제목", client=SimpleNamespace(responses=SimpleNamespace(
            parse=lambda **kw: SimpleNamespace(output_parsed=SimpleNamespace(text="짧음", uncertain=False)))))


def test_s04_enhanced_table_stays_on_its_original_page(tmp_path):
    from capstone_rag_mcp.ingest import from_pdf
    source = {"id": "S04", "title": "2026 표시광고 지침", "reported_version_date": "2026",
              "provider": "mfds", "official_page": "https://www.mfds.go.kr/test",
              "acquisition": {"method": "pdf", "collection": "mixed_rules_cases", "source_type": "guideline"}}
    path = tmp_path / "S04.pdf"
    path.write_bytes(b"%PDF-test")
    pages = ["표지", "규정 A " * 200, "규정 B " * 200,
             "추출물 함량\n[이미지 자동 전사 - 원본 대조 필요]\n개정 전: 용매 제외\n개정 후: 용매 포함\n<예시> 녹차추출물 2%"]
    doc = from_pdf(source, path, pages=pages, page_range=(2, 4), collection="rules", key_suffix=":rules")
    assert [chunk["page_from"] for chunk in doc.chunks] == [2, 3, 4]
    assert "개정 후: 용매 포함" in doc.chunks[2]["body"]
    assert "규정 B" not in doc.chunks[2]["body"]


