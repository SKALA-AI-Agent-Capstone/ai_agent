"""Image table extraction must stay selective and fail closed on uncertain text."""
import sys
from types import SimpleNamespace

import pytest

from capstone_rag_mcp.pdf_vision import (PdfVisionError, merge_page_text, needs_vision,
                                         read_pages_for_index, transcribe_page)


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


def test_only_image_page_is_transcribed_before_chunking(monkeypatch, tmp_path):
    first = Page("본문 " * 130)
    second = Page("추출물 함량 표시 기준")
    fake_pdf = SimpleNamespace(pages=[first, second], __enter__=lambda self: self, __exit__=lambda *args: None)
    class OpenPdf:
        def __enter__(self):
            return fake_pdf
        def __exit__(self, *args):
            pass
    monkeypatch.setitem(sys.modules, "pdfplumber", SimpleNamespace(open=lambda path: OpenPdf()))
    (tmp_path / "S04.pdf").write_bytes(b"%PDF-1.7 synthetic")
    calls = []
    def transcribe(image, native):
        calls.append((image, native))
        return "개정 전: 추출용매 제외\n개정 후: 추출용매 포함\n<예시> 녹차추출물 2%"
    pages = read_pages_for_index(tmp_path / "S04.pdf", page_numbers={2}, transcriber=transcribe,
                                 renderer=lambda path, index: f"image:{index}")
    assert pages[0] == first.text.strip()  # the shared reader trims the text layer
    assert calls == [("image:1", second.text)]
    assert "개정 후: 추출용매 포함" in pages[1]
    assert pages[1].startswith(second.text)


def test_uncertain_vision_result_is_not_indexed():
    class Responses:
        def parse(self, **kwargs):
            return SimpleNamespace(output_parsed=SimpleNamespace(text="개정 후 표의 수치가 모호함", uncertain=True))
    client = SimpleNamespace(responses=Responses())
    with pytest.raises(PdfVisionError, match="REVIEW_REQUIRED"):
        transcribe_page("data:image/jpeg;base64,fixture", "제목", client=client)
    with pytest.raises(PdfVisionError, match="NO_NEW_TEXT"):
        merge_page_text("제목", "제목")


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


def test_uncertain_page_retries_cropped_table_and_keeps_other_pages(monkeypatch, tmp_path):
    first = Page("본문 " * 130)
    second = Page("추출물 함량 표시 기준")
    class OpenPdf:
        def __enter__(self):
            return SimpleNamespace(pages=[first, second])
        def __exit__(self, *args):
            pass
    monkeypatch.setitem(sys.modules, "pdfplumber", SimpleNamespace(open=lambda path: OpenPdf()))
    (tmp_path / "S04.pdf").write_bytes(b"%PDF-1.7 synthetic")
    calls = []
    def transcribe(images, native):
        calls.append(images)
        if isinstance(images, str):
            raise PdfVisionError("RAG_PDF_VISION_REVIEW_REQUIRED")
        return "개정 전: 추출용매 제외\n개정 후: 추출용매 포함\n<예시> 녹차추출물 2%"
    warnings = []
    pages = read_pages_for_index(tmp_path / "S04.pdf", transcriber=transcribe,
                                 renderer=lambda path, index: "full",
                                 retry_renderer=lambda path, index: ["left", "right"],
                                 on_warning=warnings.append)
    assert calls == ["full", ["left", "right"]]
    assert warnings == []
    assert pages[0] == first.text.strip()  # the shared reader trims the text layer
    assert "개정 후: 추출용매 포함" in pages[1]


@pytest.mark.asyncio
async def test_uncertain_page_does_not_stop_rest_of_s04(monkeypatch, tmp_path):
    from capstone_rag_mcp import batch
    source = {"id": "S04", "title": "2026 표시광고 지침", "reported_version_date": "2026",
              "provider": "mfds", "official_page": "https://www.mfds.go.kr/test",
              "acquisition": {"method": "pdf", "collection": "mixed_rules_cases",
                              "source_type": "guideline",
                              "page_ranges": {"rules": [2, 7], "cases": [8, 9]}}}
    path = tmp_path / "S04.pdf"
    path.write_bytes(b"%PDF-test")
    monkeypatch.setattr(batch, "sources", lambda: [source])
    monkeypatch.setattr(batch, "find_pdf", lambda source_id: path)
    def read(path, *, page_numbers, on_warning):
        assert 4 in page_numbers and 10 not in page_numbers
        on_warning({"page": 4, "reason": "RAG_PDF_VISION_REVIEW_REQUIRED"})
        return ["cover", "규정 1", "규정 2", "", "규정 4", "규정 5", "규정 6", "사례 1", "사례 2"]
    monkeypatch.setattr(batch, "read_pages_for_index", read)
    from capstone_rag_mcp import pdf_curator
    # One image page among the indexed pages is below the image-heavy share that requires MD.
    monkeypatch.setattr(pdf_curator, "image_page_numbers", lambda path, numbers: ([4], 8))
    documents, warnings = await batch.collect_pdf()
    assert [d.document_key for d in documents] == ["pdf:S04:rules", "pdf:S04:cases"]
    assert warnings == [{"id": "S04", "page": 4, "reason": "RAG_PDF_VISION_REVIEW_REQUIRED"}]
    assert all(chunk["page_from"] != 4 for doc in documents for chunk in doc.chunks)


def test_persistent_uncertainty_skips_only_that_page(monkeypatch, tmp_path):
    first = Page("본문 " * 130)
    second = Page("추출물 함량 표시 기준")
    class OpenPdf:
        def __enter__(self):
            return SimpleNamespace(pages=[first, second])
        def __exit__(self, *args):
            pass
    monkeypatch.setitem(sys.modules, "pdfplumber", SimpleNamespace(open=lambda path: OpenPdf()))
    (tmp_path / "S04.pdf").write_bytes(b"%PDF-1.7 synthetic")
    warnings = []
    def uncertain(images, native):
        raise PdfVisionError("RAG_PDF_VISION_REVIEW_REQUIRED")
    pages = read_pages_for_index(tmp_path / "S04.pdf", transcriber=uncertain,
                                 renderer=lambda path, index: "full",
                                 retry_renderer=lambda path, index: ["left", "right"],
                                 on_warning=warnings.append)
    assert pages == [first.text.strip(), ""]
    assert warnings == [{"page": 2, "reason": "RAG_PDF_VISION_REVIEW_REQUIRED"}]
