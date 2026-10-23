"""A public PDF without a reviewed MD is read like ad and evidence files: text layer as it is, OCR for the rest,
a vision model comparing the figures, and every OCR page marked as machine-read."""
import hashlib
from types import SimpleNamespace

import pytest

from capstone_rag_mcp import batch, ingest, pdf_ocr
from capstone_rag_mcp.pdf_vision import AUTO_OCR, AUTO_OCR_UNCHECKED, PdfVisionError
from capstone_shared.ocr_engine import OcrOutput
from test_reading import FakeOcr, combine, picture_pdf, text_pdf


def source_pdf(tmp_path, pages=2, name="X.pdf"):
    path = tmp_path / name
    path.write_bytes(combine(text_pdf("TEXT PAGE WITH A LAYER " * 4), picture_pdf(pages)))
    return path


MATCH = lambda image, text: {"status": "match", "figures": 1}  # noqa: E731


def test_text_pages_stay_as_they_are_and_ocr_pages_are_marked_after_their_first_line(tmp_path):
    path = source_pdf(tmp_path)
    ocr = FakeOcr("표 제목\n효과 22.19% 증가")
    warnings = []
    pages, trust = pdf_ocr.read_pages_with_ocr(path, None, warnings.append, ocr=ocr, crosscheck=MATCH)
    assert pages[0].startswith("TEXT PAGE") and AUTO_OCR not in pages[0]
    assert pages[1] == f"표 제목\n{AUTO_OCR}\n효과 22.19% 증가"
    assert trust == {1: "text", 2: "auto_ocr", 3: "auto_ocr"} and ocr.calls == 2 and warnings == []


def test_a_page_where_ocr_and_the_vision_model_disagree_on_a_figure_is_left_out(tmp_path):
    path = source_pdf(tmp_path)
    verdicts = iter([{"status": "needs_review", "visionOnly": ["22.19"], "ocrOnly": ["22.18"]}, {"status": "match", "figures": 1}])
    warnings = []
    pages, trust = pdf_ocr.read_pages_with_ocr(path, None, warnings.append, ocr=FakeOcr(), crosscheck=lambda i, t: next(verdicts))
    assert pages[1] == "" and AUTO_OCR in pages[2]
    assert trust[2] == "excluded" and trust[3] == "auto_ocr"
    assert warnings == [{"page": 2, "reason": "RAG_PDF_OCR_VLM_MISMATCH", "ocrOnly": ["22.18"], "visionOnly": ["22.19"]}]


def test_without_a_vision_model_ocr_pages_are_kept_but_say_their_figures_were_not_compared(tmp_path):
    path = source_pdf(tmp_path, pages=1)
    warnings = []
    pages, trust = pdf_ocr.read_pages_with_ocr(path, None, warnings.append, ocr=FakeOcr(), crosscheck=None)
    assert AUTO_OCR_UNCHECKED in pages[1] and AUTO_OCR not in pages[1] and trust[2] == "auto_ocr_unchecked"
    assert warnings == [{"page": 2, "reason": "RAG_PDF_OCR_UNCHECKED"}]


def test_a_page_the_vision_model_cannot_read_is_kept_and_marked_not_compared(tmp_path):
    warnings = []
    pages, trust = pdf_ocr.read_pages_with_ocr(source_pdf(tmp_path, pages=1), None, warnings.append, ocr=FakeOcr(),
                                               crosscheck=lambda i, t: {"status": "unavailable", "reason": "VISION_UNCERTAIN"})
    assert AUTO_OCR_UNCHECKED in pages[1] and trust[2] == "auto_ocr_unchecked"
    assert warnings == [{"page": 2, "reason": "RAG_PDF_OCR_UNCHECKED"}]


def test_a_page_ocr_could_not_read_is_left_out_with_a_warning(tmp_path):
    def broken(image, prefer):
        raise RuntimeError("engine down")

    warnings = []
    pages, trust = pdf_ocr.read_pages_with_ocr(source_pdf(tmp_path, pages=1), None, warnings.append, ocr=broken, crosscheck=MATCH)
    assert pages[1] == "" and trust[2] == "unread" and warnings == [{"page": 2, "reason": "RAG_PDF_OCR_UNREAD"}]


def test_only_the_indexed_page_range_is_read_by_ocr(tmp_path):
    path = source_pdf(tmp_path, pages=3)
    ocr = FakeOcr()
    pages, trust = pdf_ocr.read_pages_with_ocr(path, {1, 2}, lambda w: None, ocr=ocr, crosscheck=MATCH)
    assert ocr.calls == 1 and trust[2] == "auto_ocr"
    assert pages[2] == "" and trust[3] == "text" and trust[4] == "text"   # outside the range: not read, not reported


def test_a_reviewed_md_is_used_and_nothing_is_ocrd(monkeypatch, tmp_path):
    (tmp_path / "sources").mkdir()
    path = source_pdf(tmp_path / "sources", pages=1)
    md = tmp_path / "curated" / "X.md"
    md.parent.mkdir()
    body = "\n\n".join(f"## {n}쪽\n\n{'> 원본 이미지 대조 완료' if n > 1 else ''}\n\n검수한 {n}쪽" for n in (1, 2))
    md.write_text(f"- SHA-256: `{hashlib.sha256(path.read_bytes()).hexdigest()}`\n\n{body}\n", encoding="utf-8")
    monkeypatch.setattr(batch, "read_pages_with_ocr", lambda *a, **k: pytest.fail("OCR must not run when an MD matches"))
    pages, curated = batch.read_source_pages(path, None, lambda w: None)
    assert curated and pages == ["검수한 1쪽", "검수한 2쪽"]


def test_without_an_md_the_batch_reads_the_pdf_and_reports_trust_per_page(monkeypatch, tmp_path):
    path = source_pdf(tmp_path, pages=1)
    monkeypatch.setattr(pdf_ocr, "vision_crosscheck", lambda: MATCH)
    real = pdf_ocr.read_pdf
    monkeypatch.setattr(pdf_ocr, "read_pdf", lambda raw, **kw: real(raw, **{**kw, "ocr": FakeOcr()}))
    warnings = []
    pages, curated = batch.read_source_pages(path, None, warnings.append)
    assert not curated and AUTO_OCR in pages[1]
    assert warnings[-1] == {"reason": "PDF_PAGE_TRUST", "mode": "ocr", "pages": {"text": [1], "auto_ocr": [2]}, "report": None}


def test_a_new_reader_version_gives_ocr_pages_a_new_document_version(monkeypatch, tmp_path):
    path = source_pdf(tmp_path, pages=1)
    source = {"id": "X", "title": "t", "official_page": "u", "acquisition": {"collection": "rules", "source_type": "guideline"}}
    pages = ["본문", pdf_ocr.mark("표\n효과 1.5%")]
    first = ingest.from_pdf(source, path, pages=pages).content_sha256
    monkeypatch.setattr(ingest, "READER_VERSION", "a-newer-reader")
    assert ingest.from_pdf(source, path, pages=pages).content_sha256 != first
    plain = ["본문", "다음 쪽"]
    assert ingest.from_pdf(source, path, pages=plain).content_sha256 == ingest.from_pdf(source, path, pages=plain).content_sha256


def test_the_vision_check_compares_figures_and_an_unreadable_page_is_not_compared_rather_than_excluded(monkeypatch):
    from PIL import Image
    image = Image.new("RGB", (40, 40), "white")
    ocr = "효과 22.19% 증가 22.19% 개선 수치 표시 기준"
    check = pdf_ocr.vision_crosscheck(client=object(), transcriber=lambda url, native, client=None: "효과 22.19% 증가 22.19% 개선")
    assert check(image, ocr)["status"] == "match"
    check = pdf_ocr.vision_crosscheck(client=object(), transcriber=lambda url, native, client=None: "효과 22.91% 증가 22.19% 개선")
    assert check(image, ocr)["status"] == "needs_review"

    def uncertain(url, native, client=None):
        raise PdfVisionError("RAG_PDF_VISION_REVIEW_REQUIRED")

    verdict = pdf_ocr.vision_crosscheck(client=object(), transcriber=uncertain)(image, ocr)
    assert verdict == {"status": "unavailable", "reason": "VISION_UNCERTAIN"}
    # a page with almost no text has no figure to disagree about, and the model is not asked
    assert pdf_ocr.vision_crosscheck(client=object(), transcriber=uncertain)(image, "표지")["status"] == "match"
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert pdf_ocr.vision_crosscheck() is None
