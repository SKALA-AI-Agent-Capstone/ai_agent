"""The shared page reader: text layer first, OCR only for pages without one, every page says how it was read."""
from io import BytesIO

from PIL import Image, ImageDraw
import pypdfium2 as pdfium
import pytest

from capstone_shared import ocr_engine, reading
from capstone_shared.ocr_engine import OcrOutput
from capstone_shared.reading import classify_page, merge_unique, read_document, read_image_file, read_pdf


def text_pdf(text: str) -> bytes:
    content = f"BT /F1 12 Tf 50 730 Td ({text}) Tj ET".encode()
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
               b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
               b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
               b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream"]
    raw, offsets = b"%PDF-1.4\n", []
    for number, body in enumerate(objects, 1):
        offsets.append(len(raw))
        raw += str(number).encode() + b" 0 obj\n" + body + b"\nendobj\n"
    xref = len(raw)
    raw += b"xref\n0 6\n0000000000 65535 f \n" + b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    return raw + f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()


def picture_pdf(pages: int = 1) -> bytes:
    """Pages that are only a picture: no text layer at all."""
    images = []
    for n in range(pages):
        image = Image.new("RGB", (600, 800), "white")
        ImageDraw.Draw(image).rectangle((50, 50, 550, 300), outline="black", width=4)
        images.append(image)
    out = BytesIO()
    images[0].save(out, format="PDF", save_all=True, append_images=images[1:])
    return out.getvalue()


def combine(*documents: bytes) -> bytes:
    merged = pdfium.PdfDocument.new()
    for raw in documents:
        source = pdfium.PdfDocument(raw)
        merged.import_pages(source)
    out = BytesIO()
    merged.save(out)
    return out.getvalue()


class FakeOcr:
    def __init__(self, text="스캔 본문 첫 줄\n스캔 본문 둘째 줄", engine="paddle", fell_back=False):
        self.calls, self.text, self.engine, self.fell_back = 0, text, engine, fell_back

    def __call__(self, image, prefer):
        self.calls += 1
        self.prefer = prefer
        return OcrOutput(self.text, 93.0, self.engine, self.fell_back)


REPORT = "SYNTHETIC TEST REPORT STUDY CODE HBSE-MGE-24032 CLINICAL EVIDENCE SAMPLE TEXT"


def test_page_kinds_follow_what_the_page_holds():
    assert classify_page("가" * 400, 0.0) == "text"
    assert classify_page("가" * 10, 0.0) == "scan"                       # no usable text layer
    assert classify_page("가" * 60, 0.7) == "mixed"                      # a title above a picture of the content
    assert classify_page("가" * 600, 0.7) == "text"                      # plenty of text: the picture is decoration
    assert classify_page("가" * 400, 0.95) == "layered_scan"             # a scanner's own OCR on top of a picture
    assert classify_page("가" * 300 + "(cid:1279)(cid:1280)" * 12, 0.0) == "garbled"
    assert classify_page("가" * 600 + "(cid:1279)", 0.0) == "text"        # one stray glyph is not garbled


def test_text_pages_are_read_from_the_layer_and_never_sent_to_ocr():
    ocr = FakeOcr()
    result = read_pdf(text_pdf(REPORT), ocr=ocr)
    assert ocr.calls == 0
    page = result.pages[0]
    assert (page.kind, page.method) == ("text", "text_layer") and "HBSE-MGE-24032" in page.text
    assert result.reader_version == reading.READER_VERSION and result.unread_pages == []


def test_picture_only_pages_are_read_by_ocr_and_say_which_engine():
    ocr = FakeOcr()
    result = read_pdf(combine(text_pdf(REPORT), picture_pdf()), ocr=ocr)
    assert [p.method for p in result.pages] == ["text_layer", "paddle"]
    assert result.pages[1].kind == "scan" and "스캔 본문" in result.pages[1].text
    assert result.methods == ["paddle", "text_layer"] and ocr.calls == 1


def test_a_failed_paddle_read_that_fell_back_is_marked():
    result = read_pdf(picture_pdf(), ocr=FakeOcr(engine="tesseract", fell_back=True))
    assert result.pages[0].method == "tesseract"
    assert "PADDLE_FAILED_TESSERACT_USED" in result.pages[0].warnings


def test_ocr_budget_leaves_the_rest_unread_instead_of_running_past_the_limit():
    ocr = FakeOcr()
    result = read_pdf(picture_pdf(pages=4), ocr=ocr, max_ocr_pages=2)
    assert ocr.calls == 2
    assert result.unread_pages == [3, 4] and result.warnings == ["UNREAD_PAGES:3,4"]
    assert all("OCR_SKIPPED" in p.warnings for p in result.pages[2:])


def test_zero_budget_reads_only_the_text_layer():
    ocr = FakeOcr()
    result = read_pdf(combine(text_pdf(REPORT), picture_pdf()), ocr=ocr, max_ocr_pages=0)
    assert ocr.calls == 0 and result.unread_pages == [2] and result.pages[0].text


def test_ocr_that_raises_marks_the_page_unread_and_keeps_going():
    def broken(image, prefer):
        raise RuntimeError("engine down")

    result = read_pdf(combine(picture_pdf(), text_pdf(REPORT)), ocr=broken)
    assert result.unread_pages == [1] and "OCR_FAILED" in result.pages[0].warnings
    assert result.pages[1].method == "text_layer"


def test_prefer_tesseract_is_passed_through_for_short_time_limits():
    ocr = FakeOcr()
    read_pdf(picture_pdf(), ocr=ocr, prefer="tesseract")
    assert ocr.prefer == "tesseract"


def test_mixed_page_keeps_the_layer_and_adds_only_the_lines_it_lacks():
    layer = "제목 줄\n본문 한 줄"
    merged = merge_unique(layer, "제목 줄\n그림 속 새 줄\n본문 한 줄")
    assert merged == "제목 줄\n본문 한 줄\n그림 속 새 줄"


def test_scanner_text_layer_is_trusted_only_where_ocr_agrees(monkeypatch):
    assert reading.agreement("같은 문장입니다 숫자 9.71", "같은 문장입니다 숫자 9.71") == 1.0
    assert reading.agreement("같은 문장입니다 숫자 9.71", "전혀 다른 내용의 글") < reading.LAYER_AGREEMENT


def test_images_are_one_scanned_page():
    out = BytesIO()
    Image.new("RGB", (300, 200), "white").save(out, format="PNG")
    result = read_document(out.getvalue(), ocr=FakeOcr())
    assert result.page_count == 1 and result.pages[0].kind == "scan" and result.pages[0].method == "paddle"
    with pytest.raises(ValueError, match="PDF_OR_IMAGE_REQUIRED"):
        read_image_file(b"not an image")


def test_limits_have_specific_error_codes():
    with pytest.raises(ValueError, match="FILE_TOO_LARGE"):
        read_pdf(b"%PDF-" + b"0" * (reading.MAX_BYTES + 1))
    with pytest.raises(ValueError, match="EVIDENCE_PAGE_LIMIT"):
        read_pdf(picture_pdf(pages=3), max_pages=2, ocr=FakeOcr())


def test_engine_choice_falls_back_to_tesseract_and_says_so(monkeypatch):
    image = Image.new("RGB", (50, 50), "white")
    monkeypatch.setattr(ocr_engine, "paddle_enabled", lambda: True)

    def broken(_image):
        raise RuntimeError("paddle failed")

    monkeypatch.setattr(ocr_engine, "paddle_read", broken)
    monkeypatch.setattr(ocr_engine, "tesseract_read", lambda _image: OcrOutput("글", 80.0, "tesseract"))
    output = ocr_engine.read_image(image)
    assert (output.engine, output.fell_back) == ("tesseract", True)
    assert ocr_engine.read_image(image, prefer="tesseract").fell_back is False
