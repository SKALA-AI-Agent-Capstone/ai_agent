"""One page reader for PDFs and images: the text layer where it exists, OCR only where it does not.

Per page, by what the page holds:
  text          the text layer is enough              -> its text, as is (exact; OCR would only add errors)
  scan          (almost) no text layer                -> OCR (PaddleOCR; Tesseract if it is missing or fails)
  mixed         little text and a large picture       -> text layer plus the OCR lines the layer lacks
  garbled       many (cid:N) glyphs without a mapping -> OCR
  layered_scan  one picture of a page with a text layer on top (a scanner's own OCR)
                                                      -> OCR too; the layer is kept only if the two agree

Every page records how it was read, so a result can be trusted or re-read by what produced it. READER_VERSION
changes whenever a threshold or an engine changes; stored results carry it, and a different version means the
file is read again.
"""
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from io import BytesIO
import re
import time
from typing import Callable

from PIL import Image
import pdfplumber
import pypdfium2 as pdfium

from . import ocr_engine

READER_VERSION = "2026.10.23.1"
MAX_BYTES = 25 * 1024 * 1024
MAX_PAGES = 80
MAX_PIXELS = 40_000_000

TEXT_MIN = 40             # characters (without spaces) below which a page has no usable text layer
MIXED_TEXT_MAX = 350      # a page with less text and a large picture reads its picture too
MIXED_IMAGE_RATIO = 0.35
LAYERED_SCAN_RATIO = 0.9  # picture share of the page above which a text layer is a scanner's OCR
GARBLED_RATIO = 0.05      # share of the text made of (cid:N) glyphs
LAYER_AGREEMENT = 0.6

CID = re.compile(r"\(cid:\d+\)")


def squash(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


@dataclass
class PageText:
    number: int
    kind: str
    method: str            # text_layer | paddle | tesseract | text_layer+paddle | ... | none
    text: str
    confidence: float | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def chars(self) -> int:
        return len(squash(self.text))


@dataclass
class ReadResult:
    pages: list[PageText]
    page_count: int
    reader_version: str = READER_VERSION
    warnings: list[str] = field(default_factory=list)
    unread_pages: list[int] = field(default_factory=list)

    @property
    def methods(self) -> list[str]:
        return sorted({page.method for page in self.pages})

    def text(self) -> str:
        return "\n".join(page.text for page in self.pages if page.text)


OcrCall = Callable[[Image.Image, str], ocr_engine.OcrOutput]


def classify_page(text: str, image_ratio: float) -> str:
    cid_chars = sum(len(token) for token in CID.findall(text))
    if cid_chars and cid_chars / max(1, len(text)) >= GARBLED_RATIO:
        return "garbled"
    size = len(squash(CID.sub("", text)))
    if size < TEXT_MIN:
        return "scan"
    if image_ratio >= LAYERED_SCAN_RATIO:
        return "layered_scan"
    if image_ratio >= MIXED_IMAGE_RATIO and size < MIXED_TEXT_MAX:
        return "mixed"
    return "text"


def image_ratio(page) -> float:
    area = float(page.width * page.height)
    covered = sum(max(0, image["x1"] - image["x0"]) * max(0, image["bottom"] - image["top"]) for image in page.images)
    return min(1.0, covered / area) if area else 0.0


def merge_unique(layer: str, ocr: str) -> str:
    """The text layer, then the OCR lines it does not already contain."""
    known = squash(layer)
    extra = [line for line in ocr.splitlines() if len(squash(line)) >= 2 and squash(line) not in known]
    return "\n".join(part for part in (layer.strip(), "\n".join(extra)) if part)


def agreement(layer: str, ocr: str) -> float:
    a, b = squash(layer)[:2000], squash(ocr)[:2000]
    return SequenceMatcher(None, a, b, autojunk=False).ratio() if a and b else 0.0


def _render(pdf_page, scale: float) -> Image.Image:
    width, height = pdf_page.get_size()
    if width * height * scale * scale > MAX_PIXELS:
        scale = max(0.5, (MAX_PIXELS / (width * height)) ** 0.5)
    bitmap = pdf_page.render(scale=scale)
    try:
        return bitmap.to_pil().convert("RGB")
    finally:
        bitmap.close()


def read_pdf(raw: bytes, *, max_pages: int = MAX_PAGES, prefer: str = "auto", ocr_budget_seconds: float | None = None,
             max_ocr_pages: int | None = None, scale: float = 2.0, ocr: OcrCall | None = None) -> ReadResult:
    """Read every page. ocr_budget_seconds and max_ocr_pages bound the OCR; pages left unread are listed."""
    if len(raw) > MAX_BYTES:
        raise ValueError("FILE_TOO_LARGE")
    ocr = ocr or ocr_engine.read_image
    deadline = time.monotonic() + ocr_budget_seconds if ocr_budget_seconds else None
    pages: list[PageText] = []
    unread: list[int] = []
    warnings: list[str] = []
    ocr_pages = 0
    with pdfplumber.open(BytesIO(raw)) as text_pdf, pdfium.PdfDocument(raw) as visual_pdf:
        count = len(text_pdf.pages)
        if not 1 <= count <= max_pages:
            raise ValueError("EVIDENCE_PAGE_LIMIT")
        for index, page in enumerate(text_pdf.pages):
            number = index + 1
            layer_raw = page.extract_text() or ""
            cid_count = len(CID.findall(layer_raw))
            layer = CID.sub("", layer_raw).strip()
            kind = classify_page(layer_raw, image_ratio(page))
            notes = [f"CID_GLYPHS:{cid_count}"] if cid_count else []
            if kind == "text":
                pages.append(PageText(number, kind, "text_layer", layer, None, notes))
                continue
            over_budget = (deadline is not None and time.monotonic() > deadline) or \
                (max_ocr_pages is not None and ocr_pages >= max_ocr_pages)
            if over_budget:
                pages.append(PageText(number, kind, "text_layer" if layer else "none", layer, None, notes + ["OCR_SKIPPED"]))
                if len(squash(layer)) < TEXT_MIN:
                    unread.append(number)
                continue
            try:
                image = _render(visual_pdf[index], scale)
                output = ocr(image, prefer)
            except Exception:
                pages.append(PageText(number, kind, "text_layer" if layer else "none", layer, None, notes + ["OCR_FAILED"]))
                if len(squash(layer)) < TEXT_MIN:
                    unread.append(number)
                continue
            ocr_pages += 1
            if output.fell_back:
                notes.append("PADDLE_FAILED_TESSERACT_USED")
            if kind == "mixed":
                pages.append(PageText(number, kind, f"text_layer+{output.engine}", merge_unique(layer, output.text),
                                      output.confidence, notes))
            elif kind == "layered_scan" and agreement(layer, output.text) >= LAYER_AGREEMENT:
                pages.append(PageText(number, kind, "text_layer", layer, output.confidence, notes))
            else:
                if kind == "layered_scan":
                    notes.append("TEXT_LAYER_DISAGREES_WITH_OCR")
                text = output.text if output.text.strip() else layer
                pages.append(PageText(number, kind, output.engine if output.text.strip() else "text_layer", text,
                                      output.confidence, notes))
            if not pages[-1].text.strip():
                unread.append(number)
    if unread:
        warnings.append("UNREAD_PAGES:" + ",".join(str(n) for n in unread))
    return ReadResult(pages, count, READER_VERSION, warnings, unread)


def read_image_file(raw: bytes, *, prefer: str = "auto", ocr: OcrCall | None = None) -> ReadResult:
    if len(raw) > MAX_BYTES:
        raise ValueError("FILE_TOO_LARGE")
    try:
        image = Image.open(BytesIO(raw))
        image.load()
    except Exception:
        raise ValueError("PDF_OR_IMAGE_REQUIRED") from None
    ocr = ocr or ocr_engine.read_image
    try:
        output = ocr(image.convert("RGB"), prefer)
    except Exception:
        return ReadResult([PageText(1, "scan", "none", "", None, ["OCR_FAILED"])], 1, READER_VERSION,
                          ["UNREAD_PAGES:1"], [1])
    notes = ["PADDLE_FAILED_TESSERACT_USED"] if output.fell_back else []
    page = PageText(1, "scan", output.engine, output.text, output.confidence, notes)
    return ReadResult([page], 1, READER_VERSION, [] if page.text.strip() else ["UNREAD_PAGES:1"],
                      [] if page.text.strip() else [1])


def read_document(raw: bytes, **options) -> ReadResult:
    """A PDF or an image file."""
    if raw.startswith(b"%PDF-"):
        return read_pdf(raw, **options)
    options.pop("max_pages", None), options.pop("ocr_budget_seconds", None)
    options.pop("max_ocr_pages", None), options.pop("scale", None)
    return read_image_file(raw, **options)
