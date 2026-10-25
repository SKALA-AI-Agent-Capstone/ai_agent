"""One page reader for PDFs and images: the text layer where it exists, OCR only where it does not.

Per page, by what the page holds:
  text          the text layer is enough              -> its text, as is (exact; OCR would only add errors)
  scan          (almost) no text layer                -> OCR (PaddleOCR; Tesseract if it is missing or fails)
  mixed         little text and a large picture       -> text layer plus the OCR lines the layer lacks
  garbled       many (cid:N) glyphs without a mapping -> OCR
  layered_scan  one picture of a page with a text layer on top (a scanner's own OCR)
                                                      -> OCR too; the layer is kept only if the two agree

A page read by OCR can also be cross-checked by a second reader (a vision model, passed in as `crosscheck`): it says
whether the two agree on the figures, and a page where they differ is listed in check_pages for a person to look at.
The cross-check never replaces the OCR text; it only marks the page.

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
import pypdfium2 as pdfium

from . import ocr_engine

READER_VERSION = "2026.10.26.1"
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
    check: dict | None = None   # second reader's verdict: {"status": match | needs_review | skipped | unavailable, ...}
    image_ratio: float = 0.0    # share of the page covered by pictures
    lines: list[dict] | None = None  # with_lines: text-layer lines {"index", "text", "bbox" (page ratios)}

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
    check_pages: list[int] = field(default_factory=list)  # OCR pages where the second reader disagreed

    @property
    def methods(self) -> list[str]:
        return sorted({page.method for page in self.pages})

    def text(self) -> str:
        return "\n".join(page.text for page in self.pages if page.text)


OcrCall = Callable[[Image.Image, str], ocr_engine.OcrOutput]
CrossCheck = Callable[[Image.Image, str], dict]   # (page image, OCR text) -> verdict; may raise
CHECK_MAX_PAGES = 30
CHECK_BUDGET_SECONDS = 200


class Checker:
    """Runs the cross-check on OCR pages within a page and time budget, and never lets it break the read."""

    def __init__(self, crosscheck: CrossCheck | None, max_pages: int | None = None, budget_seconds: float | None = None):
        self.crosscheck, self.checked = crosscheck, 0
        self.max_pages = CHECK_MAX_PAGES if max_pages is None else max_pages
        self.deadline = time.monotonic() + (CHECK_BUDGET_SECONDS if budget_seconds is None else budget_seconds)

    def run(self, page: "PageText", image: Image.Image, ocr_text: str) -> None:
        if self.crosscheck is None:
            return
        if self.checked >= self.max_pages or time.monotonic() > self.deadline:
            page.check = {"status": "skipped"}
            page.warnings.append("VLM_CHECK_SKIPPED")
            return
        self.checked += 1
        try:
            verdict = self.crosscheck(image, ocr_text)
        except Exception:
            page.check = {"status": "unavailable"}
            page.warnings.append("VLM_CHECK_FAILED")
            return
        page.check = verdict
        if verdict.get("status") == "needs_review":
            page.warnings.append("OCR_VLM_MISMATCH")


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


def text_lines(page) -> list[dict]:
    """Text-layer lines with their boxes as page ratios. index counts every line pdfplumber returns, so ids built from
    it stay the same whether or not a line turns out empty. A glyph without a Unicode mapping stays as "(cid:N)": the
    line points at an exact place, and silently dropping the glyph would turn "탈락" into "탈" without a trace."""
    lines = []
    for index, line in enumerate(page.extract_text_lines()):
        text = line["text"].strip()
        if text:
            lines.append({"index": index, "text": text,
                          "bbox": [line["x0"] / page.width, line["top"] / page.height,
                                   line["x1"] / page.width, line["bottom"] / page.height]})
    return lines


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
             max_ocr_pages: int | None = None, scale: float = 2.0, ocr: OcrCall | None = None,
             crosscheck: CrossCheck | None = None, with_lines: bool = False, max_bytes: int | None = MAX_BYTES,
             page_limit_error: str = "EVIDENCE_PAGE_LIMIT", only_pages: set[int] | None = None) -> ReadResult:
    """Read every page. ocr_budget_seconds and max_ocr_pages bound the OCR; pages left unread are listed.

    max_ocr_pages=0 reads the text layer only and still classifies every page (the RAG index and the ad reader send
    picture pages to a vision model themselves). with_lines adds each text-layer line with its box, for callers that
    must point at where a phrase is. only_pages limits OCR to those page numbers (the RAG index reads only the page
    ranges it indexes); other pages keep their text layer and are not reported as unread."""
    if max_bytes is not None and len(raw) > max_bytes:
        raise ValueError("FILE_TOO_LARGE")
    ocr = ocr or ocr_engine.read_image
    checker = Checker(crosscheck)
    deadline = time.monotonic() + ocr_budget_seconds if ocr_budget_seconds else None
    # pdfium is opened only when a page has to be rendered for OCR: a text-layer read (the RAG index, ad text lines)
    # never loads it.
    rendered: list = []

    def visual(index: int):
        if not rendered:
            rendered.append(pdfium.PdfDocument(raw))
        return rendered[0][index]

    try:
        return _read_pages(raw, max_pages, page_limit_error, with_lines, deadline, max_ocr_pages, ocr, prefer, scale,
                           checker, visual, only_pages)
    finally:
        for document in rendered:
            document.close()


def _read_pages(raw, max_pages, page_limit_error, with_lines, deadline, max_ocr_pages, ocr, prefer, scale, checker,
                visual, only_pages) -> ReadResult:
    import pdfplumber  # looked up per call, so a caller's stand-in module is honoured
    pages: list[PageText] = []
    unread: list[int] = []
    warnings: list[str] = []
    ocr_pages = 0
    with pdfplumber.open(BytesIO(raw)) as text_pdf:
        count = len(text_pdf.pages)
        if not 1 <= count <= max_pages:
            raise ValueError(page_limit_error)
        for index, page in enumerate(text_pdf.pages):
            number = index + 1
            layer_raw = page.extract_text() or ""
            cid_count = len(CID.findall(layer_raw))
            layer = CID.sub("", layer_raw).strip()
            ratio = image_ratio(page)
            kind = classify_page(layer_raw, ratio)
            notes = [f"CID_GLYPHS:{cid_count}"] if cid_count else []
            page_lines = text_lines(page) if with_lines else None

            def add(item: PageText) -> None:
                item.image_ratio, item.lines = ratio, page_lines
                pages.append(item)
            if kind == "text":
                add(PageText(number, kind, "text_layer", layer, None, notes))
                continue
            if only_pages is not None and number not in only_pages:
                add(PageText(number, kind, "text_layer" if layer else "none", layer, None, notes))
                continue
            over_budget = (deadline is not None and time.monotonic() > deadline) or \
                (max_ocr_pages is not None and ocr_pages >= max_ocr_pages)
            if over_budget:
                add(PageText(number, kind, "text_layer" if layer else "none", layer, None, notes + ["OCR_SKIPPED"]))
                if len(squash(layer)) < TEXT_MIN:
                    unread.append(number)
                continue
            try:
                image = _render(visual(index), scale)
                output = ocr(image, prefer)
            except Exception:
                add(PageText(number, kind, "text_layer" if layer else "none", layer, None, notes + ["OCR_FAILED"]))
                if len(squash(layer)) < TEXT_MIN:
                    unread.append(number)
                continue
            ocr_pages += 1
            if output.fell_back:
                notes.append("PADDLE_FAILED_TESSERACT_USED")
            if kind == "mixed":
                add(PageText(number, kind, f"text_layer+{output.engine}", merge_unique(layer, output.text),
                                      output.confidence, notes))
            elif kind == "layered_scan" and agreement(layer, output.text) >= LAYER_AGREEMENT:
                add(PageText(number, kind, "text_layer", layer, output.confidence, notes))
            else:
                if kind == "layered_scan":
                    notes.append("TEXT_LAYER_DISAGREES_WITH_OCR")
                text = output.text if output.text.strip() else layer
                add(PageText(number, kind, output.engine if output.text.strip() else "text_layer", text,
                                      output.confidence, notes))
            if not pages[-1].text.strip():
                unread.append(number)
            checker.run(pages[-1], image, output.text)
    if unread:
        warnings.append("UNREAD_PAGES:" + ",".join(str(n) for n in unread))
    return ReadResult(pages, count, READER_VERSION, warnings, unread,
                      [page.number for page in pages if page.check and page.check.get("status") == "needs_review"])


def read_image_file(raw: bytes, *, prefer: str = "auto", ocr: OcrCall | None = None,
                    crosscheck: CrossCheck | None = None) -> ReadResult:
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
    Checker(crosscheck).run(page, image.convert("RGB"), output.text)
    return ReadResult([page], 1, READER_VERSION, [] if page.text.strip() else ["UNREAD_PAGES:1"],
                      [] if page.text.strip() else [1],
                      [1] if page.check and page.check.get("status") == "needs_review" else [])


def read_document(raw: bytes, **options) -> ReadResult:
    """A PDF or an image file."""
    if raw.startswith(b"%PDF-"):
        return read_pdf(raw, **options)
    options.pop("max_pages", None), options.pop("ocr_budget_seconds", None)
    options.pop("max_ocr_pages", None), options.pop("scale", None)
    return read_image_file(raw, **options)
