"""Read a public PDF that has no reviewed MD: the shared page reader, with OCR for the pages that need it.

Pages with a text layer are read as they are. Pages without one (scans, picture tables, slides) are read by PaddleOCR
(Tesseract if it fails), the same as ad and evidence files. A vision model then transcribes each OCR page on its own
and the figures are compared; a page where they differ is left out, because a wrong number in a rule or a case would
be quoted as a basis. A page the model cannot read reliably is kept but marked as not compared. Every OCR page carries AUTO_OCR, so a hit from it shows as machine-read and not yet reviewed.
A reviewed MD always wins over this path (batch.read_source_pages).
"""
import os
from pathlib import Path

from capstone_shared import figure_check
from capstone_shared.reading import read_pdf

from .pdf_vision import AUTO_OCR, AUTO_OCR_UNCHECKED, PdfVisionError, _image_url, transcribe_page

SHORT_OCR = 20  # characters; a page this empty has no figure to disagree about


def vision_crosscheck(client=None, transcriber=transcribe_page):
    """A page cross-check that asks a vision model, or None when no model is configured (pages stay unverified)."""
    if client is None:
        key = os.getenv("OPENAI_API_KEY", "").strip()
        if not key:
            return None
        from openai import OpenAI
        client = OpenAI(api_key=key, timeout=90, max_retries=2)

    def check(image, ocr_text: str) -> dict:
        if len("".join(ocr_text.split())) < SHORT_OCR:
            return {"status": "match", "figures": 0}
        try:
            seen = transcriber(_image_url(image), "", client=client)
        except PdfVisionError:
            # The model could not read the page reliably, so it can neither confirm nor contradict the OCR. That is
            # "not compared" (the page is kept and marked so), not a disagreement: only a figure that differs excludes.
            return {"status": "unavailable", "reason": "VISION_UNCERTAIN"}
        return figure_check.compare(seen, ocr_text)

    return check


def mark(text: str, marker: str = AUTO_OCR) -> str:
    """The marker on its own line after the page's first line, where the curated markers go."""
    title, _, rest = text.partition("\n")
    return f"{title}\n{marker}\n{rest}".rstrip()


def read_pages_with_ocr(path: Path, page_numbers: set[int] | None, on_warning, *, ocr=None, crosscheck="vision"
                        ) -> tuple[list[str], dict[int, str]]:
    """(page texts, trust per page). trust: text | auto_ocr | auto_ocr_unchecked | excluded | unread."""
    check = vision_crosscheck() if crosscheck == "vision" else crosscheck
    result = read_pdf(path.read_bytes(), max_pages=500, max_bytes=None, page_limit_error="PUBLIC_DOCUMENT_PAGE_LIMIT",
                      only_pages=page_numbers, crosscheck=check, ocr=ocr)
    pages, trust = [], {}
    for page in result.pages:
        if page.kind == "text" or not (page_numbers is None or page.number in page_numbers):
            pages.append(page.text)
            trust[page.number] = "text"
        elif page.number in result.unread_pages or not page.text.strip():
            pages.append("")
            trust[page.number] = "unread"
            on_warning({"page": page.number, "reason": "RAG_PDF_OCR_UNREAD"})
        elif page.check and page.check.get("status") == "needs_review":
            pages.append("")
            trust[page.number] = "excluded"
            on_warning({"page": page.number, "reason": "RAG_PDF_OCR_VLM_MISMATCH",
                        "ocrOnly": page.check.get("ocrOnly"), "visionOnly": page.check.get("visionOnly")})
        else:
            checked = bool(page.check) and page.check.get("status") == "match"
            pages.append(mark(page.text, AUTO_OCR if checked else AUTO_OCR_UNCHECKED))
            trust[page.number] = "auto_ocr" if checked else "auto_ocr_unchecked"
            if not checked:
                on_warning({"page": page.number, "reason": "RAG_PDF_OCR_UNCHECKED"})
    return pages, trust


