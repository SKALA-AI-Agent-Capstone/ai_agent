"""Read one evidence file and say what it is: pages, issuer, codes, version, and the pages of each product.

This is the single entry the evidence flow calls (the Agent's parse endpoint, and business-mcp's structure_evidence
when a review carries its own evidence). The result is plain JSON, ready to store next to the file's SHA-256.
"""
from . import evidence_identity as identity
from .reading import READER_VERSION, ReadResult, read_document


def describe(result: ReadResult, product_name: str | None = None) -> dict:
    texts = [page.text for page in result.pages]
    who = identity.identify(texts)
    names = identity.product_names(texts)
    ranges = identity.product_pages(texts, names) if names else []
    results = identity.extract_results(texts, names)
    products = [{"name": name, "pages": pages, "ranges": identity.group_ranges(pages),
                 "results": [r for r in results if r["product"] == name]}
                for name, pages in zip(names, ranges)]
    match = identity.match_product(product_name, names) if product_name and names else None
    warnings = list(result.warnings)
    if product_name and names and match is None:
        warnings.append("PRODUCT_NOT_IN_REPORT")
    if result.check_pages:
        warnings.append("OCR_VLM_MISMATCH:" + ",".join(str(n) for n in result.check_pages))
    if any(code["status"] in ("ambiguous", "unconfirmed", "single_page") for code in who.codes) or (not who.codes):
        warnings.append("CODE_NEEDS_CONFIRMATION")
    return {
        "readerVersion": result.reader_version,
        "pageCount": result.page_count,
        "pages": [{"n": p.number, "kind": p.kind, "method": p.method, "chars": p.chars, "text": p.text,
                   "confidence": p.confidence, "warnings": p.warnings, "check": p.check} for p in result.pages],
        "unreadPages": result.unread_pages,
        "checkPages": result.check_pages,
        "issuer": {"key": who.issuer_key, "name": who.issuer_name} if who.issuer_key else None,
        "codes": who.codes,
        "version": who.version,
        "issuedOn": who.issued_on,
        "products": products,
        # Pages of this product where OCR and the vision model disagree: the figures there need a person's look.
        "productMatch": ({**products[match[0]], "score": round(match[1], 3),
                          "checkPages": [n for n in products[match[0]]["pages"] if n in result.check_pages]} if match else None),
        "warnings": warnings,
    }


def parse_evidence(raw: bytes, product_name: str | None = None, *, prefer: str = "auto",
                   ocr_budget_seconds: float | None = 240, max_ocr_pages: int | None = 40, ocr=None, crosscheck=None) -> dict:
    """Raises ValueError with a specific code (FILE_TOO_LARGE, EVIDENCE_PAGE_LIMIT, PDF_OR_IMAGE_REQUIRED)."""
    options = {"prefer": prefer, "ocr_budget_seconds": ocr_budget_seconds, "max_ocr_pages": max_ocr_pages}
    if ocr is not None:
        options["ocr"] = ocr
    if crosscheck is not None:
        options["crosscheck"] = crosscheck
    return describe(read_document(raw, **options), product_name)


__all__ = ["READER_VERSION", "describe", "parse_evidence"]
