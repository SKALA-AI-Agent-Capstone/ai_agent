"""Read reviewed page text from data/curated/<id>.md instead of parsing the PDF again.

Each page is a `## N쪽` section. An image page carries one review note right under its heading,
which sets the page's trust level:
  human_verified  `> 원본 이미지 대조 완료 ...` (a person compared it with the image) is indexed as is.
  auto_verified   `> 자동 원본 대조 완료 · 사람 검수 전` (pdf_curator's model re-check) gets AUTO_VERIFIED.
  draft           any other note, e.g. `> 이미지 판독 초안 · 원본 대조 필요`, gets AUTO_TRANSCRIPTION.
  excluded        `> ... 불확실 ...` is left out (never index an uncertain read).
  skipped         `> ... 판독 생략` was outside the indexed page range and has no text.
  text            no note: the PDF text layer, or text a person edited. Sections that are not pages (S01 별표 행별 연결) are ignored;
the annex rows still come from pdf_tables.
"""
import hashlib
from pathlib import Path
import re

from .pdf_vision import AUTO_TRANSCRIPTION, AUTO_VERIFIED

PAGE_HEADING = re.compile(r"^## (\d+)쪽\s*$")
OTHER_HEADING = re.compile(r"^## ")
MD_HEADING = re.compile(r"^#{1,6}\s+")
SHA_LINE = re.compile(r"^- SHA-256: `([0-9a-f]{64})`")
VERIFIED = "원본 이미지 대조 완료"
AUTO_CHECKED = "자동 원본 대조 완료"
UNCERTAIN = "불확실"
SKIPPED = "판독 생략"
MARKERS = {"auto_verified": AUTO_VERIFIED, "draft": AUTO_TRANSCRIPTION}


class CuratedMismatch(ValueError):
    """The markdown was made from a different PDF or does not cover every page."""


def curated_path(pdf_path: Path) -> Path:
    return pdf_path.parent.parent / "curated" / f"{pdf_path.stem}.md"


def read_curated_pages(md_path: Path, pdf_path: Path, page_count: int, *, on_warning=None,
                       trust: dict[int, str] | None = None) -> list[str]:
    return parse_curated_text(md_path.read_text(encoding="utf-8"), pdf_path, page_count,
                              on_warning=on_warning, trust=trust)


def trust_level(note: str | None) -> str:
    if note is None:
        return "text"
    if UNCERTAIN in note:
        return "excluded"
    if SKIPPED in note:
        return "skipped"
    if VERIFIED in note:
        return "human_verified"
    return "auto_verified" if AUTO_CHECKED in note else "draft"


def parse_curated_text(text: str, pdf_path: Path, page_count: int, *, on_warning=None,
                       trust: dict[int, str] | None = None) -> list[str]:
    """Use the same page and SHA checks for a saved MD or an in-memory auto review.

    `trust`, when given, receives each page's trust level for the operator report.
    """
    sha = next((m.group(1) for m in map(SHA_LINE.match, text.splitlines()) if m), None)
    if sha != hashlib.sha256(pdf_path.read_bytes()).hexdigest():
        raise CuratedMismatch("CURATED_SHA_MISMATCH")
    sections: dict[int, list[str]] = {}
    current = None
    for line in text.splitlines():
        heading = PAGE_HEADING.match(line)
        if heading:
            current = sections.setdefault(int(heading.group(1)), [])
        elif OTHER_HEADING.match(line):
            current = None
        elif current is not None:
            current.append(line)
    if set(sections) != set(range(1, page_count + 1)):
        raise CuratedMismatch("CURATED_PAGE_MISMATCH")
    pages = []
    for number in range(1, page_count + 1):
        lines = sections[number]
        notes = [line[1:].strip() for line in lines if line.startswith(">")]
        body = "\n".join(MD_HEADING.sub("", line) for line in lines if not line.startswith(">")).strip()
        level = trust_level(notes[0] if notes else None)
        if level == "excluded":
            if on_warning is not None:
                on_warning({"page": number, "reason": "CURATED_PAGE_UNCERTAIN"})
            body = ""
        elif level == "skipped":
            body = ""
        elif level in MARKERS and body:
            title, _, rest = body.partition("\n")
            body = f"{title}\n{MARKERS[level]}\n{rest}".rstrip()
        if trust is not None:
            trust[number] = level
        pages.append(body)
    return pages
