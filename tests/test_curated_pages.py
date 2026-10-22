"""Reviewed markdown replaces PDF parsing only when it matches the PDF page for page."""
import hashlib

import pytest

from capstone_rag_mcp.curated import CuratedMismatch, curated_path, read_curated_pages
from capstone_rag_mcp.pdf_vision import AUTO_TRANSCRIPTION


def write_pair(tmp_path, pages_md: str, pdf_bytes: bytes = b"%PDF-1.7 test"):
    (tmp_path / "sources").mkdir()
    (tmp_path / "curated").mkdir()
    pdf = tmp_path / "sources" / "S04.pdf"
    pdf.write_bytes(pdf_bytes)
    md = curated_path(pdf)
    md.write_text(f"# S04 PDF 변환본\n\n- SHA-256: `{hashlib.sha256(pdf_bytes).hexdigest()}`\n\n{pages_md}",
                  encoding="utf-8")
    return md, pdf


PAGES = """## 1쪽

본문 텍스트

## 2쪽

> 원본 이미지 대조 완료 · 위반 대상 문구 중심.

# 부당광고 적발 사례: 니들
- ‘피부 깊숙히 침투’

## 3쪽

> 이미지 판독 초안 · 원본 대조 필요

부당광고 적발 사례
질염에 도움

## 4쪽

> 이미지 판독 초안 · 불확실한 글자 있음

부당광고 적발 사례
?? 판독 불가

## 별표 1·2: 행별 연결

### [별표 1] > 기타 (원본 1쪽)

금지 표현: · 메디슨
"""


def test_review_note_decides_how_each_page_is_indexed(tmp_path):
    md, pdf = write_pair(tmp_path, PAGES)
    warnings, trust = [], {}
    pages = read_curated_pages(md, pdf, 4, on_warning=warnings.append, trust=trust)
    assert trust == {1: "text", 2: "human_verified", 3: "draft", 4: "excluded"}
    assert pages[0] == "본문 텍스트"
    assert pages[1] == "부당광고 적발 사례: 니들\n- ‘피부 깊숙히 침투’"
    assert pages[2] == f"부당광고 적발 사례\n{AUTO_TRANSCRIPTION}\n질염에 도움"
    assert pages[3] == ""
    assert warnings == [{"page": 4, "reason": "CURATED_PAGE_UNCERTAIN"}]
    assert "메디슨" not in "".join(pages)  # annex rows come from pdf_tables, not this section


def test_markdown_for_another_pdf_or_missing_page_is_rejected(tmp_path):
    md, pdf = write_pair(tmp_path, PAGES)
    pdf.write_bytes(b"%PDF-1.7 newer edition")
    with pytest.raises(CuratedMismatch, match="CURATED_SHA_MISMATCH"):
        read_curated_pages(md, pdf, 4)
    pdf.write_bytes(b"%PDF-1.7 test")
    with pytest.raises(CuratedMismatch, match="CURATED_PAGE_MISMATCH"):
        read_curated_pages(md, pdf, 5)
