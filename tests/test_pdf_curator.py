"""Automatic MD must keep page identity and exclude uncertain image text from RAG."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from capstone_rag_mcp.curated import read_curated_pages
from capstone_rag_mcp.pdf_curator import build_markdown, write_markdown


class Page:
    width = height = 100

    def __init__(self, text, images=None, tables=None, chars=None, cells=None):
        self.text = text
        self.images = images or []
        self.tables = tables or []
        self.chars = chars or []
        self.cells = cells or []

    def extract_text(self):
        return self.text

    def find_tables(self):
        rows = [SimpleNamespace(cells=self.cells)]
        return [SimpleNamespace(extract=lambda table=table: table, rows=rows) for table in self.tables]


def test_generated_markdown_preserves_rows_and_excludes_uncertain_image(monkeypatch, tmp_path):
    pdf = tmp_path / "X01.pdf"
    pdf.write_bytes(b"%PDF-1.7 synthetic")
    pages = [Page("광고 표현 기준", tables=[["표현 기준"],
                                                ["광고 표현", "실증 자료 기준"]]),
             Page("위반 사례", images=[{"x0": 0, "x1": 90, "top": 0, "bottom": 90}])]
    pages[0].tables = [[['광고 표현', '실증 자료 기준'], ['피부결 개선', '인체적용시험 자료']]]

    class Document:
        def __enter__(self):
            return SimpleNamespace(pages=pages)
        def __exit__(self, *args):
            pass

    monkeypatch.setitem(sys.modules, "pdfplumber", SimpleNamespace(open=lambda path: Document()))

    def image_reader(path, index, native, client, model):
        return "위반 사례\n[판독 불가] 22.19%", True, ["숫자 앞 문구 판독 불가"], ["UNREADABLE_MARK"]

    markdown, report = build_markdown(pdf, client=object(), image_reader=image_reader)
    assert report["sha256"] == hashlib.sha256(pdf.read_bytes()).hexdigest()
    assert report["issues"][0]["uncertain"]
    assert [r["trust"] for r in report["page_reports"]] == ["native_text", "excluded"]
    assert report["page_reports"][1]["reasons"] == ["UNREADABLE_MARK"]
    assert "- 광고 표현: 피부결 개선\n- 실증 자료 기준: 인체적용시험 자료" in markdown
    md = tmp_path / "X01.md"
    md.write_text(markdown)
    warnings = []
    indexed = read_curated_pages(md, pdf, 2, on_warning=warnings.append)
    assert "피부결 개선" in indexed[0]
    assert indexed[1] == ""
    assert warnings == [{"page": 2, "reason": "CURATED_PAGE_UNCERTAIN"}]


def test_existing_reviewed_markdown_is_not_overwritten(tmp_path):
    pdf = tmp_path / "X01.pdf"
    pdf.write_bytes(b"%PDF-1.7")
    md = tmp_path / "X01.md"
    md.write_text("reviewed")
    with pytest.raises(FileExistsError):
        write_markdown(pdf, md)
    assert md.read_text() == "reviewed"


def test_batch_can_use_auto_verified_pages_without_creating_md(monkeypatch, tmp_path):
    from capstone_rag_mcp import batch
    from capstone_rag_mcp import pdf_curator

    pdf = tmp_path / "sources" / "X01.pdf"
    pdf.parent.mkdir()
    pdf.write_bytes(b"%PDF-1.7 synthetic")
    native = Page("규정 본문")

    class Document:
        def __enter__(self):
            return SimpleNamespace(pages=[native])
        def __exit__(self, *args):
            pass

    monkeypatch.setitem(sys.modules, "pdfplumber", SimpleNamespace(open=lambda path: Document()))
    monkeypatch.setenv("RAG_PDF_AUTO_VERIFY", "true")
    md = (f"# X01\n\n- SHA-256: `{hashlib.sha256(pdf.read_bytes()).hexdigest()}`\n\n"
          "## 1쪽\n\n검증된 규정 본문\n")
    monkeypatch.setenv("RAG_PDF_QA_DIR", str(tmp_path / "qa"))
    report = {"sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(), "issues": []}
    monkeypatch.setattr(pdf_curator, "build_markdown", lambda path, page_numbers=None: (md, report))
    warnings = []
    pages, derived = batch.read_source_pages(pdf, None, warnings.append)
    assert pages == ["검증된 규정 본문"]
    assert derived
    assert warnings == [{"reason": "PDF_PAGE_TRUST", "mode": "auto_verify", "pages": {"text": [1]},
                         "report": str(tmp_path / "qa" / f"X01.{report['sha256'][:12]}.auto.json")}]
    assert not (tmp_path / "curated" / "X01.md").exists()


def test_comparison_table_without_separate_columns_is_excluded(monkeypatch, tmp_path):
    from capstone_rag_mcp import pdf_curator

    calls = []
    monkeypatch.setattr(pdf_curator, "_vision", lambda *args: SimpleNamespace(text="개정 전 개정 후\n추출물 2%", uncertain=False))
    monkeypatch.setattr(pdf_curator, "_cross_check", lambda *args: SimpleNamespace(
        corrected_text="개정 전 개정 후\n추출물 2%", uncertain=False, issues=[]))
    result = pdf_curator._read_image_page(tmp_path / "X01.pdf", 0, "개정 전 개정 후", object(), "model",
                                          renderer=lambda *_: "full",
                                          crop_renderer=lambda *_: calls.append("crop") or ["left", "right"])
    assert result[1] is True
    assert calls == ["crop"]


def _read_with(monkeypatch, tmp_path, reads, checks, native=""):
    """Run _read_image_page with scripted model answers; returns the result and the calls made."""
    from capstone_rag_mcp import pdf_curator

    reads, checks, calls = list(reads), list(checks), []
    monkeypatch.setattr(pdf_curator, "_vision", lambda client, model, images, hint:
                        calls.append(("read", images)) or SimpleNamespace(text=reads.pop(0), uncertain=False))
    monkeypatch.setattr(pdf_curator, "_cross_check", lambda client, model, images, draft:
                        calls.append(("check", images)) or checks.pop(0))
    result = pdf_curator._read_image_page(tmp_path / "X01.pdf", 0, native, object(), "model",
                                          renderer=lambda *_: "full", crop_renderer=lambda *_: ["left", "right"])
    return result, calls


def checked(text, issues=(), uncertain=False):
    return SimpleNamespace(corrected_text=text, uncertain=uncertain, issues=list(issues))


@pytest.mark.parametrize("final, independent, native, reason", [
    # The checker "corrected" a number the independent read saw differently.
    ("주름 개선 22.18%", "주름 개선 22.19%", "", "NUMBER_MISMATCH_BETWEEN_READS"),
    ("함량 10g", "함량 10mg", "", "NUMBER_MISMATCH_BETWEEN_READS"),
    # Both reads agree but miss a value the PDF text layer has.
    ("미백 효과", "미백 효과", "미백 효과 2.5%", "NATIVE_NUMBER_MISSING"),
    ("피부 개선", "피부 개선*", "", "NOTE_OR_EXCEPTION_MISMATCH"),
    ("천연 표현 금지", "천연 표현 금지\n단, 인증 제품은 제외", "", "NOTE_OR_EXCEPTION_MISMATCH"),
    ("천연 표현 금지", "천연 표현 금지", "※ 협회 안내서 참고", "NOTE_OR_EXCEPTION_MISMATCH"),
    # Comparison table found only in the model's own text, columns mixed on one line.
    ("개정 전 개정 후\n용매 제외 용매 포함", "개정 전 개정 후\n용매 제외 용매 포함", "", "COMPARISON_COLUMNS_NOT_SEPARATED"),
    ("### 개정 전\n### 개정 후\n용매 포함", "### 개정 전\n### 개정 후\n용매 포함", "", "COMPARISON_COLUMN_EMPTY"),
    ("개정 전\n용매 포함\n개정 후\n용매 포함", "개정 전\n용매 포함\n개정 후\n용매 포함", "개정 전 개정 후",
     "COMPARISON_COLUMNS_IDENTICAL"),
    ("변경 전 변경 후\n10% 20%", "변경 전 변경 후\n10% 20%", "", "COMPARISON_COLUMNS_NOT_SEPARATED"),
])
def test_deterministic_checks_override_model_confidence(final, independent, native, reason):
    from capstone_rag_mcp.pdf_curator import check_reasons

    assert reason in check_reasons(final, independent, native)


def test_checks_pass_for_matching_reads_and_ignore_markdown_formatting():
    from capstone_rag_mcp.pdf_curator import check_reasons

    final = ("## 개정 전\n- **추출물** 함량 2% (용매 제외)\n\n## 개정 후\n* 추출물 2%, 용매 포함 표시\n"
             "※ 2025. 8. 14. 시행")
    independent = "개정 전: \n추출물 함량 2% 용매 제외\n개정 후:\n추출물 2% 용매 포함 표시\n※ 2025.8.14. 시행"
    assert check_reasons(final, independent, "개정 전 개정 후 - 4 -") == []
    assert check_reasons("x" * 10, "x" * 10, "", draft_length=18001) == ["DRAFT_TOO_LONG_FOR_CHECK"]


def test_number_disagreement_retries_crops_then_excludes_page(monkeypatch, tmp_path):
    (text, uncertain, issues, reasons), calls = _read_with(
        monkeypatch, tmp_path, ["주름 개선 22.19%", "주름 개선 22.19%"],
        [checked("주름 개선 22.18%", ["22.19 → 22.18"]), checked("주름 개선 22.18%", ["22.19 → 22.18"])])
    assert uncertain and reasons == ["NUMBER_MISMATCH_BETWEEN_READS"]
    assert [images for _, images in calls] == [["full"], ["full"], ["left", "right"], ["left", "right"]]
    assert issues == ["22.19 → 22.18", "22.19 → 22.18"]


def test_correction_must_survive_a_second_check(monkeypatch, tmp_path):
    # Only wording changed, so the reads agree on numbers, but the confirmation check reads 3% again.
    (_, uncertain, _, reasons), _ = _read_with(
        monkeypatch, tmp_path, ["보습 지속 3%", "보습 지속 3%"],
        [checked("보습 유지 3%", ["지속 → 유지"]), checked("보습 유지 8%"),
         checked("보습 유지 3%", ["지속 → 유지"]), checked("보습 유지 8%")])
    assert uncertain and reasons == ["CORRECTION_NOT_CONFIRMED"]


def test_confirmed_correction_is_auto_verified_without_retry(monkeypatch, tmp_path):
    (text, uncertain, issues, reasons), calls = _read_with(
        monkeypatch, tmp_path, ["보습 지속 3%"], [checked("보습 유지 3%", ["지속 → 유지"]), checked("보습 유지 3%")])
    assert (text, uncertain, issues, reasons) == ("보습 유지 3%", False, ["지속 → 유지"], [])
    assert len(calls) == 3


def test_model_headings_and_quotes_stay_inside_their_page(monkeypatch, tmp_path):
    """'## 개정 전' from a model used to end the page section, silently dropping the rest."""
    pdf = tmp_path / "X01.pdf"
    pdf.write_bytes(b"%PDF-1.7 synthetic")
    pages = [Page("개정", images=[{"x0": 0, "x1": 90, "top": 0, "bottom": 90}]), Page("다음 쪽 본문")]

    class Document:
        def __enter__(self):
            return SimpleNamespace(pages=pages)
        def __exit__(self, *args):
            pass

    monkeypatch.setitem(sys.modules, "pdfplumber", SimpleNamespace(open=lambda path: Document()))
    body = "# 개정사항\n## 개정 전\n용매 제외\n## 개정 후\n> 용매 포함 2%"
    markdown, _ = build_markdown(pdf, client=object(),
                                 image_reader=lambda *args: (body, False, [], []))
    from capstone_rag_mcp.curated import parse_curated_text
    from capstone_rag_mcp.pdf_vision import AUTO_VERIFIED

    trust = {}
    indexed = parse_curated_text(markdown, pdf, 2, trust=trust)
    assert indexed[0] == f"개정사항\n{AUTO_VERIFIED}\n개정 전\n용매 제외\n개정 후\n용매 포함 2%"
    assert indexed[1] == "다음 쪽 본문"
    assert trust == {1: "auto_verified", 2: "text"}


def test_image_pages_outside_the_indexed_range_are_not_sent_to_the_model(monkeypatch, tmp_path):
    pdf = tmp_path / "X01.pdf"
    pdf.write_bytes(b"%PDF-1.7 synthetic")
    image = [{"x0": 0, "x1": 90, "top": 0, "bottom": 90}]
    pages = [Page("표지", images=image), Page("사례", images=image), Page("본문 " * 200, images=image)]

    class Document:
        def __enter__(self):
            return SimpleNamespace(pages=pages)
        def __exit__(self, *args):
            pass

    monkeypatch.setitem(sys.modules, "pdfplumber", SimpleNamespace(open=lambda path: Document()))
    read = []
    markdown, report = build_markdown(pdf, client=object(), page_numbers={2, 3},
                                      image_reader=lambda p, index, *a: read.append(index) or ("사례 1%", False, [], []))
    assert read == [1]
    assert report["page_reports"] == [
        {"page": 1, "trust": "skipped"},
        {"page": 2, "trust": "auto_verified", "reasons": [], "corrections": []},
        {"page": 3, "trust": "native_text", "reasons": ["IMAGE_NOT_READ"]}]
    from capstone_rag_mcp.curated import parse_curated_text

    trust = {}
    assert parse_curated_text(markdown, pdf, 3, trust=trust)[0] == ""
    assert trust == {1: "skipped", 2: "auto_verified", 3: "text"}


def test_operator_report_lists_excluded_pages_without_touching_curated(monkeypatch, tmp_path):
    from capstone_rag_mcp import batch, pdf_curator

    pdf = tmp_path / "sources" / "X02.pdf"
    pdf.parent.mkdir()
    pdf.write_bytes(b"%PDF-1.7 synthetic")
    image = [{"x0": 0, "x1": 90, "top": 0, "bottom": 90}]
    # 2 image pages out of 7 stays below the image-heavy share, so the auto check still runs.
    pages = [Page("사례 A", images=image), Page("사례 B", images=image)] + [Page("규정 본문") for _ in range(5)]

    class Document:
        def __enter__(self):
            return SimpleNamespace(pages=pages)
        def __exit__(self, *args):
            pass

    monkeypatch.setitem(sys.modules, "pdfplumber", SimpleNamespace(open=lambda path: Document()))
    monkeypatch.setenv("RAG_PDF_AUTO_VERIFY", "true")
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("RAG_PDF_QA_DIR", raising=False)
    answers = {0: ("사례 A 효과 5%", False, ["4% → 5%"], []),
               1: ("사례 B 10%", True, [], ["NUMBER_MISMATCH_BETWEEN_READS"])}
    monkeypatch.setattr(pdf_curator, "_read_image_page", lambda p, index, *a, **k: answers[index])
    monkeypatch.setattr(pdf_curator, "_model_client", lambda: object())
    original = pdf_curator.build_markdown
    monkeypatch.setattr(pdf_curator, "build_markdown", lambda path, **kw: original(
        path, image_reader=pdf_curator._read_image_page, **kw))
    warnings = []
    indexed, _ = batch.read_source_pages(pdf, None, warnings.append)
    assert indexed[1] == ""
    assert {"page": 2, "reason": "CURATED_PAGE_UNCERTAIN"} in warnings
    assert {"page": 1, "reason": "AUTO_PDF_CHECK_NOTES", "count": 1} in warnings
    assert {"page": 2, "reason": "AUTO_PDF_CHECK_FAILED", "checks": ["NUMBER_MISMATCH_BETWEEN_READS"]} in warnings
    summary = warnings[-1]
    assert summary["pages"] == {"auto_verified": [1], "excluded": [2], "text": [3, 4, 5, 6, 7]}
    saved = json.loads(Path(summary["report"]).read_text())
    assert saved["page_reports"][0]["corrections"] == ["4% → 5%"]
    assert Path(summary["report"]).parent == tmp_path / "pdf-qa"
    assert not (tmp_path / "curated").exists()


def test_table_whose_cell_cuts_a_character_is_left_out_of_row_text(monkeypatch, tmp_path):
    """S04 9쪽: '질염 등' straddles the row label cell, so the drug cell read '염 등 질병의'."""
    pdf = tmp_path / "X01.pdf"
    pdf.write_bytes(b"%PDF-1.7 synthetic")
    text = "사용목적 질염 등 질병의 치료"
    table = [["구분", "의약품"], [None, "염 등 질병의 치료"]]
    char = lambda ch, x0: {"text": ch, "x0": x0, "x1": x0 + 4, "top": 50, "bottom": 54}
    clipped = Page(text, tables=[table], chars=[char("질", 38), char("염", 42)], cells=[(10, 40, 40, 60), (40, 40, 90, 60)])
    whole = Page(text, tables=[table], chars=[char("질", 30), char("염", 42)], cells=[(10, 40, 40, 60), (40, 40, 90, 60)])

    class Document:
        def __init__(self, page):
            self.page = page
        def __enter__(self):
            return SimpleNamespace(pages=[self.page])
        def __exit__(self, *args):
            pass

    monkeypatch.setitem(sys.modules, "pdfplumber", SimpleNamespace(open=lambda path: Document(clipped)))
    markdown, report = build_markdown(pdf, client=object())
    assert "염 등 질병의" not in markdown.replace(text, "")
    assert text in markdown
    assert report["page_reports"] == [{"page": 1, "trust": "native_text",
                                       "reasons": ["TABLE_ROWS_DROPPED_CLIPPED_CELL"]}]
    monkeypatch.setitem(sys.modules, "pdfplumber", SimpleNamespace(open=lambda path: Document(whole)))
    markdown, report = build_markdown(pdf, client=object())
    assert "- 의약품: 염 등 질병의 치료" in markdown
    assert report["page_reports"] == [{"page": 1, "trust": "native_text"}]


IMAGE = [{"x0": 0, "x1": 90, "top": 0, "bottom": 90}]


def _image_heavy_pdf(monkeypatch, tmp_path, source_id="X04"):
    """3 image pages out of 4: S04-like slides."""
    pdf = tmp_path / "sources" / f"{source_id}.pdf"
    pdf.parent.mkdir(exist_ok=True)
    pdf.write_bytes(b"%PDF-1.7 image slides")
    pages = [Page("표지 본문 " * 80)] + [Page(f"사례 {n}", images=IMAGE) for n in (2, 3, 4)]

    class Document:
        def __enter__(self):
            return SimpleNamespace(pages=pages)
        def __exit__(self, *args):
            pass

    monkeypatch.setitem(sys.modules, "pdfplumber", SimpleNamespace(open=lambda path: Document()))
    return pdf


def _no_model(*args, **kwargs):
    raise AssertionError("an image-heavy PDF without MD must not reach a vision model")


@pytest.mark.parametrize("auto_verify", ["true", "false"])
def test_image_heavy_pdf_without_md_is_left_out_without_model_calls(monkeypatch, tmp_path, auto_verify):
    from capstone_rag_mcp import batch, pdf_curator

    pdf = _image_heavy_pdf(monkeypatch, tmp_path)
    monkeypatch.setenv("RAG_PDF_AUTO_VERIFY", auto_verify)
    monkeypatch.setattr(pdf_curator, "build_markdown", _no_model)
    monkeypatch.setattr(batch, "read_pages_for_index", _no_model)
    with pytest.raises(batch.CuratedMdRequired) as raised:
        batch.read_source_pages(pdf, None, [].append)
    assert (raised.value.image_pages, raised.value.total) == ([2, 3, 4], 4)


def test_image_heavy_pdf_uses_matching_md_and_rejects_stale_md(monkeypatch, tmp_path):
    from capstone_rag_mcp import batch

    pdf = _image_heavy_pdf(monkeypatch, tmp_path)
    monkeypatch.setattr(batch, "read_pages_for_index", _no_model)
    md = tmp_path / "curated" / "X04.md"
    md.parent.mkdir()
    body = "".join(f"## {n}쪽\n\n> 원본 이미지 대조 완료\n\n사례 {n} 22.19%\n\n" for n in (2, 3, 4))
    md.write_text(f"- SHA-256: `{hashlib.sha256(pdf.read_bytes()).hexdigest()}`\n\n## 1쪽\n\n표지\n\n{body}",
                  encoding="utf-8")
    warnings = []
    pages, curated = batch.read_source_pages(pdf, None, warnings.append)
    assert curated and pages[3] == "사례 4 22.19%"
    assert warnings[-1]["pages"] == {"text": [1], "human_verified": [2, 3, 4]}

    pdf.write_bytes(b"%PDF-1.7 newer edition")  # MD made from another edition
    warnings = []
    with pytest.raises(batch.CuratedMdRequired):
        batch.read_source_pages(pdf, None, warnings.append)
    assert warnings == [{"file": "X04.md", "reason": "CURATED_SHA_MISMATCH", "fallback": "pdf"}]


def test_image_share_counts_only_the_indexed_page_range(monkeypatch, tmp_path):
    from capstone_rag_mcp import batch, pdf_curator

    pdf = _image_heavy_pdf(monkeypatch, tmp_path)
    monkeypatch.setattr(batch, "read_pages_for_index", lambda path, **kw: ["표지 본문", "", "", ""])
    assert pdf_curator.image_page_numbers(pdf, {1}) == ([], 1)
    pages, curated = batch.read_source_pages(pdf, {1}, [].append)
    assert pages[0] == "표지 본문" and not curated


async def test_batch_skips_image_heavy_pdf_and_says_md_is_needed(monkeypatch, tmp_path):
    from capstone_rag_mcp import batch

    pdf = _image_heavy_pdf(monkeypatch, tmp_path)
    monkeypatch.setattr(batch, "read_pages_for_index", _no_model)
    monkeypatch.setattr(batch, "find_pdf", lambda source_id: pdf)
    monkeypatch.setattr(batch, "sources", lambda: [{
        "id": "X04", "title": "이미지 사례집", "official_page": "https://example.invalid",
        "acquisition": {"method": "pdf", "collection": "cases", "source_type": "guideline"}}])
    documents, skipped = await batch.collect_pdf()
    assert documents == []
    assert skipped == [{"id": "X04", "reason": "PDF_NEEDS_CURATED_MD", "image_pages": [2, 3, 4], "pages": 4,
                        "expected": "data/curated/X04.md"}]
