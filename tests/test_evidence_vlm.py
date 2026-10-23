"""Scanned evidence pages get a second reader: where OCR and a vision model disagree on the figures, the page is marked."""
import asyncio
from types import SimpleNamespace

from capstone_agent import evidence_vlm
from capstone_shared import reading
from capstone_shared.evidence_parse import describe
from capstone_shared.reading import PageText, ReadResult, read_pdf
from test_reading import FakeOcr, REPORT, combine, picture_pdf, text_pdf


def test_equal_figures_agree_even_when_words_differ_and_a_different_figure_is_flagged():
    assert evidence_vlm.compare("변화율 9.71% 증가", "변 화 율 9.71 % 증가") == {"status": "match", "figures": 1}
    verdict = evidence_vlm.compare("변화율 9.71% 증가", "변화율 9.17% 증가")
    assert verdict == {"status": "needs_review", "visionOnly": ["9.71"], "ocrOnly": ["9.17"]}
    # a figure only one reader saw is also a disagreement
    assert evidence_vlm.compare("값 10.5", "값")["visionOnly"] == ["10.5"]


def test_dates_are_not_figures():
    # the cover prints a study period; one reader spacing it differently is not a disagreement
    assert evidence_vlm.compare("연구 기간 2024. 03. 19 ~ 04. 03 변화율 9.71%", "연구 기간 2024.03.19 ~ 04.03 변화율 9.71%")["status"] == "match"
    assert evidence_vlm.compare("연구 기간 2024. 03. 19 ~ 04. 03", "연구 기간 2024. 03. 19 ~")["status"] == "match"
    assert evidence_vlm.compare("값 4.03%", "값 4.30%")["status"] == "needs_review"   # a real figure still counts


def test_only_ocr_pages_are_cross_checked_and_a_mismatch_is_listed():
    calls = []

    def check(image, text):
        calls.append(text)
        return {"status": "needs_review", "visionOnly": ["9.71"], "ocrOnly": []}

    result = read_pdf(combine(text_pdf(REPORT), picture_pdf()), ocr=FakeOcr("변화율 9.17%"), crosscheck=check)
    assert calls == ["변화율 9.17%"]                       # the text-layer page is exact and is not checked
    assert result.pages[0].check is None and result.pages[1].check["status"] == "needs_review"
    assert "OCR_VLM_MISMATCH" in result.pages[1].warnings and result.check_pages == [2]


def test_a_failing_second_reader_never_breaks_the_read_and_is_recorded():
    def broken(image, text):
        raise RuntimeError("vision model down")

    result = read_pdf(picture_pdf(pages=2), ocr=FakeOcr(), crosscheck=broken)
    assert [p.check["status"] for p in result.pages] == ["unavailable", "unavailable"]
    assert result.check_pages == [] and all("VLM_CHECK_FAILED" in p.warnings for p in result.pages)
    assert all(p.text for p in result.pages)               # the OCR text is still there


def test_pages_past_the_check_budget_are_marked_skipped_not_silently_unchecked(monkeypatch):
    monkeypatch.setattr(reading, "CHECK_MAX_PAGES", 1)
    result = read_pdf(picture_pdf(pages=3), ocr=FakeOcr(), crosscheck=lambda image, text: {"status": "match", "figures": 0})
    assert [p.check["status"] for p in result.pages] == ["match", "skipped", "skipped"]
    assert "VLM_CHECK_SKIPPED" in result.pages[2].warnings


def test_no_crosscheck_leaves_pages_untouched():
    result = read_pdf(picture_pdf(), ocr=FakeOcr())
    assert result.pages[0].check is None and result.check_pages == []


def test_description_lists_check_pages_and_the_registered_products_own_pages():
    pages = ["#1: 알파 크림\n#2: 베타 크림\n알파 크림 시험", "베타 크림 변화율 9.71%", "알파 크림 표 9.17%"]
    result = ReadResult([PageText(n, "scan", "paddle", t) for n, t in enumerate(pages, 1)], 3, check_pages=[2, 3])
    done = describe(result, "알파 크림")
    assert done["checkPages"] == [2, 3]
    assert "OCR_VLM_MISMATCH:2,3" in done["warnings"]
    assert done["productMatch"]["checkPages"] == [3]       # only the registered product's pages matter to the person


async def test_crosscheck_calls_the_model_on_the_event_loop_from_the_reader_thread():
    seen = {}

    class Provider:
        client = object()

        async def parse(self, prompt, schema, *, image=None, text=""):
            seen["loop"] = asyncio.get_running_loop()
            seen["image"] = image
            return schema(lines=["변화율 9.71% 증가"])

    check = evidence_vlm.make_crosscheck(Provider(), asyncio.get_running_loop())
    from PIL import Image
    verdict = await asyncio.to_thread(check, Image.new("RGB", (40, 40), "white"), "변화율 9.71% 증가")
    assert verdict["status"] == "match" and seen["loop"] is asyncio.get_running_loop()
    assert seen["image"].startswith("data:image/jpeg;base64,")


def test_without_a_model_there_is_no_crosscheck():
    assert evidence_vlm.make_crosscheck(SimpleNamespace(client=None), None) is None
    assert evidence_vlm.make_crosscheck(None, None) is None
