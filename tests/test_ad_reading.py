import pytest
from PIL import Image

from capstone_agent import ad_reading
from capstone_agent.ad_reading import AdCopy, assign_lines, read_tile, strip_product
from capstone_shared.documents import tile_image


def tile():
    return tile_image(Image.new("RGB", (400, 600), "white"), "a", "ad.png")[0]


def tesseract_lines(t, scale=1.0):
    if scale != 1.0:
        # The half-size pass finds the headline the full-size pass missed, and repeats one line it already has.
        width, height = t.source_size
        return [{"id": f"{t.id}:ocr:9:x0.5", "text": "큰 제목", "ocrConfidence": 80.0,
                 "location": t.location(precise_box=[20/width, 5/height, 200/width, 30/height], method="TESSERACT")},
                {"id": f"{t.id}:ocr:1:x0.5", "text": "2 주 안에", "ocrConfidence": 80.0,
                 "location": t.location(precise_box=[22/width, 42/height, 290/width, 78/height], method="TESSERACT")}]
    width, height = t.source_size
    def line(i, text, box):
        return {"id": f"{t.id}:ocr:{i}", "text": text, "ocrConfidence": 90.0,
                "location": t.location(precise_box=[box[0]/width, box[1]/height, box[2]/width, box[3]/height], method="TESSERACT")}
    return [line(1, "2 주 안에 개 선", (20, 40, 300, 80)), line(2, "Pyunkang Yul", (30, 400, 200, 430))]


async def test_reads_ad_copy_in_groups_and_leaves_product_label_text_out(monkeypatch):
    t = tile()
    monkeypatch.setattr(ad_reading, "local_ocr", tesseract_lines)
    seen = {}

    class FakeProvider:
        client = True

        async def parse(self, prompt, schema, **kwargs):
            seen["images"] = kwargs["image"]
            seen["text"] = kwargs["text"]
            assert schema is AdCopy
            return AdCopy(layout=[], on_product_text=["Pyunkang Yul"], groups=[
                {"kind": "copy", "role": "claim", "segments": ["2주 안에 개선"], "labels": [], "box_numbers": [2], "box": [40, 60, 800, 150]},
                {"kind": "copy", "role": "claim", "segments": ["Pyunkang Yul"], "labels": [], "box_numbers": [], "box": []},  # model repeated the label: code removes it
                {"kind": "footnote", "role": "condition", "segments": ["*연구 기간 : 2025.02.24~03.14", "20명 대상"], "labels": [], "box_numbers": [3], "box": [50, 800, 900, 900]},
                {"kind": "table", "role": "result", "segments": ["12.08%", "눈가 주름 감소"], "labels": ["사용 전", "사용 후"], "box_numbers": [9], "box": [0, 0, 1, 1]}])  # too small to be a box

    blocks, warnings = await read_tile(FakeProvider(), t)
    assert [(b["role"], b["text"]) for b in blocks] == [
        ("claim", "2주 안에 개선"), ("condition", "*연구 기간 : 2025.02.24~03.14 20명 대상"), ("result", "12.08% 눈가 주름 감소")]
    assert blocks[2]["labels"] == ["사용 전", "사용 후"]
    assert len(seen["images"]) == 2 and seen["images"][0].startswith("data:image/jpeg")  # original and the box drawing
    # The half-size headline is added above the first line; its overlapping repeat is not.
    assert "1: x 20~200, y 5~30" in seen["text"] and "2: x 20~300, y 40~80" in seen["text"] and "4:" not in seen["text"]
    # Position comes from the matching Tesseract line, never from the wording the model wrote.
    # The OCR line gives the edges; the model's own box is used only for a group no OCR line belongs to.
    assert blocks[0]["location"]["coordinateSource"] == "TESSERACT"
    assert blocks[0]["location"]["bbox"] == pytest.approx([0.05, 40 / 600, 0.75, 80 / 600])
    assert blocks[1]["location"]["coordinateSource"] == "VISION"
    assert blocks[1]["location"]["bbox"] == pytest.approx([0.05, 0.8, 0.9, 0.9])
    assert blocks[2]["location"]["coordinateSource"] == "VISION_COPY_REGION" and blocks[2]["location"]["bbox"] is None
    assert warnings == []


async def test_falls_back_to_tesseract_lines_when_the_vision_read_fails(monkeypatch):
    t = tile()
    monkeypatch.setattr(ad_reading, "local_ocr", tesseract_lines)

    class BrokenProvider:
        client = True

        async def parse(self, *args, **kwargs):
            raise RuntimeError("synthetic")

    blocks, warnings = await read_tile(BrokenProvider(), t)
    assert [b["text"] for b in blocks] == ["큰 제목", "2 주 안에 개 선", "Pyunkang Yul"] and {b["role"] for b in blocks} == {"claim"}
    assert warnings == [f"{t.id}:OCR_PROVIDER_FAILED"]


def test_product_text_the_model_repeats_is_removed_but_short_ad_words_stay():
    assert strip_product(["Red Ginseng Propolis Repairing Moisturizer", "홍삼 케어"], ["Red Ginseng Propolis Repairing Moisturizer"]) == ["홍삼 케어"]
    assert strip_product(["30 ml / 1.01 Fl. oz."], ["30 ml 1.01 Fl. oz."]) == []
    assert strip_product(["30%"], ["30 ml"]) == ["30%"]


def test_the_ad_copy_prompt_and_schema_fit_the_model_input_budget():
    from capstone_agent.context_budget import prepare_parse
    from capstone_agent.instructions import instruction
    # Two images are sent (original and box drawing), so the fixed cost must still leave room for the box list.
    prepare_parse(instruction("literal_reread"), "잘린 영역의 글자를 한 줄씩 그대로 옮긴다.", ad_reading.Reread, image=True)
    bounded, output_cap, compacted = prepare_parse(instruction("ad_copy_reading"), "1: x 0~10, y 0~10\n" * 120, AdCopy, image=True)
    assert output_cap >= 3000 and not compacted


def test_each_ocr_line_goes_to_the_group_it_matches_best():
    lines = [{"text": "임 상 으로 검 증 된"}, {"text": "개 선 효과"}, {"text": "*편강 3종 시리즈 인체피부 개선효과 평가"}]
    headline = "임상으로 검증된 7종 피부 노화 징후 개선 효과"
    footnote = "*편강 레드 진생 프로폴리스 3종 시리즈에 대한 인체피부 개선효과 평가시험"
    foot, head = assign_lines([footnote, headline], [[], []], lines)  # footnote first: order must not decide
    # '개선 효과' is in both, but it is the whole headline line and only a fragment of the footnote.
    assert [l["text"] for l in head] == ["임 상 으로 검 증 된", "개 선 효과"]
    assert [l["text"] for l in foot] == ["*편강 3종 시리즈 인체피부 개선효과 평가"]


async def test_recall_pass_reads_ocr_lines_the_first_read_skipped(monkeypatch):
    t = tile()
    width, height = t.source_size

    def lines(_, scale=1.0):
        if scale != 1.0:
            return []
        return [{"id": f"{t.id}:ocr:{i}", "text": text, "ocrConfidence": 90.0, "words": [],
                 "location": t.location(precise_box=[20/width, y/height, 300/width, (y+30)/height], method="TESSERACT")}
                for i, (text, y) in enumerate([("2주 안에 개선", 40), ("*연구기관 피부과학연구소", 500)])]
    monkeypatch.setattr(ad_reading, "local_ocr", lines)
    calls = []

    class Provider:
        client = True

        async def parse(self, prompt, schema, **kwargs):
            calls.append(kwargs["text"])
            if len(calls) == 1:  # the first read skips the small footnote
                return AdCopy(layout=[], on_product_text=[], groups=[{"kind": "copy", "role": "claim", "segments": ["2주 안에 개선"], "labels": [],
                                                          "box_numbers": [1], "box": []}])
            return AdCopy(layout=[], on_product_text=[], groups=[
                {"kind": "footnote", "role": "condition", "segments": ["*연구기관: 피부과학연구소"], "labels": [], "box_numbers": [1], "box": []},
                {"kind": "copy", "role": "claim", "segments": ["2주 안에 개선"], "labels": [], "box_numbers": [], "box": []}])  # not anchored

    blocks, warnings = await read_tile(Provider(), t)
    assert len(calls) == 2 and "1: x 20~300, y 500~530" in calls[1] and "y 40~70" not in calls[1]
    assert [(b["role"], b["text"]) for b in blocks] == [("claim", "2주 안에 개선"), ("condition", "*연구기관: 피부과학연구소")]
    assert blocks[1]["location"]["coordinateSource"] == "TESSERACT"


def paddle_fake(t, lines, confidence=99.0):
    width, height = t.source_size
    return [{"id": f"{t.id}:pocr:{i}", "text": text, "ocrConfidence": confidence, "words": [],
             "location": t.location(precise_box=[20/width, y/height, 300/width, (y+30)/height], method="PADDLE_OCR")}
            for i, (text, y) in enumerate(lines)]


async def test_paddle_read_that_disagrees_is_reread_and_follows_the_ocr_letters(monkeypatch):
    t = tile()
    monkeypatch.setenv("OCR_ENGINE", "paddle")
    monkeypatch.setattr(ad_reading.paddle_ocr, "enabled", lambda: True)
    monkeypatch.setattr(ad_reading.paddle_ocr, "paddle_lines",
                        lambda tile: paddle_fake(tile, [("7종 피부 노화 징후", 40), ("탄성 복원 63.91% 개선", 100)]))
    asked = []

    class Provider:
        client = True

        async def parse(self, prompt, schema, **kwargs):
            asked.append(schema.__name__)
            if schema is AdCopy:
                return AdCopy(layout=[], on_product_text=[], groups=[
                    {"kind": "copy", "role": "claim", "segments": ["7종 피부 노화 증후"], "labels": [], "box_numbers": [1], "box": []},
                    {"kind": "table", "role": "result", "segments": ["탄성 복원 63.91% 개선"], "labels": [], "box_numbers": [2], "box": []}])
            assert schema is ad_reading.Reread and isinstance(kwargs["image"], str)  # one enlarged crop
            return ad_reading.Reread(lines=["7종 피부 노화 징후"])

    blocks, warnings = await read_tile(Provider(), t)
    # The vision read rewrote 징후 to 증후; the enlarged literal reread agrees with the OCR letters and wins.
    assert blocks[0]["text"] == "7종 피부 노화 징후" and blocks[0]["readCheck"]["status"] == "corrected"
    assert blocks[0]["location"]["coordinateSource"] == "PADDLE_OCR"
    assert blocks[1]["readCheck"] == {"status": "agreed"}
    assert asked.count("Reread") == 1 and warnings == []


async def test_a_number_the_reread_cannot_settle_is_left_for_the_reviewer(monkeypatch):
    t = tile()
    monkeypatch.setattr(ad_reading.paddle_ocr, "enabled", lambda: True)
    monkeypatch.setattr(ad_reading.paddle_ocr, "paddle_lines", lambda tile: paddle_fake(tile, [("탄성 복원 63.91% 개선", 40)]))

    class Provider:
        client = True

        async def parse(self, prompt, schema, **kwargs):
            if schema is AdCopy:
                return AdCopy(layout=[], on_product_text=[], groups=[
                    {"kind": "table", "role": "result", "segments": ["탄성 복원 68.91% 개선"], "labels": [], "box_numbers": [1], "box": []}])
            return ad_reading.Reread(lines=["탄성 복원 68.91% 개선"])

    [block], _ = await read_tile(Provider(), t)
    assert block["text"] == "탄성 복원 68.91% 개선"
    assert block["readCheck"]["status"] == "needs_review" and block["readCheck"]["ocr"] == "탄성 복원 63.91% 개선"


async def test_paddle_failure_falls_back_to_tesseract(monkeypatch):
    t = tile()
    monkeypatch.setattr(ad_reading.paddle_ocr, "enabled", lambda: True)

    def broken(tile):
        raise RuntimeError("synthetic")
    monkeypatch.setattr(ad_reading.paddle_ocr, "paddle_lines", broken)
    monkeypatch.setattr(ad_reading, "local_ocr", tesseract_lines)
    lines, warnings, engine = await ad_reading.ocr_lines(t)
    assert engine == "TESSERACT" and warnings == [f"{t.id}:PADDLE_OCR_FAILED"] and lines


def test_paddle_word_boxes_split_a_line_by_character_share():
    from capstone_agent.paddle_ocr import split_words
    words = split_words("눈가 주름 개선", [0.0, 0.1, 0.8, 0.2])
    assert [w["text"] for w in words] == ["눈가", "주름", "개선"]
    assert words[0]["bbox"][0] == 0.0 and words[-1]["bbox"][2] == 0.8 and words[1]["bbox"][0] == pytest.approx(0.3)


def test_list_numbers_and_neighbour_text_in_the_ocr_are_not_disagreements():
    assert ad_reading.disagrees("임상으로 검증된 7종 피부 노화 전후 개선 효과", "임상으로 검증된 7종피부노화징후 개선 효과")  # one letter
    assert not ad_reading.disagrees("팔자 주름 9.83% 개선", "3 팔자 주름 9.83% 개선")
    assert not ad_reading.disagrees("원료적 특성에 한함", "상기 내용은 원료적 특성에 한함")
    assert ad_reading.disagrees("탄성 복원 68.91% 개선", "1 탄성 복원 63.91% 개선")
    assert ad_reading.disagrees("피부보호자수 평가", "피부노화지수평가")


async def test_a_reread_that_repeats_the_vision_text_confirms_it(monkeypatch):
    t = tile()
    monkeypatch.setattr(ad_reading.paddle_ocr, "enabled", lambda: True)
    monkeypatch.setattr(ad_reading.paddle_ocr, "paddle_lines", lambda tile: paddle_fake(tile, [("색이 열어질 수 있으나", 40)], confidence=72.0))

    class Provider:
        client = True

        async def parse(self, prompt, schema, **kwargs):
            if schema is AdCopy:
                return AdCopy(layout=[], on_product_text=[], groups=[
                    {"kind": "footnote", "role": "condition", "segments": ["색이 옅어질 수 있으나"], "labels": [], "box_numbers": [1], "box": []}])
            return ad_reading.Reread(lines=["색이 옅어질 수 있으나"])

    [block], _ = await read_tile(Provider(), t)
    assert block["text"] == "색이 옅어질 수 있으나" and block["readCheck"]["status"] == "confirmed"


def test_one_letter_lines_do_not_join_a_group():
    lines = [{"text": "2"}, {"text": "20층 보습 20.48% 개선"}, {"text": "3"}]
    [hits] = assign_lines(["20층 보습 20.48% 개선"], [[1, 3]], lines)
    assert [l["text"] for l in hits] == ["20층 보습 20.48% 개선"]


async def test_a_very_confident_ocr_line_corrects_a_letter_two_vision_reads_got_wrong(monkeypatch):
    t = tile()
    monkeypatch.setattr(ad_reading.paddle_ocr, "enabled", lambda: True)

    class Provider:
        client = True

        async def parse(self, prompt, schema, **kwargs):
            if schema is AdCopy:
                return AdCopy(layout=[], on_product_text=[], groups=[
                    {"kind": "copy", "role": "claim", "segments": ["7종 피부 노화 전후"], "labels": [], "box_numbers": [1], "box": []}])
            return ad_reading.Reread(lines=["7종 피부 노화 전후"])  # the reread repeats the plausible rewrite

    # The OCR's letters win only when it is very sure of the line (>= 95); spacing stays the vision's.
    monkeypatch.setattr(ad_reading.paddle_ocr, "paddle_lines", lambda tile: paddle_fake(tile, [("7종피부노화징후", 40)], 100.0))
    [block], _ = await read_tile(Provider(), t)
    assert block["text"] == "7종 피부 노화 징후" and block["readCheck"]["status"] == "corrected_by_ocr"
    monkeypatch.setattr(ad_reading.paddle_ocr, "paddle_lines", lambda tile: paddle_fake(tile, [("7종피부노화징후", 40)], 92.6))
    [block], _ = await read_tile(Provider(), t)
    assert block["text"] == "7종 피부 노화 전후" and block["readCheck"]["status"] == "needs_review"


async def test_a_patent_number_in_the_image_survives_when_the_vision_read_leaves_the_certificate_out(monkeypatch):
    from capstone_agent.patent_certificate import patent_numbers
    t = tile()
    monkeypatch.setattr(ad_reading.paddle_ocr, "enabled", lambda: True)
    # PaddleOCR drops the first letter of the spaced-out label: '허 제 0440357 호'.
    monkeypatch.setattr(ad_reading.paddle_ocr, "paddle_lines",
                        lambda tile: paddle_fake(tile, [("허 제 0440357 호", 40), ("불꺼짐 탄력섬유재생!", 100)]))

    class Provider:
        client = True

        async def parse(self, prompt, schema, **kwargs):
            return AdCopy(layout=[], on_product_text=[], groups=[
                {"kind": "copy", "role": "claim", "segments": ["불꺼짐 탄력섬유재생!"], "labels": [], "box_numbers": [2], "box": []}])

    blocks, _ = await read_tile(Provider(), t)
    certificate = [b for b in blocks if b.get("documentContext") == "patent_certificate"]
    assert [b["text"] for b in certificate] == ["특허 제 0440357 호"] and certificate[0]["role"] == "condition"
    assert patent_numbers("\n".join(b["text"] for b in blocks)) == ["10-0440357"]


def test_a_label_spelled_out_in_one_letter_lines_still_yields_the_patent_number():
    from capstone_agent.patent_certificate import patent_numbers
    t = tile()
    width, height = t.source_size

    def line(text, x0, x1):
        return {"id": text, "text": text, "words": [], "ocrConfidence": 90.0,
                "location": t.location(precise_box=[x0/width, 100/height, x1/width, 130/height], method="PADDLE_OCR")}
    row = [line("0440357호", 200, 300), line("특", 20, 40), line("제", 120, 140), line("허", 70, 90),
           line("탄력섬유재생!", 20, 300)]
    row[-1]["location"] = t.location(precise_box=[20/width, 300/height, 300/width, 330/height], method="PADDLE_OCR")
    [block] = ad_reading.certificate_blocks(t, row, [])
    assert block["text"] == "특허 제 0440357 호" and patent_numbers(block["text"]) == ["10-0440357"]
    assert block["location"]["bbox"][0] == pytest.approx(20 / width) and block["location"]["bbox"][2] == pytest.approx(300 / width)


def test_a_bare_number_read_by_the_vision_does_not_hide_the_certificate_label():
    t = tile()
    width, height = t.source_size

    def line(text, x0, x1):
        return {"id": text, "text": text, "words": [], "ocrConfidence": 90.0,
                "location": t.location(precise_box=[x0/width, 100/height, x1/width, 130/height], method="PADDLE_OCR")}
    bare = {"text": "0440357 호", "role": "result", "segments": ["0440357 호"]}
    [block] = ad_reading.certificate_blocks(t, [line("특", 20, 40), line("허 제", 70, 120), line("0440357호", 200, 300)], [bare])
    assert block["text"] == "특허 제 0440357 호"
    assert ad_reading.certificate_blocks(t, [line("특허 제 0440357 호", 20, 300)], [{"text": "특허 제 0440357 호"}]) == []


def test_letters_spread_out_by_design_are_joined_into_the_word():
    assert ad_reading.join_spaced("특 허 증") == "특허증"
    assert ad_reading.join_spaced("출 원 일 2001년 10월 25일") == "출원일 2001년 10월 25일"
    assert ad_reading.join_spaced("눈가 주름 12.08% 개선") == "눈가 주름 12.08% 개선"


def test_ocr_letters_swap_letters_only_never_figures():
    assert ad_reading.ocr_letters("특 허 중", "특허증") == "특 허 증"
    assert ad_reading.ocr_letters("탄성 복원 68.91% 개선", "탄성 복원 63.91% 개선") is None
    assert ad_reading.ocr_letters("팔자 주름 9.83% 개선", "3 팔자 주름 9.83% 개선") is None  # nothing to swap
    assert ad_reading.ocr_letters("화장품 책임판매업자", "책임판매업자") is None  # a dropped word is not a swap


async def test_a_certificate_is_one_context_unit_and_its_number_reaches_the_patent_lookup(monkeypatch):
    from capstone_agent.patent_certificate import patent_numbers
    t = tile()
    monkeypatch.setattr(ad_reading, "local_ocr", tesseract_lines)

    class Provider:
        client = True

        async def parse(self, prompt, schema, **kwargs):
            return AdCopy(layout=["특허증", "광고 제목"], on_product_text=[], groups=[
                {"kind": "certificate", "role": "claim", "labels": [], "box_numbers": [], "box": [],
                 "segments": ["특 허 증", "특허 제 0440357 호", "출원번호 제 2001-0065893 호", "출원일 2001년 10월 25일", "등록일 2004년 07월 05일"]},
                {"kind": "copy", "role": "claim", "segments": ["볼꺼짐 탄력섬유재생!"], "labels": [], "box_numbers": [], "box": []}])

    blocks, _ = await read_tile(Provider(), t)
    certificate, headline = blocks[0], blocks[1]
    assert certificate["kind"] == "certificate" and certificate["role"] == "condition"
    assert certificate["documentContext"] == "patent_certificate" and certificate["text"].startswith("특허증 특허 제 0440357 호")
    assert headline["role"] == "claim" and "documentContext" not in headline
    assert patent_numbers(certificate["text"]) == ["10-0440357"]


async def test_a_certificate_split_into_field_blocks_is_put_back_together(monkeypatch):
    t = tile()
    monkeypatch.setattr(ad_reading, "local_ocr", tesseract_lines)

    class Provider:
        client = True

        async def parse(self, prompt, schema, **kwargs):
            def g(kind, text):
                return {"kind": kind, "role": "claim", "segments": [text], "labels": [], "box_numbers": [], "box": []}
            return AdCopy(layout=[], on_product_text=[], groups=[
                g("copy", "특허증"), g("copy", "볼꺼짐 탄력섬유재생!"), g("footnote", "특허 제 0440357 호"),
                g("footnote", "출원번호 제 2001-0065893 호"), g("footnote", "출원일 2001년 10월 25일")])

    blocks, _ = await read_tile(Provider(), t)
    assert [b["text"] for b in blocks] == ["특허증 특허 제 0440357 호 출원번호 제 2001-0065893 호 출원일 2001년 10월 25일",
                                          "볼꺼짐 탄력섬유재생!"]
    assert blocks[0]["kind"] == "certificate" and blocks[0]["documentContext"] == "patent_certificate"


def test_a_confident_line_corrects_its_own_part_of_the_group():
    fixed = ad_reading.ocr_letters("임상으로 검증된 7종 피부 노화 정후 개선 효과", "7종피부노화징후", partial=True)
    assert fixed == "임상으로 검증된 7종 피부 노화 징후 개선 효과"


def test_a_certificate_gets_back_the_title_the_vision_read_dropped():
    t = tile()
    width, height = t.source_size
    loc = t.location(precise_box=[20/width, 200/height, 300/width, 230/height], method="PADDLE_OCR")
    cert = {"id": "c", "text": "특허 제 0440357 호 출원일 2001년 10월 25일", "segments": ["특허 제 0440357 호", "출원일 2001년 10월 25일"],
            "kind": "certificate", "role": "condition", "labels": [], "words": [], "location": loc}
    title = [{"id": x, "text": x, "words": [], "location": t.location(precise_box=[x0/width, 100/height, (x0+20)/width, 130/height], method="PADDLE_OCR")}
             for x, x0 in (("특", 20), ("허", 60), ("증", 100))]
    [merged] = ad_reading.merge_certificate(t, [cert], title)
    assert merged["segments"][0] == "특허증" and merged["text"].startswith("특허증 특허 제 0440357 호")
