import pytest
from PIL import Image

from capstone_agent import ad_reading
from capstone_agent.ad_reading import AdCopy, read_tile, strip_product
from capstone_shared.documents import tile_image


def tile():
    return tile_image(Image.new("RGB", (400, 600), "white"), "a", "ad.png")[0]


def tesseract_lines(t):
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
            return AdCopy(on_product_text=["Pyunkang Yul"], groups=[
                {"role": "claim", "segments": ["2주 안에 개선"], "labels": [], "box": [40, 60, 800, 150]},
                {"role": "claim", "segments": ["Pyunkang Yul"], "labels": [], "box": []},  # model repeated the label: code removes it
                {"role": "condition", "segments": ["*연구 기간 : 2025.02.24~03.14", "20명 대상"], "labels": [], "box": [50, 800, 900, 900]},
                {"role": "result", "segments": ["12.08%", "눈가 주름 감소"], "labels": ["사용 전", "사용 후"], "box": [0, 0, 1, 1]}])  # too small to be a box

    blocks, warnings = await read_tile(FakeProvider(), t)
    assert [(b["role"], b["text"]) for b in blocks] == [
        ("claim", "2주 안에 개선"), ("condition", "*연구 기간 : 2025.02.24~03.14 20명 대상"), ("result", "12.08% 눈가 주름 감소")]
    assert blocks[2]["labels"] == ["사용 전", "사용 후"]
    assert len(seen["images"]) == 2 and seen["images"][0].startswith("data:image/jpeg")  # original and the box drawing
    assert "1: x 20~300, y 40~80" in seen["text"]  # boxes are in section pixels
    # Position comes from the matching Tesseract line, never from the wording the model wrote.
    # A model box that overlaps the Tesseract line widens it; one with no Tesseract line is used as it is.
    assert blocks[0]["location"]["coordinateSource"] == "TESSERACT+VISION"
    assert blocks[0]["location"]["bbox"] == pytest.approx([0.04, 0.06, 0.8, 0.15])  # union of the line and the model box
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
    assert [b["text"] for b in blocks] == ["2 주 안에 개 선", "Pyunkang Yul"] and {b["role"] for b in blocks} == {"claim"}
    assert warnings == [f"{t.id}:OCR_PROVIDER_FAILED"]


def test_product_text_the_model_repeats_is_removed_but_short_ad_words_stay():
    assert strip_product(["Red Ginseng Propolis Repairing Moisturizer", "홍삼 케어"], ["Red Ginseng Propolis Repairing Moisturizer"]) == ["홍삼 케어"]
    assert strip_product(["30 ml / 1.01 Fl. oz."], ["30 ml 1.01 Fl. oz."]) == []
    assert strip_product(["30%"], ["30 ml"]) == ["30%"]


def test_the_ad_copy_prompt_and_schema_fit_the_model_input_budget():
    from capstone_agent.context_budget import prepare_parse
    from capstone_agent.instructions import instruction
    # Two images are sent (original and box drawing), so the fixed cost must still leave room for the box list.
    bounded, output_cap, compacted = prepare_parse(instruction("ad_copy_reading"), "1: x 0~10, y 0~10\n" * 120, AdCopy, image=True)
    assert output_cap >= 3000 and not compacted
