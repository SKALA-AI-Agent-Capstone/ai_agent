"""OCR and vision reading comparison (experiment only, never merged into main).

The same uploaded image is cut into content sections (capstone_shared.documents.tile_image), and every
section is read two ways:

  Method 1  Tesseract reads the lines, a text model cleans spacing and broken OCR without seeing the
            image, then the vision model reads the section itself and merges the cleaned OCR lines in,
            dropping duplicates and marking what only OCR found.
  Method 2  Tesseract only says where text is (line boxes). The vision model gets the section with the
            numbered boxes drawn on it, reads the text in each box, adds text outside the boxes, and
            describes the non-text visuals.

Both return the same shape: text blocks of quoted segments, and factual [이미지] descriptions with no
judgment. Run with experiments/ocr_test/run.sh and open http://127.0.0.1:8090.
"""
import asyncio
import base64
from io import BytesIO
import os
from pathlib import Path
import time

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from openai import AsyncOpenAI
from PIL import Image, ImageDraw, ImageOps
from typing import Literal

from pydantic import BaseModel

from capstone_shared.documents import local_ocr, tile_image

HERE = Path(__file__).parent
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-5.4-mini")
TEXT_MODEL = os.getenv("TEXT_MODEL", "gpt-5.4-mini")
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", ""), timeout=90, max_retries=1)
limit = asyncio.Semaphore(int(os.getenv("OCR_TEST_CONCURRENCY", "4")))

AD_ONLY = (
    "이 서비스는 광고에 쓰인 광고 문구와 홍보 문구의 문제를 찾는다. 따라서 옮길 글자는 광고가 소비자에게 말하는 문구뿐이다: "
    "제목, 효능과 효과 주장, 수치와 시험 결과, 설문 결과, 시험 조건 각주, 성분이나 기술 소개 문구. "
    "제품 자체에 인쇄된 글자(용기, 상자, 라벨, 포장에 적힌 제품명, 용량, 성분 표기, 슬로건, 로고)는 옮기지 않고 on_product_text에만 따로 적는다. "
    "구분 기준: 사진 속 사물 위에 인쇄된 글자면 제품 표면 글자, 광고 제작자가 이미지 위에 얹은 글자면 광고 문구다. "
    "사진이나 그래프의 내용은 묘사하지 않는다. 글자만 옮긴다. "
)
SEGMENTS = (
    "글자는 이미지에 실제로 보이는 그대로 옮긴다. 띄어쓰기만 바르게 고치고, 단어나 숫자를 바꾸거나 추측해서 채우지 않는다. "
    "읽을 수 없는 부분은 [판독 불가]로 둔다. 시각적으로 구분되는 글자 단위(한 줄, 한 문구, 한 항목)마다 segments의 한 칸으로 나누고, "
    "제목과 설명처럼 한 덩어리로 읽히는 글자들은 같은 block에 묶는다. 위에서 아래, 왼쪽에서 오른쪽 순서로 쓴다."
)


# ---------- output shapes ----------

class Block(BaseModel):
    segments: list[str]
    source: Literal["image", "ocr"]  # "ocr": found only because of the OCR reference lines


class Reading(BaseModel):
    blocks: list[Block]
    on_product_text: list[str]


class CleanLine(BaseModel):
    raw: str
    fixed: str
    dropped: bool


class Cleaned(BaseModel):
    lines: list[CleanLine]


class BoxBlock(BaseModel):
    segments: list[str]
    box_ids: list[int]


class Group(BaseModel):
    role: Literal["claim", "condition", "result"]
    segments: list[str]
    labels: list[str]


class GroupReading(BaseModel):
    groups: list[Group]
    on_product_text: list[str]  # text printed on the product itself, listed only so the exclusion can be checked


class BoxReading(BaseModel):
    blocks: list[BoxBlock]
    outside_boxes: list[str]
    on_product_text: list[str]


# ---------- helpers ----------

def data_url(image: Image.Image, longest=2048) -> str:
    image = image.copy().convert("RGB")
    image.thumbnail((longest, longest))
    out = BytesIO()
    image.save(out, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(out.getvalue()).decode()


async def parse(model, system, schema, text, images=()):
    content = [{"type": "input_text", "text": text}]
    content += [{"type": "input_image", "image_url": url, "detail": "high"} for url in images]
    async with limit:
        response = await client.responses.parse(
            model=model, input=[{"role": "system", "content": system}, {"role": "user", "content": content}],
            text_format=schema, store=False, max_output_tokens=6000)
    if response.output_parsed is None:
        raise RuntimeError("MODEL_REFUSED_OR_INCOMPLETE")
    usage = response.usage
    return response.output_parsed, {"input": usage.input_tokens, "output": usage.output_tokens} if usage else {}


def ocr_lines(tile):
    """Tesseract lines of one section with boxes in section pixels."""
    width, height = tile.source_size
    x0, y0 = tile.box[0], tile.box[1]
    lines = []
    for line in local_ocr(tile):
        bx0, by0, bx1, by1 = line["location"]["bbox"]
        lines.append({"text": line["text"], "conf": round(line.get("ocrConfidence") or 0, 1),
                      "box": [round(bx0 * width - x0), round(by0 * height - y0), round(bx1 * width - x0), round(by1 * height - y0)]})
    return lines


def annotate(image: Image.Image, lines) -> Image.Image:
    """Numbered thin boxes, labels outside the box so small text stays visible."""
    marked = image.copy().convert("RGB")
    draw = ImageDraw.Draw(marked)
    for i, line in enumerate(lines, 1):
        x0, y0, x1, y1 = line["box"]
        draw.rectangle((x0 - 2, y0 - 2, x1 + 2, y1 + 2), outline=(255, 0, 80), width=2)
        label = str(i)
        lx, ly = max(0, x0 - 4 - 8 * len(label)), max(0, y0 - 2)
        draw.rectangle((lx, ly, lx + 8 * len(label) + 2, ly + 13), fill=(255, 0, 80))
        draw.text((lx + 1, ly), label, fill=(255, 255, 255))
    return marked


# ---------- the two methods ----------

async def method_one(tile, lines):
    started, calls, tokens = time.monotonic(), 0, []
    cleaned = []
    if lines:
        result, usage = await parse(
            TEXT_MODEL,
            "로컬 OCR이 광고 이미지에서 읽은 줄들이다. 이미지는 볼 수 없다. 각 줄의 띄어쓰기를 고치고, OCR이 깨뜨린 글자는 "
            "문맥상 분명할 때만 바로잡는다. 의미를 알 수 없는 잡음 줄은 dropped=true로 둔다. 줄 수와 순서는 입력과 같게 유지한다.",
            Cleaned, "\n".join(line["text"] for line in lines))
        calls += 1
        tokens.append(usage)
        cleaned = [c.model_dump() for c in result.lines]
    reference = "\n".join(f"- {c['fixed']}" for c in cleaned if not c["dropped"]) or "(없음)"
    reading, usage = await parse(
        VISION_MODEL,
        "화장품 광고 이미지의 한 구간을 사실대로 옮긴다. " + SEGMENTS + " "
        "아래 OCR 참고 줄은 틀릴 수 있다. 이미지에서 직접 확인되는 글자만 쓰고, 이미지에서 확인되지 않는 OCR 줄은 버린다. "
        "직접 읽다가 놓쳤지만 OCR 참고 줄 덕분에 이미지에서 확인한 글자는 source='ocr', 나머지는 source='image'로 쓴다. "
        "같은 글자를 두 번 쓰지 않는다. " + AD_ONLY,
        Reading, f"OCR 참고 줄:\n{reference}", [data_url(tile.image)])
    calls += 1
    tokens.append(usage)
    return {"ocr": lines, "cleaned": cleaned, "blocks": [b.model_dump() for b in reading.blocks], "on_product": reading.on_product_text,
            "seconds": round(time.monotonic() - started, 1), "calls": calls, "tokens": tokens}


async def method_two(tile, lines):
    started = time.monotonic()
    marked = annotate(tile.image, lines)
    boxes = "\n".join(f"{i}: x {l['box'][0]}~{l['box'][2]}, y {l['box'][1]}~{l['box'][3]}" for i, l in enumerate(lines, 1)) or "(상자 없음)"
    reading, usage = await parse(
        VISION_MODEL,
        "화장품 광고 이미지의 한 구간이다. 첫 이미지는 원본, 두 번째는 글자가 있을 만한 위치에 번호 상자를 그린 것이다. "
        "상자마다 원본에서 그 위치의 글자를 읽어라. 상자 위치가 어긋나 있으면 가까운 글자를 읽는다. " + SEGMENTS + " "
        "각 block에는 그 글자가 들어 있던 상자 번호(box_ids)를 쓴다. 상자 밖에 있는데 상자가 놓친 글자는 outside_boxes에 쓴다. "
        "글자가 없는 상자는 무시한다. 상자가 하나도 없으면 box_ids는 빈 목록으로 두고 모든 글자를 outside_boxes에 쓴다. " + AD_ONLY,
        BoxReading, f"상자 좌표(구간 픽셀 기준, 너비 {tile.image.width}, 높이 {tile.image.height}):\n{boxes}",
        [data_url(tile.image), data_url(marked)])
    # The model sometimes cites box numbers that were never drawn; keep only real ones and move the rest outside.
    blocks, outside = [], list(reading.outside_boxes)
    for block in reading.blocks:
        ids = [i for i in block.box_ids if 1 <= i <= len(lines)]
        if ids:
            blocks.append({"segments": block.segments, "box_ids": ids})
        else:
            outside += block.segments
    return {"boxes": lines, "annotated": data_url(marked, 1200), "blocks": blocks,
            "outside": outside, "on_product": reading.on_product_text,
            "seconds": round(time.monotonic() - started, 1), "calls": 1, "tokens": [usage]}


GROUPING = (
    AD_ONLY +
    "글자는 이미지에 실제로 보이는 그대로 옮긴다. 띄어쓰기만 바르게 고치고 단어나 숫자를 바꾸거나 추측해서 채우지 않는다. 읽을 수 없으면 [판독 불가]. "
    "글자를 **의미 덩어리**로 묶는다. 덩어리는 읽는 사람이 하나의 항목으로 이해하는 단위이고 줄 단위가 아니다. "
    "큰 수치, 그 라벨, 평가 방법 괄호, 짝이 되는 그래프의 '사용 전/후' 라벨은 하나의 덩어리다. 예: ['12.08%', '눈가 주름 감소', '(피부노화지수 평가)']. "
    "반복되는 라벨(사용 전, 사용 후)은 덩어리마다 따로 쓰지 말고 해당 덩어리의 labels에 한 번만 쓴다. "
    "제목과 부연 문장은 한 덩어리, 각주(시험 기관, 기간, 대상, 기준)는 한 덩어리로 묶는다. "
    "덩어리마다 role을 정한다: claim(효능, 홍보 주장과 제목), condition(시험 조건 각주), result(수치나 설문 결과). "
    "사진이나 그래프 안에 들어 있는 짧은 캡션('사용 전' 같은 글자)은 글자 덩어리가 아니라 이미지 묘사에 포함한다. "
    "위에서 아래, 왼쪽에서 오른쪽 순서로 쓴다."
)
def norm(text: str) -> str:
    import re
    return re.sub(r"[^0-9A-Za-z가-힣]", "", text).lower()


def numbers(text: str) -> set[str]:
    import re
    return set(re.findall(r"\d+(?:[.,]\d+)?", text))


def verify(group_text: str, lines):
    """Code check of one group against Tesseract lines. Returns (status, box ids)."""
    from difflib import SequenceMatcher
    target = norm(group_text)
    if not target:
        return "vision_only", []
    hits, status = [], "vision_only"
    for i, line in enumerate(lines, 1):
        seen = norm(line["text"])
        if not seen:
            continue
        if seen in target or SequenceMatcher(None, seen, target).find_longest_match(0, len(seen), 0, len(target)).size >= max(3, int(len(seen) * 0.6)):
            hits.append(i)
    if hits:
        ocr_numbers = set().union(*(numbers(lines[i - 1]["text"]) for i in hits))
        status = "confirmed" if numbers(group_text) <= ocr_numbers or not numbers(group_text) else "number_mismatch"
    return status, hits


async def method_three(tile, lines):
    started = time.monotonic()
    marked = annotate(tile.image, lines)
    boxes = "\n".join(f"{i}: x {l['box'][0]}~{l['box'][2]}, y {l['box'][1]}~{l['box'][3]}" for i, l in enumerate(lines, 1)) or "(상자 없음)"
    reading, usage = await parse(
        VISION_MODEL,
        "화장품 광고 이미지의 한 구간이다. 첫 이미지는 원본, 두 번째는 글자가 있을 만한 위치에 번호 상자를 그린 것이다. "
        "상자는 읽을 위치를 알려 주는 참고일 뿐이니 글자는 원본에서 직접 읽고, 상자가 놓친 글자도 읽는다. " + GROUPING,
        GroupReading, f"상자 좌표(구간 픽셀 기준, 너비 {tile.image.width}, 높이 {tile.image.height}):\n{boxes}",
        [data_url(tile.image), data_url(marked)])
    groups = []
    for group in reading.groups:
        group.segments = [x for x in group.segments if x.strip()]
        if not group.segments:
            continue  # the model sometimes emits an empty group
        status, hits = verify(" ".join(group.segments), lines)
        groups.append({**group.model_dump(), "status": status, "box_ids": hits})
    # Only what the judging step needs goes on: the ad copy with its role and how sure the reading is, and the
    # visual notes. Boxes, labels of position, the model's skipped product text and raw OCR stay here.
    forward = {"copy": [{"role": g["role"], "text": g["segments"], "series": g["labels"],
                          "reading": {"confirmed": "ocr_confirmed", "number_mismatch": "check_numbers", "vision_only": "vision_only"}[g["status"]]}
                         for g in groups],
               }
    return {"boxes": lines, "annotated": data_url(marked, 1200), "groups": groups,
            "on_product": reading.on_product_text, "forward": forward,
            "seconds": round(time.monotonic() - started, 1), "calls": 1, "tokens": [usage]}


async def safely(coro):
    try:
        return await coro
    except Exception as error:  # Show the failure in the page instead of failing the whole comparison.
        return {"error": f"{type(error).__name__}: {str(error)[:300]}"}


async def section(index, tile):
    lines = await asyncio.to_thread(ocr_lines, tile)
    first, second, third = await asyncio.gather(
        safely(method_one(tile, lines)), safely(method_two(tile, lines)), safely(method_three(tile, lines)))
    return {"index": index, "box": list(tile.box), "image": data_url(tile.image, 1200),
            "method1": first, "method2": second, "method3": third}


app = FastAPI(title="OCR test")


@app.get("/")
async def page():
    return FileResponse(HERE / "index.html")


@app.post("/api/analyze")
async def analyze(request: Request):
    raw = await request.body()
    if not raw:
        return JSONResponse({"error": "이미지를 올려 주세요."}, status_code=400)
    if not os.getenv("OPENAI_API_KEY"):
        return JSONResponse({"error": "OPENAI_API_KEY가 없습니다."}, status_code=503)
    image = ImageOps.exif_transpose(Image.open(BytesIO(raw))).convert("RGB")
    started = time.monotonic()
    tiles = await asyncio.to_thread(tile_image, image, "upload", request.headers.get("x-file-name", "upload"))
    sections = await asyncio.gather(*(section(i, t) for i, t in enumerate(tiles)))
    return {"size": list(image.size), "sections": sections, "seconds": round(time.monotonic() - started, 1),
            "models": {"vision": VISION_MODEL, "text": TEXT_MODEL}}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("OCR_TEST_PORT", "8090")))
