"""Read the ad copy of an image section: Tesseract says where text is, one vision call reads it.

Only the copy the ad speaks to consumers (headline, effect claims, numbers and results, test-condition footnotes) is
kept. Text printed on the product itself (label, jar, box) and what photos or graphs show are not read at all, because
this service looks for problems in advertising copy. Each section is read in one vision call that gets the original
and a copy with numbered Tesseract boxes, and returns the copy grouped into meaning units with a role.

A block is one meaning unit: {id, text, segments, role, location}. role is claim, condition (test footnote) or
result (numbers, survey results). Condition blocks are context for the evidence comparison, not claims to review.
"""
import asyncio
from dataclasses import replace
import os
import re
from difflib import SequenceMatcher
from typing import Literal

from pydantic import BaseModel

from capstone_shared.documents import Tile, local_ocr
from .instructions import instruction


# Tesseract is CPU bound. Every section reads at two sizes, so unbounded threads time out under load.
OCR_SLOTS = asyncio.Semaphore(int(os.getenv("OCR_CONCURRENCY", "3")))


class Group(BaseModel):
    role: Literal["claim", "condition", "result"]
    segments: list[str]
    labels: list[str]
    # Numbered boxes on the second image that hold this group's letters.
    box_numbers: list[int]
    # Where the group sits in the first image, [left, top, right, bottom] in 0~1000 of its width and height.
    box: list[float]


class AdCopy(BaseModel):
    # Written first so the model decides what to leave out before it writes the copy.
    on_product_text: list[str]
    groups: list[Group]


def norm(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]", "", text or "").lower()


def strip_product(segments: list[str], on_product: list[str]) -> list[str]:
    """Drop segments the model itself listed as product-surface text but repeated in the copy."""
    banned = [norm(x) for x in on_product if norm(x)]
    kept = []
    for segment in segments:
        n = norm(segment)
        if n and any(n == b or (len(n) >= 6 and (n in b or b in n)) for b in banned):
            continue
        kept.append(segment)
    return kept


def line_score(line: dict, text: str) -> float:
    """How well an OCR line spells part of a group: 1 when the whole line is in the group, else the share of the
    line in the longest common run, 0 below the match bar. Used for the position only, never for the wording."""
    seen, target = norm(line["text"]), norm(text)
    if not seen or not target:
        return 0.0
    if seen in target:
        return 1.0
    size = SequenceMatcher(None, seen, target).find_longest_match(0, len(seen), 0, len(target)).size
    return size / len(seen) if size >= max(3, int(len(seen) * 0.6)) else 0.0


def assign_lines(texts: list[str], numbers: list[list[int]], lines: list[dict]) -> list[list[dict]]:
    """Each OCR line goes to the one group it matches best. A footnote that repeats the headline's words
    ('개선 효과') must not take the headline's line. A box the model named counts when its text still
    resembles the group (Tesseract garbles large styled Korean)."""
    hits = [[] for _ in texts]
    for i, line in enumerate(lines, 1):
        scores = []
        for g, text in enumerate(texts):
            score = line_score(line, text)
            if i in numbers[g] and SequenceMatcher(None, norm(line["text"]), norm(text)).ratio() >= 0.3:
                score = max(score, 0.9)
            # Ties go to the group the line makes up more of (a headline line, not a footnote fragment).
            scores.append(score + (0.01 * len(norm(line["text"])) / max(1, len(norm(text))) if score else 0))
        if scores and max(scores) > 0:
            hits[scores.index(max(scores))].append(line)
    return hits


def overlaps(a, b) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def union_box(lines: list[dict]):
    boxes = [line["location"]["bbox"] for line in lines if line["location"].get("bbox")]
    if not boxes:
        return None
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def vision_box(tile: Tile, box: list[float]):
    """A model box (0~1000 of the section) as a page-ratio bbox, or None when it is not a usable box."""
    if len(box) != 4 or not all(0 <= v <= 1000 for v in box):
        return None
    x0, y0, x1, y1 = box
    if x1 - x0 < 5 or y1 - y0 < 5:
        return None
    width, height = tile.source_size
    left, top = tile.box[0], tile.box[1]
    return [(left + x0 / 1000 * tile.image.width) / width, (top + y0 / 1000 * tile.image.height) / height,
            (left + x1 / 1000 * tile.image.width) / width, (top + y1 / 1000 * tile.image.height) / height]


def merge_boxes(seen, model):
    """Tesseract lines give the exact edges. The model box is used only when no OCR line belongs to the group."""
    if seen:
        return seen, "TESSERACT"
    return (model, "VISION") if model else (None, None)


def section_box(tile: Tile, line: dict) -> list[float]:
    """A line's box in the pixels of the section image (Tesseract boxes are page ratios)."""
    width, height = tile.source_size
    x0, y0, x1, y1 = line["location"]["bbox"]
    return [x0 * width - tile.box[0], y0 * height - tile.box[1], x1 * width - tile.box[0], y1 * height - tile.box[1]]


def annotate(tile: Tile, lines: list[dict]):
    """The section with numbered boxes drawn around each Tesseract line (box labels sit outside the box)."""
    from PIL import ImageDraw
    marked = tile.image.copy().convert("RGB")
    draw = ImageDraw.Draw(marked)
    for i, line in enumerate(lines, 1):
        x0, y0, x1, y1 = section_box(tile, line)
        draw.rectangle((x0 - 2, y0 - 2, x1 + 2, y1 + 2), outline=(255, 0, 80), width=2)
        label = str(i)
        lx, ly = max(0, x0 - 4 - 8 * len(label)), max(0, y0 - 2)
        draw.rectangle((lx, ly, lx + 8 * len(label) + 2, ly + 13), fill=(255, 0, 80))
        draw.text((lx + 1, ly), label, fill=(255, 255, 255))
    return marked


def data_url(image, longest=2048) -> str:
    import base64
    from io import BytesIO
    image = image.copy().convert("RGB")
    image.thumbnail((longest, longest))
    out = BytesIO()
    image.save(out, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(out.getvalue()).decode()


def tesseract_blocks(tile: Tile, lines: list[dict]) -> list[dict]:
    """Fallback when the vision read fails: Tesseract lines as they are, role unknown (treated as claims)."""
    return [{"id": line["id"], "text": line["text"], "segments": [line["text"]], "role": "claim", "location": line["location"]}
            for line in lines]


async def read_copy(provider, tile: Tile, lines: list[dict], note: str = "") -> AdCopy:
    boxes = "\n".join(
        "{}: x {:.0f}~{:.0f}, y {:.0f}~{:.0f}".format(i, *(lambda b: (b[0], b[2], b[1], b[3]))(section_box(tile, line)))
        for i, line in enumerate(lines, 1)) or "(상자 없음)"
    return await provider.parse(
        instruction("ad_copy_reading"), AdCopy,
        image=[data_url(tile.image), data_url(annotate(tile, lines))],
        text=f"{note}\n글자가 있을 만한 위치의 번호 상자(구간 이미지 픽셀 기준, 너비 {tile.image.width}, 높이 {tile.image.height}):\n{boxes}".strip())


def copy_groups(reading: AdCopy) -> list[tuple]:
    groups = []
    for n, group in enumerate(reading.groups):
        segments = strip_product([x.strip() for x in group.segments if x.strip()], reading.on_product_text)
        if segments:
            groups.append((n, group, segments, " ".join(segments)))
    return groups


async def snap_to_ocr(tile: Tile, text: str, model_box: list[float]) -> list[dict]:
    """Model boxes are rough. Read just the area around one (padded) at three sizes and keep the OCR lines that
    spell the group: small crops let Tesseract find text over photos and very large or very small letters."""
    width, height = tile.source_size
    x0, y0, x1, y1 = model_box[0] * width, model_box[1] * height, model_box[2] * width, model_box[3] * height
    pad_x, pad_y = (x1 - x0) * 0.25 + 10, (y1 - y0) * 0.5 + 10
    left, top = max(tile.box[0], int(x0 - pad_x)), max(tile.box[1], int(y0 - pad_y))
    right, bottom = min(tile.box[2], int(x1 + pad_x)), min(tile.box[3], int(y1 + pad_y))
    if right - left < 8 or bottom - top < 8:
        return []
    crop = replace(tile, box=(left, top, right, bottom),
                   image=tile.image.crop((left - tile.box[0], top - tile.box[1], right - tile.box[0], bottom - tile.box[1])))
    found = []
    for scale in (1.0, 2.0, 0.5):
        try:
            async with OCR_SLOTS:
                lines = await asyncio.to_thread(local_ocr, crop, scale)
        except Exception:
            continue
        found += [line for line in lines if line["location"].get("bbox") and line_score(line, text) > 0
                  and not any(overlaps(line["location"]["bbox"], kept["location"]["bbox"]) for kept in found)]
    return found


def block(tile: Tile, key: str, group: Group, segments: list[str], text: str, hits: list[dict]) -> dict:
    box, method = merge_boxes(union_box(hits), vision_box(tile, group.box))
    return {"id": f"{tile.id}:copy:{key}", "text": text, "segments": segments, "role": group.role, "labels": group.labels,
            "words": [w for line in hits for w in line.get("words", [])],
            "location": tile.location(precise_box=box, method=method or "VISION_COPY_REGION")}


async def read_tile(provider, tile: Tile) -> tuple[list[dict], list[str]]:
    """One section -> (blocks, warnings). Raises only when both Tesseract and the model are unavailable."""
    warnings = []
    try:
        async with OCR_SLOTS:
            lines = await asyncio.to_thread(local_ocr, tile)
        lines = [line for line in lines if line["location"].get("bbox")]
    except Exception:
        lines, warnings = [], [f"{tile.id}:OCR_NOT_AVAILABLE"]
    try:
        # Large headline letters are often missed at full size; a half-size read finds them. Its failure only
        # loses those extra lines.
        async with OCR_SLOTS:
            small = await asyncio.to_thread(local_ocr, tile, 0.5)
        lines += [line for line in small if line["location"].get("bbox") and len(norm(line["text"])) >= 2
                  and not any(overlaps(line["location"]["bbox"], kept["location"]["bbox"]) for kept in lines)]
        lines.sort(key=lambda line: (line["location"]["bbox"][1], line["location"]["bbox"][0]))
    except Exception:
        warnings.append(f"{tile.id}:OCR_SMALL_PASS_FAILED")
    if not provider.client:
        return tesseract_blocks(tile, lines), warnings
    try:
        reading = await read_copy(provider, tile, lines)
    except Exception:
        return tesseract_blocks(tile, lines), warnings + [f"{tile.id}:OCR_PROVIDER_FAILED"]
    groups = copy_groups(reading)
    assigned = assign_lines([g[3] for g in groups], [g[1].box_numbers for g in groups], lines)
    for (n, group, segments, text), hits in zip(groups, assigned):
        model = vision_box(tile, group.box)
        if not hits and model:
            hits += await snap_to_ocr(tile, text, model)
    blocks = [block(tile, f"{n}", group, segments, text, hits) for (n, group, segments, text), hits in zip(groups, assigned)]
    # Recall pass (the OCR-first idea): OCR lines no group took and that are not product text are shown to the
    # model once more, alone, so copy the first read skipped is not lost. Only groups anchored on those lines count.
    taken = {id(line) for hits in assigned for line in hits}
    product = " ".join(reading.on_product_text)
    missed = [line for line in lines if id(line) not in taken and len(norm(line["text"])) >= 4
              and line.get("ocrConfidence", 100) >= 50 and not (product and line_score(line, product) > 0)]
    if missed:
        try:
            again = await read_copy(provider, tile, missed, note="앞에서 읽지 않은 상자들이다. 상자 안의 글자만 읽고, 광고 문구가 아니면 groups를 비운다.")
            extra = copy_groups(again)
            for (n, group, segments, text), hits in zip(extra, assign_lines([g[3] for g in extra], [g[1].box_numbers for g in extra], missed)):
                if hits:
                    blocks.append(block(tile, f"r{n}", group, segments, text, hits))
        except Exception:
            warnings.append(f"{tile.id}:RECALL_PASS_FAILED")
    return blocks, warnings
