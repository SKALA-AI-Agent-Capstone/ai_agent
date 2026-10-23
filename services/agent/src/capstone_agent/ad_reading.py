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
from . import paddle_ocr
from .patent_certificate import CERTIFICATE_NUMBER


# Tesseract is CPU bound. Every section reads at two sizes, so unbounded threads time out under load.
OCR_SLOTS = asyncio.Semaphore(int(os.getenv("OCR_CONCURRENCY", "3")))


class Group(BaseModel):
    # A unit a person reads as one: ad copy, one document (certificate), one footnote, one row of a figure list.
    kind: Literal["copy", "certificate", "footnote", "table"]
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
    # The units of the section, named before any is written out, so far-apart fields of one document are
    # planned as one unit instead of being grouped line by line.
    layout: list[str]
    groups: list[Group]


ROLE_OF_KIND = {"copy": "claim", "table": "result", "footnote": "condition", "certificate": "condition"}


def join_spaced(text: str) -> str:
    """Letters spread out by design ('특 허 증') become the word they spell ('특허증')."""
    return re.sub(r"(?<![가-힣])[가-힣](?: [가-힣]){2,}(?![가-힣])", lambda m: m.group(0).replace(" ", ""), text or "")


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
    # A one-letter line (a list index '2', a stray mark) is in almost every group and would stretch its box.
    if len(seen) < 2 or not target:
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
        segments = strip_product([join_spaced(x.strip()) for x in group.segments if x.strip()], reading.on_product_text)
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


def block(tile: Tile, key: str, group: Group, segments: list[str], text: str, hits: list[dict], engine="TESSERACT",
          check=None) -> dict:
    box, method = merge_boxes(union_box(hits), vision_box(tile, group.box))
    out = {"id": f"{tile.id}:copy:{key}", "text": text, "segments": segments, "kind": group.kind,
           "role": ROLE_OF_KIND.get(group.kind, group.role), "labels": group.labels,
           "words": [w for line in hits for w in line.get("words", [])],
           "location": tile.location(precise_box=box, method=(engine if method == "TESSERACT" else method) or "VISION_COPY_REGION")}
    if group.kind == "certificate":
        # One document's fields travel as one context block: the patent lookup reads its number, and neither
        # claim review nor the rule sweep treats its dates and numbers as claims.
        out["documentContext"] = "patent_certificate" if "특허" in text else "certificate"
    if check:
        out["readCheck"] = check
    return out


class Reread(BaseModel):
    lines: list[str]


def numbers(text: str) -> list[str]:
    """Claim figures only: decimals and percents. A bare list index ('3 팔자 주름') is not a claim figure."""
    return re.findall(r"\d+\.\d+|\d+(?=\s*%)", (text or "").replace(",", ""))


def figure_conflict(vision: str, ocr: str) -> bool:
    found, read = numbers(ocr), numbers(vision)
    return any(found.count(n) < read.count(n) for n in set(read))


def letters_apart(vision: str, ocr: str) -> int:
    """Fewest letter edits that turn the vision text into some stretch of the OCR text (extra OCR text around it,
    such as list numbers or a neighbouring line, costs nothing)."""
    a, b = norm(vision), norm(ocr)
    previous = [0] * (len(b) + 1)
    for i, x in enumerate(a, 1):
        current = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y))
        previous = current
    return min(previous)


def disagrees(vision: str, ocr: str) -> bool:
    """The OCR lines under a group do not support its vision read. Any letter that differs counts: a similarity
    ratio let one wrong letter through ('전후' and '진후' for '징후' in a 20-letter line)."""
    if not norm(vision) or not norm(ocr):
        return False
    return figure_conflict(vision, ocr) or letters_apart(vision, ocr) > 0


async def reread(provider, tile: Tile, hits: list[dict]) -> str:
    """Read just the OCR box of one group, enlarged, letter by letter (no grouping, no product rules)."""
    x0, y0, x1, y1 = (lambda boxes: (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes),
                                     max(b[3] for b in boxes)))([section_box(tile, h) for h in hits])
    crop = tile.image.crop((max(0, x0 - 8), max(0, y0 - 8), min(tile.image.width, x1 + 8), min(tile.image.height, y1 + 8)))
    crop = crop.resize((crop.width * 2, crop.height * 2))
    out = await provider.parse(instruction("literal_reread"), Reread, image=data_url(crop),
                               text="잘린 영역의 글자를 한 줄씩 그대로 옮긴다.")
    return join_spaced(" ".join(line.strip() for line in out.lines if line.strip()))


def ocr_letters(vision: str, ocr: str, limit: int = 2, partial: bool = False) -> str | None:
    """The vision text with a few of its letters replaced by the OCR's, its spacing kept. Only one-for-one letter
    swaps ('특허중' against '특허증'); no letter added or dropped, never a digit, so a figure is not touched.
    Extra OCR text before or after the read (a list number, a neighbouring line) is ignored."""
    keep = [i for i, ch in enumerate(vision) if norm(ch)]
    a, b = "".join(vision[i] for i in keep).lower(), norm(ocr)
    if not a or a == b:
        return None
    out, swaps = list(vision), 0
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal" or (tag == "insert" and i1 in (0, len(a))):
            continue
        if partial and tag == "delete" and j1 in (0, len(b)):
            continue  # the OCR line covers only part of the group
        if tag != "replace" or i2 - i1 != j2 - j1:
            return None
        for k in range(i2 - i1):
            old, new = a[i1 + k], b[j1 + k]
            if old.isdigit() or new.isdigit() or "%" in (old, new) or "." in (old, new):
                return None
            out[keep[i1 + k]] = new
            swaps += 1
    return "".join(out) if 0 < swaps <= limit else None


async def check_reading(provider, tile: Tile, text: str, hits: list[dict]) -> tuple[str, dict | None]:
    """OCR and vision read the same letters independently. Agreement passes; a disagreement gets one enlarged
    literal reread. When the reread sides with the OCR letters it replaces the vision text, when it repeats the
    vision text and the OCR was unsure (confirmed); otherwise both readings are kept for the reviewer
    (needs_review). VLMs tend to rewrite odd words into common ones."""
    ocr = " ".join(h["text"] for h in hits)
    if not hits or not disagrees(text, ocr):
        return text, ({"status": "agreed"} if hits else None)
    try:
        again = await reread(provider, tile, hits)
    except Exception:
        return text, {"status": "needs_review", "vision": text, "ocr": ocr}
    if again and not disagrees(again, ocr) and disagrees(text, again):
        # The reread sides with the OCR. Keep the first read's spacing when only letters differ.
        fixed = ocr_letters(text, ocr)
        return (fixed if fixed and not disagrees(fixed, ocr) else again), {"status": "corrected", "vision": text, "ocr": ocr}
    sure = sum(h.get("ocrConfidence", 0) for h in hits) / len(hits) >= 90
    if again and not disagrees(text, again) and not figure_conflict(text, ocr) and not sure:
        # Two vision reads agree and the OCR is unsure of its own letters (small gray text, '열어질' for '옅어질').
        # A confident OCR line that differs is never overruled: a vision model repeats the same plausible
        # rewrite twice, so both readings go to the reviewer. Figures always do.
        return text, {"status": "confirmed", "ocr": ocr}
    fixed = text
    for h in hits:
        # Each very confident OCR line corrects only the part of the group it covers, so one unsure line
        # ('개선 효과' at 93) does not block the letter another line is sure of ('징후' at 100).
        if h.get("ocrConfidence", 0) >= 95:
            fixed = ocr_letters(fixed, h["text"], partial=True) or fixed
    if fixed != text and not figure_conflict(fixed, ocr) and letters_apart(fixed, ocr) < letters_apart(text, ocr):
        return fixed, {"status": "corrected_by_ocr", "vision": text, "ocr": ocr, "reread": again}
    if min(h.get("ocrConfidence", 0) for h in hits) >= 95:
        # A very confident OCR line that differs from two vision reads by a letter or two ('특허중' for '특허증'):
        # take its letters and keep the vision spacing, which the OCR does not give. Measured on the two test
        # ads: the OCR lines it was right on scored 96.5 to 100, the one it was wrong on ('열어질') 92.6.
        fixed = ocr_letters(text, ocr)
        if fixed and not disagrees(fixed, ocr):
            return fixed, {"status": "corrected_by_ocr", "vision": text, "ocr": ocr, "reread": again}
    return text, {"status": "needs_review", "vision": text, "ocr": ocr, "reread": again}


async def ocr_lines(tile: Tile) -> tuple[list[dict], list[str], str]:
    """PaddleOCR when available; Tesseract (full and half size) otherwise or when PaddleOCR fails."""
    if paddle_ocr.enabled():
        try:
            async with OCR_SLOTS:
                lines = await asyncio.to_thread(paddle_ocr.paddle_lines, tile)
            return [line for line in lines if line["location"].get("bbox")], [], "PADDLE_OCR"
        except Exception:
            warnings = [f"{tile.id}:PADDLE_OCR_FAILED"]
    else:
        warnings = []
    try:
        async with OCR_SLOTS:
            lines = await asyncio.to_thread(local_ocr, tile)
        lines = [line for line in lines if line["location"].get("bbox")]
    except Exception:
        return [], warnings + [f"{tile.id}:OCR_NOT_AVAILABLE"], "TESSERACT"
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
    return lines, warnings, "TESSERACT"


def same_row(lines: list[dict]) -> list[list[dict]]:
    """OCR lines grouped by row, left to right. Widely spaced letters ('특 허 제 0440357 호') come back from
    PaddleOCR as separate one-letter lines, so a label and its number only meet once the row is put together."""
    rows: list[list[dict]] = []
    for line in sorted(lines, key=lambda l: (l["location"]["bbox"][1] + l["location"]["bbox"][3]) / 2):
        box = line["location"]["bbox"]
        middle = (box[1] + box[3]) / 2
        for row in rows:
            first = row[0]["location"]["bbox"]
            if first[1] <= middle <= first[3]:
                row.append(line)
                break
        else:
            rows.append([line])
    return [sorted(row, key=lambda l: l["location"]["bbox"][0]) for row in rows]


CERTIFICATE_FIELD = re.compile(r"특허증|특허\s*제|허\s*제\s*\d{7}|출원\s*번호|출원일|등록일|공개일|등록번호|"
                               r"제\s*\d{4}-\d{7}\s*호|\d{7}\s*호|\d{4}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일")


def certificate_field(text: str) -> bool:
    """The block is a field of a certificate and nothing else ('출원일 2001년 10월 25일', '특허 제 0440357 호')."""
    rest = norm(CERTIFICATE_FIELD.sub("", text))
    return bool(CERTIFICATE_FIELD.search(text)) and len(rest) <= 4


def merge_certificate(tile: Tile, blocks: list[dict], lines: list[dict] = ()) -> list[dict]:
    """When the vision read split one certificate into field blocks anyway, put them back into one unit, and give
    it back its title when the OCR saw '특허증' but the vision read dropped it (a short title is easy to skip)."""
    fields = [b for b in blocks if b.get("kind") == "certificate" or certificate_field(b["text"])]
    if not any(re.search(r"특허증|특허\s*제|허\s*제", b["text"]) for b in fields):
        return blocks
    titled = any("특허증" in norm(b["text"]) for b in fields)
    title_line = next((row for row in same_row([l for l in lines if l["location"].get("bbox")])
                       if norm("".join(l["text"] for l in row)) == "특허증"), None)
    if not titled and title_line:
        box = [min(l["location"]["bbox"][0] for l in title_line), min(l["location"]["bbox"][1] for l in title_line),
               max(l["location"]["bbox"][2] for l in title_line), max(l["location"]["bbox"][3] for l in title_line)]
        fields.append({"id": f"{tile.id}:copy:certificate-title", "text": "특허증", "segments": ["특허증"], "kind": "certificate",
                       "role": "condition", "labels": [], "words": [], "location": tile.location(precise_box=box, method="PADDLE_OCR")})
    if len(fields) < 2:
        return blocks
    fields.sort(key=lambda b: ((b["location"].get("bbox") or b["location"]["region"])[1]))
    segments = [s for b in fields for s in b["segments"]]
    boxes = [b["location"]["bbox"] for b in fields if b["location"].get("bbox")]
    box = [min(x[0] for x in boxes), min(x[1] for x in boxes), max(x[2] for x in boxes), max(x[3] for x in boxes)] if boxes else None
    merged = {**fields[0], "id": f"{tile.id}:copy:certificate", "text": " ".join(segments), "segments": segments,
              "kind": "certificate", "role": "condition", "documentContext": "patent_certificate",
              "words": [w for b in fields for w in b.get("words", [])],
              "location": tile.location(precise_box=box, method=fields[0]["location"].get("coordinateSource", "PADDLE_OCR"))}
    merged.pop("readCheck", None)
    present = [b for b in fields if b in blocks]
    first = blocks.index(min(present, key=blocks.index)) if present else 0
    out = [b for b in blocks if b not in fields]
    out.insert(min(first, len(out)), merged)
    return out


def certificate_blocks(tile: Tile, lines: list[dict], blocks: list[dict]) -> list[dict]:
    """A patent number printed in the image reaches the patent lookup even when the vision read left the
    certificate out (it may treat the certificate as a picture, not as ad copy). The block carries the OCR
    wording as it stands and is context, never a claim to review."""
    # A number the vision read already kept together with its label needs no second copy. A bare '0440357 호'
    # does not count: without '특허 제' it is not recognised as a patent number downstream.
    have = norm(" ".join(b["text"] for b in blocks if CERTIFICATE_NUMBER.search(b["text"])))
    found = []
    for row in same_row([l for l in lines if l["location"].get("bbox")]):
        joined = " ".join(l["text"] for l in row)
        for body in CERTIFICATE_NUMBER.findall(joined):
            if body in have or any(body in norm(f["text"]) for f in found):
                continue
            box = [min(l["location"]["bbox"][0] for l in row), min(l["location"]["bbox"][1] for l in row),
                   max(l["location"]["bbox"][2] for l in row), max(l["location"]["bbox"][3] for l in row)]
            found.append({"id": f"{tile.id}:copy:cert{len(found)}", "text": f"특허 제 {body} 호",
                          "segments": [f"특허 제 {body} 호"], "role": "condition", "labels": [],
                          "words": [w for l in row for w in l.get("words", [])], "documentContext": "patent_certificate",
                          "location": tile.location(precise_box=box, method=row[0]["location"].get("coordinateSource", "PADDLE_OCR"))})
    return found


async def read_tile(provider, tile: Tile) -> tuple[list[dict], list[str]]:
    """One section -> (blocks, warnings). Raises only when both Tesseract and the model are unavailable."""
    lines, warnings, engine = await ocr_lines(tile)
    if not provider.client:
        return tesseract_blocks(tile, lines), warnings
    try:
        reading = await read_copy(provider, tile, lines)
    except Exception:
        return tesseract_blocks(tile, lines), warnings + [f"{tile.id}:OCR_PROVIDER_FAILED"]
    groups = copy_groups(reading)
    assigned = assign_lines([g[3] for g in groups], [g[1].box_numbers for g in groups], lines)
    blocks = []
    for (n, group, segments, text), hits in zip(groups, assigned):
        model = vision_box(tile, group.box)
        if not hits and model and engine == "TESSERACT":
            hits += await snap_to_ocr(tile, text, model)
        check = None
        if engine == "PADDLE_OCR":
            corrected, check = await check_reading(provider, tile, text, hits)
            if corrected != text:
                text, segments = corrected, [corrected]
        blocks.append(block(tile, f"{n}", group, segments, text, hits, engine, check))
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
                    blocks.append(block(tile, f"r{n}", group, segments, text, hits, engine))
        except Exception:
            warnings.append(f"{tile.id}:RECALL_PASS_FAILED")
    return merge_certificate(tile, blocks + certificate_blocks(tile, lines, blocks), lines), warnings
