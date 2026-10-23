"""Keep patent-certificate metadata together without treating dates as ad claims."""
import re


# OCR often drops the first letter of the spaced-out "특 허 제" label, so 특 is optional.
CERTIFICATE_NUMBER = re.compile(r"(?:특\s*)?허\s*제\s*(\d{7})\s*호")
STANDARD_NUMBER = re.compile(r"\b10[-– ]\d{7}\b")
DATE_VALUE = re.compile(r"\d{4}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*[일%]?")
LABELS = ("출원번호", "출원일", "등록일", "공개일")


def patent_numbers(text: str) -> list[str]:
    """Accept the seven-digit body printed on a Korean patent certificate."""
    numbers = [re.sub(r"[-– ]", "-", n) for n in STANDARD_NUMBER.findall(text)]
    numbers += [f"10-{body}" for body in CERTIFICATE_NUMBER.findall(text)]
    return list(dict.fromkeys(numbers))


def _kind(text: str) -> str | None:
    compact = re.sub(r"\s+", "", text)
    if compact == "특허증":
        return "title"
    if CERTIFICATE_NUMBER.search(text):
        return "number"
    return next((label for label in LABELS if compact.startswith(label)), None)


def group_patent_certificate(blocks: list[dict]) -> list[dict]:
    """Replace certificate fields on one image tile with a single display block.

    Promotional copy on the same tile stays separate. Prefer VLM's complete field when
    it duplicates a broken Tesseract read; never infer an absent date or patent number.
    """
    groups: dict[tuple, list[tuple[int, dict, str]]] = {}
    for index, block in enumerate(blocks):
        location = block.get("location") or {}
        key = (location.get("fileId"), location.get("page"), location.get("tileId"))
        kind = _kind(block.get("text", ""))
        if key[0] and key[2] and kind:
            groups.setdefault(key, []).append((index, block, kind))
    replacements: dict[int, dict] = {}
    removed: set[int] = set()
    for key, items in groups.items():
        if not any(kind == "title" for _, _, kind in items) or not any(kind == "number" for _, _, kind in items):
            continue
        chosen = {}
        for index, block, kind in items:
            previous = chosen.get(kind)
            if previous is None or (block["location"].get("coordinateSource") == "VLM_OCR_REGION"
                                    and previous[1]["location"].get("coordinateSource") != "VLM_OCR_REGION"):
                chosen[kind] = (index, block)
        if len(chosen) < 3:
            continue
        removed.update(index for index, _, _ in items)
        number_block = chosen["number"][1]
        patent = CERTIFICATE_NUMBER.search(number_block["text"])
        fields = ["특허증", f"특허 제 {patent.group(1)} 호"]
        fields.extend(chosen[label][1]["text"] for label in LABELS if label in chosen)
        anchor = min(index for index, _, _ in items)
        replacements[anchor] = {**chosen["title"][1], "id": f"{key[2]}:patent-certificate",
                                "text": " · ".join(fields), "documentContext": "patent_certificate"}
        # A short OCR-only date in the same certificate row can be a duplicate of
        # VLM's complete date (e.g. 2004년 07월 05% instead of 05일).
        number_box = number_block["location"].get("bbox")
        if number_box and any(label in chosen for label in ("출원일", "등록일", "공개일")):
            for index, block in enumerate(blocks):
                if index in removed or index in replacements:
                    continue
                location = block.get("location") or {}
                if (location.get("fileId"), location.get("page"), location.get("tileId")) != key:
                    continue
                box = location.get("bbox")
                if box and number_box[1] <= box[1] <= number_box[3] + 0.04 and DATE_VALUE.fullmatch(block["text"].strip()):
                    removed.add(index)
    return [replacements[i] if i in replacements else block for i, block in enumerate(blocks)
            if i not in removed or i in replacements]
